"""Online Q-Planning candidate scoring for frozen PI0.5 policies."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .config import QPlanningModelConfig
from .model import QPlanningCritic


class QPlanningScorer:
    """Loaded online critic plus immutable checkpoint provenance."""

    def __init__(self, model: QPlanningCritic, *, checkpoint_id: str,
                 checkpoint_path: str, snapshot_id: str, update: int,
                 source_policy: dict, device):
        self.model = model
        self.checkpoint_id = checkpoint_id
        self.checkpoint_path = checkpoint_path
        self.snapshot_id = snapshot_id
        self.update = int(update)
        self.source_policy = dict(source_policy)
        self.device = torch.device(device)

    @property
    def horizon(self) -> int:
        return int(self.model.config.action_horizon)

    @torch.no_grad()
    def score(self, prefix, prefix_valid, robot, proprio, actions,
              *, batch_size: int = 64) -> torch.Tensor:
        """Return one expected Q value per normalized policy-space action chunk."""
        actions = torch.as_tensor(actions, device=self.device)
        if actions.ndim != 3:
            raise ValueError("candidate actions must be [candidates, time, action_dim]")
        n = len(actions)
        horizon = self.horizon
        action_dim = self.model.config.action_dim
        if actions.shape[1] < horizon or actions.shape[2] < action_dim:
            raise ValueError(
                f"Q{horizon} needs [{horizon},{action_dim}] candidate actions, "
                f"found {tuple(actions.shape[1:])}")

        prefix = torch.as_tensor(prefix, device=self.device)
        prefix_valid = torch.as_tensor(prefix_valid, device=self.device).bool()
        if prefix.ndim == 2:
            prefix = prefix[None]
        if prefix_valid.ndim == 1:
            prefix_valid = prefix_valid[None]
        if len(prefix) != 1 or len(prefix_valid) != 1:
            raise ValueError("online scorer expects one shared observation prefix")
        robot = torch.as_tensor(robot, device=self.device, dtype=torch.float32).reshape(1, -1)
        proprio = torch.as_tensor(
            proprio, device=self.device, dtype=torch.float32).reshape(1, -1)
        if robot.shape[1] != self.model.robot_dim:
            raise ValueError(
                f"critic expects robot_dim={self.model.robot_dim}, found {robot.shape[1]}")
        if proprio.shape[1] != self.model.proprio_dim:
            raise ValueError(
                f"critic expects proprio_dim={self.model.proprio_dim}, "
                f"found {proprio.shape[1]}")

        values = []
        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        for start in range(0, n, int(batch_size)):
            stop = min(n, start + int(batch_size))
            width = stop - start
            with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16,
                    enabled=use_amp):
                value = self.model.expected_value(
                    prefix.expand(width, -1, -1),
                    prefix_valid.expand(width, -1),
                    robot.expand(width, -1), proprio.expand(width, -1),
                    actions[start:stop, :horizon, :action_dim],
                    torch.ones((width, horizon), dtype=torch.bool, device=self.device))
            values.append(value.float())
        return torch.cat(values)


class _QPlanningPrefixTap:
    """Capture the live PI prefix while leaving candidate denoising unchanged."""

    invasive = True
    capture_training_prefix = False
    shared_prefix_singleton = True

    def __init__(self):
        self.prefix = None
        self.prefix_valid = None

    @staticmethod
    def selected(step, s):
        return False

    def finish(self, ctx):
        if ctx.prefix_embeddings is None or ctx.prefix_pad_masks is None:
            raise RuntimeError("Q-Planning prefix capture produced no live PI context")
        self.prefix = ctx.prefix_embeddings[:1].detach()
        self.prefix_valid = ctx.prefix_pad_masks[:1].detach()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qplanning_scorer(checkpoint_path: str | Path, *, device=None,
                          expected_horizon: int | None = None,
                          expected_source_revision: str | None = None) -> QPlanningScorer:
    """Load one training checkpoint and reject horizon/source mismatches loudly."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Q-Planning checkpoint not found: {path}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checksum = _sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "qplanning_critic_v1":
        raise ValueError(f"unsupported Q-Planning checkpoint format: {path}")
    architecture = dict(payload["architecture"])
    prefix_dim = int(architecture.pop("prefix_dim"))
    robot_dim = int(architecture.pop("robot_dim"))
    proprio_dim = int(architecture.pop("proprio_dim"))
    config = QPlanningModelConfig(**architecture)
    if expected_horizon is not None and config.action_horizon != int(expected_horizon):
        raise ValueError(
            f"requested Q{expected_horizon}, checkpoint contains Q{config.action_horizon}")
    source_policy = dict(payload.get("source_policy") or {})
    if not source_policy.get("repo_id") or not source_policy.get("revision"):
        raise ValueError("Q-Planning checkpoint lacks immutable source-policy provenance")
    if (expected_source_revision is not None
            and source_policy["revision"] != expected_source_revision):
        raise ValueError(
            "Q-Planning checkpoint source revision differs from the requested PI checkpoint")
    model = QPlanningCritic(
        prefix_dim=prefix_dim, robot_dim=robot_dim,
        proprio_dim=proprio_dim, config=config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_id = (
        f"{payload.get('snapshot_id', 'unknown')}:q{config.action_horizon}:"
        f"step{int(payload.get('update', 0))}:{checksum[:16]}")
    scorer = QPlanningScorer(
        model, checkpoint_id=checkpoint_id, checkpoint_path=str(path),
        snapshot_id=str(payload.get("snapshot_id", "")),
        update=int(payload.get("update", 0)), source_policy=source_policy,
        device=device)
    del payload
    print(
        f"[qplanning] loaded Q{scorer.horizon} step {scorer.update} | "
        f"{checkpoint_id} | source "
        f"{source_policy['repo_id']}@{source_policy['revision']}")
    return scorer


def q_weighted_average(candidates: torch.Tensor, q_values: torch.Tensor, *,
                       n_elites: int, temperature: float):
    """Softmax-average the highest-Q complete chunks."""
    if candidates.ndim != 3 or q_values.ndim != 1 or len(candidates) != len(q_values):
        raise ValueError("candidate/Q shapes must be [N,T,A] and [N]")
    if not 1 <= int(n_elites) <= len(candidates):
        raise ValueError("n_elites must lie in [1, number of candidates]")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(q_values).all():
        raise ValueError("critic produced non-finite Q values")
    elite_q, elite_indices = torch.topk(q_values, int(n_elites), largest=True, sorted=True)
    weights = torch.softmax(elite_q / float(temperature), dim=0)
    blended = torch.sum(
        candidates.index_select(0, elite_indices) * weights[:, None, None], dim=0,
        keepdim=True)
    return blended, elite_indices, weights


def _repeat_policy_batch(value, repeats: int):
    if torch.is_tensor(value):
        if value.ndim < 1 or len(value) != 1:
            raise ValueError("online Q-Planning expects singleton preprocessor tensors")
        return value.expand(repeats, *value.shape[1:])
    if isinstance(value, dict):
        return {key: _repeat_policy_batch(item, repeats) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_repeat_policy_batch(item, repeats) for item in value)
    if isinstance(value, list):
        return [_repeat_policy_batch(item, repeats) for item in value]
    if value is None or isinstance(value, (bool, int, float, str, np.generic)):
        return value
    raise TypeError(f"unsupported preprocessor value {type(value).__name__}")


def _diversity_summary(actions: torch.Tensor, horizon: int, action_dim: int = 7) -> dict:
    values = actions[:, :horizon, :action_dim].float()
    n = len(values)
    if n < 2:
        return {"actions_compared": int(horizon), "mean_pairwise_rms": 0.0,
                "max_pairwise_rms": 0.0, "mean_pairwise_cosine": 1.0}
    flat = values.reshape(n, -1)
    differences = flat[:, None, :] - flat[None, :, :]
    rms = differences.square().mean(-1).sqrt()
    normalized = torch.nn.functional.normalize(flat, dim=-1)
    cosine = normalized @ normalized.T
    mask = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=actions.device), diagonal=1)
    return {
        "actions_compared": int(horizon),
        "mean_pairwise_rms": float(rms[mask].mean()),
        "max_pairwise_rms": float(rms[mask].max()),
        "mean_pairwise_cosine": float(cosine[mask].mean()),
    }


def qplanning_select(policy, batch, candidate_noises: torch.Tensor, *,
                     scorer: QPlanningScorer, robot_state, policy_proprio,
                     n_elites: int, temperature: float,
                     candidate_batch_size: int):
    """Generate same-observation candidates, Q-blend elites, and return telemetry."""
    from .. import sampler

    if candidate_noises.ndim != 3:
        raise ValueError("candidate_noises must be [N, chunk, latent_action_dim]")
    n_candidates = len(candidate_noises)
    if not 1 <= int(candidate_batch_size) <= n_candidates:
        raise ValueError("invalid candidate_batch_size")
    chunks = []
    shared_prefix = shared_valid = None
    previous_strategy = policy.model._pnp.strategy
    try:
        for start in range(0, n_candidates, int(candidate_batch_size)):
            stop = min(n_candidates, start + int(candidate_batch_size))
            width = stop - start
            capture = _QPlanningPrefixTap()
            sampler.set_strategy(policy.model, capture)
            with torch.no_grad():
                chunks.append(policy.predict_action_chunk(
                    _repeat_policy_batch(batch, width),
                    noise=candidate_noises[start:stop]))
            if shared_prefix is None:
                shared_prefix, shared_valid = capture.prefix, capture.prefix_valid
    finally:
        sampler.set_strategy(policy.model, previous_strategy)
    candidates = torch.cat(chunks, dim=0)
    denoise_steps = int(
        policy.model._pnp.num_steps or policy.config.num_inference_steps)
    u20_telemetry = {}
    if getattr(scorer, "requires_current_u20", False):
        from .. import sampler

        # Training U20 was measured on the ordinary 10-step decoder at Euler
        # probes 3 and 4. Candidate generation remains the paper-style 3-step
        # path; this additional pass supplies only the live state-risk context.
        previous_num_steps = policy.model._pnp.num_steps
        try:
            policy.model._pnp.num_steps = 10
            _, current_u20 = sampler.measure_chunk_uncertainty(
                policy, batch, candidate_noises[:1],
                probe_steps=(3, 4), num_iterations=5,
                uncertainty_horizon=20)
        finally:
            policy.model._pnp.num_steps = previous_num_steps
        q_values, predicted_future_u20, selection_scores = scorer.score_components(
            shared_prefix, shared_valid, robot_state, policy_proprio, candidates,
            current_u20=float(current_u20), batch_size=candidate_batch_size)
        _, elite_indices = torch.topk(
            selection_scores, int(n_elites), largest=True, sorted=True)
        elite_q = q_values.index_select(0, elite_indices)
        elite_weights = torch.softmax(elite_q / float(temperature), dim=0)
        blended = torch.sum(
            candidates.index_select(0, elite_indices)
            * elite_weights[:, None, None], dim=0, keepdim=True)
        best_index = int(torch.argmax(selection_scores))
        u20_telemetry = {
            "current_u20": float(current_u20),
            "predicted_future_u20": [
                float(value) for value in predicted_future_u20.detach().cpu()],
            "selection_scores": [
                float(value) for value in selection_scores.detach().cpu()],
            "uncertainty_beta": float(scorer.uncertainty_beta),
        }
    else:
        q_values = scorer.score(
            shared_prefix, shared_valid, robot_state, policy_proprio, candidates)
        blended, elite_indices, elite_weights = q_weighted_average(
            candidates, q_values, n_elites=n_elites, temperature=temperature)
        best_index = int(torch.argmax(q_values))
    telemetry = {
        "q_values": [float(value) for value in q_values.detach().cpu()],
        "best_index": best_index,
        "elite_indices": [int(value) for value in elite_indices.detach().cpu()],
        "elite_weights": [float(value) for value in elite_weights.detach().cpu()],
        "q_mean": float(q_values.mean()), "q_std": float(q_values.std(unbiased=False)),
        "q_min": float(q_values.min()), "q_max": float(q_values.max()),
        "n_candidates": int(n_candidates), "n_elites": int(n_elites),
        "temperature": float(temperature), "denoise_steps": denoise_steps,
        "candidate_batch_size": int(candidate_batch_size),
        "candidate_equivalent_vf_evals": int(n_candidates * denoise_steps),
        "first10_diversity": _diversity_summary(candidates, 10),
        "full50_diversity": _diversity_summary(candidates, 50),
        **u20_telemetry,
    }
    return blended, telemetry
