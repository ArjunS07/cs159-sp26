"""Small, explicit selection rules for SmolVLA Q10 experiments.

Candidate provenance is part of the API.  A Q trained on P&P trees cannot be
silently presented as a P&P-independent reranker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FRESH8_EXPERIMENT = "smolvla-libero-depth1-fresh8-trees-v4-bellman"
FRESH8_KINDS = ("stored_source", *(f"fresh_seed_{i}" for i in range(1, 9)))


def fresh8_snapshot_key(tree_limit: int) -> str:
    return f"smolvla_trees/manifests/v4_fresh8_success_q10_{tree_limit}_v1.json"


@dataclass(frozen=True)
class Selection:
    index: int
    stock_score: float
    chosen_score: float
    margin: float
    switched: bool


def choose_existing(scores, kinds, *, allowed_families: set[str],
                    margin: float = 0.0) -> Selection:
    """Choose from explicit proposal families, retaining stock unless Q clears margin.

    `kinds[0]` must be the stock chunk.  The caller defines `allowed_families`:
    e.g. {"vanilla"} for an ordinary reranker or {"pnp"} for a P&P gate.
    This function never treats a P&P proposal as a vanilla policy draw.
    """
    q = np.asarray(scores, dtype=np.float64)
    if q.ndim != 1 or len(q) != len(kinds) or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("need one finite score per candidate")
    if kinds[0] != "stock" or not np.isfinite(margin) or margin < 0:
        raise ValueError("candidate zero must be stock and margin must be nonnegative")
    indices = [i for i, kind in enumerate(kinds) if i and kind in allowed_families]
    if not indices:
        raise ValueError(f"no proposals in allowed families {sorted(allowed_families)}")
    best = max(indices, key=lambda i: q[i])
    delta = float(q[best] - q[0])
    chosen = best if delta > margin else 0
    return Selection(chosen, float(q[0]), float(q[chosen]), delta, bool(chosen))


def bounded_q_ascent(score_fn, action: torch.Tensor, *,
                     max_normalized_rms: float, action_std: torch.Tensor):
    """One local action-space proposal; does not assert simulator improvement.

    `score_fn` maps [B,10,7] to [B] Q values.  The update is in units of
    training-action standard deviations, with a per-candidate RMS trust radius.
    Re-score and execute in the simulator before attributing benefit to PCP.
    """
    if action.ndim != 3 or action.shape[-2:] != (10, 7):
        raise ValueError("action must be [batch,10,7]")
    if max_normalized_rms <= 0 or not np.isfinite(max_normalized_rms):
        raise ValueError("max_normalized_rms must be positive and finite")
    std = torch.as_tensor(action_std, device=action.device, dtype=action.dtype).reshape(1, 1, 7)
    if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("action_std must be positive and finite")
    base = action.detach().clone().requires_grad_(True)
    before = score_fn(base)
    if before.shape != (len(base),):
        raise ValueError("score_fn must return one score per action chunk")
    gradient, = torch.autograd.grad(before.sum(), base)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("non-finite critic gradient")
    direction = gradient * std
    rms = direction.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    delta = max_normalized_rms * (direction / rms) * std
    return (base + delta).detach(), {
        "q_before": before.detach(),
        "normalized_update_rms": (delta / std).square().mean((1, 2)).sqrt().detach(),
    }


def load_success_q10(checkpoint: str | Path, *, expected_snapshot: str | None = None,
                     device: str | torch.device = "cpu"):
    """Load the SmolVLA success-Q checkpoint, rejecting mismatched snapshots."""
    from .qplanning_critic.config import QPlanningModelConfig
    from .qplanning_critic.model import QPlanningCritic
    from .smolvla_success_critic import FORMAT

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload.get("objective") != "success_probability_after_one_intervention":
        raise ValueError("checkpoint is not a SmolVLA success-Q10 model")
    if expected_snapshot is not None and payload.get("snapshot_digest") != expected_snapshot:
        raise ValueError("checkpoint snapshot does not match dataset")
    dims = dict(payload["architecture"])
    prefix_dim = dims.pop("prefix_dim")
    robot_dim = dims.pop("robot_dim")
    proprio_dim = dims.pop("proprio_dim")
    model = QPlanningCritic(prefix_dim=prefix_dim, robot_dim=robot_dim,
                            proprio_dim=proprio_dim, config=QPlanningModelConfig(**dims))
    model.load_state_dict(payload["model"])
    model.to(device).eval().requires_grad_(False)
    return model, payload


class SmolSuccessScorer:
    """Adapter for ordinary online candidate reranking with a timed Q10 critic."""

    def __init__(self, checkpoint: str | Path, *, device="cuda"):
        from .qplanning_critic.inference import QPlanningScorer

        model, payload = load_success_q10(checkpoint, device=device)
        if payload.get("representation", "frozen_prefill_prefix_v1") != "frozen_prefill_prefix_v1":
            raise ValueError("online RLT extraction has not been integrated; use prefill Q checkpoint")
        if payload.get("continuation_policy") != "smolvla_pnp_steps123_k311":
            raise ValueError("unexpected continuation policy")
        self.scorer = QPlanningScorer(
            model, checkpoint_id=f"smolvla-success-q10:{payload['snapshot_digest']}:{payload['update']}",
            checkpoint_path=str(checkpoint), snapshot_id=payload["snapshot_digest"],
            update=payload["update"], source_policy={"repo_id": "HuggingFaceVLA/smolvla_libero",
            "continuation": payload["continuation_policy"]}, device=device)
        self.checkpoint_id = self.scorer.checkpoint_id
        self.model = model
        self.requires_current_u20 = False
        self.remaining_fraction = None

    def set_remaining_fraction(self, value: float):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("remaining time must be a fraction in [0,1]")
        self.remaining_fraction = float(value)

    def score(self, prefix, prefix_valid, robot, proprio, actions, **kwargs):
        if self.remaining_fraction is None:
            raise RuntimeError("set remaining time before online Q scoring")
        timed_robot = np.r_[np.asarray(robot, np.float32).reshape(-1),
                            np.float32(self.remaining_fraction)]
        return self.scorer.score(prefix, prefix_valid, timed_robot, proprio, actions, **kwargs)

    def score_with_grad(self, prefix, prefix_valid, robot, proprio, actions):
        if self.remaining_fraction is None:
            raise RuntimeError("set remaining time before online Q guidance")
        timed_robot = np.r_[np.asarray(robot, np.float32).reshape(-1),
                            np.float32(self.remaining_fraction)]
        return self.scorer.score_with_grad(
            prefix, prefix_valid, timed_robot, proprio, actions)


@torch.no_grad()
def score_root_trees(model, roots, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """Score all recorded candidates, preserving within-root grouping."""
    from .smolvla_tree_bellman_finetune import _root_batch, _root_scores, _to
    scores, outcomes = [], []
    for start in range(0, len(roots), 8):
        batch = _to(_root_batch([roots[i] for i in range(start, min(start + 8, len(roots)))]),
                    torch.device(device))
        scores.append(_root_scores(model, batch).float().cpu().numpy())
        outcomes.append(batch["success"].cpu().numpy().astype(bool))
    return np.concatenate(scores), np.concatenate(outcomes)


def paired_counts(outcomes: np.ndarray, selected: np.ndarray) -> dict:
    """Count per-root rescues and spoils against recorded stock continuation."""
    y = np.asarray(outcomes, dtype=bool)
    i = np.asarray(selected, dtype=int)
    if y.ndim != 2 or i.shape != (len(y),) or np.any((i < 0) | (i >= y.shape[1])):
        raise ValueError("outcomes must be [roots,candidates], selection [roots]")
    stock, chosen = y[:, 0], y[np.arange(len(y)), i]
    return {"roots": len(y), "stock_successes": int(stock.sum()),
            "selected_successes": int(chosen.sum()),
            "rescues": int((~stock & chosen).sum()),
            "spoils": int((stock & ~chosen).sum()),
            "oracle_successes": int(y.any(1).sum())}
