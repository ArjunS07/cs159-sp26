"""Step-aligned residual U20 critic and direct uncertainty-gradient diagnostic.

The v1 critic mostly learned state-level uncertainty.  V2 separates the state
mean from the candidate-specific residual, aligns each ``z_hat`` with U20 from
the same Euler step, and makes within-observation ranking the primary objective.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import copy
import io
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .pnp import _pnp_seed_perturb, run_probe
from .sampler import _temp_strategy
from .uncertainty_critic import CANDIDATE_COUNT, PROBE_STEPS
from .uncertainty_critic_train import (
    ACTION_DIM, ACTION_HORIZON, CandidateGroupData, dataset_hash,
    fetch_candidate_rows)
from .verifier.model import CompactAdvantageVerifier


@dataclass
class ResidualTrainConfig:
    seed: int = 42
    epochs: int = 120
    patience: int = 20
    batch_groups: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    value_weight: float = 0.25
    residual_weight: float = 1.0
    ranking_weight: float = 3.0
    grad_clip: float = 1.0
    dropout: float = 0.10
    action_horizon: int = ACTION_HORIZON


@dataclass
class StepAlignedGroupData:
    """One row is one (observation, Euler probe step), with six candidates."""

    obs: np.ndarray
    initial_actions: np.ndarray
    z_hat_actions: np.ndarray
    targets_u20: np.ndarray
    chunk_position: np.ndarray
    probe_step: np.ndarray
    probe_s: np.ndarray
    episode_idx: np.ndarray
    task_idx: np.ndarray
    suite: np.ndarray
    rollout_id: np.ndarray
    chunk_idx: np.ndarray

    def __len__(self):
        return len(self.episode_idx)

    def subset(self, mask):
        mask = np.asarray(mask)
        return StepAlignedGroupData(**{
            name: np.asarray(value)[mask] for name, value in self.__dict__.items()})

    @property
    def train_mask(self):
        return np.isin(self.episode_idx, tuple(range(20, 36)))

    @property
    def validation_mask(self):
        return np.isin(self.episode_idx, tuple(range(36, 40)))


class StepAlignedDataset(Dataset):
    def __init__(self, groups, representation):
        if representation not in {"z_hat_obs", "z_hat_action_only", "initial_obs"}:
            raise ValueError(f"unknown representation: {representation}")
        self.groups = groups
        self.representation = representation

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        actions = (self.groups.initial_actions[index]
                   if self.representation == "initial_obs"
                   else self.groups.z_hat_actions[index])
        return {
            "obs": torch.from_numpy(self.groups.obs[index]),
            "actions": torch.from_numpy(actions),
            "position": torch.tensor(self.groups.chunk_position[index], dtype=torch.float32),
            "probe_s": torch.tensor(self.groups.probe_s[index], dtype=torch.float32),
            "target": torch.from_numpy(self.groups.targets_u20[index]),
        }


class StepAlignedResidualCritic(nn.Module):
    """V(obs, position, s) plus action-sensitive residual A(obs, action, position, s)."""

    def __init__(self, obs_dim=2048, action_dim=ACTION_DIM,
                 action_horizon=ACTION_HORIZON, dropout=.10, zero_context=False):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.zero_context = bool(zero_context)
        # Append the scalar sampler noise level to the frozen observation encoding.
        self.backbone = CompactAdvantageVerifier(
            obs_dim=self.obs_dim + 1, action_dim=self.action_dim,
            dropout=dropout, conditioning="multiplicative")
        self.register_buffer("group_log_mean", torch.zeros(()))
        self.register_buffer("group_log_std", torch.ones(()))
        self.register_buffer("residual_log_std", torch.ones(()))

    def set_statistics(self, action_mean, action_std, target_u20):
        self.backbone.set_action_statistics(action_mean, action_std)
        log_u = torch.as_tensor(target_u20, dtype=torch.float32).clamp_min(1e-8).log()
        group_mean = log_u.mean(dim=1)
        residual = log_u - group_mean[:, None]
        self.group_log_mean.copy_(group_mean.mean())
        self.group_log_std.copy_(group_mean.std(unbiased=False).clamp_min(1e-6))
        self.residual_log_std.copy_(residual.std(unbiased=False).clamp_min(1e-6))

    def encode_context(self, obs, chunk_position, probe_s):
        augmented = torch.cat([obs, probe_s.reshape(-1, 1).to(obs.dtype)], dim=-1)
        return self.backbone.encode_context(augmented, chunk_position)

    def forward(self, obs, actions, chunk_position, probe_s):
        if actions.ndim != 4:
            raise ValueError("actions must be [batch,candidates,horizon,action_dim]")
        mask = torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
        context = self.encode_context(obs, chunk_position, probe_s)
        value = self.backbone.value(context)
        residual = self.backbone.rank_candidates(
            context, actions, mask, prefix_length=self.action_horizon,
            zero_context=self.zero_context)
        return value, residual

    def action_score(self, obs, actions, chunk_position, probe_s):
        """Action-dependent score used for steering; lower is lower predicted U20."""
        return self(obs, actions, chunk_position, probe_s)[1]

    def predict_uncertainty(self, obs, actions, chunk_position, probe_s):
        value, residual = self(obs, actions, chunk_position, probe_s)
        mean_log = value * self.group_log_std + self.group_log_mean
        return torch.exp(mean_log[:, None] + residual * self.residual_log_std)


def _download_with_retry(store, path, attempts=5):
    for attempt in range(attempts):
        try:
            return store._download(path)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def _decode_step_aligned_row(store, row):
    with np.load(io.BytesIO(_download_with_retry(
            store, row["generated_chunks_path"])), allow_pickle=False) as artifact:
        chunks = np.asarray(artifact["group_chunk_idx"], dtype=np.int16)
        positions = np.asarray(artifact["group_chunk_pos"], dtype=np.float32)
        initial = np.asarray(artifact["candidate_initial_action_chunk"], dtype=np.float32)
        z_hat = np.asarray(artifact["candidate_z_hat"], dtype=np.float32)
        u_time = np.asarray(artifact["candidate_u_time"], dtype=np.float32)
        obs = np.asarray(artifact["obs_enc"], dtype=np.float32)
        steps = np.asarray(artifact["step_indices"], dtype=np.int16)
    if not np.array_equal(steps, np.asarray(PROBE_STEPS, dtype=np.int16)):
        raise ValueError(f"{row['rollout_id']} has probe steps {steps.tolist()}")
    group_count = len(chunks)
    expected_z = (group_count, CANDIDATE_COUNT, len(PROBE_STEPS), 50, ACTION_DIM)
    expected_u = (group_count, CANDIDATE_COUNT, len(PROBE_STEPS), 50)
    if z_hat.shape != expected_z or u_time.shape != expected_u:
        raise ValueError(
            f"{row['rollout_id']} step shapes z_hat={z_hat.shape}, u={u_time.shape}")
    # Flow uses ten Euler steps: step 3 -> s=.7 and step 4 -> s=.6.
    probe_s = 1.0 - steps.astype(np.float32) / 10.0
    repeats = len(steps)
    return {
        "obs": np.repeat(obs, repeats, axis=0),
        "initial_actions": np.repeat(
            initial[:, :, :ACTION_HORIZON, :ACTION_DIM], repeats, axis=0),
        "z_hat_actions": np.transpose(
            z_hat[:, :, :, :ACTION_HORIZON, :ACTION_DIM], (0, 2, 1, 3, 4)
            ).reshape(group_count * repeats, CANDIDATE_COUNT, ACTION_HORIZON, ACTION_DIM),
        "targets_u20": np.transpose(
            u_time[..., :ACTION_HORIZON].mean(axis=-1), (0, 2, 1)
            ).reshape(group_count * repeats, CANDIDATE_COUNT).astype(np.float32),
        "chunk_position": np.repeat(positions, repeats),
        "probe_step": np.tile(steps, group_count),
        "probe_s": np.tile(probe_s, group_count),
        "episode_idx": np.full(group_count * repeats, int(row["episode_idx"]), np.int16),
        "task_idx": np.full(group_count * repeats, int(row["task_idx"]), np.int16),
        "suite": np.full(group_count * repeats, row["suite"], dtype="U40"),
        "rollout_id": np.full(group_count * repeats, row["rollout_id"], dtype="U32"),
        "chunk_idx": np.repeat(chunks, repeats),
    }


def validate_step_aligned_groups(groups):
    n = len(groups)
    if not n:
        raise ValueError("step-aligned dataset is empty")
    if groups.z_hat_actions.shape != (n, CANDIDATE_COUNT, ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"z_hat_actions shape {groups.z_hat_actions.shape}")
    if groups.targets_u20.shape != (n, CANDIDATE_COUNT):
        raise ValueError(f"targets_u20 shape {groups.targets_u20.shape}")
    if set(np.unique(groups.probe_step)) != set(PROBE_STEPS):
        raise ValueError(f"expected probe steps {PROBE_STEPS}")
    if not np.isfinite(groups.targets_u20).all():
        raise ValueError("non-finite U20 targets")
    if not groups.train_mask.any() or not groups.validation_mask.any():
        raise ValueError("training and validation groups are both required")
    return groups


def save_step_cache(groups, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **groups.__dict__)


def load_step_aligned_groups(store, rows=None, *, cache_path=None,
                             workers=12, progress=None):
    if cache_path is not None and Path(cache_path).exists():
        with np.load(Path(cache_path), allow_pickle=False) as cached:
            groups = StepAlignedGroupData(**{key: cached[key] for key in cached.files})
        return validate_step_aligned_groups(groups)
    rows = rows or fetch_candidate_rows(store, require_complete=True)
    parts = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(_decode_step_aligned_row, store, row): index
                   for index, row in enumerate(rows)}
        iterator = as_completed(futures)
        if progress is not None:
            iterator = progress(iterator, total=len(futures), desc="step-aligned artifacts")
        for future in iterator:
            parts[futures[future]] = future.result()
    groups = StepAlignedGroupData(**{
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]})
    validate_step_aligned_groups(groups)
    if cache_path is not None:
        save_step_cache(groups, cache_path)
    return groups


def _loader(groups, representation, config, shuffle=False, seed_offset=0):
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    return DataLoader(
        StepAlignedDataset(groups, representation), batch_size=config.batch_groups,
        shuffle=shuffle, generator=generator if shuffle else None, num_workers=0)


def _centered_targets(model, target):
    log_u = target.clamp_min(1e-8).log()
    group_mean = log_u.mean(dim=1)
    value_target = (group_mean - model.group_log_mean) / model.group_log_std
    residual_target = (log_u - group_mean[:, None]) / model.residual_log_std
    return value_target, residual_target


def _ranking_loss(prediction, target):
    pdiff = prediction[:, :, None] - prediction[:, None, :]
    tdiff = target[:, :, None] - target[:, None, :]
    mask = torch.triu(torch.ones_like(tdiff, dtype=torch.bool), diagonal=1)
    mask &= tdiff.abs() > 1e-7
    if not bool(mask.any()):
        return prediction.sum() * 0
    return nn.functional.softplus(-tdiff[mask].sign() * pdiff[mask]).mean()


def _loss(model, batch, device, config):
    target = batch["target"].to(device)
    value_target, residual_target = _centered_targets(model, target)
    value, raw_residual = model(
        batch["obs"].to(device), batch["actions"].to(device),
        batch["position"].to(device), batch["probe_s"].to(device))
    # Remove every state group's additive shortcut before residual regression.
    residual = raw_residual - raw_residual.mean(dim=1, keepdim=True)
    value_loss = nn.functional.smooth_l1_loss(value, value_target)
    residual_loss = nn.functional.smooth_l1_loss(residual, residual_target)
    ranking_loss = _ranking_loss(raw_residual, residual_target)
    total = (config.value_weight * value_loss
             + config.residual_weight * residual_loss
             + config.ranking_weight * ranking_loss)
    return total, value_loss, residual_loss, ranking_loss


@torch.no_grad()
def predict_step_groups(model, groups, device, representation, config=None):
    config = config or ResidualTrainConfig()
    model.eval()
    predictions, scores = [], []
    for batch in _loader(groups, representation, config):
        obs, actions = batch["obs"].to(device), batch["actions"].to(device)
        position, probe_s = batch["position"].to(device), batch["probe_s"].to(device)
        predictions.append(model.predict_uncertainty(
            obs, actions, position, probe_s).cpu().numpy())
        scores.append(model.action_score(obs, actions, position, probe_s).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(scores)


def evaluate_residual_critic(model, groups, device, representation, config=None):
    prediction, score = predict_step_groups(
        model, groups, device, representation, config)
    target = groups.targets_u20
    ranking = []
    for predicted_group, target_group in zip(score, target):
        pdiff = predicted_group[:, None] - predicted_group[None, :]
        tdiff = target_group[:, None] - target_group[None, :]
        mask = np.triu(np.ones_like(tdiff, dtype=bool), 1) & (np.abs(tdiff) > 1e-7)
        if mask.any():
            ranking.append(np.mean(np.sign(pdiff[mask]) == np.sign(tdiff[mask])))
    chosen = score.argmin(1)
    rows = np.arange(len(groups))
    selected, default, oracle = target[rows, chosen], target[:, 0], target.min(1)
    metrics = {
        "groups": len(groups), "candidates": int(target.size),
        "mae_u20": float(np.abs(prediction - target).mean()),
        "within_group_ranking_accuracy": float(np.mean(ranking)),
        "default_candidate_u20": float(default.mean()),
        "critic_selected_u20": float(selected.mean()),
        "oracle_candidate_u20": float(oracle.mean()),
        "selected_minus_default_u20": float((selected - default).mean()),
    }
    for step in PROBE_STEPS:
        use = groups.probe_step == step
        step_rows = np.flatnonzero(use)
        step_chosen = score[use].argmin(1)
        step_target = target[use]
        step_selected = step_target[np.arange(len(step_rows)), step_chosen]
        step_default = step_target[:, 0]
        metrics[f"step{step}_selected_minus_default_u20"] = float(
            (step_selected - step_default).mean())
    return metrics


def train_residual_critic(train_groups, validation_groups, device, *,
                          representation="z_hat_obs", config=None, progress=True):
    config = config or ResidualTrainConfig()
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    actions = (train_groups.initial_actions if representation == "initial_obs"
               else train_groups.z_hat_actions)
    model = StepAlignedResidualCritic(
        obs_dim=train_groups.obs.shape[1], action_dim=actions.shape[-1],
        action_horizon=actions.shape[-2], dropout=config.dropout,
        zero_context=representation == "z_hat_action_only").to(device)
    model.set_statistics(
        actions.reshape(-1, actions.shape[-1]).mean(0),
        actions.reshape(-1, actions.shape[-1]).std(0),
        train_groups.targets_u20)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 30)
    best_rank, best_selected, best_state, best_epoch, bad = -1., float("inf"), None, -1, 0
    history = []
    iterator = range(config.epochs)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc=representation, unit="epoch", dynamic_ncols=True)
    for epoch in iterator:
        model.train(); losses = []
        for batch in _loader(train_groups, representation, config, True, epoch):
            optimizer.zero_grad()
            values = _loss(model, batch, device, config)
            values[0].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip)
            optimizer.step()
            losses.append([float(value.detach()) for value in values]
                          + [float(gradient_norm)])
        scheduler.step()
        val = evaluate_residual_critic(
            model, validation_groups, device, representation, config)
        mean_loss = np.mean(losses, axis=0)
        history.append({
            "epoch": epoch, "train_total": mean_loss[0], "train_value": mean_loss[1],
            "train_residual": mean_loss[2], "train_ranking": mean_loss[3],
            "gradient_norm": mean_loss[4],
            "validation_ranking": val["within_group_ranking_accuracy"],
            "validation_selected_minus_default_u20": val["selected_minus_default_u20"],
        })
        rank = val["within_group_ranking_accuracy"]
        selected = val["critic_selected_u20"]
        improved = rank > best_rank + 1e-4 or (
            abs(rank - best_rank) <= 1e-4 and selected < best_selected)
        if improved:
            best_rank, best_selected, best_epoch, bad = rank, selected, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if progress:
            iterator.set_postfix(rank=f"{rank:.3f}", delta=f"{val['selected_minus_default_u20']:.5f}")
        if bad >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("residual critic produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, {
        "representation": representation, "best_epoch": best_epoch,
        "epochs_ran": len(history), "best_validation_ranking": best_rank,
        "best_validation_selected_u20": best_selected,
        "train_config": asdict(config),
    }


def residual_gradient_diagnostic(model, groups, device, representation, max_groups=16):
    subset = groups.subset(np.arange(len(groups)) < min(len(groups), max_groups))
    batch = next(iter(DataLoader(
        StepAlignedDataset(subset, representation), batch_size=len(subset))))
    actions = batch["actions"].to(device).requires_grad_(True)
    score = model.action_score(
        batch["obs"].to(device), actions, batch["position"].to(device),
        batch["probe_s"].to(device))
    gradient = torch.autograd.grad(score.mean(), actions)[0]
    return {
        "groups_checked": len(subset), "all_finite": bool(torch.isfinite(gradient).all()),
        "nonzero_fraction": float((gradient.abs() > 1e-12).float().mean()),
        "mean_gradient_l2": float(gradient.flatten(2).norm(dim=-1).mean()),
        "max_abs_gradient": float(gradient.abs().max()),
    }


def save_residual_checkpoint(path, model, metadata, metrics, groups, representation, config):
    payload = {
        "format": "pnp_step_aligned_residual_u20_critic_v2",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "obs_dim": model.obs_dim, "action_dim": model.action_dim,
            "action_horizon": model.action_horizon, "dropout": config.dropout,
            "zero_context": model.zero_context,
        },
        "representation": representation,
        "target": "same-Euler-step U20; candidate log-U centered within observation",
        "source_dataset_hash": dataset_hash(CandidateGroupData(
            obs=groups.obs, initial_actions=groups.initial_actions,
            z_hat_actions=groups.z_hat_actions, targets_u10=groups.targets_u20,
            targets_u20=groups.targets_u20, targets_u50=groups.targets_u20,
            chunk_position=groups.chunk_position, episode_idx=groups.episode_idx,
            task_idx=groups.task_idx, suite=groups.suite, rollout_id=groups.rollout_id,
            chunk_idx=groups.chunk_idx)),
        "metadata": metadata, "validation_metrics": metrics,
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


class DirectUncertaintyGradientTap:
    """Non-feedback diagnostic of d(measured U20)/d(live flow latent)."""

    invasive = True

    def __init__(self, *, probe_step=3, k=5, horizon=20,
                 epsilons=(.0025, .005, .01), perturb_seed=12345,
                 random_seed=54321, checkpoint_vfield=True):
        self.probe_step = int(probe_step)
        self.k = int(k)
        self.horizon = int(horizon)
        self.epsilons = tuple(float(value) for value in epsilons)
        self.perturb_seed = int(perturb_seed)
        self.random_seed = int(random_seed)
        self.checkpoint_vfield = bool(checkpoint_vfield)
        self.records = []

    def begin_chunk(self):
        pass

    def selected(self, step, s):
        return int(step) == self.probe_step

    def step(self, x_t, s, vf, ctx):
        from torch.utils.checkpoint import checkpoint

        def differentiable_vf(value):
            if self.checkpoint_vfield:
                return checkpoint(vf, value, use_reentrant=False)
            return vf(value)

        with torch.enable_grad():
            x = x_t.detach().clone().requires_grad_(True)
            _pnp_seed_perturb(self.perturb_seed)
            probe = run_probe(x, s, differentiable_vf, k=self.k, adim=ACTION_DIM)
            u20 = probe.u_time[:self.horizon].mean()
            gradient = torch.autograd.grad(u20, x)[0]
        gradient_rms = gradient.float().square().mean().sqrt().clamp_min(1e-12)
        descent = gradient / gradient_rms.to(gradient.dtype)
        generator = torch.Generator(device=x_t.device).manual_seed(self.random_seed)
        random_direction = torch.empty_like(x_t).normal_(generator=generator)
        random_direction = random_direction / random_direction.float().square().mean().sqrt().to(
            random_direction.dtype).clamp_min(1e-12)

        def measured_u(value):
            _pnp_seed_perturb(self.perturb_seed)
            with torch.no_grad():
                measured = run_probe(value, s, vf, k=self.k, adim=ACTION_DIM)
                return float(measured.u_time[:self.horizon].mean())

        baseline = float(u20.detach())
        for epsilon in self.epsilons:
            lower = measured_u(x_t - epsilon * descent)
            higher = measured_u(x_t + epsilon * descent)
            random = measured_u(x_t + epsilon * random_direction)
            self.records.append({
                "probe_step": self.probe_step, "s": float(s), "epsilon_rms": epsilon,
                "baseline_u20": baseline, "descent_u20": lower,
                "ascent_u20": higher, "random_u20": random,
                "descent_delta_u20": lower - baseline,
                "ascent_delta_u20": higher - baseline,
                "random_delta_u20": random - baseline,
                "gradient_rms": float(gradient_rms),
                "gradient_finite": bool(torch.isfinite(gradient).all()),
            })
        del probe, u20, gradient, x
        return x_t

    def finish(self, ctx):
        pass


def direct_uncertainty_gradient_test(policy, batch, noise, *, probe_step=3, k=5,
                                     horizon=20, epsilons=(.0025, .005, .01),
                                     perturb_seed=12345, random_seed=54321):
    """Run a no-feedback local test using common perturbation randomness.

    π0.5 weights are frozen temporarily; gradients are retained only with respect
    to the live sampler latent at ``probe_step``. The returned action is ignored.
    """
    tap = DirectUncertaintyGradientTap(
        probe_step=probe_step, k=k, horizon=horizon, epsilons=epsilons,
        perturb_seed=perturb_seed, random_seed=random_seed)
    parameters = list(policy.model.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        with _temp_strategy(policy.model, tap):
            policy.predict_action_chunk(batch, noise=noise)
    finally:
        for parameter, required in zip(parameters, requires_grad):
            parameter.requires_grad_(required)
    if not tap.records:
        raise RuntimeError(f"sampler never reached direct-gradient probe step {probe_step}")
    return tap.records
