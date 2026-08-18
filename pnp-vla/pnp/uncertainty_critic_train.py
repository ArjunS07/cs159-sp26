"""Training utilities for the same-observation U20 gradient critic.

Worker 45 stores six independently sampled action chunks at the same observation.
This module turns those artifacts into group-level examples, then trains a
differentiable critic to regress U20 and preserve the within-observation ordering.
The primary representation is the ordinary, pre-refinement action prediction;
action-only and intermediate ``z_hat`` variants are explicit ablations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .uncertainty_critic import (
    CANDIDATE_COUNT, COLLECTION_EPISODE_INDICES, EXPERIMENT, METHOD,
    PROBE_STEPS, SOURCE_MODEL_REVISION, TARGET_CHUNKS,
    TRAIN_EPISODE_INDICES, VALIDATION_EPISODE_INDICES, logical_config,
)
from .verifier.model import CompactAdvantageVerifier


ROLLOUT_COLUMNS = (
    "rollout_id,experiment,benchmark,suite,task_idx,episode_idx,status,"
    "method,config_hash,config_json,generated_chunks_path"
)
ACTION_HORIZON = 20
ACTION_DIM = 7


@dataclass
class CriticTrainConfig:
    seed: int = 42
    epochs: int = 120
    patience: int = 18
    batch_groups: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    regression_weight: float = 1.0
    ranking_weight: float = 0.5
    grad_clip: float = 1.0
    dropout: float = 0.10
    action_horizon: int = ACTION_HORIZON


@dataclass
class CandidateGroupData:
    """Compact in-memory representation: one row is one observation state."""

    obs: np.ndarray
    initial_actions: np.ndarray
    z_hat_actions: np.ndarray
    targets_u10: np.ndarray
    targets_u20: np.ndarray
    targets_u50: np.ndarray
    chunk_position: np.ndarray
    episode_idx: np.ndarray
    task_idx: np.ndarray
    suite: np.ndarray
    rollout_id: np.ndarray
    chunk_idx: np.ndarray

    def __len__(self):
        return len(self.episode_idx)

    def subset(self, mask):
        mask = np.asarray(mask)
        return CandidateGroupData(**{
            name: np.asarray(value)[mask]
            for name, value in self.__dict__.items()
        })

    @property
    def train_mask(self):
        return np.isin(self.episode_idx, TRAIN_EPISODE_INDICES)

    @property
    def validation_mask(self):
        return np.isin(self.episode_idx, VALIDATION_EPISODE_INDICES)


class CandidateGroupDataset(Dataset):
    def __init__(self, groups: CandidateGroupData, representation: str):
        if representation not in {"initial_obs", "initial_action_only", "z_hat_obs"}:
            raise ValueError(f"unknown representation: {representation}")
        self.groups = groups
        self.representation = representation

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        actions = (self.groups.z_hat_actions[index]
                   if self.representation == "z_hat_obs"
                   else self.groups.initial_actions[index])
        return {
            "obs": torch.from_numpy(self.groups.obs[index]),
            "actions": torch.from_numpy(actions),
            "position": torch.tensor(
                self.groups.chunk_position[index], dtype=torch.float32),
            "target": torch.from_numpy(self.groups.targets_u20[index]),
            "group_index": torch.tensor(index),
        }


class UncertaintyGradientCritic(nn.Module):
    """Differentiable U20 predictor over observation-conditioned action chunks.

    ``forward`` returns standardized log-U20. ``predict_uncertainty`` maps it
    back to the original U20 scale. Lower scores are preferred.
    """

    def __init__(self, obs_dim=2048, action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON,
                 dropout=0.10, zero_context=False):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.zero_context = bool(zero_context)
        self.backbone = CompactAdvantageVerifier(
            obs_dim=obs_dim, action_dim=action_dim, dropout=dropout,
            conditioning="multiplicative")
        self.register_buffer("target_log_mean", torch.zeros(()))
        self.register_buffer("target_log_std", torch.ones(()))

    def set_statistics(self, action_mean, action_std, target_u20):
        self.backbone.set_action_statistics(action_mean, action_std)
        target = torch.as_tensor(target_u20, dtype=torch.float32).clamp_min(1e-8).log()
        self.target_log_mean.copy_(target.mean())
        self.target_log_std.copy_(target.std(unbiased=False).clamp_min(1e-6))

    def normalize_target(self, target_u20):
        return (target_u20.clamp_min(1e-8).log() - self.target_log_mean) / self.target_log_std

    def denormalize_prediction(self, prediction):
        return torch.exp(prediction * self.target_log_std + self.target_log_mean)

    def forward(self, obs, actions, chunk_position):
        squeeze_candidate = actions.ndim == 3
        if squeeze_candidate:
            actions = actions[:, None]
        if actions.ndim != 4:
            raise ValueError("actions must be [batch,horizon,action_dim] or "
                             "[batch,candidates,horizon,action_dim]")
        if actions.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"expected action shape (*,{self.action_horizon},{self.action_dim}), "
                f"got {tuple(actions.shape)}")
        mask = torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
        context = self.backbone.encode_context(obs, chunk_position)
        score = self.backbone.rank_candidates(
            context, actions, mask, prefix_length=self.action_horizon,
            zero_context=self.zero_context)
        return score.squeeze(1) if squeeze_candidate else score

    def predict_uncertainty(self, obs, actions, chunk_position):
        return self.denormalize_prediction(self(obs, actions, chunk_position))


def fetch_candidate_rows(store, *, require_complete=True):
    """Fetch and strictly validate the immutable 800-row worker-45 cohort."""
    expected_hash = store.config_hash(logical_config())
    rows = store.fetch_all(
        "rollouts", ROLLOUT_COLUMNS,
        configure=lambda query: query.eq("experiment", EXPERIMENT).eq(
            "method", METHOD).eq("config_hash", expected_hash).eq(
            "status", "completed"),
        order_by=("suite", "task_idx", "episode_idx"))
    if not rows:
        raise ValueError(f"no completed rows found for {EXPERIMENT}")
    duplicates = len(rows) - len({row["rollout_id"] for row in rows})
    if duplicates:
        raise ValueError(f"candidate cohort contains {duplicates} duplicate rollout IDs")
    wrong = [row for row in rows if row.get("benchmark") != "libero"]
    if wrong:
        raise ValueError("candidate cohort unexpectedly contains non-LIBERO rows")
    missing_paths = [row["rollout_id"] for row in rows
                     if not row.get("generated_chunks_path")]
    if missing_paths:
        raise ValueError(f"{len(missing_paths)} rows lack generated_chunks_path")
    expected_episodes = set(COLLECTION_EPISODE_INDICES)
    task_counts = {}
    for row in rows:
        key = (row["suite"], int(row["task_idx"]))
        task_counts.setdefault(key, set()).add(int(row["episode_idx"]))
    bad_tasks = {key: sorted(indices) for key, indices in task_counts.items()
                 if indices != expected_episodes}
    expected_rows = 4 * 10 * len(expected_episodes)
    if require_complete and (len(rows) != expected_rows or len(task_counts) != 40 or bad_tasks):
        raise ValueError(
            f"expected complete 800-row cohort (40 tasks x 20 states); "
            f"found rows={len(rows)}, tasks={len(task_counts)}, bad_tasks={len(bad_tasks)}")
    hashes = {row.get("config_hash") for row in rows}
    if hashes != {expected_hash}:
        raise ValueError(f"unexpected config hashes: {sorted(hashes)}")
    return rows


def _download_with_retry(store, path, attempts=5):
    for attempt in range(attempts):
        try:
            return store._download(path)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def _decode_row(store, row):
    with np.load(io.BytesIO(_download_with_retry(
            store, row["generated_chunks_path"])), allow_pickle=False) as artifact:
        required = {
            "group_chunk_idx", "group_chunk_pos", "candidate_initial_action_chunk",
            "candidate_z_hat", "candidate_u_time", "obs_enc", "step_indices",
        }
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"{row['rollout_id']} artifact missing {sorted(missing)}")
        chunk_idx = np.asarray(artifact["group_chunk_idx"], dtype=np.int16)
        position = np.asarray(artifact["group_chunk_pos"], dtype=np.float32)
        initial = np.asarray(
            artifact["candidate_initial_action_chunk"], dtype=np.float32)
        z_hat_all = np.asarray(artifact["candidate_z_hat"], dtype=np.float32)
        u_time = np.asarray(artifact["candidate_u_time"], dtype=np.float32)
        obs = np.asarray(artifact["obs_enc"], dtype=np.float32)
        steps = np.asarray(artifact["step_indices"], dtype=np.int16)
    groups = len(chunk_idx)
    if not np.array_equal(steps, np.asarray(PROBE_STEPS, dtype=np.int16)):
        raise ValueError(f"{row['rollout_id']} has probe steps {steps.tolist()}")
    if initial.shape[:2] != (groups, CANDIDATE_COUNT) or initial.shape[-2:] != (50, ACTION_DIM):
        raise ValueError(f"{row['rollout_id']} initial-action shape {initial.shape}")
    if z_hat_all.shape[:3] != (groups, CANDIDATE_COUNT, len(PROBE_STEPS)):
        raise ValueError(f"{row['rollout_id']} z_hat shape {z_hat_all.shape}")
    if u_time.shape != (groups, CANDIDATE_COUNT, len(PROBE_STEPS), 50):
        raise ValueError(f"{row['rollout_id']} u_time shape {u_time.shape}")
    if obs.ndim != 2 or len(obs) != groups:
        raise ValueError(f"{row['rollout_id']} obs_enc shape {obs.shape}")
    if not set(chunk_idx).issubset(TARGET_CHUNKS):
        raise ValueError(f"{row['rollout_id']} has unexpected chunks {chunk_idx.tolist()}")
    # Use the later collected correction state (Euler step 4) for the z_hat ablation.
    z_hat = z_hat_all[:, :, -1, :ACTION_HORIZON, :ACTION_DIM]
    targets = {
        width: u_time[..., :width].mean(axis=(-2, -1)).astype(np.float32)
        for width in (10, 20, 50)
    }
    if not all(np.isfinite(value).all() for value in (
            initial, z_hat, u_time, obs, *targets.values())):
        raise ValueError(f"{row['rollout_id']} artifact contains non-finite values")
    return {
        "obs": obs,
        "initial_actions": initial[:, :, :ACTION_HORIZON, :ACTION_DIM],
        "z_hat_actions": z_hat,
        "targets_u10": targets[10], "targets_u20": targets[20],
        "targets_u50": targets[50], "chunk_position": position,
        "episode_idx": np.full(groups, int(row["episode_idx"]), dtype=np.int16),
        "task_idx": np.full(groups, int(row["task_idx"]), dtype=np.int16),
        "suite": np.full(groups, row["suite"], dtype="U40"),
        "rollout_id": np.full(groups, row["rollout_id"], dtype="U32"),
        "chunk_idx": chunk_idx,
    }


def _concatenate(parts):
    if not parts:
        raise ValueError("no candidate artifacts decoded")
    return CandidateGroupData(**{
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    })


def dataset_hash(groups: CandidateGroupData):
    digest = hashlib.sha256()
    for rollout_id, chunk_idx in zip(groups.rollout_id, groups.chunk_idx):
        digest.update(f"{rollout_id}|{int(chunk_idx)}".encode())
    digest.update(np.asarray(groups.targets_u20, dtype=np.float32).tobytes())
    return digest.hexdigest()


def save_group_cache(groups: CandidateGroupData, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **groups.__dict__)
    return path


def load_group_cache(path):
    with np.load(Path(path), allow_pickle=False) as cached:
        return CandidateGroupData(**{key: cached[key] for key in cached.files})


def load_candidate_groups(store, rows, *, cache_path=None, workers=12, progress=None):
    """Download/decode artifacts in parallel or reuse one compact NPZ cache."""
    if cache_path is not None and Path(cache_path).exists():
        groups = load_group_cache(cache_path)
        validate_candidate_groups(groups)
        return groups
    parts = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(_decode_row, store, row): index
                   for index, row in enumerate(rows)}
        iterator = as_completed(futures)
        if progress is not None:
            iterator = progress(iterator, total=len(futures), desc="candidate artifacts")
        for future in iterator:
            parts[futures[future]] = future.result()
    groups = _concatenate(parts)
    validate_candidate_groups(groups)
    if cache_path is not None:
        save_group_cache(groups, cache_path)
    return groups


def validate_candidate_groups(groups: CandidateGroupData):
    if len(groups) == 0:
        raise ValueError("candidate dataset contains no observation groups")
    shapes = {
        "obs": (len(groups), None),
        "initial_actions": (len(groups), CANDIDATE_COUNT, ACTION_HORIZON, ACTION_DIM),
        "z_hat_actions": (len(groups), CANDIDATE_COUNT, ACTION_HORIZON, ACTION_DIM),
        "targets_u10": (len(groups), CANDIDATE_COUNT),
        "targets_u20": (len(groups), CANDIDATE_COUNT),
        "targets_u50": (len(groups), CANDIDATE_COUNT),
    }
    for name, expected in shapes.items():
        actual = np.asarray(getattr(groups, name)).shape
        if len(actual) != len(expected) or any(
                want is not None and got != want for got, want in zip(actual, expected)):
            raise ValueError(f"{name} has shape {actual}; expected {expected}")
    if not groups.train_mask.any() or not groups.validation_mask.any():
        raise ValueError("both train (20-35) and validation (36-39) groups are required")
    if bool((groups.train_mask & groups.validation_mask).any()):
        raise ValueError("train and validation groups overlap")
    if not np.isfinite(groups.targets_u20).all():
        raise ValueError("U20 targets contain non-finite values")
    return groups


def _loader(groups, representation, config, shuffle=False, seed_offset=0):
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    return DataLoader(
        CandidateGroupDataset(groups, representation),
        batch_size=config.batch_groups, shuffle=shuffle,
        generator=generator if shuffle else None, num_workers=0)


def _pairwise_ranking_loss(prediction, target):
    pred_difference = prediction[:, :, None] - prediction[:, None, :]
    target_difference = target[:, :, None] - target[:, None, :]
    mask = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
    mask &= target_difference.abs() > 1e-7
    if not bool(mask.any()):
        return prediction.sum() * 0
    # Predicted and target U differences should have the same sign.
    return nn.functional.softplus(
        -target_difference[mask].sign() * pred_difference[mask]).mean()


def _batch_loss(model, batch, device, config):
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    position = batch["position"].to(device)
    target = batch["target"].to(device)
    target_normalized = model.normalize_target(target)
    prediction = model(obs, actions, position)
    regression = nn.functional.smooth_l1_loss(prediction, target_normalized)
    ranking = _pairwise_ranking_loss(prediction, target_normalized)
    total = config.regression_weight * regression + config.ranking_weight * ranking
    return total, regression, ranking


@torch.no_grad()
def predict_groups(model, groups, device, representation, config=None):
    config = config or CriticTrainConfig()
    model.eval()
    predictions = []
    for batch in _loader(groups, representation, config):
        normalized = model(
            batch["obs"].to(device), batch["actions"].to(device),
            batch["position"].to(device))
        predictions.append(model.denormalize_prediction(normalized).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _rankdata(values):
    from scipy.stats import rankdata
    return rankdata(np.asarray(values), method="average")


def evaluate_critic(model, groups, device, representation, config=None):
    prediction = predict_groups(model, groups, device, representation, config)
    target = np.asarray(groups.targets_u20)
    flat_prediction, flat_target = prediction.reshape(-1), target.reshape(-1)
    pearson = float(np.corrcoef(flat_prediction, flat_target)[0, 1])
    spearman = float(np.corrcoef(
        _rankdata(flat_prediction), _rankdata(flat_target))[0, 1])
    pair_scores = []
    for predicted_group, target_group in zip(prediction, target):
        pdiff = predicted_group[:, None] - predicted_group[None, :]
        tdiff = target_group[:, None] - target_group[None, :]
        mask = np.triu(np.ones_like(tdiff, dtype=bool), 1) & (np.abs(tdiff) > 1e-7)
        if mask.any():
            pair_scores.append(np.mean(np.sign(pdiff[mask]) == np.sign(tdiff[mask])))
    chosen = prediction.argmin(axis=1)
    rows = np.arange(len(groups))
    selected = target[rows, chosen]
    default = target[:, 0]
    oracle = target.min(axis=1)
    return {
        "groups": len(groups), "candidates": int(target.size),
        "mae_u20": float(np.mean(np.abs(flat_prediction - flat_target))),
        "rmse_u20": float(np.sqrt(np.mean((flat_prediction - flat_target) ** 2))),
        "pearson": pearson, "spearman": spearman,
        "within_group_ranking_accuracy": float(np.mean(pair_scores)),
        "default_candidate_u20": float(default.mean()),
        "critic_selected_u20": float(selected.mean()),
        "oracle_candidate_u20": float(oracle.mean()),
        "selected_minus_default_u20": float((selected - default).mean()),
    }


def train_uncertainty_critic(train_groups, validation_groups, device, *,
                             representation="initial_obs", config=None,
                             progress=True):
    config = config or CriticTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    actions = (train_groups.z_hat_actions if representation == "z_hat_obs"
               else train_groups.initial_actions)
    model = UncertaintyGradientCritic(
        obs_dim=train_groups.obs.shape[1], action_dim=actions.shape[-1],
        action_horizon=actions.shape[-2], dropout=config.dropout,
        zero_context=representation == "initial_action_only").to(device)
    model.set_statistics(
        actions.reshape(-1, actions.shape[-1]).mean(0),
        actions.reshape(-1, actions.shape[-1]).std(0),
        train_groups.targets_u20.reshape(-1))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 30)
    best_loss, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []
    iterator = range(config.epochs)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc=representation, unit="epoch", dynamic_ncols=True)
    for epoch in iterator:
        model.train()
        train_losses = []
        for batch in _loader(
                train_groups, representation, config, shuffle=True, seed_offset=epoch):
            optimizer.zero_grad()
            total, regression, ranking = _batch_loss(model, batch, device, config)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append((float(total.detach()), float(regression.detach()),
                                 float(ranking.detach()), float(gradient_norm)))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in _loader(validation_groups, representation, config):
                validation_losses.append(tuple(float(value) for value in
                    _batch_loss(model, batch, device, config)))
        train_mean = np.mean(train_losses, axis=0)
        validation_mean = np.mean(validation_losses, axis=0)
        history.append({
            "epoch": epoch, "train_loss": train_mean[0],
            "train_regression": train_mean[1], "train_ranking": train_mean[2],
            "gradient_norm": train_mean[3], "validation_loss": validation_mean[0],
            "validation_regression": validation_mean[1],
            "validation_ranking": validation_mean[2],
        })
        if progress:
            iterator.set_postfix(
                val=f"{validation_mean[0]:.4f}", best=f"{best_loss:.4f}")
        if validation_mean[0] < best_loss - 1e-5:
            best_loss, best_epoch, bad = validation_mean[0], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        scheduler.step()
        if bad >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    metadata = {
        "representation": representation, "best_epoch": best_epoch,
        "epochs_ran": len(history), "best_validation_loss": float(best_loss),
        "train_config": asdict(config),
    }
    return model, history, metadata


def gradient_diagnostic(model, groups, device, representation, max_groups=32):
    subset = groups.subset(np.arange(len(groups)) < min(len(groups), max_groups))
    dataset = CandidateGroupDataset(subset, representation)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    actions = batch["actions"].to(device).requires_grad_(True)
    predicted = model.predict_uncertainty(
        batch["obs"].to(device), actions, batch["position"].to(device))
    gradient = torch.autograd.grad(predicted.mean(), actions)[0]
    return {
        "groups_checked": len(subset),
        "all_finite": bool(torch.isfinite(gradient).all()),
        "nonzero_fraction": float((gradient.abs() > 1e-12).float().mean()),
        "mean_gradient_l2": float(gradient.flatten(2).norm(dim=-1).mean()),
        "max_abs_gradient": float(gradient.abs().max()),
    }


def checkpoint_payload(model, metadata, metrics, groups, representation, config):
    return {
        "format": "pnp_uncertainty_gradient_critic_v1",
        "model_class": "pnp.uncertainty_critic_train.UncertaintyGradientCritic",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "obs_dim": model.obs_dim, "action_dim": model.action_dim,
            "action_horizon": model.action_horizon,
            "dropout": config.dropout, "zero_context": model.zero_context,
        },
        "representation": representation,
        "target": "mean U20 across P&P probe steps (3,4)",
        "source_experiment": EXPERIMENT,
        "source_model_revision": SOURCE_MODEL_REVISION,
        "dataset_hash": dataset_hash(groups),
        "train_episode_indices": list(TRAIN_EPISODE_INDICES),
        "validation_episode_indices": list(VALIDATION_EPISODE_INDICES),
        "metadata": metadata, "validation_metrics": metrics,
    }


def save_checkpoint(path, model, metadata, metrics, groups, representation, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(
        model, metadata, metrics, groups, representation, config), path)
    return path


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "pnp_uncertainty_gradient_critic_v1":
        raise ValueError("not an uncertainty-gradient critic v1 checkpoint")
    model = UncertaintyGradientCritic(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def dataset_audit(groups):
    return {
        "observation_groups": len(groups),
        "candidate_examples": len(groups) * CANDIDATE_COUNT,
        "train_groups": int(groups.train_mask.sum()),
        "validation_groups": int(groups.validation_mask.sum()),
        "rollouts": int(len(np.unique(groups.rollout_id))),
        "suites": int(len(np.unique(groups.suite))),
        "tasks": int(len(set(zip(groups.suite.tolist(), groups.task_idx.tolist())))),
        "obs_dim": int(groups.obs.shape[1]),
        "groups_by_chunk": {
            str(int(key)): int(value) for key, value in zip(
                *np.unique(groups.chunk_idx, return_counts=True))},
        "dataset_hash": dataset_hash(groups),
    }
