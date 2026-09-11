"""Q50 critic with current-U20 context and a future-U20 auxiliary head."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..pcp_critic.data import DatasetSnapshot, eligible_rollout_rows
from ..pcp_critic.registry import PCPCriticRegistry
from ..pcp_critic.resumable_snapshot import _download_with_retry
from ..store import SupabaseStore
from .config import QPlanningModelConfig, QPlanningTrainConfig
from .data import (
    QPlanningCacheIndex, QPlanningWindowDataset, collate_windows,
    prepare_qplanning_streaming_cache)
from .model import QPlanningCritic
from .train import RolloutBatchSampler, _auc


U20_CACHE_SCHEMA_VERSION = 1
FUTURE_BOUNDARIES = 4
U_LOSS_WEIGHT = 0.25
_U_TIME_KEY = re.compile(r"^c(?P<chunk>\d+)_s(?P<step>\d+)_u_time$")


@dataclass(frozen=True)
class U20LabelCache:
    snapshot_id: str
    cache_dir: str
    rollouts: tuple[dict, ...]
    current_log_mean: float
    current_log_std: float
    target_log_mean: float
    target_log_std: float
    n_labels: int
    n_future_labels: int

    @property
    def digest(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if key != "cache_dir"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _u20_labels(payload: bytes, *, expected_windows: int) -> dict[str, np.ndarray]:
    by_chunk: dict[int, list[float]] = defaultdict(list)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        for name in archive.files:
            match = _U_TIME_KEY.match(name)
            if not match:
                continue
            profile = np.asarray(archive[name], np.float32).reshape(-1)
            if len(profile) < 20:
                raise ValueError(f"{name} has only {len(profile)} action positions")
            by_chunk[int(match.group("chunk"))].append(float(profile[:20].mean()))
    expected = set(range(expected_windows))
    if set(by_chunk) != expected:
        missing = sorted(expected - set(by_chunk))
        extra = sorted(set(by_chunk) - expected)
        raise ValueError(
            f"U20 chunks do not match Q windows; missing={missing[:5]}, extra={extra[:5]}")
    current = np.asarray(
        [np.mean(by_chunk[index]) for index in range(expected_windows)], np.float32)
    if not np.isfinite(current).all() or np.any(current <= 0):
        raise ValueError("U20 labels must be finite and positive")
    future = np.zeros_like(current)
    future_valid = np.zeros(expected_windows, bool)
    for index in range(expected_windows - 1):
        values = current[index + 1:min(expected_windows, index + 1 + FUTURE_BOUNDARIES)]
        future[index] = float(values.mean())
        future_valid[index] = True
    next_current = np.zeros_like(current)
    if expected_windows > 5:
        next_current[:-5] = current[5:]
    return {
        "current_u20": current,
        "next_current_u20": next_current,
        "future_u20": future,
        "future_u20_valid": future_valid,
    }


def prepare_u20_label_cache(store, snapshot: DatasetSnapshot,
                            q_cache: QPlanningCacheIndex, *,
                            cache_root: str | Path,
                            download_workers: int = 4) -> U20LabelCache:
    """Extract tiny per-boundary labels without retaining full a-hat artifacts."""
    if q_cache.horizon != 50:
        raise ValueError("the U-aware experiment is defined only for Q50")
    if download_workers < 1:
        raise ValueError("download_workers must be positive")
    root = Path(cache_root).expanduser() / snapshot.snapshot_id / "u20_labels_v1"
    root.mkdir(parents=True, exist_ok=True)
    rows = eligible_rollout_rows(store, rollout_ids=snapshot.rollout_ids)
    by_id = {row["rollout_id"]: row for row in rows}
    entries = []
    pending = []
    for q_entry in q_cache.rollouts:
        rollout_id = q_entry["rollout_id"]
        row = by_id[rollout_id]
        ahats_path = row.get("ahats_path")
        if not ahats_path:
            raise ValueError(f"eligible rollout {rollout_id} has no ahats_path")
        path = root / f"{rollout_id}.npz"
        entry = {
            "rollout_id": rollout_id, "path": path.name,
            "n_windows": int(q_entry["n_windows"]),
            "ahats_path": str(ahats_path),
        }
        entries.append(entry)
        if not path.is_file() or not path.stat().st_size:
            pending.append((entry, path))

    worker_state = threading.local()

    def extract(item):
        entry, path = item
        worker_store = getattr(worker_state, "store", None)
        if worker_store is None:
            fork = getattr(store, "fork_for_thread", None)
            worker_store = fork() if fork is not None else store
            worker_state.store = worker_store
        payload = _download_with_retry(worker_store, entry["ahats_path"])
        labels = _u20_labels(payload, expected_windows=entry["n_windows"])
        _atomic_npz(path, **labels)
        return entry["rollout_id"]

    if pending:
        print(f"[qplanning-u20] extracting {len(pending)} rollout label files "
              f"with {download_workers} workers", flush=True)
        completed = 0
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            for _ in executor.map(extract, pending):
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(f"[qplanning-u20] labels: {completed}/{len(pending)} rollouts",
                          flush=True)
    else:
        print(f"[qplanning-u20] restored {len(entries)} cached rollout label files",
              flush=True)

    train_ids = set(snapshot.train_rollout_ids)
    current_logs, target_logs = [], []
    n_labels = n_future = 0
    for entry in entries:
        with np.load(root / entry["path"], allow_pickle=False) as archive:
            current = np.asarray(archive["current_u20"], np.float64)
            future = np.asarray(archive["future_u20"], np.float64)
            valid = np.asarray(archive["future_u20_valid"], bool)
        n_labels += len(current)
        n_future += int(valid.sum())
        if entry["rollout_id"] in train_ids:
            current_logs.extend(np.log(current).tolist())
            target_logs.extend(np.log(future[valid]).tolist())
    if not current_logs or not target_logs:
        raise ValueError("training split has no usable U20 labels")
    current_log_std = max(float(np.std(current_logs)), 1e-6)
    target_log_std = max(float(np.std(target_logs)), 1e-6)
    index = U20LabelCache(
        snapshot_id=snapshot.snapshot_id, cache_dir=str(root),
        rollouts=tuple(entries),
        current_log_mean=float(np.mean(current_logs)),
        current_log_std=current_log_std,
        target_log_mean=float(np.mean(target_logs)),
        target_log_std=target_log_std,
        n_labels=n_labels, n_future_labels=n_future)
    (root / "index.json").write_text(json.dumps(asdict(index), indent=2, sort_keys=True))
    return index


class QPlanningU20WindowDataset(QPlanningWindowDataset):
    """Q50 windows augmented with current and next-four-boundary U20 labels."""

    def __init__(self, cache: QPlanningCacheIndex, labels: U20LabelCache,
                 rollout_ids, **kwargs):
        super().__init__(cache, rollout_ids, **kwargs)
        if cache.snapshot_id != labels.snapshot_id:
            raise ValueError("Q-window and U20-label snapshots differ")
        self.label_dir = Path(labels.cache_dir)
        self.label_paths = {
            entry["rollout_id"]: entry["path"] for entry in labels.rollouts}
        self._label_open: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def _label_arrays(self, rollout_id: str) -> dict[str, np.ndarray]:
        arrays = self._label_open.pop(rollout_id, None)
        if arrays is None:
            with np.load(
                    self.label_dir / self.label_paths[rollout_id],
                    allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if len(self._label_open) >= self.max_open_rollouts:
                self._label_open.popitem(last=False)
        self._label_open[rollout_id] = arrays
        return arrays

    def __getitem__(self, index) -> dict:
        item = super().__getitem__(index)
        entry, offset = self.indices[index]
        labels = self._label_arrays(entry["rollout_id"])
        item.update({name: labels[name][offset] for name in (
            "current_u20", "next_current_u20", "future_u20", "future_u20_valid")})
        return item


class QPlanningU20Critic(QPlanningCritic):
    """RL-token Q head plus a jointly learned future-U20 token/head."""

    def __init__(self, *, prefix_dim: int, robot_dim: int, proprio_dim: int,
                 config: QPlanningModelConfig):
        if config.action_horizon != 50:
            raise ValueError("QPlanningU20Critic requires action_horizon=50")
        super().__init__(
            prefix_dim=prefix_dim, robot_dim=robot_dim,
            proprio_dim=proprio_dim, config=config)
        self.current_u_projection = nn.Sequential(
            nn.Linear(1, config.width), nn.LayerNorm(config.width))
        self.u_token = nn.Parameter(torch.empty(1, 1, config.width))
        self.u_head = nn.Linear(config.width, 1)
        self.register_buffer("current_u_log_mean", torch.tensor(0.0))
        self.register_buffer("current_u_log_std", torch.tensor(1.0))
        self.register_buffer("target_u_log_mean", torch.tensor(0.0))
        self.register_buffer("target_u_log_std", torch.tensor(1.0))
        nn.init.normal_(self.u_token, std=0.02)

    def architecture_config(self) -> dict:
        return {
            **super().architecture_config(),
            "u20_auxiliary": True,
            "future_u20_boundaries": FUTURE_BOUNDARIES,
        }

    def set_u20_statistics(self, labels: U20LabelCache) -> None:
        self.current_u_log_mean.fill_(labels.current_log_mean)
        self.current_u_log_std.fill_(labels.current_log_std)
        self.target_u_log_mean.fill_(labels.target_log_mean)
        self.target_u_log_std.fill_(labels.target_log_std)

    def normalize_current_u20(self, value: torch.Tensor) -> torch.Tensor:
        return (
            value.float().clamp_min(1e-8).log() - self.current_u_log_mean
        ) / self.current_u_log_std

    def normalize_target_u20(self, value: torch.Tensor) -> torch.Tensor:
        return (
            value.float().clamp_min(1e-8).log() - self.target_u_log_mean
        ) / self.target_u_log_std

    def denormalize_target_u20(self, value: torch.Tensor) -> torch.Tensor:
        return torch.exp(value.float() * self.target_u_log_std + self.target_u_log_mean)

    def forward(self, prefix, prefix_valid, robot, proprio, action, action_valid,
                current_u20):
        c = self.config
        if action.shape[-2:] != (c.action_horizon, c.action_dim):
            raise ValueError(
                f"expected action [batch,{c.action_horizon},{c.action_dim}], "
                f"got {tuple(action.shape)}")
        current_u20 = current_u20.reshape(len(action), 1)
        memory = torch.cat([
            self.prefix_projection(prefix.float()),
            self.robot_projection(robot.float())[:, None, :],
            self.proprio_projection(proprio.float())[:, None, :],
            self.current_u_projection(
                self.normalize_current_u20(current_u20))[:, None, :],
        ], dim=1)
        memory_valid = torch.cat([
            prefix_valid.bool(),
            torch.ones((len(prefix), 3), dtype=torch.bool, device=prefix.device),
        ], dim=1)
        normalized = (action.float() - self.action_mean) / self.action_std
        action_tokens = self.action_projection(normalized) + self.action_positions[None]
        target = torch.cat([
            self.rl_token.expand(len(action), -1, -1),
            self.u_token.expand(len(action), -1, -1),
            action_tokens,
        ], dim=1)
        target_ignored = torch.cat([
            torch.zeros((len(action), 2), dtype=torch.bool, device=action.device),
            ~action_valid.bool(),
        ], dim=1)
        decoded = self.decoder(
            target, memory, tgt_key_padding_mask=target_ignored,
            memory_key_padding_mask=~memory_valid)
        return self.value_head(decoded[:, 0]), self.u_head(decoded[:, 1]).squeeze(-1)

    def expected_value(self, *args, **kwargs) -> torch.Tensor:
        logits, _ = self(*args, **kwargs)
        return (logits.softmax(-1) * self.value_bins).sum(-1)


class QPlanningU20Scorer:
    """Frozen Q50+U20 checkpoint with one deployment tradeoff coefficient."""

    requires_current_u20 = True

    def __init__(self, model: QPlanningU20Critic, *, checkpoint_id: str,
                 checkpoint_path: str, snapshot_id: str, update: int,
                 source_policy: dict, uncertainty_beta: float, device):
        if float(uncertainty_beta) not in (0.25, 0.5, 1.0):
            raise ValueError("uncertainty_beta must be one of 0.25, 0.5, or 1.0")
        self.model = model
        self.base_checkpoint_id = checkpoint_id
        self.checkpoint_id = f"{checkpoint_id}:beta{float(uncertainty_beta):.2f}"
        self.checkpoint_path = checkpoint_path
        self.snapshot_id = snapshot_id
        self.update = int(update)
        self.source_policy = dict(source_policy)
        self.uncertainty_beta = float(uncertainty_beta)
        self.device = torch.device(device)

    @property
    def horizon(self) -> int:
        return 50

    @torch.no_grad()
    def score_components(self, prefix, prefix_valid, robot, proprio, actions,
                         *, current_u20: float, batch_size: int = 64):
        actions = torch.as_tensor(actions, device=self.device)
        if actions.ndim != 3 or actions.shape[1] < 50:
            raise ValueError("Q50+U20 candidates must be [N,>=50,action_dim]")
        prefix = torch.as_tensor(prefix, device=self.device)
        prefix_valid = torch.as_tensor(prefix_valid, device=self.device).bool()
        if prefix.ndim == 2:
            prefix = prefix[None]
        if prefix_valid.ndim == 1:
            prefix_valid = prefix_valid[None]
        robot = torch.as_tensor(
            robot, device=self.device, dtype=torch.float32).reshape(1, -1)
        proprio = torch.as_tensor(
            proprio, device=self.device, dtype=torch.float32).reshape(1, -1)
        q_parts, u_parts = [], []
        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        for start in range(0, len(actions), int(batch_size)):
            stop = min(len(actions), start + int(batch_size))
            width = stop - start
            with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16,
                    enabled=use_amp):
                logits, normalized_u = self.model(
                    prefix.expand(width, -1, -1),
                    prefix_valid.expand(width, -1),
                    robot.expand(width, -1), proprio.expand(width, -1),
                    actions[start:stop, :50, :self.model.config.action_dim],
                    torch.ones((width, 50), dtype=torch.bool, device=self.device),
                    torch.full(
                        (width,), float(current_u20), dtype=torch.float32,
                        device=self.device))
                q_parts.append(
                    (logits.softmax(-1) * self.model.value_bins).sum(-1).float())
                u_parts.append(
                    self.model.denormalize_target_u20(normalized_u).float())
        q_values, future_u20 = torch.cat(q_parts), torch.cat(u_parts)

        def standardize(values):
            return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)

        selection = (
            standardize(q_values)
            - self.uncertainty_beta * standardize(future_u20))
        return q_values, future_u20, selection


def load_qplanning_u20_scorer(checkpoint_path: str | Path, *,
                              uncertainty_beta: float, device=None,
                              expected_source_revision: str | None = None):
    """Load the final Q50+U20 checkpoint for one declared inference beta."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Q50+U20 checkpoint not found: {path}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "qplanning_q50_u20_v1":
        raise ValueError("checkpoint is not Q50+U20 v1")
    architecture = dict(payload["architecture"])
    if architecture.pop("u20_auxiliary", None) is not True:
        raise ValueError("checkpoint lacks the U20 auxiliary architecture marker")
    if int(architecture.pop("future_u20_boundaries", -1)) != FUTURE_BOUNDARIES:
        raise ValueError("checkpoint uses a different future-U20 target")
    prefix_dim = int(architecture.pop("prefix_dim"))
    robot_dim = int(architecture.pop("robot_dim"))
    proprio_dim = int(architecture.pop("proprio_dim"))
    config = QPlanningModelConfig(**architecture)
    source_policy = dict(payload.get("source_policy") or {})
    if not source_policy.get("repo_id") or not source_policy.get("revision"):
        raise ValueError("checkpoint lacks immutable source-policy provenance")
    if (expected_source_revision is not None
            and source_policy["revision"] != expected_source_revision):
        raise ValueError("checkpoint source revision differs from requested PI checkpoint")
    model = QPlanningU20Critic(
        prefix_dim=prefix_dim, robot_dim=robot_dim,
        proprio_dim=proprio_dim, config=config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_id = (
        f"{payload.get('snapshot_id', 'unknown')}:q50u20:"
        f"step{int(payload.get('update', 0))}:{digest.hexdigest()[:16]}")
    scorer = QPlanningU20Scorer(
        model, checkpoint_id=checkpoint_id, checkpoint_path=str(path),
        snapshot_id=str(payload.get("snapshot_id", "")),
        update=int(payload.get("update", 0)), source_policy=source_policy,
        uncertainty_beta=uncertainty_beta, device=device)
    print(
        f"[qplanning-u20] loaded step {scorer.update}, beta={scorer.uncertainty_beta:g} | "
        f"{scorer.checkpoint_id}", flush=True)
    return scorer


def _loader(dataset, batch_size: int, *, shuffle: bool, seed: int = 0):
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=RolloutBatchSampler(dataset, batch_size, seed),
            num_workers=0, collate_fn=collate_windows)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_windows)


def _to(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _ema(target, online, rate):
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), online.parameters()):
            target_parameter.lerp_(parameter, rate)
        for target_buffer, buffer in zip(target.buffers(), online.buffers()):
            if target_buffer.dtype.is_floating_point:
                target_buffer.lerp_(buffer, rate)
            else:
                target_buffer.copy_(buffer)


def _target_value(target: QPlanningU20Critic, batch):
    with torch.no_grad():
        next_value = target.expected_value(
            batch["next_prefix"], batch["next_pad"], batch["next_robot"],
            batch["next_proprio"], batch["next_action"], batch["next_action_valid"],
            batch["next_current_u20"])
        return (batch["reward"] + batch["discount"] * next_value).clamp(
            target.config.value_min, target.config.value_max)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def _spearman(left, right) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2:
        return float("nan")
    a, b = _rankdata(left), _rankdata(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate_q50_u20(model, target, dataset, device, *, batch_size: int) -> dict:
    model.eval(); target.eval()
    q_losses, predictions, returns, successes = [], [], [], []
    u_predictions, u_targets, u_successes, sensitivities = [], [], [], []
    for raw in _loader(dataset, batch_size, shuffle=False):
        batch = _to(raw, device)
        target_value = _target_value(target, batch)
        logits, predicted_normalized = model(
            batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
            batch["action"], batch["action_valid"], batch["current_u20"])
        q_losses.append(float(model.categorical_loss(logits, target_value).cpu()))
        q = (logits.softmax(-1) * model.value_bins).sum(-1)
        predictions.extend(q.float().cpu().tolist())
        returns.extend(batch["mc_return"].float().cpu().tolist())
        successes.extend(batch["success"].cpu().tolist())
        valid = batch["future_u20_valid"]
        if valid.any():
            predicted = model.denormalize_target_u20(predicted_normalized)
            u_predictions.extend(predicted[valid].float().cpu().tolist())
            u_targets.extend(batch["future_u20"][valid].float().cpu().tolist())
            u_successes.extend(batch["success"][valid].cpu().tolist())
        if len(batch["action"]) > 1:
            _, shuffled_u = model(
                batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
                batch["action"].roll(1, 0), batch["action_valid"].roll(1, 0),
                batch["current_u20"])
            sensitivities.extend(
                (shuffled_u - predicted_normalized).abs().float().cpu().tolist())
    q = np.asarray(predictions)
    mc = np.asarray(returns)
    success = np.asarray(successes, bool)
    up = np.asarray(u_predictions)
    ut = np.asarray(u_targets)
    us = np.asarray(u_successes, bool)
    return {
        "hl_gauss_ce": float(np.mean(q_losses)),
        "return_mae": float(np.mean(np.abs(q - mc))),
        "q_success": float(q[success].mean()) if success.any() else float("nan"),
        "q_failure": float(q[~success].mean()) if (~success).any() else float("nan"),
        "q_failure_auc": _auc(~success, -q),
        "future_u20_mae": float(np.mean(np.abs(up - ut))),
        "future_u20_spearman": _spearman(up, ut),
        "future_u20_failure_auc": _auc(~us, up),
        "u_action_shuffle_sensitivity": float(np.mean(sensitivities)),
        "n_transitions": int(len(q)),
        "n_future_u20": int(len(up)),
    }


def _cpu_state(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _latest(directory: Path):
    paths = list(directory.glob("checkpoint_step_*.pt"))
    return max(paths, key=lambda path: int(path.stem.rsplit("_", 1)[-1])) if paths else None


def _save(path, *, model, target, optimizer, update, snapshot_id, cache_digest,
          source_policy, config, history):
    payload = {
        "format": "qplanning_q50_u20_v1", "update": update,
        "snapshot_id": snapshot_id, "cache_digest": cache_digest,
        "source_policy": source_policy, "architecture": model.architecture_config(),
        "train_config": config.to_dict(), "u_loss_weight": U_LOSS_WEIGHT,
        "model": _cpu_state(model), "target": _cpu_state(target),
        "optimizer": optimizer.state_dict(), "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    for candidate in path.parent.glob("checkpoint_step_*.pt"):
        if candidate != path:
            candidate.unlink()


def _restore(path, *, model, target, optimizer, snapshot_id, cache_digest):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "qplanning_q50_u20_v1":
        raise ValueError("checkpoint is not Q50+U20 v1")
    if payload.get("snapshot_id") != snapshot_id or payload.get("cache_digest") != cache_digest:
        raise ValueError("checkpoint data contract differs from this run")
    if payload.get("architecture") != model.architecture_config():
        raise ValueError("checkpoint architecture differs from this run")
    model.load_state_dict(payload["model"])
    target.load_state_dict(payload["target"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return int(payload["update"]), list(payload.get("history", []))


def train_q50_u20(model, train_dataset, val_dataset, device, *,
                  snapshot_id, cache_digest, source_policy,
                  output_dir, config, resume=True):
    torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)
    device = torch.device(device)
    model = model.to(device)
    target = copy.deepcopy(model).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir = Path(output_dir)
    start, history = 0, []
    latest = _latest(output_dir) if resume else None
    if latest:
        start, history = _restore(
            latest, model=model, target=target, optimizer=optimizer,
            snapshot_id=snapshot_id, cache_digest=cache_digest)
        model.to(device); target.to(device)
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        print(f"[qplanning-u20] resumed {latest.name} at step {start}", flush=True)
    loader = _loader(
        train_dataset, config.micro_batch_size, shuffle=True, seed=config.seed)
    iterator = iter(loader)
    use_amp = bool(
        config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    rolling = defaultdict(float)
    rolling_count = 0
    started = time.perf_counter()
    for update in range(start + 1, config.updates + 1):
        lr = config.learning_rate_at(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.accumulation_steps):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = _to(raw, device)
            with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                q_target = _target_value(target, batch)
                logits, predicted_u = model(
                    batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
                    batch["action"], batch["action_valid"], batch["current_u20"])
                q_loss = model.categorical_loss(logits, q_target)
                valid = batch["future_u20_valid"]
                if valid.any():
                    u_target = model.normalize_target_u20(batch["future_u20"])
                    u_loss = F.smooth_l1_loss(predicted_u[valid], u_target[valid])
                    raw_u_mae = (
                        model.denormalize_target_u20(predicted_u[valid]).detach()
                        - batch["future_u20"][valid]).abs().mean()
                else:
                    u_loss = predicted_u.sum() * 0
                    raw_u_mae = predicted_u.detach().sum() * 0
                loss = (q_loss + U_LOSS_WEIGHT * u_loss) / config.accumulation_steps
            loss.backward()
            prediction = (logits.detach().float().softmax(-1) * model.value_bins).sum(-1)
            rolling["q_ce"] += float(q_loss.detach())
            rolling["u_loss"] += float(u_loss.detach())
            rolling["u_mae"] += float(raw_u_mae)
            rolling["q"] += float(prediction.mean())
            rolling["target"] += float(q_target.mean())
            rolling_count += 1
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        _ema(target, model, config.target_rate)
        if update % config.print_interval == 0 or update == config.updates:
            elapsed = time.perf_counter() - started
            completed = update - start
            eta = elapsed / max(1, completed) * (config.updates - update)
            gpu = (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else 0.0)
            print(
                f"Q50+U20 step {update}/{config.updates} | "
                f"q_ce {rolling['q_ce']/rolling_count:.4f} | "
                f"u_loss {rolling['u_loss']/rolling_count:.4f} | "
                f"u_mae {rolling['u_mae']/rolling_count:.5f} | "
                f"q {rolling['q']/rolling_count:.4f} | "
                f"target {rolling['target']/rolling_count:.4f} | "
                f"grad {float(grad):.3f} | lr {lr:.2e} | GPU {gpu:.1f} GB | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m",
                flush=True)
            rolling.clear(); rolling_count = 0
        if update % config.eval_interval == 0 or update == config.updates:
            metrics = evaluate_q50_u20(
                model, target, val_dataset, device,
                batch_size=config.micro_batch_size)
            history.append({"update": update, "validation": metrics})
            print(
                "validation | Q CE {hl_gauss_ce:.4f} | Q MAE {return_mae:.4f} | "
                "Q failure AUC {q_failure_auc:.3f} | U20 MAE {future_u20_mae:.5f} | "
                "U20 rho {future_u20_spearman:.3f} | "
                "U20 failure AUC {future_u20_failure_auc:.3f} | "
                "action sensitivity {u_action_shuffle_sensitivity:.4f} | "
                "n {n_transitions}".format(**metrics),
                flush=True)
        if update % config.checkpoint_interval == 0 or update == config.updates:
            path = output_dir / f"checkpoint_step_{update:06d}.pt"
            _save(
                path, model=model, target=target, optimizer=optimizer, update=update,
                snapshot_id=snapshot_id, cache_digest=cache_digest,
                source_policy=source_policy, config=config, history=history)
            print(f"[qplanning-u20] saved {path}", flush=True)
    final = evaluate_q50_u20(
        model, target, val_dataset, device, batch_size=config.micro_batch_size)
    return {
        "snapshot_id": snapshot_id, "cache_digest": cache_digest,
        "updates": config.updates, "train_windows": len(train_dataset),
        "validation_windows": len(val_dataset), "validation": final,
        "history": history,
        "final_checkpoint": str(
            output_dir / f"checkpoint_step_{config.updates:06d}.pt"),
    }


def run_q50_u20_training(*, snapshot_id: str,
                         cache_root: str | Path = "/content/qplanning_cache",
                         output_root: str | Path = "/content/drive/MyDrive/pnp_qplanning_u20",
                         micro_batch_size: int = 64,
                         cache_download_workers: int = 8,
                         label_download_workers: int = 8,
                         device=None, resume: bool = True,
                         store: SupabaseStore | None = None) -> dict:
    """Notebook entry point for the fixed 8,000-update Q50+U20 experiment."""
    if not snapshot_id.startswith("pcpcds-"):
        raise ValueError("paste the immutable pcpcds-* snapshot ID produced by notebook 56")
    store = store or SupabaseStore()
    snapshot = PCPCriticRegistry(store).load_snapshot(snapshot_id)
    q_cache = prepare_qplanning_streaming_cache(
        store, snapshot, horizon=50, gamma=.99, cache_root=cache_root,
        download_workers=cache_download_workers)
    labels = prepare_u20_label_cache(
        store, snapshot, q_cache, cache_root=cache_root,
        download_workers=label_download_workers)
    train = QPlanningU20WindowDataset(
        q_cache, labels, snapshot.train_rollout_ids)
    validation = QPlanningU20WindowDataset(
        q_cache, labels, snapshot.val_rollout_ids, max_windows=4096)
    model = QPlanningU20Critic(
        prefix_dim=q_cache.prefix_dim, robot_dim=q_cache.robot_dim,
        proprio_dim=q_cache.proprio_dim,
        config=QPlanningModelConfig(action_horizon=50, action_dim=q_cache.action_dim))
    model.set_action_statistics(q_cache.action_mean, q_cache.action_std)
    model.set_u20_statistics(labels)
    config = QPlanningTrainConfig(
        effective_batch_size=64, micro_batch_size=micro_batch_size,
        updates=8_000, warmup_updates=500, print_interval=100,
        eval_interval=500, checkpoint_interval=1_000,
        max_validation_transitions=4096)
    combined_digest = hashlib.sha256(
        f"{q_cache.digest}|{labels.digest}".encode()).hexdigest()[:24]
    output_dir = Path(output_root).expanduser() / snapshot_id / "q50_u20_full"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\nQ50+U20 training contract")
    print(f"  snapshot: {snapshot.snapshot_id}")
    print(f"  source PI: {snapshot.policy_repo_id}@{snapshot.policy_revision}")
    print(f"  rollouts: {len(snapshot.train_rollout_ids)} train + "
          f"{len(snapshot.val_rollout_ids)} validation")
    print(f"  windows: {len(train)} train + {len(validation)} validation")
    print("  input U: current-boundary U20, train-log-normalized")
    print(f"  auxiliary target: mean U20 over next {FUTURE_BOUNDARIES} boundaries")
    print(f"  loss: Q HL-Gauss CE + {U_LOSS_WEIGHT} * U20 SmoothL1")
    print("  U10/U50/contraction: excluded")
    print("  held-out position PRO episodes: excluded by immutable snapshot eligibility")
    print(f"  updates: {config.updates}; effective batch: 64; "
          f"microbatch: {micro_batch_size}; accumulation: {config.accumulation_steps}")
    print(f"  checkpoint directory: {output_dir}", flush=True)
    report = train_q50_u20(
        model, train, validation, device,
        snapshot_id=snapshot_id, cache_digest=combined_digest,
        source_policy={
            "repo_id": snapshot.policy_repo_id,
            "revision": snapshot.policy_revision},
        output_dir=output_dir, config=config, resume=resume)
    report["u20_label_cache_digest"] = labels.digest
    return report
