"""Executed-trajectory Q10/Q50 windows backed by a bounded local cache."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..pcp_critic.data import DatasetSnapshot, eligible_rollout_rows
from ..pcp_search.data import validate_training_artifact


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QPlanningCacheIndex:
    snapshot_id: str
    horizon: int
    cache_dir: str
    rollouts: tuple[dict, ...]
    prefix_dim: int
    robot_dim: int
    proprio_dim: int
    action_dim: int
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    n_windows: int
    n_train_windows: int
    n_val_windows: int
    n_bootstrap_windows: int
    generated_executed_first10_mae: float
    generated_executed_first10_max_abs: float

    @property
    def digest(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if key != "cache_dir"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _prefix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"expected prefix [tokens,width], got {value.shape}")
    return value.astype(np.float16, copy=False)


def _mask(value: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    value = value.reshape(-1).astype(bool, copy=False)
    if len(value) != length:
        raise ValueError(f"prefix mask length {len(value)} does not match {length}")
    return value


def _padded_window(values: np.ndarray, start: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    result = np.zeros((horizon, *values.shape[1:]), dtype=values.dtype)
    width = max(0, min(horizon, len(values) - start))
    if width:
        result[:width] = values[start:start + width]
    valid = np.zeros(horizon, bool)
    valid[:width] = True
    return result, valid


def qplanning_windows_from_artifact(row: dict, arrays: dict[str, np.ndarray], *,
                                    horizon: int, gamma: float) -> dict[str, np.ndarray]:
    """Create windows anchored only at saved 10-action planning boundaries.

    Q10 uses one executed interval. Q50 concatenates five intervals from the
    actual closed-loop trajectory; it never substitutes the generated 50-action
    chunk whose tail was not executed.
    """
    if horizon not in (10, 50):
        raise ValueError("horizon must be 10 or 50")
    validate_training_artifact(arrays)
    actions = np.asarray(arrays["actions_normalized"], np.float32)
    rewards = np.asarray(arrays["rewards"], np.float32)
    terminated = np.asarray(arrays["terminated"], bool)
    truncated = np.asarray(arrays["truncated"], bool)
    success_steps = np.asarray(arrays["step_success"], bool)
    boundary_steps = np.asarray(arrays["boundary/step"], np.int64)
    prefixes = arrays["prefix/prefix_embeddings"]
    masks = arrays["prefix/prefix_pad_masks"]
    robots = np.asarray(arrays["boundary/raw_robot_state"], np.float32)
    proprios = np.asarray(arrays["boundary/policy_proprio"], np.float32)
    if len(boundary_steps) != len(prefixes) or len(boundary_steps) != len(robots):
        raise ValueError("boundary state arrays do not align")
    if np.any(np.diff(boundary_steps) <= 0):
        raise ValueError("decision-boundary steps must increase strictly")
    if boundary_steps[0] != 0 or boundary_steps[-1] != len(actions):
        raise ValueError(
            f"expected boundary range [0,{len(actions)}], got "
            f"[{boundary_steps[0]},{boundary_steps[-1]}]")
    boundary_index = {int(step): index for index, step in enumerate(boundary_steps)}
    done = terminated | truncated
    episode_success = bool(success_steps.any())
    gamma_powers = gamma ** np.arange(len(rewards), dtype=np.float64)
    records = []
    for state_index, start_value in enumerate(boundary_steps[:-1]):
        start = int(start_value)
        action, action_valid = _padded_window(actions, start, horizon)
        width = int(action_valid.sum())
        if width == 0:
            continue
        local_done = done[start:start + width]
        if local_done.any():
            width = int(np.flatnonzero(local_done)[0]) + 1
            action_valid[width:] = False
            action[width:] = 0
        reward = float(sum((gamma ** j) * float(rewards[start + j]) for j in range(width)))
        target_step = start + horizon
        may_bootstrap = width == horizon and not local_done[:width].any() and target_step in boundary_index
        next_index = boundary_index[target_step] if may_bootstrap else len(boundary_steps) - 1
        next_start = int(boundary_steps[next_index])
        next_action, next_action_valid = _padded_window(actions, next_start, horizon)
        if next_action_valid.any():
            next_width = int(next_action_valid.sum())
            next_done = done[next_start:next_start + next_width]
            if next_done.any():
                next_width = int(np.flatnonzero(next_done)[0]) + 1
                next_action_valid[next_width:] = False
                next_action[next_width:] = 0
        state_prefix = _prefix(prefixes[state_index])
        next_prefix = _prefix(prefixes[next_index])
        future = rewards[start:]
        mc_return = float(np.dot(gamma_powers[:len(future)], future.astype(np.float64)))
        records.append({
            "prefix": state_prefix,
            "pad": _mask(masks[state_index], len(state_prefix)),
            "robot": robots[state_index],
            "proprio": proprios[state_index],
            "action": action,
            "action_valid": action_valid,
            "next_prefix": next_prefix,
            "next_pad": _mask(masks[next_index], len(next_prefix)),
            "next_robot": robots[next_index],
            "next_proprio": proprios[next_index],
            "next_action": next_action,
            "next_action_valid": next_action_valid,
            "reward": np.float32(reward),
            "discount": np.float32(gamma ** horizon if may_bootstrap else 0.0),
            "mc_return": np.float32(mc_return),
            "success": np.bool_(episode_success),
            "start_step": np.int32(start),
        })
    if not records:
        raise ValueError(f"rollout {row['rollout_id']} has no usable windows")
    return {key: np.stack([record[key] for record in records]) for key in records[0]}


def _write_rollout(path: Path, row: dict, arrays: dict[str, np.ndarray], *,
                   horizon: int, gamma: float) -> tuple[dict, np.ndarray, np.ndarray]:
    packed = qplanning_windows_from_artifact(row, arrays, horizon=horizon, gamma=gamma)
    generated = np.asarray(arrays["bellman/action"], np.float32)
    executed = np.asarray(arrays["bellman/executed_normalized"], np.float32)
    valid = np.asarray(arrays["bellman/validity_mask"], bool)
    differences = []
    for index in range(min(len(generated), len(executed))):
        width = int(valid[index].sum())
        if width:
            differences.append(np.abs(generated[index, :width] - executed[index, :width]))
    differences = np.concatenate(differences) if differences else np.empty((0,), np.float32)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(temporary, path)
    entry = {
        "rollout_id": row["rollout_id"], "path": path.name,
        "n_windows": len(packed["reward"]),
        "n_bootstrap": int(np.count_nonzero(packed["discount"])),
        "benchmark": str(row.get("benchmark") or ""),
        "suite": str(row.get("suite") or ""),
        "task_idx": int(row.get("task_idx") or 0),
        "run_id": str(row.get("run_id") or ""),
        "prefix_dim": int(packed["prefix"].shape[-1]),
        "robot_dim": int(packed["robot"].shape[-1]),
        "proprio_dim": int(packed["proprio"].shape[-1]),
        "action_dim": int(packed["action"].shape[-1]),
        "generated_executed_abs_sum": float(differences.sum()) if len(differences) else 0.0,
        "generated_executed_count": int(differences.size),
        "generated_executed_max_abs": float(differences.max()) if len(differences) else 0.0,
    }
    return entry, np.asarray(arrays["actions_normalized"], np.float64), differences


def prepare_qplanning_cache(store, snapshot: DatasetSnapshot, *, horizon: int,
                            gamma: float, cache_root: str | Path) -> QPlanningCacheIndex:
    """Create/resume a local, horizon-specific cache without loading all prefixes into RAM."""
    cache_dir = Path(cache_root).expanduser() / snapshot.snapshot_id / f"q{horizon}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "index.json"
    rows = eligible_rollout_rows(store, rollout_ids=snapshot.rollout_ids)
    old = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text())
            if (payload.get("cache_schema_version") == CACHE_SCHEMA_VERSION
                    and payload.get("snapshot_id") == snapshot.snapshot_id
                    and payload.get("horizon") == horizon
                    and payload.get("gamma") == gamma):
                old = {entry["rollout_id"]: entry for entry in payload.get("rollouts", [])}
        except (OSError, json.JSONDecodeError):
            pass
    entries = []
    train_ids = set(snapshot.train_rollout_ids)
    action_count = 0
    action_sum = action_sumsq = None
    for position, row in enumerate(rows, 1):
        path = cache_dir / f"{row['rollout_id']}.npz"
        entry = old.get(row["rollout_id"])
        need_arrays = entry is None or not path.exists() or row["rollout_id"] in train_ids
        arrays = store.load_training_data(row["training_data_path"]) if need_arrays else None
        try:
            if entry is None or not path.exists():
                entry, actions, _ = _write_rollout(
                    path, row, arrays, horizon=horizon, gamma=gamma)
            elif row["rollout_id"] in train_ids:
                actions = np.asarray(arrays["actions_normalized"], np.float64)
            if row["rollout_id"] in train_ids:
                batch_sum = actions.sum(0)
                batch_sumsq = np.square(actions).sum(0)
                action_sum = batch_sum if action_sum is None else action_sum + batch_sum
                action_sumsq = batch_sumsq if action_sumsq is None else action_sumsq + batch_sumsq
                action_count += len(actions)
            entries.append(entry)
        finally:
            if arrays is not None:
                del arrays
        if position % 25 == 0 or position == len(rows):
            print(f"[qplanning] q{horizon} cache: {position}/{len(rows)} rollouts")
    if not action_count:
        raise ValueError("snapshot has no train actions")
    dimensions = {(e["prefix_dim"], e["robot_dim"], e["proprio_dim"], e["action_dim"])
                  for e in entries}
    if len(dimensions) != 1:
        raise ValueError(f"mixed dimensions in one snapshot: {dimensions}")
    prefix_dim, robot_dim, proprio_dim, action_dim = next(iter(dimensions))
    mean = action_sum / action_count
    variance = np.maximum(action_sumsq / action_count - mean ** 2, 1e-12)
    train_count = sum(e["n_windows"] for e in entries if e["rollout_id"] in train_ids)
    val_ids = set(snapshot.val_rollout_ids)
    count = sum(e["generated_executed_count"] for e in entries)
    result = QPlanningCacheIndex(
        snapshot_id=snapshot.snapshot_id, horizon=horizon, cache_dir=str(cache_dir),
        rollouts=tuple(entries), prefix_dim=prefix_dim, robot_dim=robot_dim,
        proprio_dim=proprio_dim, action_dim=action_dim,
        action_mean=tuple(float(x) for x in mean),
        action_std=tuple(float(x) for x in np.sqrt(variance)),
        n_windows=sum(e["n_windows"] for e in entries), n_train_windows=train_count,
        n_val_windows=sum(e["n_windows"] for e in entries if e["rollout_id"] in val_ids),
        n_bootstrap_windows=sum(e["n_bootstrap"] for e in entries),
        generated_executed_first10_mae=(
            sum(e["generated_executed_abs_sum"] for e in entries) / count if count else float("nan")),
        generated_executed_first10_max_abs=max(
            (e["generated_executed_max_abs"] for e in entries), default=float("nan")))
    payload = {"cache_schema_version": CACHE_SCHEMA_VERSION, "gamma": gamma, **asdict(result)}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temporary, manifest_path)
    return result


class QPlanningWindowDataset(Dataset):
    """Lazy per-rollout NPZ dataset with a bounded LRU."""

    def __init__(self, cache: QPlanningCacheIndex, rollout_ids: Iterable[str], *,
                 max_open_rollouts: int = 2, max_windows: int | None = None):
        wanted = set(rollout_ids)
        entries = [entry for entry in cache.rollouts if entry["rollout_id"] in wanted]
        indices = [(entry, i) for entry in entries for i in range(entry["n_windows"])]
        if max_windows is not None and len(indices) > max_windows:
            indices.sort(key=lambda pair: hashlib.sha256(
                f"{pair[0]['rollout_id']}|{pair[1]}".encode()).hexdigest())
            indices = indices[:max_windows]
        self.cache_dir = cache.cache_dir
        self.indices = indices
        self.max_open_rollouts = max_open_rollouts
        self._open: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self):
        return len(self.indices)

    def _arrays(self, entry: dict) -> dict[str, np.ndarray]:
        name = entry["path"]
        arrays = self._open.pop(name, None)
        if arrays is None:
            with np.load(Path(self.cache_dir) / name, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            if len(self._open) >= self.max_open_rollouts:
                self._open.popitem(last=False)
        self._open[name] = arrays
        return arrays

    def __getitem__(self, index) -> dict:
        entry, offset = self.indices[index]
        arrays = self._arrays(entry)
        return {key: value[offset] for key, value in arrays.items()}


def collate_windows(items: list[dict]) -> dict[str, torch.Tensor]:
    def padded(name: str, mask_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        arrays = [np.asarray(item[name], np.float32) for item in items]
        masks = [np.asarray(item[mask_name], bool) for item in items]
        length = max(map(len, arrays))
        width = {array.shape[-1] for array in arrays}
        if len(width) != 1:
            raise ValueError("mixed prefix widths")
        result = torch.zeros(len(items), length, next(iter(width)), dtype=torch.float32)
        valid = torch.zeros(len(items), length, dtype=torch.bool)
        for i, (array, mask) in enumerate(zip(arrays, masks)):
            result[i, :len(array)] = torch.from_numpy(array)
            valid[i, :len(mask)] = torch.from_numpy(mask)
        return result, valid

    prefix, pad = padded("prefix", "pad")
    next_prefix, next_pad = padded("next_prefix", "next_pad")
    tensor = lambda name: torch.from_numpy(np.stack([item[name] for item in items]))
    return {
        "prefix": prefix, "pad": pad,
        "robot": tensor("robot").float(), "proprio": tensor("proprio").float(),
        "action": tensor("action").float(), "action_valid": tensor("action_valid").bool(),
        "next_prefix": next_prefix, "next_pad": next_pad,
        "next_robot": tensor("next_robot").float(), "next_proprio": tensor("next_proprio").float(),
        "next_action": tensor("next_action").float(),
        "next_action_valid": tensor("next_action_valid").bool(),
        "reward": tensor("reward").float(), "discount": tensor("discount").float(),
        "mc_return": tensor("mc_return").float(), "success": tensor("success").bool(),
        "start_step": tensor("start_step").long(),
    }
