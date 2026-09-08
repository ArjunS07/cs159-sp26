"""Executed-trajectory Q10/Q50 windows backed by a bounded local cache."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..pcp_critic.data import DatasetSnapshot, eligible_rollout_rows
from ..pcp_critic.resumable_snapshot import load_training_fields_with_retry


CACHE_SCHEMA_VERSION = 2
SOURCE_CACHE_SCHEMA_VERSION = 1
DEFAULT_DOWNLOAD_WORKERS = 4
PROGRESS_INTERVAL = 10
QPLANNING_ARTIFACT_FIELDS = (
    "actions_normalized", "rewards", "terminated", "truncated", "step_success",
    "boundary/step", "boundary/raw_robot_state", "boundary/policy_proprio",
    "prefix/prefix_embeddings", "prefix/prefix_pad_masks",
    "bellman/action", "bellman/executed_normalized", "bellman/validity_mask",
)


def _validate_qplanning_fields(arrays: dict[str, np.ndarray]) -> None:
    missing = sorted(set(QPLANNING_ARTIFACT_FIELDS) - set(arrays))
    if missing:
        raise ValueError(f"Q-planning artifact is missing {missing}")
    n_steps = len(arrays["actions_normalized"])
    for name in ("rewards", "terminated", "truncated", "step_success"):
        if len(arrays[name]) != n_steps:
            raise ValueError(f"{name} does not align with executed actions")
    boundaries = len(arrays["boundary/step"])
    for name in ("boundary/raw_robot_state", "boundary/policy_proprio",
                 "prefix/prefix_embeddings", "prefix/prefix_pad_masks"):
        if len(arrays[name]) != boundaries:
            raise ValueError(f"{name} does not align with planning boundaries")
    transitions = boundaries - 1
    for name in ("bellman/action", "bellman/executed_normalized", "bellman/validity_mask"):
        if len(arrays[name]) != transitions:
            raise ValueError(f"{name} does not align with planning transitions")


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
    storage_mode: str = "materialized"
    gamma: float = .99

    @property
    def digest(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if key != "cache_dir"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temporary, path)


def _rollout_digest(rollout_ids: Iterable[str]) -> str:
    value = "\n".join(sorted(rollout_ids))
    return hashlib.sha256(value.encode()).hexdigest()


def _bounded_parallel_map(function, items: Iterable, *, max_workers: int):
    """Yield ordered results without queueing an unbounded number of large jobs."""
    iterator = iter(items)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    pending = deque()
    try:
        for _ in range(max_workers * 2):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


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
    _validate_qplanning_fields(arrays)
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


def _generated_executed_statistics(arrays: dict[str, np.ndarray]) -> tuple[float, int, float]:
    generated = np.asarray(arrays["bellman/action"], np.float32)
    executed = np.asarray(arrays["bellman/executed_normalized"], np.float32)
    valid = np.asarray(arrays["bellman/validity_mask"], bool)
    differences = []
    for index in range(min(len(generated), len(executed))):
        width = int(valid[index].sum())
        if width:
            differences.append(np.abs(generated[index, :width] - executed[index, :width]))
    differences = np.concatenate(differences) if differences else np.empty((0,), np.float32)
    return (
        float(differences.sum()) if len(differences) else 0.0,
        int(differences.size),
        float(differences.max()) if len(differences) else 0.0,
    )


def _write_source_rollout(path: Path, row: dict,
                          arrays: dict[str, np.ndarray]) -> dict:
    """Persist the expensive remote fields once for both Q10 and Q50."""
    _validate_qplanning_fields(arrays)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{
            name: arrays[name] for name in QPLANNING_ARTIFACT_FIELDS})
    os.replace(temporary, path)
    actions = np.asarray(arrays["actions_normalized"], np.float64)
    abs_sum, difference_count, max_abs = _generated_executed_statistics(arrays)
    return {
        "rollout_id": row["rollout_id"], "path": path.name,
        "benchmark": str(row.get("benchmark") or ""),
        "suite": str(row.get("suite") or ""),
        "task_idx": int(row.get("task_idx") or 0),
        "run_id": str(row.get("run_id") or ""),
        "action_count": int(len(actions)),
        "action_sum": [float(value) for value in actions.sum(0)],
        "action_sumsq": [float(value) for value in np.square(actions).sum(0)],
        "generated_executed_abs_sum": abs_sum,
        "generated_executed_count": difference_count,
        "generated_executed_max_abs": max_abs,
    }


def _download_source_rollout(store, source_dir: Path, row: dict) -> dict:
    arrays = load_training_fields_with_retry(
        store, row["training_data_path"], QPLANNING_ARTIFACT_FIELDS)
    try:
        return _write_source_rollout(
            source_dir / f"{row['rollout_id']}.npz", row, arrays)
    finally:
        del arrays


def _source_progress_payload(snapshot: DatasetSnapshot, entries: list[dict], *,
                             complete: bool) -> dict:
    return {
        "source_cache_schema_version": SOURCE_CACHE_SCHEMA_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "rollout_digest": _rollout_digest(snapshot.rollout_ids),
        "complete": complete,
        "rollouts": entries,
    }


def _load_source_entries(index_path: Path, source_dir: Path,
                         snapshot: DatasetSnapshot) -> dict[str, dict]:
    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if (payload.get("source_cache_schema_version") != SOURCE_CACHE_SCHEMA_VERSION
            or payload.get("snapshot_id") != snapshot.snapshot_id
            or payload.get("rollout_digest") != _rollout_digest(snapshot.rollout_ids)):
        return {}
    wanted = set(snapshot.rollout_ids)
    return {
        entry["rollout_id"]: entry for entry in payload.get("rollouts", [])
        if entry.get("rollout_id") in wanted
        and (source_dir / entry.get("path", "")).is_file()
    }


def _prepare_source_cache(store, snapshot: DatasetSnapshot, *, cache_root: Path,
                          download_workers: int) -> tuple[Path, tuple[dict, ...]]:
    """Create/resume the shared remote-artifact cache with bounded parallelism."""
    source_dir = cache_root / snapshot.snapshot_id / "source_v1"
    source_dir.mkdir(parents=True, exist_ok=True)
    index_path = source_dir / "index.json"
    entries_by_id = _load_source_entries(index_path, source_dir, snapshot)
    wanted_ids = tuple(snapshot.rollout_ids)
    if len(entries_by_id) == len(wanted_ids):
        print(
            f"[qplanning] shared source cache: reused {len(wanted_ids)}/{len(wanted_ids)} rollouts",
            flush=True)
        return source_dir, tuple(entries_by_id[rollout_id] for rollout_id in wanted_ids)

    rows = eligible_rollout_rows(store, rollout_ids=wanted_ids)
    rows_by_id = {row["rollout_id"]: row for row in rows}
    missing_ids = [rollout_id for rollout_id in wanted_ids if rollout_id not in entries_by_id]
    print(
        f"[qplanning] shared source cache: {len(entries_by_id)}/{len(wanted_ids)} ready; "
        f"downloading {len(missing_ids)} with {download_workers} workers",
        flush=True,
    )
    completed_since_save = 0
    worker_state = threading.local()

    def download(rollout_id: str) -> dict:
        worker_store = getattr(worker_state, "store", None)
        if worker_store is None:
            fork = getattr(store, "fork_for_thread", None)
            worker_store = fork() if fork is not None else store
            worker_state.store = worker_store
        return _download_source_rollout(
            worker_store, source_dir, rows_by_id[rollout_id])

    try:
        results = _bounded_parallel_map(
            download, missing_ids, max_workers=download_workers)
        for entry in results:
            entries_by_id[entry["rollout_id"]] = entry
            completed_since_save += 1
            completed = len(entries_by_id)
            if completed_since_save >= PROGRESS_INTERVAL:
                ordered = [entries_by_id[rid] for rid in wanted_ids if rid in entries_by_id]
                _atomic_json(index_path, _source_progress_payload(
                    snapshot, ordered, complete=False))
                completed_since_save = 0
            if completed % PROGRESS_INTERVAL == 0 or completed == len(wanted_ids):
                print(
                    f"[qplanning] shared source cache: {completed}/{len(wanted_ids)} rollouts",
                    flush=True,
                )
    finally:
        ordered = [entries_by_id[rid] for rid in wanted_ids if rid in entries_by_id]
        _atomic_json(index_path, _source_progress_payload(
            snapshot, ordered, complete=len(ordered) == len(wanted_ids)))
    if len(entries_by_id) != len(wanted_ids):
        raise RuntimeError(
            f"shared source cache is incomplete: {len(entries_by_id)}/{len(wanted_ids)}")
    return source_dir, tuple(entries_by_id[rollout_id] for rollout_id in wanted_ids)


def _load_source_arrays(source_dir: Path, entry: dict) -> dict[str, np.ndarray]:
    with np.load(source_dir / entry["path"], allow_pickle=False) as archive:
        return {name: archive[name] for name in QPLANNING_ARTIFACT_FIELDS}


def _write_rollout(path: Path, row: dict, arrays: dict[str, np.ndarray], *,
                   horizon: int, gamma: float, source_entry: dict | None = None) -> dict:
    packed = qplanning_windows_from_artifact(row, arrays, horizon=horizon, gamma=gamma)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(temporary, path)
    source_entry = source_entry or {}
    return {
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
        "generated_executed_abs_sum": float(
            source_entry.get("generated_executed_abs_sum", 0.0)),
        "generated_executed_count": int(
            source_entry.get("generated_executed_count", 0)),
        "generated_executed_max_abs": float(
            source_entry.get("generated_executed_max_abs", 0.0)),
    }


def _cache_from_payload(payload: dict, cache_dir: Path) -> QPlanningCacheIndex:
    return QPlanningCacheIndex(
        snapshot_id=payload["snapshot_id"], horizon=int(payload["horizon"]),
        cache_dir=str(cache_dir), rollouts=tuple(payload["rollouts"]),
        prefix_dim=int(payload["prefix_dim"]), robot_dim=int(payload["robot_dim"]),
        proprio_dim=int(payload["proprio_dim"]), action_dim=int(payload["action_dim"]),
        action_mean=tuple(payload["action_mean"]), action_std=tuple(payload["action_std"]),
        n_windows=int(payload["n_windows"]), n_train_windows=int(payload["n_train_windows"]),
        n_val_windows=int(payload["n_val_windows"]),
        n_bootstrap_windows=int(payload["n_bootstrap_windows"]),
        generated_executed_first10_mae=float(
            payload["generated_executed_first10_mae"]),
        generated_executed_first10_max_abs=float(
            payload["generated_executed_first10_max_abs"]),
        storage_mode=str(payload.get("storage_mode", "materialized")),
        gamma=float(payload.get("gamma", .99)),
    )


def _load_finished_cache(manifest_path: Path, cache_dir: Path, snapshot: DatasetSnapshot,
                         *, horizon: int, gamma: float) -> QPlanningCacheIndex | None:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or payload.get("snapshot_id") != snapshot.snapshot_id
            or payload.get("horizon") != horizon or payload.get("gamma") != gamma
            or not payload.get("complete")):
        return None
    entries = payload.get("rollouts", [])
    if ({entry.get("rollout_id") for entry in entries} != set(snapshot.rollout_ids)
            or any(not (cache_dir / entry.get("path", "")).is_file() for entry in entries)):
        return None
    return _cache_from_payload(payload, cache_dir)


def prepare_qplanning_cache(store, snapshot: DatasetSnapshot, *, horizon: int,
                            gamma: float, cache_root: str | Path,
                            download_workers: int = DEFAULT_DOWNLOAD_WORKERS) -> QPlanningCacheIndex:
    """Create/resume a shared download cache and local horizon-specific windows."""
    if not 1 <= download_workers <= 8:
        raise ValueError("download_workers must be in [1, 8]")
    root = Path(cache_root).expanduser()
    cache_dir = root / snapshot.snapshot_id / f"q{horizon}_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "index.json"
    finished = _load_finished_cache(
        manifest_path, cache_dir, snapshot, horizon=horizon, gamma=gamma)
    if finished is not None:
        print(
            f"[qplanning] Q{horizon} window cache: reused {len(finished.rollouts)} rollouts",
            flush=True)
        return finished

    source_dir, source_entries = _prepare_source_cache(
        store, snapshot, cache_root=root, download_workers=download_workers)
    source_by_id = {entry["rollout_id"]: entry for entry in source_entries}
    progress_path = cache_dir / "progress.json"
    old = {}
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
            if (progress.get("cache_schema_version") == CACHE_SCHEMA_VERSION
                    and progress.get("snapshot_id") == snapshot.snapshot_id
                    and progress.get("horizon") == horizon and progress.get("gamma") == gamma):
                old = {
                    entry["rollout_id"]: entry for entry in progress.get("rollouts", [])
                    if (cache_dir / entry.get("path", "")).is_file()
                }
        except (OSError, json.JSONDecodeError):
            pass
    entries_by_id = old
    wanted_ids = tuple(snapshot.rollout_ids)
    missing_ids = [rollout_id for rollout_id in wanted_ids if rollout_id not in entries_by_id]
    print(
        f"[qplanning] Q{horizon} local windows: {len(entries_by_id)}/{len(wanted_ids)} ready; "
        f"building {len(missing_ids)} with 2 workers",
        flush=True,
    )
    completed_since_save = 0
    try:
        def convert(rollout_id: str) -> dict:
            source_entry = source_by_id[rollout_id]
            arrays = _load_source_arrays(source_dir, source_entry)
            try:
                return _write_rollout(
                    cache_dir / f"{rollout_id}.npz", source_entry, arrays,
                    horizon=horizon, gamma=gamma, source_entry=source_entry)
            finally:
                del arrays

        for entry in _bounded_parallel_map(convert, missing_ids, max_workers=2):
            entries_by_id[entry["rollout_id"]] = entry
            completed_since_save += 1
            completed = len(entries_by_id)
            if completed_since_save >= PROGRESS_INTERVAL:
                ordered = [entries_by_id[rid] for rid in wanted_ids if rid in entries_by_id]
                _atomic_json(progress_path, {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "snapshot_id": snapshot.snapshot_id, "horizon": horizon,
                    "gamma": gamma, "rollouts": ordered,
                })
                completed_since_save = 0
            if completed % PROGRESS_INTERVAL == 0 or completed == len(wanted_ids):
                print(
                    f"[qplanning] Q{horizon} local windows: "
                    f"{completed}/{len(wanted_ids)} rollouts",
                    flush=True,
                )
    finally:
        ordered = [entries_by_id[rid] for rid in wanted_ids if rid in entries_by_id]
        _atomic_json(progress_path, {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "snapshot_id": snapshot.snapshot_id, "horizon": horizon,
            "gamma": gamma, "rollouts": ordered,
        })
    entries = [entries_by_id[rollout_id] for rollout_id in wanted_ids]
    train_ids = set(snapshot.train_rollout_ids)
    train_sources = [source_by_id[rollout_id] for rollout_id in snapshot.train_rollout_ids]
    action_count = sum(entry["action_count"] for entry in train_sources)
    if not action_count:
        raise ValueError("snapshot has no train actions")
    action_sum = np.sum([entry["action_sum"] for entry in train_sources], axis=0)
    action_sumsq = np.sum([entry["action_sumsq"] for entry in train_sources], axis=0)
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
            (e["generated_executed_max_abs"] for e in entries), default=float("nan")),
        gamma=gamma)
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION, "gamma": gamma,
        "complete": True, **asdict(result),
    }
    _atomic_json(manifest_path, payload)
    return result


def _streaming_bootstrap_count(n_steps: int, horizon: int) -> int:
    # Collected trajectories replan every ten actions and mark success/truncation on the final
    # executed action. A target exactly at the terminal boundary must therefore not bootstrap.
    return sum(start + horizon < n_steps for start in range(0, n_steps, 10))


def prepare_qplanning_streaming_cache(
        store, snapshot: DatasetSnapshot, *, horizon: int, gamma: float,
        cache_root: str | Path,
        download_workers: int = DEFAULT_DOWNLOAD_WORKERS) -> QPlanningCacheIndex:
    """Index Q10/Q50 windows directly over the shared source cache.

    This avoids a second horizon-specific copy of the large saved prefix embeddings. Windows are
    materialized in RAM one rollout at a time by ``QPlanningWindowDataset``.
    """
    if horizon not in (10, 50):
        raise ValueError("horizon must be 10 or 50")
    if not 1 <= download_workers <= 8:
        raise ValueError("download_workers must be in [1, 8]")
    root = Path(cache_root).expanduser()
    source_dir, source_entries = _prepare_source_cache(
        store, snapshot, cache_root=root, download_workers=download_workers)
    train_ids = set(snapshot.train_rollout_ids)
    val_ids = set(snapshot.val_rollout_ids)
    train_sources = [entry for entry in source_entries if entry["rollout_id"] in train_ids]
    action_count = sum(entry["action_count"] for entry in train_sources)
    if not action_count:
        raise ValueError("snapshot has no train actions")
    action_sum = np.sum([entry["action_sum"] for entry in train_sources], axis=0)
    action_sumsq = np.sum([entry["action_sumsq"] for entry in train_sources], axis=0)
    mean = action_sum / action_count
    variance = np.maximum(action_sumsq / action_count - mean ** 2, 1e-12)

    first = source_entries[0]
    with np.load(source_dir / first["path"], allow_pickle=False) as archive:
        prefix_dim = int(archive["prefix/prefix_embeddings"].shape[-1])
        robot_dim = int(archive["boundary/raw_robot_state"].shape[-1])
        proprio_dim = int(archive["boundary/policy_proprio"].shape[-1])
        action_dim = int(archive["actions_normalized"].shape[-1])

    entries = []
    for source in source_entries:
        n_steps = int(source["action_count"])
        entries.append({
            **source,
            "n_windows": (n_steps + 9) // 10,
            "n_bootstrap": _streaming_bootstrap_count(n_steps, horizon),
            "prefix_dim": prefix_dim,
            "robot_dim": robot_dim,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
        })
    difference_count = sum(entry["generated_executed_count"] for entry in entries)
    result = QPlanningCacheIndex(
        snapshot_id=snapshot.snapshot_id, horizon=horizon, cache_dir=str(source_dir),
        rollouts=tuple(entries), prefix_dim=prefix_dim, robot_dim=robot_dim,
        proprio_dim=proprio_dim, action_dim=action_dim,
        action_mean=tuple(float(value) for value in mean),
        action_std=tuple(float(value) for value in np.sqrt(variance)),
        n_windows=sum(entry["n_windows"] for entry in entries),
        n_train_windows=sum(
            entry["n_windows"] for entry in entries if entry["rollout_id"] in train_ids),
        n_val_windows=sum(
            entry["n_windows"] for entry in entries if entry["rollout_id"] in val_ids),
        n_bootstrap_windows=sum(entry["n_bootstrap"] for entry in entries),
        generated_executed_first10_mae=(
            sum(entry["generated_executed_abs_sum"] for entry in entries) / difference_count
            if difference_count else float("nan")),
        generated_executed_first10_max_abs=max(
            (entry["generated_executed_max_abs"] for entry in entries),
            default=float("nan")),
        storage_mode="source_stream",
        gamma=gamma,
    )
    print(
        f"[qplanning] Q{horizon} windows will stream from source_v1; "
        "no horizon-specific disk cache will be created",
        flush=True,
    )
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
            indices.sort(key=lambda pair: (pair[0]["rollout_id"], pair[1]))
        self.cache_dir = cache.cache_dir
        self.horizon = cache.horizon
        self.gamma = cache.gamma
        self.storage_mode = cache.storage_mode
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
                if self.storage_mode == "source_stream":
                    source = {key: archive[key] for key in QPLANNING_ARTIFACT_FIELDS}
                    arrays = qplanning_windows_from_artifact(
                        entry, source, horizon=self.horizon, gamma=self.gamma)
                else:
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
