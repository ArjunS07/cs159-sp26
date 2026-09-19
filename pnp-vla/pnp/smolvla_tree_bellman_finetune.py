"""Fine-tune the demonstration-pretrained SmolVLA Q10 critic on v3 trees.

Both declared arms use the same immutable tree snapshot, 50/50 demonstration/tree
transition batches, genuine EMA Bellman targets, and the same initialization.
The fork-aware arm changes only two terms: early branch transitions receive larger
loss weights, and mixed trees receive a small same-root listwise selection loss.
No synthetic reward is added and the Bellman target is never reversed.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import copy
import glob
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .pcp_critic.resumable_snapshot import (
    _download_with_retry, load_training_fields_with_retry)
from .qplanning_critic.config import QPlanningModelConfig
from .qplanning_critic.data import (
    QPLANNING_ARTIFACT_FIELDS, QPlanningCacheIndex, QPlanningWindowDataset, collate_windows,
    qplanning_windows_from_artifact)
from .qplanning_critic.model import QPlanningCritic, pool_prefix_tokens
from .smolvla_tree_collection import (
    SMOLVLA_TREE_CANDIDATES, SMOLVLA_TREE_EXPERIMENT)
from .smolvla_tree_critic import TREE_Q10_KINDS
from .store import SupabaseStore


SNAPSHOT_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
CHECKPOINT_FORMAT = "smolvla_tree_bellman_finetune_v1"
DEFAULT_SNAPSHOT_KEY = (
    "smolvla_trees/manifests/v3_bellman_finetune_480_20260919.json")
# Tree collection writes the same Bellman contract as Q-planning. Request the
# complete contract: the shared window builder validates its generated/executed
# action diagnostics even though this fine-tuner does not optimize those arrays.
TREE_FIELDS = QPLANNING_ARTIFACT_FIELDS
ROOT_SOURCE_FIELDS = (
    "prefix/prefix_embeddings", "prefix/prefix_pad_masks",
    "boundary/raw_robot_state", "boundary/policy_proprio", "boundary/step",
    "bellman/action",
)


@dataclass(frozen=True)
class TreeBellmanFineTuneConfig:
    seed: int = 42
    updates: int = 2_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_updates: int = 100
    batch_size: int = 64
    demo_fraction: float = .50
    root_trees_per_update: int = 4
    root_loss_weight: float = 0.0
    root_temperature: float = .10
    fork_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    target_rate: float = .005
    gamma: float = .99
    print_interval: int = 100
    eval_interval: int = 250
    checkpoint_interval: int = 500
    grad_clip: float = 1.0
    use_bf16: bool = True

    def __post_init__(self):
        if self.updates < 1 or self.batch_size < 2:
            raise ValueError("updates and batch_size must be positive")
        if not 0 < self.demo_fraction < 1:
            raise ValueError("demo_fraction must lie in (0,1)")
        if self.root_loss_weight < 0 or self.root_temperature <= 0:
            raise ValueError("invalid root objective")
        if len(self.fork_weights) != 3 or min(self.fork_weights) <= 0:
            raise ValueError("fork_weights must contain three positive values")
        if not 0 < self.target_rate <= 1 or not 0 < self.gamma <= 1:
            raise ValueError("invalid EMA/Bellman parameters")

    def learning_rate_at(self, update: int) -> float:
        if self.warmup_updates and update <= self.warmup_updates:
            return self.learning_rate * update / self.warmup_updates
        span = max(1, self.updates - self.warmup_updates)
        progress = min(1.0, max(0.0, (update - self.warmup_updates) / span))
        return self.learning_rate * .5 * (1 + math.cos(math.pi * progress))


def _json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _fetch_tree_snapshot(store, *, tree_limit: int) -> dict:
    groups = store.fetch_all(
        "verifier_candidate_groups",
        "candidate_group_id,suite,task_idx,episode_idx,chunk_idx,metadata_json",
        configure=lambda query: query.eq("experiment", SMOLVLA_TREE_EXPERIMENT),
        order_by=("candidate_group_id",))
    group_ids = [str(row["candidate_group_id"]) for row in groups]
    candidates = []
    for start in range(0, len(group_ids), 100):
        ids = group_ids[start:start + 100]
        candidates.extend(store.fetch_all(
            "verifier_candidates",
            "candidate_id,candidate_group_id,candidate_kind,success,n_steps,metadata_json",
            configure=lambda query, ids=ids: query.in_("candidate_group_id", ids),
            order_by=("candidate_group_id", "candidate_kind")))
    by_group = defaultdict(list)
    for row in candidates:
        by_group[str(row["candidate_group_id"])].append(row)
    expected = set(TREE_Q10_KINDS)
    complete = []
    for group in groups:
        gid = str(group["candidate_group_id"])
        rows = by_group[gid]
        kinds = {str(row["candidate_kind"]) for row in rows}
        paths = [bool((row.get("metadata_json") or {}).get("training_data_path"))
                 for row in rows]
        if len(rows) == SMOLVLA_TREE_CANDIDATES and kinds == expected and all(paths):
            complete.append(group)
    if len(complete) < tree_limit:
        raise ValueError(
            f"need at least {tree_limit} complete v3 Bellman trees; found {len(complete)}")
    complete.sort(key=lambda row: hashlib.sha256(
        str(row["candidate_group_id"]).encode()).hexdigest())
    selected = complete[:tree_limit]
    rank = {kind: index for index, kind in enumerate(TREE_Q10_KINDS)}
    result = []
    for group in selected:
        gid = str(group["candidate_group_id"])
        rows = sorted(by_group[gid], key=lambda row: rank[str(row["candidate_kind"])])
        result.append({
            "candidate_group_id": gid,
            "suite": str(group["suite"]), "task_idx": int(group["task_idx"]),
            "episode_idx": int(group["episode_idx"]),
            "chunk_idx": int(group["chunk_idx"]),
            "candidates": [{
                "candidate_id": str(row["candidate_id"]),
                "candidate_kind": str(row["candidate_kind"]),
                "success": bool(row["success"]), "n_steps": int(row["n_steps"]),
                "training_data_path": str((row.get("metadata_json") or {})[
                    "training_data_path"]),
                "training_data_start_boundary": int((row.get("metadata_json") or {}).get(
                    "training_data_start_boundary", 0)),
            } for row in rows],
        })
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "experiment": SMOLVLA_TREE_EXPERIMENT,
        "tree_limit": int(tree_limit), "groups": result,
    }
    payload["snapshot_digest"] = _json_digest(payload)
    return payload


def load_or_create_tree_snapshot(*, store, snapshot_key: str,
                                 tree_limit: int) -> dict:
    try:
        payload = json.loads(_download_with_retry(store, snapshot_key))
        created = False
    except Exception:
        payload = _fetch_tree_snapshot(store, tree_limit=tree_limit)
        store._upload(snapshot_key, json.dumps(payload, sort_keys=True).encode())
        # Re-read the shared object so two simultaneously launched workers use
        # whichever immutable contract won the write race.
        payload = json.loads(_download_with_retry(store, snapshot_key))
        created = True
    check = dict(payload)
    digest = check.pop("snapshot_digest", None)
    if (_json_digest(check) != digest
            or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
            or payload.get("experiment") != SMOLVLA_TREE_EXPERIMENT
            or int(payload.get("tree_limit", -1)) != int(tree_limit)):
        raise ValueError(f"shared tree snapshot is incompatible: {snapshot_key}")
    print({
        "tree_snapshot": snapshot_key, "created_now": created,
        "snapshot_digest": digest, "trees": len(payload["groups"]),
    }, flush=True)
    return payload


def _pool_windows(windows: dict[str, np.ndarray], tokens: int = 128) -> None:
    for name, mask_name in (("prefix", "pad"), ("next_prefix", "next_pad")):
        values = torch.from_numpy(np.asarray(windows[name], np.float32))
        masks = torch.from_numpy(np.asarray(windows[mask_name], bool))
        pooled, valid = pool_prefix_tokens(values, masks, tokens)
        windows[name] = pooled.numpy().astype(np.float16)
        windows[mask_name] = valid.numpy().astype(bool)


def _materialize_candidate(store, candidate: dict, group_id: str,
                           destination: Path, *, gamma: float) -> dict:
    path = destination / f"{candidate['candidate_id']}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return {
                "rollout_id": candidate["candidate_id"], "path": path.name,
                "n_windows": int(len(archive["reward"])),
                "candidate_group_id": group_id,
                "candidate_kind": candidate["candidate_kind"],
                "success": bool(candidate["success"]),
            }
    arrays = load_training_fields_with_retry(
        store, candidate["training_data_path"], TREE_FIELDS)
    windows = qplanning_windows_from_artifact(
        {"rollout_id": candidate["candidate_id"]}, arrays, horizon=10, gamma=gamma)
    start = int(candidate.get("training_data_start_boundary", 0))
    if not 0 <= start < len(windows["reward"]):
        raise ValueError(
            f"invalid source boundary {start} for {candidate['candidate_id']} with "
            f"{len(windows['reward'])} windows")
    windows = {name: np.asarray(value[start:]).copy() for name, value in windows.items()}
    _pool_windows(windows)
    windows["fork_offset"] = np.arange(len(windows["reward"]), dtype=np.int32)
    if bool(np.asarray(windows["success"])[0]) != bool(candidate["success"]):
        raise ValueError(f"candidate outcome mismatch for {candidate['candidate_id']}")
    _atomic_npz(path, windows)
    return {
        "rollout_id": candidate["candidate_id"], "path": path.name,
        "n_windows": int(len(windows["reward"])),
        "candidate_group_id": group_id,
        "candidate_kind": candidate["candidate_kind"],
        "success": bool(candidate["success"]),
    }


def _split_groups(groups: list[dict], *, seed: int = 42,
                  train_fraction: float = .80) -> tuple[tuple[str, ...], tuple[str, ...]]:
    buckets = defaultdict(list)
    for group in groups:
        stock_success = bool(group["candidates"][0]["success"])
        buckets[(group["suite"], stock_success)].append(group)
    train, validation = [], []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda row: hashlib.sha256(
            f"{seed}|{row['candidate_group_id']}".encode()).hexdigest())
        n_val = max(1, int(round((1 - train_fraction) * len(rows)))) if len(rows) > 1 else 0
        validation.extend(row["candidate_group_id"] for row in rows[:n_val])
        train.extend(row["candidate_group_id"] for row in rows[n_val:])
    return tuple(sorted(train)), tuple(sorted(validation))


def prepare_tree_bellman_cache(*, snapshot: dict, cache_root: str | Path,
                               download_workers: int = 8, gamma: float = .99,
                               store=None) -> dict:
    store = store or SupabaseStore()
    root = Path(cache_root).expanduser() / snapshot["snapshot_digest"]
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "cache_index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        if (payload.get("schema_version") == CACHE_SCHEMA_VERSION
                and payload.get("snapshot_digest") == snapshot["snapshot_digest"]
                and all((root / entry["path"]).is_file()
                        for entry in payload.get("candidate_entries", []))
                and all((root / entry["path"]).is_file()
                        for entry in payload.get("group_entries", []))):
            print(
                f"[tree-bellman] cache ready: {len(payload['candidate_entries'])} branches, "
                f"{len(payload['group_entries'])} trees", flush=True)
            return payload

    tasks = []
    for group in snapshot["groups"]:
        for candidate in group["candidates"]:
            tasks.append((group["candidate_group_id"], candidate))
    local = threading.local()

    def build(task):
        gid, candidate = task
        worker_store = getattr(local, "store", None)
        if worker_store is None:
            worker_store = store.fork_for_thread()
            local.store = worker_store
        return _materialize_candidate(
            worker_store, candidate, gid, root, gamma=gamma)

    entries = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = [executor.submit(build, task) for task in tasks]
        for future in as_completed(futures):
            entries.append(future.result())
            done = len(entries)
            if done % 20 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (len(tasks) - done)
                print(
                    f"[tree-bellman] branch artifacts {done}/{len(tasks)} | "
                    f"{done / max(elapsed, 1e-6):.2f}/s | ETA {eta / 60:.1f}m",
                    flush=True)
    by_candidate = {entry["rollout_id"]: entry for entry in entries}
    group_entries = []
    for position, group in enumerate(snapshot["groups"], 1):
        rows = [by_candidate[row["candidate_id"]] for row in group["candidates"]]
        root_rows = []
        for row in rows:
            with np.load(root / row["path"], allow_pickle=False) as archive:
                root_rows.append({name: np.asarray(archive[name][0]).copy() for name in (
                    "prefix", "pad", "robot", "proprio", "action", "action_valid")})
        root_file = root / f"group_{group['candidate_group_id']}.npz"
        _atomic_npz(root_file, {
            "prefix": root_rows[0]["prefix"], "pad": root_rows[0]["pad"],
            "robot": root_rows[0]["robot"], "proprio": root_rows[0]["proprio"],
            "actions": np.stack([row["action"] for row in root_rows]),
            "action_valid": np.stack([row["action_valid"] for row in root_rows]),
            "success": np.asarray([row["success"] for row in group["candidates"]], bool),
            "candidate_kinds": np.asarray(TREE_Q10_KINDS, dtype="U32"),
        })
        outcomes = [bool(row["success"]) for row in group["candidates"]]
        group_entries.append({
            "candidate_group_id": group["candidate_group_id"],
            "suite": group["suite"], "task_idx": group["task_idx"],
            "stock_success": outcomes[0], "mixed": len(set(outcomes)) > 1,
            "path": root_file.name,
        })
        if position % 50 == 0 or position == len(snapshot["groups"]):
            print(f"[tree-bellman] root groups {position}/{len(snapshot['groups'])}", flush=True)
    train_groups, validation_groups = _split_groups(snapshot["groups"])
    train_set = set(train_groups)
    validation_set = set(validation_groups)
    candidate_entries = sorted(entries, key=lambda row: row["rollout_id"])
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "snapshot_digest": snapshot["snapshot_digest"],
        "cache_dir": str(root), "candidate_entries": candidate_entries,
        "group_entries": group_entries,
        "train_group_ids": sorted(train_set),
        "validation_group_ids": sorted(validation_set),
        "train_windows": sum(row["n_windows"] for row in candidate_entries
                             if row["candidate_group_id"] in train_set),
        "validation_windows": sum(row["n_windows"] for row in candidate_entries
                                  if row["candidate_group_id"] in validation_set),
    }
    _atomic_json(index_path, payload)
    return payload


class TreeWindowDataset(Dataset):
    def __init__(self, cache: dict, group_ids, *, max_open: int = 8):
        wanted = set(group_ids)
        self.root = Path(cache["cache_dir"])
        self.entries = [row for row in cache["candidate_entries"]
                        if row["candidate_group_id"] in wanted]
        self.indices = [(row, offset) for row in self.entries
                        for offset in range(int(row["n_windows"]))]
        self.max_open = max_open
        self.open: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        entry, offset = self.indices[index]
        arrays = self.open.pop(entry["path"], None)
        if arrays is None:
            with np.load(self.root / entry["path"], allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
            if len(self.open) >= self.max_open:
                self.open.popitem(last=False)
        self.open[entry["path"]] = arrays
        return {name: value[offset] for name, value in arrays.items()}


class RootTreeDataset(Dataset):
    def __init__(self, cache: dict, group_ids):
        wanted = set(group_ids)
        self.root = Path(cache["cache_dir"])
        self.entries = [row for row in cache["group_entries"]
                        if row["candidate_group_id"] in wanted]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        with np.load(self.root / entry["path"], allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _resolve_single(pattern: str | Path) -> Path:
    path = Path(pattern).expanduser()
    if path.is_file():
        return path
    # The snapshot digest is an intermediate wildcard component, e.g.
    # checkpoints/*/demo_q10_pretrain/checkpoint_step_004000.pt. Path.glob on
    # path.parent cannot expand that parent wildcard, so expand the complete
    # pattern instead.
    matches = [Path(value) for value in sorted(glob.glob(str(path)))]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one match for {path}; found {matches}")
    return matches[0]


def _root_prefix(value) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"expected root prefix [tokens,width], got {value.shape}")
    return value.astype(np.float16, copy=False)


def _root_mask(value, length: int) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    value = value.reshape(-1).astype(bool, copy=False)
    if len(value) != length:
        raise ValueError("root prefix/mask lengths differ")
    return value


def _read_policy_chunk(store, path: str, *, horizon: int, action_dim: int):
    payload = _download_with_retry(store, str(path))
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if "actions" not in archive.files:
            raise ValueError(f"candidate policy chunk {path} has no actions")
        value = np.asarray(archive["actions"], np.float32)
    if value.ndim != 2 or value.shape[0] < horizon or value.shape[1] < action_dim:
        raise ValueError(f"candidate policy chunk {path} has shape {value.shape}")
    return value[:horizon, :action_dim].copy()


def prepare_tree_validation_roots(*, snapshot: dict, cache_root: str | Path,
                                  horizon: int, action_dim: int,
                                  download_workers: int = 8, store=None) -> dict:
    """Materialize only held-out fork roots for fast threshold diagnostics."""
    store = store or SupabaseStore()
    _, validation_ids = _split_groups(snapshot["groups"])
    wanted = set(validation_ids)
    root = Path(cache_root).expanduser() / snapshot["snapshot_digest"]
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "validation_root_index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        if (payload.get("snapshot_digest") == snapshot["snapshot_digest"]
                and payload.get("horizon") == horizon
                and payload.get("action_dim") == action_dim
                and set(payload.get("validation_group_ids", ())) == wanted
                and all((root / row["path"]).is_file()
                        for row in payload.get("group_entries", ()) )):
            print(f"[tree-threshold] root cache ready: {len(wanted)} trees", flush=True)
            return payload

    groups = store.fetch_all(
        "verifier_candidate_groups",
        "candidate_group_id,suite,task_idx,episode_idx,metadata_json",
        configure=lambda query: query.eq("experiment", SMOLVLA_TREE_EXPERIMENT),
        order_by=("candidate_group_id",))
    groups = {str(row["candidate_group_id"]): row for row in groups
              if str(row["candidate_group_id"]) in wanted}
    if set(groups) != wanted:
        raise ValueError(f"validation root query found {len(groups)}/{len(wanted)} groups")
    candidates = []
    ids = sorted(wanted)
    for start in range(0, len(ids), 100):
        subset = ids[start:start + 100]
        candidates.extend(store.fetch_all(
            "verifier_candidates",
            "candidate_id,candidate_group_id,candidate_kind,success,policy_chunk_path",
            configure=lambda query, subset=subset: query.in_("candidate_group_id", subset),
            order_by=("candidate_group_id", "candidate_kind")))
    by_group = defaultdict(list)
    for row in candidates:
        by_group[str(row["candidate_group_id"])].append(row)
    rank = {kind: index for index, kind in enumerate(TREE_Q10_KINDS)}
    for gid in wanted:
        rows = by_group[gid]
        if (len(rows) != len(TREE_Q10_KINDS)
                or {str(row["candidate_kind"]) for row in rows} != set(TREE_Q10_KINDS)):
            raise ValueError(f"validation tree {gid} is incomplete")
        rows.sort(key=lambda row: rank[str(row["candidate_kind"])])

    local = threading.local()

    def materialize(gid: str):
        path = root / f"root_{gid}.npz"
        group = groups[gid]
        metadata = group.get("metadata_json") or {}
        worker_store = getattr(local, "store", None)
        if worker_store is None:
            worker_store = store.fork_for_thread()
            local.store = worker_store
        source = load_training_fields_with_retry(
            worker_store, str(metadata["source_training_data_path"]), ROOT_SOURCE_FIELDS)
        boundary = int(metadata["source_boundary_index"])
        prefix = _root_prefix(source["prefix/prefix_embeddings"][boundary])
        pad = _root_mask(source["prefix/prefix_pad_masks"][boundary], len(prefix))
        rows = by_group[gid]
        actions = np.stack([
            _read_policy_chunk(
                worker_store, row["policy_chunk_path"],
                horizon=horizon, action_dim=action_dim)
            for row in rows])
        exact_source = np.asarray(source["bellman/action"][boundary], np.float32)[
            :horizon, :action_dim]
        error = float(np.max(np.abs(exact_source - actions[0])))
        if error > 1e-6:
            raise ValueError(f"stored source mismatch at {gid}: {error:.3g}")
        _atomic_npz(path, {
            "prefix": prefix, "pad": pad,
            "robot": np.asarray(
                source["boundary/raw_robot_state"][boundary], np.float32).reshape(-1),
            "proprio": np.asarray(
                source["boundary/policy_proprio"][boundary], np.float32).reshape(-1),
            "actions": actions,
            "action_valid": np.ones((len(rows), horizon), bool),
            "success": np.asarray([bool(row["success"]) for row in rows], bool),
            "candidate_kinds": np.asarray(TREE_Q10_KINDS, dtype="U32"),
        })
        return {
            "candidate_group_id": gid, "suite": str(group["suite"]),
            "task_idx": int(group["task_idx"]),
            "stock_success": bool(rows[0]["success"]),
            "mixed": len({bool(row["success"]) for row in rows}) > 1,
            "path": path.name,
        }

    entries = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = [executor.submit(materialize, gid) for gid in sorted(wanted)]
        for future in as_completed(futures):
            entries.append(future.result())
            done = len(entries)
            if done % 10 == 0 or done == len(wanted):
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (len(wanted) - done)
                print(
                    f"[tree-threshold] roots {done}/{len(wanted)} | "
                    f"ETA {eta / 60:.1f}m", flush=True)
    entries.sort(key=lambda row: row["candidate_group_id"])
    payload = {
        "snapshot_digest": snapshot["snapshot_digest"],
        "cache_dir": str(root), "horizon": horizon, "action_dim": action_dim,
        "validation_group_ids": sorted(wanted), "group_entries": entries,
    }
    _atomic_json(index_path, payload)
    return payload


def _load_demo_contract(cache_root: str | Path, checkpoint_path: str | Path):
    checkpoint_path = _resolve_single(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "qplanning_critic_v1":
        raise ValueError("not a demonstration Q-Planning checkpoint")
    indices = list(Path(cache_root).expanduser().glob("*/cache_index.json"))
    if len(indices) != 1:
        raise ValueError(f"expected one notebook-94 cache index, found {indices}")
    payload = json.loads(indices[0].read_text())
    entries = tuple(payload["entries"])
    dimensions = payload["dimensions"]
    action_mean = tuple(float(x) for x in checkpoint["model"]["action_mean"].tolist())
    action_std = tuple(float(x) for x in checkpoint["model"]["action_std"].tolist())
    cache = QPlanningCacheIndex(
        snapshot_id=str(checkpoint["snapshot_id"]), horizon=10,
        cache_dir=str(indices[0].parent), rollouts=entries,
        prefix_dim=int(dimensions["prefix"]), robot_dim=int(dimensions["robot"]),
        proprio_dim=int(dimensions["proprio"]), action_dim=int(dimensions["action"]),
        action_mean=action_mean, action_std=action_std,
        n_windows=sum(int(row["n_windows"]) for row in entries),
        n_train_windows=sum(int(row["n_windows"]) for row in entries
                            if row["split"] == "train"),
        n_val_windows=sum(int(row["n_windows"]) for row in entries
                          if row["split"] == "validation"),
        n_bootstrap_windows=0, generated_executed_first10_mae=float("nan"),
        generated_executed_first10_max_abs=float("nan"), gamma=.99)
    train_ids = [row["rollout_id"] for row in entries if row["split"] == "train"]
    val_ids = [row["rollout_id"] for row in entries if row["split"] == "validation"]
    return checkpoint_path, checkpoint, cache, train_ids, val_ids


def _model_from_checkpoint(payload: dict, device):
    architecture = dict(payload["architecture"])
    prefix_dim = int(architecture.pop("prefix_dim"))
    robot_dim = int(architecture.pop("robot_dim"))
    proprio_dim = int(architecture.pop("proprio_dim"))
    model = QPlanningCritic(
        prefix_dim=prefix_dim, robot_dim=robot_dim, proprio_dim=proprio_dim,
        config=QPlanningModelConfig(**architecture))
    target = copy.deepcopy(model)
    model.load_state_dict(payload["model"])
    target.load_state_dict(payload["target"])
    model.to(device)
    target.to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return model, target


def _to(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _target_value(target, batch):
    with torch.no_grad():
        value = target.expected_value(
            batch["next_prefix"], batch["next_pad"], batch["next_robot"],
            batch["next_proprio"], batch["next_action"], batch["next_action_valid"])
        return (batch["reward"] + batch["discount"] * value).clamp(
            target.config.value_min, target.config.value_max)


def _per_sample_hl_loss(model, logits, targets):
    distribution = model.hl_gauss_targets(targets)
    return -(distribution * F.log_softmax(logits.float(), -1)).sum(-1)


def _ema(target, model, rate: float):
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), model.parameters()):
            target_parameter.lerp_(parameter, rate)
        for target_buffer, buffer in zip(target.buffers(), model.buffers()):
            if target_buffer.dtype.is_floating_point:
                target_buffer.lerp_(buffer, rate)
            else:
                target_buffer.copy_(buffer)


def _root_batch(items: list[dict]):
    tokens = max(len(item["prefix"]) for item in items)
    width = items[0]["prefix"].shape[-1]
    prefix = np.zeros((len(items), tokens, width), np.float16)
    pad = np.zeros((len(items), tokens), bool)
    for index, item in enumerate(items):
        length = len(item["prefix"])
        prefix[index, :length] = item["prefix"]
        pad[index, :length] = item["pad"]
    return {
        "prefix": torch.from_numpy(prefix), "pad": torch.from_numpy(pad),
        "robot": torch.from_numpy(np.stack([item["robot"] for item in items])).float(),
        "proprio": torch.from_numpy(np.stack([item["proprio"] for item in items])).float(),
        "action": torch.from_numpy(np.stack([item["actions"] for item in items])).float(),
        "action_valid": torch.from_numpy(np.stack([
            item["action_valid"] for item in items])).bool(),
        "success": torch.from_numpy(np.stack([item["success"] for item in items])).bool(),
    }


def _root_scores(model, batch):
    trees, candidates = batch["action"].shape[:2]
    repeat = lambda value: value.repeat_interleave(candidates, 0)
    logits = model(
        repeat(batch["prefix"]), repeat(batch["pad"]), repeat(batch["robot"]),
        repeat(batch["proprio"]),
        batch["action"].reshape(trees * candidates, 10, -1),
        batch["action_valid"].reshape(trees * candidates, 10))
    return ((logits.float().softmax(-1) * model.value_bins.float()).sum(-1)
            .reshape(trees, candidates))


def _listwise_root_loss(scores, success, *, temperature: float):
    per_tree = []
    scaled = scores / temperature
    for values, labels in zip(scaled, success.bool()):
        if bool(labels.any()) and bool((~labels).any()):
            per_tree.append(torch.logsumexp(values, 0) - torch.logsumexp(values[labels], 0))
    return torch.stack(per_tree).mean() if per_tree else scores.sum() * 0.0


def _auc(labels, scores):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float((positive[:, None] > negative[None, :]).mean()
                 + .5 * (positive[:, None] == negative[None, :]).mean())


@torch.no_grad()
def _evaluate_windows(model, target, dataset, device, *, maximum: int = 4096,
                      seed: int = 7):
    model.eval(); target.eval()
    if not len(dataset):
        return {"ce": float("nan"), "mae": float("nan"), "failure_auc": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    indices = (np.arange(len(dataset)) if len(dataset) <= maximum else
               np.sort(rng.choice(len(dataset), size=maximum, replace=False)))
    losses, predictions, targets, successes = [], [], [], []
    for start in range(0, len(indices), 64):
        items = [dataset[int(index)] for index in indices[start:start + 64]]
        batch = _to(collate_windows(items), device)
        target_value = _target_value(target, batch)
        logits = model(
            batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
            batch["action"], batch["action_valid"])
        losses.extend(_per_sample_hl_loss(model, logits, target_value).cpu().tolist())
        predictions.extend(((logits.float().softmax(-1) * model.value_bins.float()).sum(-1))
                           .cpu().tolist())
        targets.extend(target_value.cpu().tolist())
        successes.extend(batch["success"].cpu().tolist())
    prediction = np.asarray(predictions)
    target_values = np.asarray(targets)
    success = np.asarray(successes, bool)
    return {
        "ce": float(np.mean(losses)), "mae": float(np.abs(prediction - target_values).mean()),
        "failure_auc": _auc(~success, -prediction), "n": int(len(prediction)),
    }


@torch.no_grad()
def _evaluate_roots(model, dataset, device):
    model.eval()
    all_scores, all_success = [], []
    for start in range(0, len(dataset), 8):
        batch = _to(_root_batch([
            dataset[index] for index in range(start, min(len(dataset), start + 8))]), device)
        all_scores.append(_root_scores(model, batch).cpu().numpy())
        all_success.append(batch["success"].cpu().numpy())
    scores = np.concatenate(all_scores)
    success = np.concatenate(all_success)
    selected = scores.argmax(1)
    rows = np.arange(len(scores))
    selected_success = success[rows, selected]
    stock_success = success[:, 0]
    pair_correct = pair_total = 0.0
    for values, labels in zip(scores, success):
        good, bad = np.flatnonzero(labels), np.flatnonzero(~labels)
        if len(good) and len(bad):
            comparisons = values[good, None] - values[bad]
            pair_correct += float((comparisons > 0).sum() + .5 * (comparisons == 0).sum())
            pair_total += comparisons.size
    return {
        "within_tree": float(pair_correct / pair_total) if pair_total else float("nan"),
        "selected_minus_stock_pp": float(100 * (selected_success.mean() - stock_success.mean())),
        "failure_to_success": int((~stock_success & selected_success).sum()),
        "success_to_failure": int((stock_success & ~selected_success).sum()),
        "stock_selected_pct": float(100 * (selected == 0).mean()),
        "trees": int(len(scores)), "mixed_trees": int(sum(len(set(row)) > 1 for row in success)),
    }


def _latest(directory: Path):
    values = list(directory.glob("checkpoint_step_*.pt"))
    return max(values, key=lambda path: int(path.stem.rsplit("_", 1)[-1])) if values else None


def _cpu_state(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _save(path: Path, *, model, target, optimizer, update: int, history: list,
          config: TreeBellmanFineTuneConfig, snapshot_digest: str,
          demo_checkpoint_digest: str, run_name: str):
    payload = {
        "format": CHECKPOINT_FORMAT, "update": update, "history": history,
        "run_name": run_name, "train_config": asdict(config),
        "snapshot_digest": snapshot_digest,
        "demo_checkpoint_digest": demo_checkpoint_digest,
        "architecture": model.architecture_config(),
        "model": _cpu_state(model), "target": _cpu_state(target),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    for candidate in path.parent.glob("checkpoint_step_*.pt"):
        if candidate != path:
            candidate.unlink()


def run_smolvla_tree_bellman_finetune(*,
                                      run_name: str,
                                      fork_aware: bool,
                                      demo_cache_root: str | Path,
                                      demo_checkpoint_path: str | Path,
                                      tree_cache_root: str | Path,
                                      output_root: str | Path,
                                      tree_limit: int = 480,
                                      snapshot_key: str = DEFAULT_SNAPSHOT_KEY,
                                      updates: int = 2_000,
                                      batch_size: int = 64,
                                      download_workers: int = 8,
                                      resume: bool = True,
                                      device=None,
                                      store=None) -> dict:
    """Run one of the two matched demonstration-to-tree fine-tuning arms."""
    store = store or SupabaseStore()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path, demo_payload, demo_cache, demo_train_ids, demo_val_ids = _load_demo_contract(
        demo_cache_root, demo_checkpoint_path)
    demo_digest = _file_digest(checkpoint_path)
    snapshot = load_or_create_tree_snapshot(
        store=store, snapshot_key=snapshot_key, tree_limit=tree_limit)
    tree_cache = prepare_tree_bellman_cache(
        snapshot=snapshot, cache_root=tree_cache_root,
        download_workers=download_workers, gamma=.99, store=store)
    demo_train = QPlanningWindowDataset(demo_cache, demo_train_ids, max_open_rollouts=8)
    demo_validation = QPlanningWindowDataset(demo_cache, demo_val_ids, max_open_rollouts=8)
    tree_train = TreeWindowDataset(tree_cache, tree_cache["train_group_ids"])
    tree_validation = TreeWindowDataset(tree_cache, tree_cache["validation_group_ids"])
    roots_train = RootTreeDataset(tree_cache, tree_cache["train_group_ids"])
    roots_validation = RootTreeDataset(tree_cache, tree_cache["validation_group_ids"])
    mixed_root_indices = np.asarray([
        index for index, entry in enumerate(roots_train.entries) if entry["mixed"]], np.int64)
    if fork_aware and not len(mixed_root_indices):
        raise ValueError("fork-aware training requires at least one mixed training tree")

    config = TreeBellmanFineTuneConfig(
        updates=updates, batch_size=batch_size,
        root_loss_weight=.10 if fork_aware else 0.0,
        fork_weights=(3.0, 2.0, 1.5) if fork_aware else (1.0, 1.0, 1.0))
    # Seed before constructing the optimizer/model. A resumed checkpoint restores
    # its RNG state below and must not be re-seeded afterward.
    torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)
    model, target = _model_from_checkpoint(demo_payload, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir = (Path(output_root).expanduser() / snapshot["snapshot_digest"] / run_name)
    start_update, history = 0, []
    latest = _latest(output_dir) if resume else None
    if latest is not None:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if (payload.get("format") != CHECKPOINT_FORMAT
                or payload.get("snapshot_digest") != snapshot["snapshot_digest"]
                or payload.get("demo_checkpoint_digest") != demo_digest
                or payload.get("train_config") != asdict(config)):
            raise ValueError("existing checkpoint does not match this fine-tuning contract")
        model.load_state_dict(payload["model"]); target.load_state_dict(payload["target"])
        optimizer.load_state_dict(payload["optimizer"])
        model.to(device); target.to(device)
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        start_update = int(payload["update"]); history = list(payload.get("history", []))
        torch.set_rng_state(payload["torch_rng_state"])
        if payload.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        np.random.set_state(payload["numpy_rng_state"]); random.setstate(payload["python_rng_state"])
        print(f"[tree-bellman] resumed {latest.name} at step {start_update}", flush=True)
    demo_batch = int(round(config.batch_size * config.demo_fraction))
    tree_batch = config.batch_size - demo_batch
    print("Training contract", flush=True)
    print({
        "run_name": run_name, "fork_aware": fork_aware,
        "initialization": str(checkpoint_path), "tree_snapshot": snapshot["snapshot_digest"],
        "trees": len(snapshot["groups"]), "mixed_train_trees": len(mixed_root_indices),
        "demo_train_windows": len(demo_train), "tree_train_windows": len(tree_train),
        "demo_tree_batch": [demo_batch, tree_batch],
        "fork_weights_j0_j1_j2": config.fork_weights,
        "root_listwise_weight": config.root_loss_weight,
        "updates": config.updates, "device": str(device), "output_dir": str(output_dir),
    }, flush=True)
    use_amp = bool(config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    started = time.perf_counter(); rolling = defaultdict(float); rolling_count = 0
    for update in range(start_update + 1, config.updates + 1):
        model.train(); target.eval()
        learning_rate = config.learning_rate_at(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        rng = np.random.default_rng(config.seed * 1_000_003 + update)
        demo_indices = rng.choice(len(demo_train), demo_batch, replace=len(demo_train) < demo_batch)
        tree_indices = rng.choice(len(tree_train), tree_batch, replace=len(tree_train) < tree_batch)
        demo_items = [demo_train[int(index)] for index in demo_indices]
        tree_items = [tree_train[int(index)] for index in tree_indices]
        batch = _to(collate_windows(demo_items + tree_items), device)
        weights = torch.ones(config.batch_size, dtype=torch.float32, device=device)
        if fork_aware:
            offsets = np.asarray([int(item["fork_offset"]) for item in tree_items], np.int64)
            local_weights = np.ones(len(offsets), np.float32)
            for boundary, value in enumerate(config.fork_weights):
                local_weights[offsets == boundary] = value
            weights[demo_batch:] = torch.from_numpy(local_weights).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            target_value = _target_value(target, batch)
            logits = model(
                batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
                batch["action"], batch["action_valid"])
            per_sample = _per_sample_hl_loss(model, logits, target_value)
            bellman_loss = (per_sample * weights).sum() / weights.sum()
            root_loss = bellman_loss * 0.0
            if fork_aware:
                root_indices = rng.choice(
                    mixed_root_indices, config.root_trees_per_update,
                    replace=len(mixed_root_indices) < config.root_trees_per_update)
                root_batch = _to(_root_batch([
                    roots_train[int(index)] for index in root_indices]), device)
                root_scores = _root_scores(model, root_batch)
                root_loss = _listwise_root_loss(
                    root_scores, root_batch["success"], temperature=config.root_temperature)
            loss = bellman_loss + config.root_loss_weight * root_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step(); _ema(target, model, config.target_rate)
        predictions = (logits.detach().float().softmax(-1) * model.value_bins.float()).sum(-1)
        rolling["loss"] += float(loss.detach()); rolling["bellman"] += float(bellman_loss.detach())
        rolling["root"] += float(root_loss.detach()); rolling["q"] += float(predictions.mean())
        rolling["target"] += float(target_value.mean()); rolling_count += 1
        if update % config.print_interval == 0 or update == config.updates:
            elapsed = time.perf_counter() - started
            eta = elapsed / max(1, update - start_update) * (config.updates - update)
            gpu = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
            print(
                f"{run_name} step {update}/{config.updates} | "
                f"loss {rolling['loss']/rolling_count:.4f} | "
                f"bellman {rolling['bellman']/rolling_count:.4f} | "
                f"root {rolling['root']/rolling_count:.4f} | "
                f"q {rolling['q']/rolling_count:.4f} | target {rolling['target']/rolling_count:.4f} | "
                f"grad {float(grad_norm):.3f} | lr {learning_rate:.2e} | GPU {gpu:.1f}GB | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m", flush=True)
            rolling.clear(); rolling_count = 0
        if update % config.eval_interval == 0 or update == config.updates:
            demo_metrics = _evaluate_windows(model, target, demo_validation, device, maximum=2048)
            tree_metrics = _evaluate_windows(model, target, tree_validation, device, maximum=4096)
            root_metrics = _evaluate_roots(model, roots_validation, device)
            record = {"update": update, "demo": demo_metrics,
                      "tree": tree_metrics, "root": root_metrics}
            history.append(record)
            print(
                f"validation | demo CE {demo_metrics['ce']:.4f} MAE {demo_metrics['mae']:.4f} | "
                f"tree CE {tree_metrics['ce']:.4f} MAE {tree_metrics['mae']:.4f} "
                f"failure AUC {tree_metrics['failure_auc']:.3f} | "
                f"root within-tree {root_metrics['within_tree']:.3f} | "
                f"selected-stock {root_metrics['selected_minus_stock_pp']:+.1f} pp | "
                f"F->S {root_metrics['failure_to_success']} | "
                f"S->F {root_metrics['success_to_failure']} | "
                f"stock chosen {root_metrics['stock_selected_pct']:.1f}% | "
                f"mixed {root_metrics['mixed_trees']}/{root_metrics['trees']}", flush=True)
        if update % config.checkpoint_interval == 0 or update == config.updates:
            path = output_dir / f"checkpoint_step_{update:06d}.pt"
            _save(
                path, model=model, target=target, optimizer=optimizer, update=update,
                history=history, config=config, snapshot_digest=snapshot["snapshot_digest"],
                demo_checkpoint_digest=demo_digest, run_name=run_name)
            print(f"[tree-bellman] saved {path}", flush=True)
    final = {
        "demo": _evaluate_windows(model, target, demo_validation, device, maximum=2048),
        "tree": _evaluate_windows(model, target, tree_validation, device, maximum=4096),
        "root": _evaluate_roots(model, roots_validation, device),
    }
    return {
        "run_name": run_name, "fork_aware": fork_aware,
        "snapshot_digest": snapshot["snapshot_digest"], "updates": config.updates,
        "validation": final, "history": history,
        "final_checkpoint": str(output_dir / f"checkpoint_step_{config.updates:06d}.pt"),
    }


def analyze_smolvla_tree_advantage_thresholds(*,
                                              checkpoint_path: str | Path,
                                              cache_root: str | Path =
                                              "/content/smolvla_tree_threshold_cache",
                                              tree_limit: int = 480,
                                              snapshot_key: str = DEFAULT_SNAPSHOT_KEY,
                                              download_workers: int = 8,
                                              device=None,
                                              store=None) -> dict:
    """Rescore held-out roots and sweep a stock-relative Q override margin."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    checkpoint_path = _resolve_single(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported tree checkpoint: {checkpoint_path}")
    architecture = payload["architecture"]
    horizon = int(architecture["action_horizon"])
    action_dim = int(architecture["action_dim"])
    store = store or SupabaseStore()
    snapshot = load_or_create_tree_snapshot(
        store=store, snapshot_key=snapshot_key, tree_limit=tree_limit)
    if payload.get("snapshot_digest") != snapshot["snapshot_digest"]:
        raise ValueError("checkpoint and shared tree snapshot differ")
    root_cache = prepare_tree_validation_roots(
        snapshot=snapshot, cache_root=cache_root, horizon=horizon,
        action_dim=action_dim, download_workers=download_workers, store=store)
    dataset = RootTreeDataset(root_cache, root_cache["validation_group_ids"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = _model_from_checkpoint(payload, device)
    model.eval()
    score_parts, success_parts = [], []
    with torch.no_grad():
        for start in range(0, len(dataset), 16):
            batch = _to(_root_batch([
                dataset[index] for index in range(start, min(len(dataset), start + 16))]),
                device)
            score_parts.append(_root_scores(model, batch).cpu().numpy())
            success_parts.append(batch["success"].cpu().numpy())
    scores = np.concatenate(score_parts)
    success = np.concatenate(success_parts).astype(bool)
    suites = np.asarray([entry["suite"] for entry in dataset.entries])
    rows = np.arange(len(scores))
    best_alternative = scores[:, 1:].argmax(1) + 1
    delta = scores[rows, best_alternative] - scores[:, 0]
    alternative_success = success[rows, best_alternative]
    stock_success = success[:, 0]

    def metrics(threshold: float) -> dict:
        override = delta > threshold
        selected = np.where(override, alternative_success, stock_success)
        f_to_s = int((~stock_success & selected).sum())
        s_to_f = int((stock_success & ~selected).sum())
        return {
            "threshold": float(threshold), "overrides": int(override.sum()),
            "override_pct": float(100 * override.mean()),
            "failure_to_success": f_to_s, "success_to_failure": s_to_f,
            "net_flips": f_to_s - s_to_f,
            "selected_minus_stock_pp": float(100 * (selected.mean() - stock_success.mean())),
            "selected_sr_pct": float(100 * selected.mean()),
        }

    sweep = pd.DataFrame([metrics(float("-inf"))] + [
        metrics(value) for value in np.unique(delta)])
    ranked = sweep.sort_values(
        ["net_flips", "success_to_failure", "overrides"],
        ascending=[False, True, True])
    best = ranked.iloc[0]
    conservative_pool = sweep[
        (sweep["overrides"] > 0)
        & (sweep["failure_to_success"] >= sweep["success_to_failure"])]
    conservative = (conservative_pool.sort_values(
        ["net_flips", "success_to_failure", "overrides"],
        ascending=[False, True, True]).iloc[0]
        if len(conservative_pool) else None)
    requested = [
        ("ungated argmax", float("-inf")),
        ("positive advantage", 0.0),
        ("median delta", float(np.quantile(delta, .50))),
        ("top quartile delta", float(np.quantile(delta, .75))),
        ("top decile delta", float(np.quantile(delta, .90))),
        ("best validation net", float(best["threshold"])),
    ]
    if conservative is not None:
        requested.append(("best F->S >= S->F", float(conservative["threshold"])))
    operating = pd.DataFrame([
        {"rule": label, **metrics(threshold)} for label, threshold in requested])
    operating = operating.drop_duplicates(subset=["threshold"], keep="first")

    transition = np.full(len(delta), "same outcome", dtype=object)
    transition[~stock_success & alternative_success] = "F->S"
    transition[stock_success & ~alternative_success] = "S->F"
    delta_summary = pd.DataFrame({"transition": transition, "delta_q": delta}).groupby(
        "transition")["delta_q"].agg(["count", "mean", "median", "min", "max"])
    chosen_threshold = float(
        conservative["threshold"] if conservative is not None else best["threshold"])
    per_suite = []
    for suite in sorted(set(suites)):
        keep = suites == suite
        override = (delta > chosen_threshold) & keep
        selected = stock_success.copy()
        selected[override] = alternative_success[override]
        per_suite.append({
            "suite": suite, "n": int(keep.sum()),
            "overrides": int(override.sum()),
            "failure_to_success": int((keep & ~stock_success & selected).sum()),
            "success_to_failure": int((keep & stock_success & ~selected).sum()),
            "stock_sr_pct": float(100 * stock_success[keep].mean()),
            "selected_sr_pct": float(100 * selected[keep].mean()),
        })
    per_suite = pd.DataFrame(per_suite)

    print({
        "checkpoint": str(checkpoint_path), "checkpoint_update": int(payload["update"]),
        "validation_trees": len(dataset), "mixed_validation_trees": int(sum(
            len(set(row)) > 1 for row in success)),
        "note": "best/safe thresholds are exploratory choices on this validation set",
    }, flush=True)
    print("\nOperating points:")
    display(operating)
    print("\nDelta-Q by outcome of the ungated best alternative:")
    display(delta_summary)
    print(f"\nPer-suite result at exploratory threshold {chosen_threshold:.6f}:")
    display(per_suite)

    finite = sweep[np.isfinite(sweep["threshold"])]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(finite["threshold"], finite["selected_minus_stock_pp"])
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Require best alternative Q - stock Q > threshold",
                ylabel="Selected SR minus stock (pp)", title="Validation threshold sweep")
    groups = [delta[transition == name] for name in ("F->S", "S->F", "same outcome")
              if np.any(transition == name)]
    labels = [name for name in ("F->S", "S->F", "same outcome")
              if np.any(transition == name)]
    axes[1].boxplot(groups, labels=labels, showfliers=True)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Best-alternative Q minus stock Q",
                title="Does the margin separate helpful and harmful overrides?")
    plt.tight_layout(); plt.show()
    return {
        "operating_points": operating, "full_sweep": sweep,
        "delta_summary": delta_summary, "per_suite": per_suite,
        "scores": scores, "success": success, "delta_q": delta,
        "suggested_exploratory_threshold": chosen_threshold,
    }
