"""Frozen 1,600-tree snapshot and disk-small root-MC cache for local training."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import threading

import numpy as np
import torch

from .pcp_critic.resumable_snapshot import (
    _download_with_retry, load_training_fields_with_retry,
)
from .qplanning_critic.model import pool_prefix_tokens
from .smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS
from .smolvla_scaling_trees import TREE_EXPERIMENT as NEW_EXPERIMENT
from .smolvla_tree_bellman_finetune import (
    SNAPSHOT_SCHEMA_VERSION, _atomic_json, _atomic_npz,
    _fetch_tree_snapshot, _json_digest, _root_mask, _root_prefix, _split_groups,
)
from .store import SupabaseStore


COMBINED_EXPERIMENT = "smolvla-libero-fresh8-combined-1600-v1"
COMBINED_SNAPSHOT_KEY = "smolvla_trees/manifests/combined_fresh8_success_q10_1600_v1.json"
SOURCE_EXPERIMENTS = (FRESH8_EXPERIMENT, NEW_EXPERIMENT)
ROOT_CACHE_FORMAT = "smolvla-success-q10-root-only-v1"
ROOT_FIELDS = (
    "prefix/prefix_embeddings", "prefix/prefix_pad_masks",
    "boundary/raw_robot_state", "boundary/policy_proprio",
    "boundary/step", "bellman/action",
)


def load_or_create_combined_snapshot(store=None) -> dict:
    """Freeze both complete 800-tree cohorts without changing either source snapshot."""
    store = store or SupabaseStore()
    source_snapshots = [
        _fetch_tree_snapshot(
            store, tree_limit=800, experiment=experiment,
            candidate_kinds=FRESH8_KINDS, include_root_sources=True)
        for experiment in SOURCE_EXPERIMENTS
    ]
    groups = [group for source in source_snapshots for group in source["groups"]]
    ids = [group["candidate_group_id"] for group in groups]
    if len(groups) != 1600 or len(set(ids)) != 1600:
        raise ValueError("combined snapshot needs 1,600 distinct complete trees")
    candidate_ids = [candidate["candidate_id"] for group in groups
                     for candidate in group["candidates"]]
    if len(candidate_ids) != 14_400 or len(set(candidate_ids)) != 14_400:
        raise ValueError("combined snapshot needs 14,400 distinct candidates")
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "experiment": COMBINED_EXPERIMENT,
        "tree_limit": 1600,
        "candidate_kinds": list(FRESH8_KINDS),
        "source_snapshots": [
            {"experiment": experiment, "digest": source["snapshot_digest"], "trees": 800}
            for experiment, source in zip(SOURCE_EXPERIMENTS, source_snapshots)
        ],
        "groups": groups,
    }
    payload["snapshot_digest"] = _json_digest(payload)
    try:
        saved = json.loads(_download_with_retry(store, COMBINED_SNAPSHOT_KEY))
    except Exception as error:
        if not any(token in str(error).lower() for token in ("404", "not found", "does not exist")):
            raise
        store._upload(COMBINED_SNAPSHOT_KEY, json.dumps(payload, sort_keys=True).encode())
        saved = json.loads(_download_with_retry(store, COMBINED_SNAPSHOT_KEY))
    if saved != payload:
        raise ValueError("persisted combined snapshot differs from the 1,600 completed trees")
    print({"snapshot": COMBINED_SNAPSHOT_KEY, "digest": saved["snapshot_digest"],
           "trees": len(saved["groups"]), "sources": saved["source_snapshots"]}, flush=True)
    return saved


def _read_root_group(store, group: dict, destination: Path) -> dict:
    path = destination / f"group_{group['candidate_group_id']}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if (set(("prefix", "pad", "robot", "proprio", "actions",
                     "action_valid", "success")) <= set(archive.files)
                    and archive["actions"].shape == (9, 10, 7)):
                return _group_entry(group, path)
    arrays = load_training_fields_with_retry(
        store, group["source_training_data_path"], ROOT_FIELDS)
    boundary = int(group["source_boundary_index"])
    prefix = _root_prefix(arrays["prefix/prefix_embeddings"][boundary])
    pad = _root_mask(arrays["prefix/prefix_pad_masks"][boundary], len(prefix))
    pooled, valid = pool_prefix_tokens(
        torch.from_numpy(prefix.astype(np.float32))[None],
        torch.from_numpy(pad)[None], 128)
    root_step = int(np.asarray(arrays["boundary/step"])[boundary])
    actions, masks, outcomes = [], [], []
    for candidate in group["candidates"]:
        with np.load(io.BytesIO(_download_with_retry(store, candidate["policy_chunk_path"])),
                     allow_pickle=False) as archive:
            chunk = np.asarray(archive["actions"], np.float32)
        if chunk.ndim != 2 or chunk.shape[0] < 10 or chunk.shape[1] < 7:
            raise ValueError(f"invalid candidate action shape for {candidate['candidate_id']}")
        width = min(10, int(candidate["n_steps"]) - root_step)
        if width < 1:
            raise ValueError(f"candidate has no executed action at root: {candidate['candidate_id']}")
        action = np.zeros((10, 7), np.float32)
        action[:width] = chunk[:width, :7]
        mask = np.arange(10) < width
        actions.append(action)
        masks.append(mask)
        outcomes.append(bool(candidate["success"]))
    source_action = np.asarray(arrays["bellman/action"][boundary], np.float32)
    width = int(masks[0].sum())
    if float(np.max(np.abs(source_action[:width, :7] - actions[0][:width]))) > 1e-6:
        raise ValueError(f"stored source action mismatch for {group['candidate_group_id']}")
    _atomic_npz(path, {
        "prefix": pooled[0].numpy().astype(np.float16),
        "pad": valid[0].numpy(),
        "robot": np.asarray(arrays["boundary/raw_robot_state"][boundary], np.float32).reshape(-1),
        "proprio": np.asarray(arrays["boundary/policy_proprio"][boundary], np.float32).reshape(-1),
        "actions": np.stack(actions), "action_valid": np.stack(masks),
        "success": np.asarray(outcomes, bool),
    })
    return _group_entry(group, path)


def _group_entry(group: dict, path: Path) -> dict:
    outcomes = [bool(row["success"]) for row in group["candidates"]]
    return {"candidate_group_id": group["candidate_group_id"], "path": path.name,
            "suite": group["suite"], "task_idx": group["task_idx"],
            "stock_success": outcomes[0], "mixed": len(set(outcomes)) > 1,
            "source_experiment": group["source_experiment"]}


def prepare_combined_root_cache(*, snapshot: dict, cache_root: str | Path,
                                store=None, download_workers: int = 4) -> dict:
    """Cache one state and nine executed Q10 chunks per tree; no TD continuations."""
    if snapshot.get("experiment") != COMBINED_EXPERIMENT or len(snapshot.get("groups", ())) != 1600:
        raise ValueError("expected the frozen 1,600-root combined snapshot")
    if not 1 <= download_workers <= 8:
        raise ValueError("download_workers must be in [1,8]")
    store = store or SupabaseStore()
    root = Path(cache_root).expanduser() / ROOT_CACHE_FORMAT / snapshot["snapshot_digest"]
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "cache_index.json"
    groups = snapshot["groups"]
    by_id = {group["candidate_group_id"]: group for group in groups}
    if len(by_id) != len(groups):
        raise ValueError("duplicate combined tree identity")
    ready = {}
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        if index.get("format") == ROOT_CACHE_FORMAT and index.get("snapshot_digest") == snapshot["snapshot_digest"]:
            ready = {entry["candidate_group_id"]: entry
                     for entry in index.get("group_entries", ())
                     if entry["candidate_group_id"] in by_id and (root / entry["path"]).is_file()}
    missing = [group for group in groups if group["candidate_group_id"] not in ready]
    local = threading.local()

    def build(group):
        worker_store = getattr(local, "store", None)
        if worker_store is None:
            worker_store = store.fork_for_thread()
            local.store = worker_store
        return _read_root_group(worker_store, group, root)

    print({"root_cache": str(root), "ready": len(ready), "missing": len(missing),
           "download_workers": download_workers}, flush=True)
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = [executor.submit(build, group) for group in missing]
        for future in as_completed(futures):
            entry = future.result()
            ready[entry["candidate_group_id"]] = entry
            if len(ready) % 20 == 0 or len(ready) == len(groups):
                print({"root_cache_trees": len(ready), "total": len(groups)}, flush=True)
                _atomic_json(index_path, {"format": ROOT_CACHE_FORMAT,
                                          "snapshot_digest": snapshot["snapshot_digest"],
                                          "group_entries": list(ready.values())})
    if len(ready) != len(groups):
        raise ValueError("root cache is incomplete")
    train, validation = [], []
    for experiment in SOURCE_EXPERIMENTS:
        cohort = [group for group in groups if group["source_experiment"] == experiment]
        if len(cohort) != 800:
            raise ValueError(f"expected 800 roots from {experiment}, found {len(cohort)}")
        cohort_train, cohort_validation = _split_groups(cohort)
        train.extend(cohort_train)
        validation.extend(cohort_validation)
    if set(train) & set(validation) or len(train) + len(validation) != 1600:
        raise ValueError("combined train/validation split is not disjoint and complete")
    index = {"format": ROOT_CACHE_FORMAT, "snapshot_digest": snapshot["snapshot_digest"],
             "cache_dir": str(root), "group_entries": [ready[group["candidate_group_id"]]
                                                       for group in groups],
             "candidate_entries": [], "train_windows": 0, "validation_windows": 0,
             "train_group_ids": sorted(train), "validation_group_ids": sorted(validation),
             "gamma": 1.0, "success_reward": True, "root_only": True,
             "cohort_counts": dict(Counter(group["source_experiment"] for group in groups))}
    _atomic_json(index_path, index)
    print({"root_cache_ready": len(groups), "train_roots": len(train),
           "validation_roots": len(validation)}, flush=True)
    return index
