"""Leakage-safe trajectory pretraining followed by same-root Q10 ranking.

Only training groups from the frozen combined snapshot are downloaded here.
Each group contributes its full source rollout and two predeclared fresh-noise
continuations. The branch root window is omitted: all nine root actions are
used together during ranking fine-tuning instead.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import torch
import torch.nn.functional as F

from .config import resolve_max_steps
from .pcp_critic.resumable_snapshot import load_training_fields_with_retry
from .qplanning_critic.model import pool_prefix_tokens
from .smolvla_tree_bellman_finetune import _atomic_json, _atomic_npz


CACHE_FORMAT = "smolvla_two_stage_trajectory_v2_full_chunks"
LEGACY_CACHE_FORMAT = "smolvla_two_stage_trajectory_v1"
PRETRAIN_KINDS = ("stored_source", "fresh_seed_1", "fresh_seed_5")
TRAJECTORY_FIELDS = (
    "prefix/prefix_embeddings", "prefix/prefix_pad_masks",
    "boundary/raw_robot_state", "boundary/policy_proprio", "boundary/step",
    "bellman/action", "bellman/validity_mask", "step_success",
)


def trajectory_spec(group: dict, kind: str) -> dict:
    candidate = next(row for row in group["candidates"]
                     if row["candidate_kind"] == kind)
    if kind == "stored_source":
        path = group["source_training_data_path"]
        start, step_offset = 0, 0
    else:
        path = candidate["training_data_path"]
        # The branch artifact starts at the fork and has local boundary steps.
        # Its first window is the same root comparison used in stage two.
        start, step_offset = 1, 10 * int(group["chunk_idx"])
    return {
        "group_id": group["candidate_group_id"], "kind": kind,
        "path": str(path), "start": start, "step_offset": step_offset,
        "suite": group["suite"], "success": bool(candidate["success"]),
    }


def _materialize(store, spec: dict, destination: Path,
                 legacy_dir: Path | None = None) -> dict:
    key = hashlib.sha256(
        (spec["group_id"] + "|" + spec["kind"]).encode()).hexdigest()[:24]
    path = destination / f"trajectory_{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as old:
            if (str(old["group_id"]) != spec["group_id"]
                    or str(old["kind"]) != spec["kind"]
                    or str(old["format"]) != CACHE_FORMAT
                    or len(old["success"]) < 1):
                raise ValueError(f"invalid cached trajectory {path}")
            return {**spec, "file": path.name, "windows": len(old["success"])}
    if legacy_dir is not None:
        legacy_path = legacy_dir / path.name
        if legacy_path.is_file():
            with np.load(legacy_path, allow_pickle=False) as old:
                if (str(old["group_id"]) != spec["group_id"]
                        or str(old["kind"]) != spec["kind"]
                        or str(old["format"]) != LEGACY_CACHE_FORMAT
                        or not np.all(old["success"] == spec["success"])):
                    raise ValueError(f"invalid legacy trajectory {legacy_path}")
                keep = np.asarray(old["action_valid"], bool).all(axis=1)
                if not keep.any():
                    return {**spec, "file": None, "windows": 0}
                # The old cache preserved all proposed actions on full ten-
                # action windows. Its partial terminal windows zeroed future
                # actions, so discard those rather than importing leakage.
                _atomic_npz(path, {
                    "format": np.asarray(CACHE_FORMAT),
                    "group_id": np.asarray(spec["group_id"]),
                    "kind": np.asarray(spec["kind"]),
                    **{name: np.asarray(old[name])[keep].copy() for name in (
                        "prefix", "pad", "robot", "proprio", "action",
                        "action_valid", "success")},
                })
                return {**spec, "file": path.name, "windows": int(keep.sum())}
    arrays = load_training_fields_with_retry(
        store, spec["path"], TRAJECTORY_FIELDS)
    actions = np.asarray(arrays["bellman/action"], np.float32)
    valid = np.asarray(arrays["bellman/validity_mask"], bool)
    steps = np.asarray(arrays["boundary/step"], np.int32)
    robot = np.asarray(arrays["boundary/raw_robot_state"], np.float32)
    proprio = np.asarray(arrays["boundary/policy_proprio"], np.float32)
    prefix = np.asarray(arrays["prefix/prefix_embeddings"], np.float16)
    pad = np.asarray(arrays["prefix/prefix_pad_masks"], bool)
    if prefix.ndim == 4 and prefix.shape[1] == 1:
        prefix = prefix[:, 0]
    if pad.ndim == 3 and pad.shape[1] == 1:
        pad = pad[:, 0]
    n = len(actions)
    if (actions.ndim != 3 or actions.shape[1] < 10 or actions.shape[2] < 7
            or valid.shape != (n, 10)
            or not (len(steps) == len(robot) == len(proprio)
                    == len(prefix) == len(pad) == n + 1)):
        raise ValueError(f"malformed trajectory {spec['group_id']}:{spec['kind']}")
    if bool(np.asarray(arrays["step_success"]).any()) != spec["success"]:
        raise ValueError(f"trajectory success disagrees with candidate label: {spec['group_id']}")
    start = spec["start"]
    # A full proposed chunk is available at every decision boundary, but the
    # executed-validity mask reveals when the episode ended. Exclude partial
    # transitions from pretraining; root comparisons retain the full proposal.
    kept = np.flatnonzero(valid[start:n].all(axis=1)) + start
    if not len(kept):
        if spec["kind"] == "stored_source":
            raise ValueError(f"source has no full Q10 windows: {spec['group_id']}")
        # A branch can finish during the ten intervened actions. It still
        # belongs in the root-ranking stage, but has no continuation example.
        return {**spec, "file": None, "windows": 0}
    pooled, pooled_valid = pool_prefix_tokens(
        torch.from_numpy(prefix[kept]), torch.from_numpy(pad[kept]), 128)
    maximum = resolve_max_steps(spec["suite"])
    absolute_steps = steps[kept] + spec["step_offset"]
    if np.any(absolute_steps < 0) or np.any(absolute_steps + 10 > maximum):
        raise ValueError(f"trajectory boundary time out of range: {spec['group_id']}")
    time_left = ((maximum - absolute_steps) / maximum).astype(np.float32)
    robot_with_time = np.concatenate(
        [robot[kept], time_left[:, None]], axis=1)
    action = np.asarray(actions[kept, :10, :7], np.float32).copy()
    mask = np.ones((len(kept), 10), bool)
    _atomic_npz(path, {
        "format": np.asarray(CACHE_FORMAT), "group_id": np.asarray(spec["group_id"]),
        "kind": np.asarray(spec["kind"]),
        "prefix": pooled.numpy().astype(np.float16),
        "pad": pooled_valid.numpy().astype(bool),
        "robot": robot_with_time,
        "proprio": proprio[kept].copy(),
        "action": action, "action_valid": mask,
        "success": np.full(len(kept), spec["success"], bool),
    })
    return {**spec, "file": path.name, "windows": len(kept)}


def prepare_trajectory_cache(*, snapshot: dict, root_cache: dict,
                             cache_root: str | Path, store,
                             workers: int = 4) -> dict:
    """Persist only training-group source/branch windows; resume per trajectory."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    train_ids = set(root_cache["train_group_ids"])
    validation_ids = set(root_cache["validation_group_ids"])
    groups = snapshot["groups"]
    if (len(train_ids) != 1280 or len(validation_ids) != 320
            or train_ids & validation_ids
            or {g["candidate_group_id"] for g in groups} != train_ids | validation_ids):
        raise ValueError("trajectory cache requires the frozen 1280/320 root split")
    tasks = [trajectory_spec(group, kind)
             for group in groups if group["candidate_group_id"] in train_ids
             for kind in PRETRAIN_KINDS]
    if len(tasks) != 3840 or len({row["path"] for row in tasks}) != 3840:
        raise ValueError("trajectory pretraining paths are missing or duplicated")
    destination = Path(cache_root).expanduser() / CACHE_FORMAT / snapshot["snapshot_digest"]
    legacy_dir = (Path(cache_root).expanduser() / LEGACY_CACHE_FORMAT
                  / snapshot["snapshot_digest"])
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "cache_index.json"
    ready = {}
    if index_path.exists():
        old = json.loads(index_path.read_text())
        if old.get("format") == CACHE_FORMAT and old.get("snapshot_digest") == snapshot["snapshot_digest"]:
            ready = {(row["group_id"], row["kind"]): row for row in old["entries"]
                     if row["file"] is None or (destination / row["file"]).is_file()}
    # Recover files completed by workers after the last index flush.
    for spec in tasks:
        key = (spec["group_id"], spec["kind"])
        if key not in ready:
            digest = hashlib.sha256((spec["group_id"] + "|" + spec["kind"]).encode()).hexdigest()[:24]
            path = destination / f"trajectory_{digest}.npz"
            if path.is_file():
                ready[key] = _materialize(store, spec, destination, legacy_dir)
    pending = [row for row in tasks if (row["group_id"], row["kind"]) not in ready]
    local = threading.local()

    def build(spec):
        worker_store = getattr(local, "store", None)
        if worker_store is None:
            worker_store = store.fork_for_thread()
            local.store = worker_store
        return _materialize(worker_store, spec, destination, legacy_dir)

    print({"trajectory_cache": str(destination), "ready": len(ready),
           "pending": len(pending), "total": len(tasks)}, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build, spec): spec for spec in pending}
        for future in as_completed(futures):
            try:
                entry = future.result()
            except Exception as error:
                executor.shutdown(wait=False, cancel_futures=True)
                spec = futures[future]
                raise RuntimeError(
                    f"failed trajectory {spec['group_id']}:{spec['kind']}") from error
            ready[(entry["group_id"], entry["kind"])] = entry
            if len(ready) % 40 == 0 or len(ready) == len(tasks):
                index = {
                    "format": CACHE_FORMAT, "snapshot_digest": snapshot["snapshot_digest"],
                    "train_group_ids": sorted(train_ids), "validation_group_ids": sorted(validation_ids),
                    "entries": [ready[(row["group_id"], row["kind"])]
                                for row in tasks if (row["group_id"], row["kind"]) in ready],
                }
                _atomic_json(index_path, index)
                print({"trajectory_cache_ready": len(ready), "total": len(tasks)}, flush=True)
    index = json.loads(index_path.read_text())
    if len(index["entries"]) != len(tasks):
        raise ValueError("trajectory cache incomplete")
    index["cache_dir"] = str(destination)
    print({"trajectory_windows": sum(row["windows"] for row in index["entries"]),
           "trajectories": sum(row["windows"] > 0 for row in index["entries"]),
           "root_finished_branches": sum(row["windows"] == 0 for row in index["entries"])},
          flush=True)
    return index


class TrajectorySampler:
    """Select roots uniformly, then one of their trajectories and a window."""

    def __init__(self, index: dict, *, max_open: int = 24):
        self.root = Path(index["cache_dir"])
        self.group_ids = sorted(index["train_group_ids"])
        self.entries = {(row["group_id"], row["kind"]): row for row in index["entries"]}
        self.available = {
            group_id: [kind for kind in PRETRAIN_KINDS
                       if self.entries[(group_id, kind)]["windows"] > 0]
            for group_id in self.group_ids
        }
        self.open: OrderedDict[str, dict] = OrderedDict()
        self.max_open = max_open

    def sample(self, rng: np.random.Generator, batch_size: int) -> list[dict]:
        groups = rng.choice(self.group_ids, size=batch_size,
                            replace=batch_size > len(self.group_ids))
        items = []
        for group_id in groups:
            available = self.available[str(group_id)]
            kind = available[int(rng.integers(len(available)))]
            row = self.entries[(str(group_id), kind)]
            arrays = self.open.pop(row["file"], None)
            if arrays is None:
                with np.load(self.root / row["file"], allow_pickle=False) as archive:
                    arrays = {name: np.asarray(archive[name]).copy()
                              for name in ("prefix", "pad", "robot", "proprio",
                                           "action", "action_valid", "success")}
                if len(self.open) >= self.max_open:
                    self.open.popitem(last=False)
            self.open[row["file"]] = arrays
            offset = int(rng.integers(row["windows"]))
            items.append({name: values[offset] for name, values in arrays.items()})
        return items


def single_batch(items: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "prefix": torch.from_numpy(np.stack([x["prefix"] for x in items])),
        "pad": torch.from_numpy(np.stack([x["pad"] for x in items])),
        "robot": torch.from_numpy(np.stack([x["robot"] for x in items])).float(),
        "proprio": torch.from_numpy(np.stack([x["proprio"] for x in items])).float(),
        "action": torch.from_numpy(np.stack([x["action"] for x in items])).float(),
        "action_valid": torch.from_numpy(np.stack([x["action_valid"] for x in items])).bool(),
        "success": torch.from_numpy(np.asarray([x["success"] for x in items])).float(),
    }


def score_single(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.expected_value(
        batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
        batch["action"], batch["action_valid"]).float().clamp(1e-5, 1 - 1e-5)


def pairwise_root_loss(scores: torch.Tensor, labels: torch.Tensor,
                       *, temperature: float = 0.1) -> torch.Tensor:
    """Mean logistic loss over unlike-outcome pairs, averaged by mixed root."""
    per_root = []
    for values, success in zip(scores, labels.bool()):
        good, bad = values[success], values[~success]
        if len(good) and len(bad):
            per_root.append(F.softplus(
                -(good[:, None] - bad[None, :]) / temperature).mean())
    return torch.stack(per_root).mean() if per_root else scores.sum() * 0
