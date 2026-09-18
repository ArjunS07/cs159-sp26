"""Matched Q10 training on the SmolVLA depth-1 counterfactual trees.

The two intended arms differ only in ``difference_weight``. Candidate actions are
the ten policy-space actions actually executed at the fork. Every candidate is
supervised with its observed discounted terminal return; the optional auxiliary
loss explicitly matches each candidate's predicted return gap to the exact stored
source candidate's observed return gap at the same state.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
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

from .pcp_critic.resumable_snapshot import (
    _download_with_retry, load_training_fields_with_retry)
from .qplanning_critic.config import QPlanningModelConfig
from .qplanning_critic.model import QPlanningCritic
from .smolvla_tree_collection import SMOLVLA_TREE_CANDIDATES, SMOLVLA_TREE_EXPERIMENT
from .store import SupabaseStore


TREE_Q10_SCHEMA_VERSION = 1
TREE_Q10_HORIZON = 10
TREE_Q10_ACTION_DIM = 7
TREE_Q10_EXPECTED_TREES = 521
TREE_Q10_GAMMA = 0.99
TREE_Q10_KINDS = (
    "stored_source",
    "fresh_seed_1", "fresh_seed_2", "fresh_seed_3", "fresh_seed_4",
    "pnp_perturb_1", "pnp_perturb_2", "pnp_perturb_3", "pnp_perturb_4",
)
_SOURCE_FIELDS = (
    "prefix/prefix_embeddings", "prefix/prefix_pad_masks",
    "boundary/raw_robot_state", "boundary/policy_proprio", "boundary/step",
    "bellman/action",
)


@dataclass(frozen=True)
class SmolVLATreeQ10TrainConfig:
    seed: int = 42
    gamma: float = TREE_Q10_GAMMA
    updates: int = 2_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_updates: int = 200
    effective_tree_batch: int = 8
    micro_tree_batch: int = 2
    difference_weight: float = 0.0
    print_interval: int = 100
    eval_interval: int = 250
    checkpoint_interval: int = 500
    grad_clip: float = 1.0
    use_bf16: bool = True

    def __post_init__(self):
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0,1]")
        if self.updates < 1 or self.warmup_updates < 0:
            raise ValueError("invalid update schedule")
        if self.effective_tree_batch < 1 or self.micro_tree_batch < 1:
            raise ValueError("tree batches must be positive")
        if self.effective_tree_batch % self.micro_tree_batch:
            raise ValueError("effective_tree_batch must divide by micro_tree_batch")
        if self.difference_weight < 0:
            raise ValueError("difference_weight must be nonnegative")

    @property
    def accumulation_steps(self) -> int:
        return self.effective_tree_batch // self.micro_tree_batch

    def learning_rate_at(self, update: int) -> float:
        if self.warmup_updates and update <= self.warmup_updates:
            return self.learning_rate * update / self.warmup_updates
        span = max(1, self.updates - self.warmup_updates)
        progress = min(1.0, max(0.0, (update - self.warmup_updates) / span))
        return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def discounted_fork_return(success: bool, *, n_steps: int, root_step: int,
                           gamma: float = TREE_Q10_GAMMA) -> float:
    """Terminal reward discounted per executed action from the fork boundary."""
    if not success:
        return 0.0
    remaining = max(1, int(n_steps) - int(root_step))
    return float(gamma ** (remaining - 1))


def stock_relative_difference_loss(prediction: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
    """Huber loss on eight candidate-minus-stock gaps in each nine-way tree."""
    if prediction.ndim != 2 or prediction.shape[1] != SMOLVLA_TREE_CANDIDATES:
        raise ValueError("prediction must be [trees,9] with stored_source first")
    if target.shape != prediction.shape:
        raise ValueError("target shape differs from prediction")
    predicted_gap = prediction[:, 1:] - prediction[:, :1]
    target_gap = target[:, 1:] - target[:, :1]
    return F.smooth_l1_loss(predicted_gap, target_gap)


def _json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _complete_tree_rows(store, *, expected_trees: int):
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
            "candidate_id,candidate_group_id,candidate_kind,success,n_steps,"
            "policy_chunk_path,metadata_json",
            configure=lambda query, ids=ids: query.in_("candidate_group_id", ids),
            order_by=("candidate_group_id", "candidate_kind")))
    by_group = defaultdict(list)
    for row in candidates:
        by_group[str(row["candidate_group_id"])].append(row)
    complete = []
    expected_kinds = set(TREE_Q10_KINDS)
    for group in groups:
        rows = by_group[str(group["candidate_group_id"])]
        if (len(rows) == SMOLVLA_TREE_CANDIDATES
                and {str(row["candidate_kind"]) for row in rows} == expected_kinds):
            complete.append(group)
    if len(complete) != expected_trees:
        raise ValueError(
            f"expected exactly {expected_trees} frozen complete trees, found {len(complete)}; "
            "do not let the two ablation workers train on different database snapshots")
    complete_ids = {str(row["candidate_group_id"]) for row in complete}
    by_group = {key: value for key, value in by_group.items() if key in complete_ids}
    contract = []
    for group in complete:
        gid = str(group["candidate_group_id"])
        contract.append({
            "group": gid,
            "candidates": [(str(row["candidate_id"]), str(row["candidate_kind"]),
                            bool(row["success"]), int(row["n_steps"]),
                            str(row["policy_chunk_path"]))
                           for row in sorted(by_group[gid], key=lambda row: row["candidate_kind"])],
        })
    return complete, by_group, _json_digest(contract)


def _candidate_order(rows: list[dict]) -> list[dict]:
    rank = {kind: index for index, kind in enumerate(TREE_Q10_KINDS)}
    ordered = sorted(rows, key=lambda row: rank[str(row["candidate_kind"])])
    if [str(row["candidate_kind"]) for row in ordered] != list(TREE_Q10_KINDS):
        raise ValueError("tree candidate kinds do not match the frozen nine-way contract")
    return ordered


def _read_chunk(store, path: str) -> np.ndarray:
    payload = _download_with_retry(store, str(path))
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if "actions" not in archive.files:
            raise ValueError(f"candidate chunk {path} has no actions")
        value = np.asarray(archive["actions"], np.float32)
    if (value.ndim != 2 or value.shape[0] < TREE_Q10_HORIZON
            or value.shape[1] < TREE_Q10_ACTION_DIM):
        raise ValueError(f"invalid candidate chunk {value.shape} at {path}")
    return value[:TREE_Q10_HORIZON, :TREE_Q10_ACTION_DIM].copy()


def _prefix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"expected root prefix [tokens,width], got {value.shape}")
    return value.astype(np.float16, copy=False)


def _mask(value: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    value = value.reshape(-1).astype(bool, copy=False)
    if len(value) != length:
        raise ValueError("root prefix mask length mismatch")
    return value


def _write_group(path: Path, store, group: dict, candidates: list[dict], *, gamma: float):
    metadata = group.get("metadata_json") or {}
    arrays = load_training_fields_with_retry(
        store, str(metadata["source_training_data_path"]), _SOURCE_FIELDS)
    boundary_index = int(metadata["source_boundary_index"])
    prefix = _prefix(arrays["prefix/prefix_embeddings"][boundary_index])
    pad = _mask(arrays["prefix/prefix_pad_masks"][boundary_index], len(prefix))
    robot = np.asarray(arrays["boundary/raw_robot_state"][boundary_index], np.float32).reshape(-1)
    proprio = np.asarray(arrays["boundary/policy_proprio"][boundary_index], np.float32).reshape(-1)
    root_step = int(np.asarray(arrays["boundary/step"])[boundary_index])
    ordered = _candidate_order(candidates)
    actions = np.stack([_read_chunk(store, row["policy_chunk_path"]) for row in ordered])
    source_action = np.asarray(arrays["bellman/action"][boundary_index], np.float32)
    source_action = source_action[:TREE_Q10_HORIZON, :TREE_Q10_ACTION_DIM]
    source_error = float(np.max(np.abs(source_action - actions[0])))
    if source_error > 1e-6:
        raise ValueError(
            f"stored-source candidate differs from source training action by {source_error:.3g}")
    successes = np.asarray([bool(row["success"]) for row in ordered], bool)
    n_steps = np.asarray([int(row["n_steps"]) for row in ordered], np.int32)
    targets = np.asarray([
        discounted_fork_return(success, n_steps=steps, root_step=root_step, gamma=gamma)
        for success, steps in zip(successes, n_steps)], np.float32)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, prefix=prefix, pad=pad, robot=robot, proprio=proprio,
            actions=actions, targets=targets, successes=successes, n_steps=n_steps,
            root_step=np.asarray(root_step, np.int32),
            candidate_kinds=np.asarray(TREE_Q10_KINDS, dtype="U32"))
    os.replace(temporary, path)
    return {
        "candidate_group_id": str(group["candidate_group_id"]), "path": path.name,
        "suite": str(group["suite"]), "task_idx": int(group["task_idx"]),
        "episode_idx": int(group["episode_idx"]), "root_step": root_step,
        "stock_success": bool(successes[0]), "prefix_tokens": int(len(prefix)),
        "prefix_dim": int(prefix.shape[-1]), "robot_dim": int(len(robot)),
        "proprio_dim": int(len(proprio)), "action_dim": TREE_Q10_ACTION_DIM,
    }


def _split_group_ids(entries: list[dict], *, seed: int = 42,
                     train_fraction: float = .80):
    buckets = defaultdict(list)
    for entry in entries:
        buckets[(entry["suite"], bool(entry["stock_success"]))].append(entry)
    train, validation = [], []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda row: hashlib.sha256(
            f"{seed}|{row['candidate_group_id']}".encode()).hexdigest())
        n_val = max(1, int(round((1 - train_fraction) * len(rows)))) if len(rows) > 1 else 0
        validation.extend(row["candidate_group_id"] for row in rows[:n_val])
        train.extend(row["candidate_group_id"] for row in rows[n_val:])
    return tuple(sorted(train)), tuple(sorted(validation))


def _write_index(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temporary, path)


def prepare_smolvla_tree_q10_cache(*, store=None,
                                   cache_root: str | Path = "/content/smolvla_tree_q10_cache",
                                   expected_trees: int = TREE_Q10_EXPECTED_TREES,
                                   gamma: float = TREE_Q10_GAMMA,
                                   download_workers: int = 8) -> dict:
    if not 1 <= download_workers <= 8:
        raise ValueError("download_workers must be in [1,8]")
    store = store or SupabaseStore()
    groups, candidates_by_group, dataset_digest = _complete_tree_rows(
        store, expected_trees=expected_trees)
    root = Path(cache_root).expanduser() / dataset_digest
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    existing = {}
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text())
            if (payload.get("schema_version") == TREE_Q10_SCHEMA_VERSION
                    and payload.get("dataset_digest") == dataset_digest
                    and payload.get("gamma") == gamma):
                existing = {row["candidate_group_id"]: row for row in payload.get("entries", [])
                            if (root / row.get("path", "")).is_file()}
        except (OSError, json.JSONDecodeError):
            pass
    group_by_id = {str(row["candidate_group_id"]): row for row in groups}
    wanted = sorted(group_by_id)
    missing = [gid for gid in wanted if gid not in existing]
    print(
        f"[smolvla-tree-q10] cache {len(existing)}/{len(wanted)} trees ready; "
        f"building {len(missing)} with {download_workers} workers", flush=True)
    worker_state = threading.local()

    def convert(gid: str):
        worker_store = getattr(worker_state, "store", None)
        if worker_store is None:
            fork = getattr(store, "fork_for_thread", None)
            worker_store = fork() if callable(fork) else store
            worker_state.store = worker_store
        return _write_group(
            root / f"{gid}.npz", worker_store, group_by_id[gid], candidates_by_group[gid],
            gamma=gamma)

    completed_since_save = 0
    try:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            for entry in executor.map(convert, missing):
                existing[entry["candidate_group_id"]] = entry
                completed_since_save += 1
                completed = len(existing)
                if completed_since_save >= 10:
                    _write_index(index_path, {
                        "schema_version": TREE_Q10_SCHEMA_VERSION,
                        "dataset_digest": dataset_digest, "gamma": gamma,
                        "entries": [existing[gid] for gid in wanted if gid in existing],
                        "complete": False})
                    completed_since_save = 0
                if completed % 10 == 0 or completed == len(wanted):
                    print(f"[smolvla-tree-q10] cache {completed}/{len(wanted)} trees", flush=True)
    finally:
        ordered = [existing[gid] for gid in wanted if gid in existing]
        _write_index(index_path, {
            "schema_version": TREE_Q10_SCHEMA_VERSION,
            "dataset_digest": dataset_digest, "gamma": gamma,
            "entries": ordered, "complete": len(ordered) == len(wanted)})
    entries = [existing[gid] for gid in wanted]
    dimensions = {(row["prefix_dim"], row["robot_dim"], row["proprio_dim"], row["action_dim"])
                  for row in entries}
    if len(dimensions) != 1:
        raise ValueError(f"mixed tree dimensions: {dimensions}")
    train_ids, validation_ids = _split_group_ids(entries)
    train_set = set(train_ids)
    action_sum = np.zeros(TREE_Q10_ACTION_DIM, np.float64)
    action_sumsq = np.zeros(TREE_Q10_ACTION_DIM, np.float64)
    action_count = 0
    for entry in entries:
        if entry["candidate_group_id"] not in train_set:
            continue
        with np.load(root / entry["path"], allow_pickle=False) as archive:
            action = np.asarray(archive["actions"], np.float64)
        action_sum += action.sum(axis=(0, 1))
        action_sumsq += np.square(action).sum(axis=(0, 1))
        action_count += int(action.shape[0] * action.shape[1])
    mean = action_sum / action_count
    std = np.sqrt(np.maximum(action_sumsq / action_count - mean ** 2, 1e-12))
    prefix_dim, robot_dim, proprio_dim, action_dim = next(iter(dimensions))
    final = {
        "schema_version": TREE_Q10_SCHEMA_VERSION, "dataset_digest": dataset_digest,
        "gamma": gamma, "cache_dir": str(root), "entries": entries,
        "train_group_ids": list(train_ids), "validation_group_ids": list(validation_ids),
        "prefix_dim": prefix_dim, "robot_dim": robot_dim, "proprio_dim": proprio_dim,
        "action_dim": action_dim, "action_mean": mean.tolist(), "action_std": std.tolist(),
    }
    _write_index(index_path, final)
    print({
        "dataset_digest": dataset_digest, "trees": len(entries),
        "train_trees": len(train_ids), "validation_trees": len(validation_ids),
        "candidate_rows": len(entries) * SMOLVLA_TREE_CANDIDATES,
    }, flush=True)
    return final


class SmolVLATreeDataset:
    def __init__(self, cache: dict, group_ids):
        self.root = Path(cache["cache_dir"])
        by_id = {row["candidate_group_id"]: row for row in cache["entries"]}
        self.entries = [by_id[gid] for gid in group_ids]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        with np.load(self.root / entry["path"], allow_pickle=False) as archive:
            result = {name: np.asarray(archive[name]).copy() for name in archive.files}
        result["candidate_group_id"] = entry["candidate_group_id"]
        result["suite"] = entry["suite"]
        return result


def _collate_groups(groups: list[dict]):
    batch = len(groups)
    tokens = max(len(group["prefix"]) for group in groups)
    width = groups[0]["prefix"].shape[-1]
    prefix = np.zeros((batch, tokens, width), np.float16)
    pad = np.zeros((batch, tokens), bool)
    for index, group in enumerate(groups):
        length = len(group["prefix"])
        prefix[index, :length] = group["prefix"]
        pad[index, :length] = group["pad"]
    return {
        "prefix": torch.from_numpy(prefix), "pad": torch.from_numpy(pad),
        "robot": torch.from_numpy(np.stack([g["robot"] for g in groups]).astype(np.float32)),
        "proprio": torch.from_numpy(np.stack([g["proprio"] for g in groups]).astype(np.float32)),
        "action": torch.from_numpy(np.stack([g["actions"] for g in groups]).astype(np.float32)),
        "target": torch.from_numpy(np.stack([g["targets"] for g in groups]).astype(np.float32)),
        "success": torch.from_numpy(np.stack([g["successes"] for g in groups]).astype(bool)),
    }


def _to(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _forward_groups(model: QPlanningCritic, batch):
    trees, candidates = batch["action"].shape[:2]
    repeat = lambda value: value.repeat_interleave(candidates, dim=0)
    action = batch["action"].reshape(
        trees * candidates, TREE_Q10_HORIZON, TREE_Q10_ACTION_DIM)
    valid = torch.ones(action.shape[:2], dtype=torch.bool, device=action.device)
    logits = model(
        repeat(batch["prefix"]), repeat(batch["pad"]), repeat(batch["robot"]),
        repeat(batch["proprio"]), action, valid)
    expected = (logits.float().softmax(-1) * model.value_bins.float()).sum(-1)
    return logits, expected.reshape(trees, candidates)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float((positive[:, None] > negative[None, :]).mean()
                 + .5 * (positive[:, None] == negative[None, :]).mean())


@torch.no_grad()
def evaluate_smolvla_tree_q10(model, dataset, device, *, micro_tree_batch: int):
    model.eval()
    predictions, targets, successes, losses = [], [], [], []
    for start in range(0, len(dataset), micro_tree_batch):
        raw = _collate_groups([
            dataset[index] for index in range(start, min(len(dataset), start + micro_tree_batch))])
        batch = _to(raw, device)
        logits, expected = _forward_groups(model, batch)
        losses.append(float(model.categorical_loss(logits, batch["target"].reshape(-1)).cpu()))
        predictions.append(expected.cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        successes.append(batch["success"].cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    success = np.concatenate(successes)
    selected = prediction.argmax(1)
    rows = np.arange(len(prediction))
    selected_success = success[rows, selected]
    stock_success = success[:, 0]
    oracle_success = success.any(1)
    pair_correct, pair_total = 0.0, 0
    for q, outcome in zip(prediction, success):
        good, bad = np.flatnonzero(outcome), np.flatnonzero(~outcome)
        if len(good) and len(bad):
            comparisons = q[good, None] - q[bad]
            pair_correct += float((comparisons > 0).sum() + .5 * (comparisons == 0).sum())
            pair_total += int(comparisons.size)
    delta_prediction = prediction[:, 1:] - prediction[:, :1]
    delta_target = target[:, 1:] - target[:, :1]
    return {
        "hl_gauss_ce": float(np.mean(losses)),
        "return_mae": float(np.abs(prediction - target).mean()),
        "failure_auc": _auc((~success).reshape(-1), (-prediction).reshape(-1)),
        "within_tree_pairwise_accuracy": (
            float(pair_correct / pair_total) if pair_total else float("nan")),
        "difference_mae": float(np.abs(delta_prediction - delta_target).mean()),
        "stock_sr_pct": float(100 * stock_success.mean()),
        "selected_sr_pct": float(100 * selected_success.mean()),
        "oracle_sr_pct": float(100 * oracle_success.mean()),
        "selected_minus_stock_pp": float(100 * (selected_success.mean() - stock_success.mean())),
        "failure_to_success": int((~stock_success & selected_success).sum()),
        "success_to_failure": int((stock_success & ~selected_success).sum()),
        "stock_selected_pct": float(100 * (selected == 0).mean()),
        "trees": int(len(prediction)), "candidate_rows": int(prediction.size),
        "decidable_pairs": int(pair_total),
    }


def _state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def _save_checkpoint(path: Path, *, model, optimizer, update: int, cache: dict,
                     config: SmolVLATreeQ10TrainConfig, history: list[dict], run_name: str):
    payload = {
        "format": "smolvla_tree_q10_critic_v1", "update": update,
        "run_name": run_name, "dataset_digest": cache["dataset_digest"],
        "train_group_ids": cache["train_group_ids"],
        "validation_group_ids": cache["validation_group_ids"],
        "architecture": model.architecture_config(), "train_config": asdict(config),
        "source_policy": {"model": "HuggingFaceVLA/smolvla_libero",
                          "executed_actions": 10, "generated_actions": 50},
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    for candidate in path.parent.glob("checkpoint_step_*.pt"):
        if candidate != path:
            candidate.unlink()


def _latest_checkpoint(directory: Path):
    values = list(directory.glob("checkpoint_step_*.pt"))
    return max(values, key=lambda path: int(path.stem.rsplit("_", 1)[-1])) if values else None


def run_smolvla_tree_q10_training(*, run_name: str, difference_weight: float,
                                  expected_trees: int = TREE_Q10_EXPECTED_TREES,
                                  updates: int = 2_000,
                                  cache_root: str | Path = "/content/smolvla_tree_q10_cache",
                                  output_root: str | Path = "/content/drive/MyDrive/pnp_smolvla_tree_q10",
                                  micro_tree_batch: int = 2,
                                  download_workers: int = 8,
                                  device: str | None = None,
                                  resume: bool = True, store=None):
    if difference_weight not in (0.0, 1.0):
        raise ValueError("the declared ablation requires difference_weight 0.0 or 1.0")
    store = store or SupabaseStore()
    config = SmolVLATreeQ10TrainConfig(
        updates=updates, micro_tree_batch=micro_tree_batch,
        difference_weight=float(difference_weight))
    cache = prepare_smolvla_tree_q10_cache(
        store=store, cache_root=cache_root, expected_trees=expected_trees,
        gamma=config.gamma, download_workers=download_workers)
    train = SmolVLATreeDataset(cache, cache["train_group_ids"])
    validation = SmolVLATreeDataset(cache, cache["validation_group_ids"])
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = QPlanningCritic(
        prefix_dim=int(cache["prefix_dim"]), robot_dim=int(cache["robot_dim"]),
        proprio_dim=int(cache["proprio_dim"]),
        config=QPlanningModelConfig(action_horizon=10, action_dim=int(cache["action_dim"])))
    model.set_action_statistics(cache["action_mean"], cache["action_std"])
    initial_digest = _state_digest(model)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir = Path(output_root).expanduser() / cache["dataset_digest"] / run_name
    start_update, history = 0, []
    latest = _latest_checkpoint(output_dir) if resume else None
    if latest is not None:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if (payload.get("format") != "smolvla_tree_q10_critic_v1"
                or payload.get("dataset_digest") != cache["dataset_digest"]
                or payload.get("train_config") != asdict(config)):
            raise ValueError("existing checkpoint does not match this exact ablation contract")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        model.to(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_update = int(payload["update"])
        history = list(payload.get("history", []))
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        print(f"[smolvla-tree-q10] resumed {latest.name} at step {start_update}", flush=True)
    print("Training contract", flush=True)
    print({
        "run_name": run_name, "difference_weight": difference_weight,
        "updates": updates, "gamma_per_action": config.gamma,
        "effective_tree_batch": config.effective_tree_batch,
        "micro_tree_batch": config.micro_tree_batch,
        "effective_candidate_rows": config.effective_tree_batch * SMOLVLA_TREE_CANDIDATES,
        "dataset_digest": cache["dataset_digest"], "initial_model_digest": initial_digest,
        "train_trees": len(train), "validation_trees": len(validation),
        "device": str(device), "output_dir": str(output_dir),
    }, flush=True)
    if start_update == 0:
        initial = evaluate_smolvla_tree_q10(
            model, validation, device, micro_tree_batch=micro_tree_batch)
        history.append({"update": 0, "validation": initial})
        print(f"validation step 0 | {initial}", flush=True)
    use_amp = bool(config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    started = time.perf_counter()
    rolling = defaultdict(float)
    rolling_count = 0
    for update in range(start_update + 1, config.updates + 1):
        learning_rate = config.learning_rate_at(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        # Update-indexed sampling makes both arms and resumed runs consume identical trees.
        rng = np.random.default_rng(config.seed * 1_000_003 + update)
        indices = rng.choice(len(train), size=config.effective_tree_batch, replace=False)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for offset in range(0, len(indices), config.micro_tree_batch):
            raw = _collate_groups([
                train[int(i)] for i in indices[offset:offset + config.micro_tree_batch]])
            batch = _to(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits, expected = _forward_groups(model, batch)
                absolute = model.categorical_loss(logits, batch["target"].reshape(-1))
                difference = stock_relative_difference_loss(expected, batch["target"])
                loss = (absolute + config.difference_weight * difference) / config.accumulation_steps
            loss.backward()
            rolling["absolute"] += float(absolute.detach())
            rolling["difference"] += float(difference.detach())
            rolling["q"] += float(expected.detach().mean())
            rolling["target"] += float(batch["target"].mean())
            rolling_count += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if update % config.print_interval == 0 or update == config.updates:
            elapsed = time.perf_counter() - started
            completed = update - start_update
            eta = elapsed / max(completed, 1) * (config.updates - update)
            gpu = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
            print(
                f"tree-Q10 step {update}/{config.updates} | "
                f"abs_ce {rolling['absolute']/rolling_count:.4f} | "
                f"gap_huber {rolling['difference']/rolling_count:.5f} | "
                f"q {rolling['q']/rolling_count:.4f} | target {rolling['target']/rolling_count:.4f} | "
                f"grad {float(grad_norm):.3f} | lr {learning_rate:.2e} | GPU {gpu:.1f} GB | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m", flush=True)
            rolling.clear()
            rolling_count = 0
        if update % config.eval_interval == 0 or update == config.updates:
            metrics = evaluate_smolvla_tree_q10(
                model, validation, device, micro_tree_batch=micro_tree_batch)
            history.append({"update": update, "validation": metrics})
            print(
                "validation | CE {hl_gauss_ce:.4f} | return MAE {return_mae:.4f} | "
                "gap MAE {difference_mae:.4f} | failure AUC {failure_auc:.3f} | "
                "within-tree {within_tree_pairwise_accuracy:.3f} | "
                "selected-stock {selected_minus_stock_pp:+.1f} pp | "
                "F->S {failure_to_success} | S->F {success_to_failure} | "
                "stock chosen {stock_selected_pct:.1f}% | n {trees}".format(**metrics), flush=True)
        if update % config.checkpoint_interval == 0 or update == config.updates:
            path = output_dir / f"checkpoint_step_{update:06d}.pt"
            _save_checkpoint(
                path, model=model, optimizer=optimizer, update=update, cache=cache,
                config=config, history=history, run_name=run_name)
            print(f"[smolvla-tree-q10] saved {path}", flush=True)
    final = evaluate_smolvla_tree_q10(
        model, validation, device, micro_tree_batch=micro_tree_batch)
    return {
        "run_name": run_name, "difference_weight": difference_weight,
        "dataset_digest": cache["dataset_digest"], "initial_model_digest": initial_digest,
        "updates": config.updates, "validation": final, "history": history,
        "final_checkpoint": str(output_dir / f"checkpoint_step_{config.updates:06d}.pt"),
    }
