"""LIBERO-only Q50 base training for staged continual-learning experiments."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..pcp_critic.data import DatasetSnapshot, eligible_rollout_rows
from ..pcp_critic.registry import PCPCriticRegistry
from ..store import SupabaseStore
from .config import QPlanningModelConfig, QPlanningTrainConfig
from .data import QPlanningWindowDataset, prepare_qplanning_streaming_cache
from .model import QPlanningCritic
from .train import train_qplanning_critic
from .workflow import _print_preflight


LIBERO_BASE_UPDATES = 4_000
LIBERO_BASE_EXPECTED_ROLLOUTS = 600


def build_libero_only_snapshot(
        parent: DatasetSnapshot, rows: list[dict], *,
        expected_rollouts: int = LIBERO_BASE_EXPECTED_ROLLOUTS) -> DatasetSnapshot:
    """Create an immutable in-memory view preserving the parent's grouped split."""
    by_id = {row["rollout_id"]: row for row in rows}
    missing = set(parent.rollout_ids) - set(by_id)
    if missing:
        raise ValueError(f"snapshot metadata query is missing rollout IDs: {sorted(missing)[:3]}")
    selected = sorted(
        rollout_id for rollout_id in parent.rollout_ids
        if str(by_id[rollout_id].get("benchmark") or "") == "libero")
    if len(selected) != expected_rollouts:
        raise ValueError(
            f"expected {expected_rollouts} standard-LIBERO rollouts, found {len(selected)}")
    train_parent, val_parent = set(parent.train_rollout_ids), set(parent.val_rollout_ids)
    train = tuple(rollout_id for rollout_id in selected if rollout_id in train_parent)
    val = tuple(rollout_id for rollout_id in selected if rollout_id in val_parent)
    if not train or not val or set(train) | set(val) != set(selected):
        raise ValueError("LIBERO-only view has an invalid train/validation split")
    payload = {
        "parent_snapshot_id": parent.snapshot_id,
        "selection": "benchmark=libero",
        "rollout_ids": selected,
        "train_rollout_ids": train,
        "val_rollout_ids": val,
    }
    snapshot_id = "pcpcds-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        rollout_ids=tuple(selected),
        train_rollout_ids=train,
        val_rollout_ids=val,
        policy_repo_id=parent.policy_repo_id,
        policy_revision=parent.policy_revision,
        artifact_schema_version=parent.artifact_schema_version,
        # The streaming-cache builder recomputes these from LIBERO train actions.
        action_mean=parent.action_mean,
        action_std=parent.action_std,
        provenance={
            **payload,
            "filtered_view": True,
            "expected_rollouts": expected_rollouts,
        },
    )


def _print_outcomes(snapshot: DatasetSnapshot, rows: list[dict]) -> None:
    train_ids = set(snapshot.train_rollout_ids)
    counts = Counter()
    for row in rows:
        if row["rollout_id"] not in set(snapshot.rollout_ids):
            continue
        if row.get("success") not in (True, False, 0, 1):
            raise ValueError(f"rollout {row['rollout_id']} has no Boolean success label")
        split = "train" if row["rollout_id"] in train_ids else "validation"
        outcome = "success" if bool(row["success"]) else "failure"
        counts[(split, str(row["suite"]), outcome)] += 1
    print("\nLIBERO-only rollout outcomes")
    for (split, suite, outcome), count in sorted(counts.items()):
        print(f"  {split:10s} {suite:24s} {outcome:7s} {count:4d}")


def run_q50_libero_base_training(
        *, parent_snapshot_id: str,
        updates: int = LIBERO_BASE_UPDATES,
        expected_rollouts: int = LIBERO_BASE_EXPECTED_ROLLOUTS,
        cache_root: str | Path = "/content/qplanning_libero_base_cache",
        output_root: str | Path = "/content/qplanning_libero_base",
        micro_batch_size: int = 64,
        cache_download_workers: int = 8,
        device=None, resume: bool = True,
        store: SupabaseStore | None = None) -> dict:
    """Train a fresh ordinary Q50 critic using standard LIBERO and nothing else."""
    if not parent_snapshot_id.startswith("pcpcds-"):
        raise ValueError("paste the immutable pcpcds-* snapshot ID produced by notebook 56")
    if updates < 1:
        raise ValueError("updates must be positive")
    store = store or SupabaseStore()
    parent = PCPCriticRegistry(store).load_snapshot(parent_snapshot_id)
    all_rows = eligible_rollout_rows(store, rollout_ids=parent.rollout_ids)
    snapshot = build_libero_only_snapshot(
        parent, all_rows, expected_rollouts=expected_rollouts)
    selected_ids = set(snapshot.rollout_ids)
    selected_rows = [row for row in all_rows if row["rollout_id"] in selected_ids]

    # Passing the filtered snapshot here is important: only the 600 LIBERO
    # artifacts are downloaded, and normalization uses only its train split.
    cache = prepare_qplanning_streaming_cache(
        store, snapshot, horizon=50, gamma=.99, cache_root=cache_root,
        download_workers=cache_download_workers)
    if set(entry["benchmark"] for entry in cache.rollouts) != {"libero"}:
        raise AssertionError("non-LIBERO data escaped into the base cache")
    _print_preflight(snapshot, cache)
    _print_outcomes(snapshot, selected_rows)
    print(f"\nParent snapshot: {parent_snapshot_id}")
    print(f"Filtered LIBERO-only snapshot: {snapshot.snapshot_id}")
    print("PRO rollouts/actions/normalization: EXCLUDED")

    train = QPlanningWindowDataset(cache, snapshot.train_rollout_ids)
    validation = QPlanningWindowDataset(
        cache, snapshot.val_rollout_ids, max_windows=4096)
    train_config = QPlanningTrainConfig(
        effective_batch_size=64, micro_batch_size=micro_batch_size,
        updates=updates, warmup_updates=min(500, updates),
        print_interval=100, eval_interval=500,
        checkpoint_interval=1_000, max_validation_transitions=4096)
    model_config = QPlanningModelConfig(
        action_horizon=50, action_dim=cache.action_dim)

    # Reset before model allocation so this is a true from-scratch base.
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)
    model = QPlanningCritic(
        prefix_dim=cache.prefix_dim, robot_dim=cache.robot_dim,
        proprio_dim=cache.proprio_dim, config=model_config)
    model.set_action_statistics(cache.action_mean, cache.action_std)
    output_dir = (
        Path(output_root) / parent_snapshot_id /
        f"q50_libero_only_{updates:06d}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\nTraining contract")
    print(f"  initialization=fresh seed {train_config.seed}")
    print(f"  device={device}; updates={updates}")
    print(f"  effective batch={train_config.effective_batch_size}; "
          f"microbatch={train_config.micro_batch_size}; "
          f"accumulation={train_config.accumulation_steps}")
    print("  replay=uniform standard-LIBERO windows")
    print(f"  checkpoint directory: {output_dir}")
    _, _, report = train_qplanning_critic(
        model, train, validation, device,
        snapshot_id=snapshot.snapshot_id,
        cache_digest=cache.digest,
        source_policy={
            "repo_id": snapshot.policy_repo_id,
            "revision": snapshot.policy_revision,
        },
        output_dir=output_dir, config=train_config, resume=resume)
    report.update({
        "parent_snapshot_id": parent_snapshot_id,
        "filtered_snapshot_id": snapshot.snapshot_id,
        "benchmark": "libero",
        "expected_rollouts": expected_rollouts,
        "pro_data_used": False,
    })
    return report
