"""Twenty-action intervention variant of the fixed Q-planning fork pilot.

Root identities remain tied to the source policy's ordinary ten-action planning
boundaries. Each candidate instead executes its first twenty predicted actions
before ordinary ten-action closed-loop replanning resumes.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import numpy as np

from .pcp_critic.data import eligible_rollout_rows
from .pcp_critic.registry import PCPCriticRegistry
from .pcp_critic.resumable_snapshot import _download_with_retry
from .pcp_search.pro import PRO_TEN_STATE_SUITES, PRO_TRAIN_QUOTAS
from .qplanning_fork_pilot import (
    FORK_PILOT_CANDIDATES,
    FORK_PILOT_PROPOSAL_DENOISE_STEPS,
    FORK_PILOT_RENDER_LEAD,
    FORK_PILOT_SEED,
    FORK_PILOT_SHARDS,
    FORK_PILOT_SKIP_UNUSED_RENDERS,
    FORK_PILOT_SNAPSHOT_ID,
    FORK_PILOT_STOCK_DENOISE_STEPS,
    FORK_PILOT_STRATEGIES,
    FORK_PILOT_TRAIN_PRIORITY_FRACTION,
    FORK_PILOT_TREES_PER_STRATEGY,
    FORK_PILOT_U20_BOUNDARIES,
    FORK_PILOT_MANIFEST_PATH,
    _balanced_take,
    _canonical_json,
    _digest,
    _fetch_candidate_rows,
    _group_id,
    _rank,
    _root_item,
    _three_boundary_score,
    load_fork_pilot_manifest,
    run_fork_pilot_worker,
    run_fork_restoration_preflight,
    u20_profile_from_ahats,
)


FORK20_VERSION = 1
FORK20_MANIFEST_PATH = (
    "qplanning_forks/manifests/fixed_three_priority_20action_v1.json")
FORK20_INTERVENTION_ACTIONS = 20
FORK20_REPLAN_ACTIONS = 10
FORK20_REQUIRED_BOUNDARIES = 4


def _eligible_chunks(profile: tuple[float, ...]) -> list[int]:
    """Require 20 intervention actions and at least 20 continuation actions."""
    return list(range(1, len(profile) - FORK20_REQUIRED_BOUNDARIES + 1))


def build_fixed_fork20_manifest(
        source_rows: list[dict], profiles: dict[str, tuple[float, ...]], *,
        reference_trees: list[dict] | None = None,
        reference_manifest_hash: str = "",
        trees_per_strategy: int = FORK_PILOT_TREES_PER_STRATEGY,
        seed: int = FORK_PILOT_SEED,
        snapshot_id: str = FORK_PILOT_SNAPSHOT_ID) -> dict:
    """Build a deterministic four-boundary manifest, retaining v4 roots when eligible."""
    if trees_per_strategy < 1 or trees_per_strategy % FORK_PILOT_SHARDS:
        raise ValueError("trees_per_strategy must be positive and divisible by four")
    by_identity = {}
    for row in source_rows:
        if row.get("benchmark") != "libero_pro":
            continue
        if row.get("suite") not in PRO_TRAIN_QUOTAS:
            raise AssertionError(
                f"non-training PRO suite reached fork20 pool: {row.get('suite')}")
        if row.get("suite") in PRO_TEN_STATE_SUITES:
            continue
        identity = (row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
        previous = by_identity.get(identity)
        if previous is None or str(row["rollout_id"]) < str(previous["rollout_id"]):
            by_identity[identity] = row
    by_rollout = {str(row["rollout_id"]): row for row in by_identity.values()}

    slots = {
        strategy: {ordinal: None for ordinal in range(trees_per_strategy)}
        for strategy in FORK_PILOT_STRATEGIES}
    selected_rollouts = {strategy: set() for strategy in FORK_PILOT_STRATEGIES}
    for reference in reference_trees or []:
        strategy = str(reference.get("strategy"))
        ordinal = int(reference.get("ordinal_within_strategy", -1))
        rollout_id = str(reference.get("source_rollout_id"))
        row = by_rollout.get(rollout_id)
        profile = tuple(profiles.get(rollout_id, ()))
        chunk_idx = int(reference.get("chunk_idx", -1))
        if (strategy not in slots or ordinal not in slots[strategy] or row is None
                or chunk_idx not in _eligible_chunks(profile)):
            continue
        item = _root_item(row, profile, chunk_idx, strategy, seed=seed)
        item["paired_with_10action_tree"] = True
        slots[strategy][ordinal] = item
        selected_rollouts[strategy].add(rollout_id)

    candidates = {strategy: [] for strategy in FORK_PILOT_STRATEGIES}
    for row in by_identity.values():
        rollout_id = str(row["rollout_id"])
        profile = tuple(profiles.get(rollout_id, ()))
        eligible = _eligible_chunks(profile)
        if not eligible:
            continue
        random_chunk = eligible[int(
            _rank(seed, "fork20-chunk", rollout_id), 16) % len(eligible)]
        proposed = {
            "random": random_chunk,
            "u20": min(
                eligible,
                key=lambda index: (-_three_boundary_score(profile, index), index)),
        }
        if not bool(row["success"]):
            proposed["failure"] = random_chunk
        for strategy, chunk_idx in proposed.items():
            if rollout_id in selected_rollouts[strategy]:
                continue
            item = _root_item(row, profile, chunk_idx, strategy, seed=seed)
            item["paired_with_10action_tree"] = False
            candidates[strategy].append(item)

    for strategy in FORK_PILOT_STRATEGIES:
        missing = [ordinal for ordinal, item in slots[strategy].items()
                   if item is None]
        replacements = _balanced_take(
            candidates[strategy], len(missing), strategy=strategy)
        for ordinal, item in zip(missing, replacements):
            slots[strategy][ordinal] = item

    trees = []
    for ordinal in range(trees_per_strategy):
        for strategy in FORK_PILOT_STRATEGIES:
            item = dict(slots[strategy][ordinal])
            item["ordinal_within_strategy"] = ordinal
            item["shard_index"] = ordinal % FORK_PILOT_SHARDS
            trees.append(item)
    paired_count = sum(bool(item["paired_with_10action_tree"]) for item in trees)
    payload = {
        "version": FORK20_VERSION,
        "variant": "twenty_action_intervention",
        "snapshot_id": snapshot_id,
        "seed": seed,
        "strategies": list(FORK_PILOT_STRATEGIES),
        "trees_per_strategy": trees_per_strategy,
        "candidate_count": FORK_PILOT_CANDIDATES,
        "shard_count": FORK_PILOT_SHARDS,
        "candidate_intervention_actions": FORK20_INTERVENTION_ACTIONS,
        "continuation_replan_actions": FORK20_REPLAN_ACTIONS,
        "root_boundary_stride": FORK20_REPLAN_ACTIONS,
        "required_complete_boundaries": FORK20_REQUIRED_BOUNDARIES,
        "u20_score_boundaries": FORK_PILOT_U20_BOUNDARIES,
        "stock_denoise_steps": FORK_PILOT_STOCK_DENOISE_STEPS,
        "proposal_denoise_steps": FORK_PILOT_PROPOSAL_DENOISE_STEPS,
        "skip_unused_renders": FORK_PILOT_SKIP_UNUSED_RENDERS,
        "render_lead": FORK_PILOT_RENDER_LEAD,
        "future_training_priority_fraction": FORK_PILOT_TRAIN_PRIORITY_FRACTION,
        "reference_10action_manifest_path": FORK_PILOT_MANIFEST_PATH,
        "reference_10action_manifest_hash": reference_manifest_hash,
        "paired_with_10action_tree_count": paired_count,
        "source_scope": (
            "Q-planning snapshot training split; LIBERO-PRO train suites only; "
            "position and ten-state behavior-seed suites excluded; roots restored "
            "from exact stored source-trajectory environment actions"),
        "trees": trees,
    }
    initial_hash = _digest(payload)
    for item in payload["trees"]:
        item["experiment"] = (
            f"qfork20-pilot-{item['strategy']}-{initial_hash}")
    return {"manifest_hash": _digest(payload), "payload": payload}


def _validate_fork20_manifest(document: dict) -> dict:
    payload = document.get("payload")
    if not isinstance(payload, dict) or payload.get("version") != FORK20_VERSION:
        raise ValueError("unsupported fork20 manifest")
    if payload.get("variant") != "twenty_action_intervention":
        raise ValueError("fork20 manifest has the wrong variant")
    if _digest(payload) != document.get("manifest_hash"):
        raise ValueError("fork20 manifest hash mismatch")
    expected_fields = {
        "snapshot_id": FORK_PILOT_SNAPSHOT_ID,
        "candidate_count": FORK_PILOT_CANDIDATES,
        "shard_count": FORK_PILOT_SHARDS,
        "candidate_intervention_actions": FORK20_INTERVENTION_ACTIONS,
        "continuation_replan_actions": FORK20_REPLAN_ACTIONS,
        "root_boundary_stride": FORK20_REPLAN_ACTIONS,
        "required_complete_boundaries": FORK20_REQUIRED_BOUNDARIES,
        "u20_score_boundaries": FORK_PILOT_U20_BOUNDARIES,
        "skip_unused_renders": True,
        "render_lead": FORK_PILOT_RENDER_LEAD,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"fork20 manifest {field}={payload.get(field)!r}, expected {expected!r}")
    expected_count = FORK_PILOT_TREES_PER_STRATEGY * len(FORK_PILOT_STRATEGIES)
    if len(payload.get("trees", [])) != expected_count:
        raise ValueError(f"fork20 manifest must contain {expected_count} trees")
    counts = defaultdict(int)
    for item in payload["trees"]:
        counts[item["strategy"]] += 1
        if item["suite"] not in PRO_TRAIN_QUOTAS:
            raise AssertionError("held-out PRO suite appears in fork20 manifest")
        if int(item["chunk_idx"]) + FORK20_REQUIRED_BOUNDARIES > int(
                item["chunk_count"]):
            raise AssertionError("fork20 root lacks four complete boundaries")
    if set(counts.values()) != {FORK_PILOT_TREES_PER_STRATEGY}:
        raise ValueError(f"unbalanced fork20 manifest: {dict(counts)}")
    return document


def load_fork20_pilot_manifest(store, path: str = FORK20_MANIFEST_PATH) -> dict:
    return _validate_fork20_manifest(json.loads(store._download(path)))


def create_fork20_pilot_manifest(
        *, store, cache_root: str | Path,
        snapshot_id: str = FORK_PILOT_SNAPSHOT_ID,
        reference_manifest_path: str = FORK_PILOT_MANIFEST_PATH,
        download_workers: int = 8) -> dict:
    """Publish a four-boundary manifest maximally paired to the 10-action v4 roots."""
    if snapshot_id != FORK_PILOT_SNAPSHOT_ID:
        raise ValueError("fork20 is frozen to the prior critic snapshot")
    if not 1 <= int(download_workers) <= 8:
        raise ValueError("download_workers must be in [1, 8]")
    reference = load_fork_pilot_manifest(store, reference_manifest_path)
    snapshot = PCPCriticRegistry(store).load_snapshot(snapshot_id)
    rows = eligible_rollout_rows(store, rollout_ids=snapshot.train_rollout_ids)
    rows = [row for row in rows
            if row.get("benchmark") == "libero_pro"
            and row.get("suite") in PRO_TRAIN_QUOTAS
            and row.get("suite") not in PRO_TEN_STATE_SUITES]
    if not rows:
        raise ValueError("snapshot contains no eligible non-position PRO source rows")

    cache = Path(cache_root).expanduser() / snapshot_id / "fork_u20_profiles"
    cache.mkdir(parents=True, exist_ok=True)
    profiles, pending = {}, []
    for row in rows:
        path = cache / f"{row['rollout_id']}.json"
        if path.is_file() and path.stat().st_size:
            profiles[row["rollout_id"]] = tuple(json.loads(path.read_text()))
        else:
            pending.append((row, path))
    print(f"[qfork20] U20 profiles: {len(profiles)}/{len(rows)} cached; "
          f"extracting {len(pending)}", flush=True)
    state = threading.local()
    lock = threading.Lock()

    def extract(item):
        row, path = item
        worker_store = getattr(state, "store", None)
        if worker_store is None:
            fork = getattr(store, "fork_for_thread", None)
            worker_store = fork() if callable(fork) else store
            state.store = worker_store
        profile = u20_profile_from_ahats(
            _download_with_retry(worker_store, row["ahats_path"]))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(profile))
        temporary.replace(path)
        with lock:
            profiles[row["rollout_id"]] = profile
        return row["rollout_id"]

    if pending:
        completed = 0
        with ThreadPoolExecutor(max_workers=int(download_workers)) as executor:
            for _ in executor.map(extract, pending):
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"[qfork20] U20 profiles extracted: {completed}/{len(pending)}",
                        flush=True)
    document = build_fixed_fork20_manifest(
        rows, profiles, reference_trees=reference["payload"]["trees"],
        reference_manifest_hash=reference["manifest_hash"],
        snapshot_id=snapshot.snapshot_id)
    document["payload"]["policy_repo_id"] = snapshot.policy_repo_id
    document["payload"]["policy_revision"] = snapshot.policy_revision
    document["manifest_hash"] = _digest(document["payload"])
    store._upload(FORK20_MANIFEST_PATH, _canonical_json(document))
    _validate_fork20_manifest(document)
    print({
        "manifest_path": FORK20_MANIFEST_PATH,
        "manifest_hash": document["manifest_hash"],
        "trees": len(document["payload"]["trees"]),
        "paired_with_10action_trees": document["payload"][
            "paired_with_10action_tree_count"],
        "required_complete_boundaries": FORK20_REQUIRED_BOUNDARIES,
    }, flush=True)
    return document


def run_fork20_pilot_worker(
        *, shard_index: int, shard_count: int = FORK_PILOT_SHARDS,
        tree_limit_per_strategy: int | None = None,
        manifest_path: str = FORK20_MANIFEST_PATH, store=None) -> dict:
    return run_fork_pilot_worker(
        shard_index=shard_index, shard_count=shard_count,
        tree_limit_per_strategy=tree_limit_per_strategy,
        manifest_path=manifest_path, store=store,
        _manifest_loader=load_fork20_pilot_manifest,
        _intervention_actions=FORK20_INTERVENTION_ACTIONS,
        _replan_actions=FORK20_REPLAN_ACTIONS,
        _run_name="qplanning_fork20_pilot")


def run_fork20_restoration_preflight(
        *, manifest_path: str = FORK20_MANIFEST_PATH, store=None) -> list[dict]:
    return run_fork_restoration_preflight(
        manifest_path=manifest_path, store=store,
        _manifest_loader=load_fork20_pilot_manifest,
        _intervention_actions=FORK20_INTERVENTION_ACTIONS,
        _replan_actions=FORK20_REPLAN_ACTIONS)


def load_fork20_pilot_results(
        *, store=None, manifest_path: str = FORK20_MANIFEST_PATH):
    """Return per-tree and per-strategy diagnostics for the 20-action pilot."""
    import pandas as pd
    from .store import SupabaseStore

    store = store or SupabaseStore()
    document = load_fork20_pilot_manifest(store, manifest_path)
    items = document["payload"]["trees"]
    expected = {_group_id(item): item for item in items}
    by_group = defaultdict(list)
    for row in _fetch_candidate_rows(store, sorted(expected)):
        if row["candidate_group_id"] in expected:
            by_group[row["candidate_group_id"]].append(row)
    records = []
    for group_id, item in expected.items():
        rows = by_group.get(group_id, [])
        complete = len(rows) == FORK_PILOT_CANDIDATES
        if not complete:
            records.append({
                **item, "candidate_group_id": group_id,
                "complete": False, "branches": len(rows)})
            continue
        stock = next(row for row in rows if row["candidate_kind"] == "default")
        outcomes = [bool(row["success"]) for row in rows]
        successful_steps = [int(row["n_steps"]) for row in rows if row["success"]]
        records.append({
            **item, "candidate_group_id": group_id, "complete": True,
            "branches": len(rows), "stock_success": bool(stock["success"]),
            "any_success": any(outcomes), "mixed_outcomes": len(set(outcomes)) > 1,
            "branch_success_fraction": float(np.mean(outcomes)),
            "stock_n_steps": int(stock["n_steps"]),
            "best_success_n_steps": min(successful_steps)
            if successful_steps else np.nan,
        })
    trees = pd.DataFrame(records)
    complete = trees[trees.complete].copy()
    summary_rows = []
    for strategy in FORK_PILOT_STRATEGIES:
        group = complete[complete.strategy == strategy]
        summary_rows.append({
            "strategy": strategy,
            "complete_trees": len(group),
            "branches": int(group.branches.sum()) if len(group) else 0,
            "paired_with_10action_trees": int(
                group.paired_with_10action_tree.sum()) if len(group) else 0,
            "mixed_outcome_trees_pct": 100 * float(
                group.mixed_outcomes.mean()) if len(group) else np.nan,
            "stock_branch_sr_pct": 100 * float(
                group.stock_success.mean()) if len(group) else np.nan,
            "any_branch_success_pct": 100 * float(
                group.any_success.mean()) if len(group) else np.nan,
            "oracle_gain_over_stock_pp": 100 * float(
                (group.any_success.astype(int)
                 - group.stock_success.astype(int)).mean()) if len(group) else np.nan,
            "mean_branch_success_fraction": float(
                group.branch_success_fraction.mean()) if len(group) else np.nan,
            "mean_selected_current_u20": float(
                group.source_current_u20.mean()) if len(group) else np.nan,
            "mean_selected_three_boundary_u20": float(
                group.source_three_boundary_u20.mean()) if len(group) else np.nan,
        })
    return trees, pd.DataFrame(summary_rows)
