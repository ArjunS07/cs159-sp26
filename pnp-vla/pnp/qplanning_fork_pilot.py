"""Fixed-budget same-state fork pilot for Q-planning data acquisition.

The pilot compares three ways to choose tree roots from the immutable critic
training split: uniform random, terminal-failure priority, and four-boundary
U20 priority.  Every selected root receives one ordinary 10-step PI0.5
candidate and eight 3-step proposal candidates; every branch executes ten
actions and then continues with the ordinary 10-step policy.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import threading

import numpy as np

from .pcp_critic.data import eligible_rollout_rows
from .pcp_critic.registry import PCPCriticRegistry
from .pcp_critic.resumable_snapshot import _download_with_retry
from .pcp_search.pro import PRO_TRAIN_QUOTAS, PRO_TEN_STATE_SUITES
from .verifier.collection import (
    candidate_group_id, collect_replay_candidate_group)


FORK_PILOT_VERSION = 2
FORK_PILOT_SNAPSHOT_ID = "pcpcds-98d1f32dff8213841529b3c6"
FORK_PILOT_MANIFEST_PATH = "qplanning_forks/manifests/fixed_three_priority_v2.json"
FORK_PILOT_STRATEGIES = ("random", "u20", "failure")
FORK_PILOT_TREES_PER_STRATEGY = 64
FORK_PILOT_CANDIDATES = 9
FORK_PILOT_SHARDS = 4
FORK_PILOT_EXECUTED_ACTIONS = 10
FORK_PILOT_STOCK_DENOISE_STEPS = 10
FORK_PILOT_PROPOSAL_DENOISE_STEPS = 3
FORK_PILOT_TRAIN_PRIORITY_FRACTION = .65
FORK_PILOT_SEED = 20260914
FORK_PILOT_PRINT_EVERY_TREES = 4
_U_TIME_KEY = re.compile(r"^c(?P<chunk>\d+)_s(?P<step>\d+)_u_time$")


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()[:20]


def _rank(seed: int, namespace: str, *values) -> str:
    payload = "|".join(map(str, (seed, namespace, *values))).encode()
    return hashlib.sha256(payload).hexdigest()


def u20_profile_from_ahats(payload: bytes) -> tuple[float, ...]:
    """Return one first-20-action uncertainty value per planning boundary."""
    by_chunk: dict[int, list[float]] = defaultdict(list)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        for name in archive.files:
            match = _U_TIME_KEY.match(name)
            if not match:
                continue
            profile = np.asarray(archive[name], np.float32).reshape(-1)
            if len(profile) < 20:
                raise ValueError(f"{name} contains only {len(profile)} action positions")
            by_chunk[int(match.group("chunk"))].append(float(profile[:20].mean()))
    if not by_chunk:
        raise ValueError("a-hat artifact contains no per-action uncertainty profiles")
    expected = list(range(max(by_chunk) + 1))
    if sorted(by_chunk) != expected:
        raise ValueError("a-hat artifact has non-contiguous chunk indices")
    values = tuple(float(np.mean(by_chunk[index])) for index in expected)
    if not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise ValueError("U20 profile must be finite and positive")
    return values


def _four_boundary_score(profile: tuple[float, ...], chunk_idx: int) -> float:
    return float(np.mean(profile[chunk_idx:min(len(profile), chunk_idx + 4)]))


def _root_item(row: dict, profile: tuple[float, ...], chunk_idx: int,
               strategy: str, *, seed: int) -> dict:
    return {
        "strategy": strategy,
        "source_rollout_id": str(row["rollout_id"]),
        "benchmark": "libero_pro",
        "suite": str(row["suite"]),
        "task_idx": int(row["task_idx"]),
        "episode_idx": int(row["episode_idx"]),
        "init_state_hash": str(row.get("init_state_hash") or ""),
        "source_success": bool(row["success"]),
        "source_behavior_seed_index": 0,
        "chunk_idx": int(chunk_idx),
        "chunk_count": len(profile),
        "normalized_chunk_position": float(chunk_idx / max(len(profile) - 1, 1)),
        "source_current_u20": float(profile[chunk_idx]),
        "source_four_boundary_u20": _four_boundary_score(profile, chunk_idx),
        "source_u20_profile": [float(value) for value in profile],
        "selection_tiebreak": _rank(
            seed, strategy, row["rollout_id"], chunk_idx),
    }


def _balanced_take(candidates: list[dict], count: int, *, strategy: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        buckets[item["suite"]].append(item)
    for suite, values in buckets.items():
        if strategy == "u20":
            values.sort(key=lambda item: (
                -item["source_four_boundary_u20"], item["selection_tiebreak"]))
        else:
            values.sort(key=lambda item: item["selection_tiebreak"])
    selected = []
    suites = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for suite in suites:
            if buckets[suite] and len(selected) < count:
                selected.append(buckets[suite].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(
            f"{strategy}: requested {count} roots but found only {len(selected)}")
    return selected


def build_fixed_fork_manifest(source_rows: list[dict], profiles: dict[str, tuple[float, ...]],
                              *, trees_per_strategy: int = FORK_PILOT_TREES_PER_STRATEGY,
                              seed: int = FORK_PILOT_SEED,
                              snapshot_id: str = FORK_PILOT_SNAPSHOT_ID) -> dict:
    """Build deterministic, suite-balanced, equal-count root lists."""
    if trees_per_strategy < 1 or trees_per_strategy % FORK_PILOT_SHARDS:
        raise ValueError("trees_per_strategy must be positive and divisible by four")
    by_identity = {}
    for row in source_rows:
        if row.get("benchmark") != "libero_pro":
            continue
        if row.get("suite") not in PRO_TRAIN_QUOTAS:
            raise AssertionError(f"non-training PRO suite reached fork pool: {row.get('suite')}")
        if row.get("suite") in PRO_TEN_STATE_SUITES:
            continue
        identity = (row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
        previous = by_identity.get(identity)
        if previous is None or str(row["rollout_id"]) < str(previous["rollout_id"]):
            by_identity[identity] = row

    candidates = {strategy: [] for strategy in FORK_PILOT_STRATEGIES}
    for row in by_identity.values():
        profile = tuple(profiles.get(str(row["rollout_id"]), ()))
        # Chunk zero is intentionally excluded: the acquisition question concerns
        # mid-trajectory correction rather than restarting the whole task.
        eligible = list(range(1, len(profile)))
        if not eligible:
            continue
        random_chunk = eligible[int(_rank(
            seed, "chunk", row["rollout_id"]), 16) % len(eligible)]
        candidates["random"].append(
            _root_item(row, profile, random_chunk, "random", seed=seed))
        if not bool(row["success"]):
            candidates["failure"].append(
                _root_item(row, profile, random_chunk, "failure", seed=seed))
        u_chunk = min(
            eligible,
            key=lambda index: (-_four_boundary_score(profile, index), index))
        candidates["u20"].append(
            _root_item(row, profile, u_chunk, "u20", seed=seed))

    chosen = {
        strategy: _balanced_take(candidates[strategy], trees_per_strategy,
                                 strategy=strategy)
        for strategy in FORK_PILOT_STRATEGIES}
    trees = []
    for ordinal in range(trees_per_strategy):
        for strategy in FORK_PILOT_STRATEGIES:
            item = dict(chosen[strategy][ordinal])
            item["ordinal_within_strategy"] = ordinal
            item["shard_index"] = ordinal % FORK_PILOT_SHARDS
            trees.append(item)

    payload = {
        "version": FORK_PILOT_VERSION,
        "snapshot_id": snapshot_id,
        "seed": seed,
        "strategies": list(FORK_PILOT_STRATEGIES),
        "trees_per_strategy": trees_per_strategy,
        "candidate_count": FORK_PILOT_CANDIDATES,
        "shard_count": FORK_PILOT_SHARDS,
        "executed_actions": FORK_PILOT_EXECUTED_ACTIONS,
        "stock_denoise_steps": FORK_PILOT_STOCK_DENOISE_STEPS,
        "proposal_denoise_steps": FORK_PILOT_PROPOSAL_DENOISE_STEPS,
        "future_training_priority_fraction": FORK_PILOT_TRAIN_PRIORITY_FRACTION,
        "source_scope": (
            "Q-planning snapshot training split; LIBERO-PRO train suites only; "
            "position and ten-state behavior-seed suites excluded; roots restored "
            "from exact stored source-trajectory environment actions"),
        "trees": trees,
    }
    manifest_hash = _digest(payload)
    for item in payload["trees"]:
        item["experiment"] = f"qfork-pilot-{item['strategy']}-{manifest_hash}"
    # Experiment labels are behavior-defining and therefore covered by the final hash.
    manifest_hash = _digest(payload)
    return {"manifest_hash": manifest_hash, "payload": payload}


def _validate_manifest(document: dict) -> dict:
    payload = document.get("payload")
    if not isinstance(payload, dict) or payload.get("version") != FORK_PILOT_VERSION:
        raise ValueError("unsupported fork-pilot manifest")
    if _digest(payload) != document.get("manifest_hash"):
        raise ValueError("fork-pilot manifest hash mismatch")
    if payload.get("snapshot_id") != FORK_PILOT_SNAPSHOT_ID:
        raise ValueError("fork-pilot manifest uses the wrong immutable dataset snapshot")
    expected = FORK_PILOT_TREES_PER_STRATEGY * len(FORK_PILOT_STRATEGIES)
    if len(payload.get("trees", [])) != expected:
        raise ValueError(f"fork-pilot manifest must contain {expected} trees")
    counts = {strategy: 0 for strategy in FORK_PILOT_STRATEGIES}
    for item in payload["trees"]:
        counts[item["strategy"]] += 1
        if item["suite"] not in PRO_TRAIN_QUOTAS:
            raise AssertionError("held-out PRO suite appears in fork manifest")
    if set(counts.values()) != {FORK_PILOT_TREES_PER_STRATEGY}:
        raise ValueError(f"unbalanced fork manifest: {counts}")
    return document


def load_fork_pilot_manifest(store, path: str = FORK_PILOT_MANIFEST_PATH) -> dict:
    return _validate_manifest(json.loads(store._download(path)))


def create_fork_pilot_manifest(*, store, cache_root: str | Path,
                               snapshot_id: str = FORK_PILOT_SNAPSHOT_ID,
                               download_workers: int = 8) -> dict:
    """Extract small U20 profiles and publish the frozen 192-tree manifest."""
    if snapshot_id != FORK_PILOT_SNAPSHOT_ID:
        raise ValueError("the pilot is frozen to the critic snapshot used by prior experiments")
    if not 1 <= int(download_workers) <= 8:
        raise ValueError("download_workers must be in [1, 8]")
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
    profiles = {}
    pending = []
    for row in rows:
        path = cache / f"{row['rollout_id']}.json"
        if path.is_file() and path.stat().st_size:
            profiles[row["rollout_id"]] = tuple(json.loads(path.read_text()))
        else:
            pending.append((row, path))
    print(f"[qfork] U20 profiles: {len(profiles)}/{len(rows)} cached; "
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
                    print(f"[qfork] U20 profiles extracted: {completed}/{len(pending)}",
                          flush=True)
    document = build_fixed_fork_manifest(
        rows, profiles, snapshot_id=snapshot.snapshot_id)
    document["payload"]["policy_repo_id"] = snapshot.policy_repo_id
    document["payload"]["policy_revision"] = snapshot.policy_revision
    # Adding policy provenance changes the immutable content hash.
    document["manifest_hash"] = _digest(document["payload"])
    store._upload(FORK_PILOT_MANIFEST_PATH, _canonical_json(document))
    _validate_manifest(document)
    print({
        "manifest_path": FORK_PILOT_MANIFEST_PATH,
        "manifest_hash": document["manifest_hash"],
        "trees": len(document["payload"]["trees"]),
        "trees_per_strategy": FORK_PILOT_TREES_PER_STRATEGY,
        "future_training_mix": "65% fork-priority / 35% ordinary replay",
    })
    return document


def _episode_lookup(items: list[dict]):
    from .experiments import _prepare_libero_pro_expanded_episodes

    suites = sorted({item["suite"] for item in items})
    episode_idxs = sorted({int(item["episode_idx"]) for item in items})
    episodes = _prepare_libero_pro_expanded_episodes(
        suites=suites, episode_idxs=episode_idxs)
    lookup = {(ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"])): ep
              for ep in episodes}
    missing = sorted({
        (item["suite"], int(item["task_idx"]), int(item["episode_idx"]))
        for item in items} - set(lookup))
    if missing:
        raise ValueError(f"fork manifest identities cannot be reconstructed: {missing[:3]}")
    return lookup


def _trajectory_actions_from_payload(payload: bytes) -> np.ndarray:
    """Read the compact trajectory artifact used for exact parent replay."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if "actions" not in archive.files:
            raise ValueError("source trajectory artifact has no actions array")
        actions = np.asarray(archive["actions"], dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"invalid source environment actions {actions.shape}")
    return actions


def _load_source_replay_actions(store, items: list[dict]) -> dict[str, np.ndarray]:
    """Download each tiny source trajectory once and retain its exact env actions."""
    rollout_ids = sorted({str(item["source_rollout_id"]) for item in items})
    rows = []
    for start in range(0, len(rollout_ids), 100):
        batch = rollout_ids[start:start + 100]
        rows.extend(store.fetch_all(
            "rollouts", "rollout_id,trajectory_path",
            configure=lambda query, batch=batch: query.in_("rollout_id", batch),
            order_by=("rollout_id",)))
    paths = {str(row["rollout_id"]): row.get("trajectory_path") for row in rows}
    missing = [rollout_id for rollout_id in rollout_ids if not paths.get(rollout_id)]
    if missing:
        raise ValueError(
            f"fork roots require compact source trajectories; missing {missing[:3]}")
    result = {}
    for index, rollout_id in enumerate(rollout_ids, 1):
        result[rollout_id] = _trajectory_actions_from_payload(
            _download_with_retry(store, paths[rollout_id]))
        if index % 10 == 0 or index == len(rollout_ids):
            print(f"[qfork] source trajectories: {index}/{len(rollout_ids)}", flush=True)
    return result


def _group_id(item: dict) -> str:
    return candidate_group_id(
        "libero_pro", item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=item["experiment"])


def _fetch_candidate_rows(store, group_ids: list[str]) -> list[dict]:
    rows = []
    for start in range(0, len(group_ids), 100):
        batch = group_ids[start:start + 100]
        rows.extend(store.fetch_all(
            "verifier_candidates",
            "candidate_id,candidate_group_id,candidate_kind,success,n_steps,metadata_json",
            configure=lambda query, batch=batch: query.in_("candidate_group_id", batch),
            order_by=("candidate_group_id", "candidate_id")))
    return rows


def _progress(store, items: list[dict]) -> tuple[set[str], str]:
    expected = {_group_id(item): item for item in items}
    rows = _fetch_candidate_rows(store, sorted(expected))
    by_group = defaultdict(list)
    for row in rows:
        if row["candidate_group_id"] in expected:
            by_group[row["candidate_group_id"]].append(row)
    complete = {group_id for group_id, values in by_group.items()
                if len(values) == FORK_PILOT_CANDIDATES}
    lines = [
        "Exact fork-pilot progress in THIS shard; partial trees are excluded.",
        f"{'strategy':<12}{'trees':>8}{'branches':>11}{'mixed':>10}"
        f"{'stock SR':>12}{'any-success':>14}{'oracle gain':>14}",
    ]
    for strategy in FORK_PILOT_STRATEGIES:
        group_ids = [group_id for group_id, item in expected.items()
                     if item["strategy"] == strategy and group_id in complete]
        values = [by_group[group_id] for group_id in group_ids]
        stock = [next(row for row in group if row["candidate_kind"] == "default")
                 for group in values]
        mixed = sum(len({bool(row["success"]) for row in group}) > 1 for group in values)
        any_success = sum(any(bool(row["success"]) for row in group) for group in values)
        stock_success = sum(bool(row["success"]) for row in stock)
        n = len(values)
        pct = lambda value: "-" if not n else f"{100 * value / n:.0f}%"
        lines.append(
            f"{strategy:<12}{n:>8}{n * FORK_PILOT_CANDIDATES:>11}"
            f"{pct(mixed):>10}{pct(stock_success):>12}{pct(any_success):>14}"
            f"{pct(any_success - stock_success):>14}")
    return complete, "\n".join(lines)


def run_fork_pilot_worker(*, shard_index: int, shard_count: int = FORK_PILOT_SHARDS,
                          tree_limit_per_strategy: int | None = None,
                          manifest_path: str = FORK_PILOT_MANIFEST_PATH,
                          store=None) -> dict:
    """Collect one resume-safe shard of all three root-selection strategies."""
    from . import libero_env, models
    from .store import SupabaseStore

    if shard_count != FORK_PILOT_SHARDS or not 0 <= int(shard_index) < shard_count:
        raise ValueError("fork pilot requires shard_count=4 and shard_index in [0,4)")
    store = store or SupabaseStore()
    document = load_fork_pilot_manifest(store, manifest_path)
    payload = document["payload"]
    items = [dict(item) for item in payload["trees"]
             if int(item["shard_index"]) == int(shard_index)]
    if tree_limit_per_strategy is not None:
        if int(tree_limit_per_strategy) < 1:
            raise ValueError("tree_limit_per_strategy must be positive")
        items = [item for item in items
                 if int(item["ordinal_within_strategy"]) // shard_count
                 < int(tree_limit_per_strategy)]
    per_strategy = {strategy: sum(item["strategy"] == strategy for item in items)
                    for strategy in FORK_PILOT_STRATEGIES}
    if len(set(per_strategy.values())) != 1:
        raise AssertionError(f"fork worker is not balanced: {per_strategy}")

    complete, table = _progress(store, items)
    print(table, flush=True)
    pending = [item for item in items if _group_id(item) not in complete]
    if not pending:
        return {"new_trees": 0, "requested_trees": len(items), "complete": len(complete)}

    lookup = _episode_lookup(items)
    source_replay_actions = _load_source_replay_actions(store, pending)
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    policy.model._pnp.num_steps = FORK_PILOT_STOCK_DENOISE_STEPS
    experiments = sorted({item["experiment"] for item in items})
    store.start_run(
        "qplanning_fork_pilot", "libero_pro", experiments[0],
        config={
            "manifest_path": manifest_path,
            "manifest_hash": document["manifest_hash"],
            "shard_count": shard_count, "shard_index": int(shard_index),
            "trees": len(items), "candidate_count": FORK_PILOT_CANDIDATES,
            "executed_actions": FORK_PILOT_EXECUTED_ACTIONS,
            "stock_denoise_steps": FORK_PILOT_STOCK_DENOISE_STEPS,
            "proposal_denoise_steps": FORK_PILOT_PROPOSAL_DENOISE_STEPS,
            "videos": False,
        })
    new_trees = 0
    try:
        for item in pending:
            ep = dict(lookup[(
                item["suite"], int(item["task_idx"]), int(item["episode_idx"]))])
            ep["behavior_seed_index"] = 0
            env = libero_env.make_env(ep["bddl_path"])
            try:
                result = collect_replay_candidate_group(
                    env, ep, policy, preprocess, postprocess, device,
                    chunk_idx=int(item["chunk_idx"]),
                    uncertainty_stratum=item["strategy"],
                    prefix_length=FORK_PILOT_EXECUTED_ACTIONS,
                    candidate_count=FORK_PILOT_CANDIDATES,
                    experiment=item["experiment"],
                    collection_split="fork_pilot",
                    manifest_hash=document["manifest_hash"],
                    model_revision=payload["policy_revision"],
                    n_action_steps=FORK_PILOT_EXECUTED_ACTIONS,
                    candidate_num_inference_steps=FORK_PILOT_PROPOSAL_DENOISE_STEPS,
                    replay_actions_override=source_replay_actions[
                        item["source_rollout_id"]][
                            :int(item["chunk_idx"]) * FORK_PILOT_EXECUTED_ACTIONS])
            finally:
                env.close()
            if result is None:
                raise RuntimeError(
                    f"source replay terminated before root {_group_id(item)}")
            group, candidates = result
            group["metadata_json"].update({
                "root_selection_strategy": item["strategy"],
                "source_rollout_id": item["source_rollout_id"],
                "source_success": item["source_success"],
                "source_current_u20": item["source_current_u20"],
                "source_four_boundary_u20": item["source_four_boundary_u20"],
                "source_u20_profile": item["source_u20_profile"],
                "normalized_chunk_position": item["normalized_chunk_position"],
                "future_training_priority_fraction": FORK_PILOT_TRAIN_PRIORITY_FRACTION,
            })
            store.register_candidate_group(group, candidates)
            new_trees += 1
            if new_trees % FORK_PILOT_PRINT_EVERY_TREES == 0:
                _, table = _progress(store, items)
                print(table, flush=True)
    finally:
        store.finish_run(n_rollouts=new_trees * FORK_PILOT_CANDIDATES)
    complete, table = _progress(store, items)
    print(table, flush=True)
    return {
        "manifest_hash": document["manifest_hash"],
        "shard_index": int(shard_index), "requested_trees": len(items),
        "new_trees": new_trees, "complete_trees": len(complete),
        "new_branch_outcomes": new_trees * FORK_PILOT_CANDIDATES,
    }


def run_fork_restoration_preflight(*, manifest_path: str = FORK_PILOT_MANIFEST_PATH,
                                   store=None) -> list[dict]:
    """Replay one root per strategy twice and require identical stock branches."""
    from . import libero_env, models
    from .store import SupabaseStore

    store = store or SupabaseStore()
    document = load_fork_pilot_manifest(store, manifest_path)
    payload = document["payload"]
    roots = [next(item for item in payload["trees"] if item["strategy"] == strategy)
             for strategy in FORK_PILOT_STRATEGIES]
    lookup = _episode_lookup(roots)
    source_replay_actions = _load_source_replay_actions(store, roots)
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    policy.model._pnp.num_steps = FORK_PILOT_STOCK_DENOISE_STEPS
    reports = []
    for item in roots:
        ep = dict(lookup[(item["suite"], item["task_idx"], item["episode_idx"])])
        ep["behavior_seed_index"] = 0
        env = libero_env.make_env(ep["bddl_path"])
        runs = []
        try:
            for _ in range(2):
                result = collect_replay_candidate_group(
                    env, ep, policy, preprocess, postprocess, device,
                    chunk_idx=item["chunk_idx"], uncertainty_stratum=item["strategy"],
                    prefix_length=FORK_PILOT_EXECUTED_ACTIONS, candidate_count=1,
                    experiment=item["experiment"], collection_split="restoration_preflight",
                    manifest_hash=document["manifest_hash"],
                    model_revision=payload["policy_revision"],
                    n_action_steps=FORK_PILOT_EXECUTED_ACTIONS,
                    replay_actions_override=source_replay_actions[
                        item["source_rollout_id"]][
                            :int(item["chunk_idx"]) * FORK_PILOT_EXECUTED_ACTIONS])
                if result is None:
                    raise RuntimeError("preflight source terminated before selected root")
                runs.append(result)
        finally:
            env.close()
        (group_a, candidates_a), (group_b, candidates_b) = runs
        candidate_a, candidate_b = candidates_a[0], candidates_b[0]
        policy_error = float(np.max(np.abs(
            candidate_a["blobs"]["policy_chunk"]["actions"]
            - candidate_b["blobs"]["policy_chunk"]["actions"])))
        env_error = float(np.max(np.abs(
            candidate_a["blobs"]["env_chunk"]["actions"]
            - candidate_b["blobs"]["env_chunk"]["actions"])))
        metadata_a = group_a["metadata_json"]
        metadata_b = group_b["metadata_json"]
        restoration_fields = (
            "parent_replay_actions_sha256", "root_sim_state_sha256",
            "root_policy_input_sha256")
        restoration_exact = all(
            metadata_a[field] == metadata_b[field] for field in restoration_fields)
        noise_exact = (
            candidate_a["metadata_json"]["candidate_noise_sha256"]
            == candidate_b["metadata_json"]["candidate_noise_sha256"])
        prediction_repeat_exact = policy_error <= 1e-6 and env_error <= 1e-6
        passed = (
            metadata_a["exact_sim_state_validated"]
            and metadata_b["exact_sim_state_validated"]
            and metadata_a["parent_replay_source"] == "stored_source_trajectory"
            and metadata_b["parent_replay_source"] == "stored_source_trajectory"
            and restoration_exact and noise_exact)
        report = {
            "strategy": item["strategy"], "suite": item["suite"],
            "task_idx": item["task_idx"], "episode_idx": item["episode_idx"],
            "chunk_idx": item["chunk_idx"], "policy_chunk_max_abs": policy_error,
            "env_chunk_max_abs": env_error,
            "root_restoration_exact": bool(restoration_exact),
            "root_noise_exact": bool(noise_exact),
            "prediction_repeat_exact": bool(prediction_repeat_exact),
            "outcomes_match": bool(candidate_a["success"] == candidate_b["success"]),
            "step_counts_match": bool(candidate_a["n_steps"] == candidate_b["n_steps"]),
            "outcome": bool(candidate_a["success"]),
            "n_steps": int(candidate_a["n_steps"]), "passed": bool(passed),
        }
        print(report, flush=True)
        if not passed:
            raise AssertionError(f"fork restoration preflight failed: {report}")
        reports.append(report)
    return reports


def load_fork_pilot_results(*, store=None,
                            manifest_path: str = FORK_PILOT_MANIFEST_PATH):
    """Return per-tree and per-strategy acquisition diagnostics."""
    import pandas as pd
    from .store import SupabaseStore

    store = store or SupabaseStore()
    document = load_fork_pilot_manifest(store, manifest_path)
    items = document["payload"]["trees"]
    expected = {_group_id(item): item for item in items}
    candidates = _fetch_candidate_rows(store, sorted(expected))
    by_group = defaultdict(list)
    for row in candidates:
        if row["candidate_group_id"] in expected:
            by_group[row["candidate_group_id"]].append(row)
    records = []
    for group_id, item in expected.items():
        rows = by_group.get(group_id, [])
        complete = len(rows) == FORK_PILOT_CANDIDATES
        if not complete:
            records.append({**item, "candidate_group_id": group_id,
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
            "best_success_n_steps": min(successful_steps) if successful_steps else np.nan,
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
            "mixed_outcome_trees_pct": 100 * float(group.mixed_outcomes.mean()) if len(group) else np.nan,
            "stock_branch_sr_pct": 100 * float(group.stock_success.mean()) if len(group) else np.nan,
            "any_branch_success_pct": 100 * float(group.any_success.mean()) if len(group) else np.nan,
            "oracle_gain_over_stock_pp": 100 * float(
                (group.any_success.astype(int) - group.stock_success.astype(int)).mean())
                if len(group) else np.nan,
            "mean_branch_success_fraction": float(group.branch_success_fraction.mean())
                if len(group) else np.nan,
            "mean_selected_current_u20": float(group.source_current_u20.mean())
                if len(group) else np.nan,
            "mean_selected_four_boundary_u20": float(group.source_four_boundary_u20.mean())
                if len(group) else np.nan,
        })
    return trees, pd.DataFrame(summary_rows)
