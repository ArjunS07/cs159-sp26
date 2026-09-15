"""Fixed-budget same-state fork pilot for Q-planning data acquisition.

The pilot compares three ways to choose tree roots from the immutable critic
training split: uniform random, terminal-failure priority, and three-boundary
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
import time

import numpy as np

from .pcp_critic.data import eligible_rollout_rows
from .pcp_critic.registry import PCPCriticRegistry
from .pcp_critic.resumable_snapshot import _download_with_retry
from .pcp_search.pro import PRO_TRAIN_QUOTAS, PRO_TEN_STATE_SUITES
from .verifier.collection import (
    _reset_and_replay_actions, _unwrap_sim,
    candidate_group_id, collect_replay_candidate_group, postprocess_chunk,
    predict_clean_chunk)


FORK_PILOT_VERSION = 5
FORK_PILOT_SNAPSHOT_ID = "pcpcds-98d1f32dff8213841529b3c6"
FORK_PILOT_MANIFEST_PATH = "qplanning_forks/manifests/fixed_three_priority_v5.json"
FORK_PILOT_STRATEGIES = ("random", "u20", "failure")
FORK_PILOT_TREES_PER_STRATEGY = 64
FORK_PILOT_CANDIDATES = 9
FORK_PILOT_SHARDS = 4
FORK_PILOT_EXECUTED_ACTIONS = 10
FORK_PILOT_STOCK_DENOISE_STEPS = 10
FORK_PILOT_PROPOSAL_DENOISE_STEPS = 3
FORK_PILOT_U20_BOUNDARIES = 3
FORK_PILOT_TRAIN_PRIORITY_FRACTION = .65
FORK_PILOT_SEED = 20260914
FORK_PILOT_PRINT_EVERY_TREES = 4
FORK_PILOT_SKIP_UNUSED_RENDERS = True
FORK_PILOT_RENDER_LEAD = 2
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


def _three_boundary_score(profile: tuple[float, ...], chunk_idx: int) -> float:
    values = profile[chunk_idx:chunk_idx + FORK_PILOT_U20_BOUNDARIES]
    if len(values) != FORK_PILOT_U20_BOUNDARIES:
        raise ValueError("U20 root does not have three complete boundaries remaining")
    return float(np.mean(values))


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
        "source_three_boundary_u20": _three_boundary_score(profile, chunk_idx),
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
                -item["source_three_boundary_u20"], item["selection_tiebreak"]))
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
        # Every arm gets the same intervention opportunity: the selected
        # 10-action chunk plus two complete 10-action continuation chunks.
        eligible = list(range(
            1, len(profile) - FORK_PILOT_U20_BOUNDARIES + 1))
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
            key=lambda index: (-_three_boundary_score(profile, index), index))
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
        "u20_boundaries": FORK_PILOT_U20_BOUNDARIES,
        "skip_unused_renders": FORK_PILOT_SKIP_UNUSED_RENDERS,
        "render_lead": FORK_PILOT_RENDER_LEAD,
        "future_training_priority_fraction": FORK_PILOT_TRAIN_PRIORITY_FRACTION,
        "source_scope": (
            "Q-planning snapshot training split; LIBERO-PRO train suites only; "
            "position and ten-state behavior-seed suites excluded; roots restored "
            "from persisted source simulator states, policy inputs, and actions"),
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
    if payload.get("skip_unused_renders") is not True:
        raise ValueError("fork-pilot manifest must use validated sparse rendering")
    if payload.get("render_lead") != FORK_PILOT_RENDER_LEAD:
        raise ValueError("fork-pilot manifest uses the wrong camera render lead")
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


_SOURCE_FIDELITY_ARRAYS = (
    "initial_state", "episode_seed", "actions_env", "sim_state_t_plus_1", "chunk_start_steps",
    "chunk_noise_seeds", "bellman/action", "boundary/step",
    "boundary/raw_agentview", "boundary/raw_wrist", "boundary/raw_robot_state",
    "boundary/policy_proprio", "boundary/sim_state",
)


def _load_selected_training_arrays(store, path: str,
                                   names=_SOURCE_FIDELITY_ARRAYS
                                   ) -> dict[str, np.ndarray]:
    """Load only source-fidelity arrays from a legacy or multipart artifact."""
    wanted = tuple(names)
    payload = _download_with_retry(store, path)
    if path.endswith(".npz"):
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            missing = sorted(set(wanted) - set(archive.files))
            if missing:
                raise ValueError(f"training artifact is missing {missing}")
            return {name: np.asarray(archive[name]).copy() for name in wanted}

    manifest = json.loads(payload)
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError(f"invalid multipart training-data manifest at {path}")
    missing = sorted(set(wanted) - set(arrays))
    if missing:
        raise ValueError(f"training artifact is missing {missing}")
    part_cache: dict[str, dict[str, np.ndarray]] = {}
    result = {}
    for name in wanted:
        spec = arrays[name]
        shape, dtype = tuple(spec["shape"]), np.dtype(spec["dtype"])
        output = np.empty(shape, dtype=dtype)
        scalar = not shape
        populated_scalar = False
        for part in spec["parts"]:
            key = str(part["path"])
            if key not in part_cache:
                part_payload = _download_with_retry(store, key)
                with np.load(io.BytesIO(part_payload), allow_pickle=False) as archive:
                    part_cache[key] = {
                        item: np.asarray(archive[item]).copy() for item in archive.files}
            value = part_cache[key][name]
            start, stop = part.get("start"), part.get("stop")
            if start is None:
                output[...] = value
                populated_scalar = scalar
            else:
                output[int(start):int(stop)] = value
        if scalar and not populated_scalar:
            raise ValueError(f"multipart scalar {name} has no payload")
        result[name] = output
    return result


def _load_source_fidelity_bundles(store, items: list[dict]) -> dict[str, dict]:
    rollout_ids = sorted({str(item["source_rollout_id"]) for item in items})
    rows = []
    for start in range(0, len(rollout_ids), 100):
        batch = rollout_ids[start:start + 100]
        rows.extend(store.fetch_all(
            "rollouts",
            "rollout_id,trajectory_path,training_data_path,success,n_steps,"
            "init_state_hash,bddl_sha256",
            configure=lambda query, batch=batch: query.in_("rollout_id", batch),
            order_by=("rollout_id",)))
    by_id = {str(row["rollout_id"]): row for row in rows}
    missing = [rollout_id for rollout_id in rollout_ids
               if rollout_id not in by_id
               or not by_id[rollout_id].get("trajectory_path")
               or not by_id[rollout_id].get("training_data_path")]
    if missing:
        raise ValueError(
            "source-fidelity preflight requires trajectory and training artifacts; "
            f"missing {missing[:3]}")
    bundles = {}
    for index, rollout_id in enumerate(rollout_ids, 1):
        row = by_id[rollout_id]
        compact_actions = _trajectory_actions_from_payload(
            _download_with_retry(store, row["trajectory_path"]))
        arrays = _load_selected_training_arrays(store, row["training_data_path"])
        training_actions = np.asarray(arrays["actions_env"], np.float32)
        if not np.array_equal(compact_actions, training_actions):
            raise AssertionError(
                f"source {rollout_id} compact/training actions differ")
        bundles[rollout_id] = {
            "row": row, "arrays": arrays, "actions": compact_actions}
        print(f"[qfork] source-fidelity artifacts: {index}/{len(rollout_ids)}",
              flush=True)
    return bundles


def _raw_robot_state_from_obs(obs) -> np.ndarray:
    return np.concatenate([
        obs["robot0_eef_pos"], obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32, copy=True)


def _source_policy_observation(arrays: dict, boundary_index: int,
                               task_desc: str) -> dict:
    """Recreate obs_to_policy's input from the exact stored source boundary."""
    from .libero_env import obs_to_policy

    robot = np.asarray(
        arrays["boundary/raw_robot_state"][boundary_index], np.float32)
    raw = {
        "agentview_image": np.asarray(
            arrays["boundary/raw_agentview"][boundary_index]).copy(),
        "robot0_eye_in_hand_image": np.asarray(
            arrays["boundary/raw_wrist"][boundary_index]).copy(),
        "robot0_eef_pos": robot[:3].copy(),
        "robot0_eef_quat": robot[3:7].copy(),
        "robot0_gripper_qpos": robot[7:9].copy(),
    }
    return obs_to_policy(raw, task_desc)


def _set_flat_sim_state(env, state: np.ndarray) -> float:
    _, sim = _unwrap_sim(env)
    value = np.asarray(state).copy()
    setter = getattr(sim, "set_state_from_flattened", None)
    if not callable(setter):
        raise RuntimeError("MuJoCo simulator has no set_state_from_flattened")
    setter(value)
    sim.forward()
    return float(np.max(np.abs(np.asarray(sim.get_state().flatten()) - value)))


def _execute_source_actions(env, actions: np.ndarray, *, capture_frames=False,
                            initial_frame=None) -> dict:
    frames = []
    if capture_frames and initial_frame is not None:
        frames.append(np.ascontiguousarray(initial_frame[::-1, ::-1]))
    success = False
    done_at = None
    obs = None
    for index, action in enumerate(np.asarray(actions, np.float32)):
        obs, _, done, _ = env.step(action)
        success = bool(env.check_success())
        if capture_frames:
            frames.append(np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]))
        if success or done:
            done_at = index + 1
            break
    return {
        "success": success, "steps": done_at or len(actions), "obs": obs,
        "frames": frames,
    }


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
                          store=None, _manifest_loader=None,
                          _intervention_actions: int = FORK_PILOT_EXECUTED_ACTIONS,
                          _replan_actions: int = FORK_PILOT_EXECUTED_ACTIONS,
                          _run_name: str = "qplanning_fork_pilot") -> dict:
    """Collect one resume-safe shard of all three root-selection strategies."""
    from . import libero_env, models
    from .store import SupabaseStore

    if shard_count != FORK_PILOT_SHARDS or not 0 <= int(shard_index) < shard_count:
        raise ValueError("fork pilot requires shard_count=4 and shard_index in [0,4)")
    store = store or SupabaseStore()
    manifest_loader = _manifest_loader or load_fork_pilot_manifest
    document = manifest_loader(store, manifest_path)
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

    print(f"[qfork] preparing {len(pending)} pending trees: resolving LIBERO-PRO tasks", flush=True)
    lookup = _episode_lookup(items)
    print("[qfork] downloading persisted source states, inputs, and actions", flush=True)
    source_bundles = _load_source_fidelity_bundles(store, pending)
    print("[qfork] loading PI0.5 policy", flush=True)
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    policy.model._pnp.num_steps = FORK_PILOT_STOCK_DENOISE_STEPS
    experiments = sorted({item["experiment"] for item in items})
    store.start_run(
        _run_name, "libero_pro", experiments[0],
        config={
            "manifest_path": manifest_path,
            "manifest_hash": document["manifest_hash"],
            "shard_count": shard_count, "shard_index": int(shard_index),
            "trees": len(items), "candidate_count": FORK_PILOT_CANDIDATES,
            "executed_actions": int(_intervention_actions),
            "candidate_intervention_actions": int(_intervention_actions),
            "continuation_replan_actions": int(_replan_actions),
            "stock_denoise_steps": FORK_PILOT_STOCK_DENOISE_STEPS,
            "proposal_denoise_steps": FORK_PILOT_PROPOSAL_DENOISE_STEPS,
            "skip_unused_renders": FORK_PILOT_SKIP_UNUSED_RENDERS,
            "render_lead": FORK_PILOT_RENDER_LEAD,
            "videos": False,
        })
    new_trees = 0
    collection_started = time.monotonic()
    try:
        for item in pending:
            tree_started = time.monotonic()
            print(
                f"[qfork] starting tree {new_trees + 1}/{len(pending)} | "
                f"{item['strategy']} | {item['suite']} | chunk {item['chunk_idx']}",
                flush=True)
            ep = dict(lookup[(
                item["suite"], int(item["task_idx"]), int(item["episode_idx"]))])
            ep["behavior_seed_index"] = 0
            source_bundle = source_bundles[str(item["source_rollout_id"])]
            source = _source_boundary(source_bundle, item)
            if str(ep.get("init_state_hash") or "") != str(item["init_state_hash"]):
                raise AssertionError(
                    f"source identity mismatch for {item['source_rollout_id']}")
            env = libero_env.make_env(ep["bddl_path"])
            try:
                result = collect_replay_candidate_group(
                    env, ep, policy, preprocess, postprocess, device,
                    chunk_idx=int(item["chunk_idx"]),
                    uncertainty_stratum=item["strategy"],
                    prefix_length=int(_intervention_actions),
                    candidate_count=FORK_PILOT_CANDIDATES,
                    experiment=item["experiment"],
                    collection_split="fork_pilot",
                    manifest_hash=document["manifest_hash"],
                    model_revision=payload["policy_revision"],
                    n_action_steps=int(_replan_actions),
                    candidate_num_inference_steps=FORK_PILOT_PROPOSAL_DENOISE_STEPS,
                    skip_unused_renders=FORK_PILOT_SKIP_UNUSED_RENDERS,
                    render_lead=FORK_PILOT_RENDER_LEAD,
                    replay_actions_override=source_bundle["actions"][
                        :int(item["chunk_idx"]) * int(_replan_actions)],
                    source_sim_state_override=source["sim_state"],
                    source_policy_observation_override=_source_policy_observation(
                        source_bundle["arrays"], source["boundary_index"],
                        ep["task_desc"])),
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
                "source_three_boundary_u20": item["source_three_boundary_u20"],
                "source_u20_profile": item["source_u20_profile"],
                "normalized_chunk_position": item["normalized_chunk_position"],
                "future_training_priority_fraction": FORK_PILOT_TRAIN_PRIORITY_FRACTION,
                "candidate_intervention_actions": int(_intervention_actions),
                "continuation_replan_actions": int(_replan_actions),
                "source_restoration": "persisted_sim_state_and_policy_input",
            })
            store.register_candidate_group(group, candidates)
            new_trees += 1
            successes = [bool(candidate["success"]) for candidate in candidates]
            stock_success = bool(candidates[0]["success"])
            elapsed = time.monotonic() - collection_started
            seconds_per_tree = elapsed / new_trees
            eta_seconds = seconds_per_tree * (len(pending) - new_trees)
            print(
                f"[qfork] completed tree {new_trees}/{len(pending)} | "
                f"branches={sum(successes)}/{len(successes)} success | "
                f"stock={'S' if stock_success else 'F'} | "
                f"mixed={len(set(successes)) > 1} | "
                f"tree={time.monotonic() - tree_started:.1f}s | "
                f"elapsed={elapsed / 60:.1f}m | ETA={eta_seconds / 60:.1f}m",
                flush=True)
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
                                   store=None, _manifest_loader=None,
                                   _intervention_actions: int = FORK_PILOT_EXECUTED_ACTIONS,
                                   _replan_actions: int = FORK_PILOT_EXECUTED_ACTIONS
                                   ) -> list[dict]:
    """Replay one root per strategy twice and require identical stock branches."""
    from . import libero_env, models
    from .store import SupabaseStore

    store = store or SupabaseStore()
    manifest_loader = _manifest_loader or load_fork_pilot_manifest
    document = manifest_loader(store, manifest_path)
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
                    prefix_length=int(_intervention_actions), candidate_count=2,
                    experiment=item["experiment"], collection_split="restoration_preflight",
                    manifest_hash=document["manifest_hash"],
                    model_revision=payload["policy_revision"],
                    n_action_steps=int(_replan_actions),
                    candidate_num_inference_steps=FORK_PILOT_PROPOSAL_DENOISE_STEPS,
                    skip_unused_renders=FORK_PILOT_SKIP_UNUSED_RENDERS,
                    render_lead=FORK_PILOT_RENDER_LEAD,
                    replay_actions_override=source_replay_actions[
                        item["source_rollout_id"]][
                            :int(item["chunk_idx"]) * int(_replan_actions)])
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
        repeat_equal = {
            field: metadata_a[field] == metadata_b[field]
            for field in restoration_fields}
        independent_root_replay_exact = all(repeat_equal.values())
        noise_exact = (
            candidate_a["metadata_json"]["candidate_noise_sha256"]
            == candidate_b["metadata_json"]["candidate_noise_sha256"])
        prediction_repeat_exact = policy_error <= 1e-6 and env_error <= 1e-6
        passed = (
            metadata_a["exact_sim_state_validated"]
            and metadata_b["exact_sim_state_validated"]
            and metadata_a["parent_replay_source"] == "stored_source_trajectory"
            and metadata_b["parent_replay_source"] == "stored_source_trajectory"
            and metadata_a["parent_replay_actions_sha256"]
            == metadata_b["parent_replay_actions_sha256"]
            and noise_exact)
        report = {
            "strategy": item["strategy"], "suite": item["suite"],
            "task_idx": item["task_idx"], "episode_idx": item["episode_idx"],
            "chunk_idx": item["chunk_idx"], "policy_chunk_max_abs": policy_error,
            "env_chunk_max_abs": env_error,
            "branch_restore_max_abs": float(max(
                metadata_a["replay_state_max_abs_after_correction"],
                metadata_b["replay_state_max_abs_after_correction"])),
            "parent_actions_repeat_exact": bool(
                repeat_equal["parent_replay_actions_sha256"]),
            "parent_terminal_events_a": len(
                metadata_a["canonical_parent_replay_reported_terminal_events"]),
            "parent_terminal_events_b": len(
                metadata_b["canonical_parent_replay_reported_terminal_events"]),
            "root_sim_repeat_exact": bool(
                repeat_equal["root_sim_state_sha256"]),
            "root_policy_input_repeat_exact": bool(
                repeat_equal["root_policy_input_sha256"]),
            "independent_root_replay_exact": bool(independent_root_replay_exact),
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


def _source_boundary(bundle: dict, item: dict) -> dict:
    arrays = bundle["arrays"]
    chunk_index = int(item["chunk_idx"])
    starts = np.asarray(arrays["chunk_start_steps"], np.int64)
    boundary_steps = np.asarray(arrays["boundary/step"], np.int64)
    if chunk_index >= len(starts):
        raise ValueError(
            f"source {item['source_rollout_id']} has no chunk {chunk_index}")
    root_step = int(starts[chunk_index])
    matches = np.flatnonzero(boundary_steps == root_step)
    if len(matches) != 1:
        raise ValueError(
            f"source root step {root_step} maps to {len(matches)} boundaries")
    boundary_index = int(matches[0])
    if root_step != chunk_index * FORK_PILOT_EXECUTED_ACTIONS:
        raise AssertionError(
            f"source chunk stride is not 10: chunk={chunk_index}, step={root_step}")
    return {
        "root_step": root_step,
        "boundary_index": boundary_index,
        "sim_state": np.asarray(
            arrays["sim_state_t_plus_1"][root_step]).copy(),
        "boundary_sim_state": np.asarray(
            arrays["boundary/sim_state"][boundary_index]).copy(),
        "raw_agentview": np.asarray(
            arrays["boundary/raw_agentview"][boundary_index]).copy(),
        "raw_wrist": np.asarray(
            arrays["boundary/raw_wrist"][boundary_index]).copy(),
        "raw_robot": np.asarray(
            arrays["boundary/raw_robot_state"][boundary_index], np.float32).copy(),
        "policy_proprio": np.asarray(
            arrays["boundary/policy_proprio"][boundary_index], np.float32).copy(),
        "policy_chunk": np.asarray(
            arrays["bellman/action"][boundary_index], np.float32).copy(),
        "noise_seed": int(np.asarray(
            arrays["chunk_noise_seeds"])[boundary_index]),
    }


def _reset_to_source_root(env, ep, policy, parent_actions: np.ndarray,
                          source_state: np.ndarray, *, sparse_rendering: bool):
    obs, terminal_events = _reset_and_replay_actions(
        env, ep, policy, parent_actions,
        skip_unused_renders=sparse_rendering,
        render_lead=FORK_PILOT_RENDER_LEAD)
    _, sim = _unwrap_sim(env)
    replay_state = np.asarray(sim.get_state().flatten()).copy()
    correction_error = _set_flat_sim_state(env, source_state)
    return obs, terminal_events, replay_state, correction_error


def _collector_continuation_video(env, obs, ep, policy, preprocess, postprocess,
                                  device, *, prefix: np.ndarray,
                                  branch_seed: int, steps_already: int,
                                  initial_frame: np.ndarray) -> dict:
    """Video-enabled mirror of the 10-action collector continuation."""
    from .libero_env import obs_to_policy
    from .rollout import _draw_chunk_noise, chunk_noise_seed

    frames = [np.ascontiguousarray(initial_frame[::-1, ::-1])]
    steps = int(steps_already)
    success = False
    for action in np.asarray(prefix, np.float32):
        obs, _, done, _ = env.step(action)
        steps += 1
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if env.check_success():
            success = True
            return {"success": True, "steps": steps, "frames": frames}
        if done or steps >= int(ep["max_steps"]):
            return {"success": False, "steps": steps, "frames": frames}

    est_chunks = max(1, round(int(ep["max_steps"]) / FORK_PILOT_EXECUTED_ACTIONS))
    queue, replan = [], 0
    while steps < int(ep["max_steps"]):
        if not queue:
            chunk_index = steps // FORK_PILOT_EXECUTED_ACTIONS
            policy.model._pnp.chunk_pos = min(chunk_index / est_chunks, 1.0)
            batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
            noise = _draw_chunk_noise(
                policy, device, chunk_noise_seed(branch_seed, replan))
            chunk, _ = predict_clean_chunk(policy, batch, noise)
            queue = list(chunk.squeeze(0).detach().cpu().numpy()[
                :FORK_PILOT_EXECUTED_ACTIONS])
            replan += 1
        action = postprocess_chunk(
            np.asarray(queue.pop(0))[None], postprocess, device)[0]
        obs, _, done, _ = env.step(action)
        steps += 1
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if env.check_success():
            success = True
            break
        if done:
            break
    return {"success": success, "steps": steps, "frames": frames}


def run_fork_source_fidelity_preflight(
        *, manifest_path: str = FORK_PILOT_MANIFEST_PATH, store=None,
        video_dir: str | Path | None = None, state_atol: float = 1e-6,
        _manifest_loader=None,
        _intervention_actions: int = FORK_PILOT_EXECUTED_ACTIONS) -> dict:
    """Audit current fork restoration against the original stored source rollout.

    Unlike :func:`run_fork_restoration_preflight`, this test does not merely ask
    whether two fresh replays agree.  It compares each fresh root to the exact
    source simulator state and action/policy tensors saved during collection,
    checks both full and root-suffix source replay outcomes, and emits one video
    pair that isolates the collector's newly sampled continuation.
    """
    from . import libero_env, models
    from .libero_env import init_state_hash, obs_to_policy
    from .rollout import _draw_chunk_noise, _encode_mp4, episode_seed
    from .store import SupabaseStore

    store = store or SupabaseStore()
    manifest_loader = _manifest_loader or load_fork_pilot_manifest
    document = manifest_loader(store, manifest_path)
    payload = document["payload"]
    roots = [next(item for item in payload["trees"] if item["strategy"] == strategy)
             for strategy in FORK_PILOT_STRATEGIES]
    video_item = next(
        (item for item in payload["trees"]
         if item["strategy"] == "random" and bool(item["source_success"])),
        next(item for item in payload["trees"] if bool(item["source_success"])))
    requested = roots + ([] if video_item in roots else [video_item])
    lookup = _episode_lookup(requested)
    bundles = _load_source_fidelity_bundles(store, requested)
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    policy.model._pnp.num_steps = FORK_PILOT_STOCK_DENOISE_STEPS

    reports = []
    for item in roots:
        key = (item["suite"], int(item["task_idx"]), int(item["episode_idx"]))
        ep = dict(lookup[key]); ep["behavior_seed_index"] = 0
        bundle = bundles[str(item["source_rollout_id"])]
        arrays, actions, row = bundle["arrays"], bundle["actions"], bundle["row"]
        source = _source_boundary(bundle, item)
        root_step = source["root_step"]
        source_success = bool(item["source_success"])
        source_row_success_exact = bool(row["success"]) == source_success
        identity_exact = (
            str(ep.get("init_state_hash") or init_state_hash(ep["init_state"]))
            == str(item["init_state_hash"])
            == str(row.get("init_state_hash") or ""))
        initial_state_exact = np.array_equal(
            np.asarray(ep["init_state"]), np.asarray(arrays["initial_state"]))
        bddl_exact = str(ep.get("bddl_sha256") or "") == str(
            row.get("bddl_sha256") or "")
        boundary_sim_exact = np.array_equal(
            source["sim_state"], source["boundary_sim_state"])
        source_seed = int(np.asarray(arrays["episode_seed"]))
        expected_seed = episode_seed(ep["init_state"], int(ep["ep_idx"]))
        source_seed_exact = source_seed == expected_seed

        env = libero_env.make_env(ep["bddl_path"])
        try:
            replay_obs, root_events = _reset_and_replay_actions(
                env, ep, policy, actions[:root_step],
                skip_unused_renders=FORK_PILOT_SKIP_UNUSED_RENDERS,
                render_lead=FORK_PILOT_RENDER_LEAD)
            _, sim = _unwrap_sim(env)
            replay_state = np.asarray(sim.get_state().flatten()).copy()
            root_state_max_abs = float(np.max(np.abs(
                replay_state - source["sim_state"])))
            replay_robot = _raw_robot_state_from_obs(replay_obs)
            robot_state_max_abs = float(np.max(np.abs(
                replay_robot - source["raw_robot"])))
            agentview_mae = float(np.mean(np.abs(
                np.asarray(replay_obs["agentview_image"], np.float32)
                - np.asarray(source["raw_agentview"], np.float32))))
            wrist_mae = float(np.mean(np.abs(
                np.asarray(replay_obs["robot0_eye_in_hand_image"], np.float32)
                - np.asarray(source["raw_wrist"], np.float32))))

            policy.model._pnp.chunk_pos = min(
                int(item["chunk_idx"]) /
                max(1, round(int(ep["max_steps"]) / FORK_PILOT_EXECUTED_ACTIONS)), 1.0)
            noise = _draw_chunk_noise(policy, device, source["noise_seed"])
            source_policy_observation = _source_policy_observation(
                arrays, source["boundary_index"], ep["task_desc"])
            source_policy_proprio = source_policy_observation["observation.state"]
            if hasattr(source_policy_proprio, "detach"):
                source_policy_proprio = source_policy_proprio.detach().cpu().numpy()
            source_policy_proprio_error = float(np.max(np.abs(
                np.asarray(source_policy_proprio, np.float32)
                - source["policy_proprio"])))
            stored_batch = preprocess(source_policy_observation)
            replay_batch = preprocess(obs_to_policy(replay_obs, ep["task_desc"]))
            stored_prediction, _ = predict_clean_chunk(policy, stored_batch, noise)
            replay_prediction, _ = predict_clean_chunk(policy, replay_batch, noise)
            stored_prediction = stored_prediction.squeeze(0).detach().cpu().numpy()
            replay_prediction = replay_prediction.squeeze(0).detach().cpu().numpy()
            source_chunk = source["policy_chunk"]
            stored_input_chunk_rms = float(np.sqrt(np.mean(
                (stored_prediction[:10] - source_chunk[:10]) ** 2)))
            replay_input_chunk_rms = float(np.sqrt(np.mean(
                (replay_prediction[:10] - source_chunk[:10]) ** 2)))

            _reset_and_replay_actions(
                env, ep, policy, [], skip_unused_renders=False,
                render_lead=FORK_PILOT_RENDER_LEAD)
            full = _execute_source_actions(env, actions)
            full_replay_success = bool(full["success"])
            full_replay_steps_exact = int(full["steps"]) == int(row["n_steps"])

            _, _, _, snap_error = _reset_to_source_root(
                env, ep, policy, actions[:root_step], source["sim_state"],
                sparse_rendering=False)
            suffix = _execute_source_actions(env, actions[root_step:])
            suffix_replay_success = bool(suffix["success"])
        finally:
            env.close()

        pre_snap_replay_exact = root_state_max_abs <= float(state_atol)
        behavior_exact = (
            full_replay_success == source_success
            and suffix_replay_success == source_success
            and full_replay_steps_exact and not root_events)
        passed = (
            source_row_success_exact and identity_exact and initial_state_exact
            and bddl_exact and boundary_sim_exact and source_seed_exact
            and source_policy_proprio_error <= float(state_atol)
            and snap_error <= float(state_atol)
            and behavior_exact)
        report = {
            "strategy": item["strategy"], "suite": item["suite"],
            "task_idx": int(item["task_idx"]),
            "episode_idx": int(item["episode_idx"]),
            "chunk_idx": int(item["chunk_idx"]), "root_step": root_step,
            "source_success": source_success,
            "source_row_success_exact": bool(source_row_success_exact),
            "identity_exact": bool(identity_exact),
            "initial_state_exact": bool(initial_state_exact),
            "bddl_exact": bool(bddl_exact),
            "artifact_boundary_sim_exact": bool(boundary_sim_exact),
            "source_seed_exact": bool(source_seed_exact),
            "source_full_replay_success": bool(full_replay_success),
            "source_full_replay_steps": int(full["steps"]),
            "source_logged_steps": int(row["n_steps"]),
            "source_full_replay_steps_exact": bool(full_replay_steps_exact),
            "source_suffix_from_exact_root_success": bool(suffix_replay_success),
            "behavioral_replay_exact": bool(behavior_exact),
            "collector_root_sim_max_abs_vs_source": root_state_max_abs,
            "collector_root_robot_max_abs_vs_source": robot_state_max_abs,
            "collector_root_agentview_mae_0_255": agentview_mae,
            "collector_root_wrist_mae_0_255": wrist_mae,
            "persisted_policy_proprio_reconstruction_max_abs":
                source_policy_proprio_error,
            "stored_input_prediction_rms_vs_logged_first10": stored_input_chunk_rms,
            "collector_input_prediction_rms_vs_logged_first10": replay_input_chunk_rms,
            "exact_source_state_snap_max_abs": float(snap_error),
            "pre_snap_action_replay_exact": bool(pre_snap_replay_exact),
            "parent_replay_terminal_events": len(root_events),
            "current_collector_source_fidelity_passed": bool(passed),
            "passed": bool(passed),
        }
        print(report, flush=True)
        reports.append(report)

    video = None
    if video_dir is not None:
        item = video_item
        key = (item["suite"], int(item["task_idx"]), int(item["episode_idx"]))
        ep = dict(lookup[key]); ep["behavior_seed_index"] = 0
        bundle = bundles[str(item["source_rollout_id"])]
        source = _source_boundary(bundle, item)
        actions = bundle["actions"]
        root_step = source["root_step"]
        env = libero_env.make_env(ep["bddl_path"])
        try:
            _reset_to_source_root(
                env, ep, policy, actions[:root_step], source["sim_state"],
                sparse_rendering=False)
            fixed = _execute_source_actions(
                env, actions[root_step:], capture_frames=True,
                initial_frame=source["raw_agentview"])

            branch_obs, _, _, _ = _reset_to_source_root(
                env, ep, policy, actions[:root_step], source["sim_state"],
                sparse_rendering=False)
            branch = _collector_continuation_video(
                env, branch_obs, ep, policy, preprocess, postprocess, device,
                prefix=actions[root_step:root_step + int(_intervention_actions)],
                branch_seed=int(np.asarray(bundle["arrays"]["episode_seed"])) ^ 0x51A7,
                steps_already=root_step, initial_frame=source["raw_agentview"])
        finally:
            env.close()
        output = Path(video_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        stem = (f"{item['strategy']}_{item['suite']}_t{item['task_idx']}_"
                f"e{item['episode_idx']}_c{item['chunk_idx']}")
        fixed_path = output / f"{stem}_stored_source_suffix.mp4"
        branch_path = output / f"{stem}_collector_continuation.mp4"
        fixed_path.write_bytes(_encode_mp4(fixed["frames"]))
        branch_path.write_bytes(_encode_mp4(branch["frames"]))
        video = {
            "identity": stem,
            "left_label": "stored source suffix from exact source root",
            "left_path": str(fixed_path), "left_success": bool(fixed["success"]),
            "right_label": (
                f"same first {int(_intervention_actions)} actions, "
                "then collector continuation"),
            "right_path": str(branch_path), "right_success": bool(branch["success"]),
        }
        print(video, flush=True)
    return {"reports": reports, "video": video}


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
            "mean_selected_three_boundary_u20": float(group.source_three_boundary_u20.mean())
                if len(group) else np.nan,
        })
    return trees, pd.DataFrame(summary_rows)
