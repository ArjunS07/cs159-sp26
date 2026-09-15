"""Primitives for deterministic same-state candidate collection in LIBERO."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json

import numpy as np
import torch

from ..config import LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT
from ..libero_env import obs_to_policy, set_camera_observables
from ..rollout import _draw_chunk_noise, chunk_noise_seed, episode_seed
from ..sampler import _temp_strategy


def _env_chain(env):
    """Yield an environment and its nested ``.env`` wrappers once each."""
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)
        if current is None:
            break


def _unwrap_sim(env):
    for current in _env_chain(env):
        sim = getattr(current, "sim", None)
        if sim is not None:
            return current, sim
    raise TypeError("environment does not expose a MuJoCo sim")


def _content_digest(value) -> str:
    """Stable diagnostic hash for nested policy inputs and replay arrays."""
    digest = hashlib.sha256()

    def update(item):
        if torch.is_tensor(item):
            item = (item.detach().to("cpu", torch.float32) if item.is_floating_point()
                    else item.detach().cpu()).numpy()
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(f"array:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}".encode())
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def candidate_group_id(benchmark, suite, task_idx, episode_idx, chunk_idx, *, namespace="",
                       trajectory_seed=None) -> str:
    identity = f"{benchmark}|{suite}|{task_idx}|{episode_idx}|{chunk_idx}"
    if trajectory_seed is not None:
        identity += f"|trajectory={trajectory_seed}"
    raw = (f"{namespace}|{identity}" if namespace else identity).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def build_stratified_manifest(rollout_rows, euler_rows, targets=None, seed=42):
    """Choose one low/medium/high-uncertainty branch state per source rollout."""
    targets = targets or {"libero": 50, "libero_pro": 75}
    outcomes = defaultdict(list)
    by_id = {}
    for row in rollout_rows:
        by_id[row["rollout_id"]] = row
        outcomes[(row["benchmark"], row["suite"], row["task_idx"])].append(bool(row["success"]))
    hard = {key for key, values in outcomes.items() if .1 < np.mean(values) < .9}
    uncertainty = defaultdict(list)
    for row in euler_rows:
        uncertainty[(row["rollout_id"], int(row["chunk_idx"]))].append(float(row["u_mean"]))
    candidates = []
    for (rollout_id, chunk_idx), values in uncertainty.items():
        rollout = by_id.get(rollout_id)
        if rollout is None:
            continue
        key = (rollout["benchmark"], rollout["suite"], rollout["task_idx"])
        if key not in hard:
            continue
        candidates.append({**rollout, "chunk_idx": chunk_idx,
                           "u_mean": float(np.mean(values))})
    selected = []
    for benchmark, target in targets.items():
        pool = [row for row in candidates if row["benchmark"] == benchmark]
        if not pool:
            continue
        q1, q2 = np.quantile([row["u_mean"] for row in pool], [1 / 3, 2 / 3])
        for row in pool:
            row["uncertainty_stratum"] = ("low" if row["u_mean"] <= q1 else
                                           "mid" if row["u_mean"] <= q2 else "high")
        per = {name: target // 3 + (1 if i < target % 3 else 0)
               for i, name in enumerate(("low", "mid", "high"))}
        used_identities = set()
        for stratum in ("low", "mid", "high"):
            stratum_pool = [row for row in pool if row["uncertainty_stratum"] == stratum]
            stratum_pool.sort(key=lambda row: hashlib.sha256(
                f"{seed}|{row['rollout_id']}|{row['chunk_idx']}".encode()).hexdigest())
            for row in stratum_pool:
                identity = (row["suite"], row["task_idx"], row["episode_idx"])
                if identity in used_identities:
                    continue
                selected.append(row); used_identities.add(identity)
                if sum(r["benchmark"] == benchmark and r["uncertainty_stratum"] == stratum
                       for r in selected) >= per[stratum]:
                    break
    return selected


def collection_manifest_hash(rows) -> str:
    """Hash a collection manifest independently of row/query ordering."""
    fields = (
        "benchmark", "suite", "task_idx", "episode_idx", "rollout_id",
        "chunk_idx", "u_mean", "uncertainty_stratum", "success",
        "trajectory_seed", "collection_split",
    )
    canonical = [
        {field: row.get(field) for field in fields}
        for row in sorted(rows, key=lambda row: (
            row["benchmark"], row["suite"], int(row["task_idx"]),
            int(row["episode_idx"]), int(row["chunk_idx"])))
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_seeded_pro_manifest(rows, *, development_target=240, test_target=160,
                              seed=20260728):
    """Create deterministic, disjoint trajectory-seeded PRO development/test cohorts.

    Source candidate outcomes are used only to balance the development cohort.
    Test membership is assigned first and is independent of future candidate outcomes.
    """
    pro = [dict(row) for row in rows if row.get("benchmark") == "libero_pro"]
    if not pro:
        raise ValueError("seeded PRO manifest requires LIBERO-PRO source rows")
    by_identity = {}
    for row in pro:
        identity = (row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
        previous = by_identity.get(identity)
        rank = (-float(row.get("u_mean", 0)), int(row["chunk_idx"]), row["rollout_id"])
        if previous is None or rank < previous[0]:
            by_identity[identity] = (rank, row)
    base = [value[1] for value in by_identity.values()]
    if len(base) < development_target + test_target:
        raise ValueError(
            "seeded PRO manifest requires one distinct base identity per requested group")

    def with_seed(row, split):
        raw = (f"{seed}|{split}|{row['suite']}|{row['task_idx']}|"
               f"{row['episode_idx']}").encode()
        return {**row,
                "trajectory_seed": int(hashlib.sha256(raw).hexdigest()[:8], 16),
                "collection_split": split}

    test_buckets = defaultdict(list)
    for row in base:
        test_buckets[row["suite"]].append(row)
    for suite in test_buckets:
        test_buckets[suite].sort(key=lambda row: hashlib.sha256(
            f"{seed}|test|{row['suite']}|{row['task_idx']}|{row['episode_idx']}".encode()
        ).hexdigest())
    test_base = []
    while len(test_base) < test_target:
        progressed = False
        for suite in sorted(test_buckets):
            if test_buckets[suite] and len(test_base) < test_target:
                test_base.append(test_buckets[suite].pop())
                progressed = True
        if not progressed:
            break
    test = [with_seed(row, "confirmatory_test") for row in test_base]
    test_identities = {(row["suite"], row["task_idx"], row["episode_idx"])
                       for row in test_base}
    available = [row for row in base if (
        row["suite"], row["task_idx"], row["episode_idx"]) not in test_identities]
    # Interleave source failures and successes, then suites, to avoid an easy cohort.
    buckets = defaultdict(list)
    for row in available:
        buckets[(row["suite"], bool(row.get("success")))].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda row: hashlib.sha256(
            f"{seed}|development|{row['suite']}|{row['task_idx']}|"
            f"{row['episode_idx']}".encode()).hexdigest())
    development = []
    while len(development) < development_target and any(buckets.values()):
        for key in sorted(buckets, key=str):
            if buckets[key] and len(development) < development_target:
                development.append(with_seed(buckets[key].pop(), "development"))
    if len(development) != development_target or len(test) != test_target:
        raise ValueError("insufficient source identities for requested seeded PRO manifest")
    return {"development": development, "confirmatory_test": test}


def build_targeted_manifests(
        rollout_rows, euler_rows, excluded_identities, *,
        development_targets=None, test_targets=None,
        development_failure_fraction=.70, seed=42, allow_shortfall=False):
    """Build outcome-blind prospective-test and failure-enriched development manifests.

    One high-uncertainty state is selected per previously unused episode. Test
    membership is assigned before development enrichment and never depends on
    candidate outcomes.
    """
    development_targets = development_targets or {"libero": 180, "libero_pro": 270}
    test_targets = test_targets or {"libero": 60, "libero_pro": 90}
    if not 0 <= development_failure_fraction <= 1:
        raise ValueError("development_failure_fraction must be in [0, 1]")
    excluded = {tuple(identity) for identity in excluded_identities}
    by_id = {row["rollout_id"]: row for row in rollout_rows}
    uncertainty = defaultdict(list)
    for row in euler_rows:
        uncertainty[(row["rollout_id"], int(row["chunk_idx"]))].append(
            float(row["u_mean"]))

    candidates = []
    for (rollout_id, chunk_idx), values in uncertainty.items():
        rollout = by_id.get(rollout_id)
        if rollout is None:
            continue
        identity = (
            rollout["benchmark"], rollout["suite"], int(rollout["task_idx"]),
            int(rollout["episode_idx"]),
        )
        if identity in excluded:
            continue
        candidates.append({
            **rollout, "chunk_idx": chunk_idx,
            "u_mean": float(np.mean(values)), "uncertainty_stratum": "high",
        })

    episode_best = {}
    for candidate in candidates:
        identity = (
            candidate["benchmark"], candidate["suite"], int(candidate["task_idx"]),
            int(candidate["episode_idx"]),
        )
        previous = episode_best.get(identity)
        candidate_rank = (
            -candidate["u_mean"], int(candidate["chunk_idx"]),
            candidate["rollout_id"])
        previous_rank = (
            -previous["u_mean"], int(previous["chunk_idx"]),
            previous["rollout_id"]) if previous is not None else None
        if previous_rank is None or candidate_rank < previous_rank:
            episode_best[identity] = candidate

    result = {"development": [], "test": []}
    for benchmark in sorted(set(development_targets) | set(test_targets)):
        pool = [row for row in episode_best.values() if row["benchmark"] == benchmark]
        pool.sort(key=lambda row: hashlib.sha256(
            f"{seed}|test|{row['rollout_id']}|{row['chunk_idx']}".encode()
        ).hexdigest())
        n_test = test_targets.get(benchmark, 0)
        if allow_shortfall:
            n_test = min(n_test, len(pool))
        test = pool[:n_test]
        if len(test) != n_test:
            raise ValueError(f"{benchmark}: requested {n_test} test states, found {len(test)}")
        result["test"].extend(test)

        used = {
            (row["benchmark"], row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
            for row in test
        }
        available = [row for row in pool if (
            row["benchmark"], row["suite"], int(row["task_idx"]), int(row["episode_idx"])
        ) not in used]
        target = development_targets.get(benchmark, 0)
        if allow_shortfall:
            target = min(target, len(available))
        failure_target = round(target * development_failure_fraction)
        failures = [row for row in available if not bool(row["success"])]
        successes = [row for row in available if bool(row["success"])]
        for label, rows in (("failure", failures), ("success", successes)):
            rows.sort(key=lambda row: (
                -row["u_mean"],
                hashlib.sha256(
                    f"{seed}|development|{label}|{row['rollout_id']}|"
                    f"{row['chunk_idx']}".encode()).hexdigest(),
            ))
        development = failures[:failure_target]
        development.extend(successes[:target - len(development)])
        if len(development) < target:
            development.extend(
                failures[failure_target:failure_target + target - len(development)])
        if len(development) != target:
            raise ValueError(
                f"{benchmark}: requested {target} development states, found {len(development)}")
        result["development"].extend(development)
    return result


class _ContextCapture:
    """Run the hooked vanilla Euler loop once and retain its observation embedding."""
    # The hooked loop is used only to expose obs_enc. Returning its separately
    # reconstructed action would make candidate zero subtly different from the
    # stock policy; the sampler's non-invasive path returns the saved original
    # sampler result instead.
    invasive = False

    def __init__(self):
        self.obs_enc = None

    def selected(self, step, s):
        return False

    def finish(self, ctx):
        self.obs_enc = ctx.obs_enc.detach().float().cpu().numpy()


def predict_clean_chunk(policy, batch, noise, *, capture_context=False):
    if not capture_context:
        with torch.no_grad():
            return policy.predict_action_chunk(batch, noise=noise), None
    tap = _ContextCapture()
    # The hooked sampler temporarily requires eager attention. It mutates this
    # config internally, so restore it here; otherwise every policy call after
    # the first captured chunk uses a different attention backend.
    language_config = (
        policy.model.paligemma_with_expert.paligemma.model.language_model.config)
    previous_attention = language_config._attn_implementation
    try:
        with _temp_strategy(policy.model, tap), torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
    finally:
        language_config._attn_implementation = previous_attention
    return chunk, tap.obs_enc


def postprocess_chunk(chunk, postprocess, device):
    """Convert a policy-space clean chunk to the environment coordinates used by the verifier."""
    result = []
    for action in np.asarray(chunk):
        value = postprocess(torch.as_tensor(action, device=device).unsqueeze(0))
        if isinstance(value, torch.Tensor):
            value = value.squeeze(0).detach().cpu().numpy()
        result.append(np.asarray(value).reshape(-1)[:7])
    return np.asarray(result, dtype=np.float32)


def _run_continuation(env, obs, ep, policy, preprocess, postprocess, device, *,
                      prefix, branch_seed, steps_already, n_action_steps=None,
                      skip_unused_renders: bool = False, render_lead: int = 2):
    success = False
    steps = steps_already
    chunk_stride = int(n_action_steps or policy.config.chunk_size)
    est_chunks = max(1, round(ep["max_steps"] / chunk_stride))
    lead = max(1, int(render_lead))
    skipping = bool(skip_unused_renders and set_camera_observables(env, True))

    def _render_next(needed: bool) -> None:
        if skipping:
            set_camera_observables(env, needed)

    prefix = list(prefix)
    try:
        for index, action in enumerate(prefix):
            # The observation after the prefix is the first one consumed by the
            # continuation policy. Warm both cameras for the validated lead.
            _render_next(len(prefix) - index <= lead)
            obs, _, done, _ = env.step(action)
            steps += 1
            if env.check_success():
                return True, steps
            if done or steps >= ep["max_steps"]:
                return False, steps
        queue, replan = [], 0
        while steps < ep["max_steps"]:
            if not queue:
                # Match run_episode's time conditioning. Derive it from the
                # environment step count so every independently restored branch is
                # insensitive to whatever chunk_pos a previous branch left behind.
                chunk_index = steps // chunk_stride
                policy.model._pnp.chunk_pos = min(chunk_index / est_chunks, 1.0)
                batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
                noise = _draw_chunk_noise(
                    policy, device, chunk_noise_seed(branch_seed, replan))
                chunk, _ = predict_clean_chunk(policy, batch, noise)
                # Closed-loop continuation: execute only the first n_action_steps of each
                # generated chunk before replanning (mirrors run_episode).
                rows = list(chunk.squeeze(0).detach().cpu().numpy())
                queue = rows if n_action_steps is None else rows[:int(n_action_steps)]
                replan += 1
            # This is equivalent to run_episode's `len(queue) < lead` check after pop().
            _render_next(len(queue) <= lead)
            action = queue.pop(0)
            action = postprocess_chunk(np.asarray(action)[None], postprocess, device)[0]
            obs, _, done, _ = env.step(action)
            steps += 1
            if env.check_success():
                success = True
                break
            if done:
                break
        return success, steps
    finally:
        if skipping:
            set_camera_observables(env, True)


def _reset_and_replay_actions(env, ep, policy, replay_actions, *,
                              skip_unused_renders: bool = False,
                              render_lead: int = 2):
    """Reach a branch state through fixed actions and report transient terminal flags.

    A restored replay can cross the task-success predicate slightly earlier than
    the source run because independent MuJoCo replays are not bit-exact. The
    selected root is defined by the full stored action count, and callers correct
    every branch to one canonical root state afterward, so these flags are
    diagnostics rather than reasons to truncate parent restoration.
    """
    replay_actions = list(replay_actions)
    terminal_events = []
    lead = max(1, int(render_lead))
    skipping = bool(skip_unused_renders and set_camera_observables(env, True))

    def _render_next(needed: bool) -> None:
        if skipping:
            set_camera_observables(env, needed)

    try:
        env.reset(); policy.reset()
        obs = env.set_init_state(ep["init_state"])
        for wait_step in range(NUM_STEPS_WAIT):
            # If there is no parent replay, the final settling observation is
            # consumed immediately. Otherwise the final parent actions provide
            # the camera warm-up for the root observation.
            _render_next(
                not replay_actions and wait_step >= NUM_STEPS_WAIT - lead)
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        for index, action in enumerate(replay_actions):
            _render_next(len(replay_actions) - index <= lead)
            obs, _, done, _ = env.step(action)
            reported_success = bool(env.check_success())
            if reported_success or done:
                terminal_events.append({
                    "replay_action_index": int(index),
                    "success": reported_success,
                    "done": bool(done),
                })
        return obs, terminal_events
    finally:
        if skipping:
            set_camera_observables(env, True)


def collect_replay_candidate_group(env, ep, policy, preprocess, postprocess, device, *,
                                   chunk_idx: int, uncertainty_stratum: str,
                                   prefix_length: int = 10, candidate_count: int = 4,
                                   experiment: str = "verifier-clean-pairs-v3",
                                   state_atol: float = 1e-6,
                                   trajectory_seed: int | None = None,
                                   collection_split: str = "development",
                                   manifest_hash: str = "",
                                   model_revision: str = "",
                                   n_action_steps: int | None = None,
                                   candidate_num_inference_steps: int | None = None,
                                   replay_actions_override: np.ndarray | None = None,
                                   source_sim_state_override: np.ndarray | None = None,
                                   source_policy_observation_override: dict | None = None,
                                   skip_unused_renders: bool = False,
                                   render_lead: int = 2):
    """Collect candidates at a mid-rollout state by deterministic action replay.

    Unlike simulator snapshots, replay reconstructs wrapper state, contacts, and
    observations through normal resets and steps. Candidate generation happens
    once at the canonical replay state; each outcome branch independently
    replays the exact same environment-space action prefix before intervention.
    """
    seed = (int(trajectory_seed) if trajectory_seed is not None else
            episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0))))
    replay_actions = []
    steps = 0
    chunk_stride = int(n_action_steps or policy.config.chunk_size)
    est_chunks = max(1, round(ep["max_steps"] / chunk_stride))
    if replay_actions_override is not None:
        supplied = np.asarray(replay_actions_override, dtype=np.float32)
        expected = int(chunk_idx) * chunk_stride
        if supplied.ndim != 2 or supplied.shape != (expected, 7):
            raise ValueError(
                f"stored parent replay must have shape ({expected}, 7), got {supplied.shape}")
        replay_source = "stored_source_trajectory"
        replay_actions = [action.copy() for action in supplied]
        obs, canonical_parent_terminal_events = _reset_and_replay_actions(
            env, ep, policy, replay_actions,
            skip_unused_renders=skip_unused_renders, render_lead=render_lead)
        steps = len(replay_actions)
    else:
        obs, canonical_parent_terminal_events = _reset_and_replay_actions(
            env, ep, policy, [], skip_unused_renders=skip_unused_renders,
            render_lead=render_lead)
        replay_source = "regenerated_policy"
        for ci in range(chunk_idx):
            # Do not inherit time conditioning from a previous tree/branch. This
            # is the same boundary position used by run_episode for a 10-action
            # receding-horizon rollout.
            policy.model._pnp.chunk_pos = min(ci / est_chunks, 1.0)
            batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
            noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, ci))
            chunk, _ = predict_clean_chunk(policy, batch, noise)
            env_chunk = postprocess_chunk(
                chunk.squeeze(0).detach().cpu().numpy(), postprocess, device)
            replay_chunk = (env_chunk if n_action_steps is None
                            else env_chunk[:int(n_action_steps)])
            for action in replay_chunk:
                obs, _, done, _ = env.step(action)
                replay_actions.append(action.copy()); steps += 1
                if env.check_success() or done:
                    return None

    _, canonical_sim = _unwrap_sim(env)
    replay_root_state = np.asarray(canonical_sim.get_state().flatten()).copy()
    source_state_set_error = None
    if source_sim_state_override is not None:
        source_state = np.asarray(source_sim_state_override).copy()
        setter = getattr(canonical_sim, "set_state_from_flattened", None)
        if not callable(setter):
            raise RuntimeError("MuJoCo simulator has no set_state_from_flattened")
        setter(source_state)
        canonical_sim.forward()
        restored = np.asarray(canonical_sim.get_state().flatten()).copy()
        if restored.shape != source_state.shape:
            raise ValueError(
                f"persisted source state shape {source_state.shape} does not match "
                f"simulator state {restored.shape}")
        source_state_set_error = float(np.max(np.abs(restored - source_state)))
        if source_state_set_error > state_atol:
            raise RuntimeError(
                "persisted source simulator-state restoration failed "
                f"(max_abs={source_state_set_error:.3g})")
    canonical_sim_state = copy.deepcopy(canonical_sim.get_state())
    canonical_state = np.asarray(canonical_sim_state.flatten()).copy()
    policy.model._pnp.chunk_pos = min(chunk_idx / est_chunks, 1.0)
    policy_observation = (
        copy.deepcopy(source_policy_observation_override)
        if source_policy_observation_override is not None
        else obs_to_policy(obs, ep["task_desc"]))
    batch = preprocess(policy_observation)
    root_policy_input_digest = _content_digest(batch)
    policy_chunks = {}
    obs_enc = None
    previous_num_steps = policy.model._pnp.num_steps
    candidate_steps = {}
    candidate_noise_digests = {}
    try:
        for index in range(candidate_count):
            kind = "default" if index == 0 else f"fresh_noise_{index}"
            requested_steps = (previous_num_steps if index == 0
                               else candidate_num_inference_steps)
            policy.model._pnp.num_steps = requested_steps
            effective_steps = int(
                requested_steps or getattr(policy.config, "num_inference_steps", 10))
            noise_index = chunk_idx if index == 0 else chunk_idx * 1000 + index
            noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, noise_index))
            candidate_noise_digests[kind] = _content_digest(noise)
            chunk, captured = predict_clean_chunk(
                policy, batch, noise, capture_context=(index == 0))
            if captured is not None:
                obs_enc = captured
            policy_chunks[kind] = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
            candidate_steps[kind] = effective_steps
    finally:
        policy.model._pnp.num_steps = previous_num_steps
    env_chunks = {kind: postprocess_chunk(chunk, postprocess, device)
                  for kind, chunk in policy_chunks.items()}

    group_id = candidate_group_id(
        ep.get("benchmark", "libero"), ep["suite"], ep["task_idx"],
        ep.get("ep_idx", ep.get("episode_idx", 0)), chunk_idx, namespace=experiment,
        trajectory_seed=trajectory_seed)
    candidates, replay_state_errors, post_correction_errors = [], [], []
    branch_parent_terminal_events = []
    for kind in policy_chunks:
        branch_obs, replay_terminal_events = _reset_and_replay_actions(
            env, ep, policy, replay_actions,
            skip_unused_renders=skip_unused_renders, render_lead=render_lead)
        branch_parent_terminal_events.append(replay_terminal_events)
        _, branch_sim = _unwrap_sim(env)
        branch_state = np.asarray(branch_sim.get_state().flatten())
        replay_state_error = float(np.max(np.abs(branch_state - canonical_state)))
        replay_state_errors.append(replay_state_error)
        state_corrected = not np.allclose(
            branch_state, canonical_state, atol=state_atol, rtol=0)
        if state_corrected:
            # Replay reconstructs wrapper/controller state and clocks. Correct
            # only MuJoCo's physical state so all interventions start from the
            # exact same qpos/qvel/act/time state. The stale pre-prefix
            # observation is never consumed: _run_continuation steps the
            # intervention prefix before its first policy query.
            branch_sim.set_state(copy.deepcopy(canonical_sim_state))
            branch_sim.forward()
            corrected = np.asarray(branch_sim.get_state().flatten())
            if not np.allclose(corrected, canonical_state, atol=state_atol, rtol=0):
                corrected_error = float(np.max(np.abs(corrected - canonical_state)))
                raise RuntimeError(
                    f"sim state correction failed (max_abs={corrected_error:.3g})")
        final_branch_state = np.asarray(branch_sim.get_state().flatten())
        post_correction_error = float(np.max(np.abs(
            final_branch_state - canonical_state)))
        post_correction_errors.append(post_correction_error)
        success, n_steps = _run_continuation(
            env, branch_obs, ep, policy, preprocess, postprocess, device,
            prefix=env_chunks[kind][:prefix_length], branch_seed=seed ^ 0x51A7,
            steps_already=steps, n_action_steps=n_action_steps,
            skip_unused_renders=skip_unused_renders, render_lead=render_lead)
        candidate_id = hashlib.sha256(f"{group_id}|{kind}".encode()).hexdigest()[:24]
        candidates.append({
            "candidate_id": candidate_id, "candidate_kind": kind, "success": success,
            "n_steps": n_steps, "rollout_id": None,
            "metadata_json": {
                "source_episode_seed": seed, "chunk_idx": chunk_idx,
                "replay_state_max_abs_before_correction": replay_state_error,
                "replay_state_max_abs_after_correction": post_correction_error,
                "sim_state_corrected": state_corrected,
                "denoise_steps": candidate_steps[kind],
                "executed_prefix_length": int(prefix_length),
                "candidate_noise_sha256": candidate_noise_digests[kind],
                "parent_replay_reported_terminal_events": replay_terminal_events,
            },
            "blobs": {
                "policy_chunk": {"actions": policy_chunks[kind]},
                "env_chunk": {"actions": env_chunks[kind],
                              "mask": np.ones(len(env_chunks[kind]), dtype=np.bool_)},
                "observation": {"obs_enc": obs_enc},
            },
        })
    group = {
        "candidate_group_id": group_id, "experiment": experiment,
        "benchmark": ep.get("benchmark", "libero"), "suite": ep["suite"],
        "task_idx": ep["task_idx"], "episode_idx": ep.get("ep_idx", ep.get("episode_idx", 0)),
        "chunk_idx": chunk_idx, "uncertainty_stratum": uncertainty_stratum,
        "pairing_mode": "deterministic_replay", "prefix_length": prefix_length,
        "snapshot_validated": False,
        "trajectory_seed": trajectory_seed,
        "collection_split": collection_split,
        "manifest_hash": manifest_hash,
        "model_revision": model_revision,
        "metadata_json": {
                          "replay_validated": not any(
                              error > state_atol for error in replay_state_errors),
                          "exact_sim_state_validated": not any(
                              error > state_atol for error in post_correction_errors),
                          "state_initialization": (
                              "replay_plus_sim_correction" if any(
                                  error > state_atol for error in replay_state_errors)
                              else "exact_replay"),
                          "sim_state_corrected": any(
                              error > state_atol for error in replay_state_errors),
                          "replay_state_max_abs_before_correction": max(
                              replay_state_errors, default=0.0),
                          "replay_state_max_abs_after_correction": max(
                              post_correction_errors, default=0.0),
                          "replay_action_count": len(replay_actions),
                          "parent_replay_source": replay_source,
                          "canonical_root_source": (
                              "persisted_source_artifact"
                              if source_sim_state_override is not None
                              else "fresh_action_replay"),
                          "policy_input_source": (
                              "persisted_source_artifact"
                              if source_policy_observation_override is not None
                              else "fresh_action_replay"),
                          "replay_root_max_abs_vs_canonical": float(np.max(np.abs(
                              replay_root_state - canonical_state))),
                          "persisted_source_state_set_max_abs": source_state_set_error,
                          "parent_replay_actions_sha256": _content_digest(
                              np.asarray(replay_actions, dtype=np.float32)),
                          "canonical_parent_replay_reported_terminal_events":
                              canonical_parent_terminal_events,
                          "branch_parent_replay_terminal_event_count": sum(
                              len(events) for events in branch_parent_terminal_events),
                          "root_sim_state_sha256": _content_digest(canonical_state),
                          "root_policy_input_sha256": root_policy_input_digest,
                          "chunk_position": float(policy.model._pnp.chunk_pos),
                          "trajectory_seed": trajectory_seed,
                          "collection_split": collection_split,
                          "collection_manifest_hash": manifest_hash,
                          "model_revision": model_revision,
                          "candidate_count": candidate_count,
                          "default_candidate_denoise_steps": candidate_steps.get("default"),
                          "alternative_candidate_denoise_steps": candidate_num_inference_steps,
                          "parent_replay_n_action_steps": n_action_steps,
                          "continuation_n_action_steps": n_action_steps,
                          "skip_unused_renders": bool(skip_unused_renders),
                          "render_lead": int(render_lead)},
    }
    return group, candidates
