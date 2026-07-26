"""Primitives for deterministic same-state candidate collection in LIBERO."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json

import numpy as np
import torch

from ..config import LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT
from ..libero_env import obs_to_policy
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


def candidate_group_id(benchmark, suite, task_idx, episode_idx, chunk_idx, *, namespace="") -> str:
    identity = f"{benchmark}|{suite}|{task_idx}|{episode_idx}|{chunk_idx}"
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
    )
    canonical = [
        {field: row.get(field) for field in fields}
        for row in sorted(rows, key=lambda row: (
            row["benchmark"], row["suite"], int(row["task_idx"]),
            int(row["episode_idx"]), int(row["chunk_idx"])))
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_targeted_manifests(
        rollout_rows, euler_rows, excluded_identities, *,
        development_targets=None, test_targets=None,
        development_failure_fraction=.70, seed=42):
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
    invasive = True

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
    with _temp_strategy(policy.model, tap), torch.no_grad():
        chunk = policy.predict_action_chunk(batch, noise=noise)
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
                      prefix, branch_seed, steps_already):
    success = False
    steps = steps_already
    for action in prefix:
        obs, _, done, _ = env.step(action)
        steps += 1
        if env.check_success():
            return True, steps
        if done or steps >= ep["max_steps"]:
            return False, steps
    queue, replan = [], 0
    while steps < ep["max_steps"]:
        if not queue:
            batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
            noise = _draw_chunk_noise(policy, device, chunk_noise_seed(branch_seed, replan))
            chunk, _ = predict_clean_chunk(policy, batch, noise)
            queue = list(chunk.squeeze(0).detach().cpu().numpy())
            replan += 1
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


def _reset_and_replay_actions(env, ep, policy, replay_actions):
    """Reach a branch state through the public environment API using fixed actions."""
    env.reset(); policy.reset()
    obs = env.set_init_state(ep["init_state"])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    for action in replay_actions:
        obs, _, done, _ = env.step(action)
        if env.check_success() or done:
            return None
    return obs


def collect_replay_candidate_group(env, ep, policy, preprocess, postprocess, device, *,
                                   chunk_idx: int, uncertainty_stratum: str,
                                   prefix_length: int = 10, candidate_count: int = 4,
                                   experiment: str = "verifier-clean-pairs-v3",
                                   state_atol: float = 1e-6):
    """Collect candidates at a mid-rollout state by deterministic action replay.

    Unlike simulator snapshots, replay reconstructs wrapper state, contacts, and
    observations through normal resets and steps. Candidate generation happens
    once at the canonical replay state; each outcome branch independently
    replays the exact same environment-space action prefix before intervention.
    """
    seed = episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
    obs = _reset_and_replay_actions(env, ep, policy, [])
    replay_actions = []
    steps = 0
    for ci in range(chunk_idx):
        batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
        noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, ci))
        chunk, _ = predict_clean_chunk(policy, batch, noise)
        env_chunk = postprocess_chunk(chunk.squeeze(0).detach().cpu().numpy(), postprocess, device)
        for action in env_chunk:
            obs, _, done, _ = env.step(action)
            replay_actions.append(action.copy()); steps += 1
            if env.check_success() or done:
                return None

    _, canonical_sim = _unwrap_sim(env)
    canonical_sim_state = copy.deepcopy(canonical_sim.get_state())
    canonical_state = np.asarray(canonical_sim_state.flatten()).copy()
    chunk_size = policy.config.chunk_size
    est_chunks = max(1, round(ep["max_steps"] / chunk_size))
    policy.model._pnp.chunk_pos = min(chunk_idx / est_chunks, 1.0)
    batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
    policy_chunks = {}
    obs_enc = None
    for index in range(candidate_count):
        kind = "default" if index == 0 else f"fresh_noise_{index}"
        noise_index = chunk_idx if index == 0 else chunk_idx * 1000 + index
        noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, noise_index))
        chunk, captured = predict_clean_chunk(
            policy, batch, noise, capture_context=(index == 0))
        if captured is not None:
            obs_enc = captured
        policy_chunks[kind] = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
    env_chunks = {kind: postprocess_chunk(chunk, postprocess, device)
                  for kind, chunk in policy_chunks.items()}

    group_id = candidate_group_id(
        ep.get("benchmark", "libero"), ep["suite"], ep["task_idx"],
        ep.get("ep_idx", ep.get("episode_idx", 0)), chunk_idx, namespace=experiment)
    candidates, replay_state_errors = [], []
    for kind in policy_chunks:
        branch_obs = _reset_and_replay_actions(env, ep, policy, replay_actions)
        if branch_obs is None:
            raise RuntimeError("fixed replay terminated before the branch state")
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
        success, n_steps = _run_continuation(
            env, branch_obs, ep, policy, preprocess, postprocess, device,
            prefix=env_chunks[kind][:prefix_length], branch_seed=seed ^ 0x51A7,
            steps_already=steps)
        candidate_id = hashlib.sha256(f"{group_id}|{kind}".encode()).hexdigest()[:24]
        candidates.append({
            "candidate_id": candidate_id, "candidate_kind": kind, "success": success,
            "n_steps": n_steps, "rollout_id": None,
            "metadata_json": {
                "source_episode_seed": seed, "chunk_idx": chunk_idx,
                "replay_state_max_abs_before_correction": replay_state_error,
                "sim_state_corrected": state_corrected,
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
        "metadata_json": {
                          "replay_validated": not any(
                              error > state_atol for error in replay_state_errors),
                          "exact_sim_state_validated": True,
                          "state_initialization": (
                              "replay_plus_sim_correction" if any(
                                  error > state_atol for error in replay_state_errors)
                              else "exact_replay"),
                          "sim_state_corrected": any(
                              error > state_atol for error in replay_state_errors),
                          "replay_state_max_abs_before_correction": max(
                              replay_state_errors, default=0.0),
                          "replay_action_count": len(replay_actions),
                          "chunk_position": float(policy.model._pnp.chunk_pos)},
    }
    return group, candidates
