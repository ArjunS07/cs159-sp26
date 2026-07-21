"""Primitives for deterministic same-state candidate collection in LIBERO."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import random

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


def _observations(env):
    # OffScreenRenderEnv may expose ``sim`` on its outer wrapper while the
    # robosuite observation getter lives on the wrapped task environment.
    for owner in _env_chain(env):
        getter = getattr(owner, "_get_observations", None)
        if getter is None:
            continue
        try:
            return getter(force_update=True)
        except TypeError:
            return getter()
    raise TypeError("environment does not expose _get_observations after restoration")


@dataclass
class SimulatorSnapshot:
    sim_state: np.ndarray
    numpy_state: tuple
    python_state: object
    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor] | None


def capture_snapshot(env) -> SimulatorSnapshot:
    _, sim = _unwrap_sim(env)
    return SimulatorSnapshot(
        sim_state=np.asarray(sim.get_state().flatten()).copy(),
        numpy_state=np.random.get_state(), python_state=random.getstate(),
        torch_state=torch.random.get_rng_state(),
        cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_snapshot(env, snapshot: SimulatorSnapshot):
    _, sim = _unwrap_sim(env)
    sim.set_state_from_flattened(snapshot.sim_state.copy())
    sim.forward()
    np.random.set_state(snapshot.numpy_state)
    random.setstate(snapshot.python_state)
    torch.random.set_rng_state(snapshot.torch_state)
    if snapshot.cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.cuda_states)
    return _observations(env)


def _obs_arrays(obs):
    return {key: np.asarray(value).copy() for key, value in obs.items()
            if isinstance(value, (np.ndarray, list, tuple))}


def validate_snapshot_replay(env, actions, *, state_atol=1e-7, obs_atol=1e-6):
    """Replay an action prefix twice and verify state, observations, and success agree."""
    snapshot = capture_snapshot(env)
    runs = []
    for _ in range(2):
        restore_snapshot(env, snapshot)
        final_obs, done = None, False
        for action in np.asarray(actions):
            final_obs, _, done, _ = env.step(action)
            if done:
                break
        _, sim = _unwrap_sim(env)
        runs.append((np.asarray(sim.get_state().flatten()).copy(), _obs_arrays(final_obs),
                     bool(env.check_success()), bool(done)))
    restore_snapshot(env, snapshot)
    state_ok = np.allclose(runs[0][0], runs[1][0], atol=state_atol, rtol=0)
    common = set(runs[0][1]) & set(runs[1][1])
    obs_ok = all(np.allclose(runs[0][1][key], runs[1][1][key], atol=obs_atol, rtol=0)
                 for key in common)
    return {"deterministic": bool(state_ok and obs_ok and runs[0][2:] == runs[1][2:]),
            "state_match": bool(state_ok), "observation_match": bool(obs_ok),
            "success_match": runs[0][2] == runs[1][2], "done_match": runs[0][3] == runs[1][3]}


def candidate_group_id(benchmark, suite, task_idx, episode_idx, chunk_idx) -> str:
    raw = f"{benchmark}|{suite}|{task_idx}|{episode_idx}|{chunk_idx}".encode()
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
        used_rollouts = set()
        for stratum in ("low", "mid", "high"):
            stratum_pool = [row for row in pool if row["uncertainty_stratum"] == stratum]
            stratum_pool.sort(key=lambda row: hashlib.sha256(
                f"{seed}|{row['rollout_id']}|{row['chunk_idx']}".encode()).hexdigest())
            for row in stratum_pool:
                if row["rollout_id"] in used_rollouts:
                    continue
                selected.append(row); used_rollouts.add(row["rollout_id"])
                if sum(r["benchmark"] == benchmark and r["uncertainty_stratum"] == stratum
                       for r in selected) >= per[stratum]:
                    break
    return selected


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


def collect_candidate_pair(env, ep, policy, preprocess, postprocess, device, *,
                           chunk_idx: int, uncertainty_stratum: str, prefix_length: int = 10,
                           validate_snapshot: bool = True):
    """Collect default-vs-fresh outcomes from one replayed mid-episode state.

    Returns ``(group_row, candidate_rows)`` ready for ``SupabaseStore.register_candidate_group``.
    If the vanilla replay succeeds before ``chunk_idx``, returns ``None``.
    """
    env.reset(); policy.reset()
    obs = env.set_init_state(ep["init_state"])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    seed = episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
    chunk_size = policy.config.chunk_size
    steps = 0
    for ci in range(chunk_idx):
        batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
        noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, ci))
        chunk, _ = predict_clean_chunk(policy, batch, noise)
        env_chunk = postprocess_chunk(chunk.squeeze(0).detach().cpu().numpy(), postprocess, device)
        for action in env_chunk:
            obs, _, done, _ = env.step(action); steps += 1
            if env.check_success() or done:
                return None

    est_chunks = max(1, round(ep["max_steps"] / chunk_size))
    policy.model._pnp.chunk_pos = min(chunk_idx / est_chunks, 1.0)
    batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
    default_noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, chunk_idx))
    fresh_noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, chunk_idx * 1000 + 1))
    default, obs_enc = predict_clean_chunk(policy, batch, default_noise, capture_context=True)
    fresh, _ = predict_clean_chunk(policy, batch, fresh_noise)
    policy_chunks = {
        "default": default.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "fresh_noise": fresh.squeeze(0).detach().cpu().numpy().astype(np.float32),
    }
    env_chunks = {kind: postprocess_chunk(chunk, postprocess, device)
                  for kind, chunk in policy_chunks.items()}
    snapshot = capture_snapshot(env)
    replay = (validate_snapshot_replay(env, env_chunks["default"][:prefix_length])
              if validate_snapshot else {"deterministic": False})
    if validate_snapshot and not replay["deterministic"]:
        raise RuntimeError(f"snapshot replay is not deterministic: {replay}")
    group_id = candidate_group_id(ep.get("benchmark", "libero"), ep["suite"], ep["task_idx"],
                                  ep.get("ep_idx", ep.get("episode_idx", 0)), chunk_idx)
    candidates = []
    for offset, kind in enumerate(("default", "fresh_noise")):
        restored_obs = restore_snapshot(env, snapshot)
        success, n_steps = _run_continuation(
            env, restored_obs, ep, policy, preprocess, postprocess, device,
            prefix=env_chunks[kind][:prefix_length], branch_seed=seed ^ 0x51A7,
            steps_already=steps)
        candidate_id = hashlib.sha256(f"{group_id}|{kind}".encode()).hexdigest()[:24]
        candidates.append({
            "candidate_id": candidate_id, "candidate_kind": kind, "success": success,
            "n_steps": n_steps, "rollout_id": None,
            "metadata_json": {"source_episode_seed": seed, "chunk_idx": chunk_idx},
            "blobs": {
                "policy_chunk": {"actions": policy_chunks[kind]},
                "env_chunk": {"actions": env_chunks[kind],
                              "mask": np.ones(len(env_chunks[kind]), dtype=np.bool_)},
                "observation": {"obs_enc": obs_enc},
            },
        })
    group = {
        "candidate_group_id": group_id, "experiment": "verifier-clean-pairs-v1",
        "benchmark": ep.get("benchmark", "libero"), "suite": ep["suite"],
        "task_idx": ep["task_idx"], "episode_idx": ep.get("ep_idx", ep.get("episode_idx", 0)),
        "chunk_idx": chunk_idx, "uncertainty_stratum": uncertainty_stratum,
        "pairing_mode": "snapshot", "prefix_length": prefix_length,
        "snapshot_validated": bool(replay["deterministic"]),
        "metadata_json": {**replay, "chunk_position": float(policy.model._pnp.chunk_pos)},
    }
    restore_snapshot(env, snapshot)
    return group, candidates


def collect_initial_pair_fallback(env, ep, policy, preprocess, postprocess, device, *,
                                  uncertainty_stratum: str, prefix_length: int = 10,
                                  source_chunk_idx: int = 0):
    """Fallback paired full episodes: alternatives share only the initial simulator state."""
    seed = episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
    group_id = candidate_group_id(ep.get("benchmark", "libero"), ep["suite"], ep["task_idx"],
                                  ep.get("ep_idx", ep.get("episode_idx", 0)), source_chunk_idx)
    candidates, obs_enc = [], None
    for offset, kind in enumerate(("default", "fresh_noise")):
        env.reset(); policy.reset()
        obs = env.set_init_state(ep["init_state"])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
        noise_index = 0 if kind == "default" else 1
        noise = _draw_chunk_noise(policy, device, chunk_noise_seed(seed, noise_index))
        chunk, captured = predict_clean_chunk(policy, batch, noise, capture_context=(offset == 0))
        if captured is not None:
            obs_enc = captured
        policy_chunk = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
        env_chunk = postprocess_chunk(policy_chunk, postprocess, device)
        success, n_steps = _run_continuation(
            env, obs, ep, policy, preprocess, postprocess, device,
            prefix=env_chunk[:prefix_length], branch_seed=seed ^ 0x51A7, steps_already=0)
        candidate_id = hashlib.sha256(f"{group_id}|{kind}".encode()).hexdigest()[:24]
        candidates.append({
            "candidate_id": candidate_id, "candidate_kind": kind, "success": success,
            "n_steps": n_steps, "rollout_id": None,
            "metadata_json": {"source_episode_seed": seed, "chunk_idx": 0},
            "blobs": {"policy_chunk": {"actions": policy_chunk},
                      "env_chunk": {"actions": env_chunk,
                                    "mask": np.ones(len(env_chunk), dtype=np.bool_)},
                      "observation": {"obs_enc": obs_enc}},
        })
    group = {
        "candidate_group_id": group_id, "experiment": "verifier-clean-pairs-v1",
        "benchmark": ep.get("benchmark", "libero"), "suite": ep["suite"],
        "task_idx": ep["task_idx"], "episode_idx": ep.get("ep_idx", ep.get("episode_idx", 0)),
        "chunk_idx": 0, "uncertainty_stratum": uncertainty_stratum,
        "pairing_mode": "paired_full_episode", "prefix_length": prefix_length,
        "snapshot_validated": False,
        "metadata_json": {"fallback": "snapshot_nondeterministic",
                          "source_chunk_idx": source_chunk_idx, "chunk_position": 0.0},
    }
    return group, candidates
