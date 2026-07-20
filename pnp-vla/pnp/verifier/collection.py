"""Primitives for deterministic same-state candidate collection in LIBERO."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import numpy as np
import torch


def _unwrap_sim(env):
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        sim = getattr(current, "sim", None)
        if sim is not None:
            return current, sim
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    raise TypeError("environment does not expose a MuJoCo sim")


def _observations(owner):
    getter = getattr(owner, "_get_observations", None)
    if getter is None:
        raise TypeError("environment does not expose _get_observations after restoration")
    try:
        return getter(force_update=True)
    except TypeError:
        return getter()


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
    owner, sim = _unwrap_sim(env)
    sim.set_state_from_flattened(snapshot.sim_state.copy())
    sim.forward()
    np.random.set_state(snapshot.numpy_state)
    random.setstate(snapshot.python_state)
    torch.random.set_rng_state(snapshot.torch_state)
    if snapshot.cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.cuda_states)
    return _observations(owner)


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
