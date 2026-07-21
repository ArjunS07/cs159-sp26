import numpy as np

from pnp.verifier.collection import (
    build_stratified_manifest, candidate_group_id, capture_snapshot, restore_snapshot,
    validate_snapshot_replay,
)


class _State:
    def __init__(self, value): self.value = np.asarray(value)
    def flatten(self): return self.value.copy()


class _Sim:
    def __init__(self): self.value = np.zeros(2)
    def get_state(self): return _State(self.value)
    def set_state_from_flattened(self, value): self.value = np.asarray(value).copy()
    def forward(self): pass


class _Env:
    def __init__(self): self.sim = _Sim()
    def _get_observations(self, force_update=True): return {"state": self.sim.value.copy()}
    def step(self, action):
        self.sim.value += np.asarray(action)[:2]
        return self._get_observations(), 0, False, {}
    def check_success(self): return bool(self.sim.value.sum() > 100)


class _OuterEnv:
    """Match LIBERO's split wrapper API: sim outside, observations inside."""
    def __init__(self):
        self.env = _Env()
        self.sim = self.env.sim


def test_snapshot_restore_and_determinism_check():
    env = _Env()
    snapshot = capture_snapshot(env)
    env.step([1, 2])
    obs = restore_snapshot(env, snapshot)
    assert np.allclose(obs["state"], 0)
    report = validate_snapshot_replay(env, [[1, 2], [3, 4]])
    assert report["deterministic"]


def test_snapshot_restore_finds_observations_below_sim_wrapper():
    env = _OuterEnv()
    snapshot = capture_snapshot(env)
    env.sim.value[:] = 9
    obs = restore_snapshot(env, snapshot)
    assert np.array_equal(obs["state"], snapshot.sim_state)


def test_candidate_group_id_is_stable_and_identity_sensitive():
    a = candidate_group_id("libero", "s", 1, 2, 3)
    assert a == candidate_group_id("libero", "s", 1, 2, 3)
    assert a != candidate_group_id("libero", "s", 1, 2, 4)


def test_manifest_uses_mixed_tasks_and_one_state_per_rollout():
    rollouts, steps = [], []
    for task in range(2):
        for episode in range(10):
            rid = f"r-{task}-{episode}"
            rollouts.append({"rollout_id": rid, "benchmark": "libero", "suite": "s",
                             "task_idx": task, "episode_idx": episode,
                             "success": episode < 5 if task == 0 else True})
            for chunk in range(2):
                steps.append({"rollout_id": rid, "chunk_idx": chunk,
                              "u_mean": episode + chunk / 10})
    manifest = build_stratified_manifest(rollouts, steps, targets={"libero": 6})
    assert len(manifest) == 6
    assert {row["task_idx"] for row in manifest} == {0}
    assert len({row["rollout_id"] for row in manifest}) == 6
    assert {row["uncertainty_stratum"] for row in manifest} == {"low", "mid", "high"}
