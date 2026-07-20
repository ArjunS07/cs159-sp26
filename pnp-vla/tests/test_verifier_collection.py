import numpy as np

from pnp.verifier.collection import (
    candidate_group_id, capture_snapshot, restore_snapshot, validate_snapshot_replay,
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


def test_snapshot_restore_and_determinism_check():
    env = _Env()
    snapshot = capture_snapshot(env)
    env.step([1, 2])
    obs = restore_snapshot(env, snapshot)
    assert np.allclose(obs["state"], 0)
    report = validate_snapshot_replay(env, [[1, 2], [3, 4]])
    assert report["deterministic"]


def test_candidate_group_id_is_stable_and_identity_sensitive():
    a = candidate_group_id("libero", "s", 1, 2, 3)
    assert a == candidate_group_id("libero", "s", 1, 2, 3)
    assert a != candidate_group_id("libero", "s", 1, 2, 4)
