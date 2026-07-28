from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pnp.config import RolloutConfig
from pnp.rollout import run_episode, run_episode_batch


def _observation():
    return {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.zeros(4),
        "robot0_gripper_qpos": np.zeros(2),
    }


class _Env:
    def __init__(self):
        self.actions = []

    def reset(self):
        return _observation()

    def set_init_state(self, _state):
        return _observation()

    def step(self, action):
        self.actions.append(np.asarray(action))
        return _observation(), 0.0, False, {}

    def check_success(self):
        return bool(self.actions and np.allclose(self.actions[-1], 2.0))


class _Policy:
    def __init__(self):
        self.model = SimpleNamespace(_pnp=SimpleNamespace(
            action_dim=7, num_steps=None, vf_evals=0, strategy=None, chunk_pos=0.0,
        ))
        self.config = SimpleNamespace(
            chunk_size=2, max_action_dim=7, num_inference_steps=10,
        )

    def reset(self):
        pass

    def predict_action_chunk(self, _batch, noise=None):
        return torch.ones((1, 2, 7))


class _BatchPolicy(_Policy):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def predict_action_chunk(self, _batch, noise=None):
        self.batch_sizes.append(noise.shape[0])
        # Lane output is determined only by its explicit initial noise.
        value = noise[:, :1, :1].expand(-1, 2, 7)
        return value


class _TimedEnv(_Env):
    def __init__(self, success_after):
        super().__init__()
        self.success_after = success_after

    def check_success(self):
        return len(self.actions) >= 10 + self.success_after


def test_run_episode_postprocesses_actions_before_environment_step():
    env = _Env()
    ep = {
        "task_desc": "test task", "max_steps": 1,
        "init_state": np.zeros(3), "ep_idx": 0,
        "suite": "test", "task_idx": 0,
    }

    with patch("pnp.rollout.obs_to_policy", return_value={}):
        result = run_episode(
            env, ep, _Policy(), lambda obs: obs, lambda action: action * 2,
            torch.device("cpu"), RolloutConfig(),
        )

    assert result["success"] is True
    assert result["terminated_reason"] == "success"
    assert result["started_at"] <= result["finished_at"]
    assert result["perturb_seed"] == result["episode_seed"]
    assert result["inference_ms_total"] >= 0.0
    assert np.allclose(env.actions[-1], 2.0)
    assert np.allclose(result["trajectory"]["actions"][0], 2.0)


def test_run_episode_can_save_exact_policy_space_chunks():
    env = _Env()
    ep = {
        "task_desc": "test task", "max_steps": 1,
        "init_state": np.zeros(3), "ep_idx": 0,
        "suite": "test", "task_idx": 0,
    }
    with patch("pnp.rollout.obs_to_policy", return_value={}):
        result = run_episode(
            env, ep, _Policy(), lambda obs: obs, lambda action: action * 2,
            torch.device("cpu"), RolloutConfig(save_generated_chunks=True),
        )
    assert result["generated_chunks"]["chunks"].shape == (1, 2, 7)
    assert np.allclose(result["generated_chunks"]["chunks"], 1.0)
    assert np.allclose(result["trajectory"]["actions"], 2.0)


def test_run_episode_batch_preserves_order_and_shrinks_active_lanes():
    envs = [_TimedEnv(1), _TimedEnv(3)]
    episodes = [{
        "task_desc": "test task", "max_steps": 4, "init_state": np.full(3, i),
        "ep_idx": i, "suite": "test", "task_idx": 0,
    } for i in range(2)]
    policy = _BatchPolicy()
    with patch("pnp.rollout.obs_to_policy", return_value={}):
        results = run_episode_batch(
            envs, episodes, policy, lambda obs: obs, lambda action: action,
            torch.device("cpu"), RolloutConfig(save_generated_chunks=True))

    assert [result["n_steps"] for result in results] == [1, 3]
    assert [result["terminated_reason"] for result in results] == ["success", "success"]
    assert policy.batch_sizes == [2, 1]
    assert results[0]["episode_seed"] != results[1]["episode_seed"]


def test_batch_lane_noise_is_order_independent():
    episodes = [{
        "task_desc": "test task", "max_steps": 1, "init_state": np.full(3, i),
        "ep_idx": i, "suite": "test", "task_idx": 0,
    } for i in range(2)]
    with patch("pnp.rollout.obs_to_policy", return_value={}):
        together = run_episode_batch(
            [_TimedEnv(1), _TimedEnv(1)], episodes, _BatchPolicy(), lambda obs: obs,
            lambda action: action, torch.device("cpu"), RolloutConfig(save_generated_chunks=True))
        reversed_results = run_episode_batch(
            [_TimedEnv(1), _TimedEnv(1)], list(reversed(episodes)), _BatchPolicy(), lambda obs: obs,
            lambda action: action, torch.device("cpu"), RolloutConfig(save_generated_chunks=True))
    by_seed = {result["episode_seed"]: result["generated_chunks"]["chunks"]
               for result in reversed_results}
    for result in together:
        assert np.array_equal(result["generated_chunks"]["chunks"],
                              by_seed[result["episode_seed"]])
