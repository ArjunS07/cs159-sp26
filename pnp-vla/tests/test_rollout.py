from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pnp.config import RolloutConfig
from pnp.rollout import run_episode


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
