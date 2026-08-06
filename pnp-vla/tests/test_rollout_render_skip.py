"""skip_unused_renders must be invisible to the policy.

Camera rendering is ~90% of a LIBERO step and the policy reads an observation only at chunk
boundaries, so 49 of 50 renders are discarded. Skipping them is pure speed -- unless the toggle is
off by one, in which case the policy silently receives a stale image and success rate collapses
with no error. These tests stamp every rendered frame with its step index so a stale or missing
image is detectable, and assert the executed actions are identical with and without skipping.
"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pnp.config import NUM_STEPS_WAIT, RolloutConfig
from pnp.rollout import run_episode

CHUNK = 5
MAX_STEPS = 17
N_DECISIONS = -(-MAX_STEPS // CHUNK)
# reset() + set_init_state() + one per settling step + one per executed step.
ALWAYS_RENDERED = 2 + NUM_STEPS_WAIT + MAX_STEPS
# The settling loop consumes stamps 0..NUM_STEPS_WAIT-1, so main-loop step j returns stamp
# NUM_STEPS_WAIT + j. Decision i re-plans at main step CHUNK*i from the step before it.
LIVE_STAMPS = [NUM_STEPS_WAIT - 1 + CHUNK * i for i in range(N_DECISIONS)]


class _Env:
    """Records which steps rendered, and stamps each image with the step that produced it."""

    def __init__(self, supports_toggle=True):
        self.supports_toggle = supports_toggle
        self.cameras_on = True
        self.render_steps = []
        self.actions = []
        self.toggles = []
        self._step = -1        # -1 == set_init_state, 0..n-1 == settle steps, then main loop

    # robosuite exposes the toggle on the wrapped env; mirror that shape.
    @property
    def env(self):
        return self

    def modify_observable(self, name, attribute, value):
        if not self.supports_toggle:
            raise NotImplementedError("no observable toggle")
        assert attribute == "enabled"
        self.cameras_on = bool(value)
        self.toggles.append((name, bool(value)))

    def _observation(self):
        obs = {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": np.zeros(2),
        }
        if self.cameras_on:
            self.render_steps.append(self._step)
            stamp = np.full((4, 4, 3), self._step, dtype=np.uint8)
            obs["agentview_image"] = stamp
            obs["robot0_eye_in_hand_image"] = stamp
        return obs

    def reset(self):
        return self._observation()

    def set_init_state(self, _state):
        return self._observation()

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self._step += 1
        return self._observation(), 0.0, False, {}

    def check_success(self):
        return False


class _Policy:
    """Records the image stamp it was given at each decision point."""

    def __init__(self):
        self.model = SimpleNamespace(_pnp=SimpleNamespace(
            action_dim=7, num_steps=None, vf_evals=0, strategy=None, chunk_pos=0.0))
        self.config = SimpleNamespace(
            chunk_size=CHUNK, max_action_dim=7, num_inference_steps=10)
        self.seen = []

    def reset(self):
        pass

    def predict_action_chunk(self, batch, noise=None):
        self.seen.append(batch["stamp"])
        return torch.ones((1, CHUNK, 7))


def _episode():
    return {"task_desc": "t", "max_steps": MAX_STEPS, "init_state": np.zeros(3),
            "ep_idx": 0, "suite": "s", "task_idx": 0}


def _obs_to_policy(obs, _task):
    # Fails loudly if the image is absent -- exactly the failure skipping could introduce.
    return {"stamp": int(obs["agentview_image"][0, 0, 0])}


def _run(env, policy, **config_kwargs):
    with patch("pnp.rollout.obs_to_policy", _obs_to_policy):
        return run_episode(env, _episode(), policy, lambda b: b, lambda a: a,
                           torch.device("cpu"), RolloutConfig(**config_kwargs))


def test_policy_sees_a_live_image_at_every_decision_point():
    env, policy = _Env(), _Policy()
    _run(env, policy, skip_unused_renders=True)

    # One decision per chunk boundary over MAX_STEPS executed actions.
    assert len(policy.seen) == N_DECISIONS
    # The first decision uses the last settling observation; each later one uses the step
    # immediately before it. A stale frame would show up as a smaller stamp here.
    assert policy.seen == LIVE_STAMPS


def test_skipping_matches_always_rendering_exactly():
    baseline_env, baseline_policy = _Env(), _Policy()
    _run(baseline_env, baseline_policy, skip_unused_renders=False)
    skipped_env, skipped_policy = _Env(), _Policy()
    _run(skipped_env, skipped_policy, skip_unused_renders=True)

    assert skipped_policy.seen == baseline_policy.seen
    assert len(skipped_env.actions) == len(baseline_env.actions)
    for skipped, baseline in zip(skipped_env.actions, baseline_env.actions):
        assert np.array_equal(skipped, baseline)


def test_skipping_renders_far_fewer_frames():
    baseline = _Env()
    _run(baseline, _Policy(), skip_unused_renders=False)
    skipped = _Env()
    _run(skipped, _Policy(), skip_unused_renders=True)
    assert len(skipped.render_steps) < len(baseline.render_steps)
    # One render per decision, plus reset/set_init_state, never one per step.
    assert len(skipped.render_steps) <= len(baseline.render_steps) // 2


def test_frame_sinks_force_every_render():
    """save_observations/video consume every frame, so skipping must switch itself off."""
    env = _Env()
    _run(env, _Policy(), skip_unused_renders=True, save_observations=True)
    assert env.toggles == []
    assert len(env.render_steps) == ALWAYS_RENDERED


def test_unsupported_toggle_falls_back_to_always_rendering():
    """An older robosuite without modify_observable must not lose the policy's images."""
    env, policy = _Env(supports_toggle=False), _Policy()
    _run(env, policy, skip_unused_renders=True)
    assert len(env.render_steps) == ALWAYS_RENDERED
    assert policy.seen == LIVE_STAMPS


def test_cameras_are_restored_for_the_next_rollout_on_the_same_env():
    env = _Env()
    _run(env, _Policy(), skip_unused_renders=True)
    assert env.cameras_on is True
