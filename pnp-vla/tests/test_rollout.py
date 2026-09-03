from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pnp.config import RolloutConfig
from pnp.rollout import (_run_episode_serial, candidate_action_disagreement,
                         candidate_chunk_noise_seed, chunk_noise_seed, run_episode,
                         run_episode_batch, _stack_batches, _stack_policy_batches)


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
            chunk_size=2, n_action_steps=2, max_action_dim=7, num_inference_steps=10,
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


def test_run_episode_executes_configured_prefix_but_saves_full_generated_chunk():
    class HorizonPolicy(_Policy):
        def __init__(self):
            super().__init__()
            self.config.chunk_size = 4
            self.config.n_action_steps = 1

        def predict_action_chunk(self, _batch, noise=None):
            values = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
            return values.expand(1, 4, 7)

    env = _Env()
    policy = HorizonPolicy()
    ep = {
        "task_desc": "test task", "max_steps": 3,
        "init_state": np.zeros(3), "ep_idx": 0,
        "suite": "test", "task_idx": 0,
    }
    config = RolloutConfig(n_action_steps=2, save_generated_chunks=True)
    with patch("pnp.rollout.obs_to_policy", return_value={}):
        result = run_episode(
            env, ep, policy, lambda obs: obs, lambda action: action,
            torch.device("cpu"), config)

    assert [float(action[0]) for action in env.actions[-3:]] == [0.0, 1.0, 0.0]
    assert result["n_chunks"] == 2
    assert result["generated_chunks"]["chunks"].shape == (2, 4, 7)
    assert policy.config.n_action_steps == 1


def test_dual_policy_episode_selects_and_logs_clean_candidate_diversity():
    env = _Env()
    ep = {
        "task_desc": "test task", "max_steps": 1,
        "init_state": np.zeros(3), "ep_idx": 0,
        "suite": "test", "task_idx": 0,
    }
    source, member = _Policy(), _Policy()
    candidate_actions = [torch.ones((1, 2, 7)), torch.full((1, 2, 7), 2.0)]
    bundles = [
        ("source", source, lambda obs: obs, lambda action: action * 2),
        ("model_1", member, lambda obs: obs, lambda action: action * 3),
    ]
    config = RolloutConfig(
        num_samples=2, pnp_k=5, ms_probe_steps=(3, 4),
        candidate_set_id="source|m1", save_generated_chunks=True)
    with (patch("pnp.rollout.obs_to_policy", return_value={}),
          patch("pnp.rollout.multi_policy_select", return_value=(
              candidate_actions[1], 1, [.2, .1], candidate_actions))):
        result = run_episode(
            env, ep, source, lambda obs: obs, lambda action: action,
            torch.device("cpu"), config, candidate_bundles=bundles)

    assert np.allclose(env.actions[-1], 6.0)
    assert result["ms_selections"][0]["chosen_label"] == "model_1"
    assert result["ms_selections"][0]["action_disagreement"]["action_l2_mean"] > 0
    assert result["generated_chunks"]["candidate_chunks"].shape == (1, 2, 2, 7)
    assert candidate_chunk_noise_seed(123, 4, 0) == chunk_noise_seed(123, 4)
    assert candidate_chunk_noise_seed(123, 4, 1) != chunk_noise_seed(123, 4)


def test_same_policy_candidates_preserve_stock_slot_zero_and_log_all_chunks():
    env = _Env()
    ep = {
        "task_desc": "test task", "max_steps": 2,
        "init_state": np.zeros(3), "ep_idx": 0,
        "suite": "test", "task_idx": 0,
    }
    policy = _Policy()
    drawn_seeds = []

    def draw(_policy, _device, seed):
        drawn_seeds.append(seed)
        return torch.zeros((1, 2, 7))

    def select(_policy, _batch, _base_seed, _chunk_idx, num_samples, _probe_steps,
               noise_of, **kwargs):
        assert kwargs["return_actions"] is True
        assert kwargs["return_details"] is True
        assert "perturb_seed_of" in kwargs
        for index in range(num_samples):
            noise_of(index)
        actions = [torch.full((1, 2, 7), float(index + 1))
                   for index in range(num_samples)]
        profiles = [
            {"u10": .3 - index * .1, "u20": .3 - index * .1,
             "u_full": .3 - index * .1, "contraction10": 0.0,
             "contraction20": 0.0, "contraction_full": 0.0}
            for index in range(num_samples)]
        return actions[2], 2, [.3, .2, .1], actions, profiles

    config = RolloutConfig(
        num_samples=3, pnp_k=5, ms_probe_steps=(2, 3),
        selection_uncertainty_horizon=2, candidate_seed_scheme="stock_slot0_v1",
        candidate_set_id="source|source|source", num_inference_steps=5,
        n_action_steps=1, save_generated_chunks=True)
    with (patch("pnp.rollout.obs_to_policy", return_value={}),
          patch("pnp.rollout._draw_chunk_noise", side_effect=draw),
          patch("pnp.rollout.multi_sample_select", side_effect=select)):
        result = run_episode(
            env, ep, policy, lambda obs: obs, lambda action: action,
            torch.device("cpu"), config)

    # Per chunk: ordinary seed draw, followed by candidate slots 0, 1, 2.
    assert drawn_seeds[0] == drawn_seeds[1]
    assert drawn_seeds[4] == drawn_seeds[5]
    assert result["generated_chunks"]["candidate_chunks"].shape == (2, 3, 2, 7)
    assert result["chunk_noise_seeds"] == [
        candidate_chunk_noise_seed(result["episode_seed"], 0, 2),
        candidate_chunk_noise_seed(result["episode_seed"], 1, 2),
    ]
    np.testing.assert_array_equal(
        result["generated_chunks"]["chunk_noise_seeds"],
        result["chunk_noise_seeds"])
    assert len(result["ms_selections"]) == 2
    for selection in result["ms_selections"]:
        assert selection["chosen"] == 2
        assert selection["u_spread"] == pytest.approx(.2)
        assert selection["action_disagreement"]["n_candidate_pairs"] == 3
        assert selection["executed_prefix_disagreement"]["actions_compared"] == 1
        assert selection["inference_ms"] >= 0
        assert selection["n_vf_evals"] >= 0
        assert selection["selected_noise_seed"] == selection["candidate_noise_seeds"][2]


def test_same_policy_select_then_refine_is_an_explicit_second_stage():
    env = _Env()
    ep = {
        "task_desc": "test task", "max_steps": 1,
        "init_state": np.zeros(3), "ep_idx": 0,
        "suite": "test", "task_idx": 0,
    }
    candidates = [torch.ones((1, 2, 7)), torch.full((1, 2, 7), 2.0),
                  torch.full((1, 2, 7), 3.0)]
    profiles = [
        {"u10": value, "u20": value, "u_full": value,
         "contraction10": 0.0, "contraction20": 0.0, "contraction_full": 0.0}
        for value in (.3, .2, .4)]
    refined = torch.full((1, 2, 7), 4.0)
    config = RolloutConfig(
        num_samples=3, pnp_k=5, ms_probe_steps=(2, 3),
        selection_uncertainty_horizon=2, candidate_seed_scheme="stock_slot0_v1",
        multi_sample_refine_selected=True, candidate_set_id="source|source|source",
        num_inference_steps=5, n_action_steps=1, save_generated_chunks=True)
    with (patch("pnp.rollout.obs_to_policy", return_value={}),
          patch("pnp.rollout.multi_sample_select", return_value=(
              candidates[1], 1, [.3, .2, .4], candidates, profiles)),
          patch("pnp.sampler.refine_action_chunk", return_value=(
              refined, .1, {**profiles[1], "u20": .1})) as refine):
        result = run_episode(
            env, ep, _Policy(), lambda obs: obs, lambda action: action,
            torch.device("cpu"), config)

    assert refine.call_count == 1
    assert np.allclose(env.actions[-1], 4.0)
    telemetry = result["ms_selections"][0]["selected_refinement"]
    assert telemetry["pre_u"] == pytest.approx(.2)
    assert telemetry["refined_path_u"] == pytest.approx(.1)
    assert telemetry["delta_u"] == pytest.approx(-.1)
    assert telemetry["lowered_u"] is True
    assert telemetry["initial_noise_seed"] == result["ms_selections"][0][
        "selected_noise_seed"]
    assert result["chunk_noise_seeds"] == [
        result["ms_selections"][0]["selected_noise_seed"]]
    assert telemetry["selected_prefix_movement"]["action_l2_mean"] > 0


def test_candidate_action_disagreement_is_zero_for_identical_chunks():
    action = torch.ones((1, 2, 7))
    metrics = candidate_action_disagreement([action, action.clone()])
    assert metrics["action_l2_mean"] == 0.0
    assert metrics["action_l2_normalized"] == 0.0
    assert np.isclose(metrics["action_cosine"], 1.0)


def test_candidate_action_disagreement_supports_three_pairwise_candidates():
    actions = [torch.full((1, 2, 7), value) for value in (0.0, 1.0, 3.0)]
    metrics = candidate_action_disagreement(actions)
    assert metrics["n_candidates"] == 3
    assert metrics["n_candidate_pairs"] == 3
    assert metrics["action_l2_mean"] > 0
    assert np.isclose(metrics["action_l2_max"], np.sqrt(7) * 3)


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


def test_stack_batches_preserves_optional_processor_fields():
    batched = _stack_batches([
        {"tokens": torch.ones(1, 2), "pixel_mask": None, "metadata": 0.5},
        {"tokens": torch.zeros(1, 2), "pixel_mask": None, "metadata": 0.5},
    ])
    assert batched["tokens"].shape == (2, 2)
    assert batched["pixel_mask"] is None
    assert batched["metadata"] == 0.5


def test_stack_policy_batches_discards_transition_bookkeeping_fields():
    batched = _stack_policy_batches([
        {"observation.state": torch.ones(1, 2), "reward": 0.1,
         "done": False, "info": {"elapsed": 1.0}},
        {"observation.state": torch.zeros(1, 2), "reward": 0.2,
         "done": False, "info": {"elapsed": 2.0}},
    ])
    assert set(batched) == {"observation.state"}
    assert batched["observation.state"].shape == (2, 2)


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


def _horizon_episodes():
    return [{
        "task_desc": "test task", "max_steps": 4, "init_state": np.full(3, i),
        "ep_idx": i, "suite": "test", "task_idx": 0,
    } for i in range(2)]


def test_run_episode_batch_executes_configured_horizon_then_replans():
    # _BatchPolicy generates a 2-action chunk; n_action_steps=1 executes one action then
    # replans, so a chunk is generated on every executed step (closed-loop, not open-loop).
    with patch("pnp.rollout.obs_to_policy", return_value={}):
        results = run_episode_batch(
            [_TimedEnv(9), _TimedEnv(9)], _horizon_episodes(), _BatchPolicy(), lambda obs: obs,
            lambda action: action, torch.device("cpu"), RolloutConfig(n_action_steps=1))
    for result in results:
        assert result["n_steps"] == 4      # never succeeds (needs 19 actions), runs to max_steps
        assert result["n_chunks"] == 4     # one chunk generated per executed action


def test_run_episode_batch_matches_serial_under_closed_loop_horizon():
    cfg = RolloutConfig(n_action_steps=1, save_generated_chunks=True)
    with patch("pnp.rollout.obs_to_policy", return_value={}):
        batched = run_episode_batch(
            [_TimedEnv(9), _TimedEnv(9)], _horizon_episodes(), _BatchPolicy(), lambda o: o,
            lambda a: a, torch.device("cpu"), cfg)
        serial = [
            _run_episode_serial(env, ep, _BatchPolicy(), lambda o: o, lambda a: a,
                                torch.device("cpu"), cfg)
            for env, ep in zip([_TimedEnv(9), _TimedEnv(9)], _horizon_episodes())]
    for lane, single in zip(batched, serial):
        assert lane["n_steps"] == single["n_steps"]
        assert lane["n_chunks"] == single["n_chunks"]
        assert np.array_equal(lane["generated_chunks"]["chunks"],
                              single["generated_chunks"]["chunks"])
