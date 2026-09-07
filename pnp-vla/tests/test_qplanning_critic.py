import numpy as np
import torch

from pnp.pcp_search.data import build_training_artifact
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.qplanning_critic.data import collate_windows, qplanning_windows_from_artifact
from pnp.qplanning_critic.model import QPlanningCritic


def _prefix(value):
    return {
        "processed_images": [np.zeros((3, 2, 2), np.float32)],
        "processed_image_masks": [np.ones((1,), bool)],
        "token_ids": np.asarray([1, 2], np.int64),
        "token_masks": np.asarray([True, True]),
        "prefix_embeddings": np.full((5, 8), value, np.float16),
        "prefix_pad_masks": np.ones(5, bool),
        "prefix_attention_masks": np.zeros(5, bool),
        "prefix_attention_2d_masks": np.zeros((5, 5), bool),
        "prefix_position_ids": np.arange(5),
    }


def _artifact(n_steps=120):
    boundaries = list(range(0, n_steps + 1, 10))
    decisions = [{
        "step": step, "raw_agentview": np.zeros((2, 2, 3), np.uint8),
        "raw_wrist": np.zeros((2, 2, 3), np.uint8),
        "raw_robot_state": np.full(3, index, np.float32),
        "policy_proprio": np.full(2, index, np.float32),
        "sim_state": np.zeros(4), "instruction": "task",
    } for index, step in enumerate(boundaries)]
    actions = np.arange(n_steps * 7, dtype=np.float32).reshape(n_steps, 7) / 1000
    generated = np.zeros((len(boundaries), 50, 7), np.float32)
    for index, start in enumerate(boundaries[:-1]):
        generated[index, :min(10, n_steps - start)] = actions[start:start + 10]
    rewards = np.zeros(n_steps, np.float32); rewards[-1] = 1
    terminated = np.zeros(n_steps, bool); terminated[-1] = True
    success = terminated.copy()
    return build_training_artifact(
        decisions=decisions, prefixes=[_prefix(i) for i in range(len(boundaries))],
        generated_chunks=generated, normalized_actions=actions, env_actions=actions,
        rewards=rewards, terminated=terminated, truncated=np.zeros(n_steps, bool),
        step_success=success, robot_states=np.zeros((n_steps + 1, 3), np.float32),
        sim_states=np.zeros((n_steps + 1, 4)), chunk_start_steps=boundaries[:-1],
        chunk_noise_seeds=list(range(len(boundaries))), episode_seed=1,
        perturb_seed=2, initial_state=np.zeros(4))


def test_q50_uses_executed_trajectory_across_five_replans():
    arrays = _artifact()
    packed = qplanning_windows_from_artifact(
        {"rollout_id": "r"}, arrays, horizon=50, gamma=.99)
    np.testing.assert_allclose(packed["action"][0], arrays["actions_normalized"][:50])
    np.testing.assert_allclose(packed["next_action"][0], arrays["actions_normalized"][50:100])
    assert packed["discount"][0] == np.float32(.99 ** 50)
    assert packed["next_robot"][0, 0] == 5
    assert packed["action_valid"][-1].sum() == 10
    assert packed["discount"][-1] == 0


def test_q10_bootstraps_at_the_next_planning_boundary():
    arrays = _artifact()
    packed = qplanning_windows_from_artifact(
        {"rollout_id": "r"}, arrays, horizon=10, gamma=.99)
    np.testing.assert_allclose(packed["action"][0], arrays["actions_normalized"][:10])
    np.testing.assert_allclose(packed["next_action"][0], arrays["actions_normalized"][10:20])
    assert packed["next_robot"][0, 0] == 1
    assert packed["discount"][0] == np.float32(.99 ** 10)
    assert packed["discount"][-1] == 0


def test_single_q_rl_token_decoder_and_hl_gauss_head():
    arrays = _artifact(20)
    packed = qplanning_windows_from_artifact(
        {"rollout_id": "r"}, arrays, horizon=10, gamma=.99)
    items = [{key: value[i] for key, value in packed.items()} for i in range(len(packed["reward"]))]
    batch = collate_windows(items)
    model = QPlanningCritic(
        prefix_dim=8, robot_dim=3, proprio_dim=2,
        config=QPlanningModelConfig(
            action_horizon=10, width=32, n_layers=1, n_heads=4,
            ffn_width=64, dropout=0, n_bins=11, hl_gauss_sigma=.1))
    logits = model(
        batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
        batch["action"], batch["action_valid"])
    targets = model.hl_gauss_targets(torch.tensor([0.0, 1.0]))
    assert logits.shape == (2, 11)
    assert model.rl_token.shape == (1, 1, 32)
    assert len(model.value_head.weight) == 11
    assert torch.allclose(targets.sum(-1), torch.ones(2), atol=1e-6)
    assert model.categorical_loss(logits, torch.tensor([0.0, 1.0])).isfinite()
