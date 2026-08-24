import numpy as np
import torch

from pnp.pcp_critic.config import PCPCriticModelConfig, PCPCriticTrainConfig, PCPSearchAdapterConfig
from pnp.pcp_critic.data import transitions_from_artifact
from pnp.pcp_critic.deploy import PCPSearchAdapter
from pnp.pcp_critic.model import PCPCritic
from pnp.pcp_critic.train import collate_transitions, train_critic
from pnp.pcp_search.data import build_training_artifact


def _prefix(value):
    return {
        "processed_images": [np.zeros((3, 2, 2), np.float32)],
        "processed_image_masks": [np.ones((1,), bool)],
        "token_ids": np.asarray([1, 2], np.int64), "token_masks": np.asarray([True, True]),
        "prefix_embeddings": np.full((5, 8), value, np.float16),
        "prefix_pad_masks": np.ones(5, bool), "prefix_attention_masks": np.zeros(5, bool),
        "prefix_attention_2d_masks": np.zeros((5, 5), bool), "prefix_position_ids": np.arange(5),
    }


def _artifact():
    decisions = [{"step": i * 10, "raw_agentview": np.zeros((2, 2, 3), np.uint8),
                  "raw_wrist": np.zeros((2, 2, 3), np.uint8),
                  "raw_robot_state": np.full(3, i, np.float32),
                  "policy_proprio": np.full(2, i, np.float32), "sim_state": np.zeros(4),
                  "instruction": "task"} for i in range(3)]
    return build_training_artifact(
        decisions=decisions, prefixes=[_prefix(i) for i in range(3)],
        generated_chunks=np.zeros((3, 50, 7), np.float32), normalized_actions=np.zeros((20, 7), np.float32),
        env_actions=np.zeros((20, 7), np.float32), rewards=np.r_[np.ones(10), np.full(10, 2)].astype(np.float32),
        terminated=np.r_[np.zeros(19, bool), True], truncated=np.zeros(20, bool),
        step_success=np.r_[np.zeros(19, bool), True], robot_states=np.zeros((21, 3), np.float32),
        sim_states=np.zeros((21, 4)), chunk_start_steps=[0, 10], chunk_noise_seeds=[1, 2, 3],
        episode_seed=1, perturb_seed=2, initial_state=np.zeros(4))


def test_pcp_critic_transition_uses_h10_return_and_never_exposes_sim_state():
    items = transitions_from_artifact({"rollout_id": "r", "benchmark": "libero", "suite": "s",
                                       "task_idx": 0, "init_state_hash": "i"}, _artifact(), gamma=.99)
    assert len(items) == 2
    assert items[1].terminal and items[1].discount == 0
    assert items[0].mc_return > items[0].reward
    assert not hasattr(items[0], "sim_state")


def test_twin_critic_is_action_sensitive_and_adapter_cannot_mutate_offline():
    items = transitions_from_artifact({"rollout_id": "r", "benchmark": "libero", "suite": "s",
                                       "task_idx": 0, "init_state_hash": "i"}, _artifact(), gamma=.99)
    batch = collate_transitions(items)
    model = PCPCritic(prefix_dim=8, robot_dim=3, proprio_dim=2,
                      config=PCPCriticModelConfig(width=32, n_blocks=2, dropout=0))
    q0 = model.minimum_q(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"], batch["action"])
    changed = batch["action"].clone(); changed[:, :, 0] += 0.5
    q1 = model.minimum_q(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"], changed)
    assert not torch.allclose(q0, q1)
    adapter = PCPSearchAdapter(model)
    try:
        adapter.corrected_action(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"], batch["action"])
    except RuntimeError as exc:
        assert "offline_only" in str(exc)
    else:
        raise AssertionError("offline adapter unexpectedly mutated an action")


def test_pcp_critic_smoke_training_supports_both_objectives():
    items = transitions_from_artifact({"rollout_id": "r", "benchmark": "libero", "suite": "s",
                                       "task_idx": 0, "init_state_hash": "i"}, _artifact(), gamma=.99)
    for objective in ("calql", "iql"):
        model = PCPCritic(prefix_dim=8, robot_dim=3, proprio_dim=2,
                          config=PCPCriticModelConfig(width=32, n_blocks=1, dropout=0))
        _, report = train_critic(model, items, items, torch.device("cpu"), config=PCPCriticTrainConfig(
            objective=objective, batch_size=2, updates=2, eval_interval=1, patience=2,
            n_local_actions=1, n_broad_actions=1))
        assert report["objective"] == objective
        assert report["updates_ran"] >= 1
