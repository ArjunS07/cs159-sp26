import numpy as np
import torch

from pnp.qplanning_critic.config import QPlanningModelConfig, QPlanningTrainConfig
from pnp.qplanning_critic.data import QPlanningCacheIndex, QPlanningWindowDataset
from pnp.qplanning_critic.model import QPlanningCritic
from pnp.qplanning_critic.train import train_qplanning_critic


def test_training_can_stop_before_original_lr_schedule_horizon():
    config = QPlanningTrainConfig(
        updates=6_000, lr_schedule_updates=8_000, warmup_updates=500)
    assert config.learning_rate_at(6_000) > 0
    assert QPlanningTrainConfig(
        updates=8_000, warmup_updates=500).learning_rate_at(6_000) == (
            config.learning_rate_at(6_000))


def test_qplanning_trainer_saves_resumable_single_q_checkpoint(tmp_path):
    n, horizon = 4, 10
    arrays = {
        "prefix": np.zeros((n, 3, 8), np.float16),
        "pad": np.ones((n, 3), bool),
        "robot": np.zeros((n, 3), np.float32),
        "proprio": np.zeros((n, 2), np.float32),
        "action": np.zeros((n, horizon, 7), np.float32),
        "action_valid": np.ones((n, horizon), bool),
        "next_prefix": np.zeros((n, 3, 8), np.float16),
        "next_pad": np.ones((n, 3), bool),
        "next_robot": np.zeros((n, 3), np.float32),
        "next_proprio": np.zeros((n, 2), np.float32),
        "next_action": np.zeros((n, horizon, 7), np.float32),
        "next_action_valid": np.ones((n, horizon), bool),
        "reward": np.asarray([0, 0, 1, 1], np.float32),
        "discount": np.zeros(n, np.float32),
        "mc_return": np.asarray([0, 0, 1, 1], np.float32),
        "success": np.asarray([False, False, True, True]),
        "start_step": np.arange(n, dtype=np.int32) * 10,
    }
    np.savez(tmp_path / "r.npz", **arrays)
    entry = {"rollout_id": "r", "path": "r.npz", "n_windows": n}
    cache = QPlanningCacheIndex(
        "snapshot", horizon, str(tmp_path), (entry,), 8, 3, 2, 7,
        tuple(np.zeros(7)), tuple(np.ones(7)), n, n, n, 0, 0.0, 0.0)
    dataset = QPlanningWindowDataset(cache, {"r"})
    model = QPlanningCritic(
        prefix_dim=8, robot_dim=3, proprio_dim=2,
        config=QPlanningModelConfig(
            action_horizon=horizon, width=32, n_layers=1, n_heads=4,
            ffn_width=64, dropout=0, n_bins=11, hl_gauss_sigma=.1))
    config = QPlanningTrainConfig(
        effective_batch_size=2, micro_batch_size=2, updates=2,
        warmup_updates=1, print_interval=1, eval_interval=1,
        checkpoint_interval=1, max_validation_transitions=4, use_bf16=False)
    _, _, report = train_qplanning_critic(
        model, dataset, dataset, torch.device("cpu"), snapshot_id="snapshot",
        cache_digest=cache.digest, source_policy={"repo_id": "pi", "revision": "rev"},
        output_dir=tmp_path / "checkpoints", config=config, resume=False)
    checkpoint = torch.load(report["final_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["format"] == "qplanning_critic_v1"
    assert checkpoint["source_policy"] == {"repo_id": "pi", "revision": "rev"}
    assert checkpoint["update"] == 2
    assert "target" in checkpoint and "optimizer" in checkpoint
    assert [path.name for path in (tmp_path / "checkpoints").glob("checkpoint_step_*.pt")] == [
        "checkpoint_step_000002.pt"]
