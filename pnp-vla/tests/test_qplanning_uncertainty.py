import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.qplanning_critic.data import collate_windows
from pnp.qplanning_critic.uncertainty import (
    QPlanningU20Critic, _u20_labels, load_qplanning_u20_scorer)
from pnp.qplanning_critic.inference import qplanning_select
from pnp.qplanning_u20_eval_experiment import (
    QPLANNING_U20_BETAS, QPLANNING_U20_SHARDS, build_qplanning_u20_method)


def test_u20_labels_align_boundaries_and_exclude_current_boundary():
    payload = io.BytesIO()
    arrays = {}
    for chunk in range(6):
        for step in (3, 4):
            arrays[f"c{chunk}_s{step}_u_time"] = np.full(
                50, chunk + 1 + 0.1 * (step - 3), np.float32)
    np.savez_compressed(payload, **arrays)
    labels = _u20_labels(payload.getvalue(), expected_windows=6)
    np.testing.assert_allclose(
        labels["current_u20"], np.arange(1, 7) + 0.05, rtol=1e-6)
    np.testing.assert_allclose(
        labels["future_u20"][0], np.mean([2.05, 3.05, 4.05, 5.05]), rtol=1e-6)
    np.testing.assert_allclose(labels["next_current_u20"][0], 6.05, rtol=1e-6)
    assert labels["future_u20_valid"].tolist() == [True, True, True, True, True, False]


def test_u20_critic_outputs_q_and_future_uncertainty():
    items = []
    for index in range(2):
        items.append({
            "prefix": np.full((3, 8), index, np.float32),
            "pad": np.ones(3, bool),
            "robot": np.zeros(3, np.float32),
            "proprio": np.zeros(2, np.float32),
            "action": np.zeros((50, 7), np.float32),
            "action_valid": np.ones(50, bool),
            "next_prefix": np.full((3, 8), index + 1, np.float32),
            "next_pad": np.ones(3, bool),
            "next_robot": np.zeros(3, np.float32),
            "next_proprio": np.zeros(2, np.float32),
            "next_action": np.zeros((50, 7), np.float32),
            "next_action_valid": np.ones(50, bool),
            "reward": np.float32(0), "discount": np.float32(.99 ** 50),
            "mc_return": np.float32(index), "success": np.bool_(index),
            "start_step": np.int32(index * 10),
            "current_u20": np.float32(.02 + .001 * index),
            "next_current_u20": np.float32(.021 + .001 * index),
            "future_u20": np.float32(.022 + .001 * index),
            "future_u20_valid": np.bool_(True),
        })
    batch = collate_windows(items)
    model = QPlanningU20Critic(
        prefix_dim=8, robot_dim=3, proprio_dim=2,
        config=QPlanningModelConfig(
            action_horizon=50, width=32, n_layers=1, n_heads=4,
            ffn_width=64, dropout=0, n_bins=11, hl_gauss_sigma=.1))
    logits, predicted_u = model(
        batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
        batch["action"], batch["action_valid"], batch["current_u20"])
    assert logits.shape == (2, 11)
    assert predicted_u.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(predicted_u).all()
    assert batch["future_u20_valid"].dtype == torch.bool


def test_q50_u20_notebook_uses_full_8000_update_entrypoint():
    notebook = json.loads(
        open("notebooks/70_train_qplanning_q50_u20.ipynb", encoding="utf-8").read())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_q50_u20_training" in source
    assert "run_mode" not in source
    assert "PASTE_PCPCDS_ID_FROM_NOTEBOOK_56" in source


def test_u20_checkpoint_loader_and_eval_method(tmp_path):
    model = QPlanningU20Critic(
        prefix_dim=8, robot_dim=3, proprio_dim=2,
        config=QPlanningModelConfig(
            action_horizon=50, width=16, n_layers=1, n_heads=4,
            ffn_width=32, dropout=0, n_bins=11))
    path = tmp_path / "checkpoint_step_008000.pt"
    torch.save({
        "format": "qplanning_q50_u20_v1", "update": 8000,
        "snapshot_id": "pcpcds-u20", "cache_digest": "digest",
        "source_policy": {"repo_id": "repo", "revision": "revision"},
        "architecture": model.architecture_config(),
        "model": model.state_dict(),
    }, path)
    scorer = load_qplanning_u20_scorer(
        path, uncertainty_beta=.5, device="cpu",
        expected_source_revision="revision")
    method, config = build_qplanning_u20_method(
        "repo@revision", scorer, candidate_batch_size=8)
    assert scorer.update == 8000 and scorer.uncertainty_beta == .5
    assert method == "qplanning_q50_u20_beta050"
    assert config.num_samples == 64 and config.qplanning_n_elites == 16
    assert config.num_inference_steps == 3 and config.n_action_steps == 10
    assert config.video == "off" and not config.save_generated_chunks


def test_u20_selector_measures_live_ten_step_u_and_restores_three_steps():
    class Policy:
        def __init__(self):
            self.model = SimpleNamespace(
                _pnp=SimpleNamespace(strategy=None, num_steps=3, action_dim=7))
            self.config = SimpleNamespace(num_inference_steps=3)

        def predict_action_chunk(self, batch, noise):
            strategy = self.model._pnp.strategy
            if strategy is not None:
                strategy.finish(SimpleNamespace(
                    prefix_embeddings=torch.zeros((len(noise), 3, 8)),
                    prefix_pad_masks=torch.ones((len(noise), 3), dtype=torch.bool)))
            return noise

    class Scorer:
        requires_current_u20 = True
        uncertainty_beta = .5

        def score_components(self, prefix, valid, robot, proprio, actions,
                             current_u20, batch_size):
            assert current_u20 == .02
            return (
                torch.tensor([.1, .2, .3]),
                torch.tensor([.3, .2, .1]),
                torch.tensor([-1., 0., 1.]))

    policy = Policy()
    noises = torch.stack([
        torch.full((50, 7), float(index)) for index in range(3)])

    def measure(measured_policy, batch, noise, **kwargs):
        assert measured_policy.model._pnp.num_steps == 10
        assert kwargs == {
            "probe_steps": (3, 4), "num_iterations": 5,
            "uncertainty_horizon": 20}
        return noise, .02

    with patch("pnp.sampler.measure_chunk_uncertainty", side_effect=measure):
        _, telemetry = qplanning_select(
            policy, {}, noises, scorer=Scorer(),
            robot_state=np.zeros(3), policy_proprio=np.zeros(2),
            n_elites=2, temperature=1, candidate_batch_size=3)
    assert policy.model._pnp.num_steps == 3
    assert telemetry["best_index"] == 2
    assert telemetry["current_u20"] == .02
    assert telemetry["uncertainty_beta"] == .5
    assert telemetry["elite_indices"] == [2, 1]


def test_two_u20_workers_are_fixed_three_arm_shards():
    assert QPLANNING_U20_SHARDS == 2
    assert QPLANNING_U20_BETAS == (.25, .5, 1.)
    root = Path(__file__).parents[1]
    for shard_index in range(2):
        path = root / "notebooks" / "workers" / (
            f"71_eval_qplanning_q50_u20_heldout160_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "SHARD_COUNT = 2" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "EPISODE_LIMIT = None" in source
        assert "run_qplanning_u20_heldout_worker(" in source
        assert "checkpoint_step_008000.pt" in source
        assert "reuse exact notebook-68 rows" in source
        assert "video_frames_generated_chunks" in source
