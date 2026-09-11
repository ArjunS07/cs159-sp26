import io
import json

import numpy as np
import torch

from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.qplanning_critic.data import collate_windows
from pnp.qplanning_critic.uncertainty import (
    QPlanningU20Critic, _u20_labels)


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
