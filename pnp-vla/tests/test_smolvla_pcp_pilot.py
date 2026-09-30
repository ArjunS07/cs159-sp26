import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_pcp_pilot import controlled_chunks, direction_and_benchmark, load_scalar_checkpoint
from pnp.smolvla_scalar_returns import FORMAT, ScalarCritic, LateFusionScalarCritic


def test_controls_are_paired_equal_rms_and_leave_unknown_tail_unchanged():
    stock = np.zeros((50, 7), np.float32)
    valid = np.arange(10) < 4
    direction = np.ones((10, 7), np.float32) * valid[:, None]
    extras = controlled_chunks(stock, direction, valid, 99)
    for scale, radius in (("small", .02), ("large", .06)):
        plus = extras[f"q_plus_{scale}"][0]
        minus = extras[f"q_minus_{scale}"][0]
        np.testing.assert_array_equal(plus, -minus)
        for control in ("q_plus", "q_minus", "random"):
            proposal, meta = extras[f"{control}_{scale}"]
            assert meta["realized_normalized_action_rms"] == pytest.approx(radius, abs=1e-7)
            np.testing.assert_array_equal(proposal[4:], stock[4:])
    again = controlled_chunks(stock, direction, valid, 99)
    np.testing.assert_array_equal(extras["random_large"][0], again["random_large"][0])
    with pytest.raises(ValueError, match="zero/nonfinite"):
        controlled_chunks(stock, np.zeros_like(direction), valid, 1)


def fixture_model(cls=ScalarCritic):
    return cls(prefix_dim=8, robot_dim=2, proprio_dim=3, config=QPlanningModelConfig(
        width=16, n_heads=2, n_layers=1, ffn_width=32, dropout=0,
        action_dim=7, action_horizon=10, prefix_pool_tokens=4))


@pytest.mark.parametrize("cls", [ScalarCritic, LateFusionScalarCritic])
def test_strict_checkpoint_loader_supports_both_scalar_families(tmp_path, cls):
    model = fixture_model(cls)
    path = tmp_path / "critic.pt"
    payload = {"format": FORMAT, "scalar_head": True, "snapshot_digest": "same",
               "architecture": model.architecture_config(), "model": model.state_dict(), "update": 5}
    torch.save(payload, path)
    loaded, meta = load_scalar_checkpoint(path, {"snapshot_digest": "same"})
    assert type(loaded) is cls
    assert len(meta["critic_sha256"]) == 64
    assert not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError, match="exact snapshot"):
        load_scalar_checkpoint(path, {"snapshot_digest": "other"})
    payload["model"].pop("value_head.bias")
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_scalar_checkpoint(path, {"snapshot_digest": "same"})


def test_outcomes_do_not_change_gradient_or_inputs():
    torch.manual_seed(8)
    model = fixture_model().eval().requires_grad_(False)
    rng = np.random.default_rng(6)
    row = {"prefix": rng.normal(size=(4, 8)).astype(np.float32), "pad": np.ones(4, bool),
           "robot": np.zeros(2, np.float32), "proprio": np.zeros(3, np.float32),
           "actions": rng.normal(size=(9, 10, 7)).astype(np.float32),
           "action_valid": np.ones((9, 10), bool), "success": np.arange(9) < 3}
    first, report = direction_and_benchmark(model, row)
    row["success"] = ~row["success"]
    second, inverted = direction_and_benchmark(model, row)
    np.testing.assert_array_equal(first, second)
    assert report["q_scores"] == inverted["q_scores"]
    assert all(p.grad is None for p in model.parameters())
    row["action_valid"][1, 8:] = False
    with pytest.raises(ValueError, match="preaction root"):
        direction_and_benchmark(model, row)


def test_notebook_is_thin_and_disabled_by_default():
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/workers/120_smolvla_pcp_pilot.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "RUN_INTERVENTION = False" in source
    assert "ROOT_LIMIT = 1" in source
    assert "UPLOAD_CHECKPOINT" in source and "CHECKPOINT_PATH" in source
    assert "run_pcp_pilot(" in source
    assert "not the original flow-step PCP" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")
