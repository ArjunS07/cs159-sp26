import json
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch

from pnp.config import Method, RolloutConfig
from pnp.tap import RolloutTap
from pnp.smolvla_consensus_projection_experiment import (
    PROJECTION_K_VALUES,
    PROJECTION_STEP,
    build_smolvla_consensus_projection_methods,
)


def test_consensus_projection_config_contract():
    methods = build_smolvla_consensus_projection_methods()
    assert [name for name, _ in methods] == [
        Method.SMOLVLA_CONSENSUS_PROJECT_K3,
        Method.SMOLVLA_CONSENSUS_PROJECT_K5,
    ]
    assert [config.consensus_projection_k for _, config in methods] == list(
        PROJECTION_K_VALUES) == [3, 5]
    for _, config in methods:
        assert tuple(config.pnp_steps) == (1, 2, 3)
        assert tuple(config.pnp_k_by_step) == (3, 1, 1)
        assert config.refine
        assert config.num_inference_steps == 10
        assert config.n_action_steps == 10
        assert config.consensus_projection_step == PROJECTION_STEP == 5
        assert config.save_time_uncertainty
        assert config.video == "off"


def test_consensus_projection_requires_complete_valid_settings():
    with pytest.raises(ValueError, match="both projection_k and projection_step"):
        RolloutConfig(
            pnp_steps=(1,), refine=True, num_inference_steps=10,
            consensus_projection_k=3)
    with pytest.raises(ValueError, match="P&P-refined parent"):
        RolloutConfig(
            pnp_steps=(1,), num_inference_steps=10,
            consensus_projection_k=3, consensus_projection_step=5)
    with pytest.raises(ValueError, match=r"\[1, num_inference_steps\)"):
        RolloutConfig(
            pnp_steps=(1,), refine=True, num_inference_steps=10,
            consensus_projection_k=3, consensus_projection_step=10)


def test_projection_fields_do_not_change_historical_config_hash_material():
    historical = RolloutConfig(pnp_steps=(1,), refine=True)
    logical = historical.logical_dict()
    assert "consensus_projection_k" not in logical
    assert "consensus_projection_step" not in logical


def test_projection_averages_arm_and_preserves_stock_gripper_on_disagreement():
    config = build_smolvla_consensus_projection_methods()[0][1]
    tap = RolloutTap(config, SimpleNamespace(), device="cpu", adim=7)
    tap._projection_noise = lambda value: torch.zeros_like(value)
    tap._record_probe = lambda *args, **kwargs: None
    stock = torch.zeros((1, 2, 8))
    refined = torch.full((1, 2, 8), 2.0)
    stock[..., 6] = -1.0
    refined[..., 6] = 1.0
    ctx = SimpleNamespace(num_steps=10, step=None, records=[])

    def fake_probe(x_t, s, vfield, **kwargs):
        return SimpleNamespace(x_acc=x_t)

    with patch("pnp.tap.run_probe", side_effect=fake_probe):
        projected = tap.project_consensus_action(
            stock, refined, lambda value, s: torch.zeros_like(value), ctx)

    # s=0.5 and zero projection noise leave half of the clean consensus before a zero field.
    assert torch.allclose(projected[..., :6], torch.full_like(projected[..., :6], 0.5))
    assert torch.allclose(projected[..., 6], torch.full_like(projected[..., 6], -0.5))
    assert ctx.step == 5


def test_consensus_projection_notebook_contract():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "96_smolvla_consensus_projection_k3_k5_eval.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_consensus_projection_eval_worker" in source
    assert "'identities': 400" in source
    assert "'new_rollouts': 800" in source
    assert "average + project K=3" in source
    assert "average + project K=5" in source
    assert "'projection_s': 0.5" in source
    assert "'n_action_steps': 10" in source
    assert "'video': 'off'" in source
