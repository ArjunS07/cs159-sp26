import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.smolvla_u20_gated_blend_experiment import SMOLVLA_U20_Q75_BY_SUITE
from pnp.smolvla_u20_gated_refinement_experiment import (
    build_smolvla_u20_gated_refinement_method,
)
from pnp.tap import RolloutTap


def _records(value):
    return [{
        "step": step,
        "u_time": np.full(50, value, np.float32),
    } for step in (1, 2, 3)]


def test_gated_plain_refinement_keeps_notebook_108_gate_contract():
    method, config = build_smolvla_u20_gated_refinement_method("libero_10")
    assert method == Method.SMOLVLA_U20_GATED_REFINEMENT_K311
    assert config.refine is True
    assert config.refine_chunk_gate_threshold == pytest.approx(
        SMOLVLA_U20_Q75_BY_SUITE["libero_10"])
    assert config.refine_chunk_gate_horizon == 20
    assert config.consensus_projection_k is None
    assert config.consensus_average_only is False
    assert tuple(config.pnp_steps) == (1, 2, 3)
    assert tuple(config.pnp_k_by_step) == (3, 1, 1)
    assert config.num_inference_steps == 10
    assert config.n_action_steps == 10


def test_chunk_gate_returns_exact_stock_or_completed_refinement():
    _, config = build_smolvla_u20_gated_refinement_method("libero_10")
    tap = RolloutTap(config, SimpleNamespace(), device=None, adim=7)
    stock = torch.zeros((1, 50, 8))
    refined = torch.ones_like(stock)

    tap.begin_chunk()
    tap._finish_chunk_refinement_gate(_records(0.01))
    assert torch.equal(tap.finalize_action(stock, refined), stock)

    tap.begin_chunk()
    tap._finish_chunk_refinement_gate(_records(0.20))
    assert torch.equal(tap.finalize_action(stock, refined), refined)
    telemetry = tap.refinement_gate_telemetry
    assert telemetry["n_corrections_applied"] == 1
    assert telemetry["gate_fire_rate"] == pytest.approx(0.5)
    assert telemetry["refinement_chunk_gate"]["n_considered"] == 2


def test_chunk_gate_rejects_partial_or_mixed_operator_configuration():
    with pytest.raises(ValueError, match="threshold and horizon"):
        RolloutConfig(
            pnp_steps=(1, 2, 3), pnp_k_by_step=(3, 1, 1), pnp_k=3,
            refine=True, refine_chunk_gate_threshold=0.08)
    with pytest.raises(ValueError, match="cannot also average or project"):
        RolloutConfig(
            pnp_steps=(1, 2, 3), pnp_k_by_step=(3, 1, 1), pnp_k=3,
            refine=True, refine_chunk_gate_threshold=0.08,
            refine_chunk_gate_horizon=20, consensus_average_only=True)


def test_worker_notebook_is_same_gate_with_plain_refinement_only():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "110_smolvla_a10_online_u20_q75_gated_plain_refine_eval.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_u20_gated_refinement_eval_worker" in source
    assert "ROLLOUT_BATCH_SIZE = 1" in source
    assert "'online_per_chunk': True" in source
    assert "'gate_horizon': 20" in source
    assert "'n_action_steps': 10" in source
    assert "'video': 'off'" in source
    assert "projection_s" not in source

