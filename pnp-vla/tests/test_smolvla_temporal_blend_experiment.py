import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.smolvla_temporal_blend_experiment import build_smolvla_temporal_blend_methods
from pnp.tap import BatchedRolloutTap, RolloutTap


def test_temporal_worker_has_stock_and_refined_parent_arms():
    methods = build_smolvla_temporal_blend_methods()
    assert [name for name, _ in methods] == [
        Method.SMOLVLA_TEMPORAL_STOCK_PROJECT_K3,
        Method.SMOLVLA_TEMPORAL_REFINED_PROJECT_K3,
    ]
    stock, refined = (config for _, config in methods)
    assert stock.temporal_overlap_consensus and not stock.refine
    assert refined.temporal_overlap_consensus and refined.refine
    assert tuple(refined.pnp_steps) == (1, 2, 3)
    assert tuple(refined.pnp_k_by_step) == (3, 1, 1)
    for config in (stock, refined):
        assert config.n_action_steps == 10
        assert config.num_inference_steps == 10
        assert config.consensus_projection_step == 5
        assert config.consensus_projection_k == 3


def test_temporal_overlap_requires_execution_horizon_and_projection():
    with pytest.raises(ValueError, match="explicit n_action_steps"):
        RolloutConfig(temporal_overlap_consensus=True)
    with pytest.raises(ValueError, match="requires projection settings"):
        RolloutConfig(temporal_overlap_consensus=True, n_action_steps=10)


def test_temporal_merge_uses_previous_projected_suffix_and_preserves_current_tail():
    config = build_smolvla_temporal_blend_methods()[0][1]
    tap = RolloutTap(config, SimpleNamespace(), device="cpu", adim=2)
    previous_projected = torch.arange(50.0).reshape(1, 50, 1).repeat(1, 1, 2)
    current_parent = torch.full((1, 50, 2), 100.0)
    current_parent[:, 40:] = 200.0
    tap.store_temporal_projected_action(previous_projected)
    merged = tap.temporal_consensus_action(current_parent)
    expected_prefix = 0.5 * (previous_projected[:, 10:50] + 100.0)
    assert torch.equal(merged[:, :40], expected_prefix)
    assert torch.equal(merged[:, 40:], current_parent[:, 40:])


def test_first_temporal_decision_uses_current_parent_and_stores_only_projected_result():
    config = build_smolvla_temporal_blend_methods()[0][1]
    tap = RolloutTap(config, SimpleNamespace(), device="cpu", adim=2)
    parent = torch.ones((1, 50, 2))
    assert torch.equal(tap.temporal_consensus_action(parent), parent)
    assert tap._temporal_previous_projected is None
    projected = torch.full_like(parent, 3.0)
    tap.store_temporal_projected_action(projected)
    assert torch.equal(tap._temporal_previous_projected, projected)


def test_batched_temporal_state_round_trip_is_lane_aligned():
    config = build_smolvla_temporal_blend_methods()[0][1]
    previous = [torch.zeros((1, 50, 2)), torch.ones((1, 50, 2))]
    tap = BatchedRolloutTap(
        config, [SimpleNamespace(), SimpleNamespace()], [1, 2], "cpu", 2,
        previous_projected=previous)
    current = torch.full((2, 50, 2), 4.0)
    merged = tap.temporal_consensus_action(current)
    assert torch.equal(merged[0, :40], torch.full((40, 2), 2.0))
    assert torch.equal(merged[1, :40], torch.full((40, 2), 2.5))
    projected = torch.stack([torch.full((50, 2), 6.0), torch.full((50, 2), 7.0)])
    tap.store_temporal_projected_action(projected)
    assert tap.temporal_projected_actions[0].shape == (1, 50, 2)
    assert torch.equal(tap.temporal_projected_actions[1], projected[1:2])


def test_temporal_worker_notebook_contract():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "104_smolvla_a10_temporal_overlap_blend_eval.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_temporal_blend_eval_worker" in source
    assert "'new_rollouts': 800" in source
    assert "previous[10:50] with current[0:40]" in source
    assert "'n_action_steps': 10" in source
    assert "'video': 'off'" in source
