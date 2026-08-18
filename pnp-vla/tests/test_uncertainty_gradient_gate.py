import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from analysis.uncertainty_gradient_gate import _parse_gate
from pnp.config import Method, RolloutConfig
from pnp.tap import RolloutTap
from pnp.uncertainty_gradient_gate_experiment import (
    U20_ACTION_GATE_EXPERIMENT,
    U20_ACTION_GATE_THRESHOLDS,
    build_u20_action_gate_methods,
)


ROOT = Path(__file__).parents[1]


def _config(threshold=.015):
    return RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, n_action_steps=10,
        uncertainty_gradient_mode="descent",
        uncertainty_gradient_step_size=.01,
        uncertainty_gradient_horizon=20,
        uncertainty_gradient_action_rms_max=threshold)


def test_action_gate_is_validated_and_behavior_hashed():
    config = _config()
    assert config.logical_dict()["uncertainty_gradient_action_rms_max"] == .015
    assert "uncertainty_gradient_action_rms_max" not in RolloutConfig().logical_dict()
    with pytest.raises(ValueError, match="requires descent mode"):
        RolloutConfig(
            pnp_steps=(3, 4), n_action_steps=10,
            uncertainty_gradient_mode="random",
            uncertainty_gradient_step_size=.01,
            uncertainty_gradient_action_rms_max=.015)
    with pytest.raises(ValueError, match="explicit n_action_steps"):
        RolloutConfig(
            pnp_steps=(3, 4),
            uncertainty_gradient_mode="descent",
            uncertainty_gradient_step_size=.01,
            uncertainty_gradient_action_rms_max=.015)


def test_gate_measures_postprocessed_first_ten_arm_actions_and_returns_exact_chunk():
    def postprocess(chunk):
        decoded = chunk[..., :7].clone()
        decoded[..., :6] *= 2
        return decoded

    tap = RolloutTap(
        _config(.015), SimpleNamespace(), device=None, adim=7,
        action_postprocess=postprocess)
    tap.begin_chunk()
    stock = torch.zeros(1, 50, 32)
    small = stock.clone()
    small[..., :10, :6] = .005  # decoded RMS .010: accept
    large = stock.clone()
    large[..., :10, :6] = .010  # decoded RMS .020: reject

    assert tap.needs_baseline_fallback
    assert tap.finalize_action(stock, small) is small
    assert tap.finalize_action(stock, large) is stock
    first, second = tap._gradient_action_gate_records
    assert first["action_rms"] == pytest.approx(.010)
    assert first["accepted"] is True
    assert second["action_rms"] == pytest.approx(.020)
    assert second["accepted"] is False
    assert second["actions_compared"] == 10
    assert second["motion_dimensions"] == 6


def test_existing_refinement_fallback_keeps_exact_stock_when_no_gate_fire():
    config = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True, refine_threshold=.03)
    tap = RolloutTap(config, SimpleNamespace(), device=None, adim=7)
    stock, candidate = object(), object()
    assert tap.finalize_action(stock, candidate) is stock
    tap._chunk_refine_applied = 1
    assert tap.finalize_action(stock, candidate) is candidate


def test_gate_telemetry_parser_summarizes_chunk_decisions():
    parsed = _parse_gate({"uncertainty_gradient": {"action_gate_records": [
        {"action_rms": .01, "first_action_l2": .02, "accepted": True},
        {"action_rms": .03, "first_action_l2": .04, "accepted": False,
         "gripper_sign_disagreement": True},
    ]}})
    assert parsed["gate_decisions"] == 2
    assert parsed["gate_accepted"] == 1
    assert parsed["gate_accept_rate"] == .5
    assert parsed["mean_action_rms"] == pytest.approx(.02)
    assert parsed["gripper_disagreement_rate"] == .5


def test_two_predeclared_arms_and_four_colab_workers():
    methods = build_u20_action_gate_methods()
    assert [method for method, _ in methods] == [
        Method.U20_GRADIENT_GATE_015, Method.U20_GRADIENT_GATE_020]
    assert [config.uncertainty_gradient_action_rms_max for _, config in methods] == [
        .015, .020]
    assert U20_ACTION_GATE_THRESHOLDS == (.015, .020)
    assert U20_ACTION_GATE_EXPERIMENT == "pi05-u20-gradient-action-gate-pro220-v1"

    workers = sorted((ROOT / "notebooks" / "workers").glob(
        "50_u20_action_gate_pro220_worker_*.ipynb"))
    assert len(workers) == 4
    for index, path in enumerate(workers):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert f"SHARD_INDEX = {index}" in source
        assert "SHARD_COUNT = 4" in source
        assert "run_u20_action_gate_worker(" in source
        assert "'benchmark': 'LIBERO-PRO'" in source


def test_analysis_notebook_uses_strict_full_matched_cohort():
    path = ROOT / "notebooks" / "51_analyze_u20_action_gate_pro220.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "EXPECTED_IDENTITIES = 220" in source
    assert "REQUIRE_FULL_COHORT = True" in source
    assert "match_action_gate_cohort(" in source
    assert "condition_minus_reference_pp" in source
