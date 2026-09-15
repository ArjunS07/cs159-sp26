import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pnp.config import ALL_METHODS, RolloutConfig
from pnp.qplanning_u20_gated_gradient_eval_experiment import (
    QGUIDE_U20_GATE_HORIZON, QGUIDE_U20_GATE_METHODS,
    QGUIDE_U20_GATE_PNP_K, QGUIDE_U20_GATE_PROBE_STEPS,
    QGUIDE_U20_GATE_SHARDS, QGUIDE_U20_GATE_THRESHOLD,
    build_qguide_u20_gate_method)
from pnp.pnp import PnPRecorder
from pnp.tap import RolloutTap


ROOT = Path(__file__).parents[1]


def test_gated_qguide_contract_is_distinct_and_resume_safe():
    assert set(QGUIDE_U20_GATE_METHODS.values()).issubset(ALL_METHODS)
    scorer = SimpleNamespace(checkpoint_id="critic-id")
    hashes = []
    for strategy, expected_method in QGUIDE_U20_GATE_METHODS.items():
        method, config = build_qguide_u20_gate_method(
            strategy, "repo@revision", scorer)
        assert method == expected_method
        assert config.num_samples is None
        assert config.q_guidance_gate_threshold == QGUIDE_U20_GATE_THRESHOLD
        assert config.q_guidance_gate_horizon == QGUIDE_U20_GATE_HORIZON
        assert config.q_guidance_gate_pnp_k == QGUIDE_U20_GATE_PNP_K
        assert tuple(config.q_guidance_gate_probe_steps) == QGUIDE_U20_GATE_PROBE_STEPS
        assert config.num_inference_steps == 10
        assert config.n_action_steps == 10
        assert config.video == "off" and not config.save_observations
        logical = config.logical_dict()
        assert logical["q_guidance_gate_threshold"] == 0.0225
        assert logical["q_guidance_gate_horizon"] == 20
        hashes.append((method, logical))
    assert hashes[0] != hashes[1]


def test_partial_qguide_gate_configuration_is_rejected():
    scorer = SimpleNamespace(checkpoint_id="critic-id")
    common = dict(
        q_guidance_ckpt_id="critic-id", q_guidance_step=3,
        q_guidance_step_size=0.005, q_guidance_scorer=scorer,
        num_inference_steps=10)
    with pytest.raises(ValueError, match="requires threshold, horizon"):
        RolloutConfig(**common, q_guidance_gate_threshold=0.0225)


def test_gate_telemetry_counts_rejected_boundaries_without_fake_q_updates():
    scorer = SimpleNamespace(checkpoint_id="critic-id")
    _, config = build_qguide_u20_gate_method(
        "original", "repo@revision", scorer)
    tap = RolloutTap(config, PnPRecorder(), "cpu", adim=7)
    details = {
        "u10": 0.018, "u20": 0.020, "u_full": 0.021,
        "contraction10": 0.0, "contraction20": 0.0,
        "contraction_full": 0.0,
    }
    tap.set_q_guidance_gate(0.020, details, fired=False, chunk_idx=4)
    telemetry = tap.q_guidance_telemetry
    assert telemetry["n_gate_considered"] == 1
    assert telemetry["n_gate_fired"] == 0
    assert telemetry["n_updates"] == 0
    assert telemetry["records"][0]["chunk_idx"] == 4
    assert not telemetry["records"][0]["guidance_applied"]

    tap.set_q_guidance_gate(0.025, details, fired=True, chunk_idx=5)
    tap.begin_chunk()
    assert tap._chunk_idx == 5
    assert tap._q_guidance_gate_pending["gate_fired"]


def test_worker_82_is_four_shards_two_guides_and_historical_stock():
    assert QGUIDE_U20_GATE_SHARDS == 4
    for worker in range(4):
        path = (
            ROOT / "notebooks" / "workers"
            / f"82_eval_q50_u20_gated_gradient_worker_{worker}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert f"SHARD_INDEX = {worker}" in source
        assert "SHARD_COUNT = 4" in source
        assert "EPISODE_LIMIT = None" in source
        assert "new_rollouts': 80" in source
        assert "periodic_print_every_complete_identities': 10" in source
        assert "exact matched notebook-68 outcomes" in source
        assert "U20_GATE_THRESHOLD = 0.0225" in source
        assert "LATENT_UPDATE_RMS = 0.005" in source
        assert "original_q50_step8000.pt" in source
        assert "u20_8chunk_priority_q50_step6000.pt" in source
        assert "run_qplanning_u20_gated_gradient_heldout160(" in source
        assert "validate_qplanning_u20_gated_gradient_sentinel(" in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                ast.parse(
                    "".join(cell["source"]),
                    filename=f"{path.name}:cell{index}")
