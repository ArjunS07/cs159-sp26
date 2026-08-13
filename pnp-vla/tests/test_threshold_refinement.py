import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pnp.config import Method, RolloutConfig
from pnp.experiments import format_progress_table
from pnp.tap import RolloutTap


ROOT = Path(__file__).parents[1]


def test_threshold_refinement_config_is_explicit_and_preserves_old_logical_keys():
    old = RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine=True)
    gated = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True, refine_threshold=0.03)

    assert "refine_threshold" not in old.logical_dict()
    assert gated.logical_dict()["refine_threshold"] == pytest.approx(0.03)
    assert old.logical_dict() != gated.logical_dict()
    with pytest.raises(ValueError, match="requires refine=True"):
        RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine_threshold=0.03)


def test_action_execution_horizon_is_explicit_and_hashed():
    original = RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine=True)
    horizon_20 = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True, n_action_steps=20)
    assert "n_action_steps" not in original.logical_dict()
    assert horizon_20.logical_dict()["n_action_steps"] == 20
    assert original.logical_dict() != horizon_20.logical_dict()
    with pytest.raises(ValueError, match="positive integer"):
        RolloutConfig(n_action_steps=0)


def test_threshold_refinement_applies_existing_last_refine_only_above_gate():
    config = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True, refine_threshold=0.03)
    recorder = SimpleNamespace()
    tap = RolloutTap(config, recorder, device=None, adim=7)
    assert tap.needs_baseline_fallback
    tap.begin_chunk()
    ctx = SimpleNamespace(step=3, records=[])
    x_t = object()
    refined = object()

    low_probe = SimpleNamespace(rec={"u_mean": 0.029})
    high_probe = SimpleNamespace(rec={"u_mean": 0.031})
    with patch("pnp.tap.run_probe", side_effect=[low_probe, high_probe]), patch(
            "pnp.tap.apply_refine", return_value=refined) as apply_refine:
        assert tap.step(x_t, 0.7, None, ctx) is x_t
        assert tap.step(x_t, 0.6, None, ctx) is refined

    apply_refine.assert_called_once_with(high_probe, False)
    assert tap.chunk_intervened
    assert tap.refinement_gate_telemetry == {
        "n_corrections_applied": 1,
        "gate_fire_rate": 0.5,
    }
    tap.begin_chunk()
    assert not tap.chunk_intervened


def test_delayed_refinement_preserves_five_stock_chunks_then_refines():
    config = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True, refine_start_chunk=5)
    assert config.logical_dict()["refine_start_chunk"] == 5
    assert "refine_start_chunk" not in RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True).logical_dict()
    with pytest.raises(ValueError, match="requires refine=True"):
        RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine_start_chunk=5)

    tap = RolloutTap(config, SimpleNamespace(), device=None, adim=7)
    assert tap.needs_baseline_fallback
    ctx = SimpleNamespace(step=3, records=[])
    x_t = object()
    refined = object()
    probes = [SimpleNamespace(rec={"u_mean": 0.04}) for _ in range(6)]
    with patch("pnp.tap.run_probe", side_effect=probes), patch(
            "pnp.tap.apply_refine", return_value=refined) as apply_refine:
        for _ in range(5):
            tap.begin_chunk()
            assert tap.step(x_t, 0.7, None, ctx) is x_t
            assert not tap.chunk_intervened
        tap.begin_chunk()
        assert tap.step(x_t, 0.7, None, ctx) is refined
        assert tap.chunk_intervened

    apply_refine.assert_called_once_with(probes[-1], False)


def test_progress_table_can_use_exact_historical_unrefined_rates():
    table = format_progress_table(
        {("libero_goal_swap", Method.THRESHOLD_REFINEMENT): [4, 1]},
        [Method.THRESHOLD_REFINEMENT], historical_sr={"libero_goal_swap": 0.5})
    assert "U-gated refine" in table
    assert "25% (1/4)" in table
    assert "50%" in table


def test_source_threshold_workers_are_four_fixed_shards_and_one_arm():
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"24_source_threshold_refinement_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_threshold_refinement_worker(" in source
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "THRESHOLD = DIVERSITY_FIXED_REFINEMENT_THRESHOLD" in source
        assert "SOURCE_MODEL_REVISION = manifest[\"source_model_revision\"]" in source
        assert "run_diversity_chunk_selector_worker" not in source


def test_source_multi_query_workers_default_to_two_and_keep_count_explicit():
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"26_source_multi_query_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_multi_query_worker(" in source
        assert "N_QUERIES = 2" in source
        assert "num_queries=N_QUERIES" in source
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "run_diversity_chunk_selector_worker" not in source


def test_source_delayed_refinement_workers_are_four_fixed_shards():
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"29_source_delayed_refinement_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_delayed_refinement_worker(" in source
        assert "EPISODES_PER_TASK = 10" in source
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "REFINE_START_CHUNK = 5" in source
        assert "source_model_revision=SOURCE_MODEL_REVISION" in source


def test_delayed_refinement_analysis_uses_whole_exact_matched_cohort():
    path = ROOT / "notebooks" / "30_analyze_source_delayed_refinement.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "SOURCE_DELAYED_REFINEMENT_EXPERIMENT" in source
    assert '"method", Method.DELAYED_REFINEMENT' in source
    assert "REFINE_START_CHUNK = 5" in source
    assert "EXPECTED_EPISODES = 1300" in source
    assert 'on=DIVERSITY_PAIR_KEYS, validate="one_to_one"' in source
    assert "delayed_minus_source_pp" in source
    assert "paired_bootstrap_ci" in source
    assert "failure_to_success" in source
    assert "success_to_failure" in source


def test_action_horizon_workers_default_to_refinement_on_two_shards():
    for shard_index in range(2):
        path = ROOT / "notebooks" / "workers" / (
            f"31_source_action_horizon_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_action_horizon_worker(" in source
        assert 'ARM = "refinement"' in source
        assert "N_ACTION_STEPS = 20" in source
        assert "EPISODES_PER_TASK = 1" in source
        assert "SHARD_COUNT = 2" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "arm=ARM, n_action_steps=N_ACTION_STEPS" in source
