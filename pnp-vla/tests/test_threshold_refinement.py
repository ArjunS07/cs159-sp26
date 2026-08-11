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
