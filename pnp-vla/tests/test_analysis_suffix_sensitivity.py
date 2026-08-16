import io
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.suffix_sensitivity import (
    ARMS, apply_window_by_suite, decode_artifact, failure_auc_table,
    pair_arms, sensitivity_quartiles, summarize_pair, top_windows,
    threshold_sweep, validate_cohort, window_sweep,
)
from pnp.diversity import DIVERSITY_PAIR_KEYS


ROOT = Path(__file__).parents[1]


def _rows(n=20):
    rows = []
    for index in range(n):
        identity = {
            "suite": f"suite_{index % 2}", "task_idx": index % 10,
            "episode_idx": index, "init_state_hash": f"h{index}",
        }
        for arm_index, method in enumerate(ARMS):
            rows.append({
                **identity, "rollout_id": f"r{index}_{arm_index}", "method": method,
                "status": "completed", "success": bool((index + arm_index) % 4 == 0),
            })
    return pd.DataFrame(rows)


def _artifact():
    arrays = {}
    for chunk in range(2):
        for step in (3, 4):
            stem = f"c{chunk}_s{step}"
            arrays[f"{stem}_u_time"] = np.linspace(.01, .06, 50) + chunk * .001
            reference = np.zeros((1, 10, 7), dtype=np.float32)
            predictions = np.ones((4, 1, 10, 7), dtype=np.float32) * .2
            arrays[f"{stem}_suffix_prefix_predictions"] = predictions
            arrays[f"{stem}_suffix_prefix_reference"] = reference
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def test_validate_and_pair_exact_three_arm_cohort():
    rows = _rows()
    arms = validate_cohort(rows, expected_identities=20)
    assert list(arms) == list(ARMS)
    pair = pair_arms(arms[ARMS[0]], arms[ARMS[1]])
    assert len(pair) == 20
    overall, by_suite = summarize_pair(pair)
    assert int(overall.episodes.iloc[0]) == 20
    assert by_suite.episodes.sum() == 20

    with pytest.raises(ValueError, match="identical identity sets"):
        validate_cohort(rows.iloc[:-1], expected_identities=20)


def test_decode_artifact_recovers_temporal_and_tail_features():
    records, uncertainty, sensitivity = decode_artifact(_artifact(), rollout_id="r0")
    assert len(records) == 4
    assert len(uncertainty) == 4 * 50
    assert len(sensitivity) == 4 * 10
    assert records.u_first10.lt(records.u_full50).all()
    assert records.tail_to_prefix_l2.iloc[0] == pytest.approx(np.sqrt(7) * .2)
    assert records.tail_to_prefix_abs.iloc[0] == pytest.approx(.2)


def test_window_delta_uses_every_pair_in_denominator():
    pair = pair_arms(
        validate_cohort(_rows(), expected_identities=20)[ARMS[0]],
        validate_cohort(_rows(), expected_identities=20)[ARMS[1]])
    pair["u_first10_episode"] = np.linspace(.01, .05, len(pair))
    sweep = window_sweep(
        pair, score_column="u_first10_episode", grid_size=5, min_selected=1)
    assert sweep.episodes_in_sr_denominator.eq(20).all()
    best = top_windows(sweep, n=1).iloc[0]
    selected = pair.u_first10_episode.between(best.lower, best.upper)
    expected = np.where(selected, pair.condition_success, pair.baseline_success).mean()
    assert best.window_policy_sr == pytest.approx(expected)
    assert best.delta_pp == pytest.approx(
        100 * (expected - pair.baseline_success.mean()))
    by_suite = apply_window_by_suite(
        pair, score_column="u_first10_episode", lower=best.lower, upper=best.upper)
    assert by_suite.episodes.sum() == 20

    thresholds = threshold_sweep(
        pair, score_column="u_first10_episode", grid_size=5, min_selected=1)
    assert thresholds.episodes_in_sr_denominator.eq(20).all()
    row = thresholds[thresholds.eligible].sort_values("delta_pp", ascending=False).iloc[0]
    selected = pair.u_first10_episode.ge(row.threshold)
    expected = np.where(selected, pair.condition_success, pair.baseline_success).mean()
    assert row.threshold_policy_sr == pytest.approx(expected)


def test_failure_auc_and_sensitivity_quartiles_are_readable():
    features = pd.DataFrame({
        "suite": ["a"] * 10 + ["b"] * 10,
        "success": [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5,
        "u_first10_episode": [.01] * 5 + [.05] * 5 + [.01] * 5 + [.05] * 5,
    })
    auc = failure_auc_table(
        features, score_columns=["u_first10_episode"], n_boot=50)
    assert auc.query("suite == 'pooled'").failure_auc.iloc[0] == 1.0

    rows = _rows()
    arms = validate_cohort(rows, expected_identities=20)
    full = pair_arms(arms[ARMS[0]], arms[ARMS[1]])
    tapered = pair_arms(arms[ARMS[0]], arms[ARMS[2]])
    feature_values = pd.DataFrame({
        **{key: full[key] for key in DIVERSITY_PAIR_KEYS},
        "tail_to_prefix_l2_episode": np.arange(len(full), dtype=float),
    })
    full = full.merge(feature_values, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    tapered = tapered.merge(feature_values, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    quartiles = sensitivity_quartiles(full, tapered)
    assert quartiles.episodes.sum() == 20
    assert quartiles.sensitivity_quartile.tolist() == [1, 2, 3, 4]


def test_analysis_notebook_is_clean_and_contains_primary_sections():
    path = ROOT / "notebooks" / "36_analyze_suffix_sensitivity.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
    assert "REQUIRE_FULL_COHORT = True" in source
    assert "EXPECTED_IDENTITIES = 220" in source
    assert "uncertainty_window_sweep_heatmaps.png" in source
    assert "uncertainty_failure_auc_and_roc.png" in source
    assert "tail_sensitivity_analysis.png" in source
    assert "episodes_in_sr_denominator" in source
