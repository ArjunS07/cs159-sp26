import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.horizon_diagnostics import (
    decode_horizon_artifact, failure_auc_table, pair_diagnostics_with_historical,
    prefix_feature_table, prefix_failure_auc_table, quantile_outcome_curve,
    validate_diagnostic_cohort)
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS


ROOT = Path(__file__).parents[1]


def _artifact(include_iter=True):
    buffer = io.BytesIO()
    arrays = {"c0_s3_u_time": np.arange(50, dtype=np.float32)}
    if include_iter:
        arrays["c0_s3_u_iter_time"] = np.stack([
            np.full(50, value, dtype=np.float32) for value in (4, 3, 2, 1)])
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _identity(suite, episode):
    return {
        "suite": suite, "task_idx": 0, "episode_idx": episode,
        "init_state_hash": f"h{episode}",
    }


def test_decode_horizon_artifact_keeps_u_and_contraction_horizons():
    records, positions, iterations = decode_horizon_artifact(
        _artifact(), rollout_id="rid")
    row = records.iloc[0]
    assert row.u10 == pytest.approx(4.5)
    assert row.u20 == pytest.approx(9.5)
    assert row.u50 == pytest.approx(24.5)
    assert row.contraction10 == pytest.approx(3.0)
    assert row.contraction20 == pytest.approx(3.0)
    assert row.contraction50 == pytest.approx(3.0)
    assert row.contraction_fraction20 == pytest.approx(0.75)
    assert len(positions) == 50
    assert set(iterations.horizon) == {10, 20, 50}
    assert len(iterations) == 12


def test_decode_requires_consecutive_disagreement_profile():
    with pytest.raises(ValueError, match="missing"):
        decode_horizon_artifact(_artifact(include_iter=False), rollout_id="rid")


def test_validate_diagnostic_cohort_filters_and_requires_exact_coverage():
    rows = pd.DataFrame([
        {**_identity("suite", 0), "rollout_id": "a", "status": "completed",
         "success": True, "method": Method.UNCERTAINTY, "ahats_path": "a.npz"},
        {**_identity("suite", 1), "rollout_id": "b", "status": "completed",
         "success": False, "method": Method.UNCERTAINTY, "ahats_path": "b.npz"},
        {**_identity("suite", 2), "rollout_id": "c", "status": "completed",
         "success": False, "method": Method.CHUNK_SOURCE_MULTI_QUERY,
         "ahats_path": None},
    ])
    cohort = validate_diagnostic_cohort(rows, expected_identities=2)
    assert cohort.rollout_id.tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="expected 3"):
        validate_diagnostic_cohort(rows, expected_identities=3)


def test_prefix_features_keep_every_episode_when_one_ends_early():
    records = pd.DataFrame([
        {"rollout_id": rid, "chunk_idx": chunk,
         **{f"{prefix}{horizon}": value
            for prefix in ("u", "contraction", "contraction_fraction")
            for horizon in (10, 20, 50)}}
        for rid, chunk, value in (("a", 0, .1), ("b", 0, .2), ("b", 1, .4))])
    features = pd.DataFrame([
        {"rollout_id": "a", "suite": "suite", "success": True, "n_chunks": 1},
        {"rollout_id": "b", "suite": "suite", "success": False, "n_chunks": 2},
    ])
    prefix = prefix_feature_table(records, features, max_chunks=3)
    assert prefix.groupby("first_k_chunks").size().eq(2).all()
    assert prefix[prefix.first_k_chunks.eq(3)].set_index("rollout_id").loc["a", "u20"] == .1
    auc = prefix_failure_auc_table(prefix, n_boot=20)
    assert set(auc.episodes) == {2}


def test_pairing_and_failure_tables_use_exact_identity_keys():
    features = pd.DataFrame([
        {**_identity("suite", index), "rollout_id": f"d{index}",
         "success": bool(index), "n_chunks": 2, "u20_episode": .2 - .1 * index}
        for index in range(2)])
    arms = {
        Method.UNCERTAINTY: pd.DataFrame([
            {**_identity("suite", index), "success": bool(index)} for index in range(2)]),
        Method.REFINEMENT: pd.DataFrame([
            {**_identity("suite", index), "success": True} for index in range(2)]),
    }
    paired = pair_diagnostics_with_historical(features, arms)
    assert len(paired) == 2
    assert paired.diagnostic_matches_historical_baseline.all()
    auc = failure_auc_table(features, ["u20_episode"], n_boot=20)
    assert auc[auc.suite.eq("pooled")].failure_auc.iloc[0] == 1.0
    curve = quantile_outcome_curve(features, score_column="u20_episode", bins=2)
    assert curve.episodes.sum() == 2


def test_worker41_analysis_notebook_has_required_sections():
    notebook = json.loads((ROOT / "notebooks" /
        "42_analyze_source_horizon_diagnostics.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in (
        "SOURCE_HORIZON_MULTI_QUERY_EXPERIMENT", "EXPECTED_IDENTITIES = 1300",
        "load_horizon_artifacts", "failure_auc_table", "prefix_failure_auc_table",
        "contraction", "window_sweep", "SOURCE_ACTION_HORIZON_EXPERIMENT",
        "diagnostic_matches_historical_baseline"):
        assert required in source
