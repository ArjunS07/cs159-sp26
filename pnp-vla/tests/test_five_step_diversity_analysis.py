import ast
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from analysis.five_step_diversity import (
    ARMS, KEYS, analyze_five_step, candidate_tables, diversity_summary, effect_table,
    fetch_five_step_rows, match_cohort, success_table)
from pnp.five_step_diversity_experiment import (
    FIVE_STEP_DIVERSITY_EXPERIMENT, build_five_step_diversity_methods, identity_manifest_hash)
from pnp.uncertainty_gradient_experiment import (
    DIRECT_U20_GRADIENT_EXPERIMENT, build_direct_u20_gradient_methods)
from pnp.store import SupabaseStore


ROOT = Path(__file__).parents[1]
SOURCE = "source@revision"


def _trace(count, offset=0):
    trace = {key: [] for key in (
        "chosen", "u", "candidate_profiles", "action_disagreement",
        "executed_prefix_disagreement", "inference_ms", "n_vf_evals", "selected_refinement")}
    for chunk in range(count):
        trace["chosen"].append(1)
        trace["u"].append([.3, .1, .2])
        trace["candidate_profiles"].append([
            {"u10": u / 2, "u20": u, "u_full": u * 2} for u in (.3, .1, .2)])
        for horizon, name in ((50, "action_disagreement"), (10, "executed_prefix_disagreement")):
            trace[name].append({
                "actions_compared": horizon,
                "pairs": [{
                    "left": a, "right": b,
                    "action_cosine": .99 - .03 * index - offset - chunk * .005,
                    "action_l2_mean": .01 + index * .02 + offset,
                    "gripper_sign_disagreement": .1 * index,
                } for index, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)))]})
        trace["inference_ms"].append(100)
        trace["n_vf_evals"].append(60)
        trace["selected_refinement"].append({
            "pre_u": .1, "refined_path_u": .08, "delta_u": -.02, "lowered_u": True,
            "selected_prefix_movement": {"action_l2_mean": .04 + offset}})
    return trace


def synthetic_rows(manifest=None):
    if manifest is None:
        manifest = pd.DataFrame([
            {"suite": f"suite_{index // 2}", "task_idx": index % 2,
             "episode_idx": 10, "init_state_hash": f"h{index}"} for index in range(4)])
    outcomes = {"baseline": [False, False, True, True], "single": [False, True, True, True],
                "select": [True, True, False, True], "refine": [True, False, False, True]}
    configs = [build_direct_u20_gradient_methods()[0], *build_five_step_diversity_methods(SOURCE)]
    rows = []
    for arm, (method, config) in zip(ARMS, configs):
        for index, identity in enumerate(manifest.to_dict("records")):
            count = 1 + index % 3
            rows.append({
                **identity, "arm": arm, "rollout_id": f"{arm}-{index}",
                "run_id": "baseline-run" if arm == "baseline" else "new-run",
                "experiment": (DIRECT_U20_GRADIENT_EXPERIMENT if arm == "baseline"
                               else FIVE_STEP_DIVERSITY_EXPERIMENT),
                "method": method, "status": "completed", "success": outcomes[arm][index % 4],
                "config_hash": SupabaseStore.config_hash(
                    SupabaseStore._logical_key(method, config)),
                "config_json": config.logical_dict(), "max_steps": 500,
                "n_chunks": count, "n_steps": 10 * count, "chunk_size": 50,
                "episode_seed": 100 + index, "inference_ms_total": 100 * count,
                "n_vf_evals": 60 * count,
                "ms_candidate_u": _trace(count, .02 * (index % 4))
                if arm in ("select", "refine") else None})
    return pd.DataFrame(rows), manifest


def test_matched_sr_and_paired_flips_use_one_identity_denominator():
    rows, manifest = synthetic_rows()
    paired, matched, coverage = match_cohort(rows, manifest, require_complete=True)
    assert len(paired) == 4 and len(matched) == 16
    assert coverage.matched_identities_used.eq(4).all()
    success = success_table(paired).set_index("arm")
    assert success.loc["baseline", "sr_pct"] == 50
    assert success.loc["select", "sr_pct"] == 75
    effects = effect_table(paired, n_boot=100)
    result = effects[effects.reference.eq("baseline") & effects.arm.eq("select")].iloc[0]
    assert result.delta_pp == 25 and result.F_to_S == 2 and result.S_to_F == 1


def test_preview_excludes_incomplete_identity_from_every_arm():
    rows, manifest = synthetic_rows()
    rows.loc[rows.arm.eq("refine") & rows.rollout_id.eq("refine-3"), "status"] = "error"
    paired, matched, coverage = match_cohort(rows, manifest)
    assert len(paired) == 3 and len(matched) == 12
    assert coverage.noncompleted_rows.sum() == 1
    with pytest.raises(ValueError, match="final analysis requires 4"):
        match_cohort(rows, manifest, require_complete=True)


@pytest.mark.parametrize("kind", ["duplicate", "outside", "seed", "horizon", "success"])
def test_invalid_matching_is_not_silently_dropped(kind):
    rows, manifest = synthetic_rows()
    if kind == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    elif kind == "outside":
        rows.loc[0, "init_state_hash"] = "wrong"
    elif kind == "seed":
        rows.loc[0, "episode_seed"] = -1
    elif kind == "horizon":
        rows.loc[0, "max_steps"] = 999
    else:
        rows["success"] = rows.success.astype(object)
        rows.loc[0, "success"] = None
    with pytest.raises(ValueError):
        match_cohort(rows, manifest)


def test_all_three_pairs_both_horizons_and_unrefined_u_profiles_are_retained():
    rows, manifest = synthetic_rows()
    paired, matched, _ = match_cohort(rows, manifest)
    tables = analyze_five_step(paired, matched, n_boot=100)
    pairs, candidates, boundaries = (
        tables["candidate_pairs"], tables["candidates"], tables["boundaries"])
    assert set(pairs.pair) == {"0-1", "0-2", "1-2"}
    assert set(pairs.horizon) == {10, 50}
    assert len(pairs) == 6 * len(boundaries) and len(candidates) == 3 * len(boundaries)
    np.testing.assert_allclose(pairs.cosine_distance, 1 - pairs.cosine_similarity)
    assert candidates[candidates.selected].u20.eq(.1).all()
    assert tables["refinement_by_episode"].refined_path_u20.eq(.08).all()
    assert tables["candidate_uncertainty"].u50.notna().all()


@pytest.mark.parametrize("kind", ["length", "pair", "choice", "u20"])
def test_malformed_candidate_telemetry_is_not_zip_truncated(kind):
    rows, manifest = synthetic_rows()
    _, matched, _ = match_cohort(rows, manifest)
    index = matched.index[matched.arm.eq("select")][0]
    trace = copy.deepcopy(matched.loc[index, "ms_candidate_u"])
    if kind == "length":
        trace["u"] = []
    elif kind == "pair":
        trace["action_disagreement"][0]["pairs"].pop()
    elif kind == "choice":
        trace["chosen"][0] = 0
    else:
        trace["candidate_profiles"][0][0]["u20"] = 123
    matched.at[index, "ms_candidate_u"] = trace
    with pytest.raises(ValueError):
        candidate_tables(matched)


def test_diversity_summary_weights_episodes_not_duration_and_keeps_missing_cosines():
    pairs = pd.DataFrame([
        {"arm": "select", "horizon": 10, "pair": "0-1", "rollout_id": rollout,
         "chunk_idx": index, "cosine_similarity": cosine,
         "cosine_distance": 1 - cosine, "action_l2_mean": 1., "gripper_disagreement": 0.}
        for rollout, count, cosine in (("short", 1, 0.), ("long", 9, 1.))
        for index in range(count)])
    summary = diversity_summary(pairs).iloc[0]
    assert summary.cosine_similarity == .5
    assert summary.identities == 2 and summary.boundaries == 10
    pairs["cosine_similarity"] = np.nan
    pairs["cosine_distance"] = np.nan
    summary = diversity_summary(pairs).iloc[0]
    assert np.isnan(summary.cosine_similarity) and summary.finite_cosines == 0


class _Query:
    def __init__(self):
        self.filters = []

    def eq(self, key, value):
        self.filters.append((key, value))
        return self


def _fake_store():
    manifest = pd.DataFrame([
        {"suite": f"suite_{suite}", "task_idx": task, "episode_idx": episode,
         "init_state_hash": f"{suite}-{task}-{episode}"}
        for suite in range(11) for task in range(10) for episode in (10, 11)])
    rows, _ = synthetic_rows(manifest)
    config = {"frozen_identity_manifest": manifest.to_dict("records"),
              "frozen_identity_manifest_hash": identity_manifest_hash(manifest.to_dict("records")),
              "source_model_repo_id": "source", "source_model_revision": "revision"}
    runs = [{"run_id": run_id, "experiment": experiment, "config_json": config,
             "model_repo_id": "source", "model_revision": "revision"}
            for run_id, experiment in (("new-run", FIVE_STEP_DIVERSITY_EXPERIMENT),
                                       ("baseline-run", DIRECT_U20_GRADIENT_EXPERIMENT))]
    store = SupabaseStore.__new__(SupabaseStore)
    data = {"rollouts": rows.to_dict("records"), "experiment_runs": runs}

    def fetch(table, columns, *, configure=None, **kwargs):
        query = _Query()
        if configure:
            configure(query)
        return [row for row in data[table]
                if all(row.get(key) == value for key, value in query.filters)]

    store.fetch_all = fetch
    return store, data


def test_fetch_binds_frozen_manifest_model_revision_and_exact_configs():
    store, _ = _fake_store()
    rows, manifest, provenance = fetch_five_step_rows(store)
    assert len(rows) == 880 and len(manifest) == 220
    assert provenance["source_id"] == SOURCE
    assert set(rows.arm) == set(ARMS)


@pytest.mark.parametrize("kind", ["run_id", "revision", "manifest", "config_hash"])
def test_fetch_rejects_unproven_history_or_mixed_collection(kind):
    store, data = _fake_store()
    if kind == "run_id":
        data["rollouts"][0]["run_id"] = None
    elif kind == "revision":
        data["experiment_runs"][1]["model_revision"] = "wrong-revision"
    elif kind == "manifest":
        data["experiment_runs"][0]["config_json"]["frozen_identity_manifest_hash"] = "wrong"
    else:
        next(row for row in data["rollouts"] if row["arm"] == "select")["config_hash"] = "wrong"
    with pytest.raises(ValueError):
        fetch_five_step_rows(store)


def test_generated_notebook_executes_all_analysis_cells_on_synthetic_data(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import analysis.five_step_diversity as analysis

    path = ROOT / "notebooks" / "61_analyze_five_step_diversity_pro220.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    rows, manifest = synthetic_rows()
    monkeypatch.setattr(analysis, "fetch_five_step_rows",
                        lambda *args, **kwargs: (rows, manifest, {"source_id": SOURCE}))
    monkeypatch.setattr("pnp.store.SupabaseStore", Mock(return_value=Mock()))
    monkeypatch.chdir(tmp_path)
    # The local test runtime need not install Jupyter; only stub rich display, not analysis.
    namespace = {"display": lambda *args, **kwargs: None, "Image": lambda **kwargs: kwargs}
    assert "accelerator" not in notebook["metadata"]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        source = "".join(cell["source"])
        ast.parse(source, filename=f"cell-{index}")
        if "urllib.request" not in source:
            source = source.replace("from IPython.display import display, Image", "")
            exec(compile(source, f"cell-{index}", "exec"), namespace)
    figures = list((tmp_path / "five_step_diversity_pro220_outputs" / "figures").glob("*.png"))
    assert len(figures) == 6
    assert all(path.stat().st_size > 10000 for path in figures)
