import ast
import copy
import io
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from analysis.coarse_refinement import (
    ARMS, COMPARISONS, HORIZONS, KEYS, NEW_ARMS, PROBE_LABELS, analyze_success,
    effect_table, fetch_coarse_refinement_rows, load_probe_artifacts, match_cohort,
    probe_figures, probe_tables, success_figures, compute_figure)
from pnp.coarse_refinement_experiment import (
    COARSE_REFINEMENT_EXPERIMENT, build_coarse_refinement_methods)
from pnp.five_step_diversity_experiment import identity_manifest_hash
from pnp.store import SupabaseStore


ROOT = Path(__file__).parents[1]
SOURCE = "source@revision"


def synthetic_rows(count=4):
    manifest = pd.DataFrame([
        {"suite": f"suite_{index // 2}", "task_idx": index % 2,
         "episode_idx": 10, "init_state_hash": f"h{index}"} for index in range(count)])
    outcomes = {
        "stock10": [False, False, True, True],
        "five_single": [False, True, True, True],
        "five_select": [True, True, False, True],
        "five_select_refine": [True, False, True, True],
        "five_refine": [False, True, False, True],
        "three_single": [False, False, False, True],
        "three_refine": [True, False, False, True],
    }
    rows = []
    for arm in ARMS:
        for index, identity in enumerate(manifest.to_dict("records")):
            rows.append({
                **identity, "arm": arm, "rollout_id": f"{arm}-{index}",
                "run_id": "new-run" if arm in NEW_ARMS else "old-run",
                "status": "completed", "success": outcomes[arm][index % 4],
                "config_hash": f"hash-{arm}", "max_steps": 300, "episode_seed": 100 + index,
                "chunk_size": 50, "n_chunks": index % 2 + 1,
                "n_steps": 10 * (index % 2 + 1), "elapsed_s": 2.0 * (index % 2 + 1),
                "inference_ms_total": 100.0 * (index % 2 + 1),
                "n_vf_evals": 20 * (index % 2 + 1),
                "ahats_path": f"{arm}-{index}.npz",
            })
    return pd.DataFrame(rows), manifest


def synthetic_probe_records(matched):
    records = []
    for row in matched[matched.arm.isin(NEW_ARMS)].to_dict("records"):
        for step in [value for arm, value in PROBE_LABELS if arm == row["arm"]]:
            for chunk in range(row["n_chunks"]):
                base = .1 + .01 * chunk + (.02 if not row["success"] else 0)
                record = {"rollout_id": row["rollout_id"], "arm": row["arm"],
                          "suite": row["suite"], "success": row["success"],
                          "probe": PROBE_LABELS[(row["arm"], step)],
                          "euler_step": step, "chunk_idx": chunk}
                for horizon in HORIZONS:
                    record.update({
                        f"u{horizon}": base + horizon / 1000,
                        f"start_disagreement{horizon}": base + .03,
                        f"end_disagreement{horizon}": base + .01,
                        f"contraction{horizon}": .02,
                    })
                records.append(record)
    return pd.DataFrame(records)


def test_all_seven_arms_share_one_denominator_and_expected_paired_flips():
    rows, manifest = synthetic_rows()
    paired, matched, coverage = match_cohort(rows, manifest, require_complete=True)
    assert len(paired) == 4 and len(matched) == 28
    assert coverage.matched_identities_used.eq(4).all()
    tables = analyze_success(paired, matched, n_boot=100)
    rates = tables["arm_success_rates"].set_index("arm")
    assert rates.loc["stock10", "sr_pct"] == 50
    assert rates.loc["three_refine", "sr_pct"] == 50
    effects = tables["paired_effects"]
    assert set(zip(effects.reference, effects.arm)) == set(COMPARISONS)
    result = effects[(effects.reference == "three_single")
                     & (effects.arm == "three_refine")].iloc[0]
    assert result.delta_pp == 25 and result.F_to_S == 1 and result.S_to_F == 0
    assert set(tables["compute"].arm) == set(NEW_ARMS)
    assert tables["compute"].mean_vf_per_boundary.eq(20).all()


def test_preview_excludes_partial_identity_from_every_historical_and_new_arm():
    rows, manifest = synthetic_rows()
    rows.loc[(rows.arm == "three_refine") & (rows.init_state_hash == "h3"), "status"] = "error"
    paired, matched, coverage = match_cohort(rows, manifest)
    assert len(paired) == 3 and len(matched) == 21
    assert coverage.matched_identities_used.eq(3).all()
    assert coverage.noncompleted_rows.sum() == 1
    with pytest.raises(ValueError, match="final analysis requires 4"):
        match_cohort(rows, manifest, require_complete=True)


@pytest.mark.parametrize("kind", ["duplicate", "outside", "seed", "chunk", "success"])
def test_matching_rejects_invalid_rows(kind):
    rows, manifest = synthetic_rows()
    if kind == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    elif kind == "outside":
        rows.loc[0, "init_state_hash"] = "outside"
    elif kind == "seed":
        rows.loc[0, "episode_seed"] = -1
    elif kind == "chunk":
        rows.loc[0, "chunk_size"] = 10
    else:
        rows["success"] = rows.success.astype(object)
        rows.loc[0, "success"] = None
    with pytest.raises(ValueError):
        match_cohort(rows, manifest)


def test_probe_tables_are_episode_weighted_and_keep_probe_flow_times_separate():
    rows, manifest = synthetic_rows()
    _, matched, _ = match_cohort(rows, manifest)
    records = synthetic_probe_records(matched)
    tables = probe_tables(records)
    episode = tables["probe_episode"]
    assert episode.rollout_id.nunique() == 12
    assert set(episode.probe) == set(PROBE_LABELS.values())
    summary = tables["probe_summary"]
    assert set(summary.outcome) == {"all", "success", "failure"}
    assert set(tables["failure_detection_auc"].horizon) == set(HORIZONS)
    assert len(tables["first_chunk_probe_records"]) == 16
    # Each rollout counts once even when it has two boundaries.
    row = summary[(summary.probe == PROBE_LABELS[("three_single", 2)])
                  & (summary.outcome == "all")].iloc[0]
    source = episode[episode.probe == row.probe]
    assert row.u10 == pytest.approx(source.u10.mean())


def _npz(steps, chunks=2):
    values = {}
    for chunk in range(chunks):
        for step in steps:
            stem = f"c{chunk}_s{step}"
            values[f"{stem}_u_time"] = np.linspace(.1, .2, 50)
            values[f"{stem}_u_iter_time"] = np.stack([
                np.linspace(.2 - index * .01, .3 - index * .01, 50) for index in range(4)])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def test_probe_loader_downloads_only_new_exact_artifacts_and_validates_counts():
    rows, manifest = synthetic_rows()
    _, matched, _ = match_cohort(rows, manifest)
    payloads = {}
    for row in matched[matched.arm.isin(NEW_ARMS)].to_dict("records"):
        steps = [step for arm, step in PROBE_LABELS if arm == row["arm"]]
        payloads[row["ahats_path"]] = _npz(steps, row["n_chunks"])
    store = Mock()
    store._download.side_effect = lambda path: payloads[path]
    records = load_probe_artifacts(store, matched)
    assert set(records.arm) == set(NEW_ARMS)
    assert store._download.call_count == 12
    assert len(records) == sum(row["n_chunks"] * len([
        step for arm, step in PROBE_LABELS if arm == row["arm"]])
        for row in matched[matched.arm.isin(NEW_ARMS)].to_dict("records"))
    broken = matched.copy()
    broken.loc[broken.arm == "three_single", "ahats_path"] = None
    with pytest.raises(ValueError, match="missing"):
        load_probe_artifacts(store, broken)


class _Query:
    def __init__(self):
        self.filters = []

    def eq(self, key, value):
        self.filters.append((key, value))
        return self


def test_fetch_binds_new_runs_to_prior_verified_manifest_model_and_configs(monkeypatch):
    old, manifest = synthetic_rows(220)
    old = old[old.arm.isin(ARMS[:4])].copy()
    old["arm"] = old.arm.map({"stock10": "baseline", "five_single": "single",
                              "five_select": "select", "five_select_refine": "refine"})
    old_provenance = {
        "source_id": SOURCE, "historical_experiment": "stock-experiment",
        "experiment": "old-five", "historical_config_hash": "stock-hash",
        "new_config_hashes": {"old": "hash"}}
    monkeypatch.setattr("analysis.coarse_refinement.fetch_five_step_rows",
                        lambda store: (old, manifest, old_provenance))
    methods = build_coarse_refinement_methods(SOURCE)
    store = SupabaseStore.__new__(SupabaseStore)
    new = []
    for method, config in methods:
        digest = store.config_hash(store._logical_key(method, config))
        arm = {methods[0][0]: "five_refine", methods[1][0]: "three_refine",
               methods[2][0]: "three_single"}[method]
        for index, identity in enumerate(manifest.to_dict("records")):
            new.append({**identity, "rollout_id": f"{arm}-{index}", "run_id": "new-run",
                        "experiment": COARSE_REFINEMENT_EXPERIMENT,
                        "method": method, "config_hash": digest, "status": "completed",
                        "success": True, "max_steps": 300, "episode_seed": 100 + index,
                        "chunk_size": 50})
    run_config = {"frozen_identity_manifest": manifest.to_dict("records"),
                  "frozen_identity_manifest_hash": identity_manifest_hash(manifest.to_dict("records")),
                  "source_model_repo_id": "source", "source_model_revision": "revision"}
    data = {"rollouts": new, "experiment_runs": [{
        "run_id": "new-run", "experiment": COARSE_REFINEMENT_EXPERIMENT,
        "config_json": run_config,
        "model_repo_id": "source", "model_revision": "revision"}]}

    def fetch(table, columns, *, configure=None, **kwargs):
        query = _Query()
        if configure:
            configure(query)
        return [row for row in data[table]
                if all(row.get(key) == value for key, value in query.filters)]

    store.fetch_all = fetch
    combined, returned_manifest, provenance = fetch_coarse_refinement_rows(store)
    assert len(combined) == 1540 and len(returned_manifest) == 220
    assert set(combined.arm) == set(ARMS)
    assert provenance["source_id"] == SOURCE

    data["experiment_runs"][0]["model_revision"] = "wrong"
    with pytest.raises(ValueError, match="checkpoint provenance"):
        fetch_coarse_refinement_rows(store)


def test_plot_functions_generate_five_nonempty_figures(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows, manifest = synthetic_rows()
    paired, matched, _ = match_cohort(rows, manifest)
    tables = analyze_success(paired, matched, n_boot=100)
    tables.update(probe_tables(synthetic_probe_records(matched)))
    paths = success_figures(tables, tmp_path)
    paths += probe_figures(tables, tmp_path)
    paths.append(compute_figure(tables, tmp_path))
    assert len(paths) == 5 and all(path.stat().st_size > 10_000 for path in paths)


def test_generated_notebook_executes_all_cells_with_synthetic_data(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import analysis.coarse_refinement as analysis

    path = ROOT / "notebooks" / "63_analyze_coarse_single_refinement_pro220.ipynb"
    document = json.loads(path.read_text(encoding="utf-8"))
    rows, manifest = synthetic_rows()
    paired, matched, _ = match_cohort(rows, manifest)
    records = synthetic_probe_records(matched)
    monkeypatch.setattr(analysis, "fetch_coarse_refinement_rows",
                        lambda *args, **kwargs: (rows, manifest, {"source_id": SOURCE}))
    monkeypatch.setattr(analysis, "load_probe_artifacts",
                        lambda *args, **kwargs: records)
    monkeypatch.setattr("pnp.store.SupabaseStore", Mock(return_value=Mock()))
    monkeypatch.chdir(tmp_path)
    namespace = {"display": lambda *args, **kwargs: None, "Image": lambda **kwargs: kwargs}
    assert "accelerator" not in document["metadata"]
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        source = "".join(cell["source"])
        ast.parse(source, filename=f"cell-{index}")
        if "urllib.request" not in source:
            source = source.replace("from IPython.display import display, Image", "")
            source = source.replace("from tqdm.auto import tqdm", "tqdm = lambda x, **kwargs: x")
            exec(compile(source, f"cell-{index}", "exec"), namespace)
    figures = list((tmp_path / "coarse_refinement_pro220_outputs" / "figures").glob("*.png"))
    assert len(figures) == 5 and all(path.stat().st_size > 10_000 for path in figures)


def test_notebook_is_reproducibly_generated_and_clean():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from build_coarse_refinement_analysis_notebook import analysis_notebook
        path = ROOT / "notebooks" / "63_analyze_coarse_single_refinement_pro220.ipynb"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document == analysis_notebook()
        source = "\n".join("".join(cell["source"]) for cell in document["cells"])
        assert "REQUIRE_FULL_COHORT = False" in source
        assert "U10/U20/U50" in source and "seven arms" in source
    finally:
        sys.path.pop(0)
