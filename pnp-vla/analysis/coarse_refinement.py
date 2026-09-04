"""Exact matched analysis for the single-query 5/3-step refinement follow-up."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .five_step_diversity import _keys, _object, _referenced_runs, fetch_five_step_rows
from .horizon_diagnostics import decode_horizon_artifact, _download_with_retry
from .statistics import (
    auc_metrics, discordant_test, paired_bootstrap_ci, paired_counts, wilson_interval)
from pnp.coarse_refinement_experiment import (
    COARSE_REFINEMENT_EXPERIMENT, build_coarse_refinement_methods)
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS as KEYS
from pnp.five_step_diversity_experiment import identity_manifest_hash, identity_manifest_payload


ARMS = ("stock10", "five_single", "five_select", "five_select_refine",
        "five_refine", "three_single", "three_refine")
NEW_ARMS = ("five_refine", "three_single", "three_refine")
LABELS = {
    "stock10": "10-step x1 (historical stock)",
    "five_single": "5-step x1 (historical)",
    "five_select": "5-step x3 lowest U20 (historical)",
    "five_select_refine": "5-step x3 + refine (historical)",
    "five_refine": "5-step x1 + refine",
    "three_single": "3-step x1",
    "three_refine": "3-step x1 + refine",
}
COLORS = {
    "stock10": "#4C78A8", "five_single": "#72B7B2", "five_select": "#F58518",
    "five_select_refine": "#B279A2", "five_refine": "#9C755F",
    "three_single": "#59A14F", "three_refine": "#E15759",
}
COMPARISONS = (
    ("stock10", "five_refine"), ("stock10", "three_single"),
    ("stock10", "three_refine"), ("five_single", "five_refine"),
    ("three_single", "three_refine"), ("five_refine", "three_refine"),
    ("five_select_refine", "five_refine"),
)
PROBE_LABELS = {
    ("five_refine", 2): "5s refine: step 2 (s=0.6)",
    ("five_refine", 3): "5s refine: step 3 (s=0.4)",
    ("three_single", 2): "3s x1: step 2 (s=1/3)",
    ("three_refine", 2): "3s refine: step 2 (s=1/3)",
}
HORIZONS = (10, 20, 50)


def fetch_coarse_refinement_rows(store, *, experiment=COARSE_REFINEMENT_EXPERIMENT):
    """Bind all seven arms to one frozen identity manifest and one model revision."""
    old, manifest, old_provenance = fetch_five_step_rows(store)
    old = old.copy()
    old["arm"] = old.arm.map({"baseline": "stock10", "single": "five_single",
                              "select": "five_select", "refine": "five_select_refine"})
    new = pd.DataFrame(store.fetch_all(
        "rollouts", "*", configure=lambda query: query.eq("experiment", experiment),
        order_by=("rollout_id",)))
    if new.empty:
        raise ValueError(f"no rollout rows found for {experiment}")
    runs = _referenced_runs(store, new, experiment)
    source_repo, revision = old_provenance["source_id"].split("@", 1)
    expected_manifest_hash = identity_manifest_hash(manifest.to_dict("records"))
    signatures = set()
    for run in runs:
        config = _object(run["config_json"])
        saved_manifest = config.get("frozen_identity_manifest", [])
        digest = identity_manifest_hash(saved_manifest)
        if (len(saved_manifest) != 220 or len(identity_manifest_payload(saved_manifest)) != 220
                or digest != config.get("frozen_identity_manifest_hash")
                or digest != expected_manifest_hash):
            raise ValueError("coarse-refinement run has a mismatched frozen manifest")
        if (config.get("source_model_repo_id") != source_repo
                or config.get("source_model_revision") != revision
                or run.get("model_repo_id") != source_repo
                or run.get("model_revision") != revision):
            raise ValueError("coarse-refinement checkpoint provenance is inconsistent")
        signatures.add((source_repo, revision, digest))
    if len(signatures) != 1:
        raise ValueError("refusing to pool different coarse-refinement cohorts")

    methods = build_coarse_refinement_methods(old_provenance["source_id"])
    expected_hashes = {method: store.config_hash(store._logical_key(method, config))
                       for method, config in methods}
    method_to_arm = {Method.FIVE_STEP_SINGLE_REFINE: "five_refine",
                     Method.THREE_STEP_SINGLE_QUERY: "three_single",
                     Method.THREE_STEP_SINGLE_REFINE: "three_refine"}
    if (new.method.map(expected_hashes).isna().any()
            or not new.config_hash.eq(new.method.map(expected_hashes)).all()):
        raise ValueError("coarse-refinement experiment contains unexpected methods/config hashes")
    new["arm"] = new.method.map(method_to_arm)
    if new.arm.isna().any():
        raise ValueError("coarse-refinement experiment contains an unmapped arm")
    provenance = {
        "experiment": experiment, "source_id": old_provenance["source_id"],
        "frozen_identity_manifest_hash": expected_manifest_hash,
        "new_config_hashes": expected_hashes,
        "historical_experiments": {
            "stock10": old_provenance["historical_experiment"],
            "five_step": old_provenance["experiment"]},
        "historical_config_hash": old_provenance["historical_config_hash"],
        "historical_five_step_config_hashes": old_provenance["new_config_hashes"],
    }
    return pd.concat([old, new], ignore_index=True), manifest, provenance


def match_cohort(rows, manifest, *, require_complete=False):
    """Use the intersection of all seven exact arms; never change denominator by plot."""
    expected = _keys(manifest)
    if _keys(rows) - expected:
        raise ValueError("rollout identities fall outside the frozen manifest")
    completed = rows[rows.status.eq("completed")].copy()
    if completed.duplicated(KEYS + ["arm"]).any():
        raise ValueError("duplicate completed identity/arm rows")
    if not completed.success.isin([True, False, 0, 1]).all():
        raise ValueError("completed rollouts have missing or invalid success outcomes")
    sets = {arm: _keys(completed[completed.arm.eq(arm)]) for arm in ARMS}
    common = set.intersection(*sets.values())
    if not common:
        raise ValueError("no identities have completed all seven exact arms")
    if require_complete and common != expected:
        raise ValueError(f"final analysis requires {len(expected)} identities; found {len(common)}")
    coverage = pd.DataFrame([{
        "arm": arm, "label": LABELS[arm], "target_identities": len(expected),
        "completed_identities": len(sets[arm]), "matched_identities_used": len(common),
        "completed_but_unmatched": len(sets[arm] - common),
        "noncompleted_rows": int((rows.arm.eq(arm) & ~rows.status.eq("completed")).sum()),
    } for arm in ARMS])
    matched = completed[completed[KEYS].apply(tuple, axis=1).isin(common)].copy()
    matched["success"] = matched.success.astype(bool)
    for column in ("max_steps", "episode_seed", "chunk_size"):
        if matched[column].isna().any() or matched.groupby(KEYS)[column].nunique().ne(1).any():
            raise ValueError(f"matched arms disagree on {column}")
    if not matched.chunk_size.eq(50).all():
        raise ValueError("expected 50-action model outputs")
    paired = matched.pivot(index=KEYS, columns="arm", values="success").reset_index()
    paired.columns.name = None
    paired = paired.rename(columns={arm: f"{arm}_success" for arm in ARMS})
    return paired.sort_values(KEYS).reset_index(drop=True), matched, coverage


def success_table(paired, *, by_suite=False):
    groups = paired.groupby("suite", sort=True) if by_suite else [("OVERALL", paired)]
    output = []
    for suite, group in groups:
        for arm in ARMS:
            wins, count = int(group[f"{arm}_success"].sum()), len(group)
            lo, hi = wilson_interval(wins, count)
            output.append({"suite": suite, "arm": arm, "label": LABELS[arm],
                           "identities": count, "successes": wins,
                           "sr_pct": 100 * wins / count,
                           "ci_low_pct": 100 * lo, "ci_high_pct": 100 * hi})
    return pd.DataFrame(output)


def effect_table(paired, *, by_suite=False, n_boot=5000):
    groups = paired.groupby("suite", sort=True) if by_suite else [("OVERALL", paired)]
    output = []
    for suite, group in groups:
        for reference, arm in COMPARISONS:
            baseline = group[f"{reference}_success"].to_numpy(bool)
            condition = group[f"{arm}_success"].to_numpy(bool)
            lo, hi = paired_bootstrap_ci(baseline, condition, n_boot=n_boot)
            counts = paired_counts(baseline, condition)
            output.append({
                "suite": suite, "reference": reference, "arm": arm,
                "comparison": f"{LABELS[arm]} minus {LABELS[reference]}",
                "identities": len(group), "baseline_sr_pct": 100 * baseline.mean(),
                "condition_sr_pct": 100 * condition.mean(),
                "delta_pp": 100 * (condition.mean() - baseline.mean()),
                "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi, **counts,
                "paired_p_value": discordant_test(counts["F_to_S"], counts["S_to_F"]),
            })
    return pd.DataFrame(output)


def compute_summary(matched):
    frame = matched[matched.arm.isin(NEW_ARMS)].copy()
    if frame.n_chunks.isna().any() or frame.n_chunks.le(0).any():
        raise ValueError("new completed arms have invalid boundary counts")
    frame["ms_per_boundary"] = frame.inference_ms_total / frame.n_chunks
    frame["vf_per_boundary"] = frame.n_vf_evals / frame.n_chunks
    frame["elapsed_s_per_boundary"] = frame.elapsed_s / frame.n_chunks
    return frame.groupby("arm", sort=False, as_index=False).agg(
        identities=("rollout_id", "size"), boundaries=("n_chunks", "sum"),
        mean_ms_per_boundary=("ms_per_boundary", "mean"),
        mean_vf_per_boundary=("vf_per_boundary", "mean"),
        mean_elapsed_s_per_boundary=("elapsed_s_per_boundary", "mean"))


def analyze_success(paired, matched, *, n_boot=5000):
    return {"arm_success_rates": success_table(paired),
            "paired_effects": effect_table(paired, n_boot=n_boot),
            "suite_success_rates": success_table(paired, by_suite=True),
            "suite_effects": effect_table(paired, by_suite=True, n_boot=n_boot),
            "compute": compute_summary(matched)}


def load_probe_artifacts(store, matched, *, progress=None):
    """Download only U-profile blobs for the three new arms; no trajectories/videos/actions."""
    rows = matched[matched.arm.isin(NEW_ARMS)].copy()
    if rows.ahats_path.isna().any() or rows.ahats_path.astype(str).str.strip().eq("").any():
        raise ValueError("new completed arms are missing saved U-profile artifacts")
    frames = []
    iterator = rows.to_dict("records")
    if progress is not None:
        iterator = progress(iterator, total=len(rows), desc="U10/U20/U50 artifacts")
    for row in iterator:
        records, _, _ = decode_horizon_artifact(
            _download_with_retry(store, str(row["ahats_path"])),
            rollout_id=str(row["rollout_id"]))
        expected_steps = {step for arm, step in PROBE_LABELS if arm == row["arm"]}
        if set(records.euler_step) != expected_steps:
            raise ValueError(f"{row['rollout_id']}: unexpected saved probe steps")
        if len(records) != int(row["n_chunks"]) * len(expected_steps):
            raise ValueError(f"{row['rollout_id']}: U-profile boundary count mismatch")
        records["arm"] = row["arm"]
        records["suite"] = row["suite"]
        records["success"] = bool(row["success"])
        records["probe"] = [PROBE_LABELS[(row["arm"], int(step))]
                            for step in records.euler_step]
        frames.append(records)
    return pd.concat(frames, ignore_index=True)


def probe_tables(records):
    metrics = [f"{prefix}{horizon}" for prefix in (
        "u", "start_disagreement", "end_disagreement", "contraction")
               for horizon in HORIZONS]
    if records.empty or not set(metrics).issubset(records):
        raise ValueError("probe records are empty or incomplete")
    episode = records.groupby(
        ["arm", "probe", "euler_step", "rollout_id", "success"], as_index=False)[metrics].mean()
    summaries = []
    for outcome, frame in [("all", episode), ("success", episode[episode.success]),
                           ("failure", episode[~episode.success])]:
        summary = frame.groupby(["arm", "probe", "euler_step"], as_index=False).agg(
            identities=("rollout_id", "nunique"),
            **{metric: (metric, "mean") for metric in metrics})
        summary["outcome"] = outcome
        summaries.append(summary)
    auc_rows = []
    for (arm, probe, step), group in episode.groupby(["arm", "probe", "euler_step"]):
        for horizon in HORIZONS:
            metrics_auc = auc_metrics(~group.success.to_numpy(bool), group[f"u{horizon}"])
            auc_rows.append({"arm": arm, "probe": probe, "euler_step": step,
                             "horizon": horizon, **metrics_auc})
    return {"probe_episode": episode,
            "probe_summary": pd.concat(summaries, ignore_index=True),
            "failure_detection_auc": pd.DataFrame(auc_rows),
            "first_chunk_probe_records": records[records.chunk_idx.eq(0)].copy()}


def _save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def success_figures(tables, output):
    import matplotlib.pyplot as plt
    success, effects = tables["arm_success_rates"], tables["paired_effects"]
    fig, axes = plt.subplots(1, 2, figsize=(19, 6), constrained_layout=True)
    x = np.arange(len(ARMS))
    axes[0].bar(x, success.sr_pct, color=[COLORS[arm] for arm in ARMS])
    axes[0].errorbar(x, success.sr_pct, yerr=np.maximum(0, np.vstack((
        success.sr_pct - success.ci_low_pct, success.ci_high_pct - success.sr_pct))),
        fmt="none", ecolor="black", capsize=3)
    for index, row in success.reset_index(drop=True).iterrows():
        axes[0].text(index, row.sr_pct + 2, f"{row.successes}/{row.identities}",
                     ha="center", fontsize=8)
    axes[0].set_xticks(x, [LABELS[arm] for arm in ARMS], rotation=25, ha="right")
    axes[0].set(ylabel="Success rate (%)", ylim=(0, 110),
                title=f"All seven arms on {success.identities.iloc[0]} matched identities")
    axes[0].grid(axis="y", alpha=.2)
    y = np.arange(len(effects))
    axes[1].barh(y, effects.delta_pp,
                 color=np.where(effects.delta_pp >= 0, "#54A24B", "#E45756"))
    axes[1].errorbar(effects.delta_pp, y, xerr=np.maximum(0, np.vstack((
        effects.delta_pp - effects.ci_low_pp, effects.ci_high_pp - effects.delta_pp))),
        fmt="none", ecolor="black", capsize=3)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(y, effects.comparison, fontsize=8)
    axes[1].invert_yaxis(); axes[1].grid(axis="x", alpha=.2)
    axes[1].set(xlabel="Paired SR change (percentage points)",
                title="Follow-up contrasts; paired 95% bootstrap CI")
    paths = [_save_figure(fig, output, "overall_success_and_effects")]

    suite_sr, suite_effects = tables["suite_success_rates"], tables["suite_effects"]
    suites, x = sorted(suite_sr.suite.unique()), np.arange(suite_sr.suite.nunique())
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), constrained_layout=True)
    width = .11
    for index, arm in enumerate(ARMS):
        group = suite_sr[suite_sr.arm.eq(arm)].set_index("suite").reindex(suites)
        axes[0].bar(x + (index - 3) * width, group.sr_pct, width,
                    color=COLORS[arm], label=LABELS[arm])
    axes[0].set(ylabel="Success rate (%)", ylim=(0, 105),
                title="Exact matched success rates by suite")
    axes[0].legend(ncols=3, fontsize=8); axes[0].grid(axis="y", alpha=.2)
    for index, arm in enumerate(NEW_ARMS):
        group = suite_effects[(suite_effects.reference.eq("stock10"))
                              & suite_effects.arm.eq(arm)].set_index("suite").reindex(suites)
        positions = x + (index - 1) * .25
        axes[1].bar(positions, group.delta_pp, .25,
                    color=COLORS[arm], label=f"{LABELS[arm]} minus stock 10-step")
        axes[1].errorbar(positions, group.delta_pp, yerr=np.maximum(0, np.vstack((
            group.delta_pp - group.ci_low_pp, group.ci_high_pp - group.delta_pp))),
            fmt="none", ecolor="black", capsize=2)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(ylabel="Paired SR change (pp)", title="New arms versus historical stock")
    axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=.2)
    for ax in axes:
        ax.set_xticks(x, [suite.removeprefix("libero_") for suite in suites],
                      rotation=35, ha="right")
    paths.append(_save_figure(fig, output, "success_and_suite_deltas"))
    return paths


def probe_figures(tables, output):
    import matplotlib.pyplot as plt
    overall = tables["probe_summary"].query("outcome == 'all'")
    probes = list(dict.fromkeys(overall.probe))
    palette = plt.cm.tab10(np.linspace(0, .8, len(probes)))
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    x = np.arange(len(HORIZONS))
    for probe, color in zip(probes, palette):
        row = overall[overall.probe.eq(probe)].iloc[0]
        axes[0].plot(x, [row[f"u{h}"] for h in HORIZONS], marker="o", color=color, label=probe)
        axes[1].plot(x, [row[f"contraction{h}"] for h in HORIZONS],
                     marker="o", color=color, label=probe)
    axes[0].set(ylabel="Episode-weighted mean uncertainty", title="U by action horizon")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(ylabel="First-minus-last consecutive disagreement",
                title="PnP contraction by action horizon")
    for ax in axes:
        ax.set_xticks(x, [f"U{h}" for h in HORIZONS]); ax.set_xlabel("Action horizon")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    paths = [_save_figure(fig, output, "uncertainty_and_contraction")]

    auc = tables["failure_detection_auc"]
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    width = .8 / len(probes)
    for index, (probe, color) in enumerate(zip(probes, palette)):
        group = auc[auc.probe.eq(probe)].set_index("horizon").reindex(HORIZONS)
        ax.bar(x + (index - (len(probes) - 1) / 2) * width, group.roc_auc,
               width, color=color, label=probe)
    ax.axhline(.5, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x, [f"U{h}" for h in HORIZONS]); ax.set_ylim(0, 1)
    ax.set(ylabel="Failure-detection ROC AUC", xlabel="Action horizon",
           title="Episode-mean uncertainty versus failure")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    paths.append(_save_figure(fig, output, "uncertainty_failure_auc"))
    return paths


def compute_figure(tables, output):
    import matplotlib.pyplot as plt
    compute = tables["compute"].set_index("arm").reindex(NEW_ARMS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = [LABELS[arm] for arm in NEW_ARMS]
    axes[0].bar(labels, compute.mean_ms_per_boundary,
                color=[COLORS[arm] for arm in NEW_ARMS])
    axes[1].bar(labels, compute.mean_vf_per_boundary,
                color=[COLORS[arm] for arm in NEW_ARMS])
    axes[0].set(ylabel="Mean inference ms per boundary", title="Observed inference time")
    axes[1].set(ylabel="Mean VF evaluations per boundary", title="Sampler compute")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.2)
    return _save_figure(fig, output, "new_arm_compute")
