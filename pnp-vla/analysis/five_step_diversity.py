"""Exact-cohort SR and three-candidate geometry for the five-step PRO pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import discordant_test, paired_bootstrap_ci, paired_counts, wilson_interval
from pnp.diversity import DIVERSITY_PAIR_KEYS as KEYS
from pnp.five_step_diversity_experiment import (
    FIVE_STEP_DIVERSITY_EXPERIMENT, build_five_step_diversity_methods,
    identity_manifest_hash, identity_manifest_payload)
from pnp.uncertainty_gradient_experiment import (
    DIRECT_U20_GRADIENT_EXPERIMENT, build_direct_u20_gradient_methods)


ARMS = ("baseline", "single", "select", "refine")
LABELS = {"baseline": "10-step x1 (historical)", "single": "5-step x1",
          "select": "5-step x3 lowest U20", "refine": "5-step x3 + refine"}
COLORS = {"baseline": "#4C78A8", "single": "#72B7B2",
          "select": "#F58518", "refine": "#B279A2"}
COMPARISONS = (("baseline", "single"), ("baseline", "select"),
               ("baseline", "refine"), ("single", "select"), ("select", "refine"))
PAIRS = ((0, 1), (0, 2), (1, 2))


def _object(value):
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("missing or malformed JSON telemetry/provenance")
    return value


def _keys(frame):
    return set(frame[KEYS].itertuples(index=False, name=None))


def _referenced_runs(store, rows, experiment):
    if rows.run_id.isna().any() or rows.run_id.astype(str).str.strip().eq("").any():
        raise ValueError(f"{experiment}: rows lack run_id provenance")
    runs = store.fetch_all(
        "experiment_runs", "run_id,config_json,model_repo_id,model_revision",
        configure=lambda query: query.eq("experiment", experiment), order_by=("run_id",))
    lookup = {str(run["run_id"]): run for run in runs}
    ids = set(rows.run_id.astype(str))
    if ids - lookup.keys():
        raise ValueError(f"{experiment}: missing referenced experiment_runs")
    return [lookup[run_id] for run_id in sorted(ids)]


def fetch_five_step_rows(store, *, experiment=FIVE_STEP_DIVERSITY_EXPERIMENT):
    """Read only: bind model revision, frozen identities, and all four exact config hashes.

    Uses the manifest saved by the workers, so analysis needs neither Drive nor LIBERO/model
    installation. No raw action blobs are downloaded: pair geometry is already in row JSON.
    """
    new = pd.DataFrame(store.fetch_all(
        "rollouts", "*", configure=lambda query: query.eq("experiment", experiment),
        order_by=("rollout_id",)))
    if new.empty:
        raise ValueError(f"no rollout rows found for {experiment}")
    runs = _referenced_runs(store, new, experiment)
    signatures = set()
    for run in runs:
        config = _object(run["config_json"])
        manifest = config.get("frozen_identity_manifest", [])
        digest = identity_manifest_hash(manifest)
        if (len(manifest) != 220 or len(identity_manifest_payload(manifest)) != 220
                or digest != config.get("frozen_identity_manifest_hash")):
            raise ValueError("worker run has an invalid frozen 220-identity manifest")
        source_repo, revision = config.get("source_model_repo_id"), config.get(
            "source_model_revision")
        if (not source_repo or not revision or run.get("model_repo_id") != source_repo
                or run.get("model_revision") != revision):
            raise ValueError("worker checkpoint provenance is incomplete or inconsistent")
        signatures.add((source_repo, revision, digest))
    if len(signatures) != 1:
        raise ValueError("refusing to pool different model revisions or frozen cohorts")
    source_repo, revision, digest = next(iter(signatures))
    manifest = pd.DataFrame(identity_manifest_payload(manifest))
    source_id = f"{source_repo}@{revision}"
    methods = build_five_step_diversity_methods(source_id)
    method_to_arm = {method: arm for arm, (method, _) in zip(ARMS[1:], methods)}
    expected = {method: store.config_hash(store._logical_key(method, config))
                for method, config in methods}
    if not new.config_hash.eq(new.method.map(expected)).all():
        raise ValueError("five-step experiment contains unexpected methods/config hashes")
    new["arm"] = new.method.map(method_to_arm)

    method, config = build_direct_u20_gradient_methods()[0]
    historical_hash = store.config_hash(store._logical_key(method, config))
    baseline = pd.DataFrame(store.fetch_all(
        "rollouts", "*", configure=lambda query: query.eq(
            "experiment", DIRECT_U20_GRADIENT_EXPERIMENT).eq("method", method).eq(
            "config_hash", historical_hash).eq("status", "completed"),
        order_by=("rollout_id",)))
    if baseline.empty:
        raise ValueError("no exact historical 10-step baseline rows found")
    baseline = baseline[baseline[KEYS].apply(tuple, axis=1).isin(_keys(manifest))].copy()
    if len(baseline) != 220 or _keys(baseline) != _keys(manifest):
        raise ValueError("historical baseline does not cover the exact frozen 220 identities")
    for run in _referenced_runs(store, baseline, DIRECT_U20_GRADIENT_EXPERIMENT):
        if (run.get("model_repo_id") != source_repo
                or run.get("model_revision") != revision):
            raise ValueError("historical baseline uses a different or unverified checkpoint")
    baseline["arm"] = "baseline"
    return pd.concat([baseline, new], ignore_index=True), manifest, {
        "experiment": experiment, "source_id": source_id,
        "frozen_identity_manifest_hash": digest,
        "historical_experiment": DIRECT_U20_GRADIENT_EXPERIMENT,
        "historical_config_hash": historical_hash, "new_config_hashes": expected}


def match_cohort(rows, manifest, *, require_complete=False):
    """One denominator: the intersection of all three new arms and the exact baseline."""
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
        raise ValueError("no identities have completed all three new arms plus baseline yet")
    if require_complete and common != expected:
        raise ValueError(f"final analysis requires {len(expected)} identities; found {len(common)}")
    coverage = pd.DataFrame([{
        "arm": LABELS[arm], "target_identities": len(expected),
        "completed_identities": len(sets[arm]), "matched_identities_used": len(common),
        "completed_but_unmatched": len(sets[arm] - common),
        "noncompleted_rows": int((rows.arm.eq(arm) & ~rows.status.eq("completed")).sum()),
    } for arm in ARMS])
    matched = completed[completed[KEYS].apply(tuple, axis=1).isin(common)].copy()
    matched["success"] = matched.success.astype(bool)
    # Same rollout horizon and episode seed are required, not just matching identity labels.
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
    rows = []
    for suite, group in groups:
        for arm in ARMS:
            wins = int(group[f"{arm}_success"].sum())
            lo, hi = wilson_interval(wins, len(group))
            rows.append({"suite": suite, "arm": arm, "label": LABELS[arm],
                         "identities": len(group), "successes": wins,
                         "sr_pct": 100 * wins / len(group),
                         "ci_low_pct": 100 * lo, "ci_high_pct": 100 * hi})
    return pd.DataFrame(rows)


def effect_table(paired, *, by_suite=False, n_boot=5000):
    groups = paired.groupby("suite", sort=True) if by_suite else [("OVERALL", paired)]
    rows = []
    for suite, group in groups:
        for reference, arm in COMPARISONS:
            b, c = (group[f"{name}_success"].to_numpy(bool) for name in (reference, arm))
            lo, hi = paired_bootstrap_ci(b, c, n_boot=n_boot)
            counts = paired_counts(b, c)
            rows.append({"suite": suite, "reference": reference, "arm": arm,
                         "comparison": f"{LABELS[arm]} minus {LABELS[reference]}",
                         "identities": len(group), "baseline_sr_pct": 100 * b.mean(),
                         "condition_sr_pct": 100 * c.mean(),
                         "delta_pp": 100 * (c.mean() - b.mean()),
                         "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi, **counts,
                         "paired_p_value": discordant_test(counts["F_to_S"], counts["S_to_F"])})
    return pd.DataFrame(rows)


def candidate_tables(matched):
    """One row per boundary, candidate, and unordered pair/horizon; never zip-truncate."""
    boundaries, candidates, pairs = [], [], []
    for row in matched[matched.arm.isin(("select", "refine"))].to_dict("records"):
        trace = _object(row["ms_candidate_u"])
        required = ["chosen", "u", "candidate_profiles", "action_disagreement",
                    "executed_prefix_disagreement", "inference_ms", "n_vf_evals"]
        if row["arm"] == "refine":
            required.append("selected_refinement")
        count = int(row["n_chunks"])
        if count < 1 or any(not isinstance(trace.get(key), list)
                            or len(trace[key]) != count for key in required):
            raise ValueError(f"{row['rollout_id']}: missing/misaligned boundary telemetry")
        for index in range(count):
            base = {key: row[key] for key in KEYS + ["rollout_id", "arm", "success"]}
            base["chunk_idx"] = index
            chosen, values = trace["chosen"][index], np.asarray(trace["u"][index], float)
            profiles = trace["candidate_profiles"][index]
            if (chosen not in (0, 1, 2) or values.shape != (3,)
                    or not np.isfinite(values).all() or len(profiles) != 3
                    or chosen != int(np.argmin(values))):
                raise ValueError(f"{row['rollout_id']}: invalid lowest-U20 selection")
            for slot, profile in enumerate(profiles):
                scores = [float(profile[name]) for name in ("u10", "u20", "u_full")]
                if not np.isfinite(scores).all() or not np.isclose(scores[1], values[slot]):
                    raise ValueError("candidate U20 disagrees with selection score")
                candidates.append({**base, "candidate": slot, "selected": slot == chosen,
                                   **dict(zip(("u10", "u20", "u50"), scores))})
            boundary = {**base, "chosen": chosen, "selected_u20": values[chosen],
                        "u20_spread": float(np.ptp(values)),
                        "slot0_minus_selected_u20": float(values[0] - values[chosen]),
                        "inference_ms": trace["inference_ms"][index],
                        "n_vf_evals": trace["n_vf_evals"][index]}
            if row["arm"] == "refine":
                refinement = _object(trace["selected_refinement"][index])
                boundary.update(
                    refined_path_u20=float(refinement["refined_path_u"]),
                    refined_path_delta_u20=float(refinement["delta_u"]),
                    refinement_lowered_path_u=bool(refinement["lowered_u"]),
                    refinement_prefix_l2=float(
                        refinement["selected_prefix_movement"]["action_l2_mean"]))
            boundaries.append(boundary)
            for horizon, field in ((50, "action_disagreement"),
                                   (10, "executed_prefix_disagreement")):
                geometry = _object(trace[field][index])
                entries = geometry.get("pairs", [])
                if (geometry.get("actions_compared") != horizon or len(entries) != 3
                        or {(entry["left"], entry["right"]) for entry in entries} != set(PAIRS)):
                    raise ValueError("expected all three unordered pairs at horizons 10 and 50")
                for entry in entries:
                    raw = entry.get("action_cosine")
                    cosine = float(raw) if raw is not None else np.nan
                    if np.isfinite(cosine) and abs(cosine) > 1 + 1e-5:
                        raise ValueError("invalid cosine similarity outside [-1, 1]")
                    cosine = float(np.clip(cosine, -1, 1))
                    pairs.append({**base, "horizon": horizon,
                                  "pair": f"{entry['left']}-{entry['right']}",
                                  "cosine_similarity": cosine,
                                  "cosine_distance": 1 - cosine,
                                  "action_l2_mean": entry["action_l2_mean"],
                                  "gripper_disagreement": entry.get("gripper_sign_disagreement")})
    return pd.DataFrame(boundaries), pd.DataFrame(candidates), pd.DataFrame(pairs)


def diversity_summary(pairs):
    """Episode-weighted means; pooled boundary counts are descriptive, not independent N."""
    metrics = ["cosine_similarity", "cosine_distance", "action_l2_mean", "gripper_disagreement"]
    group_keys = ["arm", "horizon", "pair"]
    per_episode = pairs.groupby(group_keys + ["rollout_id"], as_index=False)[metrics].mean()
    summary = per_episode.groupby(group_keys, as_index=False)[metrics].mean()
    counts = pairs.groupby(group_keys, as_index=False).agg(
        identities=("rollout_id", "nunique"), boundaries=("chunk_idx", "size"),
        finite_cosines=("cosine_similarity", "count"))
    return summary.merge(counts, on=group_keys, validate="one_to_one")


def compute_summary(matched):
    # Older baseline instrumentation omitted an extra measurement-only pass. Do not pretend
    # those historical VF counts are comparable to the corrected counts in the new workers.
    frame = matched[matched.arm.ne("baseline")].copy()
    frame["ms_per_boundary"] = frame.inference_ms_total / frame.n_chunks.replace(0, np.nan)
    frame["vf_per_boundary"] = frame.n_vf_evals / frame.n_chunks.replace(0, np.nan)
    return frame.groupby("arm", sort=False, as_index=False).agg(
        identities=("rollout_id", "size"), boundaries=("n_chunks", "sum"),
        mean_ms_per_boundary=("ms_per_boundary", "mean"),
        mean_vf_per_boundary=("vf_per_boundary", "mean"))


def analyze_five_step(paired, matched, *, n_boot=5000):
    boundaries, candidates, pairs = candidate_tables(matched)
    selection = boundaries.assign(**{
        f"chosen_{slot}_pct": 100 * boundaries.chosen.eq(slot) for slot in range(3)})
    columns = ["selected_u20", "u20_spread", "slot0_minus_selected_u20",
               "chosen_0_pct", "chosen_1_pct", "chosen_2_pct"]
    selection = selection.groupby(["arm", "rollout_id"])[columns].mean()
    selection = selection.groupby("arm").mean().reset_index()
    refinement = boundaries[boundaries.arm.eq("refine")].groupby("rollout_id")[[
        "selected_u20", "refined_path_u20", "refined_path_delta_u20",
        "refinement_lowered_path_u", "refinement_prefix_l2"]].mean().reset_index()
    uncertainty = candidates.groupby(["arm", "rollout_id", "candidate"])[[
        "u10", "u20", "u50"]].mean().groupby(["arm", "candidate"]).mean().reset_index()
    return {"arm_success_rates": success_table(paired),
            "paired_effects": effect_table(paired, n_boot=n_boot),
            "suite_success_rates": success_table(paired, by_suite=True),
            "suite_effects": effect_table(paired, by_suite=True, n_boot=n_boot),
            "boundaries": boundaries, "candidates": candidates, "candidate_pairs": pairs,
            "candidate_diversity": diversity_summary(pairs),
            "first_boundary_diversity": diversity_summary(pairs[pairs.chunk_idx.eq(0)]),
            "selection_summary": selection, "candidate_uncertainty": uncertainty,
            "refinement_by_episode": refinement, "compute": compute_summary(matched)}


def _save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def success_figures(tables, output):
    """Match notebooks 27/49: SR bars, paired CIs, suite deltas, and zero references."""
    import matplotlib.pyplot as plt

    success, effects = tables["arm_success_rates"], tables["paired_effects"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), constrained_layout=True)
    x = np.arange(len(ARMS))
    axes[0].bar(x, success.sr_pct, color=[COLORS[arm] for arm in ARMS])
    axes[0].errorbar(x, success.sr_pct,
                    yerr=np.maximum(0, np.vstack((
                        success.sr_pct - success.ci_low_pct,
                        success.ci_high_pct - success.sr_pct))),
                    fmt="none", ecolor="black", capsize=3)
    for index, row in success.iterrows():
        axes[0].text(index, row.sr_pct + 2, f"{row.successes}/{row.identities}",
                     ha="center", fontsize=9)
    axes[0].set_xticks(x, [LABELS[arm] for arm in ARMS], rotation=20, ha="right")
    axes[0].set(ylabel="Success rate (%)", ylim=(0, 110),
                title=f"All four arms on {success.identities.iloc[0]} matched identities")
    axes[0].grid(axis="y", alpha=.2)
    y = np.arange(len(effects))
    axes[1].barh(y, effects.delta_pp,
                 color=np.where(effects.delta_pp >= 0, "#54A24B", "#E45756"))
    axes[1].errorbar(effects.delta_pp, y,
                    xerr=np.maximum(0, np.vstack((
                        effects.delta_pp - effects.ci_low_pp,
                        effects.ci_high_pp - effects.delta_pp))),
                    fmt="none", ecolor="black", capsize=3)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(y, effects.comparison, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Paired SR change (percentage points)",
                title="Whole-matched-cohort effects; paired 95% bootstrap CI")
    axes[1].grid(axis="x", alpha=.2)
    paths = [_save_figure(fig, output, "overall_success_and_effects")]

    suite_sr = tables["suite_success_rates"]
    suites = sorted(suite_sr.suite.unique())
    x = np.arange(len(suites))
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for index, arm in enumerate(ARMS):
        group = suite_sr[suite_sr.arm.eq(arm)].set_index("suite").reindex(suites)
        axes[0].bar(x + (index - 1.5) * .2, group.sr_pct, .2,
                    color=COLORS[arm], label=LABELS[arm])
    axes[0].set(ylabel="Success rate (%)", ylim=(0, 105),
                title="Exact matched success rates by suite")
    axes[0].legend(ncols=2); axes[0].grid(axis="y", alpha=.2)
    for index, arm in enumerate(ARMS[1:]):
        group = tables["suite_effects"]
        group = group[group.reference.eq("baseline") & group.arm.eq(arm)]
        group = group.set_index("suite").reindex(suites)
        positions = x + (index - 1) * .25
        axes[1].bar(positions, group.delta_pp, .25, color=COLORS[arm],
                    label=f"{LABELS[arm]} minus historical 10-step x1")
        axes[1].errorbar(positions, group.delta_pp,
                        yerr=np.maximum(0, np.vstack((
                            group.delta_pp - group.ci_low_pp,
                            group.ci_high_pp - group.delta_pp))),
                        fmt="none", ecolor="black", capsize=2)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(ylabel="Paired SR change (pp)", title="Per-suite change versus historical stock")
    axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=.2)
    for ax in axes:
        ax.set_xticks(x, [suite.removeprefix("libero_") for suite in suites],
                      rotation=35, ha="right")
    paths.append(_save_figure(fig, output, "success_and_suite_deltas"))
    return paths


def diversity_figures(tables, output):
    """All three original candidate pairs, separate full-chunk and executed-prefix metrics."""
    import matplotlib.pyplot as plt

    pairs, summary = tables["candidate_pairs"], tables["candidate_diversity"]
    pair_labels = ["0-1", "0-2", "1-2"]
    palette = ["#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row_index, arm in enumerate(("select", "refine")):
        for column, horizon in enumerate((10, 50)):
            ax = axes[row_index, column]
            group = pairs[pairs.arm.eq(arm) & pairs.horizon.eq(horizon)]
            values = group.cosine_similarity.dropna()
            if len(values):
                low, high = float(values.min()), float(values.max())
                pad = max((high - low) * .03, .002)
                bins = np.linspace(max(-1, low - pad), min(1, high + pad), 31)
                for pair, color in zip(pair_labels, palette):
                    series = group[group.pair.eq(pair)].cosine_similarity.dropna()
                    ax.hist(series, bins=bins, histtype="step", linewidth=1.7,
                            color=color, label=f"pair {pair} (n={len(series)})")
            else:
                ax.text(.5, .5, "No finite cosines (zero-norm actions)",
                        ha="center", transform=ax.transAxes)
            ax.set(xlabel="Cosine similarity (1 = same direction)", ylabel="Boundaries",
                   title=f"{LABELS[arm]}: first {horizon} actions")
            ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.suptitle("Original candidate chunks: policy-space actions including gripper\n"
                 "Pooled boundary distributions; the tables use episode-weighted means")
    paths = [_save_figure(fig, output, "candidate_pair_cosine_distributions")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    metrics = [("cosine_distance", "Cosine distance (1 - similarity)"),
               ("action_l2_mean", "Mean per-action L2 distance"),
               ("gripper_disagreement", "Gripper-sign disagreement fraction")]
    x = np.arange(3)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for index, (arm, horizon) in enumerate(
                (("select", 10), ("select", 50), ("refine", 10), ("refine", 50))):
            group = summary[summary.arm.eq(arm) & summary.horizon.eq(horizon)]
            group = group.set_index("pair").reindex(pair_labels)
            ax.bar(x + (index - 1.5) * .2, group[metric], .2,
                   color=COLORS[arm], alpha=1 if horizon == 10 else .5,
                   label=f"{LABELS[arm]}, first {horizon}")
        ax.set_xticks(x, pair_labels)
        ax.set(xlabel="Candidate pair", ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Candidate geometry: average within each episode, then across episodes")
    paths.append(_save_figure(fig, output, "candidate_pair_distances"))
    return paths


def selection_figures(tables, output):
    import matplotlib.pyplot as plt

    selection, boundaries = tables["selection_summary"], tables["boundaries"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(3)
    for index, arm in enumerate(("select", "refine")):
        row = selection[selection.arm.eq(arm)].iloc[0]
        axes[0].bar(x + (index - .5) * .35,
                    [row[f"chosen_{slot}_pct"] for slot in range(3)], .35,
                    color=COLORS[arm], label=LABELS[arm])
        episode_means = boundaries[boundaries.arm.eq(arm)].groupby("rollout_id").u20_spread.mean()
        axes[1].hist(episode_means, bins=20, alpha=.6, color=COLORS[arm], label=LABELS[arm])
    axes[0].set_xticks(x, ["0 (stock seed)", "1", "2"])
    axes[0].set(xlabel="Selected candidate", ylabel="Episode-mean selection frequency (%)",
                ylim=(0, 105), title="Which candidate has the lowest U20?")
    axes[1].set(xlabel="Episode-mean candidate U20 spread (max - min)", ylabel="Identities",
                title="Does re-querying expose uncertainty differences?")
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=.2)
    paths = [_save_figure(fig, output, "candidate_choices_and_u_spread")]

    refinement = tables["refinement_by_episode"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(refinement.refined_path_delta_u20, bins=25, color=COLORS["refine"], alpha=.8)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Episode-mean refined-path U20 minus unrefined U20",
                ylabel="Identities", title="Refinement-path uncertainty diagnostic")
    axes[1].scatter(refinement.refinement_prefix_l2, refinement.refined_path_delta_u20,
                    color=COLORS["refine"], alpha=.65, s=30)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Episode-mean movement of selected first 10 actions (L2)",
                ylabel="Episode-mean refined-path U20 change",
                title="Uncertainty-path change versus executed-prefix movement")
    for ax in axes:
        ax.grid(alpha=.2)
    paths.append(_save_figure(fig, output, "selected_refinement_diagnostics"))
    return paths
