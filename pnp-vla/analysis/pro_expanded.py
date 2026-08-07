"""Analysis for the K=5, 13-suite expanded LIBERO-PRO collection."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from pnp.config import DIM_NAMES, Method
from .conditions import PAIR_KEYS
from . import pro
from .statistics import (bootstrap_auc, paired_bootstrap_ci, paired_counts,
                         wilson_interval)


CONTRACTION_METRICS = (
    "contraction_normalized_slope",
    "contraction_log_ratio",
    "contraction_monotonic_fraction",
    "contraction_abs",
    "contraction_within_suite_rank",
)


def _array(value) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    out = np.asarray(value, dtype=float).reshape(-1)
    return out if len(out) and np.isfinite(out).all() else None


def _profile_features(profile: np.ndarray) -> dict[str, float]:
    eps = max(1e-12, float(np.mean(profile)) * 1e-8)
    slope = float(np.polyfit(np.arange(len(profile), dtype=float), profile, 1)[0])
    return {
        **{f"u_iter_{i}": float(value) for i, value in enumerate(profile)},
        "mean_u_iter": float(profile.mean()),
        # Every contraction score is oriented so larger means stronger contraction.
        "contraction_abs": float(profile[0] - profile[-1]),
        "contraction_log_ratio": float(np.log((profile[0] + eps) / (profile[-1] + eps))),
        "contraction_normalized_slope": float(-slope / (profile.mean() + eps)),
        "contraction_monotonic_fraction": float(np.mean(np.diff(profile) < 0)),
    }


def episode_contraction(rollouts: pd.DataFrame, vectors: pd.DataFrame,
                        *, early_chunks: int = 4) -> pd.DataFrame:
    """Build one contraction row per observed rollout and analysis window.

    Features come only from the observed/no-op arm. The paired refinement outcome supplies the
    label, avoiding intervention-arm telemetry leakage. ``corrected`` is defined only among
    observed failures and means that the matched refine-last rollout succeeded.
    """
    observed = rollouts[rollouts.method == Method.UNCERTAINTY].copy()
    refined = rollouts[rollouts.method == Method.REFINEMENT].copy()
    pair = observed[PAIR_KEYS + ["rollout_id", "success", "u_mean_episode"]].merge(
        refined[PAIR_KEYS + ["success"]], on=PAIR_KEYS, validate="one_to_one",
        suffixes=("_observed", "_refined"))
    pair = pair.rename(columns={"rollout_id": "rollout_id_observed"})

    needed = {"rollout_id", "chunk_idx", "euler_step", "u_iter"}
    missing = sorted(needed - set(vectors.columns))
    if missing:
        raise ValueError(f"pnp_action_vectors lacks contraction columns: {missing}")
    telemetry = vectors[vectors.rollout_id.isin(pair.rollout_id_observed)].copy()
    telemetry["u_iter_array"] = telemetry.u_iter.map(_array)
    telemetry = telemetry[telemetry.u_iter_array.notna()]
    if telemetry.empty:
        raise ValueError("no observed-arm u_iter telemetry found; was migration 004 applied?")
    lengths = telemetry.u_iter_array.map(len)
    if set(lengths.unique()) != {4}:
        raise ValueError(f"expected K=5 telemetry with four consecutive pairs, got {sorted(lengths.unique())}")

    rows = []
    for rollout_id, group in telemetry.groupby("rollout_id", sort=False):
        for window, selected in (
            ("first_4_chunks", group[group.chunk_idx < early_chunks]),
            ("full_episode", group),
        ):
            if selected.empty:
                continue
            profile = np.stack(selected.u_iter_array.to_list()).mean(axis=0)
            rows.append({"rollout_id_observed": rollout_id, "window": window,
                         "n_probe_rows": len(selected), **_profile_features(profile)})
    features = pd.DataFrame(rows).merge(pair, on="rollout_id_observed", validate="many_to_one")
    features["observed_failure"] = ~features.success_observed.astype(bool)
    features["corrected"] = features.observed_failure & features.success_refined.astype(bool)
    features["transition"] = np.select(
        [~features.success_observed & features.success_refined,
         features.success_observed & ~features.success_refined,
         features.success_observed & features.success_refined],
        ["F_to_S", "S_to_F", "S_to_S"], default="F_to_F")
    failures = features.observed_failure
    features.loc[failures, "contraction_within_suite_rank"] = (
        features[failures].groupby(["window", "suite"])["contraction_normalized_slope"]
        .rank(method="average", pct=True))
    return features


def _spearman(labels, scores) -> tuple[float, float]:
    from scipy.stats import spearmanr
    y, score = np.asarray(labels, int), np.asarray(scores, float)
    keep = np.isfinite(score)
    if keep.sum() < 3 or len(np.unique(y[keep])) < 2 or np.nanstd(score[keep]) == 0:
        return math.nan, math.nan
    result = spearmanr(score[keep], y[keep])
    return float(result.statistic), float(result.pvalue)


def _stratified_spearman_ci(frame: pd.DataFrame, score: str, *, seed: int = 159,
                            n_boot: int = 2000) -> tuple[float, float]:
    rng, values = np.random.default_rng(seed), []
    groups = [group for _, group in frame.groupby("suite", sort=True)]
    for _ in range(n_boot):
        sample = pd.concat([
            group.iloc[rng.integers(0, len(group), len(group))] for group in groups
        ], ignore_index=True)
        rho, _ = _spearman(sample.corrected, sample[score])
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return math.nan, math.nan
    return tuple(float(value) for value in np.quantile(values, [.025, .975]))


def contraction_analysis(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Test whether observed-arm contraction predicts paired F->S correction."""
    failures = features[features.observed_failure].copy()
    summaries, suites = [], []
    for window, group in failures.groupby("window", sort=True):
        for metric in CONTRACTION_METRICS:
            rho, p_value = _spearman(group.corrected, group[metric])
            lo, hi = _stratified_spearman_ci(group, metric)
            auc = bootstrap_auc(group.corrected.astype(int), group[metric], n_boot=2000)
            summaries.append({"window": window, "metric": metric, "n_failures": len(group),
                              "n_corrected": int(group.corrected.sum()),
                              "correction_rate": float(group.corrected.mean()),
                              "spearman_rho": rho, "spearman_p": p_value,
                              "spearman_ci_low": lo, "spearman_ci_high": hi, **auc})
        for suite, suite_group in group.groupby("suite", sort=True):
            rho, p_value = _spearman(
                suite_group.corrected, suite_group.contraction_normalized_slope)
            auc = bootstrap_auc(
                suite_group.corrected.astype(int),
                suite_group.contraction_normalized_slope, n_boot=1000)
            suites.append({"window": window, "suite": suite,
                           "n_failures": len(suite_group),
                           "n_corrected": int(suite_group.corrected.sum()),
                           "correction_rate": float(suite_group.corrected.mean()),
                           "spearman_rho": rho, "spearman_p": p_value, **auc})

    quartile_rows, trend_rows, roc_rows = [], [], []
    for window, group in failures.groupby("window", sort=True):
        group = group.copy()
        group["contraction_quartile"] = pd.qcut(
            group.contraction_within_suite_rank.rank(method="first"), 4,
            labels=[1, 2, 3, 4])
        for quartile, part in group.groupby("contraction_quartile", observed=True):
            wins, n = int(part.corrected.sum()), len(part)
            lo, hi = wilson_interval(wins, n)
            quartile_rows.append({"window": window, "contraction_quartile": int(quartile),
                                  "n_failures": n, "n_corrected": wins,
                                  "correction_rate": wins / n,
                                  "ci_low": lo, "ci_high": hi,
                                  "mean_within_suite_rank":
                                      part.contraction_within_suite_rank.mean()})
        trend = group.copy()
        n_trend_bins = min(10, len(trend))
        trend["contraction_bin"] = pd.qcut(
            trend.contraction_normalized_slope.rank(method="first"), n_trend_bins,
            labels=False) + 1
        for bin_index, part in trend.groupby("contraction_bin", sort=True):
            wins, n = int(part.corrected.sum()), len(part)
            lo, hi = wilson_interval(wins, n)
            trend_rows.append({
                "window": window, "contraction_bin": int(bin_index),
                "n_failures": n, "n_corrected": wins,
                "correction_rate": wins / n, "ci_low": lo, "ci_high": hi,
                "score_min": float(part.contraction_normalized_slope.min()),
                "score_median": float(part.contraction_normalized_slope.median()),
                "score_max": float(part.contraction_normalized_slope.max()),
            })
        from sklearn.metrics import roc_curve
        for metric in ("contraction_normalized_slope", "contraction_within_suite_rank"):
            valid = group[["corrected", metric]].dropna()
            if valid.corrected.nunique() != 2:
                continue
            fpr, tpr, thresholds = roc_curve(valid.corrected.astype(int), valid[metric])
            roc_rows.extend({"window": window, "metric": metric,
                             "fpr": x, "tpr": y, "threshold": threshold}
                            for x, y, threshold in zip(fpr, tpr, thresholds))
    return {
        "expanded_contraction_episode": features,
        "expanded_contraction_summary": pd.DataFrame(summaries),
        "expanded_contraction_by_suite": pd.DataFrame(suites),
        "expanded_contraction_quartiles": pd.DataFrame(quartile_rows),
        "expanded_contraction_trend": pd.DataFrame(trend_rows),
        "expanded_contraction_roc_curves": pd.DataFrame(roc_rows),
    }


def _paired_uncertainty(rollouts: pd.DataFrame) -> pd.DataFrame:
    observed = rollouts[rollouts.method == Method.UNCERTAINTY]
    refined = rollouts[rollouts.method == Method.REFINEMENT]
    return observed[PAIR_KEYS + ["success", "u_mean_episode"]].merge(
        refined[PAIR_KEYS + ["success"]], on=PAIR_KEYS, validate="one_to_one",
        suffixes=("_observed", "_refined"))


def uncertainty_window_sweep(rollouts: pd.DataFrame, *, grid_size: int = 25,
                             min_window: int = 10) -> pd.DataFrame:
    """Reproduce the old notebook's fixed-grid selective-refinement sweep.

    The lower grid spans [0, .06] and the upper grid spans [.01, .08]. Results with fewer
    than ``min_window`` selected identities retain their sample size but have no SR estimate.
    """
    paired = _paired_uncertainty(rollouts)
    score = paired.u_mean_episode.to_numpy(float)
    observed = paired.success_observed.to_numpy(bool)
    refined = paired.success_refined.to_numpy(bool)
    lower_bounds = np.linspace(0., .06, grid_size)
    upper_bounds = np.linspace(.01, .08, grid_size)
    baseline_sr = float(observed.mean())
    rows = []
    for lower_index, lower in enumerate(lower_bounds):
        for upper_index, upper in enumerate(upper_bounds):
            if upper <= lower:
                continue
            selected = (score >= lower) & (score <= upper)
            policy = np.where(selected, refined, observed)
            eligible = int(selected.sum()) >= min_window
            rows.append({
                "lower": float(lower), "upper": float(upper),
                "lower_grid_index": lower_index, "upper_grid_index": upper_index,
                "n_refined": int(selected.sum()), "coverage_refined": float(selected.mean()),
                "eligible": eligible, "baseline_sr": baseline_sr,
                "base_sr_window": float(observed[selected].mean()) if selected.any() else math.nan,
                "ref_sr_window": float(refined[selected].mean()) if selected.any() else math.nan,
                "selective_sr": float(policy.mean()) if eligible else math.nan,
                "total_sr_policy": float(policy.mean()) if eligible else math.nan,
                "total_sr_change": float(policy.mean() - baseline_sr) if eligible else math.nan,
                "delta_pp": float(100 * (policy.mean() - baseline_sr)) if eligible else math.nan,
                "selected_F_to_S": int((selected & ~observed & refined).sum()),
                "selected_S_to_F": int((selected & observed & ~refined).sum()),
                "analysis_type": "exploratory_in_sample",
            })
    return pd.DataFrame(rows)


def optimal_window_tables(rollouts: pd.DataFrame, sweep: pd.DataFrame,
                          *, minimum_selected: int = 20) -> dict[str, pd.DataFrame]:
    """Select the best in-sample window and summarize its 13-suite policy effect."""
    eligible = sweep[(sweep.n_refined >= minimum_selected) & sweep.delta_pp.notna()].copy()
    if eligible.empty:
        raise ValueError(f"no uncertainty window selected at least {minimum_selected} identities")
    ranked = eligible.sort_values(
        ["delta_pp", "n_refined", "lower", "upper"],
        ascending=[False, False, True, True]).reset_index(drop=True)
    top = ranked.head(10).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    bottom = ranked.tail(10).sort_values(
        ["delta_pp", "n_refined"], ascending=[True, False]).reset_index(drop=True)
    bottom.insert(0, "rank", np.arange(1, len(bottom) + 1))
    extrema = pd.concat([top.assign(rank_group="top"),
                         bottom.assign(rank_group="bottom")], ignore_index=True)

    best = ranked.iloc[0]
    paired = _paired_uncertainty(rollouts).copy()
    paired["selected"] = paired.u_mean_episode.between(best.lower, best.upper)
    paired["selective_success"] = np.where(
        paired.selected, paired.success_refined, paired.success_observed).astype(bool)
    rows = []
    for suite, group in paired.groupby("suite", sort=True):
        observed = group.success_observed.astype(bool).to_numpy()
        refined = group.success_refined.astype(bool).to_numpy()
        policy = group.selective_success.astype(bool).to_numpy()
        selected = group.selected.to_numpy(bool)
        counts = paired_counts(observed, policy)
        lo, hi = paired_bootstrap_ci(observed, policy, n_boot=5000)
        rows.append({
            "suite": suite, "n": len(group), "n_refined": int(selected.sum()),
            "coverage_refined": float(selected.mean()),
            "observed_sr": float(observed.mean()), "refine_all_sr": float(refined.mean()),
            "selective_sr": float(policy.mean()),
            "delta_pp": float(100 * (policy.mean() - observed.mean())),
            "delta_ci_low_pp": float(100 * lo), "delta_ci_high_pp": float(100 * hi),
            **counts, "lower": float(best.lower), "upper": float(best.upper),
            "analysis_type": "exploratory_in_sample",
        })
    optimal = pd.DataFrame([best.to_dict()])
    optimal.insert(0, "selection_rule", f"max delta_pp among n_refined >= {minimum_selected}")
    return {
        "expanded_uncertainty_window_extrema": extrema,
        "expanded_uncertainty_optimal_window": optimal,
        "expanded_uncertainty_optimal_by_suite": pd.DataFrame(rows),
    }


def dimensional_isolation(rollouts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Old-notebook per-dimension AUC plus compact action-subspace comparisons."""

    def rank_auc(labels, scores) -> float:
        labels = np.asarray(labels, bool)
        scores = pd.Series(np.asarray(scores, float))
        positive, negative = int(labels.sum()), int((~labels).sum())
        if not positive or not negative:
            return math.nan
        rank_sum = float(scores.rank(method="average").to_numpy()[labels].sum())
        return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)

    observed = rollouts[rollouts.method == Method.UNCERTAINTY].copy()
    observed["fail"] = (~observed.success.astype(bool)).astype(int)
    definitions = {
        **{name: [index] for index, name in enumerate(DIM_NAMES)},
        "position": [0, 1, 2], "rotation": [3, 4, 5],
        "position+gripper": [0, 1, 2, 6], "all_dims": list(range(7)),
    }
    summary_rows, suite_rows = [], []
    for score_name, indices in definitions.items():
        columns = [f"u_mean_d{index}" for index in indices]
        score = observed[columns].mean(axis=1)
        values = []
        for suite, group in observed.assign(_score=score).groupby("suite", sort=True):
            valid = group[["fail", "_score"]].dropna()
            auc = (rank_auc(valid.fail, valid._score)
                   if len(valid) >= 10 and valid.fail.nunique() == 2 else math.nan)
            suite_rows.append({"score": score_name, "suite": suite, "n": len(valid),
                               "roc_auc": auc})
            if np.isfinite(auc):
                values.append(auc)
        valid = observed.assign(_score=score)[["fail", "_score"]].dropna()
        pooled_auc = (rank_auc(valid.fail, valid._score)
                      if valid.fail.nunique() == 2 else math.nan)
        summary_rows.append({
            "score": score_name,
            "score_group": "dimension" if len(indices) == 1 else "subspace",
            "dimensions": "+".join(DIM_NAMES[index] for index in indices),
            "n": len(valid), "n_suites": len(values),
            "mean_within_suite_auc": float(np.mean(values)) if values else math.nan,
            "sd_within_suite_auc": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
            "pooled_auc": pooled_auc,
        })
    return {"expanded_dimensional_isolation": pd.DataFrame(summary_rows),
            "expanded_dimensional_isolation_by_suite": pd.DataFrame(suite_rows)}


def refinement_effect_by_uncertainty_bin(rollouts: pd.DataFrame,
                                         *, n_bins: int = 10) -> pd.DataFrame:
    """Paired observed-to-refined effect within observed-uncertainty quantile bins."""
    paired = _paired_uncertainty(rollouts).copy()
    paired["uncertainty_bin"] = pd.qcut(
        paired.u_mean_episode.rank(method="first"), n_bins, labels=False) + 1
    rows = []
    for bin_index, group in paired.groupby("uncertainty_bin", sort=True):
        observed = group.success_observed.astype(bool).to_numpy()
        refined = group.success_refined.astype(bool).to_numpy()
        counts = paired_counts(observed, refined)
        lo, hi = paired_bootstrap_ci(observed, refined, n_boot=5000)
        rows.append({
            "uncertainty_bin": int(bin_index), "n": len(group),
            "u_min": float(group.u_mean_episode.min()),
            "u_median": float(group.u_mean_episode.median()),
            "u_max": float(group.u_mean_episode.max()),
            "observed_sr": float(observed.mean()), "refined_sr": float(refined.mean()),
            "delta_pp": float(100 * (refined.mean() - observed.mean())),
            "delta_ci_low_pp": float(100 * lo), "delta_ci_high_pp": float(100 * hi),
            **counts,
        })
    return pd.DataFrame(rows)


def analyze(rollouts: pd.DataFrame, steps: pd.DataFrame,
            vectors: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    frame = pro.annotate(rollouts)
    features = episode_contraction(frame, vectors)
    sweep = uncertainty_window_sweep(frame)
    tables = {
        **pro.success_and_pairs(frame),
        **pro.detector(frame, steps),
        **contraction_analysis(features),
        **dimensional_isolation(frame),
        "expanded_uncertainty_window_sweep": sweep,
        **optimal_window_tables(frame, sweep),
        "expanded_refinement_effect_by_uncertainty_bin":
            refinement_effect_by_uncertainty_bin(frame),
    }
    state = {
        "status": "available", "cohort": "expanded_13_suite_k5",
        "contraction": "available_from_observed_arm_u_iter",
        "dimensional_isolation": "available_from_rollout_u_mean_d0_to_d6",
        "multimodal_pca": "not_available_compute_multimodal_false",
        "raw_ahats_geometry": "not_needed",
        "matched_compute_control": (
            "available" if (frame.method == Method.EXTRA_STEPS).any() else "deferred"),
    }
    return tables, state
