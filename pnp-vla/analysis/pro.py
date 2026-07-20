"""Deduplicated, configuration-aware LIBERO-PRO analysis."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pnp.config import DIM_NAMES, Method
from .conditions import PAIR_KEYS, condition_label
from .statistics import (auc_metrics, bootstrap_auc, discordant_test, holm_adjust,
                         paired_bootstrap_ci, paired_counts, wilson_interval)
from .validate import pair_one_to_one


def base_suite(suite: str) -> str | None:
    for name in ("libero_object", "libero_goal", "libero_spatial", "libero_10"):
        if suite.startswith(name + "_"):
            return name
    return None


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["condition_label"] = [condition_label(r) for r in out.to_dict("records")]
    out["base_suite"] = out["suite"].map(base_suite)
    out["perturbation_group"] = np.where(
        out["suite_family"] == "position_perturb",
        "position_" + out["perturb_axis"].fillna("unknown").astype(str) + "_" +
        out["perturb_strength"].map(lambda x: f"{x:g}" if pd.notna(x) else "unknown"),
        "distractor_" + out["distractor_object"].fillna(
            out["suite"].str.replace(r"^libero_[^_]+_with_", "", regex=True)))
    return out


def _success(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(groups, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n, successes = len(group), int(group.success.sum())
        lo, hi = wilson_interval(successes, n)
        rows.append({**dict(zip(groups, keys)), "n": n, "successes": successes,
                     "sr": successes / n, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _paired_rows(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    base = df[df.method == Method.UNCERTAINTY]
    rows = []
    for config_hash, condition in df[df.method != Method.UNCERTAINTY].groupby("config_hash"):
        joined = pair_one_to_one(base, condition)
        extra_groups = [column for column in groups if column not in PAIR_KEYS]
        if extra_groups:
            joined = joined.merge(base[PAIR_KEYS + extra_groups], on=PAIR_KEYS, validate="one_to_one")
        first = condition.iloc[0]
        for keys, group in joined.groupby(groups, dropna=False, sort=True) if groups else [((), joined)]:
            keys = keys if isinstance(keys, tuple) else (keys,)
            b = group.success_baseline.astype(bool).to_numpy()
            c = group.success_condition.astype(bool).to_numpy()
            counts = paired_counts(b, c); lo, hi = paired_bootstrap_ci(b, c)
            rows.append({"config_hash": config_hash, "condition_label": condition_label(first),
                         **dict(zip(groups, keys)), "n": len(group),
                         "baseline_successes": int(b.sum()), "condition_successes": int(c.sum()),
                         "baseline_sr": b.mean(), "condition_sr": c.mean(),
                         "delta_pp": 100 * (c.mean() - b.mean()),
                         "delta_ci_low_pp": 100 * lo, "delta_ci_high_pp": 100 * hi,
                         **counts, "p_raw": discordant_test(counts["F_to_S"], counts["S_to_F"])})
    out = pd.DataFrame(rows)
    if not groups and len(out):
        out["p_holm_primary"] = holm_adjust(out.p_raw)
    return out


def success_and_pairs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "pro_success_overall": _success(df, ["config_hash", "condition_label"]),
        "pro_success_by_suite": _success(df, ["config_hash", "condition_label", "suite"]),
        "pro_success_by_task": _success(df, ["config_hash", "condition_label", "suite", "task_idx"]),
        "pro_success_by_family": _success(df, ["config_hash", "condition_label", "suite_family"]),
        "pro_success_by_perturbation": _success(
            df, ["config_hash", "condition_label", "suite_family", "perturb_axis",
                 "perturb_strength", "distractor_object", "perturbation_group"]),
        "pro_paired_overall": _paired_rows(df, []),
        "pro_paired_by_suite": _paired_rows(df, ["suite"]),
        "pro_paired_by_family": _paired_rows(df, ["suite_family"]),
    }


def _risk_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("u_mean_episode")
    rows = []
    for coverage in np.linspace(.1, 1, 10):
        kept = ordered.iloc[:max(1, round(coverage * len(ordered)))]
        rows.append({"coverage": len(kept) / len(ordered), "risk": kept.fail.mean(), "n": len(kept)})
    return pd.DataFrame(rows)


def detector(df: pd.DataFrame, steps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    observed = df[df.method == Method.UNCERTAINTY].copy()
    observed["fail"] = (~observed.success.astype(bool)).astype(int)
    pooled = bootstrap_auc(observed.fail, observed.u_mean_episode)
    suite_rows, aucs, roc_rows = [], [], []
    from sklearn.metrics import roc_curve
    for suite, group in observed.groupby("suite"):
        metrics = bootstrap_auc(group.fail, group.u_mean_episode, n_boot=1000)
        suite_rows.append({"suite": suite, **metrics}); aucs.append(metrics["roc_auc"])
        fpr, tpr, thresholds = roc_curve(group.fail, group.u_mean_episode)
        roc_rows.extend({"suite": suite, "fpr": x, "tpr": y, "threshold": t}
                        for x, y, t in zip(fpr, tpr, thresholds))
    fpr, tpr, thresholds = roc_curve(observed.fail, observed.u_mean_episode)
    roc_rows.extend({"suite": "pooled", "fpr": x, "tpr": y, "threshold": t}
                    for x, y, t in zip(fpr, tpr, thresholds))
    summary = pd.DataFrame([{**pooled, "estimate_scope": "pooled"},
                            {"estimate_scope": "macro_suite", "n": len(observed),
                             "failures": int(observed.fail.sum()), "roc_auc": float(np.mean(aucs)),
                             "pr_auc": math.nan, "n_strata": len(aucs)}])
    grouped = []
    for columns in (["suite_family"], ["perturb_axis"], ["perturb_strength"], ["perturbation_group"]):
        for keys, group in observed.groupby(columns, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            grouped.append({"grouping": "+".join(columns), "group": "+".join(map(str, keys)),
                            **auc_metrics(group.fail, group.u_mean_episode)})
    dof = []
    for i, name in enumerate(DIM_NAMES):
        dof.append({"score": name, **auc_metrics(observed.fail, observed[f"u_mean_d{i}"])})
    dof.extend([
        {"score": "position+gripper", **auc_metrics(
            observed.fail, observed[[f"u_mean_d{i}" for i in (0, 1, 2, 6)]].mean(axis=1))},
        {"score": "full_vector", **auc_metrics(
            observed.fail, observed[[f"u_mean_d{i}" for i in range(7)]].mean(axis=1))},
    ])
    score_distribution = observed.groupby("fail").u_mean_episode.agg(
        count="count", mean="mean", median="median", std="std",
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75)).reset_index()
    length = pd.DataFrame([{"n": len(observed), "spearman_uncertainty_episode_length":
                            observed[["u_mean_episode", "n_steps"]].corr(method="spearman").iloc[0, 1]}])
    early, euler, time = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not steps.empty:
        joined = steps.merge(observed[["rollout_id", "fail", "suite"]], on="rollout_id", how="inner")
        full_ep = joined.groupby("rollout_id").agg(fail=("fail", "first"), full_score=("u_mean", "mean"))
        early_ep = joined[joined.euler_step <= 4].groupby("rollout_id").u_mean.mean().rename("early_score")
        ep = full_ep.join(early_ep, how="left")
        early = pd.DataFrame([{"window": "early", **auc_metrics(ep.fail, ep.early_score)},
                              {"window": "full_episode", **auc_metrics(ep.fail, ep.full_score)}])
        euler = joined.groupby(["suite", "euler_step", "fail"]).u_mean.agg(
            ["count", "mean", "std"]).reset_index()
        euler["sem"] = euler["std"] / np.sqrt(euler["count"])
        maxima = joined.groupby("rollout_id").chunk_idx.transform("max").clip(lower=1)
        joined["episode_progress_bin"] = np.minimum(
            9, np.floor(10 * joined.chunk_idx / (maxima + 1)).astype(int))
        time = joined.groupby(["suite", "episode_progress_bin", "fail"]).u_mean.agg(
            ["count", "mean", "std"]).reset_index()
        time["sem"] = time["std"] / np.sqrt(time["count"])
    return {"pro_detector_summary": summary, "pro_detector_by_suite": pd.DataFrame(suite_rows),
            "pro_detector_by_group": pd.DataFrame(grouped), "pro_detector_roc_curves": pd.DataFrame(roc_rows),
            "pro_detector_per_dof": pd.DataFrame(dof), "pro_detector_score_distribution": score_distribution,
            "pro_detector_early_window": early, "pro_detector_length_confounding": length,
            "pro_detector_risk_coverage": _risk_coverage(observed),
            "pro_detector_euler_profile": euler, "pro_detector_time_profile": time}


def _paired_for_policy(df: pd.DataFrame, score_name: str = "full_vector") -> pd.DataFrame:
    base = df[df.method == Method.UNCERTAINTY]
    ref = df[df.method == Method.REFINEMENT]
    score = (base.u_mean_episode if score_name == "full_vector" else
             base[[f"u_mean_d{i}" for i in (0, 1, 2, 6)]].mean(axis=1))
    base_columns = base[PAIR_KEYS + ["success"]].copy()
    base_columns["score"] = score
    joined = base_columns.merge(
        ref[PAIR_KEYS + ["success"]], on=PAIR_KEYS, suffixes=("_baseline", "_refinement"),
        validate="one_to_one")
    return joined


def legacy_threshold_sweep(df: pd.DataFrame, *, score_name: str = "full_vector") -> pd.DataFrame:
    paired = _paired_for_policy(df, score_name)
    if score_name == "full_vector":
        lower, upper = np.linspace(0, .06, 25), np.linspace(.01, .08, 25)
    else:
        bounds = np.unique(paired.score.quantile(np.linspace(0, 1, 25)).to_numpy())
        lower, upper = bounds, bounds
    rows = []
    baseline_sr = paired.success_baseline.mean()
    for lo in lower:
        for hi in upper:
            if hi <= lo:
                continue
            selected = paired["score"].between(lo, hi)
            if selected.sum() < 10:
                continue
            outcome = np.where(selected, paired.success_refinement, paired.success_baseline)
            rows.append({"lower": lo, "upper": hi, "n_refined": int(selected.sum()),
                         "coverage": selected.mean(), "baseline_sr": baseline_sr,
                         "policy_sr": outcome.mean(), "delta_pp": 100 * (outcome.mean() - baseline_sr),
                         "score_name": score_name,
                         "analysis_type": "legacy_exploratory_in_sample"})
    return pd.DataFrame(rows)


def cross_validated_policy(df: pd.DataFrame, folds: int = 5,
                           score_name: str = "full_vector") -> pd.DataFrame:
    paired = _paired_for_policy(df, score_name).reset_index(drop=True)
    paired["fail"] = (~paired.success_baseline.astype(bool)).astype(int)
    paired["fold"] = -1
    for _, indices in paired.groupby(["suite", "fail"]).groups.items():
        for offset, idx in enumerate(sorted(indices)):
            paired.loc[idx, "fold"] = offset % folds
    rows = []
    for fold in range(folds):
        train, test = paired[paired.fold != fold], paired[paired.fold == fold]
        bounds = np.unique(train.score.quantile(np.linspace(0, 1, 21)).to_numpy())
        candidates = []
        for lo in bounds:
            for hi in bounds:
                selected = train.score.between(lo, hi)
                if hi <= lo or selected.sum() < 20 or selected.mean() < .1:
                    continue
                outcome = np.where(selected, train.success_refinement, train.success_baseline)
                candidates.append((outcome.mean() - train.success_baseline.mean(), lo, hi))
        _, lo, hi = max(candidates, key=lambda x: (x[0], -x[1], x[2]))
        selected = test.score.between(lo, hi)
        outcome = np.where(selected, test.success_refinement, test.success_baseline).astype(bool)
        base = test.success_baseline.astype(bool).to_numpy()
        counts = paired_counts(base, outcome)
        rows.append({"fold": fold, "n": len(test), "lower_train_only": lo, "upper_train_only": hi,
                     "n_refined": int(selected.sum()), "coverage": selected.mean(),
                     "baseline_sr": base.mean(), "policy_sr": outcome.mean(),
                     "delta_pp": 100 * (outcome.mean() - base.mean()), **counts,
                     "score_name": score_name,
                     "analysis_type": "cross_validated_held_out"})
    return pd.DataFrame(rows)


def threshold_summaries(legacy: pd.DataFrame,
                        cross_validated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep concise decision tables alongside the complete exploratory sweep."""
    extrema = []
    for score_name, group in legacy.groupby("score_name"):
        ranked = group.sort_values(["delta_pp", "n_refined"], ascending=[False, False])
        extrema.extend(ranked.head(10).assign(rank_group="top").to_dict("records"))
        extrema.extend(ranked.tail(10).sort_values("delta_pp").assign(
            rank_group="bottom").to_dict("records"))
    summary = cross_validated.groupby("score_name", sort=True).agg(
        folds=("fold", "count"), n_held_out=("n", "sum"),
        mean_coverage=("coverage", "mean"), mean_delta_pp=("delta_pp", "mean"),
        sd_delta_pp=("delta_pp", "std"), F_to_S=("F_to_S", "sum"),
        S_to_F=("S_to_F", "sum")).reset_index()
    summary["analysis_type"] = "cross_validated_held_out_fold_summary"
    return {"pro_legacy_threshold_extrema": pd.DataFrame(extrema),
            "pro_threshold_cross_validation_summary": summary}


def standard_transfer(pro_df: pd.DataFrame, standard_df: pd.DataFrame | None) -> pd.DataFrame:
    if standard_df is None or standard_df.empty:
        return pd.DataFrame(columns=["base_suite", "standard_n", "pro_n", "standard_sr", "pro_sr", "delta_pp"])
    standard = standard_df[standard_df.method == Method.UNCERTAINTY]
    pro_obs = pro_df[pro_df.method == Method.UNCERTAINTY]
    rows = []
    rng = np.random.default_rng(159)
    for suite, group in pro_obs.groupby("base_suite"):
        reference = standard[standard.suite == suite]
        a, b = reference.success.astype(float).to_numpy(), group.success.astype(float).to_numpy()
        boot = b[rng.integers(0, len(b), (5000, len(b)))].mean(1) - a[
            rng.integers(0, len(a), (5000, len(a)))].mean(1)
        rows.append({"base_suite": suite, "standard_n": len(a), "pro_n": len(b),
                     "standard_sr": a.mean(), "pro_sr": b.mean(),
                     "delta_pp": 100 * (b.mean() - a.mean()),
                     "delta_ci_low_pp": 100 * np.quantile(boot, .025),
                     "delta_ci_high_pp": 100 * np.quantile(boot, .975),
                     "comparison_type": "unpaired_base_suite"})
    return pd.DataFrame(rows)


def standard_threshold_transfer(pro_df: pd.DataFrame,
                                standard_df: pd.DataFrame | None) -> pd.DataFrame:
    """Select a detector threshold using standard labels and evaluate only on PRO."""
    if standard_df is None or standard_df.empty:
        return pd.DataFrame(columns=["scope", "threshold_standard_only", "n", "tp", "fp", "fn", "tn"])
    standard = standard_df[standard_df.method == Method.UNCERTAINTY].copy()
    standard["fail"] = (~standard.success.astype(bool)).astype(int)
    candidates = np.unique(np.quantile(standard.u_mean_episode, np.linspace(.02, .98, 97)))
    best = (-1., candidates[0])
    for threshold in candidates:
        pred, y = standard.u_mean_episode.to_numpy() >= threshold, standard.fail.to_numpy(bool)
        tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, threshold)
    pro = pro_df[pro_df.method == Method.UNCERTAINTY].copy()
    pro["fail"] = (~pro.success.astype(bool)).astype(int)
    rows = []
    for scope, group in [("pooled", pro), *pro.groupby("suite")]:
        pred, y = group.u_mean_episode.to_numpy() >= best[1], group.fail.to_numpy(bool)
        tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
        rows.append({"scope": scope, "threshold_standard_only": best[1], "n": len(group),
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                     "specificity": tn / max(1, tn + fp),
                     "accuracy": (tp + tn) / max(1, len(group)),
                     "selection_dataset": "standard_libero",
                     "evaluation_dataset": "libero_pro"})
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame, steps: pd.DataFrame, vectors: pd.DataFrame,
            standard_df: pd.DataFrame | None = None,
            artifact_validation: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    df = annotate(df)
    legacy = pd.concat([legacy_threshold_sweep(df, score_name=name)
                        for name in ("full_vector", "position+gripper")], ignore_index=True)
    cross_validated = pd.concat([cross_validated_policy(df, score_name=name)
                                 for name in ("full_vector", "position+gripper")], ignore_index=True)
    tables = {**success_and_pairs(df), **detector(df, steps),
              "pro_legacy_threshold_sweep": legacy,
              "pro_threshold_cross_validation": cross_validated,
              **threshold_summaries(legacy, cross_validated),
              "pro_standard_degradation": standard_transfer(df, standard_df),
              "pro_detector_standard_threshold_transfer": standard_threshold_transfer(df, standard_df),
              "pro_artifact_coverage": pd.DataFrame([
                  {"artifact_type": name, "referenced": value.get("referenced", 0),
                   "verified": value.get("verified", 0), "status": value.get("status")}
                  for name, value in (artifact_validation or {}).items()])}
    state = {"status": "available", "canonical_cohort": "complete",
             "expanded_cohort": "not_available_incomplete_6_of_16_suites",
             "cross_model": "not_available", "pcp_live_evaluation": "not_available",
             "sarle": "not_available_K3", "raw_ahats_geometry": "not_available"}
    return tables, state
