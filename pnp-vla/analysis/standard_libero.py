"""Configuration-aware Standard-LIBERO analyses."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pnp.config import DIM_NAMES, Method
from pnp.experiments import SCHEDULES
from .conditions import PAIR_KEYS, condition_label, schedule_family
from .statistics import (auc_metrics, bootstrap_auc, discordant_test, holm_adjust,
                         paired_bootstrap_ci, paired_counts, wilson_interval)
from .validate import pair_one_to_one


def _sr_rows(df: pd.DataFrame, group_cols: list[str], cohort: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n, successes = len(group), int(group["success"].sum())
        lo, hi = wilson_interval(successes, n)
        rows.append({**dict(zip(group_cols, keys)), "cohort": cohort, "n": n,
                     "successes": successes, "sr": successes / n,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def success_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    all_identity = df.groupby("config_hash").filter(lambda g: len(g) == 400)
    full = df[df["full_ablation_member"]]
    tables = {
        "success_all_identity": _sr_rows(all_identity, ["config_hash", "condition_label"], "all_identity"),
        "success_full_ablation": _sr_rows(full, ["config_hash", "condition_label"], "full_ablation"),
        "success_by_suite": _sr_rows(df, ["config_hash", "condition_label", "suite"], "available_identity_set"),
        "success_by_task": _sr_rows(df, ["config_hash", "condition_label", "suite", "task_idx"], "available_identity_set"),
    }
    return tables


def _observed(df):
    return df[df["method"] == Method.UNCERTAINTY]


def paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    base = _observed(df)
    rows = []
    for config_hash, condition in df[df["method"] != Method.UNCERTAINTY].groupby("config_hash", sort=True):
        base_set = base[base[PAIR_KEYS].apply(tuple, axis=1).isin(set(condition[PAIR_KEYS].apply(tuple, axis=1)))]
        joined = pair_one_to_one(base_set, condition)
        b = joined["success_baseline"].astype(bool).to_numpy()
        c = joined["success_condition"].astype(bool).to_numpy()
        counts = paired_counts(b, c)
        lo, hi = paired_bootstrap_ci(b, c)
        first = condition.iloc[0]
        rows.append({"config_hash": config_hash, "condition_label": condition_label(first),
                     "cohort": "all_identity" if len(joined) == 400 else "full_ablation",
                     "schedule_family": schedule_family(first), "n": len(joined),
                     "baseline_successes": int(b.sum()), "condition_successes": int(c.sum()),
                     "baseline_sr": b.mean(), "condition_sr": c.mean(),
                     "delta_pp": 100 * (c.mean() - b.mean()),
                     "delta_ci_low_pp": 100 * lo, "delta_ci_high_pp": 100 * hi,
                     **counts, "p_raw": discordant_test(counts["F_to_S"], counts["S_to_F"])})
    out = pd.DataFrame(rows)
    out["p_holm_eight_schedules"] = np.nan
    mask = out["condition_label"].isin([
        f"refine-last ({','.join(map(str, s))})" for s in SCHEDULES
    ]) & (out["cohort"] == "full_ablation")
    out.loc[mask, "p_holm_eight_schedules"] = holm_adjust(out.loc[mask, "p_raw"])
    return out


def _macro_suite_auc(frame: pd.DataFrame, score: str) -> tuple[float, int]:
    values = [auc_metrics(g["fail"], g[score])["roc_auc"] for _, g in frame.groupby("suite")]
    values = [x for x in values if np.isfinite(x)]
    return (float(np.mean(values)) if values else math.nan, len(values))


def detector_tables(df: pd.DataFrame, steps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    observed = _observed(df).copy()
    observed["fail"] = (~observed["success"].astype(bool)).astype(int)
    pooled = bootstrap_auc(observed["fail"], observed["u_mean_episode"])
    macro, n_suites = _macro_suite_auc(observed, "u_mean_episode")
    summary = pd.DataFrame([{**pooled, "estimate_scope": "pooled"},
                            {"estimate_scope": "macro_suite", "n": len(observed),
                             "failures": int(observed.fail.sum()), "roc_auc": macro,
                             "pr_auc": math.nan, "n_strata": n_suites}])
    suite_rows = []
    for suite, group in observed.groupby("suite"):
        suite_rows.append({"suite": suite, **bootstrap_auc(group.fail, group.u_mean_episode, n_boot=1000)})
    dof_rows = []
    for i, name in enumerate(DIM_NAMES):
        col = f"u_mean_d{i}"
        dof_rows.append({"score": name, **auc_metrics(observed.fail, observed[col])})
    position = observed[[f"u_mean_d{i}" for i in (0, 1, 2, 6)]].mean(axis=1)
    full = observed[[f"u_mean_d{i}" for i in range(7)]].mean(axis=1)
    dof_rows += [{"score": "position+gripper", **auc_metrics(observed.fail, position)},
                 {"score": "full_vector", **auc_metrics(observed.fail, full)}]

    score_dist = observed.groupby("fail")["u_mean_episode"].agg(["count", "mean", "median", "std"]).reset_index()
    suite_sr = _sr_rows(observed, ["suite"], "observed_all_identity")
    corr = observed[["u_mean_episode", "n_steps"]].corr(method="spearman").iloc[0, 1]
    confounding = pd.DataFrame([{"n": len(observed), "spearman_uncertainty_episode_length": corr}])

    reliability = observed.assign(bin=pd.qcut(observed.u_mean_episode, 10, duplicates="drop")) \
        .groupby("bin", observed=False).agg(n=("fail", "size"), mean_score=("u_mean_episode", "mean"),
                                             observed_failure_rate=("fail", "mean")).reset_index()
    reliability["bin"] = reliability["bin"].astype(str)
    ordered = observed.sort_values("u_mean_episode")
    risk_rows = []
    for coverage in np.linspace(.1, 1, 10):
        kept = ordered.iloc[:max(1, round(coverage * len(ordered)))]
        risk_rows.append({"coverage": len(kept) / len(ordered), "risk": kept.fail.mean(), "n": len(kept)})

    legacy = observed.assign(uncertainty_group=np.where(
        observed.u_mean_episode >= observed.u_mean_episode.median(), "high", "low")) \
        .groupby(["uncertainty_group", "fail"]).size().rename("n").reset_index()
    legacy["analysis_type"] = "descriptive_legacy_median_split"

    early = pd.DataFrame()
    if not steps.empty:
        joined = steps.merge(observed[["rollout_id", "fail", "suite"]], on="rollout_id", how="inner")
        ep = joined.groupby("rollout_id").agg(fail=("fail", "first"), suite=("suite", "first"),
             early_score=("u_mean", lambda x: x.iloc[:max(1, min(4, len(x)))].mean()),
             full_score=("u_mean", "mean")).reset_index()
        early = pd.DataFrame([{"window": "early", **auc_metrics(ep.fail, ep.early_score)},
                              {"window": "full_episode", **auc_metrics(ep.fail, ep.full_score)}])
    return {"detector_summary": summary, "detector_by_suite": pd.DataFrame(suite_rows),
            "detector_per_dof": pd.DataFrame(dof_rows), "detector_score_distribution": score_dist,
            "observed_success_by_suite": suite_sr, "detector_length_confounding": confounding,
            "detector_reliability": reliability, "detector_risk_coverage": pd.DataFrame(risk_rows),
            "legacy_uncertainty_taxonomy": legacy, "detector_early_window": early}


def run(df: pd.DataFrame, steps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {**success_tables(df), "paired_comparisons": paired_comparisons(df),
            **detector_tables(df, steps)}
