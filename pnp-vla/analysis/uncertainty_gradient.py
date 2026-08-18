"""Matched analysis for the online direct-U20-gradient LIBERO-PRO pilot."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from analysis.statistics import discordant_test, paired_bootstrap_ci, paired_counts
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS
from pnp.uncertainty_gradient_experiment import (
    DIRECT_U20_GRADIENT_EXPERIMENT,
    DIRECT_U20_GRADIENT_RMS,
    build_direct_u20_gradient_methods,
)


ARMS = (Method.UNCERTAINTY, Method.U20_GRADIENT, Method.LATENT_RANDOM_CONTROL)
ARM_LABELS = {
    Method.UNCERTAINTY: "unrefined baseline",
    Method.U20_GRADIENT: "true U20 gradient",
    Method.LATENT_RANDOM_CONTROL: "random control",
}


def fetch_direct_gradient_rows(
        store, *, experiment=DIRECT_U20_GRADIENT_EXPERIMENT,
        step_size=DIRECT_U20_GRADIENT_RMS) -> pd.DataFrame:
    """Fetch only rows matching the three predeclared logical configurations."""
    methods = build_direct_u20_gradient_methods(step_size)
    expected_hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    rows = pd.DataFrame(store.fetch_all(
        "rollouts",
        "rollout_id,suite,task_idx,episode_idx,init_state_hash,method,config_hash,"
        "status,success,n_steps,u_mean_episode,ms_candidate_u,error_msg",
        configure=lambda query: query.eq("experiment", experiment).in_("method", list(ARMS)),
        order_by=("rollout_id",)))
    if rows.empty:
        raise ValueError(f"no rows found for experiment {experiment!r}")
    expected = rows.method.map(expected_hashes)
    return rows[rows.config_hash.eq(expected)].copy()


def _identity_set(frame: pd.DataFrame) -> set[tuple]:
    return set(frame[DIVERSITY_PAIR_KEYS].itertuples(index=False, name=None))


def _parse_telemetry(value) -> dict:
    empty = {
        "n_updates": 0, "mean_pre_u20": math.nan, "mean_post_u20": math.nan,
        "mean_delta_u20": math.nan, "mean_update_rms": math.nan,
        "fraction_updates_lowered_u20": math.nan,
        "first_pre_u20": math.nan, "first_post_u20": math.nan,
        "first_delta_u20": math.nan,
    }
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return empty
    if not isinstance(value, dict):
        return empty
    payload = value.get("uncertainty_gradient", value)
    if not isinstance(payload, dict):
        return empty
    records = payload.get("records") or []
    records = [row for row in records if isinstance(row, dict)]
    if not records:
        return {**empty, "n_updates": int(payload.get("n_updates", 0) or 0)}

    def values(name):
        output = []
        for row in records:
            try:
                number = float(row.get(name))
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                output.append(number)
        return output

    pre, post, delta, rms = (
        values("pre_u20"), values("post_u20"), values("delta_u20"), values("update_rms"))
    first = records[0]
    return {
        "n_updates": len(records),
        "mean_pre_u20": float(np.mean(pre)) if pre else math.nan,
        "mean_post_u20": float(np.mean(post)) if post else math.nan,
        "mean_delta_u20": float(np.mean(delta)) if delta else math.nan,
        "mean_update_rms": float(np.mean(rms)) if rms else math.nan,
        "fraction_updates_lowered_u20": (
            float(np.mean(np.asarray(delta) < 0)) if delta else math.nan),
        "first_pre_u20": float(first.get("pre_u20", math.nan)),
        "first_post_u20": float(first.get("post_u20", math.nan)),
        "first_delta_u20": float(first.get("delta_u20", math.nan)),
    }


def _arm_frame(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    arm = frame[frame.method.eq(method) & frame.status.eq("completed")].copy()
    if arm.duplicated(DIVERSITY_PAIR_KEYS).any():
        duplicates = arm[arm.duplicated(DIVERSITY_PAIR_KEYS, keep=False)]
        raise ValueError(f"duplicate completed identities in {method}: {len(duplicates)} rows")
    return arm


def match_direct_gradient_cohort(
        rows: pd.DataFrame, *, expected_identities: int = 220,
        require_complete: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact three-arm matches; preview mode drops incomplete identities explicitly."""
    arms = {method: _arm_frame(rows, method) for method in ARMS}
    missing = [method for method, frame in arms.items() if frame.empty]
    if missing:
        raise ValueError(f"no completed rows for arm(s): {missing}")
    sets = {method: _identity_set(frame) for method, frame in arms.items()}
    common = set.intersection(*sets.values())
    if not common:
        raise ValueError("no identities have completed all three arms")
    if require_complete:
        counts = {method: len(values) for method, values in sets.items()}
        if len(common) != expected_identities or any(
                values != common for values in sets.values()):
            raise ValueError(
                f"expected {expected_identities} identical completed identities per arm; "
                f"found arm counts {counts}, common={len(common)}")

    coverage = pd.DataFrame([{
        "arm": ARM_LABELS[method], "completed_unique_identities": len(sets[method]),
        "matched_identities_used": len(common),
        "completed_but_not_yet_matched": len(sets[method] - common),
        "completed_success_rate_pct": 100 * arms[method].success.astype(bool).mean(),
    } for method in ARMS])

    selected = {}
    for method, arm in arms.items():
        keep = arm[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(common)
        selected[method] = arm[keep].sort_values(DIVERSITY_PAIR_KEYS).reset_index(drop=True)

    base_columns = DIVERSITY_PAIR_KEYS + ["rollout_id", "success", "n_steps", "u_mean_episode"]
    paired = selected[Method.UNCERTAINTY][base_columns].rename(columns={
        "rollout_id": "baseline_rollout_id", "success": "baseline_success",
        "n_steps": "baseline_n_steps", "u_mean_episode": "baseline_u_mean_episode"})
    for method, prefix in (
            (Method.U20_GRADIENT, "gradient"),
            (Method.LATENT_RANDOM_CONTROL, "random")):
        arm = selected[method][DIVERSITY_PAIR_KEYS + [
            "rollout_id", "success", "n_steps", "u_mean_episode", "ms_candidate_u"]].copy()
        telemetry = pd.DataFrame(
            arm.ms_candidate_u.map(_parse_telemetry).tolist(), index=arm.index)
        telemetry = telemetry.add_prefix(f"{prefix}_")
        arm = pd.concat([arm.drop(columns="ms_candidate_u"), telemetry], axis=1)
        arm = arm.rename(columns={
            "rollout_id": f"{prefix}_rollout_id", "success": f"{prefix}_success",
            "n_steps": f"{prefix}_n_steps", "u_mean_episode": f"{prefix}_u_mean_episode"})
        paired = paired.merge(arm, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    for column in ("baseline_success", "gradient_success", "random_success"):
        paired[column] = paired[column].astype(bool)
    paired["first_pre_u20_abs_difference"] = (
        paired.gradient_first_pre_u20 - paired.random_first_pre_u20).abs()
    return paired, coverage


def arm_success_table(paired: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "arm": label, "matched_episodes": len(paired),
        "successes": int(paired[column].sum()),
        "success_rate_pct": 100 * paired[column].mean(),
    } for label, column in (
        ("unrefined baseline", "baseline_success"),
        ("true U20 gradient", "gradient_success"),
        ("random control", "random_success"))])


def _effect(baseline, condition, comparison: str) -> dict:
    baseline = np.asarray(baseline, bool)
    condition = np.asarray(condition, bool)
    low, high = paired_bootstrap_ci(baseline, condition, n_boot=5000)
    counts = paired_counts(baseline, condition)
    return {
        "comparison": comparison, "matched_episodes": len(baseline),
        "baseline_sr_pct": 100 * baseline.mean(),
        "condition_sr_pct": 100 * condition.mean(),
        "condition_minus_baseline_pp": 100 * (condition.mean() - baseline.mean()),
        "delta_ci_low_pp": 100 * low, "delta_ci_high_pp": 100 * high,
        **counts,
        "paired_p_value": discordant_test(counts["F_to_S"], counts["S_to_F"]),
    }


def paired_effect_table(paired: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        _effect(paired.baseline_success, paired.gradient_success,
                "gradient minus unrefined baseline"),
        _effect(paired.baseline_success, paired.random_success,
                "random minus unrefined baseline"),
        _effect(paired.random_success, paired.gradient_success,
                "gradient minus random control"),
    ])


def suite_effect_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for suite, group in paired.groupby("suite", sort=True):
        for condition, label in (
                ("gradient_success", "gradient minus baseline"),
                ("random_success", "random minus baseline")):
            rows.append({"suite": suite, **_effect(
                group.baseline_success, group[condition], label)})
    return pd.DataFrame(rows)


def gradient_telemetry_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix, label in (("gradient", "true U20 gradient"), ("random", "random control")):
        delta = paired[f"{prefix}_mean_delta_u20"].to_numpy(float)
        finite = np.isfinite(delta)
        rows.append({
            "arm": label, "episodes": len(paired),
            "episodes_with_telemetry": int(finite.sum()),
            "total_latent_updates": int(paired[f"{prefix}_n_updates"].sum()),
            "mean_pre_u20": paired[f"{prefix}_mean_pre_u20"].mean(),
            "mean_post_u20": paired[f"{prefix}_mean_post_u20"].mean(),
            "mean_post_minus_pre_u20": np.nanmean(delta),
            "median_post_minus_pre_u20": np.nanmedian(delta),
            "episodes_with_mean_u20_reduction_pct": 100 * np.mean(delta[finite] < 0),
            "individual_updates_lowering_u20_pct": (
                100 * paired[f"{prefix}_fraction_updates_lowered_u20"].mean()),
            "mean_update_rms": paired[f"{prefix}_mean_update_rms"].mean(),
        })
    return pd.DataFrame(rows)


def telemetry_by_outcome_transition(paired: pd.DataFrame) -> pd.DataFrame:
    baseline, gradient = paired.baseline_success, paired.gradient_success
    labels = np.select(
        [~baseline & gradient, baseline & ~gradient, baseline & gradient],
        ["F→S", "S→F", "S→S"], default="F→F")
    frame = paired.assign(transition=labels)
    return (frame.groupby("transition", sort=False)
        .agg(episodes=("suite", "size"),
             first_pre_u20=("gradient_first_pre_u20", "mean"),
             mean_local_u20_change=("gradient_mean_delta_u20", "mean"),
             mean_fraction_updates_lowering_u20=(
                 "gradient_fraction_updates_lowered_u20", "mean"))
        .reset_index())


def first_u20_gate_sweep(paired: pd.DataFrame, *, grid_size: int = 41) -> pd.DataFrame:
    """Post-hoc rule: use gradient outcome only when initial pre-update U20 >= threshold."""
    scores = paired.gradient_first_pre_u20.to_numpy(float)
    finite_scores = scores[np.isfinite(scores)]
    if not len(finite_scores):
        raise ValueError("no finite first-update U20 values")
    thresholds = np.unique(np.quantile(finite_scores, np.linspace(0, 1, grid_size)))
    baseline = paired.baseline_success.to_numpy(bool)
    gradient = paired.gradient_success.to_numpy(bool)
    rows = []
    for threshold in thresholds:
        selected = np.isfinite(scores) & (scores >= threshold)
        policy = np.where(selected, gradient, baseline)
        low, high = paired_bootstrap_ci(baseline, policy, n_boot=5000)
        counts = paired_counts(baseline, policy)
        rows.append({
            "threshold": float(threshold),
            "episodes_in_sr_denominator": len(paired),
            "episodes_using_gradient": int(selected.sum()),
            "baseline_sr_pct": 100 * baseline.mean(),
            "gated_policy_sr_pct": 100 * policy.mean(),
            "gated_minus_baseline_pp": 100 * (policy.mean() - baseline.mean()),
            "delta_ci_low_pp": 100 * low, "delta_ci_high_pp": 100 * high,
            "selected_F_to_S": counts["F_to_S"],
            "selected_S_to_F": counts["S_to_F"],
        })
    return pd.DataFrame(rows)
