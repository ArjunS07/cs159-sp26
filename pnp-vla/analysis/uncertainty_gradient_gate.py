"""Exact-matched analysis for the online U20 decoded-action displacement gate."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from analysis.statistics import discordant_test, paired_bootstrap_ci, paired_counts
from analysis.uncertainty_gradient import (
    fetch_direct_gradient_rows, match_direct_gradient_cohort)
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS
from pnp.uncertainty_gradient_gate_experiment import (
    U20_ACTION_GATE_EXPERIMENT, U20_ACTION_GATE_THRESHOLDS,
    build_u20_action_gate_methods)


GATE_ARMS = (Method.U20_GRADIENT_GATE_015, Method.U20_GRADIENT_GATE_020)
GATE_LABELS = {
    Method.U20_GRADIENT_GATE_015: "online gate <= 0.015",
    Method.U20_GRADIENT_GATE_020: "online gate <= 0.020",
}


def fetch_action_gate_rows(
        store, *, experiment=U20_ACTION_GATE_EXPERIMENT,
        thresholds=U20_ACTION_GATE_THRESHOLDS) -> pd.DataFrame:
    methods = build_u20_action_gate_methods(thresholds)
    hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    rows = pd.DataFrame(store.fetch_all(
        "rollouts",
        "rollout_id,suite,task_idx,episode_idx,init_state_hash,method,config_hash,"
        "status,success,n_steps,u_mean_episode,ms_candidate_u,error_msg",
        configure=lambda query: query.eq("experiment", experiment).in_(
            "method", list(GATE_ARMS)),
        order_by=("rollout_id",)))
    if rows.empty:
        raise ValueError(f"no rows found for experiment {experiment!r}")
    expected = rows.method.map(hashes)
    return rows[rows.config_hash.eq(expected)].copy()


def _identity_set(frame):
    return set(frame[DIVERSITY_PAIR_KEYS].itertuples(index=False, name=None))


def _parse_gate(value):
    empty = {
        "gate_decisions": 0, "gate_accepted": 0, "gate_accept_rate": math.nan,
        "mean_action_rms": math.nan, "median_action_rms": math.nan,
        "p90_action_rms": math.nan, "mean_first_action_l2": math.nan,
        "gripper_disagreement_rate": math.nan,
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
    records = [row for row in (payload.get("action_gate_records") or [])
               if isinstance(row, dict)]
    if not records:
        return empty
    rms = np.asarray([float(row["action_rms"]) for row in records], float)
    first = np.asarray([float(row["first_action_l2"]) for row in records], float)
    accepted = np.asarray([bool(row["accepted"]) for row in records], bool)
    gripper = np.asarray(
        [bool(row.get("gripper_sign_disagreement", False)) for row in records], bool)
    return {
        "gate_decisions": len(records),
        "gate_accepted": int(accepted.sum()),
        "gate_accept_rate": float(accepted.mean()),
        "mean_action_rms": float(rms.mean()),
        "median_action_rms": float(np.median(rms)),
        "p90_action_rms": float(np.quantile(rms, .9)),
        "mean_first_action_l2": float(first.mean()),
        "gripper_disagreement_rate": float(gripper.mean()),
    }


def match_action_gate_cohort(
        store, *, expected_identities=220, require_complete=False):
    direct_rows = fetch_direct_gradient_rows(store)
    direct, direct_coverage = match_direct_gradient_cohort(
        direct_rows, expected_identities=expected_identities,
        require_complete=require_complete)
    rows = fetch_action_gate_rows(store)
    gate_frames = {}
    for method in GATE_ARMS:
        arm = rows[rows.method.eq(method) & rows.status.eq("completed")].copy()
        if arm.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"duplicate completed identities in {method}")
        gate_frames[method] = arm
    sets = {
        "historical": _identity_set(direct),
        **{method: _identity_set(frame) for method, frame in gate_frames.items()}}
    common = set.intersection(*sets.values())
    if not common:
        raise ValueError("no identities have historical and both online-gate arms")
    if require_complete and (
            len(common) != expected_identities
            or any(values != common for values in sets.values())):
        raise ValueError(
            f"expected {expected_identities} identical identities in every arm; "
            f"found { {name: len(values) for name, values in sets.items()} }, "
            f"common={len(common)}")

    keep = direct[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(common)
    paired = direct[keep].copy().sort_values(DIVERSITY_PAIR_KEYS).reset_index(drop=True)
    paired = paired[DIVERSITY_PAIR_KEYS + [
        "baseline_success", "gradient_success",
        "baseline_rollout_id", "gradient_rollout_id"]]
    coverage = [
        {"arm": "stock baseline", "completed_unique_identities": len(sets["historical"]),
         "matched_identities_used": len(common)},
        {"arm": "ungated U20 gradient", "completed_unique_identities": len(sets["historical"]),
         "matched_identities_used": len(common)}]
    for method, prefix in (
            (Method.U20_GRADIENT_GATE_015, "gate015"),
            (Method.U20_GRADIENT_GATE_020, "gate020")):
        arm = gate_frames[method]
        coverage.append({
            "arm": GATE_LABELS[method],
            "completed_unique_identities": len(sets[method]),
            "matched_identities_used": len(common)})
        keep = arm[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(common)
        arm = arm[keep].sort_values(DIVERSITY_PAIR_KEYS).reset_index(drop=True)
        telemetry = pd.DataFrame(
            arm.ms_candidate_u.map(_parse_gate).tolist(), index=arm.index)
        arm = pd.concat([
            arm[DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]], telemetry], axis=1)
        arm = arm.rename(columns={
            "rollout_id": f"{prefix}_rollout_id",
            "success": f"{prefix}_success",
            **{column: f"{prefix}_{column}" for column in telemetry.columns}})
        paired = paired.merge(arm, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    for column in (
            "baseline_success", "gradient_success",
            "gate015_success", "gate020_success"):
        paired[column] = paired[column].astype(bool)
    return paired, pd.DataFrame(coverage), direct_coverage


def success_table(paired):
    return pd.DataFrame([{
        "arm": label,
        "matched_episodes": len(paired),
        "successes": int(paired[column].sum()),
        "success_rate_pct": 100 * paired[column].mean(),
    } for label, column in (
        ("stock baseline", "baseline_success"),
        ("ungated U20 gradient", "gradient_success"),
        ("online gate <= 0.015", "gate015_success"),
        ("online gate <= 0.020", "gate020_success"))])


def _effect(paired, condition_column, reference_column, comparison):
    reference = paired[reference_column].to_numpy(bool)
    condition = paired[condition_column].to_numpy(bool)
    low, high = paired_bootstrap_ci(reference, condition, n_boot=5000)
    counts = paired_counts(reference, condition)
    return {
        "comparison": comparison, "matched_episodes": len(paired),
        "reference_sr_pct": 100 * reference.mean(),
        "condition_sr_pct": 100 * condition.mean(),
        "condition_minus_reference_pp": 100 * (condition.mean() - reference.mean()),
        "delta_ci_low_pp": 100 * low, "delta_ci_high_pp": 100 * high,
        **counts,
        "paired_p_value": discordant_test(counts["F_to_S"], counts["S_to_F"])}


def paired_effect_table(paired):
    rows = []
    for column, label in (
            ("gradient_success", "ungated U20 gradient"),
            ("gate015_success", "online gate <= 0.015"),
            ("gate020_success", "online gate <= 0.020")):
        rows.append(_effect(
            paired, column, "baseline_success", f"{label} minus stock baseline"))
    for column, label in (
            ("gate015_success", "online gate <= 0.015"),
            ("gate020_success", "online gate <= 0.020")):
        rows.append(_effect(
            paired, column, "gradient_success", f"{label} minus ungated gradient"))
    return pd.DataFrame(rows)


def suite_success_table(paired):
    return (paired.groupby("suite", sort=True)
        .agg(episodes=("suite", "size"),
             baseline_sr=("baseline_success", "mean"),
             gradient_sr=("gradient_success", "mean"),
             gate015_sr=("gate015_success", "mean"),
             gate020_sr=("gate020_success", "mean"))
        .reset_index())


def gate_telemetry_table(paired):
    rows = []
    for prefix, label in (
            ("gate015", "online gate <= 0.015"),
            ("gate020", "online gate <= 0.020")):
        decisions = paired[f"{prefix}_gate_decisions"].sum()
        accepted = paired[f"{prefix}_gate_accepted"].sum()
        rows.append({
            "arm": label, "matched_episodes": len(paired),
            "chunk_decisions": int(decisions), "gradient_chunks_accepted": int(accepted),
            "gradient_chunk_accept_rate_pct": 100 * accepted / max(decisions, 1),
            "mean_episode_action_rms": paired[f"{prefix}_mean_action_rms"].mean(),
            "median_episode_action_rms": paired[f"{prefix}_median_action_rms"].median(),
            "mean_first_action_l2": paired[f"{prefix}_mean_first_action_l2"].mean(),
            "mean_gripper_disagreement_rate_pct": (
                100 * paired[f"{prefix}_gripper_disagreement_rate"].mean())})
    return pd.DataFrame(rows)
