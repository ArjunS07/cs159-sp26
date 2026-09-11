"""Analysis helpers for the held-out stock/Q10/Q50 LIBERO-PRO evaluation."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from analysis.suffix_sensitivity import bootstrap_rank_auc, summarize_pair
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS


ARMS = (Method.VANILLA, Method.QPLANNING_Q10, Method.QPLANNING_Q50)
ARM_LABELS = {
    Method.VANILLA: "stock VLA",
    Method.QPLANNING_Q10: "Q10 planner",
    Method.QPLANNING_Q50: "Q50 planner",
}
Q_SIGNALS = (
    "negative_q_weighted", "negative_q_max", "q_std", "q_range",
    "elite_entropy", "first10_pairwise_rms", "full50_pairwise_rms",
    "tail_minus_prefix_rms",
)


def _json_object(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value or {}


def validate_qplanning_heldout_cohort(
        rows: pd.DataFrame, *, expected_identities: int = 160,
        require_complete: bool = True) -> dict[str, pd.DataFrame]:
    """Select one exact completed row per identity and arm, then exact-match arms."""
    required = set(DIVERSITY_PAIR_KEYS + [
        "rollout_id", "status", "success", "method", "config_hash",
        "config_json", "ms_candidate_u"])
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"held-out rows are missing columns: {sorted(missing)}")
    frame = rows[rows.status.eq("completed") & rows.method.isin(ARMS)].copy()
    missing_arms = set(ARMS) - set(frame.method.unique())
    if missing_arms:
        raise ValueError(f"missing held-out arms: {sorted(missing_arms)}")

    raw_arms, key_sets = {}, {}
    for method in ARMS:
        arm = frame[frame.method.eq(method)].copy()
        if arm.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"duplicate completed identities in {method}")
        hashes = arm.config_hash.dropna().unique()
        if len(hashes) != 1:
            raise ValueError(f"{method} contains {len(hashes)} behavior hashes")
        raw_arms[method] = arm
        key_sets[method] = set(map(
            tuple, arm[DIVERSITY_PAIR_KEYS].itertuples(index=False, name=None)))
    common = set.intersection(*key_sets.values())
    counts = {method: len(keys) for method, keys in key_sets.items()}
    if require_complete and (len(common) != expected_identities
                             or any(count != expected_identities
                                    for count in counts.values())):
        raise ValueError(
            f"expected {expected_identities} exact three-arm identities; "
            f"arm counts={counts}, common={len(common)}")
    if not common:
        raise ValueError("no complete stock/Q10/Q50 identities")

    arms = {}
    for method, arm in raw_arms.items():
        keep = arm[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(common)
        arms[method] = (arm[keep].sort_values(DIVERSITY_PAIR_KEYS)
                        .reset_index(drop=True))
    return arms


def pair_against_stock(arms: dict[str, pd.DataFrame], method: str) -> pd.DataFrame:
    """Build the standard exact pair consumed by the existing SR-delta helpers."""
    stock = arms[Method.VANILLA][
        DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]].rename(columns={
            "rollout_id": "baseline_rollout_id",
            "success": "baseline_success",
        })
    condition = arms[method][
        DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]].rename(columns={
            "rollout_id": "condition_rollout_id",
            "success": "condition_success",
        })
    paired = stock.merge(condition, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    paired["baseline_success"] = paired.baseline_success.astype(bool)
    paired["condition_success"] = paired.condition_success.astype(bool)
    return paired


def success_tables(arms: dict[str, pd.DataFrame]):
    overall, by_suite = [], []
    for method in ARMS:
        arm = arms[method]
        overall.append({
            "arm": ARM_LABELS[method], "episodes": len(arm),
            "success_rate_pct": 100 * arm.success.astype(bool).mean(),
        })
        for suite, group in arm.groupby("suite", sort=True):
            by_suite.append({
                "suite": suite, "arm": ARM_LABELS[method],
                "episodes": len(group),
                "success_rate_pct": 100 * group.success.astype(bool).mean(),
            })
    return pd.DataFrame(overall), pd.DataFrame(by_suite)


def paired_effect_tables(arms: dict[str, pd.DataFrame]):
    """Return compact overall and per-suite deltas against stock."""
    overall, suites, pairs = [], [], {}
    for method in (Method.QPLANNING_Q10, Method.QPLANNING_Q50):
        pair = pair_against_stock(arms, method)
        pairs[method] = pair
        one, per_suite = summarize_pair(pair)
        one.insert(0, "planner", ARM_LABELS[method])
        per_suite.insert(0, "planner", ARM_LABELS[method])
        overall.append(one)
        suites.append(per_suite)
    return (pd.concat(overall, ignore_index=True),
            pd.concat(suites, ignore_index=True), pairs)


def extract_qplanning_boundaries(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten compact per-boundary Q distributions and diversity summaries."""
    output = []
    for method in (Method.QPLANNING_Q10, Method.QPLANNING_Q50):
        for row in arms[method].to_dict("records"):
            trace = _json_object(row.get("ms_candidate_u")).get("qplanning")
            if not trace:
                raise ValueError(f"{row['rollout_id']} is missing Q-Planning telemetry")
            boundaries = len(trace.get("q_values", []))
            if boundaries < 1:
                raise ValueError(f"{row['rollout_id']} has no Q decision boundary")
            for field in ("elite_indices", "elite_weights", "first10_diversity",
                          "full50_diversity", "inference_ms", "n_vf_evals"):
                if len(trace.get(field, [])) != boundaries:
                    raise ValueError(
                        f"{row['rollout_id']} has inconsistent {field} boundary count")
            for boundary in range(boundaries):
                q = np.asarray(trace["q_values"][boundary], dtype=float)
                elite = np.asarray(trace["elite_indices"][boundary], dtype=int)
                weights = np.asarray(trace["elite_weights"][boundary], dtype=float)
                if len(q) != 64 or len(elite) != 16 or len(weights) != 16:
                    raise ValueError("held-out Q telemetry is not sample64/top16")
                if not np.isclose(weights.sum(), 1.0, atol=1e-5):
                    raise ValueError("elite weights do not sum to one")
                entropy = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum())
                first10 = trace["first10_diversity"][boundary]
                full50 = trace["full50_diversity"][boundary]
                q_sorted = np.sort(q)
                output.append({
                    **{key: row[key] for key in DIVERSITY_PAIR_KEYS},
                    "rollout_id": row["rollout_id"], "method": method,
                    "planner": ARM_LABELS[method], "success": bool(row["success"]),
                    "chunk_idx": boundary, "q_mean": float(q.mean()),
                    "q_std": float(q.std()), "q_min": float(q.min()),
                    "q_max": float(q.max()), "q_range": float(q.max() - q.min()),
                    "q_top_margin": float(q_sorted[-1] - q_sorted[-2]),
                    "q_weighted": float(np.dot(q[elite], weights)),
                    "negative_q_weighted": -float(np.dot(q[elite], weights)),
                    "negative_q_max": -float(q.max()),
                    "elite_entropy": entropy / math.log(len(weights)),
                    "elite_effective_n": float(math.exp(entropy)),
                    "first10_pairwise_rms": float(first10["mean_pairwise_rms"]),
                    "first10_max_pairwise_rms": float(first10["max_pairwise_rms"]),
                    "first10_cosine_disagreement": 1.0 - float(
                        first10["mean_pairwise_cosine"]),
                    "full50_pairwise_rms": float(full50["mean_pairwise_rms"]),
                    "full50_max_pairwise_rms": float(full50["max_pairwise_rms"]),
                    "full50_cosine_disagreement": 1.0 - float(
                        full50["mean_pairwise_cosine"]),
                    "tail_minus_prefix_rms": float(
                        full50["mean_pairwise_rms"] - first10["mean_pairwise_rms"]),
                    "inference_ms": float(trace["inference_ms"][boundary]),
                    "n_vf_evals": int(trace["n_vf_evals"][boundary]),
                })
    return pd.DataFrame(output)


def q_episode_features(boundaries: pd.DataFrame) -> pd.DataFrame:
    """First-boundary and whole-trajectory means for each Q signal."""
    metadata = (boundaries.groupby("rollout_id", sort=False)
                [[*DIVERSITY_PAIR_KEYS, "method", "planner", "success"]].first())
    means = boundaries.groupby("rollout_id", sort=False)[list(Q_SIGNALS)].mean()
    means.columns = [f"{column}_episode" for column in means.columns]
    first = (boundaries[boundaries.chunk_idx.eq(0)]
             .groupby("rollout_id", sort=False)[list(Q_SIGNALS)].mean())
    first.columns = [f"{column}_first_chunk" for column in first.columns]
    return metadata.join(means).join(first).reset_index()


def q_prefix_features(boundaries: pd.DataFrame, *, max_chunks: int = 8) -> pd.DataFrame:
    """Mean of the first k available Q diagnostics while retaining every episode."""
    metadata = (boundaries.groupby("rollout_id", sort=False)
                [[*DIVERSITY_PAIR_KEYS, "method", "planner", "success"]].first()
                .reset_index())
    tables = []
    for first_k in range(1, max_chunks + 1):
        values = (boundaries[boundaries.chunk_idx.lt(first_k)]
                  .groupby("rollout_id", sort=False)[list(Q_SIGNALS)].mean()
                  .reset_index())
        frame = metadata.merge(values, on="rollout_id", validate="one_to_one")
        frame["first_k_chunks"] = first_k
        tables.append(frame)
    return pd.concat(tables, ignore_index=True)


def failure_auc_table(features: pd.DataFrame, score_columns, *,
                      success_column: str = "success", by_suite: bool = True,
                      n_boot: int = 2000) -> pd.DataFrame:
    """Failure ROC-AUC; every supplied score must increase with predicted risk."""
    groups = [("pooled", features)]
    if by_suite:
        groups.extend(features.groupby("suite", sort=True))
    output = []
    for suite, group in groups:
        failure = ~group[success_column].astype(bool).to_numpy()
        for score in score_columns:
            auc, low, high = bootstrap_rank_auc(
                failure, group[score].to_numpy(float), n_boot=n_boot)
            output.append({
                "suite": suite, "score_name": score, "episodes": len(group),
                "failures": int(failure.sum()), "failure_auc": auc,
                "auc_ci_low": low, "auc_ci_high": high,
            })
    return pd.DataFrame(output)


def q_prefix_auc_table(prefix: pd.DataFrame, *, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for (method, first_k), group in prefix.groupby(
            ["method", "first_k_chunks"], sort=True):
        failure = ~group.success.astype(bool).to_numpy()
        for signal in Q_SIGNALS:
            auc, low, high = bootstrap_rank_auc(
                failure, group[signal].to_numpy(float), n_boot=n_boot)
            rows.append({
                "method": method, "planner": ARM_LABELS[method],
                "first_k_chunks": int(first_k), "score_name": signal,
                "episodes": len(group), "failures": int(failure.sum()),
                "failure_auc": auc, "auc_ci_low": low, "auc_ci_high": high,
            })
    return pd.DataFrame(rows)


def q_chunk_auc_table(boundaries: pd.DataFrame, *, n_boot: int = 2000,
                      min_episodes: int = 20) -> pd.DataFrame:
    """AUC at each exact chunk index; later rows are survivor-biased."""
    rows = []
    for (method, chunk_idx), group in boundaries.groupby(
            ["method", "chunk_idx"], sort=True):
        if len(group) < min_episodes:
            continue
        failure = ~group.success.astype(bool).to_numpy()
        for signal in Q_SIGNALS:
            auc, low, high = bootstrap_rank_auc(
                failure, group[signal].to_numpy(float), n_boot=n_boot)
            rows.append({
                "method": method, "planner": ARM_LABELS[method],
                "chunk_idx": int(chunk_idx), "score_name": signal,
                "episodes_reaching_chunk": len(group),
                "failures": int(failure.sum()), "failure_auc": auc,
                "auc_ci_low": low, "auc_ci_high": high,
            })
    return pd.DataFrame(rows)


def attach_signal_to_pair(pair: pd.DataFrame, features: pd.DataFrame,
                          score_column: str) -> pd.DataFrame:
    values = features[DIVERSITY_PAIR_KEYS + [score_column]].copy()
    if values.duplicated(DIVERSITY_PAIR_KEYS).any():
        raise ValueError(f"duplicate signal identities for {score_column}")
    return pair.merge(values, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")


def quantile_gate_sweep(paired: pd.DataFrame, *, score_column: str,
                        grid_size: int = 21, min_selected: int = 10) -> pd.DataFrame:
    """Exploratory high-threshold and bounded-window outcome-mixing proxy."""
    score = paired[score_column].to_numpy(float)
    baseline = paired.baseline_success.to_numpy(bool)
    condition = paired.condition_success.to_numpy(bool)
    finite = score[np.isfinite(score)]
    if not len(finite):
        raise ValueError(f"{score_column} contains no finite values")
    bounds = np.unique(np.quantile(finite, np.linspace(0, 1, grid_size)))
    rows = []

    def append(kind, lower, upper, selected):
        policy = np.where(selected, condition, baseline)
        rows.append({
            "score_name": score_column, "gate_kind": kind,
            "lower": float(lower), "upper": float(upper),
            "episodes_in_sr_denominator": len(paired),
            "episodes_selected": int(selected.sum()),
            "eligible": int(selected.sum()) >= min_selected,
            "stock_sr_pct": 100 * baseline.mean(),
            "proxy_policy_sr_pct": 100 * policy.mean(),
            "proxy_delta_pp": 100 * (policy.mean() - baseline.mean()),
            "selected_F_to_S": int((selected & ~baseline & condition).sum()),
            "selected_S_to_F": int((selected & baseline & ~condition).sum()),
        })

    finite_mask = np.isfinite(score)
    for lower in bounds:
        append("high_threshold", lower, math.inf,
               finite_mask & (score >= lower))
    for lower in bounds:
        for upper in bounds:
            if upper <= lower:
                continue
            append("bounded_window", lower, upper,
                   finite_mask & (score >= lower) & (score <= upper))
    return pd.DataFrame(rows)


def discordant_selector_auc(paired: pd.DataFrame, *, score_column: str,
                            n_boot: int = 3000) -> dict:
    """Can a score rank Q wins above Q losses on stock/Q-discordant episodes?"""
    discordant = paired[
        paired.baseline_success.ne(paired.condition_success)].copy()
    q_wins = (~discordant.baseline_success & discordant.condition_success).to_numpy()
    auc, low, high = bootstrap_rank_auc(
        q_wins, discordant[score_column].to_numpy(float), n_boot=n_boot)
    return {
        "score_name": score_column, "discordant_episodes": len(discordant),
        "Q_wins": int(q_wins.sum()), "Q_losses": int((~q_wins).sum()),
        "Q_win_auc": auc, "auc_ci_low": low, "auc_ci_high": high,
    }
