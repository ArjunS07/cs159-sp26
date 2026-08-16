"""Analysis primitives for the source suffix-sensitivity / tapered-P&P pilot."""
from __future__ import annotations

import io
import math
import re
import time

import numpy as np
import pandas as pd

from analysis.statistics import discordant_test, paired_bootstrap_ci
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS


ARMS = (Method.SUFFIX_SENSITIVITY, Method.REFINEMENT, Method.TAPERED_REFINEMENT)
SCORE_COLUMNS = (
    "u_first10_episode", "u_first20_episode", "u_full50_episode",
    "u_first10_first_chunk", "u_first20_first_chunk", "u_full50_first_chunk",
)
_U_KEY = re.compile(r"^c(?P<chunk>\d+)_s(?P<step>\d+)_u_time$")


def validate_cohort(rows: pd.DataFrame, *, expected_identities: int = 220,
                    require_complete: bool = True) -> dict[str, pd.DataFrame]:
    """Filter completed rows and require one exact row per identity and arm."""
    frame = rows[rows.status.eq("completed") & rows.method.isin(ARMS)].copy()
    present = set(frame.method.unique())
    missing = set(ARMS) - present
    if missing:
        raise ValueError(f"missing required arm(s): {sorted(missing)}; present={sorted(present)}")
    arms = {}
    for method in ARMS:
        arm = frame[frame.method.eq(method)].copy()
        if arm.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"duplicate identities in {method}")
        arms[method] = arm
    key_sets = {
        method: set(map(tuple, arm[DIVERSITY_PAIR_KEYS].itertuples(index=False, name=None)))
        for method, arm in arms.items()}
    common = set.intersection(*key_sets.values())
    if any(keys != common for keys in key_sets.values()):
        detail = {method: len(keys) for method, keys in key_sets.items()}
        raise ValueError(f"three arms do not contain identical identity sets: {detail}; common={len(common)}")
    if require_complete and len(common) != expected_identities:
        raise ValueError(
            f"expected {expected_identities} matched identities, found {len(common)}")
    return arms


def pair_arms(baseline: pd.DataFrame, condition: pd.DataFrame) -> pd.DataFrame:
    """Exact-match a condition against the diagnostic no-op baseline."""
    base = baseline[DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]].rename(columns={
        "rollout_id": "baseline_rollout_id", "success": "baseline_success"})
    other = condition[DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]].rename(columns={
        "rollout_id": "condition_rollout_id", "success": "condition_success"})
    paired = base.merge(other, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    paired["baseline_success"] = paired.baseline_success.astype(bool)
    paired["condition_success"] = paired.condition_success.astype(bool)
    return paired


def _paired_summary(group: pd.DataFrame) -> dict:
    baseline = group.baseline_success.to_numpy(bool)
    condition = group.condition_success.to_numpy(bool)
    lo, hi = paired_bootstrap_ci(baseline, condition, n_boot=5000)
    f_to_s = int((~baseline & condition).sum())
    s_to_f = int((baseline & ~condition).sum())
    return {
        "episodes": len(group),
        "baseline_sr_pct": 100 * float(baseline.mean()),
        "condition_sr_pct": 100 * float(condition.mean()),
        "condition_minus_baseline_pp": 100 * float(condition.mean() - baseline.mean()),
        "delta_ci_low_pp": 100 * lo, "delta_ci_high_pp": 100 * hi,
        "failure_to_success": f_to_s, "success_to_failure": s_to_f,
        "paired_p_value": discordant_test(f_to_s, s_to_f),
    }


def summarize_pair(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([_paired_summary(paired)])
    by_suite = pd.DataFrame([
        {"suite": suite, **_paired_summary(group)}
        for suite, group in paired.groupby("suite", sort=True)])
    return overall, by_suite


def summarize_all_arms(arms: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compact SR table for baseline, ordinary full P&P, and tapered P&P."""
    labels = {
        Method.SUFFIX_SENSITIVITY: "diagnostic_baseline",
        Method.REFINEMENT: "full_pnp",
        Method.TAPERED_REFINEMENT: "tapered_pnp",
    }
    overall = pd.DataFrame([{
        "arm": labels[method], "episodes": len(frame),
        "success_rate_pct": 100 * frame.success.astype(bool).mean(),
    } for method, frame in arms.items()])
    suite_rows = []
    for method, frame in arms.items():
        for suite, group in frame.groupby("suite", sort=True):
            suite_rows.append({
                "suite": suite, "arm": labels[method], "episodes": len(group),
                "success_rate_pct": 100 * group.success.astype(bool).mean(),
            })
    return overall, pd.DataFrame(suite_rows)


def _download_with_retry(store, path: str, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            return store._download(path)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def decode_artifact(payload: bytes, *, rollout_id: str = "") -> tuple[pd.DataFrame, ...]:
    """Decode one diagnostic artifact into probe records and action-position profiles."""
    record_rows, uncertainty_rows, sensitivity_rows = [], [], []
    with np.load(io.BytesIO(payload)) as archive:
        for key in archive.files:
            match = _U_KEY.match(key)
            if not match:
                continue
            chunk_idx, euler_step = int(match.group("chunk")), int(match.group("step"))
            u_time = np.asarray(archive[key], dtype=float).reshape(-1)
            if len(u_time) < 20:
                raise ValueError(f"{rollout_id}/{key} contains only {len(u_time)} actions")
            stem = key.removesuffix("_u_time")
            record = {
                "rollout_id": rollout_id, "chunk_idx": chunk_idx,
                "euler_step": euler_step,
                "u_first10": float(u_time[:10].mean()),
                "u_first20": float(u_time[:20].mean()),
                "u_full50": float(u_time.mean()),
            }
            for position, value in enumerate(u_time):
                uncertainty_rows.append({
                    "rollout_id": rollout_id, "chunk_idx": chunk_idx,
                    "euler_step": euler_step, "action_position": position,
                    "uncertainty": float(value),
                })
            prediction_key = f"{stem}_suffix_prefix_predictions"
            reference_key = f"{stem}_suffix_prefix_reference"
            if prediction_key in archive and reference_key in archive:
                predictions = np.asarray(archive[prediction_key], dtype=float)
                reference = np.asarray(archive[reference_key], dtype=float)
                if predictions.ndim != 4 or reference.ndim != 3:
                    raise ValueError(
                        f"unexpected suffix diagnostic shapes: {predictions.shape}, {reference.shape}")
                delta = predictions - reference[None, ...]
                l2_by_position = np.linalg.norm(delta, axis=-1).mean(axis=(0, 1))
                record.update({
                    "tail_to_prefix_l2": float(np.linalg.norm(delta, axis=-1).mean()),
                    "tail_to_prefix_abs": float(np.abs(delta).mean()),
                    "tail_to_prefix_std": float(
                        np.concatenate([reference[None, ...], predictions], axis=0)
                        .std(axis=0).mean()),
                    "tail_gripper_flip": float(
                        (np.sign(predictions[..., 6])
                         != np.sign(reference[..., 6])[None, ...]).mean())
                    if predictions.shape[-1] > 6 else math.nan,
                })
                for position, value in enumerate(l2_by_position):
                    sensitivity_rows.append({
                        "rollout_id": rollout_id, "chunk_idx": chunk_idx,
                        "euler_step": euler_step, "executed_action_position": position,
                        "tail_to_prefix_l2": float(value),
                    })
            record_rows.append(record)
    if not record_rows:
        raise ValueError(f"artifact for {rollout_id} has no per-action uncertainty records")
    return (pd.DataFrame(record_rows), pd.DataFrame(uncertainty_rows),
            pd.DataFrame(sensitivity_rows))


def load_artifact_tables(store, diagnostic_rows: pd.DataFrame, *, progress=None
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download all diagnostic artifacts and produce episode + profile tables."""
    required = diagnostic_rows[["rollout_id", "suite", "success", "ahats_path"]].copy()
    if required.ahats_path.isna().any():
        missing = required[required.ahats_path.isna()].rollout_id.tolist()
        raise ValueError(f"{len(missing)} diagnostic rows are missing ahats_path")
    records, uncertainty, sensitivity = [], [], []
    iterator = required.to_dict("records")
    if progress is not None:
        iterator = progress(iterator, total=len(required), desc="diagnostic artifacts")
    for row in iterator:
        decoded = decode_artifact(
            _download_with_retry(store, str(row["ahats_path"])),
            rollout_id=str(row["rollout_id"]))
        for destination, frame in zip((records, uncertainty, sensitivity), decoded):
            if not frame.empty:
                frame["suite"] = row["suite"]
                frame["success"] = bool(row["success"])
                destination.append(frame)
    record_frame = pd.concat(records, ignore_index=True)
    uncertainty_frame = pd.concat(uncertainty, ignore_index=True)
    sensitivity_frame = (pd.concat(sensitivity, ignore_index=True)
                         if sensitivity else pd.DataFrame())

    metric_columns = [
        "u_first10", "u_first20", "u_full50", "tail_to_prefix_l2",
        "tail_to_prefix_abs", "tail_to_prefix_std", "tail_gripper_flip"]
    episode = record_frame.groupby("rollout_id", sort=False)[metric_columns].mean()
    episode.columns = [f"{column}_episode" for column in episode.columns]
    first = (record_frame[record_frame.chunk_idx.eq(0)]
             .groupby("rollout_id", sort=False)[metric_columns].mean())
    first.columns = [f"{column}_first_chunk" for column in first.columns]
    metadata = required.set_index("rollout_id")[["suite", "success"]]
    episode = metadata.join(episode).join(first).reset_index()
    return episode, record_frame, uncertainty_frame, sensitivity_frame


def rank_auc(labels, scores) -> float:
    """ROC-AUC from ranks; positive labels should receive larger scores."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_rank_auc(labels, scores, *, n_boot: int = 2000,
                       seed: int = 159) -> tuple[float, float, float]:
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    auc = rank_auc(labels, scores)
    rng, samples = np.random.default_rng(seed), []
    for _ in range(n_boot):
        indices = rng.integers(0, len(labels), len(labels))
        value = rank_auc(labels[indices], scores[indices])
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return auc, math.nan, math.nan
    low, high = np.quantile(samples, [.025, .975])
    return auc, float(low), float(high)


def failure_auc_table(features: pd.DataFrame, score_columns=SCORE_COLUMNS,
                      *, by_suite: bool = True, n_boot: int = 2000) -> pd.DataFrame:
    """Failure AUC for uncertainty/sensitivity scores on the no-op baseline."""
    groups = [("pooled", features)]
    if by_suite:
        groups.extend(features.groupby("suite", sort=True))
    rows = []
    for suite, group in groups:
        failures = ~group.success.astype(bool).to_numpy()
        for score_column in score_columns:
            if score_column not in group:
                continue
            scores = group[score_column].to_numpy(float)
            auc, low, high = bootstrap_rank_auc(failures, scores, n_boot=n_boot)
            rows.append({
                "suite": suite, "score_name": score_column, "episodes": len(group),
                "failures": int(failures.sum()), "failure_auc": auc,
                "auc_ci_low": low, "auc_ci_high": high,
            })
    return pd.DataFrame(rows)


def window_sweep(paired: pd.DataFrame, *, score_column: str,
                 grid_size: int = 25, min_selected: int = 10,
                 lower_max: float = .06, upper_min: float = .01,
                 upper_max: float = .08) -> pd.DataFrame:
    """Bounded-window selective policy; every pair remains in the SR denominator."""
    score = paired[score_column].to_numpy(float)
    baseline = paired.baseline_success.to_numpy(bool)
    condition = paired.condition_success.to_numpy(bool)
    baseline_sr = float(baseline.mean())
    rows = []
    for lower in np.linspace(0, lower_max, grid_size):
        for upper in np.linspace(upper_min, upper_max, grid_size):
            if upper <= lower:
                continue
            selected = np.isfinite(score) & (score >= lower) & (score <= upper)
            policy = np.where(selected, condition, baseline)
            eligible = int(selected.sum()) >= min_selected
            rows.append({
                "score_name": score_column, "lower": float(lower), "upper": float(upper),
                "episodes_in_sr_denominator": len(paired),
                "episodes_refined": int(selected.sum()), "eligible": eligible,
                "baseline_sr": baseline_sr,
                "window_policy_sr": float(policy.mean()) if eligible else math.nan,
                "delta_pp": 100 * float(policy.mean() - baseline_sr) if eligible else math.nan,
                "selected_F_to_S": int((selected & ~baseline & condition).sum()),
                "selected_S_to_F": int((selected & baseline & ~condition).sum()),
            })
    return pd.DataFrame(rows)


def threshold_sweep(paired: pd.DataFrame, *, score_column: str,
                    grid_size: int = 33, min_selected: int = 10,
                    threshold_max: float = .08) -> pd.DataFrame:
    """One-sided U >= threshold policy; every pair remains in the SR denominator."""
    score = paired[score_column].to_numpy(float)
    baseline = paired.baseline_success.to_numpy(bool)
    condition = paired.condition_success.to_numpy(bool)
    baseline_sr = float(baseline.mean())
    rows = []
    for threshold in np.linspace(0, threshold_max, grid_size):
        selected = np.isfinite(score) & (score >= threshold)
        policy = np.where(selected, condition, baseline)
        eligible = int(selected.sum()) >= min_selected
        rows.append({
            "score_name": score_column, "threshold": float(threshold),
            "episodes_in_sr_denominator": len(paired),
            "episodes_refined": int(selected.sum()), "eligible": eligible,
            "baseline_sr": baseline_sr,
            "threshold_policy_sr": float(policy.mean()) if eligible else math.nan,
            "delta_pp": 100 * float(policy.mean() - baseline_sr) if eligible else math.nan,
            "selected_F_to_S": int((selected & ~baseline & condition).sum()),
            "selected_S_to_F": int((selected & baseline & ~condition).sum()),
        })
    return pd.DataFrame(rows)


def top_windows(sweep: pd.DataFrame, *, n: int = 10) -> pd.DataFrame:
    # A caller may attach a comparison label before ranking. Return only sweep-native
    # columns so the caller can add that label exactly once to the ranked result.
    eligible = sweep[sweep.eligible & sweep.delta_pp.notna()].copy()
    eligible = eligible.drop(columns=["comparison"], errors="ignore")
    ranked = eligible.sort_values(
        ["delta_pp", "episodes_refined", "lower", "upper"],
        ascending=[False, False, True, True]).head(n).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def apply_window_by_suite(paired: pd.DataFrame, *, score_column: str,
                          lower: float, upper: float) -> pd.DataFrame:
    frame = paired.copy()
    frame["selected"] = frame[score_column].between(lower, upper, inclusive="both")
    frame["window_success"] = np.where(
        frame.selected, frame.condition_success, frame.baseline_success).astype(bool)
    return pd.DataFrame([{
        "suite": suite, "episodes": len(group),
        "episodes_refined": int(group.selected.sum()),
        "baseline_sr_pct": 100 * group.baseline_success.mean(),
        "window_policy_sr_pct": 100 * group.window_success.mean(),
        "window_minus_baseline_pp": 100 * (
            group.window_success.mean() - group.baseline_success.mean()),
    } for suite, group in frame.groupby("suite", sort=True)])


def sensitivity_quartiles(paired_full: pd.DataFrame, paired_tapered: pd.DataFrame,
                          *, score_column: str = "tail_to_prefix_l2_episode") -> pd.DataFrame:
    """Conditional SRs by baseline tail-sensitivity quartile (descriptive, not whole-cohort)."""
    columns = DIVERSITY_PAIR_KEYS + [score_column, "baseline_success", "condition_success"]
    full = paired_full[columns].rename(columns={"condition_success": "full_success"})
    tapered = paired_tapered[DIVERSITY_PAIR_KEYS + ["condition_success"]].rename(
        columns={"condition_success": "tapered_success"})
    frame = full.merge(tapered, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    frame["sensitivity_quartile"] = pd.qcut(
        frame[score_column].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    return pd.DataFrame([{
        "sensitivity_quartile": quartile, "episodes": len(group),
        "sensitivity_mean": group[score_column].mean(),
        "baseline_sr_pct": 100 * group.baseline_success.mean(),
        "full_pnp_sr_pct": 100 * group.full_success.mean(),
        "tapered_pnp_sr_pct": 100 * group.tapered_success.mean(),
        "full_minus_baseline_pp_within_quartile": 100 * (
            group.full_success.mean() - group.baseline_success.mean()),
        "tapered_minus_baseline_pp_within_quartile": 100 * (
            group.tapered_success.mean() - group.baseline_success.mean()),
    } for quartile, group in frame.groupby("sensitivity_quartile", sort=True)])
