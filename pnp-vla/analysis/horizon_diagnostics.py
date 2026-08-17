"""Analysis primitives for the full-cohort U10/U20/U50 diagnostic run."""
from __future__ import annotations

import io
import re
import time

import numpy as np
import pandas as pd

from analysis.suffix_sensitivity import bootstrap_rank_auc
from pnp.config import Method, RolloutConfig
from pnp.diversity import DIVERSITY_PAIR_KEYS


HORIZONS = (10, 20, 50)
_U_KEY = re.compile(r"^c(?P<chunk>\d+)_s(?P<step>\d+)_u_time$")


def validate_diagnostic_cohort(rows: pd.DataFrame, *, expected_identities: int = 1300,
                               require_complete: bool = True) -> pd.DataFrame:
    """Require one completed uncertainty-diagnostic row per exact PRO identity."""
    required = set(DIVERSITY_PAIR_KEYS + [
        "rollout_id", "status", "success", "method", "ahats_path"])
    missing_columns = required - set(rows.columns)
    if missing_columns:
        raise ValueError(f"diagnostic rows are missing columns: {sorted(missing_columns)}")
    frame = rows[
        rows.status.eq("completed") & rows.method.eq(Method.UNCERTAINTY)].copy()
    if frame.duplicated(DIVERSITY_PAIR_KEYS).any():
        raise ValueError("duplicate completed diagnostic identities")
    if frame.ahats_path.isna().any():
        raise ValueError(
            f"{int(frame.ahats_path.isna().sum())} diagnostic rows are missing artifacts")
    if require_complete and len(frame) != expected_identities:
        raise ValueError(
            f"expected {expected_identities} completed diagnostic identities, found {len(frame)}")
    return frame.sort_values(DIVERSITY_PAIR_KEYS).reset_index(drop=True)


def _download_with_retry(store, path: str, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            return store._download(path)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def decode_horizon_artifact(payload: bytes, *, rollout_id: str = ""
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decode U-by-action and consecutive-disagreement profiles from one artifact."""
    records, positions, iterations = [], [], []
    with np.load(io.BytesIO(payload)) as archive:
        for key in archive.files:
            match = _U_KEY.match(key)
            if not match:
                continue
            chunk_idx = int(match.group("chunk"))
            euler_step = int(match.group("step"))
            u_time = np.asarray(archive[key], dtype=float).reshape(-1)
            if len(u_time) != 50:
                raise ValueError(
                    f"{rollout_id}/{key} expected 50 action positions, found {len(u_time)}")
            iter_key = key.removesuffix("_u_time") + "_u_iter_time"
            if iter_key not in archive:
                raise ValueError(f"{rollout_id}/{key} is missing {iter_key}")
            iter_time = np.asarray(archive[iter_key], dtype=float)
            if iter_time.ndim != 2 or iter_time.shape[1] != len(u_time):
                raise ValueError(
                    f"{rollout_id}/{iter_key} has unexpected shape {iter_time.shape}")
            if iter_time.shape[0] < 2:
                raise ValueError(
                    f"{rollout_id}/{iter_key} needs at least two perturbation transitions")

            record = {
                "rollout_id": rollout_id, "chunk_idx": chunk_idx,
                "euler_step": euler_step,
            }
            for horizon in HORIZONS:
                sequence = iter_time[:, :horizon].mean(axis=1)
                contraction = float(sequence[0] - sequence[-1])
                record.update({
                    f"u{horizon}": float(u_time[:horizon].mean()),
                    f"start_disagreement{horizon}": float(sequence[0]),
                    f"end_disagreement{horizon}": float(sequence[-1]),
                    f"contraction{horizon}": contraction,
                    f"contraction_fraction{horizon}": float(
                        contraction / max(abs(float(sequence[0])), 1e-12)),
                })
                for pair_index, disagreement in enumerate(sequence):
                    iterations.append({
                        "rollout_id": rollout_id, "chunk_idx": chunk_idx,
                        "euler_step": euler_step, "horizon": horizon,
                        "perturbation_pair": pair_index,
                        "disagreement": float(disagreement),
                    })
            for position, uncertainty in enumerate(u_time):
                positions.append({
                    "rollout_id": rollout_id, "chunk_idx": chunk_idx,
                    "euler_step": euler_step, "action_position": position,
                    "uncertainty": float(uncertainty),
                    "start_disagreement": float(iter_time[0, position]),
                    "end_disagreement": float(iter_time[-1, position]),
                    "contraction": float(
                        iter_time[0, position] - iter_time[-1, position]),
                })
            records.append(record)
    if not records:
        raise ValueError(f"artifact for {rollout_id} has no U-time records")
    return pd.DataFrame(records), pd.DataFrame(positions), pd.DataFrame(iterations)


def _metric_columns() -> list[str]:
    prefixes = (
        "u", "start_disagreement", "end_disagreement",
        "contraction", "contraction_fraction")
    return [f"{prefix}{horizon}" for prefix in prefixes for horizon in HORIZONS]


def load_horizon_artifacts(store, diagnostic_rows: pd.DataFrame, *, progress=None
                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download artifacts and return episode, record, position, and iteration tables."""
    metadata_columns = DIVERSITY_PAIR_KEYS + [
        "rollout_id", "success", "ahats_path", "n_chunks"]
    required = diagnostic_rows[metadata_columns].copy()
    record_frames, position_frames, iteration_frames = [], [], []
    iterator = required.to_dict("records")
    if progress is not None:
        iterator = progress(iterator, total=len(required), desc="worker-41 artifacts")
    for row in iterator:
        decoded = decode_horizon_artifact(
            _download_with_retry(store, str(row["ahats_path"])),
            rollout_id=str(row["rollout_id"]))
        for destination, frame in zip(
                (record_frames, position_frames, iteration_frames), decoded):
            frame["suite"] = row["suite"]
            frame["success"] = bool(row["success"])
            destination.append(frame)
    records = pd.concat(record_frames, ignore_index=True)
    positions = pd.concat(position_frames, ignore_index=True)
    iterations = pd.concat(iteration_frames, ignore_index=True)
    metrics = _metric_columns()
    episode = records.groupby("rollout_id", sort=False)[metrics].mean()
    episode.columns = [f"{column}_episode" for column in episode.columns]
    first = (records[records.chunk_idx.eq(0)]
             .groupby("rollout_id", sort=False)[metrics].mean())
    first.columns = [f"{column}_first_chunk" for column in first.columns]
    metadata = required.set_index("rollout_id").drop(columns=["ahats_path"])
    features = metadata.join(episode).join(first).reset_index()
    if len(features) != len(required):
        raise ValueError(
            f"decoded features cover {len(features)}/{len(required)} diagnostic rows")
    return features, records, positions, iterations


def prefix_feature_table(records: pd.DataFrame, features: pd.DataFrame, *, max_chunks: int = 8
                         ) -> pd.DataFrame:
    """Mean diagnostics available by each chunk boundary, retaining every episode."""
    score_columns = [
        f"{prefix}{horizon}"
        for prefix in ("u", "contraction", "contraction_fraction")
        for horizon in HORIZONS]
    metadata = features[["rollout_id", "suite", "success", "n_chunks"]].copy()
    tables = []
    for first_k in range(1, max_chunks + 1):
        values = (records[records.chunk_idx.lt(first_k)]
                  .groupby("rollout_id", sort=False)[score_columns].mean()
                  .reset_index())
        frame = metadata.merge(values, on="rollout_id", how="left", validate="one_to_one")
        if frame[score_columns].isna().any().any():
            raise ValueError(f"missing prefix diagnostics at first_k={first_k}")
        frame["first_k_chunks"] = first_k
        frame["chunks_observed"] = np.minimum(frame.n_chunks.astype(int), first_k)
        tables.append(frame)
    return pd.concat(tables, ignore_index=True)


def failure_auc_table(features: pd.DataFrame, score_columns, *, by_suite: bool = True,
                      n_boot: int = 2000) -> pd.DataFrame:
    """Failure ROC-AUC with bootstrap intervals; larger scores predict failure."""
    groups = [("pooled", features)]
    if by_suite:
        groups.extend(features.groupby("suite", sort=True))
    output = []
    for suite, group in groups:
        failure = ~group.success.astype(bool).to_numpy()
        for score in score_columns:
            auc, low, high = bootstrap_rank_auc(
                failure, group[score].to_numpy(float), n_boot=n_boot)
            output.append({
                "suite": suite, "score_name": score, "episodes": len(group),
                "failures": int(failure.sum()), "failure_auc": auc,
                "auc_ci_low": low, "auc_ci_high": high,
            })
    return pd.DataFrame(output)


def prefix_failure_auc_table(prefix: pd.DataFrame, *, n_boot: int = 2000
                             ) -> pd.DataFrame:
    """All-episode AUC by first-k chunks for U and negated contraction."""
    rows = []
    for first_k, group in prefix.groupby("first_k_chunks", sort=True):
        failure = ~group.success.astype(bool).to_numpy()
        for horizon in HORIZONS:
            for score_type, values in (
                    ("uncertainty", group[f"u{horizon}"].to_numpy(float)),
                    ("negative_contraction",
                     -group[f"contraction{horizon}"].to_numpy(float))):
                auc, low, high = bootstrap_rank_auc(failure, values, n_boot=n_boot)
                rows.append({
                    "first_k_chunks": int(first_k), "score_type": score_type,
                    "action_horizon": horizon, "episodes": len(group),
                    "failures": int(failure.sum()), "failure_auc": auc,
                    "auc_ci_low": low, "auc_ci_high": high,
                })
    return pd.DataFrame(rows)


def select_historical_10_action_arms(store, rows: pd.DataFrame, *, expected_identities: int = 1300
                                     ) -> dict[str, pd.DataFrame]:
    """Select corrected historical source arms by behavior-derived config hashes."""
    configs = {
        Method.UNCERTAINTY: RolloutConfig(
            pnp_steps=(3, 4), pnp_k=5, refine=False, refine_average=False,
            n_action_steps=10),
        Method.REFINEMENT: RolloutConfig(
            pnp_steps=(3, 4), pnp_k=5, refine=True, refine_average=False,
            n_action_steps=10),
    }
    hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in configs.items()}
    arms = {}
    for method in configs:
        frame = rows[
            rows.status.eq("completed") & rows.method.eq(method)
            & rows.config_hash.eq(hashes[method])].copy()
        if frame.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"duplicate historical {method} identities")
        if len(frame) != expected_identities:
            raise ValueError(
                f"expected {expected_identities} historical {method} rows, found {len(frame)}")
        arms[method] = frame
    return arms


def pair_diagnostics_with_historical(features: pd.DataFrame,
                                     arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach diagnostic scores to corrected baseline/refinement outcomes by exact identity."""
    excluded = set(DIVERSITY_PAIR_KEYS + ["rollout_id", "success", "n_chunks"])
    score_columns = [column for column in features if column not in excluded]
    diagnostic = features[DIVERSITY_PAIR_KEYS + ["success"] + score_columns].rename(
        columns={"success": "diagnostic_success"})
    baseline = arms[Method.UNCERTAINTY][DIVERSITY_PAIR_KEYS + ["success"]].rename(
        columns={"success": "baseline_success"})
    refinement = arms[Method.REFINEMENT][DIVERSITY_PAIR_KEYS + ["success"]].rename(
        columns={"success": "condition_success"})
    paired = (diagnostic.merge(baseline, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
              .merge(refinement, on=DIVERSITY_PAIR_KEYS, validate="one_to_one"))
    for column in ("diagnostic_success", "baseline_success", "condition_success"):
        paired[column] = paired[column].astype(bool)
    paired["diagnostic_matches_historical_baseline"] = (
        paired.diagnostic_success.eq(paired.baseline_success))
    return paired


def quantile_outcome_curve(features: pd.DataFrame, *, score_column: str,
                           outcome_column: str = "success", bins: int = 5) -> pd.DataFrame:
    """Readable score-versus-outcome curve with equal-count bins."""
    frame = features[[score_column, outcome_column]].dropna().copy()
    frame["quantile"] = pd.qcut(
        frame[score_column].rank(method="first"), bins,
        labels=np.arange(1, bins + 1)).astype(int)
    return (frame.groupby("quantile", sort=True)
            .agg(score_mean=(score_column, "mean"), episodes=(outcome_column, "size"),
                 outcome_rate=(outcome_column, "mean"))
            .reset_index())
