"""Rollout-telemetry geometry summaries with strict minimum-sample gates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pnp.config import Method

MIN_SARLE_K = 4


def analyze(rollouts: pd.DataFrame | None = None, vectors: pd.DataFrame | None = None, *, pnp_k=3):
    state = {"status": "partially_available", "reason":
             f"rollout-vector geometry available; Sarle not available because K={pnp_k} < {MIN_SARLE_K}",
             "minimum_sarle_k": MIN_SARLE_K, "observed_k": pnp_k,
             "sarle_status": "not_available", "isotropy_status": "not_available_without_full_spectrum"}
    if rollouts is None or vectors is None or vectors.empty:
        state.update(status="not_available", reason="action-vector telemetry is absent")
        return state, {}
    observed = rollouts[rollouts.method == Method.UNCERTAINTY]
    joined = vectors.merge(observed[["rollout_id", "success", "chunk_size"]], on="rollout_id", how="inner")
    rows = []
    for row in joined.to_dict("records"):
        mean_value, std_value = row.get("a_mean_vec"), row.get("a_std_vec")
        mean = np.asarray([] if mean_value is None else mean_value, float)
        std = np.asarray([] if std_value is None else std_value, float)
        if len(mean) < 3 or len(std) < 3:
            continue
        direction = mean[:3]
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        unit, variance = direction / norm, std[:3] ** 2
        parallel = float(np.sum(unit ** 2 * variance))
        lateral = float(max(0., variance.sum() - parallel))
        rows.append({"rollout_id": row["rollout_id"], "success": bool(row["success"]),
                     "parallel_variance": parallel, "lateral_variance": lateral,
                     "lateral_parallel_ratio": lateral / parallel if parallel > 0 else np.nan,
                     "pc1_fraction": row.get("mm_pc1_frac")})
    directional = pd.DataFrame(rows)
    summary = directional.groupby("success").agg(
        n_vectors=("rollout_id", "size"), parallel_variance_mean=("parallel_variance", "mean"),
        lateral_variance_mean=("lateral_variance", "mean"),
        lateral_parallel_ratio_mean=("lateral_parallel_ratio", "mean"),
        pc1_fraction_mean=("pc1_fraction", "mean")).reset_index()
    chunk_sizes = observed.chunk_size.dropna().unique()
    dimension = int(chunk_sizes[0]) * 7 if len(chunk_sizes) == 1 else np.nan
    mp_edge = (1 + np.sqrt(dimension / pnp_k)) ** 2 / dimension if np.isfinite(dimension) else np.nan
    pca = pd.DataFrame([{"pnp_k": pnp_k, "flattened_dimension": dimension,
                         "marchenko_pastur_edge": mp_edge,
                         "pc1_fraction_mean": directional.pc1_fraction.mean(),
                         "isotropy": np.nan, "isotropy_status": "not_available_without_full_spectrum"}])
    unavailable = pd.DataFrame([
        {"metric": "sarle_bimodality", "status": "not_available", "reason": f"requires K>={MIN_SARLE_K}; K={pnp_k}"},
        {"metric": "online_feature_gating", "status": "not_available", "reason": "PCP deployment telemetry absent"},
        {"metric": "cross_model_geometry", "status": "not_available", "reason": "matched model data absent"},
    ])
    return state, {"geometry_directional": summary, "geometry_pca_mp": pca,
                   "geometry_unavailable": unavailable}
