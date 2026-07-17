"""Analysis metrics, ported from pnp_pro_analysis_final.ipynb (A1-A5 + geometric B).

Pure functions over pandas DataFrames pulled by analysis.load. Variant selection is a filter
(method + refine_average) — no DB merge, no REFINE_VARIANT tag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from scipy import stats
except Exception:  # analysis extra not installed
    roc_auc_score = average_precision_score = f1_score = None
    stats = None

PAIR_KEYS = ["suite", "task_idx", "episode_idx", "init_state_hash"]


# ── A1: success-rate summaries ──────────────────────────────────────────────
def success_rate(df: pd.DataFrame, by=("method",)) -> pd.DataFrame:
    g = df.groupby(list(by))["success"]
    out = g.mean().mul(100).rename("SR_pct").reset_index()
    out["n"] = g.count().values
    return out


# ── A5: detector metrics (uncertainty -> failure) ───────────────────────────
def detector_metrics(score, fail) -> dict:
    score = np.asarray(score, float); fail = np.asarray(fail, int)
    m = np.isfinite(score); score, fail = score[m], fail[m]
    out = {"n": int(len(score)), "n_fail": int(fail.sum())}
    if roc_auc_score is None or len(np.unique(fail)) < 2 or len(score) < 3:
        out.update(roc_auc=np.nan, pr_auc=np.nan, spearman=np.nan, f1=np.nan, tau=np.nan)
        return out
    out["roc_auc"] = roc_auc_score(fail, score)
    out["pr_auc"] = average_precision_score(fail, score)
    out["spearman"] = stats.spearmanr(score, fail).correlation
    best_f1, best_t = -1.0, np.nan
    for t in np.quantile(score, np.linspace(0.05, 0.95, 19)):
        f = f1_score(fail, (score >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    out["f1"], out["tau"] = best_f1, best_t
    return out


def stratified_auc(df: pd.DataFrame, score_col: str, by="suite", fail_col="fail"):
    """Per-`by` AUC then averaged (the 6/19 fix)."""
    aucs = []
    for _, g in df.groupby(by):
        if roc_auc_score is None or g[fail_col].nunique() < 2:
            continue
        aucs.append(roc_auc_score(g[fail_col], g[score_col]))
    return (float(np.mean(aucs)) if aucs else np.nan), len(aucs)


# ── A2/A3: paired phase comparison + transitions ────────────────────────────
def _pivot_success(df, baseline_method, method, refine_average=None):
    b = df[df["method"] == baseline_method]
    m = df[df["method"] == method]
    if refine_average is not None:
        m = m[m["refine_average"] == refine_average]
    j = b.merge(m, on=PAIR_KEYS, suffixes=("_b", "_m"))
    return j


def phase_comparison(df, baseline_method="pnp_uncertainty_only",
                     method="pnp_refinement", refine_average=None) -> dict:
    j = _pivot_success(df, baseline_method, method, refine_average)
    if j.empty:
        return {"n": 0}
    sr_b = j["success_b"].mean() * 100
    sr_m = j["success_m"].mean() * 100
    return {"n": len(j), "SR_baseline": sr_b, "SR_method": sr_m, "delta_pp": sr_m - sr_b}


def transition_counts(df, baseline_method="pnp_uncertainty_only",
                      method="pnp_refinement", refine_average=None) -> dict:
    j = _pivot_success(df, baseline_method, method, refine_average)
    b = j["success_b"].astype(bool); m = j["success_m"].astype(bool)
    return {"F_to_S": int((~b & m).sum()), "S_to_F": int((b & ~m).sum()),
            "S_to_S": int((b & m).sum()), "F_to_F": int((~b & ~m).sum()), "n": len(j)}


# ── B: per-DOF localization ─────────────────────────────────────────────────
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def per_dof_auc(steps_df: pd.DataFrame, by="suite") -> pd.DataFrame:
    """Within-suite AUC of each per-DOF uncertainty (u_d0..u_d6) vs episode failure."""
    rows = []
    ep = steps_df.groupby("rollout_id").agg(
        {**{f"u_d{i}": "mean" for i in range(7)}, "fail": "first", "suite": "first"}).reset_index()
    for i, name in enumerate(DIM_NAMES):
        auc, k = stratified_auc(ep, f"u_d{i}", by=by)
        rows.append({"dim": name, "stratified_auc": auc, "n_suites": k})
    return pd.DataFrame(rows)


# ── B: geometry (Sarle bimodality + PCA isotropy) ───────────────────────────
def sarle_bc(x: np.ndarray) -> float:
    x = np.asarray(x, float).ravel(); n = x.size
    if n < 4:
        return float("nan")
    d = x - x.mean(); s = np.sqrt((d ** 2).mean()) + 1e-12
    g = (d ** 3).mean() / s ** 3
    k = (d ** 4).mean() / s ** 4 - 3.0
    return float((g ** 2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def pc1_fraction(A: np.ndarray):
    """A: (K, chunk*adim) or (K, D). Returns (PC1 variance fraction, Marchenko-Pastur edge)."""
    F = A.reshape(A.shape[0], -1).astype(np.float64)
    F = F - F.mean(0, keepdims=True)
    K, D = F.shape
    try:
        S = np.linalg.svd(F, full_matrices=False)[1]
        frac = float(S[0] ** 2 / (S ** 2).sum())
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    mp_edge = (1 + np.sqrt(D / K)) ** 2 / D    # normalized MP upper edge
    return frac, float(mp_edge)
