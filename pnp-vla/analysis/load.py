"""Pull DataFrames + Storage blobs from Supabase for analysis (off-GPU)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from pnp.config import PCP_3WAY


def require_methods(df: pd.DataFrame, names) -> None:
    """Fail LOUD (not silently empty) when an expected method is absent from the frame.

    Guards analyses that filter by `method` — if a notebook wrote a different label, this raises
    with expected-vs-present instead of returning an empty result that looks like 'no effect'."""
    present = set(df["method"].unique()) if not df.empty else set()
    missing = [n for n in names if n not in present]
    if missing:
        raise ValueError(f"expected method(s) {missing} not in data; present={sorted(present)}. "
                         "Check the notebook wrote the canonical pnp.config.Method strings.")


def _all_rows(client, table, experiment=None, select="*", page=1000, **eq):
    """Paginated fetch of a whole table (PostgREST caps rows per request)."""
    out, start = [], 0
    while True:
        q = client.table(table).select(select)
        if experiment:
            q = q.eq("experiment", experiment)
        for k, v in eq.items():
            q = q.eq(k, v)
        rows = q.range(start, start + page - 1).execute().data or []
        out.extend(rows)
        if len(rows) < page:
            return out
        start += page


def _config_label(row) -> str:
    """Compact label distinguishing a swept config (same method, different params) for grouping.

    e.g. two `pnp_uncertainty_only` rows with different (S,K) become 'pnp_uncertainty_only-s2_3-k3'
    vs '...-s5-k5'. Group/plot by `config_label` instead of `method` to compare a sweep."""
    parts = [str(row.get("method"))]
    if row.get("refine_average"):
        parts.append("avg")
    if row.get("pnp_step_indices"):
        parts.append("s" + "_".join(str(s) for s in row["pnp_step_indices"]))
    if row.get("pnp_k"):
        parts.append(f"k{row['pnp_k']}")
    if row.get("correction_lambda") is not None:
        parts.append(f"lam{row['correction_lambda']}")
    if row.get("num_samples"):
        parts.append(f"n{row['num_samples']}")
    if row.get("num_inference_steps"):
        parts.append(f"steps{row['num_inference_steps']}")
    return "-".join(parts)


def rollouts(store, experiment=None, **eq) -> pd.DataFrame:
    df = pd.DataFrame(_all_rows(store.client, "rollouts", experiment, **eq))
    if not df.empty:
        df["fail"] = (~df["success"].astype(bool)).astype(int)
        df["config_label"] = df.apply(_config_label, axis=1)   # distinguishes sweeps by config
    return df


def euler_steps(store, experiment=None) -> pd.DataFrame:
    """pnp_euler_steps joined to their rollouts' (experiment, suite, method, success)."""
    r = rollouts(store, experiment)[["rollout_id", "suite", "method", "refine_average",
                                     "success", "fail"]] if experiment else None
    steps = pd.DataFrame(_all_rows(store.client, "pnp_euler_steps"))
    if r is not None and not steps.empty:
        steps = steps.merge(r, on="rollout_id", how="inner")
    return steps


def pcp_three_way(store, experiment=None) -> pd.DataFrame:
    """PCP 3-way eval SR — the three PCP_3WAY method rows in `rollouts` (no qc_eval table)."""
    df = rollouts(store, experiment)
    if df.empty:
        return df
    require_methods(df, PCP_3WAY)
    return df[df["method"].isin(PCP_3WAY)]


def action_vectors(store) -> pd.DataFrame:
    return pd.DataFrame(_all_rows(store.client, "pnp_action_vectors"))


def load_ahats(store, rollout_id: str) -> dict:
    """Download a rollout's a_hats stacks (K,chunk,adim) keyed by chunk/step."""
    key = f"ahats/{rollout_id}.npz"
    blob = store._download(key)
    npz = np.load(io.BytesIO(blob))
    return {k: npz[k] for k in npz.files}
