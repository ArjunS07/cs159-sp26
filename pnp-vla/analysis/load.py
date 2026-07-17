"""Pull DataFrames + Storage blobs from Supabase for analysis (off-GPU)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd


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


def rollouts(store, experiment=None, **eq) -> pd.DataFrame:
    df = pd.DataFrame(_all_rows(store.client, "rollouts", experiment, **eq))
    if not df.empty:
        df["fail"] = (~df["success"].astype(bool)).astype(int)
    return df


def euler_steps(store, experiment=None) -> pd.DataFrame:
    """pnp_euler_steps joined to their rollouts' (experiment, suite, method, success)."""
    r = rollouts(store, experiment)[["rollout_id", "suite", "method", "refine_average",
                                     "success", "fail"]] if experiment else None
    steps = pd.DataFrame(_all_rows(store.client, "pnp_euler_steps"))
    if r is not None and not steps.empty:
        steps = steps.merge(r, on="rollout_id", how="inner")
    return steps


def qc_eval(store, experiment=None) -> pd.DataFrame:
    return pd.DataFrame(_all_rows(store.client, "qc_eval", experiment))


def action_vectors(store) -> pd.DataFrame:
    return pd.DataFrame(_all_rows(store.client, "pnp_action_vectors"))


def load_ahats(store, rollout_id: str) -> dict:
    """Download a rollout's a_hats stacks (K,chunk,adim) keyed by chunk/step."""
    key = f"ahats/{rollout_id}.npz"
    blob = store._download(key)
    npz = np.load(io.BytesIO(blob))
    return {k: npz[k] for k in npz.files}
