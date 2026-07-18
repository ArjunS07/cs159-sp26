"""Trust-gated backfill of legacy SQLite result DBs into Supabase (experiment='legacy').

Handles both legacy schemas:
  - the 01/02/03 RolloutDB schema (has a `method` + `policy_model` column)
  - the pnp_pro schema (has `pnp_enabled` / `pnp_mode`, no `method`)

Excludes synthetic `synthbase_`-prefixed rows. Each source DB is tagged with a trust level
(--trust post_fix|pre_fix) -> sampler_algo_version 2|1, so pre-RNG-fix data is separable by
data. The DBs live on the user's Drive; run this wherever they are reachable.

Usage:
    python -m scripts.backfill_legacy --db rollouts_jeff_2_averages_k5_large_2.db \
        --trust post_fix [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

# allow `python scripts/backfill_legacy.py` as well as `-m scripts.backfill_legacy`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEGACY_EXPERIMENT = "legacy"

# old pnp_mode -> canonical (method, refine_average)
_MODE_MAP = {
    "uncertainty": ("pnp_uncertainty_only", False),
    "both": ("pnp_refinement", False),
    "both_avg": ("pnp_refinement", True),
}


def _cols(con, table) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _method_of(row: dict) -> tuple[str, bool | None]:
    if row.get("method"):
        m = row["method"]
        return m, (m == "pnp_refinement_avg" or None)
    if not row.get("pnp_enabled"):
        return "vanilla", None
    return _MODE_MAP.get(row.get("pnp_mode", "uncertainty"), ("pnp_uncertainty_only", False))


def _map_rollout(row: dict, store, source: str, algo_version: int) -> dict | None:
    rid = row.get("rollout_id", "")
    if str(rid).startswith("synthbase_"):
        return None
    method, refine_avg = _method_of(row)
    identity = {"suite": row.get("suite"), "task_idx": row.get("task_idx"),
                "episode_idx": row.get("episode_idx"), "init_state_hash": row.get("init_state_hash"),
                "source": source, "legacy_id": rid}
    denorm = {"method": method, "refine_average": refine_avg,
              "pnp_step_indices": _jsonparse(row.get("pnp_step_indices"))}
    new_id = store.make_rollout_id(LEGACY_EXPERIMENT, identity, {**denorm, "source": source})
    return {
        "rollout_id": new_id, "experiment": LEGACY_EXPERIMENT,
        "sampler_algo_version": algo_version,
        "benchmark": "libero_pro" if "jeff" in source or "pro" in source else "libero",
        "suite": row.get("suite"), "task_idx": row.get("task_idx"),
        "task_desc": row.get("task_desc"), "episode_idx": row.get("episode_idx"),
        "init_state_hash": row.get("init_state_hash"),
        "method": method, "refine_average": refine_avg,
        "pnp_enabled": bool(row.get("pnp_enabled")) if "pnp_enabled" in row else (method != "vanilla"),
        "pnp_k": row.get("pnp_k") or row.get("pnp_num_iterations"),
        "pnp_step_indices": _jsonparse(row.get("pnp_step_indices")),
        "success": bool(row.get("success")), "n_steps": row.get("n_steps"),
        "elapsed_s": row.get("elapsed_s"),
        "u_mean_episode": row.get("u_mean_episode"), "u_max_episode": row.get("u_max_episode"),
        "mm_bc_pc1_episode": row.get("mm_bc_pc1_episode"),
        "config_hash": store.config_hash({**denorm, "source": source}),
        "config_json": {"legacy_source": source, "legacy_id": rid},
        "status": "completed",
    }


def _jsonparse(v):
    if v in (None, ""):
        return None
    try:
        return json.loads(v)
    except Exception:
        return None


def backfill_db(path: str, trust: str, dry_run: bool, limit: int | None):
    source = os.path.basename(path)
    algo_version = 2 if trust == "post_fix" else 1
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    have = _cols(con, "rollouts")
    sql = "SELECT * FROM rollouts" + (f" LIMIT {limit}" if limit else "")
    rows = [dict(r) for r in con.execute(sql)]
    con.close()
    print(f"[{source}] {len(rows)} rollouts  trust={trust}  cols={'method' if 'method' in have else 'pnp_mode'}")

    store = None
    if not dry_run:
        from pnp.store import SupabaseStore
        store = SupabaseStore()
        store.start_run(driver="backfill_legacy", benchmark="libero_pro",
                        experiment=LEGACY_EXPERIMENT, notes=f"source={source} trust={trust}")

    n_kept = n_synth = 0
    est_bytes = 0
    for r in rows:
        mapped = _map_rollout(r, store, source, algo_version) if store else \
            _map_rollout(r, _DryStore(), source, algo_version)
        if mapped is None:
            n_synth += 1
            continue
        n_kept += 1
        est_bytes += len(json.dumps(mapped, default=str))
        if not dry_run:
            store.log_episode(mapped)
    if store:
        store.finish_run(n_rollouts=n_kept)
    print(f"[{source}] kept={n_kept}  dropped_synthetic={n_synth}  ~{est_bytes/1e6:.2f} MB rows"
          + ("  (dry-run)" if dry_run else ""))


class _DryStore:
    """Minimal stand-in so --dry-run can compute rollout_ids without a Supabase client."""
    experiment = LEGACY_EXPERIMENT

    @staticmethod
    def make_rollout_id(experiment, identity, cfg):
        import hashlib
        key = f"{experiment}|{json.dumps(identity, sort_keys=True, default=str)}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    @staticmethod
    def config_hash(cfg):
        import hashlib
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True, help="path(s) to a legacy .db")
    ap.add_argument("--trust", choices=["post_fix", "pre_fix"], default="post_fix")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for db in args.db:
        backfill_db(db, args.trust, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
