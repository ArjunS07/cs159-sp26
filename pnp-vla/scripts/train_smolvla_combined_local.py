"""Train the 1,600-tree root-MC SmolVLA critic on a Mac Metal GPU.

From the repository root:
  pnp-vla/.venv/bin/python pnp-vla/scripts/train_smolvla_combined_local.py
The cache and checkpoints live in ~/pnp-vla-runs and resume on a second run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from pnp.smolvla_combined_success import (  # noqa: E402
    COMBINED_EXPERIMENT, COMBINED_SNAPSHOT_KEY, SOURCE_EXPERIMENTS,
    load_or_create_combined_snapshot, prepare_combined_root_cache,
)
from pnp.smolvla_q_selection import FRESH8_KINDS, load_success_q10  # noqa: E402
from pnp.smolvla_success_critic import (  # noqa: E402
    TimedRoots, evaluate_roots, train_smolvla_success_q10,
)
from pnp.store import SupabaseStore  # noqa: E402


def _load_local_credentials() -> None:
    path = PACKAGE_ROOT / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    missing = [key for key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY") if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"missing Supabase credentials: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-root", type=Path, default=Path.home() / "pnp-vla-runs")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1:
        parser.error("updates and batch-size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch Metal is unavailable; use a native Mac terminal")
    _load_local_credentials()
    cache_root = args.run_root.expanduser() / "cache"
    output_root = args.run_root.expanduser() / "checkpoints" / "combined_1600_root_mc_v1"
    cache_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(cache_root).free / 2**30
    if free_gib < 2:
        raise RuntimeError(f"only {free_gib:.1f} GiB free; need at least 2 GiB")
    print({"device": args.device, "free_disk_gib": round(free_gib, 1),
           "cache_root": str(cache_root), "output_root": str(output_root)}, flush=True)
    store = SupabaseStore()
    snapshot = load_or_create_combined_snapshot(store)
    cache = prepare_combined_root_cache(
        snapshot=snapshot, cache_root=cache_root, store=store,
        download_workers=args.cache_workers)
    if args.prepare_only:
        return
    result = train_smolvla_success_q10(
        arm="root_mc", output_root=output_root, cache_root=cache_root,
        tree_limit=1600, snapshot_key=COMBINED_SNAPSHOT_KEY,
        experiment=COMBINED_EXPERIMENT, candidate_kinds=FRESH8_KINDS,
        updates=args.updates, batch_size=args.batch_size, resume=True,
        root_cache_only=True, device=args.device, store=store)
    model, _ = load_success_q10(
        result["final_checkpoint"], expected_snapshot=snapshot["snapshot_digest"],
        device=args.device)
    groups = {group["candidate_group_id"]: group for group in snapshot["groups"]}
    by_cohort = {}
    for experiment in SOURCE_EXPERIMENTS:
        ids = [gid for gid in cache["validation_group_ids"]
               if groups[gid]["source_experiment"] == experiment]
        by_cohort[experiment] = evaluate_roots(
            model, TimedRoots(cache, ids, groups), torch.device(args.device))
    print(json.dumps({"checkpoint": result["final_checkpoint"],
                      "snapshot_digest": snapshot["snapshot_digest"],
                      "validation_all": result["validation"],
                      "validation_by_cohort": by_cohort}, indent=2), flush=True)


if __name__ == "__main__":
    main()
