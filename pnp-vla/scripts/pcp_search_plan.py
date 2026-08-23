#!/usr/bin/env python3
"""Query historical rollouts and build/publish the immutable PCP-search initial manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pnp.pcp_search.registry import ManifestRegistry, dump_manifest_summary
from pnp.pcp_search.task_selection import build_initial_manifest, fetch_historical_task_records
from pnp.store import SupabaseStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("pcp_search_initial_manifest.json"))
    parser.add_argument(
        "--input-json", type=Path,
        help="Optional cached rollouts JSON; filters the pinned historical experiment locally")
    parser.add_argument("--publish", action="store_true",
                        help="Upload the frozen manifest to Supabase/Storage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = None
    if args.input_json:
        source = json.loads(args.input_json.read_text(encoding="utf-8"))
        rows = [row for row in source if row.get("experiment") ==
                "libero-u20-same-state-candidates-v1"]
        provenance = {"query_source": str(args.input_json), "query_mode": "cached_json"}
    else:
        store = SupabaseStore()
        rows = fetch_historical_task_records(store)
        provenance = {"query_mode": "supabase"}
    manifest = build_initial_manifest(rows, query_provenance=provenance)
    args.output.write_text(manifest.to_json(), encoding="utf-8")
    print(json.dumps(dump_manifest_summary(manifest), indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    if args.publish:
        store = store or SupabaseStore()
        ManifestRegistry(store).publish(manifest)
        print(f"published frozen manifest {manifest.manifest_id}")


if __name__ == "__main__":
    main()
