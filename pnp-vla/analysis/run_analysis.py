"""Thin CLI for reproducible, snapshot-backed offline analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import geometry, pcp, pro, report, snapshot, standard_libero
from .validate import coverage_matrix, validate_standard


def _client():
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("snapshot", "validate", "standard", "pro", "pcp", "all"), default="all")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--refresh", action="store_true", help="contact Supabase even when a local snapshot exists")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    path = args.snapshot or snapshot.latest_snapshot(args.output_root, args.experiment)
    if args.command == "snapshot" or args.refresh or path is None:
        path = snapshot.materialize(_client(), args.experiment, args.output_root)
        if args.command == "snapshot":
            print(path); return 0
    frames = snapshot.load_snapshot(path)
    validated, validation = validate_standard(frames["rollouts"], frames["experiment_runs"])
    matrix = coverage_matrix(validated)
    print(matrix.to_string(index=False))
    (path / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest["validation"] = validation
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.command == "validate":
        return 0
    if args.command in ("pro", "pcp"):
        state = (pro if args.command == "pro" else pcp).analyze(validated)
        print(json.dumps(state, indent=2)); return 0
    tables = standard_libero.run(validated, frames["pnp_euler_steps"])
    geometry_state, geometry_tables = geometry.analyze(validated, frames["pnp_action_vectors"], pnp_k=3)
    tables.update(geometry_tables)
    availability = {"pro": pro.analyze(validated), "pcp": pcp.analyze(validated),
                    "cross_model": {"status": "not_available", "reason": "matching model conditions not present"},
                    "geometry": geometry_state}
    report.write_report(args.experiment, path, validation, tables, availability)
    print(f"analysis complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
