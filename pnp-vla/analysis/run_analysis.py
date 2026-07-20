"""Thin CLI for reproducible, snapshot-backed offline analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import geometry, pcp, pro, report, snapshot, standard_libero
from pnp.experiments import EXPERIMENT as STANDARD_EXPERIMENT
from .validate import (coverage_matrix, pro_coverage_matrix, validate_pro,
                       validate_standard)


def _client():
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("snapshot", "validate", "standard", "pro", "pcp", "all"), default="all")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--reference-snapshot", type=Path,
                        help="validated standard-LIBERO snapshot for PRO degradation")
    parser.add_argument("--reference-experiment", default=STANDARD_EXPERIMENT)
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
    snapshot_manifest = json.loads((path / "manifest.json").read_text())
    artifact_validation = snapshot_manifest.get("artifact_validation", {})
    benchmarks = set(frames["rollouts"]["benchmark"].dropna()) if "benchmark" in frames["rollouts"] else set()
    benchmark = next(iter(benchmarks)) if len(benchmarks) == 1 else None
    is_pro = benchmark == "libero_pro"
    if args.command == "standard" and is_pro:
        raise ValueError("standard command cannot analyze a LIBERO-PRO snapshot")
    if args.command == "pro" and not is_pro:
        raise ValueError("pro command requires a LIBERO-PRO snapshot")
    if is_pro:
        validated, validation = validate_pro(
            frames["rollouts"], frames["experiment_runs"], artifact_validation)
    else:
        validated, validation = validate_standard(frames["rollouts"], frames["experiment_runs"])
    matrix = pro_coverage_matrix(validated) if is_pro else coverage_matrix(validated)
    print(matrix.to_string(index=False))
    (path / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest["validation"] = validation
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.command == "validate":
        return 0
    if args.command == "pcp":
        print(json.dumps(pcp.analyze(validated), indent=2)); return 0
    if is_pro:
        reference_path = args.reference_snapshot or snapshot.latest_snapshot(
            args.output_root, args.reference_experiment)
        if reference_path is None:
            reference_path = snapshot.materialize(
                _client(), args.reference_experiment, args.output_root)
        reference_frames = snapshot.load_snapshot(reference_path)
        standard, _ = validate_standard(
            reference_frames["rollouts"], reference_frames["experiment_runs"])
        tables, pro_state = pro.analyze(
            validated, frames["pnp_euler_steps"], frames["pnp_action_vectors"], standard,
            artifact_validation)
        geometry_state, geometry_tables = geometry.analyze(
            validated, frames["pnp_action_vectors"], pnp_k=3)
        tables.update({f"pro_{name}": table for name, table in geometry_tables.items()})
        availability = {"pro": pro_state, "geometry": geometry_state,
                        "pcp": pcp.analyze(validated),
                        "cross_model": {"status": "not_available", "reason": "matching model conditions absent"}}
        report.write_pro_report(path, tables, availability)
        print(f"PRO analysis complete: {path}")
        return 0
    tables = standard_libero.run(validated, frames["pnp_euler_steps"])
    geometry_state, geometry_tables = geometry.analyze(validated, frames["pnp_action_vectors"], pnp_k=3)
    tables.update(geometry_tables)
    availability = {"pro": {"status": "not_available", "reason": "separate PRO snapshot"},
                    "pcp": pcp.analyze(validated),
                    "cross_model": {"status": "not_available", "reason": "matching model conditions not present"},
                    "geometry": geometry_state}
    report.write_report(args.experiment, path, validation, tables, availability)
    print(f"analysis complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
