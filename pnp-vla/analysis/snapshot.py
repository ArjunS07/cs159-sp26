"""Versioned, reusable local snapshots of analysis inputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SNAPSHOT_SCHEMA_VERSION = "1"
TABLES = ("rollouts", "pnp_euler_steps", "pnp_action_vectors", "experiment_runs")
ARTIFACT_COLUMNS = ("ahats_path", "pcp_chunks_path", "trajectory_path", "obs_frames_path")


def paginated_rows(client, table: str, *, experiment: str | None = None,
                   page_size: int = 1000) -> list[dict[str, Any]]:
    rows, start = [], 0
    while True:
        query = client.table(table).select("*")
        if experiment is not None and table in {"rollouts", "experiment_runs"}:
            query = query.eq("experiment", experiment)
        page = query.range(start, start + page_size - 1).execute().data or []
        rows.extend(page)
        if len(page) < page_size:
            return rows
        start += page_size


def _git_sha(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            text=True, capture_output=True, check=False)
    return result.stdout.strip()


def verify_artifact_references(client, rollouts: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Verify referenced Storage keys by listing each bucket folder once."""
    bucket = client.storage.from_("artifacts")
    references = {
        column: sorted(map(str, rollouts[column].dropna().unique()))
        for column in ARTIFACT_COLUMNS if column in rollouts
    }
    folders = {path.rsplit("/", 1)[0] for paths in references.values() for path in paths}
    available = set()
    for folder in sorted(folders):
        offset = 0
        while True:
            page = bucket.list(folder, {"limit": 1000, "offset": offset})
            available.update(f"{folder}/{row['name']}" for row in page if row.get("name"))
            if len(page) < 1000:
                break
            offset += 1000
    result = {}
    for column, paths in references.items():
        missing = sorted(set(paths) - available)
        result[column] = {"referenced": len(paths), "verified": len(paths) - len(missing),
                          "missing": missing, "status": "valid" if not missing else "invalid"}
    return result


def materialize(client, experiment: str, root: Path, *, page_size: int = 1000,
                query_time: datetime | None = None) -> Path:
    """Extract relevant tables and atomically identify them by content."""
    frames: dict[str, pd.DataFrame] = {}
    frames["rollouts"] = pd.DataFrame(paginated_rows(
        client, "rollouts", experiment=experiment, page_size=page_size))
    rollout_ids = set(frames["rollouts"].get("rollout_id", []))
    frames["experiment_runs"] = pd.DataFrame(paginated_rows(
        client, "experiment_runs", experiment=experiment, page_size=page_size))
    for table in ("pnp_euler_steps", "pnp_action_vectors"):
        all_rows = paginated_rows(client, table, page_size=page_size)
        frames[table] = pd.DataFrame(r for r in all_rows if r.get("rollout_id") in rollout_ids)

    digest = hashlib.sha256()
    for name in TABLES:
        digest.update(name.encode())
        if not frames[name].empty:
            # PostgREST JSON/array columns contain lists and dicts, which pandas cannot hash.
            # Canonical JSON also makes the content identifier stable across processes.
            records = frames[name].where(pd.notna(frames[name]), None).to_dict("records")
            digest.update(json.dumps(records, sort_keys=True, default=str,
                                     separators=(",", ":")).encode())
    snapshot_id = digest.hexdigest()[:16]
    destination = Path(root) / experiment / snapshot_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    rollouts = frames["rollouts"]
    artifact_validation = verify_artifact_references(client, rollouts)
    manifest = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "experiment": experiment,
        "query_time": (query_time or datetime.now(timezone.utc)).isoformat(),
        "row_counts": {k: len(v) for k, v in frames.items()},
        "rollout_hashes": sorted(map(str, rollouts.get("rollout_id", []))),
        "config_hashes": sorted(map(str, rollouts.get("config_hash", pd.Series(dtype=str)).dropna().unique())),
        "package_git_sha": _git_sha(Path(__file__).parents[1]),
        "schema_versions": sorted(map(str, rollouts.get("schema_version", pd.Series(dtype=str)).dropna().unique())),
        "sampler_versions": sorted(map(str, rollouts.get("sampler_algo_version", pd.Series(dtype=str)).dropna().unique())),
        "artifact_validation": artifact_validation,
        "validation": {"status": "not_run"},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def latest_snapshot(root: Path, experiment: str) -> Path | None:
    candidates = list((Path(root) / experiment).glob("*/manifest.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(Path(path) / f"{name}.parquet") for name in TABLES}
