"""Resumable, transport-tolerant PCP dataset snapshot construction.

Snapshot creation only needs action moments and immutable rollout membership.
Training-ready artifacts were fully validated when they were uploaded, so this
path reads the single ``bellman/action`` field instead of downloading raw images,
simulator states, and every other multipart field again.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from ..store import SupabaseStore, TRAINING_DATA_MULTIPART_FORMAT
from .data import DatasetSnapshot, _partition, _policy_contract, eligible_rollout_rows
from .registry import PCPCriticRegistry


PROGRESS_SCHEMA_VERSION = 1


def _is_transport_error(error: BaseException) -> bool:
    return any(cls.__module__.split(".", 1)[0] in {"httpx", "httpcore"}
               for cls in type(error).mro())


def _download_with_retry(store, path: str, *, attempts: int = 8) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            return store._download(path)
        except Exception as error:
            if not _is_transport_error(error) or attempt == attempts:
                raise
            delay = min(30, 2 ** (attempt - 1))
            print(f"[pcp-snapshot] transport reset while reading {path}; "
                  f"retry {attempt}/{attempts} in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def _load_json_with_retry(store, path: str, *, attempts: int = 8) -> dict:
    """Download a JSON object, retrying successful but invalid HTTP responses."""
    for attempt in range(1, attempts + 1):
        payload = _download_with_retry(store, path, attempts=attempts)
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if attempt == attempts:
                raise ValueError(
                    f"invalid JSON manifest after {attempts} attempts: {path} "
                    f"({len(payload)} bytes)") from error
            delay = min(30, 2 ** (attempt - 1))
            print(
                f"[pcp-snapshot] invalid JSON manifest while reading {path} "
                f"({len(payload)} bytes); retry {attempt}/{attempts} in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def load_training_fields_with_retry(store, path: str, fields: Iterable[str], *,
                                    attempts: int = 8) -> dict[str, np.ndarray]:
    """Load selected multipart arrays, retrying HTTP transport failures."""
    wanted = tuple(dict.fromkeys(fields))
    if path.endswith(".npz"):
        payload = _download_with_retry(store, path, attempts=attempts)
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            missing = sorted(set(wanted) - set(archive.files))
            if missing:
                raise ValueError(f"training artifact {path} is missing {missing}")
            return {name: archive[name] for name in wanted}
    manifest = _load_json_with_retry(store, path, attempts=attempts)
    if manifest.get("format") != TRAINING_DATA_MULTIPART_FORMAT:
        raise ValueError(f"unsupported training-data manifest at {path}")
    missing = sorted(set(wanted) - set(manifest.get("arrays", {})))
    if missing:
        raise ValueError(f"training artifact {path} is missing {missing}")
    cached: dict[str, dict[str, np.ndarray]] = {}
    result = {}
    for name in wanted:
        spec = manifest["arrays"][name]
        shape, dtype = tuple(spec["shape"]), np.dtype(spec["dtype"])
        value = np.empty(shape, dtype=dtype)
        for part in spec["parts"]:
            key = part["path"]
            if key not in cached:
                part_payload = _download_with_retry(store, key, attempts=attempts)
                with np.load(io.BytesIO(part_payload), allow_pickle=False) as archive:
                    cached[key] = {item: archive[item] for item in archive.files}
            start, stop = part["start"], part["stop"]
            if not shape or start is None or stop is None:
                value[...] = cached[key][name]
            else:
                value[start:stop] = cached[key][name]
        result[name] = value
    return result


def _row_digest(rows: list[dict]) -> str:
    payload = [(row["rollout_id"], row["training_data_path"],
                row.get("training_data_schema_version"), row.get("run_id")) for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


def _resumable_action_statistics(store, rows: list[dict], *, progress_path: str | Path,
                                 checkpoint_interval: int = 10) -> tuple[np.ndarray, np.ndarray, int]:
    path = Path(progress_path).expanduser()
    digest = _row_digest(rows)
    progress = None
    if path.exists():
        try:
            candidate = json.loads(path.read_text())
            if (candidate.get("schema_version") == PROGRESS_SCHEMA_VERSION
                    and candidate.get("row_digest") == digest):
                progress = candidate
            else:
                print("[pcp-snapshot] eligible rollout set changed; starting a new scan")
        except (OSError, json.JSONDecodeError):
            print("[pcp-snapshot] unreadable progress file; starting a new scan")
    if progress is None:
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION, "row_digest": digest,
            "processed_rollout_ids": [], "count": 0, "mean": None, "m2": None,
            "n_transitions": 0,
        }
    processed = set(progress["processed_rollout_ids"])
    count = int(progress["count"])
    mean = None if progress["mean"] is None else np.asarray(progress["mean"], np.float64)
    m2 = None if progress["m2"] is None else np.asarray(progress["m2"], np.float64)
    n_transitions = int(progress["n_transitions"])
    if processed:
        print(f"[pcp-snapshot] resuming at {len(processed)}/{len(rows)} completed rollouts")
    since_save = 0
    for position, row in enumerate(rows, 1):
        if row["rollout_id"] in processed:
            continue
        arrays = load_training_fields_with_retry(
            store, row["training_data_path"], ("bellman/action",))
        actions = np.asarray(arrays["bellman/action"], np.float64)
        if actions.ndim != 3 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"rollout {row['rollout_id']} has invalid bellman/action {actions.shape}")
        flat = actions.reshape(-1, actions.shape[-1])
        batch_count = len(flat)
        batch_mean = flat.mean(axis=0)
        batch_m2 = ((flat - batch_mean) ** 2).sum(axis=0)
        if mean is None:
            count, mean, m2 = batch_count, batch_mean, batch_m2
        else:
            delta = batch_mean - mean
            total = count + batch_count
            m2 = m2 + batch_m2 + delta ** 2 * count * batch_count / total
            mean = mean + delta * batch_count / total
            count = total
        n_transitions += len(actions)
        processed.add(row["rollout_id"])
        since_save += 1
        if since_save >= checkpoint_interval or len(processed) == len(rows):
            progress.update({
                "processed_rollout_ids": sorted(processed), "count": count,
                "mean": mean.tolist(), "m2": m2.tolist(),
                "n_transitions": n_transitions,
            })
            _save_progress(path, progress)
            since_save = 0
        if len(processed) % 25 == 0 or len(processed) == len(rows):
            print(f"[pcp-snapshot] artifacts: {len(processed)}/{len(rows)}")
    if not count:
        raise ValueError("eligible artifacts contain no actions")
    return (mean.astype(np.float32),
            np.sqrt(m2 / count).astype(np.float32).clip(1e-6), n_transitions)


def create_resumable_dataset_snapshot(*, name: str, progress_path: str | Path,
                                      seed: int = 42, train_fraction: float = .8,
                                      gamma: float = .99,
                                      store: SupabaseStore | None = None) -> str:
    """Create the same immutable snapshot contract with a resumable field-only scan."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    store = store or SupabaseStore()
    rows = eligible_rollout_rows(store)
    if not rows:
        raise ValueError("no eligible PCP-search training rows")
    contracts = {_policy_contract(row) for row in rows}
    if len(contracts) != 1:
        raise ValueError(f"mixed policy contracts require separate snapshots: {contracts}")
    versions = {int(row.get("training_data_schema_version") or 0) for row in rows}
    if len(versions) != 1:
        raise ValueError(f"mixed training artifact versions require separate snapshots: {versions}")
    mean, std, n_transitions = _resumable_action_statistics(
        store, rows, progress_path=progress_path)
    train_ids, val_ids = _partition(rows, seed=seed, train_fraction=train_fraction)
    payload = {
        "name": name, "rollout_ids": sorted(row["rollout_id"] for row in rows),
        "policy_contract": _policy_contract(rows[0]),
        "artifact_schema_version": next(iter(versions)), "seed": seed,
        "train_fraction": train_fraction, "gamma": gamma,
        "train_rollout_ids": train_ids, "val_rollout_ids": val_ids,
        "action_mean": mean.tolist(), "action_std": std.tolist(),
    }
    snapshot_id = "pcpcds-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    snapshot = DatasetSnapshot(
        snapshot_id=snapshot_id, rollout_ids=tuple(payload["rollout_ids"]),
        train_rollout_ids=tuple(train_ids), val_rollout_ids=tuple(val_ids),
        policy_repo_id=payload["policy_contract"][0],
        policy_revision=payload["policy_contract"][1],
        artifact_schema_version=next(iter(versions)),
        action_mean=tuple(float(x) for x in mean),
        action_std=tuple(float(x) for x in std),
        provenance={**payload, "n_transitions": n_transitions,
                    "snapshot_scan": "resumable_selected_fields_v1"})
    PCPCriticRegistry(store).publish_snapshot(snapshot, name=name)
    print(f"[pcp-critic] snapshot {snapshot.snapshot_id}: {len(snapshot.rollout_ids)} rollouts, "
          f"{n_transitions} transitions")
    return snapshot.snapshot_id
