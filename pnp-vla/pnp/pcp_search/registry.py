"""Supabase-backed immutable manifest registry and mutable collection status."""
from __future__ import annotations

import datetime as dt
from typing import Any

from .manifest import RolloutManifest


MANIFEST_BUCKET_PREFIX = "pcp_search/manifests"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class ManifestRegistry:
    """Store frozen membership separately from worker progress.

    Manifest rows and their JSON blob are append-only. Worker progress goes into
    ``pcp_search_manifest_results``, so retries never mutate the scientific plan.
    """

    def __init__(self, store):
        self.store = store

    def publish(self, manifest: RolloutManifest) -> str:
        manifest_id = manifest.manifest_id
        payload = manifest.to_json(indent=2).encode()
        path = f"{MANIFEST_BUCKET_PREFIX}/{manifest_id}.json"
        existing = (self.store.client.table("pcp_search_manifests")
                    .select("manifest_id,manifest_path,manifest_sha256")
                    .eq("manifest_id", manifest_id).execute().data or [])
        import hashlib
        digest = hashlib.sha256(payload).hexdigest()
        if existing:
            row = existing[0]
            existing_payload = self.store._download(row["manifest_path"])
            existing_manifest = RolloutManifest.from_json(existing_payload.decode())
            if existing_manifest.scientific_dict() != manifest.scientific_dict():
                raise ValueError(f"immutable manifest collision for {manifest_id}")
            return manifest_id

        self.store._upload(path, payload)
        row = {
            "manifest_id": manifest_id,
            "name": manifest.name,
            "schema_version": manifest.schema_version,
            "status": "frozen",
            "parent_manifest_id": manifest.parent_manifest_id,
            "n_rollouts": len(manifest.items),
            "policy_repo_id": manifest.policy_repo_id,
            "policy_revision": manifest.policy_revision,
            "collection_config": manifest.collection_config,
            "provenance_json": manifest.provenance,
            "manifest_path": path,
            "manifest_sha256": digest,
            "created_at": manifest.created_at,
            "frozen_at": _now(),
        }
        self.store.client.table("pcp_search_manifests").insert(row).execute()
        return manifest_id

    def load(self, manifest_id: str) -> RolloutManifest:
        rows = (self.store.client.table("pcp_search_manifests")
                .select("*").eq("manifest_id", manifest_id).execute().data or [])
        if len(rows) != 1:
            raise KeyError(f"unknown PCP-search manifest {manifest_id!r}")
        row = rows[0]
        if row.get("status") != "frozen":
            raise ValueError(f"manifest {manifest_id} is not frozen")
        payload = self.store._download(row["manifest_path"])
        import hashlib
        if hashlib.sha256(payload).hexdigest() != row["manifest_sha256"]:
            raise ValueError(f"manifest artifact checksum mismatch for {manifest_id}")
        return RolloutManifest.from_json(payload.decode())

    def completed_ordinals(self, manifest_id: str) -> set[int]:
        rows = self.store.fetch_all(
            "pcp_search_manifest_results", "ordinal,status",
            configure=lambda query: query.eq("manifest_id", manifest_id).eq(
                "status", "training_ready"),
            order_by=("ordinal",))
        return {int(row["ordinal"]) for row in rows}

    def record_result(self, manifest_id: str, ordinal: int, rollout_id: str, *,
                      status: str, reason: str | None = None,
                      validation: dict[str, Any] | None = None) -> None:
        allowed = {"collected", "training_ready", "excluded", "errored"}
        if status not in allowed:
            raise ValueError(f"invalid manifest result status {status!r}")
        self.store.client.table("pcp_search_manifest_results").upsert({
            "manifest_id": manifest_id,
            "ordinal": int(ordinal),
            "rollout_id": rollout_id,
            "status": status,
            "reason": reason,
            "validation_json": validation or {},
            "updated_at": _now(),
        }, on_conflict="manifest_id,ordinal").execute()


def dump_manifest_summary(manifest: RolloutManifest) -> dict[str, Any]:
    from collections import Counter
    tasks = Counter((item.suite, item.task_idx) for item in manifest.items)
    tiers = Counter(item.tier for item in manifest.items)
    reasons = Counter(item.selection_reason for item in manifest.items)
    return {
        "manifest_id": manifest.manifest_id,
        "n_rollouts": len(manifest.items),
        "n_tasks": len(tasks),
        "by_tier": dict(sorted(tiers.items())),
        "by_reason": dict(sorted(reasons.items())),
        "by_task": {f"{suite}/{task_idx}": count
                    for (suite, task_idx), count in sorted(tasks.items())},
    }
