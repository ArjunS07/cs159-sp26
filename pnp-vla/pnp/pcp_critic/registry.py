"""Supabase registry for immutable PCP critic snapshots and checkpoints."""
from __future__ import annotations

import hashlib
import io
import json

import torch

from .data import DatasetSnapshot


SNAPSHOT_PREFIX = "pcp_critic/snapshots"
CHECKPOINT_PREFIX = "pcp_critic/checkpoints"


class PCPCriticRegistry:
    def __init__(self, store):
        self.store = store

    def publish_snapshot(self, snapshot: DatasetSnapshot, *, name: str) -> str:
        payload = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        path = f"{SNAPSHOT_PREFIX}/{snapshot.snapshot_id}.json"
        rows = (self.store.client.table("pcp_critic_dataset_snapshots")
                .select("snapshot_id,snapshot_path,snapshot_sha256")
                .eq("snapshot_id", snapshot.snapshot_id).execute().data or [])
        digest = hashlib.sha256(payload).hexdigest()
        if rows:
            existing = rows[0]
            if existing["snapshot_sha256"] != digest:
                raise ValueError(f"immutable PCP critic snapshot collision {snapshot.snapshot_id}")
            return snapshot.snapshot_id
        self.store._upload(path, payload)
        self.store.client.table("pcp_critic_dataset_snapshots").insert({
            "snapshot_id": snapshot.snapshot_id, "name": name,
            "policy_repo_id": snapshot.policy_repo_id, "policy_revision": snapshot.policy_revision,
            "artifact_schema_version": snapshot.artifact_schema_version,
            "n_rollouts": len(snapshot.rollout_ids), "n_train_rollouts": len(snapshot.train_rollout_ids),
            "n_val_rollouts": len(snapshot.val_rollout_ids), "snapshot_path": path,
            "snapshot_sha256": digest, "provenance_json": snapshot.provenance,
        }).execute()
        return snapshot.snapshot_id

    def load_snapshot(self, snapshot_id: str) -> DatasetSnapshot:
        rows = (self.store.client.table("pcp_critic_dataset_snapshots").select("*")
                .eq("snapshot_id", snapshot_id).execute().data or [])
        if len(rows) != 1:
            raise KeyError(f"unknown PCP critic snapshot {snapshot_id!r}")
        row = rows[0]
        payload = self.store._download(row["snapshot_path"])
        if hashlib.sha256(payload).hexdigest() != row["snapshot_sha256"]:
            raise ValueError(f"PCP critic snapshot checksum mismatch: {snapshot_id}")
        return DatasetSnapshot.from_dict(json.loads(payload))

    def register_model(self, critic_id: str, checkpoint: bytes, *, snapshot: DatasetSnapshot,
                       architecture: dict, train_config: dict, metrics: dict,
                       objective: str, safety_status: str = "offline_only") -> str:
        if safety_status != "offline_only":
            raise ValueError("this registry only permits offline_only PCP critic checkpoints")
        path = f"{CHECKPOINT_PREFIX}/{critic_id}.pt"
        self.store._upload(path, checkpoint)
        row = {
            "critic_id": critic_id, "run_id": self.store.run_id, "experiment": self.store.experiment,
            "snapshot_id": snapshot.snapshot_id, "policy_repo_id": snapshot.policy_repo_id,
            "policy_revision": snapshot.policy_revision, "artifact_schema_version": snapshot.artifact_schema_version,
            "objective": objective, "architecture_json": architecture, "train_config_json": train_config,
            "metrics_json": metrics, "checkpoint_path": path, "safety_status": safety_status,
        }
        self.store.client.table("pcp_critic_models").upsert(row, on_conflict="critic_id").execute()
        return critic_id

    def load_model_payload(self, critic_id: str) -> tuple[dict, dict]:
        rows = (self.store.client.table("pcp_critic_models").select("*")
                .eq("critic_id", critic_id).execute().data or [])
        if len(rows) != 1:
            raise KeyError(f"unknown PCP critic {critic_id!r}")
        row = rows[0]
        payload = torch.load(io.BytesIO(self.store._download(row["checkpoint_path"])),
                             map_location="cpu", weights_only=False)
        if payload.get("format") != "pcp_critic_v1":
            raise ValueError("checkpoint is not a PCP critic v1 artifact")
        if payload.get("snapshot_id") != row["snapshot_id"]:
            raise ValueError("checkpoint/snapshot registry mismatch")
        return payload, row
