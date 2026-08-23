"""Immutable, content-addressed rollout manifests for PCP-search collection."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, order=True)
class ManifestItem:
    """One policy rollout, identified independently of mutable simulator payloads.

    ``init_state_index`` selects a state from the pinned LIBERO task.  A state may be replayed
    with multiple policy noise streams; ``behavior_seed_index`` makes those rollouts distinct in
    both the manifest and the canonical rollout ID.
    """

    ordinal: int
    suite: str
    task_idx: int
    init_state_index: int
    behavior_seed_index: int = 0
    tier: str = "coverage"
    selection_reason: str = "untouched_state"
    source_rollout_id: str | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if self.task_idx < 0 or self.init_state_index < 0:
            raise ValueError("task_idx and init_state_index must be non-negative")
        if self.behavior_seed_index < 0:
            raise ValueError("behavior_seed_index must be non-negative")
        if not self.suite.startswith("libero_"):
            raise ValueError(f"unexpected LIBERO suite {self.suite!r}")

    @property
    def task_key(self) -> tuple[str, int]:
        return self.suite, self.task_idx

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (self.suite, self.task_idx, self.init_state_index,
                self.behavior_seed_index)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManifestItem":
        return cls(**value)


@dataclass(frozen=True)
class RolloutManifest:
    """A frozen set of rollout identities whose ID is a hash of its scientific contents."""

    name: str
    items: tuple[ManifestItem, ...]
    policy_repo_id: str
    policy_revision: str
    collection_config: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    parent_manifest_id: str | None = None
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("manifest name cannot be empty")
        if not self.items:
            raise ValueError("manifest must contain at least one rollout")
        if tuple(item.ordinal for item in self.items) != tuple(range(len(self.items))):
            raise ValueError("manifest ordinals must be contiguous and start at zero")
        identities = [item.identity for item in self.items]
        if len(identities) != len(set(identities)):
            raise ValueError("manifest contains duplicate rollout identities")
        if not self.policy_revision:
            raise ValueError("policy_revision must be pinned")
        horizon = self.collection_config.get("n_action_steps")
        if horizon != 10:
            raise ValueError(f"PCP-search manifests require n_action_steps=10, got {horizon!r}")

    def scientific_dict(self) -> dict[str, Any]:
        """Fields that determine membership and collection semantics.

        Wall-clock creation time and query provenance are intentionally excluded so rebuilding
        the same plan produces the same ID.
        """
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "parent_manifest_id": self.parent_manifest_id,
            "policy_repo_id": self.policy_repo_id,
            "policy_revision": self.policy_revision,
            "collection_config": self.collection_config,
            "items": [item.to_dict() for item in self.items],
        }

    @property
    def manifest_id(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.scientific_dict()).encode()).hexdigest()
        return f"pcps-{digest[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scientific_dict(),
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent) + ("\n" if indent else "")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RolloutManifest":
        supplied_id = value.get("manifest_id")
        manifest = cls(
            name=value["name"],
            items=tuple(ManifestItem.from_dict(item) for item in value["items"]),
            policy_repo_id=value["policy_repo_id"],
            policy_revision=value["policy_revision"],
            collection_config=dict(value["collection_config"]),
            provenance=dict(value.get("provenance") or {}),
            parent_manifest_id=value.get("parent_manifest_id"),
            created_at=value.get("created_at") or dt.datetime.now(dt.timezone.utc).isoformat(),
            schema_version=int(value.get("schema_version", MANIFEST_SCHEMA_VERSION)),
        )
        if supplied_id is not None and supplied_id != manifest.manifest_id:
            raise ValueError(
                f"manifest content hash mismatch: supplied {supplied_id}, "
                f"computed {manifest.manifest_id}")
        return manifest

    @classmethod
    def from_json(cls, source: str) -> "RolloutManifest":
        return cls.from_dict(json.loads(source))


def reindex_items(items: Iterable[ManifestItem]) -> tuple[ManifestItem, ...]:
    return tuple(dataclasses.replace(item, ordinal=index)
                 for index, item in enumerate(items))
