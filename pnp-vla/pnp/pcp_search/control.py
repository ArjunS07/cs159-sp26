"""Collection-control operations used by thin survey and manifest notebooks."""
from __future__ import annotations

from collections import Counter

from ..store import SupabaseStore
from .collection import EXPERIMENT
from .pro import (build_fresh_pro_train_manifest, build_fresh_pro_train_sentinel_manifest,
                  build_pro_manifest, build_pro_sentinel_manifest, validate_pro_manifest)
from .registry import ManifestRegistry
from .task_selection import (
    HISTORICAL_EXPERIMENT,
    build_next_tranche_manifest,
    fetch_historical_task_records,
)


def build_standard_adaptive_from_store(*, parent_manifest_id: str, tranche_index: int,
                                       store: SupabaseStore | None = None):
    """Build, but do not publish, one evidence-driven standard-LIBERO tranche."""
    store = store or SupabaseStore()
    historical = fetch_historical_task_records(store, experiment=HISTORICAL_EXPERIMENT)
    current = store.fetch_all(
        "rollouts",
        "rollout_id,suite,task_idx,episode_idx,init_state_hash,status,success,n_chunks,"
        "n_steps,u_mean_episode,ms_candidate_u,training_ready",
        configure=lambda query: query.eq("experiment", EXPERIMENT).eq("benchmark", "libero"),
        order_by=("suite", "task_idx", "episode_idx", "rollout_id"))
    rows = historical + [row for row in current if row.get("training_ready")]
    coverage = Counter()
    for row in current:
        if row.get("training_ready"):
            coverage[(str(row["suite"]), int(row["task_idx"]))] += int(row.get("n_chunks") or 0)
    return build_next_tranche_manifest(
        rows, coverage, parent_manifest_id=parent_manifest_id, tranche_index=tranche_index)


def publish_pro_partition(*, store: SupabaseStore | None = None) -> dict[str, str]:
    """Freeze the approved 640/160 PRO manifests idempotently."""
    store = store or SupabaseStore()
    registry = ManifestRegistry(store)
    manifests = {split: build_pro_manifest(split=split) for split in ("train", "heldout")}
    for manifest in manifests.values():
        validate_pro_manifest(manifest)
    return {split: registry.publish(manifest) for split, manifest in manifests.items()}


def publish_pro_sentinels(*, store: SupabaseStore | None = None) -> dict[str, str]:
    """Freeze the 8+6 artifact-validation sentinels before full PRO release."""
    store = store or SupabaseStore()
    registry = ManifestRegistry(store)
    manifests = {split: build_pro_sentinel_manifest(split=split)
                 for split in ("train", "heldout")}
    for manifest in manifests.values():
        validate_pro_manifest(manifest)
    return {split: registry.publish(manifest) for split, manifest in manifests.items()}


def publish_fresh_pro_train_sentinel(*, store: SupabaseStore | None = None) -> str:
    """Freeze the eight-row fresh-PRO preflight manifest idempotently."""
    store = store or SupabaseStore()
    manifest = build_fresh_pro_train_sentinel_manifest()
    validate_pro_manifest(manifest)
    return ManifestRegistry(store).publish(manifest)


def publish_fresh_pro_train(*, store: SupabaseStore | None = None) -> str:
    """Freeze the approved second 640-row, train-eligible PRO tranche."""
    store = store or SupabaseStore()
    manifest = build_fresh_pro_train_manifest()
    validate_pro_manifest(manifest)
    return ManifestRegistry(store).publish(manifest)


def publish_standard_adaptive(*, parent_manifest_id: str, tranche_index: int,
                              store: SupabaseStore | None = None) -> str:
    """Build from the current ready set and freeze one standard adaptive tranche."""
    store = store or SupabaseStore()
    manifest = build_standard_adaptive_from_store(
        parent_manifest_id=parent_manifest_id, tranche_index=tranche_index, store=store)
    return ManifestRegistry(store).publish(manifest)
