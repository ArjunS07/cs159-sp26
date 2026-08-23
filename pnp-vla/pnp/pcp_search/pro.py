"""Frozen, suite-level LIBERO-PRO partition for PCP-search collection."""
from __future__ import annotations

from collections import Counter

from .manifest import ManifestItem, RolloutManifest
from .task_selection import COLLECTION_CONFIG, POLICY_REPO_ID, POLICY_REVISION


PRO_PARTITION_ID = "pcp-search-pro-suite-partition-v1"

# Each quota is divisible by ten tasks.  Whole suites are either train-eligible or held out;
# the three historically zero-success suites are deliberately evaluation-only.
PRO_TRAIN_QUOTAS = {
    "libero_object_temp_x0.1": 80,
    "libero_object_temp_y0.1": 80,
    "libero_object_temp_y0.2": 100,
    "libero_goal_swap": 60,
    "libero_spatial_swap": 60,
    "libero_goal_task": 80,
    "libero_goal_with_milk": 60,
    "libero_goal_with_yellow_book": 60,
    "libero_object_with_mug": 60,
}
PRO_HELDOUT_QUOTAS = {
    "libero_object_temp_x0.2": 20,
    "libero_object_temp_y0.3": 20,
    "libero_object_temp_x0.3": 20,
    "libero_object_swap": 20,
    "libero_10_swap": 20,
    "libero_object_task": 20,
    "libero_spatial_with_milk": 40,
}


def _validate_partition() -> None:
    overlap = set(PRO_TRAIN_QUOTAS) & set(PRO_HELDOUT_QUOTAS)
    if overlap:
        raise AssertionError(f"PRO train/heldout suite overlap: {sorted(overlap)}")
    if sum(PRO_TRAIN_QUOTAS.values()) != 640:
        raise AssertionError("PRO train allocation must be 640 rollouts")
    if sum(PRO_HELDOUT_QUOTAS.values()) != 160:
        raise AssertionError("PRO heldout allocation must be 160 rollouts")
    if any(quota % 10 for quota in (*PRO_TRAIN_QUOTAS.values(), *PRO_HELDOUT_QUOTAS.values())):
        raise AssertionError("PRO quotas must allocate an equal integer count per task")


def pro_partition_summary() -> dict:
    _validate_partition()
    return {
        "partition_id": PRO_PARTITION_ID,
        "train_eligible_rollouts": sum(PRO_TRAIN_QUOTAS.values()),
        "heldout_rollouts": sum(PRO_HELDOUT_QUOTAS.values()),
        "train_suites": dict(PRO_TRAIN_QUOTAS),
        "heldout_suites": dict(PRO_HELDOUT_QUOTAS),
    }


def build_pro_manifest(*, split: str, name: str | None = None) -> RolloutManifest:
    """Build one immutable train or heldout PRO manifest with exact state identities."""
    _validate_partition()
    if split not in {"train", "heldout"}:
        raise ValueError("split must be 'train' or 'heldout'")
    quotas = PRO_TRAIN_QUOTAS if split == "train" else PRO_HELDOUT_QUOTAS
    items: list[ManifestItem] = []
    for suite, quota in quotas.items():
        per_task = quota // 10
        for task_idx in range(10):
            for state_index in range(per_task):
                items.append(ManifestItem(
                    ordinal=len(items), suite=suite, task_idx=task_idx,
                    init_state_index=state_index, behavior_seed_index=0,
                    tier="pro_train" if split == "train" else "pro_heldout",
                    selection_reason="suite_partition_stock_pnp"))
    config = {
        **COLLECTION_CONFIG,
        "benchmark": "libero_pro",
        "partition_id": PRO_PARTITION_ID,
        "data_split": split,
        "train_eligible": split == "train",
    }
    return RolloutManifest(
        name=name or f"pcp-search-pro-{split}-{'640' if split == 'train' else '160'}",
        items=tuple(items), policy_repo_id=POLICY_REPO_ID, policy_revision=POLICY_REVISION,
        collection_config=config,
        provenance={
            **pro_partition_summary(),
            "benchmark": "libero_pro",
            "selection_policy": "whole-suite partition; stock P&P; execute 10 then replan",
            "zero_success_suites": ["libero_10_swap", "libero_object_task",
                                    "libero_object_temp_x0.3"],
        })


def build_pro_sentinel_manifest(*, split: str) -> RolloutManifest:
    """One task-0/state-0 rollout per suite before releasing the full PRO manifests.

    These identities are intentionally members of the corresponding full manifest, so its
    resumable worker recovers validated sentinel rollouts rather than spending them twice.
    """
    full = build_pro_manifest(split=split)
    first_by_suite = {}
    for item in full.items:
        first_by_suite.setdefault(item.suite, item)
    items = tuple(
        ManifestItem(ordinal=ordinal, suite=item.suite, task_idx=item.task_idx,
                     init_state_index=item.init_state_index,
                     behavior_seed_index=item.behavior_seed_index, tier="pro_sentinel",
                     selection_reason="preflight_artifact_validation")
        for ordinal, item in enumerate(first_by_suite.values()))
    config = {**full.collection_config, "sentinel": True}
    return RolloutManifest(
        name=f"pcp-search-pro-{split}-sentinel-{len(items)}", items=items,
        policy_repo_id=POLICY_REPO_ID, policy_revision=POLICY_REVISION,
        collection_config=config, provenance={**full.provenance,
                                               "sentinel_for_manifest": full.manifest_id})


def validate_pro_manifest(manifest: RolloutManifest) -> None:
    """Fail closed if a future edit makes a suite usable for both fit and evaluation."""
    _validate_partition()
    expected = PRO_TRAIN_QUOTAS if manifest.collection_config.get("data_split") == "train" \
        else PRO_HELDOUT_QUOTAS
    actual = Counter(item.suite for item in manifest.items)
    expected_counts = ({suite: 1 for suite in expected}
                       if manifest.collection_config.get("sentinel") else expected)
    if dict(actual) != expected_counts:
        raise ValueError("manifest membership differs from the approved PRO partition")
    if manifest.collection_config.get("benchmark") != "libero_pro":
        raise ValueError("PRO manifest must declare libero_pro benchmark")
    if bool(manifest.collection_config.get("train_eligible")) != (expected is PRO_TRAIN_QUOTAS):
        raise ValueError("PRO train eligibility disagrees with split")
