"""Frozen, suite-level LIBERO-PRO partition for PCP-search collection."""
from __future__ import annotations

from collections import Counter

from .manifest import ManifestItem, RolloutManifest
from .task_selection import COLLECTION_CONFIG, POLICY_REPO_ID, POLICY_REVISION


PRO_PARTITION_ID = "pcp-search-pro-suite-partition-v1"
FRESH_PRO_TRAIN_PARTITION_ID = "pcp-search-pro-fresh-state-train-v1"

# Each quota is divisible by ten tasks.  Position perturbations are a fully held-out category:
# no temperature/position suite is train-eligible.  The two non-position zero-success suites are
# neither collected for critic fitting nor allocated fresh budget in this first program.
PRO_TRAIN_QUOTAS = {
    "libero_goal_swap": 60,
    "libero_object_swap": 60,
    "libero_spatial_swap": 60,
    "libero_goal_task": 100,
    "libero_goal_with_milk": 90,
    "libero_spatial_with_milk": 90,
    "libero_object_with_mug": 90,
    "libero_goal_with_yellow_book": 90,
}
PRO_HELDOUT_QUOTAS = {
    "libero_object_temp_x0.1": 30,
    "libero_object_temp_y0.1": 30,
    "libero_object_temp_x0.2": 30,
    "libero_object_temp_y0.2": 30,
    "libero_object_temp_x0.3": 20,
    "libero_object_temp_y0.3": 20,
}
PRO_EVALUATION_ONLY_SUITES = ("libero_10_swap", "libero_object_task")
# These custom-asset suites expose only ten initial states/task.  The first
# tranche already used states 0--8, so a fresh rollout must vary the policy
# noise stream rather than invent unavailable physical states.
PRO_TEN_STATE_SUITES = ("libero_goal_with_milk", "libero_spatial_with_milk")
FRESH_STATE_START = 10


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
    if any("_temp_" in suite for suite in PRO_TRAIN_QUOTAS):
        raise AssertionError("position perturbations must be held out as a category")
    if not all("_temp_" in suite for suite in PRO_HELDOUT_QUOTAS):
        raise AssertionError("heldout collection must contain only position perturbations")


def pro_partition_summary() -> dict:
    _validate_partition()
    return {
        "partition_id": PRO_PARTITION_ID,
        "train_eligible_rollouts": sum(PRO_TRAIN_QUOTAS.values()),
        "heldout_rollouts": sum(PRO_HELDOUT_QUOTAS.values()),
        "train_suites": dict(PRO_TRAIN_QUOTAS),
        "heldout_suites": dict(PRO_HELDOUT_QUOTAS),
        "heldout_category": "position_perturb",
        "evaluation_only_suites": list(PRO_EVALUATION_ONLY_SUITES),
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
            "selection_policy": "whole-category position holdout; stock P&P; execute 10 then replan",
            "heldout_category": "position_perturb",
            "evaluation_only_suites": list(PRO_EVALUATION_ONLY_SUITES),
        })


def build_fresh_pro_train_manifest(*, name: str | None = None) -> RolloutManifest:
    """Second 640-row PRO training tranche on disjoint physical states when available.

    The ordinary suites have 50 states/task, so this tranche begins at state 10.
    The two ten-state milk suites instead repeat their 0--8 physical states with
    ``behavior_seed_index=1``.  That distinction is explicit in provenance and
    selection reasons; it is not treated as fresh physical-state coverage.
    """
    _validate_partition()
    items: list[ManifestItem] = []
    for suite, quota in PRO_TRAIN_QUOTAS.items():
        per_task = quota // 10
        for task_idx in range(10):
            for offset in range(per_task):
                is_ten_state = suite in PRO_TEN_STATE_SUITES
                items.append(ManifestItem(
                    ordinal=len(items), suite=suite, task_idx=task_idx,
                    init_state_index=offset if is_ten_state else FRESH_STATE_START + offset,
                    behavior_seed_index=1 if is_ten_state else 0,
                    tier="pro_train_fresh_state" if not is_ten_state else "pro_train_fresh_seed",
                    selection_reason=("fresh_behavior_seed_ten_state_suite" if is_ten_state
                                      else "fresh_physical_state")))
    config = {
        **COLLECTION_CONFIG,
        "benchmark": "libero_pro",
        "partition_id": FRESH_PRO_TRAIN_PARTITION_ID,
        "data_split": "train",
        "train_eligible": True,
        "collection_round": "fresh_state_v1",
    }
    initial = build_pro_manifest(split="train")
    return RolloutManifest(
        name=name or "pcp-search-pro-fresh-state-train-640", items=tuple(items),
        policy_repo_id=POLICY_REPO_ID, policy_revision=POLICY_REVISION,
        collection_config=config, parent_manifest_id=initial.manifest_id,
        provenance={
            **pro_partition_summary(), "benchmark": "libero_pro",
            "collection_round": "fresh_state_v1",
            "source_train_manifest_id": initial.manifest_id,
            "physical_state_start": FRESH_STATE_START,
            "ten_state_suites": list(PRO_TEN_STATE_SUITES),
            "ten_state_behavior_seed_index": 1,
            "selection_policy": (
                "same suite quotas; fresh states 10+ for 50-state suites; "
                "new behavior seeds only for ten-state milk suites; stock P&P; execute 10 then replan"),
            "heldout_category": "position_perturb",
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


def build_fresh_pro_train_sentinel_manifest() -> RolloutManifest:
    """One member per fresh-PRO train suite before releasing its full 640 rows."""
    full = build_fresh_pro_train_manifest()
    first_by_suite = {}
    for item in full.items:
        first_by_suite.setdefault(item.suite, item)
    items = tuple(
        ManifestItem(ordinal=ordinal, suite=item.suite, task_idx=item.task_idx,
                     init_state_index=item.init_state_index,
                     behavior_seed_index=item.behavior_seed_index, tier="pro_fresh_sentinel",
                     selection_reason="fresh_state_preflight_artifact_validation")
        for ordinal, item in enumerate(first_by_suite.values()))
    return RolloutManifest(
        name=f"pcp-search-pro-fresh-state-train-sentinel-{len(items)}", items=items,
        policy_repo_id=POLICY_REPO_ID, policy_revision=POLICY_REVISION,
        collection_config={**full.collection_config, "sentinel": True},
        parent_manifest_id=full.manifest_id,
        provenance={**full.provenance, "sentinel_for_manifest": full.manifest_id})


def validate_pro_manifest(manifest: RolloutManifest) -> None:
    """Fail closed if a future edit makes a suite usable for both fit and evaluation."""
    _validate_partition()
    fresh = manifest.collection_config.get("collection_round") == "fresh_state_v1"
    expected = PRO_TRAIN_QUOTAS if (fresh or manifest.collection_config.get("data_split") == "train") \
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
    if fresh:
        if manifest.collection_config.get("partition_id") != FRESH_PRO_TRAIN_PARTITION_ID:
            raise ValueError("fresh PRO manifest has the wrong partition ID")
        for item in manifest.items:
            if item.suite in PRO_TEN_STATE_SUITES:
                if not (0 <= item.init_state_index < 10 and item.behavior_seed_index == 1):
                    raise ValueError("fresh ten-state PRO suite must use state 0--9 and seed 1")
            elif not (item.init_state_index >= FRESH_STATE_START and item.behavior_seed_index == 0):
                raise ValueError("fresh PRO physical-state suite must use unseen states and seed 0")
