from collections import Counter

import pytest

from pnp.pcp_search.pro import (
    FRESH_PRO_TRAIN_PARTITION_ID,
    FRESH_STATE_START,
    PRO_HELDOUT_QUOTAS,
    PRO_TEN_STATE_SUITES,
    PRO_TRAIN_QUOTAS,
    PRO_EVALUATION_ONLY_SUITES,
    build_fresh_pro_train_manifest,
    build_fresh_pro_train_sentinel_manifest,
    build_pro_manifest,
    build_pro_sentinel_manifest,
    pro_partition_summary,
    validate_pro_manifest,
)


def test_pro_partition_is_exact_whole_suite_and_disjoint():
    summary = pro_partition_summary()
    assert summary["train_eligible_rollouts"] == 640
    assert summary["heldout_rollouts"] == 160
    assert not (set(PRO_TRAIN_QUOTAS) & set(PRO_HELDOUT_QUOTAS))
    assert not any("_temp_" in suite for suite in PRO_TRAIN_QUOTAS)
    assert all("_temp_" in suite for suite in PRO_HELDOUT_QUOTAS)
    assert set(PRO_EVALUATION_ONLY_SUITES).isdisjoint(PRO_TRAIN_QUOTAS)


@pytest.mark.parametrize("split, expected", [("train", 640), ("heldout", 160)])
def test_pro_manifests_are_frozen_ready_and_train_safe(split, expected):
    manifest = build_pro_manifest(split=split)
    assert len(manifest.items) == expected
    assert manifest.collection_config["benchmark"] == "libero_pro"
    assert manifest.collection_config["n_action_steps"] == 10
    assert manifest.collection_config["train_eligible"] is (split == "train")
    assert Counter(item.suite for item in manifest.items) == (
        PRO_TRAIN_QUOTAS if split == "train" else PRO_HELDOUT_QUOTAS)
    validate_pro_manifest(manifest)


def test_pro_manifest_round_trip_preserves_content_address():
    manifest = build_pro_manifest(split="train")
    assert type(manifest).from_json(manifest.to_json()).manifest_id == manifest.manifest_id


@pytest.mark.parametrize("split, expected", [("train", 8), ("heldout", 6)])
def test_pro_sentinel_covers_each_suite_once(split, expected):
    manifest = build_pro_sentinel_manifest(split=split)
    assert len(manifest.items) == expected
    assert manifest.collection_config["sentinel"] is True
    validate_pro_manifest(manifest)


def test_fresh_pro_train_uses_new_states_or_explicit_new_behavior_seeds():
    initial = build_pro_manifest(split="train")
    fresh = build_fresh_pro_train_manifest()
    assert len(fresh.items) == 640
    assert fresh.parent_manifest_id == initial.manifest_id
    assert fresh.collection_config["partition_id"] == FRESH_PRO_TRAIN_PARTITION_ID
    assert Counter(item.suite for item in fresh.items) == PRO_TRAIN_QUOTAS
    assert {(i.suite, i.task_idx, i.init_state_index, i.behavior_seed_index) for i in fresh.items}.isdisjoint(
        {(i.suite, i.task_idx, i.init_state_index, i.behavior_seed_index) for i in initial.items})
    for item in fresh.items:
        if item.suite in PRO_TEN_STATE_SUITES:
            assert 0 <= item.init_state_index < 10
            assert item.behavior_seed_index == 1
        else:
            assert item.init_state_index >= FRESH_STATE_START
            assert item.behavior_seed_index == 0
    validate_pro_manifest(fresh)


def test_fresh_pro_sentinel_is_a_recoverable_member_of_full_manifest():
    full = build_fresh_pro_train_manifest()
    sentinel = build_fresh_pro_train_sentinel_manifest()
    assert len(sentinel.items) == len(PRO_TRAIN_QUOTAS)
    full_identity = {(i.suite, i.task_idx, i.init_state_index, i.behavior_seed_index) for i in full.items}
    assert {(i.suite, i.task_idx, i.init_state_index, i.behavior_seed_index) for i in sentinel.items} <= full_identity
    validate_pro_manifest(sentinel)
