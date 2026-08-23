from collections import Counter

import pytest

from pnp.pcp_search.pro import (
    PRO_HELDOUT_QUOTAS,
    PRO_TRAIN_QUOTAS,
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
    assert {"libero_10_swap", "libero_object_task", "libero_object_temp_x0.3"} <= set(PRO_HELDOUT_QUOTAS)


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


@pytest.mark.parametrize("split, expected", [("train", 9), ("heldout", 7)])
def test_pro_sentinel_covers_each_suite_once(split, expected):
    manifest = build_pro_sentinel_manifest(split=split)
    assert len(manifest.items) == expected
    assert manifest.collection_config["sentinel"] is True
    validate_pro_manifest(manifest)
