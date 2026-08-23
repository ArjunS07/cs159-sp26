from collections import Counter

from pnp.pcp_search.manifest import RolloutManifest
from pnp.pcp_search.task_selection import (
    ALL_TASKS,
    TIER_A,
    TIER_B,
    TIER_C,
    build_initial_manifest,
    build_next_tranche_manifest,
    initial_task_allocation,
)


def _history():
    rows = []
    for suite, task_idx in ALL_TASKS:
        for episode_idx in range(20, 40):
            rows.append({
                "rollout_id": f"{suite}-{task_idx}-{episode_idx}",
                "suite": suite,
                "task_idx": task_idx,
                "episode_idx": episode_idx,
                "init_state_hash": f"h-{episode_idx}",
                "status": "completed",
                "success": episode_idx >= 30,
                "u20": episode_idx / 1000,
                "n_chunks": 20 + task_idx,
            })
    return rows


def test_approved_initial_allocation_is_exact():
    allocation = initial_task_allocation()
    assert len(allocation) == 40
    assert sum(allocation.values()) == 400
    assert {allocation[task] for task in TIER_A} == {30}
    assert {allocation[task] for task in TIER_B} == {20}
    assert {allocation[task] for task in TIER_C} == {13}
    suite_totals = Counter()
    for (suite, _), count in allocation.items():
        suite_totals[suite] += count
    assert suite_totals == {
        "libero_10": 118,
        "libero_goal": 133,
        "libero_object": 50,
        "libero_spatial": 99,
    }


def test_initial_manifest_is_content_addressed_and_uses_new_tier_a_behavior_seeds():
    first = build_initial_manifest(_history())
    second = build_initial_manifest(_history())
    assert first.manifest_id == second.manifest_id
    assert len(first.items) == 400
    assert len({item.identity for item in first.items}) == 400
    assert RolloutManifest.from_json(first.to_json()).manifest_id == first.manifest_id
    for task in TIER_A:
        items = [item for item in first.items if item.task_key == task]
        assert len(items) == 30
        assert sum(item.behavior_seed_index == 0 for item in items) == 20
        assert sum(item.selection_reason == "prior_failure_new_behavior_seed"
                   for item in items) == 10


def test_adaptive_tranche_is_200_in_five_rollout_blocks_with_cap():
    rows = _history()
    coverage = {task: 100 for task in ALL_TASKS}
    manifest = build_next_tranche_manifest(
        rows, coverage, parent_manifest_id="pcps-parent", tranche_index=1)
    counts = Counter(item.task_key for item in manifest.items)
    assert len(manifest.items) == 200
    assert all(count % 5 == 0 for count in counts.values())
    assert max(counts.values()) <= 50
    assert all(item.behavior_seed_index > 0 for item in manifest.items)
