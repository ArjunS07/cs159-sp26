from pnp.verifier.collection import (
    build_stratified_manifest, candidate_group_id,
)


def test_candidate_group_id_is_stable_and_identity_sensitive():
    a = candidate_group_id("libero", "s", 1, 2, 3)
    assert a == candidate_group_id("libero", "s", 1, 2, 3)
    assert a != candidate_group_id("libero", "s", 1, 2, 4)
    assert a != candidate_group_id("libero", "s", 1, 2, 3, namespace="round-2")


def test_manifest_uses_mixed_tasks_and_one_state_per_rollout():
    rollouts, steps = [], []
    for task in range(2):
        for episode in range(10):
            rid = f"r-{task}-{episode}"
            rollouts.append({"rollout_id": rid, "benchmark": "libero", "suite": "s",
                             "task_idx": task, "episode_idx": episode,
                             "success": episode < 5 if task == 0 else True})
            for chunk in range(2):
                steps.append({"rollout_id": rid, "chunk_idx": chunk,
                              "u_mean": episode + chunk / 10})
    manifest = build_stratified_manifest(rollouts, steps, targets={"libero": 6})
    assert len(manifest) == 6
    assert {row["task_idx"] for row in manifest} == {0}
    assert len({row["rollout_id"] for row in manifest}) == 6
    assert {row["uncertainty_stratum"] for row in manifest} == {"low", "mid", "high"}
