from pnp.verifier.collection import (
    build_stratified_manifest, build_targeted_manifests, candidate_group_id,
    build_seeded_pro_manifest, collection_manifest_hash,
)


def test_candidate_group_id_is_stable_and_identity_sensitive():
    a = candidate_group_id("libero", "s", 1, 2, 3)
    assert a == "db118fe14552782aa0419a80"  # Legacy resume compatibility.
    assert a == candidate_group_id("libero", "s", 1, 2, 3)
    assert a != candidate_group_id("libero", "s", 1, 2, 4)
    assert a != candidate_group_id("libero", "s", 1, 2, 3, namespace="round-2")
    assert a != candidate_group_id(
        "libero", "s", 1, 2, 3, trajectory_seed=123)


def test_seeded_pro_manifest_is_deterministic_and_split_disjoint():
    rows = [{
        "rollout_id": f"r-{episode}", "benchmark": "libero_pro",
        "suite": f"suite-{episode % 2}", "task_idx": episode % 5,
        "episode_idx": episode, "chunk_idx": 2, "u_mean": episode,
        "success": episode % 3 != 0, "uncertainty_stratum": "high",
    } for episode in range(50)]
    first = build_seeded_pro_manifest(
        rows, development_target=24, test_target=16, seed=4)
    second = build_seeded_pro_manifest(
        list(reversed(rows)), development_target=24, test_target=16, seed=4)
    assert first == second
    development = {(row["suite"], row["task_idx"], row["episode_idx"],
                    row["trajectory_seed"]) for row in first["development"]}
    test = {(row["suite"], row["task_idx"], row["episode_idx"],
             row["trajectory_seed"]) for row in first["confirmatory_test"]}
    assert len(development) == 24 and len(test) == 16
    assert development.isdisjoint(test)
    assert {row["collection_split"] for row in first["development"]} == {
        "development"}
    test_suite_counts = {
        suite: sum(row["suite"] == suite for row in first["confirmatory_test"])
        for suite in {row["suite"] for row in first["confirmatory_test"]}}
    assert max(test_suite_counts.values()) - min(test_suite_counts.values()) <= 1


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


def test_targeted_manifests_are_deterministic_disjoint_and_failure_enriched():
    rollouts, steps = [], []
    for benchmark in ("libero", "libero_pro"):
        for episode in range(30):
            rollout_id = f"{benchmark}-{episode}"
            rollouts.append({
                "rollout_id": rollout_id, "benchmark": benchmark, "suite": "s",
                "task_idx": episode % 3, "episode_idx": episode,
                "success": episode % 4 != 0,
            })
            for chunk in range(3):
                steps.append({
                    "rollout_id": rollout_id, "chunk_idx": chunk,
                    "u_mean": episode + chunk / 10,
                })
    excluded = {("libero", "s", 0, 0), ("libero_pro", "s", 0, 0)}
    kwargs = {
        "development_targets": {"libero": 6, "libero_pro": 6},
        "test_targets": {"libero": 3, "libero_pro": 3},
        "development_failure_fraction": .75,
        "seed": 9,
    }
    first = build_targeted_manifests(rollouts, steps, excluded, **kwargs)
    second = build_targeted_manifests(
        list(reversed(rollouts)), list(reversed(steps)), excluded, **kwargs)
    assert first == second
    assert collection_manifest_hash(first["development"]) == collection_manifest_hash(
        second["development"])
    development_ids = {
        (row["benchmark"], row["suite"], row["task_idx"], row["episode_idx"])
        for row in first["development"]}
    test_ids = {
        (row["benchmark"], row["suite"], row["task_idx"], row["episode_idx"])
        for row in first["test"]}
    assert len(development_ids) == 12
    assert len(test_ids) == 6
    assert development_ids.isdisjoint(test_ids | excluded)
    assert sum(not row["success"] for row in first["development"]) >= 4
    assert all(row["uncertainty_stratum"] == "high"
               for rows in first.values() for row in rows)


def test_targeted_manifest_uses_each_episodes_highest_uncertainty_state():
    rollouts = [{
        "rollout_id": f"r{episode}", "benchmark": "libero", "suite": "s",
        "task_idx": 0, "episode_idx": episode, "success": episode % 2 == 0,
    } for episode in range(20)]
    steps = [{
        "rollout_id": f"r{episode}", "chunk_idx": chunk,
        "u_mean": episode + chunk / 10,
    } for episode in range(20) for chunk in range(3)]
    manifests = build_targeted_manifests(
        rollouts, steps, set(),
        development_targets={"libero": 10},
        test_targets={"libero": 5},
        development_failure_fraction=.5,
    )
    selected = manifests["development"] + manifests["test"]
    assert len(selected) == 15
    assert len({row["episode_idx"] for row in selected}) == 15
    assert {row["chunk_idx"] for row in selected} == {2}


def test_targeted_manifest_can_accept_a_prospective_shortfall():
    rollouts = [{
        "rollout_id": f"r{i}", "benchmark": "libero", "suite": "s",
        "task_idx": 0, "episode_idx": i, "success": True,
    } for i in range(3)]
    steps = [{"rollout_id": f"r{i}", "chunk_idx": 0, "u_mean": i}
             for i in range(3)]
    manifests = build_targeted_manifests(
        rollouts, steps, set(), development_targets={"libero": 0},
        test_targets={"libero": 10}, allow_shortfall=True)
    assert len(manifests["test"]) == 3
