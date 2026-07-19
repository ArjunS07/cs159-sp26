import json
from pathlib import Path

from pnp import Method
from pnp.experiments import (
    FULL_ABLATION_TASKS,
    SCHEDULES,
    build_broad_methods,
    build_full_methods,
    build_pro_methods,
    identity_shard,
)
from pnp.store import SupabaseStore


def test_automated_worker_matrices_are_complete_and_unique():
    full = build_full_methods()
    broad = build_broad_methods(full)
    assert len(full) == 12
    assert len(broad) == 3
    assert len({
        SupabaseStore.config_hash(SupabaseStore._logical_key(name, config))
        for name, config in full
    }) == 12
    assert [name for name, _ in broad] == [
        Method.UNCERTAINTY, Method.EXTRA_STEPS, Method.REFINEMENT,
    ]
    assert broad[1][1].num_inference_steps == 16
    assert broad[2][1].pnp_steps == (4, 5)
    assert not broad[2][1].refine_average
    refinements = [config for name, config in full if name == Method.REFINEMENT]
    assert {config.pnp_steps for config in refinements} == set(SCHEDULES)
    assert all(not config.refine_average for config in refinements)


def test_automated_worker_shards_are_disjoint_and_complete():
    episodes = [
        {"suite": "suite", "task_idx": task, "ep_idx": ep, "init_state_hash": f"{task}-{ep}"}
        for task in range(3) for ep in range(5)
    ]
    shards = [identity_shard(episodes, 6, index) for index in range(6)]
    keys = lambda items: {(ep["task_idx"], ep["ep_idx"]) for ep in items}
    assert set().union(*(keys(shard) for shard in shards)) == keys(episodes)
    assert sum(len(keys(shard)) for shard in shards) == len(episodes)


def test_canonical_pro_worker_matrix_is_three_unique_configs():
    methods = build_pro_methods()
    assert [name for name, _ in methods] == [
        Method.UNCERTAINTY, Method.EXTRA_STEPS, Method.REFINEMENT,
    ]
    assert methods[0][1].pnp_steps == tuple(range(1, 10))
    assert methods[0][1].pnp_k == 3
    assert methods[0][1].save_pcp_features
    assert methods[1][1].num_inference_steps == 16
    assert methods[2][1].pnp_steps == (4, 5)
    assert methods[2][1].pnp_k == 3
    assert methods[2][1].refine and not methods[2][1].refine_average
    assert len({
        SupabaseStore.config_hash(SupabaseStore._logical_key(name, config))
        for name, config in methods
    }) == 3


def test_automated_worker_cohort_is_historical_eight_tasks():
    assert len(FULL_ABLATION_TASKS) == 8
    assert {suite for suite, _ in FULL_ABLATION_TASKS} == {"libero_spatial", "libero_goal"}


def test_six_launchers_have_fixed_unique_indices():
    worker_dir = Path(__file__).parents[1] / "notebooks" / "workers"
    launchers = sorted(worker_dir.glob("libero_worker_*.ipynb"))
    assert len(launchers) == 6
    for index, path in enumerate(launchers):
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "SHARD_COUNT = 6" in source
        assert f"SHARD_INDEX = {index}" in source
        assert "run_libero_hybrid_worker" in source


def test_six_pro_launchers_have_fixed_unique_indices():
    worker_dir = Path(__file__).parents[1] / "notebooks" / "workers"
    launchers = sorted(worker_dir.glob("libero_pro_worker_*.ipynb"))
    assert len(launchers) == 6
    for index, path in enumerate(launchers):
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "SHARD_COUNT = 6" in source
        assert f"SHARD_INDEX = {index}" in source
        assert "run_libero_pro_worker" in source
