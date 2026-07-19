import json
from pathlib import Path

from pnp import Method, RolloutConfig
from pnp.libero_pro import (
    CANONICAL_PRO_SUITES,
    EXPANDED_PRO_SUITES,
    UNION_PRO_SUITES,
    _with_dynamic_suites,
)
from pnp.store import SupabaseStore


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01_run_experiments.ipynb"


def _notebook_methods():
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(
        "".join(item["source"])
        for item in notebook["cells"]
        if "def build_schedule_methods" in "".join(item.get("source", []))
    )
    definitions = cell.split("EXPERIMENT = 'libero-hybrid-schedules-k3-v1'")[0]
    namespace = {
        "Method": Method,
        "RolloutConfig": RolloutConfig,
        "store": object.__new__(SupabaseStore),
    }
    exec(definitions, namespace)
    return namespace


def test_full_rollout_matrix_is_complete_and_unique():
    namespace = _notebook_methods()
    schedules, methods = namespace["SCHEDULES"], namespace["FULL_METHODS"]

    assert schedules == (
        (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (1, 3, 5, 7, 9), (3, 6, 9), (2, 5, 8),
    )
    assert len(methods) == 20

    hashes = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(name, config))
        for name, config in methods
    }
    assert len(hashes) == 20

    controls = sorted(
        config.num_inference_steps
        for name, config in methods
        if name == Method.EXTRA_STEPS
    )
    assert controls == [16, 19, 25]

    observed = [config for name, config in methods if name == Method.UNCERTAINTY]
    assert len(observed) == 1
    assert observed[0].pnp_steps == tuple(range(1, 10))
    assert observed[0].save_pcp_features

    refinements = [config for name, config in methods if name == Method.REFINEMENT]
    assert len(refinements) == 16
    assert {config.pnp_steps for config in refinements} == set(schedules)

    broad = namespace["BROAD_METHODS"]
    assert [name for name, _ in broad] == [
        Method.UNCERTAINTY, Method.EXTRA_STEPS, Method.REFINEMENT,
    ]
    assert broad[1][1].num_inference_steps == 16
    assert broad[2][1].pnp_steps == (4, 5)
    assert not broad[2][1].refine_average


def test_hybrid_cohort_has_eight_historical_tasks():
    tasks = _notebook_methods()["FULL_ABLATION_TASKS"]
    assert len(tasks) == 8
    assert {suite for suite, _ in tasks} == {"libero_spatial", "libero_goal"}


def test_identity_shards_are_disjoint_and_complete():
    namespace = _notebook_methods()
    episodes = [
        {"suite": "suite", "task_idx": task, "ep_idx": ep, "init_state_hash": f"{task}-{ep}"}
        for task in range(3) for ep in range(5)
    ]
    shards = []
    for shard_index in range(4):
        namespace["SHARD_COUNT"] = 4
        namespace["SHARD_INDEX"] = shard_index
        shards.append(namespace["identity_shard"](episodes))

    keys = lambda shard: {(ep["task_idx"], ep["ep_idx"]) for ep in shard}
    assert set().union(*(keys(shard) for shard in shards)) == keys(episodes)
    assert sum(len(keys(shard)) for shard in shards) == len(episodes)


def test_pro_manifest_is_a_stable_deduplicated_union():
    assert len(CANONICAL_PRO_SUITES) == 6
    assert len(EXPANDED_PRO_SUITES) == 16
    assert set(CANONICAL_PRO_SUITES) <= set(EXPANDED_PRO_SUITES)
    assert UNION_PRO_SUITES == list(dict.fromkeys(
        CANONICAL_PRO_SUITES + EXPANDED_PRO_SUITES
    ))


def test_dynamic_pro_suite_registration_fills_task_map_only_suites():
    class Benchmark:
        def __init__(self, task_order_index=0):
            self.task_order_index = task_order_index

        def _make_benchmark(self):
            self.tasks = list(FakeBenchmark.task_maps[self.name].values())
            self.n_tasks = len(self.tasks)

    class BuiltIn(Benchmark):
        pass

    class FakeBenchmark:
        task_maps = {
            "built_in": {"a": object()},
            "libero_object_temp_x0.1": {"a": object(), "b": object()},
        }

        @staticmethod
        def get_benchmark_dict():
            return {"built_in": BuiltIn}

    FakeBenchmark.Benchmark = Benchmark

    result = _with_dynamic_suites(
        FakeBenchmark, {"built_in": ["a"], "libero_object_temp_x0.1": ["a", "b"]})
    assert result["built_in"] is BuiltIn
    dynamic = result["libero_object_temp_x0.1"]()
    assert dynamic.name == "libero_object_temp_x0.1"
    assert dynamic.n_tasks == 2
