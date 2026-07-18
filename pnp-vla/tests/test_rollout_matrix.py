import json
from pathlib import Path

from pnp import Method, RolloutConfig
from pnp.libero_pro import (
    CANONICAL_PRO_SUITES,
    EXPANDED_PRO_SUITES,
    UNION_PRO_SUITES,
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
    definitions = cell.split("EXPERIMENT = 'libero-full-schedules-k3-v1'")[0]
    namespace = {
        "Method": Method,
        "RolloutConfig": RolloutConfig,
        "store": object.__new__(SupabaseStore),
    }
    exec(definitions, namespace)
    return namespace["SCHEDULES"], namespace["METHODS"]


def test_full_rollout_matrix_is_complete_and_unique():
    schedules, methods = _notebook_methods()

    assert schedules == (
        (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (1, 3, 5, 7, 9), (3, 6, 9), (2, 5, 8),
    )
    assert len(methods) == 27

    hashes = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(name, config))
        for name, config in methods
    }
    assert len(hashes) == 27

    controls = sorted(
        config.num_inference_steps
        for name, config in methods
        if name == Method.EXTRA_STEPS
    )
    assert controls == [16, 19, 25]

    observed = [config for name, config in methods if name == Method.UNCERTAINTY]
    assert len(observed) == 8
    assert [config.pnp_steps for config in observed if config.save_pcp_features] == [(7, 8)]


def test_pro_manifest_is_a_stable_deduplicated_union():
    assert len(CANONICAL_PRO_SUITES) == 6
    assert len(EXPANDED_PRO_SUITES) == 16
    assert set(CANONICAL_PRO_SUITES) <= set(EXPANDED_PRO_SUITES)
    assert UNION_PRO_SUITES == list(dict.fromkeys(
        CANONICAL_PRO_SUITES + EXPANDED_PRO_SUITES
    ))

