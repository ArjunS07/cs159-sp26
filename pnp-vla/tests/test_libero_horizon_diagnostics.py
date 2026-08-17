import ast
import json
from pathlib import Path

from pnp.config import Method
from pnp.experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_HORIZON_DIAGNOSTIC_K,
    LIBERO_HORIZON_DIAGNOSTIC_STEPS,
    LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT,
    build_libero_horizon_diagnostic_methods,
)


ROOT = Path(__file__).parents[1]


def _notebook_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_diagnostic_method_is_one_noop_arm_with_required_sinks():
    methods = build_libero_horizon_diagnostic_methods()
    assert len(methods) == 1
    method, config = methods[0]
    assert method == Method.UNCERTAINTY
    assert config.pnp_k == LIBERO_HORIZON_DIAGNOSTIC_K == 5
    assert config.pnp_steps == LIBERO_HORIZON_DIAGNOSTIC_STEPS == (3, 4)
    assert config.n_action_steps == LIBERO_ACTION_STEPS == 10
    assert not config.refine
    assert not config.refine_average
    assert config.save_pcp_features
    assert config.save_time_uncertainty
    assert config.save_trajectory
    assert config.skip_unused_renders
    assert config.render_lead == 2
    assert LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT not in {
        "libero-hybrid-schedules-k3-a10-v2",
        "worker-41-horizon-multiquery-v1",
    }


def test_four_worker_notebooks_are_disjoint_and_explicit():
    paths = sorted((ROOT / "notebooks" / "workers").glob(
        "43_libero_horizon_diagnostic_worker_*.ipynb"))
    assert len(paths) == 4
    sources = [_notebook_source(path) for path in paths]
    assert {f"SHARD_INDEX = {index}" for index in range(4)} == {
        next(line for line in source.splitlines() if line.startswith("SHARD_INDEX ="))
        for source in sources
    }
    for source in sources:
        assert "SHARD_COUNT = 4" in source
        assert "run_libero_horizon_diagnostic_worker" in source
        assert "LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT" in source
        for token in ("U10", "U20", "U50", "contraction", "PCP"):
            assert token in source


def test_analysis_notebook_contains_failure_contraction_and_q_sections():
    source = _notebook_source(
        ROOT / "notebooks" / "44_analyze_standard_libero_horizon_diagnostics.ipynb")
    required = (
        "EXPECTED_IDENTITIES = 400",
        "LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT",
        "load_horizon_artifacts",
        "failure_auc_table",
        "prefix_failure_auc_table",
        "contraction",
        "LIBERO_10STEP_EXPERIMENT",
        "window_sweep",
        "episodes_in_sr_denominator",
        "pcp_chunks_path",
        "TrainConfig(correction_steps=(3,4))",
    )
    for token in required:
        assert token in source
