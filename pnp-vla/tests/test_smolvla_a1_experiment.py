import json
from pathlib import Path

from pnp.config import Method
from pnp.smolvla_a1_experiment import (
    SMOLVLA_A1_EXPERIMENT,
    build_smolvla_a1_method,
)


def test_smolvla_a1_method_contract():
    method, config = build_smolvla_a1_method()
    assert method == Method.SMOLVLA_STOCK_A1
    assert config.n_action_steps == 1
    assert config.num_inference_steps == 10
    assert not config.has_probe
    assert not config.refine
    assert config.save_trajectory
    assert config.video == "off"
    assert SMOLVLA_A1_EXPERIMENT == "smolvla-libero-stock-a1-v1"


def test_a1_worker_notebook_is_explicit():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "98_smolvla_libero_stock_a1_eval.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_a1_eval_worker" in source
    assert "libEGL_nvidia" in source
    assert "'identities': 400" in source
    assert "'new_rollouts': 400" in source
    assert "'n_action_steps': 1" in source
    assert "historic stock A10" in source
    assert "'video': 'off'" in source


def test_projection_analysis_notebook_has_core_diagnostics():
    path = (Path(__file__).parents[1] / "notebooks"
            / "97_analyze_smolvla_consensus_projection.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "summarize_pair" in source
    assert "failure_auc_table" in source
    assert "flip_auc_table" in source
    assert "parent_weighted" in source
    assert "projection" in source
    assert "sr_delta_by_suite.png" in source

