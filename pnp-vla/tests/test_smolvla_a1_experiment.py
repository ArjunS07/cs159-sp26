import json
from pathlib import Path

from pnp.config import Method
from pnp.smolvla_a1_experiment import (
    SMOLVLA_A1_EXPERIMENT,
    SMOLVLA_A1_PNP_EXPERIMENT,
    build_smolvla_a1_method,
    build_smolvla_a1_pnp_method,
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


def test_smolvla_a1_pnp_method_contract():
    method, config = build_smolvla_a1_pnp_method()
    assert method == Method.SMOLVLA_PNP_S123_K311_A1
    assert tuple(config.pnp_steps) == (1, 2, 3)
    assert tuple(config.pnp_k_by_step) == (3, 1, 1)
    assert config.refine
    assert config.n_action_steps == 1
    assert config.num_inference_steps == 10
    assert config.save_time_uncertainty
    assert SMOLVLA_A1_PNP_EXPERIMENT == "smolvla-libero-a1-pnp-steps123-k311-v1"


def test_a1_pnp_worker_notebook_is_explicit():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "99_smolvla_libero_a1_pnp_steps123_k311_eval.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_a1_pnp_eval_worker" in source
    assert "libEGL_nvidia" in source
    assert "'identities': 400" in source
    assert "'n_action_steps': 1" in source
    assert "'pnp_steps': [1,2,3]" in source
    assert "'pnp_k_by_step': [3,1,1]" in source
    assert "notebook 98 stock A1" in source


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
