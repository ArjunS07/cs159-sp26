import ast
import json
from pathlib import Path
from types import SimpleNamespace

from pnp.config import ALL_METHODS, Method
from pnp.qplanning_priority_eval_experiment import (
    QPLANNING_PRIORITY_EVAL_UPDATE, QPLANNING_PRIORITY_METHODS,
    build_qplanning_priority_method)


ROOT = Path(__file__).parents[1]


def test_priority_eval_methods_are_distinct_fixed_q50_planners():
    assert QPLANNING_PRIORITY_EVAL_UPDATE == 6_000
    assert QPLANNING_PRIORITY_METHODS == {
        "failure": Method.QPLANNING_Q50_PRIORITY_FAILURE,
        "u20_4chunk": Method.QPLANNING_Q50_PRIORITY_U20_4CHUNK,
    }
    assert set(QPLANNING_PRIORITY_METHODS.values()).issubset(ALL_METHODS)
    configs = []
    for strategy, expected_method in QPLANNING_PRIORITY_METHODS.items():
        scorer = SimpleNamespace(horizon=50, checkpoint_id=f"critic-{strategy}")
        method, config = build_qplanning_priority_method(
            strategy, "repo@revision", scorer, candidate_batch_size=8)
        assert method == expected_method
        assert config.num_samples == 64 and config.qplanning_n_elites == 16
        assert config.num_inference_steps == 3 and config.n_action_steps == 10
        assert config.qplanning_ckpt_id == f"critic-{strategy}"
        assert config.video == "off" and not config.save_observations
        assert not config.save_generated_chunks
        configs.append(config)
    assert configs[0].candidate_set_id == configs[1].candidate_set_id


def test_single_priority_eval_notebook_runs_both_critics_with_historical_stock():
    path = ROOT / "notebooks" / "74_eval_q50_priority_heldout160.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "single full PRO160 worker" in source
    assert "EPISODE_LIMIT = None" in source
    assert "checkpoint_step_006000.pt" in source
    assert "run_qplanning_priority_heldout160(" in source
    assert "FAILURE_CHECKPOINT_PATH" in source
    assert "U20_4CHUNK_CHECKPOINT_PATH" in source
    assert "reuse exact matched notebook-68 rows" in source
    assert "periodic_print_every_complete_identities': 10" in source
    assert "video" in source.lower() and "frames" in source.lower()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"{path.name}:cell{index}")
