import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "39_analyze_standard_libero_10_action.ipynb"


def _sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_is_locked_to_corrected_ten_action_experiment():
    notebook, source = _sources()
    assert notebook["nbformat"] == 4
    assert "LIBERO_10STEP_EXPERIMENT" in source
    assert "EXPECTED_N_ACTION_STEPS = 10" in source
    assert "logged_horizons == {EXPECTED_N_ACTION_STEPS}" in source
    assert "validate_standard(rollouts, runs)" in source


def test_notebook_contains_requested_auc_delta_window_and_chunk_analyses():
    _, source = _sources()
    for required in (
        "detector_by_suite",
        "per_suite_deltas",
        "paired_bootstrap_ci",
        "window_sr_all_episodes",
        "episodes_in_sr_denominator",
        "prefix_failure_auc",
        "individual_chunk_failure_auc",
        "best_window_delta_by_chunk_horizon",
    ):
        assert required in source


def test_all_code_cells_parse():
    notebook, _ = _sources()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")
