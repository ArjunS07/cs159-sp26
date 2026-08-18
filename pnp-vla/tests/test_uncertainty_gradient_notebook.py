import json
from pathlib import Path


def test_gradient_analysis_notebook_is_clean_and_complete():
    path = Path("notebooks/49_analyze_direct_u20_gradient_pro220.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in (
            "REQUIRE_FULL_COHORT = False",
            "EXPECTED_IDENTITIES = 220",
            "fetch_direct_gradient_rows",
            "match_direct_gradient_cohort",
            "paired_effect_table",
            "gradient_telemetry_table",
            "telemetry_by_outcome_transition",
            "first_u20_gate_sweep",
            "gradient minus random control",
            "episodes_in_sr_denominator"):
        assert required in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")
