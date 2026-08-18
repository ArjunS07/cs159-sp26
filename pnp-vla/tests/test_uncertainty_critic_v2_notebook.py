import ast
import json
from pathlib import Path


def test_v2_notebook_is_step_aligned_and_contains_direct_gradient_control():
    path = Path(__file__).parents[1] / "notebooks" / "47_train_residual_u20_critic_v2.ipynb"
    document = json.loads(path.read_text(encoding="utf-8"))
    source = ""
    for cell in document["cells"]:
        text = "".join(cell.get("source", []))
        source += text
        if cell["cell_type"] == "code":
            ast.parse(text)
    assert "ranking_weight=3.0" in source
    assert "DIRECT_PROBE_STEPS = (3, 4)" in source
    assert "direct_uncertainty_gradient_test(" in source
    assert "descent_delta_u20" in source
    assert "random_delta_u20" in source
    assert "uses no LIBERO-PRO data" in source
