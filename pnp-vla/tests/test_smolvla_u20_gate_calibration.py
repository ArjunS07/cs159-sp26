import io
import json
from pathlib import Path

import numpy as np

from pnp.config import Method
from pnp.smolvla_u20_gate_calibration import (
    SMOLVLA_U20_GATE_CALIBRATION_IDENTITIES,
    SMOLVLA_U20_GATE_CALIBRATION_INDICES,
    SMOLVLA_U20_GATE_QUANTILE,
    build_smolvla_u20_gate_calibration_method,
    build_smolvla_u20_gate_calibration_methods,
    pair_weighted_u20_profile,
)


def test_calibration_method_is_stock_measurement_only():
    method, config = build_smolvla_u20_gate_calibration_method()
    assert method == Method.UNCERTAINTY
    assert not config.refine
    assert tuple(config.pnp_steps) == (1, 2, 3)
    assert tuple(config.pnp_k_by_step) == (3, 1, 1)
    assert config.num_inference_steps == 10
    assert config.n_action_steps == 10
    assert config.save_time_uncertainty
    assert not config.save_trajectory and not config.save_training_data
    assert SMOLVLA_U20_GATE_CALIBRATION_INDICES == tuple(range(10, 15))
    assert SMOLVLA_U20_GATE_CALIBRATION_IDENTITIES == 200
    assert SMOLVLA_U20_GATE_QUANTILE == 0.75

    methods = build_smolvla_u20_gate_calibration_methods()
    assert [name for name, _ in methods] == [
        Method.UNCERTAINTY, Method.SMOLVLA_CONSENSUS_PROJECT_S03_K3]
    assert methods[1][1].consensus_projection_step == 7
    assert methods[1][1].consensus_projection_k == 3
    assert methods[1][1].n_action_steps == 10


def test_pair_weighted_u20_profile_uses_311_weights():
    archive = io.BytesIO()
    arrays = {}
    for chunk in range(2):
        for step, value in ((1, 1.0 + chunk), (2, 3.0 + chunk), (3, 6.0 + chunk)):
            arrays[f"c{chunk}_s{step}_u_time"] = np.full(50, value, np.float32)
    np.savez_compressed(archive, **arrays)
    assert np.allclose(pair_weighted_u20_profile(archive.getvalue()), (2.4, 3.4))


def test_calibration_worker_notebook_contract():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "107_smolvla_stock_u20_gate_calibration_idx10_14.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_u20_gate_calibration_worker" in source
    assert "summarize_smolvla_u20_gate_calibration" in source
    assert "list(range(10, 15))" in source
    assert "'identities': 200" in source
    assert "'new_rollouts': 400" in source
    assert "'n_action_steps': 10" in source
    assert "'refine': False" in source
    assert "always-on blend" in source
    assert "'projection_s': 0.3" in source
    assert "'planned_gate_quantile': 0.75" in source
    assert "'video': 'off'" in source
