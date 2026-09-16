import json
from pathlib import Path

import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.pnp import (
    _pnp_seed_perturb,
    extend_probe_with_final_prediction,
    run_probe,
)
from pnp.smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
    build_smolvla_schedule_method,
)


def test_per_step_pnp_schedule_is_validated_and_hash_compatible():
    historical = RolloutConfig(pnp_steps=(1, 2, 3), pnp_k=3)
    assert all(historical.probe_k(step) == 3 for step in (1, 2, 3))
    assert "pnp_k_by_step" not in historical.logical_dict()

    scheduled = RolloutConfig(
        pnp_steps=(1, 2, 3), pnp_k=3,
        pnp_k_by_step=(3, 1, 1), refine=True)
    assert [scheduled.probe_k(step) for step in (1, 2, 3)] == [3, 1, 1]
    assert scheduled.logical_dict()["pnp_k_by_step"] == (3, 1, 1)
    with pytest.raises(ValueError, match="one-to-one"):
        RolloutConfig(pnp_steps=(1, 2, 3), pnp_k_by_step=(3, 1))


def test_schedule_method_contract():
    method, schedule = build_smolvla_schedule_method()
    assert method == Method.SMOLVLA_PNP_S123_K311
    assert schedule.refine
    assert tuple(schedule.pnp_steps) == SMOLVLA_SCHEDULE_STEPS == (1, 2, 3)
    assert tuple(schedule.pnp_k_by_step) == SMOLVLA_SCHEDULE_K_BY_STEP == (3, 1, 1)
    assert schedule.n_action_steps == 10
    assert schedule.save_time_uncertainty
    assert schedule.video == "off"


def test_k1_uncertainty_uses_post_perturbation_euler_prediction():
    _pnp_seed_perturb(123)
    latent = torch.zeros((1, 20, 2), dtype=torch.float32)
    probe = run_probe(
        latent, 0.8, lambda value: torch.zeros_like(value), k=1, adim=2)
    actions, rec, lane_recs, z_hat_full = extend_probe_with_final_prediction(
        probe, probe.x_acc, adim=2)

    assert actions.shape == (2, 1, 20, 2)
    assert z_hat_full.shape == latent.shape
    assert rec["u_mean"] > 0
    assert rec["u_prefix_10"] > 0
    assert rec["u_prefix_20"] > 0
    assert len(rec["u_iter"]) == 1
    assert lane_recs[0]["u_mean"] > 0


def test_schedule_worker_notebook_is_thin_and_explicit():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "85_smolvla_libero_pnp_steps123_k311_eval.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_schedule_eval_worker" in source
    assert "'pnp_steps': [1, 2, 3]" in source
    assert "'pnp_k_by_step': [3, 1, 1]" in source
    assert "separate U10/U20/full for Euler steps 1, 2, 3" in source
    assert "'integration_steps': 10" in source
    assert "'n_action_steps': 10" in source
    assert "'video': 'off'" in source
