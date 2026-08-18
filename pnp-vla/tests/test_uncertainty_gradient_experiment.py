import json
from pathlib import Path

import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.pnp import _pnp_seed_perturb
from pnp.uncertainty_critic_v2 import direct_latent_uncertainty_update
from pnp.uncertainty_gradient_experiment import (
    DIRECT_U20_GRADIENT_EXPERIMENT,
    build_direct_u20_gradient_methods,
)


def test_gradient_rollout_config_is_validated_and_hashed():
    config = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, n_action_steps=10,
        uncertainty_gradient_mode="descent",
        uncertainty_gradient_step_size=.01,
        uncertainty_gradient_horizon=20)
    logical = config.logical_dict()
    assert logical["uncertainty_gradient_mode"] == "descent"
    assert logical["uncertainty_gradient_step_size"] == .01
    assert logical["uncertainty_gradient_horizon"] == 20
    with pytest.raises(ValueError, match="requires a P&P probe"):
        RolloutConfig(
            uncertainty_gradient_mode="descent",
            uncertainty_gradient_step_size=.01)
    with pytest.raises(ValueError, match="at most one action"):
        RolloutConfig(
            pnp_steps=(3,), refine=True,
            uncertainty_gradient_mode="descent",
            uncertainty_gradient_step_size=.01)


def test_direct_gradient_lowers_exact_common_noise_u20_and_matches_rms():
    generator = torch.Generator().manual_seed(159)
    latent = torch.randn((1, 20, 8), generator=generator)

    def nonlinear_vfield(value):
        return .35 * torch.tanh(value) + .04 * value.square()

    _pnp_seed_perturb(2026)
    updated, _, telemetry = direct_latent_uncertainty_update(
        latent, .7, nonlinear_vfield, k=5, horizon=20,
        step_size=1e-3, mode="descent", checkpoint_vfield=False)
    assert telemetry["post_u20"] < telemetry["pre_u20"]
    assert telemetry["delta_u20"] < 0
    assert telemetry["update_rms"] == pytest.approx(1e-3, rel=1e-4)
    assert not torch.equal(updated, latent)


def test_three_arm_design_and_worker_notebooks():
    methods = build_direct_u20_gradient_methods()
    assert [method for method, _ in methods] == [
        Method.UNCERTAINTY, Method.U20_GRADIENT, Method.LATENT_RANDOM_CONTROL]
    assert all(config.n_action_steps == 10 for _, config in methods)
    assert methods[1][1].uncertainty_gradient_mode == "descent"
    assert methods[2][1].uncertainty_gradient_mode == "random"

    workers = sorted(Path("notebooks/workers").glob(
        "48_direct_u20_gradient_pro220_worker_*.ipynb"))
    assert len(workers) == 4
    for index, path in enumerate(workers):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][2]["source"])
        assert f"SHARD_INDEX = {index}" in source
        assert "SHARD_COUNT = 4" in source
        assert "EPISODE_INDICES = (10, 11)" in source
        assert "run_direct_u20_gradient_worker(" in source
        assert DIRECT_U20_GRADIENT_EXPERIMENT == "pi05-direct-u20-gradient-pro220-v1"
