import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.diversity import build_source_fractional_refinement_methods
from pnp.experiments import format_progress_table
from pnp.pnp import apply_fractional_refine
from pnp.store import SupabaseStore
from pnp.tap import RolloutTap


ROOT = Path(__file__).parents[1]
ARMS = [Method.UNCERTAINTY, Method.REFINEMENT,
        Method.FRACTIONAL_M2, Method.FRACTIONAL_M4]


def test_fractional_horizon_is_opt_in_validated_and_hashed():
    existing = RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine=True)
    fractional = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True, refine_horizon_m=2)

    assert "refine_horizon_m" not in existing.logical_dict()
    assert fractional.logical_dict()["refine_horizon_m"] == 2
    assert existing.logical_dict() != fractional.logical_dict()
    with pytest.raises(ValueError, match="requires refine=True"):
        RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine_horizon_m=2)
    with pytest.raises(ValueError, match="positive integer"):
        RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine=True, refine_horizon_m=0)


def test_fractional_refine_scales_by_m_over_remaining_horizon():
    x_t = torch.zeros(1)
    # At step 3 of 10, seven steps remain. Full P&P at value 7 becomes value 2 for m=2.
    pr = SimpleNamespace(x_acc=torch.tensor([7.0]))
    result = apply_fractional_refine(
        pr, x_t, False, horizon_m=2, num_steps=10, step=3)
    assert torch.equal(result, torch.tensor([2.0]))

    # At step 4, m=4 applies 4/6 of the full update.
    pr = SimpleNamespace(x_acc=torch.tensor([6.0]))
    result = apply_fractional_refine(
        pr, x_t, False, horizon_m=4, num_steps=10, step=4)
    assert torch.equal(result, torch.tensor([4.0]))

    with pytest.raises(ValueError, match="exceeds remaining sampler horizon"):
        apply_fractional_refine(
            pr, x_t, False, horizon_m=7, num_steps=10, step=4)


def test_tap_routes_fractional_config_without_changing_full_refine():
    recorder = SimpleNamespace()
    ctx = SimpleNamespace(step=3, num_steps=10, records=[])
    probe = SimpleNamespace(rec={"u_mean": 0.04})
    x_t = torch.tensor([0.0])
    fractional_result = torch.tensor([2.0])
    full_result = torch.tensor([7.0])

    fractional = RolloutTap(RolloutConfig(
        pnp_steps=(3,), pnp_k=5, refine=True, refine_horizon_m=2),
        recorder, device=None, adim=7)
    fractional.begin_chunk()
    with patch("pnp.tap.run_probe", return_value=probe), patch(
            "pnp.tap.apply_fractional_refine",
            return_value=fractional_result) as apply_fractional, patch(
                "pnp.tap.apply_refine", return_value=full_result) as apply_full:
        assert torch.equal(fractional.step(x_t, .7, None, ctx), fractional_result)
    apply_fractional.assert_called_once_with(
        probe, x_t, False, horizon_m=2, num_steps=10, step=3)
    apply_full.assert_not_called()

    full = RolloutTap(RolloutConfig(
        pnp_steps=(3,), pnp_k=5, refine=True), recorder, device=None, adim=7)
    full.begin_chunk()
    with patch("pnp.tap.run_probe", return_value=probe), patch(
            "pnp.tap.apply_refine", return_value=full_result) as apply_full:
        assert torch.equal(full.step(x_t, .7, None, ctx), full_result)
    apply_full.assert_called_once_with(probe, False)


def test_fractional_pilot_has_four_unique_paired_configs():
    methods = build_source_fractional_refinement_methods()
    assert [method for method, _ in methods] == ARMS
    assert [config.n_action_steps for _, config in methods] == [10] * 4
    assert [config.pnp_steps for _, config in methods] == [(3, 4)] * 4
    assert [config.pnp_k for _, config in methods] == [5] * 4
    assert [config.refine_horizon_m for _, config in methods] == [None, None, 2, 4]
    assert [config.refine for _, config in methods] == [False, True, True, True]
    hashes = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(method, config))
        for method, config in methods}
    assert len(hashes) == 4


def test_fractional_progress_table_has_one_balanced_column_per_arm():
    tally = {
        ("libero_goal_swap", method): [5, successes]
        for method, successes in zip(ARMS, (1, 2, 3, 4))}
    table = format_progress_table(tally, ARMS, historical_sr=False)
    header = table.splitlines()[1]
    assert header.count("observed") == 1
    assert header.count("refine") == 1
    assert header.count("fractional m=2") == 1
    assert header.count("fractional m=4") == 1
    for value in ("20% (1/5)", "40% (2/5)", "60% (3/5)", "80% (4/5)"):
        assert value in table


def test_fractional_workers_are_four_fixed_shards():
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"34_source_fractional_refinement_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_fractional_refinement_worker(" in source
        assert "EPISODE_INDICES = (10,)" in source
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "n_action_steps" not in source  # fixed and validated inside the package driver
        assert "source_model_revision=SOURCE_MODEL_REVISION" in source
