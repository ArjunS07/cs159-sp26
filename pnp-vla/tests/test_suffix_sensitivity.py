import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.diversity import build_source_suffix_sensitivity_methods
from pnp.experiments import format_probe_diagnostic_table, format_progress_table
from pnp.pnp import _pnp_seed_perturb, run_probe, temporal_decay_weights
from pnp.store import SupabaseStore
from pnp.tap import RolloutTap


ROOT = Path(__file__).parents[1]
ARMS = [Method.SUFFIX_SENSITIVITY, Method.REFINEMENT,
        Method.TAPERED_REFINEMENT]


def test_temporal_decay_weights_are_full_then_linear_then_zero():
    weights = temporal_decay_weights(
        50, 10, 20, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.equal(weights[:11], torch.ones(11))
    assert weights[15].item() == pytest.approx(0.5)
    assert weights[19].item() == pytest.approx(0.1)
    assert torch.equal(weights[20:], torch.zeros(30))


def test_new_options_are_opt_in_validated_and_hashed():
    existing = RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine=True)
    tapered = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True, n_action_steps=10,
        refine_tail_decay_end=20)
    diagnostic = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, n_action_steps=10,
        suffix_probe_samples=4)

    assert "refine_tail_decay_end" not in existing.logical_dict()
    assert "suffix_probe_samples" not in existing.logical_dict()
    assert tapered.logical_dict()["refine_tail_decay_end"] == 20
    assert diagnostic.logical_dict()["suffix_probe_samples"] == 4
    with pytest.raises(ValueError, match="above n_action_steps"):
        RolloutConfig(
            pnp_steps=(3,), refine=True, n_action_steps=10,
            refine_tail_decay_end=10)
    with pytest.raises(ValueError, match="explicit n_action_steps"):
        RolloutConfig(pnp_steps=(3,), suffix_probe_samples=4)


def test_suffix_probe_records_tail_to_prefix_sensitivity():
    _pnp_seed_perturb(123)
    x_t = torch.zeros((1, 50, 7))

    def tail_coupled_vfield(x):
        velocity = torch.zeros_like(x)
        velocity[:, :10] = x[:, 10:].mean(dim=1, keepdim=True)
        return velocity

    result = run_probe(
        x_t, 0.5, tail_coupled_vfield, k=3, adim=7,
        suffix_probe_samples=4, prefix_horizon=10)

    assert result.u_time.shape == (50,)
    assert result.rec["u_time"].shape == (50,)
    assert result.rec["suffix_prefix_predictions"].shape == (4, 1, 10, 7)
    assert result.rec["suffix_prefix_reference"].shape == (1, 10, 7)
    assert result.rec["suffix_prefix_l2_mean"] > 0


def test_tapered_probe_never_updates_far_tail():
    _pnp_seed_perturb(456)
    x_t = torch.zeros((1, 50, 7))
    weights = temporal_decay_weights(
        50, 10, 20, device=x_t.device, dtype=x_t.dtype)
    result = run_probe(
        x_t, 0.5, lambda x: torch.ones_like(x), k=2, adim=7,
        temporal_update_weights=weights)

    assert torch.equal(result.x_acc[:, 20:], x_t[:, 20:])
    assert not torch.equal(result.x_acc[:, :20], x_t[:, :20])


def test_tap_can_gate_on_prefix_uncertainty():
    probe = SimpleNamespace(
        rec={"u_mean": 0.2}, u_time=torch.tensor([0.01] * 10 + [0.2] * 40))
    config = RolloutConfig(
        pnp_steps=(3,), refine=True, refine_threshold=0.05,
        refine_uncertainty_horizon=10)
    tap = RolloutTap(config, SimpleNamespace(), device=None, adim=7)
    tap.begin_chunk()
    x_t = torch.tensor([0.0])
    ctx = SimpleNamespace(step=3, num_steps=10, records=[])

    with patch("pnp.tap.run_probe", return_value=probe), patch(
            "pnp.tap.apply_refine", return_value=torch.tensor([1.0])) as refine:
        assert torch.equal(tap.step(x_t, 0.5, None, ctx), x_t)
    refine.assert_not_called()


def test_suffix_pilot_has_three_unique_paired_configs():
    methods = build_source_suffix_sensitivity_methods()

    assert [method for method, _ in methods] == ARMS
    assert [config.n_action_steps for _, config in methods] == [10] * 3
    assert [config.pnp_steps for _, config in methods] == [(3, 4)] * 3
    assert [config.pnp_k for _, config in methods] == [5] * 3
    assert [config.refine for _, config in methods] == [False, True, True]
    assert [config.suffix_probe_samples for _, config in methods] == [4, 0, 0]
    assert [config.refine_tail_decay_end for _, config in methods] == [None, None, 20]
    assert [config.save_time_uncertainty for _, config in methods] == [True] * 3
    assert [config.save_ahats for _, config in methods] == [False] * 3
    hashes = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(method, config))
        for method, config in methods}
    assert len(hashes) == 3


def test_progress_tables_have_one_column_per_arm_and_diagnostics():
    tally = {
        ("libero_goal_swap", method): [5, successes]
        for method, successes in zip(ARMS, (1, 2, 3))}
    table = format_progress_table(tally, ARMS, historical_sr=False)
    assert table.splitlines()[1].count("diagnostic base") == 1
    assert table.splitlines()[1].count("tapered refine") == 1
    for value in ("20% (1/5)", "40% (2/5)", "60% (3/5)"):
        assert value in table

    diagnostics = format_probe_diagnostic_table({
        (Method.SUFFIX_SENSITIVITY, "u_first10"): [0.06, 2],
        (Method.SUFFIX_SENSITIVITY, "suffix_to_prefix_l2"): [0.4, 2],
    }, ARMS)
    assert "0.03000" in diagnostics
    assert "0.20000" in diagnostics
    assert "NEW rollouts" in diagnostics


def test_time_uncertainty_sink_does_not_require_full_ahat_stack():
    store = SupabaseStore.__new__(SupabaseStore)
    store.log_episode = Mock(return_value="rid")
    recorder = {"chunks": [{"chunk_idx": 0, "steps": [{
        "step": 3, "u_time": [0.1, 0.2],
        "suffix_prefix_predictions": [[[1.0]]],
        "suffix_prefix_reference": [[0.0]],
        "a_hats": [[[9.0]]],
    }]}]}
    result = {
        "recorder_episode": recorder, "chunk_noise_seeds": [], "n_chunks": 1,
        "episode_seed": 1, "success": False, "n_steps": 10, "elapsed_s": 1.0,
        "status": "completed", "error_msg": None, "nan_action_count": 0,
        "n_vf_evals": 1, "instability": {},
    }
    ep = {"suite": "suite", "task_idx": 0}
    config = RolloutConfig(
        pnp_steps=(3,), n_action_steps=10, save_time_uncertainty=True)

    with patch.object(store, "_recorder_to_rows", return_value=([], [], {})), patch.object(
            store, "_denorm", return_value={}):
        store.log_result("rid", ep, Method.SUFFIX_SENSITIVITY, config, result)

    blobs = store.log_episode.call_args.kwargs["blobs"]
    assert set(blobs["ahats"]) == {
        "c0_s3_u_time", "c0_s3_suffix_prefix_predictions",
        "c0_s3_suffix_prefix_reference"}
    assert "c0_s3" not in blobs["ahats"]


def test_suffix_workers_are_four_fixed_shards():
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"35_source_suffix_sensitivity_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_suffix_sensitivity_worker(" in source
        assert "EPISODE_INDICES = (10, 11)" in source
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "source_model_revision=SOURCE_MODEL_REVISION" in source
