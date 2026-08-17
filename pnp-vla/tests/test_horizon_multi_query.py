import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.diversity import (SOURCE_HORIZON_MULTI_QUERY_EXPERIMENT,
                           run_source_horizon_multi_query_worker)
from pnp.experiments import format_probe_diagnostic_table
from pnp.pnp import _pnp_seed_perturb, multi_policy_select, run_probe
from pnp.store import SupabaseStore


ROOT = Path(__file__).parents[1]


def test_selection_horizon_is_validated_and_behavior_hashed():
    full = RolloutConfig(num_samples=2, ms_probe_steps=(3, 4), n_action_steps=10)
    prefix = RolloutConfig(
        num_samples=2, ms_probe_steps=(3, 4), selection_uncertainty_horizon=20,
        n_action_steps=10)
    assert "selection_uncertainty_horizon" not in full.logical_dict()
    assert prefix.logical_dict()["selection_uncertainty_horizon"] == 20
    assert SupabaseStore.config_hash(full.logical_dict()) != SupabaseStore.config_hash(
        prefix.logical_dict())
    with pytest.raises(ValueError, match="requires num_samples"):
        RolloutConfig(selection_uncertainty_horizon=20)
    with pytest.raises(ValueError, match="positive integer"):
        RolloutConfig(num_samples=2, selection_uncertainty_horizon=0)


def test_probe_keeps_consecutive_disagreement_by_action_position():
    _pnp_seed_perturb(159)
    result = run_probe(
        torch.zeros((1, 50, 7)), 0.5,
        lambda value: value * 0.2 + 0.1,
        k=5, adim=7)
    assert result.rec["u_iter_time"].shape == (4, 50)
    assert result.rec["u_time"].shape == (50,)


def test_multi_policy_selector_uses_u20_and_returns_all_horizon_details():
    action0 = torch.zeros((1, 50, 7))
    action1 = torch.ones((1, 50, 7))
    profiles = [
        {"u10": .10, "u20": .30, "u_full": .05,
         "contraction10": .01, "contraction20": .02, "contraction_full": .03},
        {"u10": .20, "u20": .10, "u_full": .40,
         "contraction10": .04, "contraction20": .05, "contraction_full": .06},
    ]
    measured = [(action0, .30, profiles[0]), (action1, .10, profiles[1])]
    with patch("pnp.sampler.measure_chunk_uncertainty", side_effect=measured) as measure:
        selected = multi_policy_select(
            [(Mock(), {}), (Mock(), {})], [torch.zeros(1), torch.ones(1)],
            (3, 4), num_iterations=5, perturb_seeds=(1, 2),
            uncertainty_horizon=20, return_details=True)
    action, chosen, scores, actions, details = selected
    assert chosen == 1
    assert torch.equal(action, action1)
    assert scores == [.30, .10]
    assert len(actions) == 2
    assert torch.equal(actions[0], action0)
    assert torch.equal(actions[1], action1)
    assert details == profiles
    assert all(call.kwargs["uncertainty_horizon"] == 20 for call in measure.call_args_list)


def test_time_uncertainty_artifact_includes_iter_time_without_ahats():
    store = SupabaseStore.__new__(SupabaseStore)
    store.log_episode = Mock(return_value="rid")
    recorder = {"chunks": [{"chunk_idx": 0, "steps": [{
        "step": 3, "u_time": np.ones(50), "u_iter_time": np.ones((4, 50)),
    }]}]}
    result = {
        "recorder_episode": recorder, "chunk_noise_seeds": [], "n_chunks": 1,
        "episode_seed": 1, "success": False, "n_steps": 10, "elapsed_s": 1.0,
        "status": "completed", "error_msg": None, "nan_action_count": 0,
        "n_vf_evals": 1, "instability": {},
    }
    config = RolloutConfig(
        pnp_steps=(3,), pnp_k=5, n_action_steps=10, save_time_uncertainty=True)
    with patch.object(store, "_recorder_to_rows", return_value=([], [], {})), patch.object(
            store, "_denorm", return_value={}):
        store.log_result("rid", {"suite": "suite", "task_idx": 0},
                         Method.UNCERTAINTY, config, result)
    artifact = store.log_episode.call_args.kwargs["blobs"]["ahats"]
    assert set(artifact) == {"c0_s3_u_time", "c0_s3_u_iter_time"}


def test_probe_progress_table_prints_all_horizons_and_contractions():
    table = format_probe_diagnostic_table({
        (Method.UNCERTAINTY, "u_first10"): [.1, 2],
        (Method.UNCERTAINTY, "u_first20"): [.2, 2],
        (Method.UNCERTAINTY, "contraction_first10"): [.03, 2],
        (Method.UNCERTAINTY, "contraction_first20"): [.04, 2],
    }, [Method.UNCERTAINTY, Method.CHUNK_SOURCE_MULTI_QUERY])
    for label in ("U first10", "U first20", "C first10", "C first20", "C full"):
        assert label in table
    assert "0.05000" in table
    assert "0.10000" in table
    assert "0.01500" in table
    assert "0.02000" in table


def test_new_worker_notebooks_are_four_fixed_full_cohort_shards():
    assert (run_source_horizon_multi_query_worker.__kwdefaults__["experiment"] ==
            SOURCE_HORIZON_MULTI_QUERY_EXPERIMENT)
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"40_source_horizon_multi_query_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_horizon_multi_query_worker(" in source
        assert "EPISODES_PER_TASK = 10" in source
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "N_QUERIES = 2" in source
        assert "SELECTION_HORIZON = 20" in source
        assert "source_model_revision=SOURCE_MODEL_REVISION" in source


def test_diagnostic_only_notebooks_resume_the_same_four_shards():
    assert run_source_horizon_multi_query_worker.__kwdefaults__["include_multi_query"] is True
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"41_source_horizon_diagnostic_only_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_horizon_multi_query_worker(" in source
        assert "EPISODES_PER_TASK = 10" in source
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "include_multi_query=False" in source
        assert "EXPERIMENT = SOURCE_HORIZON_MULTI_QUERY_EXPERIMENT" in source
        assert "source_model_revision=SOURCE_MODEL_REVISION" in source
