import ast
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pnp.config import Method, RolloutConfig
from pnp.experiments import format_progress_table
from pnp.five_step_diversity_experiment import (
    FIVE_STEP_DIVERSITY_ACTION_STEPS,
    FIVE_STEP_DIVERSITY_EPISODE_INDICES,
    FIVE_STEP_DIVERSITY_EXPERIMENT,
    FIVE_STEP_DIVERSITY_NUM_INFERENCE_STEPS,
    FIVE_STEP_DIVERSITY_PROBE_STEPS,
    FIVE_STEP_DIVERSITY_PROBE_TIMES,
    build_five_step_diversity_methods,
    identity_manifest_hash,
    identity_manifest_payload,
)
from pnp.store import SupabaseStore


ROOT = Path(__file__).parents[1]


def test_three_arm_design_is_fixed_to_five_step_decode_and_ten_step_execution():
    methods = build_five_step_diversity_methods("source@revision")
    assert [method for method, _ in methods] == [
        Method.FIVE_STEP_SINGLE_QUERY,
        Method.FIVE_STEP_LOWEST_U20,
        Method.FIVE_STEP_LOWEST_U20_REFINE,
    ]
    assert FIVE_STEP_DIVERSITY_EXPERIMENT == "pi05-five-step-diversity-pro220-v1"
    assert FIVE_STEP_DIVERSITY_EPISODE_INDICES == (10, 11)
    assert FIVE_STEP_DIVERSITY_PROBE_STEPS == (2, 3)
    assert FIVE_STEP_DIVERSITY_PROBE_TIMES == (0.6, 0.4)
    assert all(config.num_inference_steps == FIVE_STEP_DIVERSITY_NUM_INFERENCE_STEPS
               for _, config in methods)
    assert all(config.n_action_steps == FIVE_STEP_DIVERSITY_ACTION_STEPS
               for _, config in methods)
    assert all(config.policy_source_id == "source@revision" for _, config in methods)
    assert methods[0][1].num_samples is None
    assert methods[0][1].pnp_steps == FIVE_STEP_DIVERSITY_PROBE_STEPS
    assert methods[1][1].num_samples == 3
    assert methods[1][1].selection_uncertainty_horizon == 20
    assert methods[1][1].candidate_seed_scheme == "stock_slot0_v1"
    assert methods[1][1].multi_sample_refine_selected is False
    assert methods[2][1].multi_sample_refine_selected is True
    assert methods[2][1].refine is False
    assert methods[0][1].save_time_uncertainty is True
    assert all(config.save_generated_chunks for _, config in methods)

    store = SupabaseStore.__new__(SupabaseStore)
    hashes = {
        store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    assert len(hashes) == 3


def test_new_candidate_behavior_is_hashed_without_rewriting_legacy_configs():
    legacy = RolloutConfig(num_samples=3, selection_uncertainty_horizon=20)
    corrected = RolloutConfig(
        num_samples=3, selection_uncertainty_horizon=20,
        candidate_seed_scheme="stock_slot0_v1")
    refined = RolloutConfig(
        num_samples=3, selection_uncertainty_horizon=20,
        candidate_seed_scheme="stock_slot0_v1", multi_sample_refine_selected=True)
    assert "candidate_seed_scheme" not in legacy.logical_dict()
    assert "multi_sample_refine_selected" not in legacy.logical_dict()
    assert corrected.logical_dict()["candidate_seed_scheme"] == "stock_slot0_v1"
    assert refined.logical_dict()["multi_sample_refine_selected"] is True
    assert len({SupabaseStore.config_hash(config.logical_dict())
                for config in (legacy, corrected, refined)}) == 3
    with pytest.raises(ValueError, match="requires num_samples"):
        RolloutConfig(candidate_seed_scheme="stock_slot0_v1")
    with pytest.raises(ValueError, match="selection_uncertainty_horizon"):
        RolloutConfig(num_samples=3, multi_sample_refine_selected=True)


def test_identity_manifest_is_exact_canonical_and_order_independent():
    episodes = [
        {"suite": "b", "task_idx": 1, "ep_idx": 11, "init_state_hash": "h2"},
        {"suite": "a", "task_idx": 0, "ep_idx": 10, "init_state_hash": "h1"},
    ]
    assert identity_manifest_payload(episodes) == [
        {"suite": "a", "task_idx": 0, "episode_idx": 10, "init_state_hash": "h1"},
        {"suite": "b", "task_idx": 1, "episode_idx": 11, "init_state_hash": "h2"},
    ]
    assert identity_manifest_hash(episodes) == identity_manifest_hash(list(reversed(episodes)))


def test_progress_table_can_show_per_arm_overall_and_exact_reference():
    tally = {
        ("suite_a", Method.FIVE_STEP_SINGLE_QUERY): [2, 1],
        ("suite_a", Method.FIVE_STEP_LOWEST_U20): [2, 2],
        ("suite_a", Method.FIVE_STEP_LOWEST_U20_REFINE): [2, 0],
        ("suite_b", Method.FIVE_STEP_SINGLE_QUERY): [2, 1],
        ("suite_b", Method.FIVE_STEP_LOWEST_U20): [2, 1],
        ("suite_b", Method.FIVE_STEP_LOWEST_U20_REFINE): [2, 2],
    }
    table = format_progress_table(
        tally,
        [Method.FIVE_STEP_SINGLE_QUERY, Method.FIVE_STEP_LOWEST_U20,
         Method.FIVE_STEP_LOWEST_U20_REFINE],
        historical_sr={"10-step x1": {"suite_a": .25, "suite_b": .75}},
        include_overall=True,
        count_label="n = complete identities")
    assert table.splitlines()[0] == "n = complete identities"
    assert "10-step x1" in table
    overall = next(line for line in table.splitlines() if line.startswith("OVERALL"))
    assert "50% (2/4)" in overall
    assert "75% (3/4)" in overall


def test_store_preserves_five_step_boundary_telemetry():
    store = SupabaseStore.__new__(SupabaseStore)
    store.log_episode = Mock(return_value="rid")
    selection = {
        "chosen": 1, "cand_u": [.3, .2, .4], "u_spread": .2,
        "candidate_noise_seeds": [11, 12, 13],
        "selected_noise_seed": 12, "selected_perturb_seed": 12,
        "candidate_profiles": [{"u10": .1, "u20": .3, "u_full": .2}] * 3,
        "selection_uncertainty_horizon": 20,
        "action_disagreement": {"n_candidate_pairs": 3},
        "executed_prefix_disagreement": {"actions_compared": 10},
        "inference_ms": 12.5, "n_vf_evals": 60,
        "selected_refinement": {"pre_u": .2, "refined_path_u": .1,
                                "delta_u": -.1, "lowered_u": True},
    }
    result = {
        "recorder_episode": None, "chunk_noise_seeds": [], "n_chunks": 1,
        "episode_seed": 1, "success": False, "n_steps": 10, "elapsed_s": 1.0,
        "status": "completed", "error_msg": None, "nan_action_count": 0,
        "n_vf_evals": 60, "inference_ms_total": 12.5, "instability": {},
        "ms_selections": [selection],
    }
    config = build_five_step_diversity_methods("source@revision")[2][1]
    with patch.object(store, "_recorder_to_rows", return_value=([], [], {})), patch.object(
            store, "_denorm", return_value={}):
        store.log_result(
            "rid", {"suite": "suite", "task_idx": 0},
            Method.FIVE_STEP_LOWEST_U20_REFINE, config, result)
    telemetry = store.log_episode.call_args.args[0]["ms_candidate_u"]
    for field in ("candidate_noise_seeds", "selected_noise_seed",
                  "selected_perturb_seed", "u_spread", "action_disagreement",
                  "executed_prefix_disagreement",
                  "candidate_profiles", "inference_ms", "n_vf_evals",
                  "selected_refinement"):
        assert field in telemetry
    assert telemetry["selected_refinement"][0]["lowered_u"] is True


def test_two_generated_workers_are_clean_fixed_110_identity_shards():
    for shard_index in range(2):
        path = ROOT / "notebooks" / "workers" / (
            f"60_five_step_diversity_pro220_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["colab"]["name"] == path.name
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        for required in (
                "run_five_step_diversity_worker(",
                "validate_five_step_diversity_sentinel(",
                "EPISODE_INDICES = FIVE_STEP_DIVERSITY_EPISODE_INDICES",
                "PROBE_STEPS = FIVE_STEP_DIVERSITY_PROBE_STEPS",
                "SHARD_COUNT = FIVE_STEP_DIVERSITY_SHARD_COUNT",
                f"SHARD_INDEX = {shard_index}",
                "identities_in_full_shard\": 110",
                "rollouts_in_full_shard\": 330",
                "absolute_uncertainty_threshold\": None"):
            assert required in source
        assert "0.03" not in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []
                ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")


def test_collection_retries_errors_without_counting_them_as_completed(capsys):
    from pnp.experiments import _run_collection

    config = build_five_step_diversity_methods("source@revision")[0][1]
    method = Method.FIVE_STEP_SINGLE_QUERY
    episode = {
        "suite": "libero_test", "task_idx": 0, "ep_idx": 10,
        "init_state_hash": "h", "bddl_path": "test.bddl"}
    store = Mock()
    store.existing_keys.return_value = set()
    store.rollout_id.return_value = "rid"
    store.iter_todo.return_value = [(episode, method, config, "rid")]
    result = {"status": "error", "error_msg": "test failure", "success": False}
    with (patch("pnp.libero_env.make_env", return_value=Mock()),
          patch("pnp.rollout.run_episode_batch", return_value=[result]),
          patch("pnp.experiments.format_progress_table") as table):
        _run_collection(
            store=store, policy=None, preprocess=None, postprocess=None, device="cpu",
            experiment="test", episodes=[episode], methods=[(method, config)],
            cohort="test", shard_count=2, shard_index=0, rollout_batch_size=1,
            report_every=0, report_every_identities=25, resume_completed_only=True)
    store.existing_keys.assert_called_once_with("test", status="completed")
    store.log_result.assert_called_once()
    table.assert_not_called()
    assert "0 identities complete in this shard" in capsys.readouterr().out


@pytest.mark.parametrize("run_id", [None, "", " "])
def test_historical_baseline_rejects_missing_run_provenance(run_id):
    from pnp.five_step_diversity_experiment import _load_verified_historical_baseline

    episode = {"suite": "suite", "task_idx": 0, "episode_idx": 10,
               "init_state_hash": "h"}
    store = SupabaseStore.__new__(SupabaseStore)
    store.fetch_all = Mock(return_value=[
        {**episode, "run_id": run_id, "status": "completed", "success": True}])
    with pytest.raises(ValueError, match="without run_id provenance"):
        _load_verified_historical_baseline(
            store, [episode], source_repo="source", source_revision="revision")
