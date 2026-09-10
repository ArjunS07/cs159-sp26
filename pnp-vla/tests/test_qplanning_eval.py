import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from pnp import experiments, libero_pro
from pnp.config import Method, RolloutConfig
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.qplanning_critic.inference import (
    load_qplanning_scorer, q_weighted_average)
from pnp.qplanning_critic.model import QPlanningCritic
from pnp.qplanning_eval_experiment import (
    QPLANNING_EVAL_ACTION_STEPS, QPLANNING_EVAL_DENOISE_STEPS,
    QPLANNING_EVAL_IDENTITIES, QPLANNING_EVAL_NUM_CANDIDATES,
    QPLANNING_EVAL_NUM_ELITES, QPLANNING_EVAL_TEMPERATURE,
    QPLANNING_HELDOUT_IDENTITIES, QPLANNING_HELDOUT_SHARDS,
    build_qplanning_eval_method, build_qplanning_stock_method)
from pnp.store import SupabaseStore


ROOT = Path(__file__).parents[1]


def _small_checkpoint(path: Path, horizon: int = 10):
    config = QPlanningModelConfig(
        action_horizon=horizon, width=16, n_layers=1, n_heads=4,
        ffn_width=32, dropout=0, n_bins=11)
    model = QPlanningCritic(
        prefix_dim=8, robot_dim=3, proprio_dim=2, config=config)
    torch.save({
        "format": "qplanning_critic_v1", "update": 8000,
        "snapshot_id": "pcpcds-test", "cache_digest": "digest",
        "source_policy": {"repo_id": "repo", "revision": "revision"},
        "architecture": model.architecture_config(),
        "model": model.state_dict(),
    }, path)
    return model


def test_checkpoint_loader_freezes_online_model_and_binds_provenance(tmp_path):
    path = tmp_path / "checkpoint_step_008000.pt"
    _small_checkpoint(path)
    scorer = load_qplanning_scorer(
        path, device="cpu", expected_horizon=10,
        expected_source_revision="revision")
    assert scorer.horizon == 10
    assert scorer.update == 8000
    assert scorer.source_policy == {"repo_id": "repo", "revision": "revision"}
    assert scorer.checkpoint_id.startswith("pcpcds-test:q10:step8000:")
    assert not any(parameter.requires_grad for parameter in scorer.model.parameters())
    with pytest.raises(ValueError, match="Q50"):
        load_qplanning_scorer(path, device="cpu", expected_horizon=50)


def test_q_weighted_average_uses_only_top_elites():
    candidates = torch.tensor([[[0.0]], [[1.0]], [[3.0]]])
    q_values = torch.tensor([0.0, 1.0, 2.0])
    blended, indices, weights = q_weighted_average(
        candidates, q_values, n_elites=2, temperature=1.0)
    assert indices.tolist() == [2, 1]
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    expected = 3 * torch.softmax(torch.tensor([2.0, 1.0]), 0)[0] + +        torch.softmax(torch.tensor([2.0, 1.0]), 0)[1]
    assert torch.allclose(blended.squeeze(), expected)


def test_eval_method_has_fixed_paper_style_search_and_ten_step_execution():
    scorer = SimpleNamespace(horizon=50, checkpoint_id="critic")
    method, config = build_qplanning_eval_method(
        50, "repo@revision", scorer, candidate_batch_size=8)
    assert method == Method.QPLANNING_Q50
    assert QPLANNING_EVAL_IDENTITIES == 220
    assert config.num_samples == QPLANNING_EVAL_NUM_CANDIDATES == 64
    assert config.qplanning_n_elites == QPLANNING_EVAL_NUM_ELITES == 16
    assert config.qplanning_temperature == QPLANNING_EVAL_TEMPERATURE == 1.0
    assert config.num_inference_steps == QPLANNING_EVAL_DENOISE_STEPS == 3
    assert config.n_action_steps == QPLANNING_EVAL_ACTION_STEPS == 10
    assert config.candidate_seed_scheme == "stock_slot0_v1"
    assert config.video == "off" and not config.save_observations
    assert not config.save_generated_chunks
    logical = config.logical_dict()
    assert logical["qplanning_ckpt_id"] == "critic"
    assert "qplanning_scorer" not in logical
    assert SupabaseStore._denorm(method, config)["pnp_step_indices"] is None
    with pytest.raises(ValueError, match="Q10"):
        build_qplanning_eval_method(10, "repo@revision", scorer)


def test_heldout_stock_method_is_exact_ordinary_ten_step_policy():
    method, config = build_qplanning_stock_method("repo@revision")
    assert method == Method.VANILLA
    assert config.num_samples is None
    assert config.num_inference_steps == 10
    assert config.n_action_steps == 10
    assert config.policy_source_id == "repo@revision"
    assert config.video == "off"
    assert not config.save_observations
    assert not config.save_generated_chunks
    assert config.save_trajectory
    assert QPLANNING_HELDOUT_IDENTITIES == 160
    assert QPLANNING_HELDOUT_SHARDS == 4


def test_explicit_heldout_loader_can_include_historical_zero_sr_suite():
    suite = "libero_object_temp_x0.3"
    assert suite in libero_pro.ZERO_SR_PRO_SUITES
    episode = {
        "suite": suite, "task_idx": 0, "ep_idx": 0,
        "init_state_hash": "state", "expanded_member": True, "max_steps": 100}
    patches = (
        patch.object(libero_pro, "clone_libero_pro", return_value="repo"),
        patch.object(libero_pro, "install_assets"),
        patch.object(libero_pro, "apply_env_patches"),
        patch.object(libero_pro, "patch_torch_load"),
        patch.object(libero_pro, "reload_benchmark", return_value={}),
        patch.object(libero_pro, "build_libero_pro_episodes",
                     return_value=[episode]),
    )
    for active in patches:
        active.start()
    try:
        with pytest.raises(AssertionError, match="0%-SR suites"):
            experiments._prepare_libero_pro_expanded_episodes(
                suites=[suite], episode_idxs=(0,))
        rows = experiments._prepare_libero_pro_expanded_episodes(
            suites=[suite], episode_idxs=(0,), allow_zero_sr_suites=True)
        assert rows == [episode]
    finally:
        for active in reversed(patches):
            active.stop()


def test_store_persists_compact_qplanning_boundary_telemetry():
    store = SupabaseStore.__new__(SupabaseStore)
    store.log_episode = Mock(return_value="rid")
    selection = {
        "q_values": list(map(float, range(64))), "best_index": 63,
        "elite_indices": list(range(63, 47, -1)), "elite_weights": [1 / 16] * 16,
        "q_mean": 31.5, "q_std": 1.0, "q_min": 0.0, "q_max": 63.0,
        "candidate_noise_seeds": list(range(64)),
        "n_candidates": 64, "n_elites": 16, "temperature": 1.0,
        "denoise_steps": 3, "candidate_batch_size": 8,
        "candidate_equivalent_vf_evals": 192,
        "first10_diversity": {"mean_pairwise_rms": .1},
        "full50_diversity": {"mean_pairwise_rms": .2},
        "inference_ms": 10.0, "n_vf_evals": 192,
    }
    result = {
        "recorder_episode": None, "chunk_noise_seeds": [], "n_chunks": 1,
        "episode_seed": 1, "success": True, "n_steps": 10, "elapsed_s": 1.0,
        "status": "completed", "error_msg": None, "nan_action_count": 0,
        "n_vf_evals": 192, "inference_ms_total": 10.0, "instability": {},
        "qplanning_selections": [selection],
    }
    scorer = SimpleNamespace(horizon=10, checkpoint_id="critic")
    method, config = build_qplanning_eval_method(10, "repo@revision", scorer)
    with patch.object(store, "_recorder_to_rows", return_value=([], [], {})):
        store.log_result(
            "rid", {"suite": "suite", "task_idx": 0}, method, config, result)
    row = store.log_episode.call_args.args[0]
    assert row["ms_chosen_idx"] == 63
    telemetry = row["ms_candidate_u"]["qplanning"]
    assert len(telemetry["q_values"][0]) == 64
    assert telemetry["denoise_steps"] == [3]


def test_two_generated_eval_workers_are_clean_single_arm_launchers():
    for number, horizon in ((66, 10), (67, 50)):
        path = ROOT / "notebooks" / "workers" / (
            f"{number}_eval_qplanning_q{horizon}_pro220.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert f"HORIZON = {horizon}" in source
        assert "EPISODE_LIMIT = None" in source
        assert "CANDIDATE_BATCH_SIZE = 8" in source
        assert "run_qplanning_eval_worker(" in source
        assert "validate_qplanning_eval_sentinel(" in source
        assert "checkpoint_step_008000.pt" in source
        assert "video" in source.lower()
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                ast.parse("".join(cell["source"]), filename=f"{path.name}:cell{index}")


def test_four_generated_heldout_workers_are_fixed_three_arm_shards():
    for shard_index in range(4):
        path = ROOT / "notebooks" / "workers" / (
            f"68_eval_qplanning_heldout160_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = chr(10).join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "SHARD_COUNT = 4" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "EPISODE_LIMIT = None" in source
        assert "CANDIDATE_BATCH_SIZE = 8" in source
        assert "run_qplanning_heldout_worker(" in source
        assert "validate_qplanning_heldout_sentinel(" in source
        assert "Q10_CHECKPOINT_PATH" in source
        assert "Q50_CHECKPOINT_PATH" in source
        assert "checkpoint_step_008000.pt" in source
        assert "online learning" in source.lower()
        assert "video" in source.lower() and "frames" in source.lower()
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                ast.parse("".join(cell["source"]), filename=f"{path.name}:cell{index}")
