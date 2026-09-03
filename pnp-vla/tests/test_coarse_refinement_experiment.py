import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from pnp.config import ALL_METHODS, Method
from pnp.coarse_refinement_experiment import (
    COARSE_REFINEMENT_EXPERIMENT, _load_exact_reference,
    build_coarse_refinement_methods, coarse_refinement_arm_settings,
    load_coarse_refinement_references, run_coarse_refinement_worker)
from pnp.experiments import _run_collection, format_matched_progress_table
from pnp.five_step_diversity_experiment import _identity_key, build_five_step_diversity_methods
from pnp.pnp import PnPRecorder, _pnp_seed_perturb
from pnp.rollout import chunk_noise_seed, episode_seed, run_episode
from pnp.sampler import _SamplerState, _sample_actions_hooked
from pnp.store import SupabaseStore
from pnp.tap import RolloutTap


ROOT = Path(__file__).parents[1]


def test_fixed_three_arm_design_and_unique_hashes():
    methods = build_coarse_refinement_methods("source@revision")
    assert [name for name, _ in methods] == [Method.FIVE_STEP_SINGLE_REFINE,
                                            Method.THREE_STEP_SINGLE_REFINE,
                                            Method.THREE_STEP_SINGLE_QUERY]
    for (name, cfg), steps, probes, refine in zip(
            methods, (5, 3, 3), ((2, 3), (2,), (2,)), (True, True, False)):
        assert name in ALL_METHODS
        assert cfg.num_inference_steps == steps
        assert cfg.pnp_steps == probes
        assert cfg.refine == refine and not cfg.refine_average
        assert cfg.pnp_k == 5 and cfg.n_action_steps == 10
        assert cfg.num_samples is None and not cfg.multi_sample_refine_selected
        assert cfg.uncertainty_gradient_mode is None and cfg.refine_threshold is None
        assert cfg.policy_source_id == "source@revision"
        assert cfg.save_generated_chunks and cfg.save_time_uncertainty and cfg.save_trajectory
        assert cfg.skip_unused_renders and cfg.render_lead == 2
        assert cfg.video == "off" and not cfg.save_observations
    settings = coarse_refinement_arm_settings(methods)
    assert settings[0]["probe_flow_times"] == pytest.approx([.6, .4])
    assert settings[1]["probe_flow_times"] == pytest.approx([1 / 3])
    all_methods = methods + build_five_step_diversity_methods("source@revision")
    hashes = {SupabaseStore.config_hash(SupabaseStore._logical_key(name, cfg))
              for name, cfg in all_methods}
    assert len(hashes) == 6
    assert COARSE_REFINEMENT_EXPERIMENT == "pi05-coarse-single-refinement-pro220-v1"


@pytest.mark.parametrize("arm", range(3))
def test_real_sampler_loop_uses_requested_schedule_and_stock_control(arm):
    """Exercise the actual sampler + PnP tap with a tiny CPU velocity-field stand-in."""
    _, cfg = build_coarse_refinement_methods("source@revision")[arm]
    recorder = PnPRecorder()
    recorder.new_episode()
    tap = RolloutTap(cfg, recorder, device="cpu", adim=7)
    stock = torch.full((1, 50, 7), 42.)
    language = SimpleNamespace(config=SimpleNamespace())
    model = SimpleNamespace(
        config=SimpleNamespace(num_inference_steps=10, chunk_size=50, max_action_dim=7),
        _pnp=_SamplerState(strategy=tap, num_steps=cfg.num_inference_steps),
        _orig_sample_actions=Mock(return_value=stock),
        embed_prefix=Mock(return_value=(torch.zeros(1, 2, 4),
                                       torch.ones(1, 2, dtype=torch.bool),
                                       torch.ones(1, 2, dtype=torch.bool))),
        _prepare_attention_masks_4d=lambda value: value,
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(model=SimpleNamespace(language_model=language)),
            forward=Mock(return_value=(None, None))),
        denoise_step=lambda **kwargs: kwargs["x_t"] * .25 + .1)
    stub = SimpleNamespace(make_att_2d_masks=lambda pad, att: pad)
    _pnp_seed_perturb(123)
    noise = torch.ones(1, 50, 7)
    with patch.dict(sys.modules, {"lerobot.policies.pi05.modeling_pi05": stub}):
        actions = _sample_actions_hooked(
            model, [], [], torch.zeros(1, 2, dtype=torch.long), None, noise=noise)
    assert actions.shape == (1, 50, 7) and torch.isfinite(actions).all()
    assert not actions.requires_grad
    chunk = recorder.current_chunks()[0]
    assert chunk["num_steps"] == cfg.num_inference_steps
    assert [step["step"] for step in chunk["steps"]] == list(cfg.pnp_steps)
    assert [step["s"] for step in chunk["steps"]] == pytest.approx(
        [1 - index / cfg.num_inference_steps for index in cfg.pnp_steps])
    assert all(len(step["u_time"]) == 50 for step in chunk["steps"])
    if cfg.refine:
        model._orig_sample_actions.assert_not_called()
        assert not torch.equal(actions, stock)
        assert tap._refine_applied == len(cfg.pnp_steps)
    else:
        assert torch.equal(actions, stock)
        assert model._orig_sample_actions.call_args.kwargs["num_steps"] == 3
        assert torch.equal(model._orig_sample_actions.call_args.kwargs["noise"], noise)
        assert tap._refine_applied == 0
    expected_vf = cfg.num_inference_steps + len(cfg.pnp_steps) * cfg.pnp_k
    if not cfg.refine:
        expected_vf += cfg.num_inference_steps  # exact stock + measurement replay
    assert model._pnp.vf_evals == expected_vf


def test_matched_printout_excludes_partial_identities_and_weights_overall_by_identity():
    methods = [name for name, _ in build_coarse_refinement_methods("s@r")]
    keys = [(suite, index, 10, str(index)) for suite, index in
            [("suite_a", 0), ("suite_a", 1), ("suite_b", 2), ("suite_b", 3)]]
    outcomes = {key: dict.fromkeys(methods, True) for key in keys[:3]}
    outcomes[keys[3]] = {methods[0]: False}
    references = {"hist stock": dict(zip(keys, [True, True, False, True]))}
    table = format_matched_progress_table(outcomes, methods, references)
    assert "3 identities completed" in table
    assert "hist stock" in table
    overall = next(line for line in table.splitlines() if line.startswith("OVERALL"))
    assert overall.count("100% (3/3)") == 3
    assert "67% (2/3)" in overall  # not unweighted suite mean of 50%
    assert "(3/4)" not in table
    with pytest.raises(ValueError, match="missing completed identity"):
        format_matched_progress_table(outcomes, methods, {"hist stock": {}})


@pytest.mark.parametrize("arm", range(3))
def test_new_arm_rollouts_execute_ten_of_fifty_and_reuse_stock_chunk_seeds(arm):
    _, cfg = build_coarse_refinement_methods("source@revision")[arm]
    obs = {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.zeros(4),
           "robot0_gripper_qpos": np.zeros(2)}
    env = Mock()
    env.set_init_state.return_value = obs
    env.step.return_value = (obs, 0., False, {})
    env.check_success.return_value = False
    policy = SimpleNamespace(
        model=SimpleNamespace(_pnp=_SamplerState()), reset=Mock(),
        config=SimpleNamespace(chunk_size=50, max_action_dim=7, num_inference_steps=10,
                               n_action_steps=50))
    calls = []

    def predict(batch, noise):
        assert policy.model._pnp.num_steps == cfg.num_inference_steps
        calls.append(noise.clone())
        return torch.arange(50.).view(1, 50, 1).expand(1, 50, 7)

    policy.predict_action_chunk = predict
    episode = {**_episode(), "task_desc": "test", "max_steps": 25}
    with (patch("pnp.rollout.obs_to_policy", return_value={}),
          patch("pnp.rollout.set_camera_observables", return_value=False)):
        result = run_episode(env, episode, policy, lambda value: value, lambda value: value,
                             torch.device("cpu"), cfg)
    assert result["status"] == "completed" and result["n_steps"] == 25
    assert len(calls) == result["n_chunks"] == 3
    assert result["generated_chunks"]["chunks"].shape == (3, 50, 7)
    assert result["chunk_noise_seeds"] == [chunk_noise_seed(result["episode_seed"], i)
                                           for i in range(3)]
    # Ignore ten settling actions. Replan at actions 10 and 20, never execute the tail.
    executed = [float(call.args[0][0]) for call in env.step.call_args_list[10:]]
    assert executed == list(range(10)) + list(range(10)) + list(range(5))
    assert policy.config.n_action_steps == 50  # restored after the rollout


def _episode(index=0, suite="libero_goal_test"):
    return {"suite": suite, "task_idx": index, "ep_idx": 10,
            "init_state_hash": f"h{index}", "init_state": np.array([index], dtype=float),
            "max_steps": 300, "bddl_path": "test.bddl"}


def _reference_fixture():
    episode = _episode()
    method, config = build_coarse_refinement_methods("source@revision")[0]
    store = SupabaseStore.__new__(SupabaseStore)
    row = {**episode, "episode_idx": 10, "method": method, "status": "completed",
           "config_hash": store.config_hash(store._logical_key(method, config)),
           "run_id": "run", "success": True, "chunk_size": 50,
           "episode_seed": episode_seed(episode["init_state"], 10)}
    run = {"run_id": "run", "model_repo_id": "source", "model_revision": "revision"}
    return store, episode, method, config, row, run


def test_reference_is_bound_to_seed_horizon_config_and_checkpoint():
    store, episode, method, config, row, run = _reference_fixture()
    store.fetch_all = Mock(side_effect=[[row], [run]])
    outcomes, digest = _load_exact_reference(
        store, [episode], experiment="old", method=method, config=config,
        source_repo="source", source_revision="revision")
    assert outcomes == {_identity_key(episode): True}
    assert digest == row["config_hash"]
    query = Mock()
    query.eq.return_value = query
    store.fetch_all.call_args_list[0].kwargs["configure"](query)
    assert ("config_hash", digest) in [call.args for call in query.eq.call_args_list]


@pytest.mark.parametrize("field,value", [
    ("episode_seed", 0), ("max_steps", 50), ("chunk_size", 10),
    ("status", "error"), ("success", None), ("method", "wrong"),
    ("config_hash", "wrong"), ("run_id", None), ("run_id", "missing"),
    ("init_state_hash", "other")])
def test_reference_rejects_mismatched_rows(field, value):
    store, episode, method, config, row, run = _reference_fixture()
    row[field] = value
    store.fetch_all = Mock(side_effect=[[row], [run]])
    with pytest.raises(ValueError):
        _load_exact_reference(store, [episode], experiment="old", method=method,
                              config=config, source_repo="source", source_revision="revision")


@pytest.mark.parametrize("case", ["missing", "duplicate", "revision"])
def test_reference_requires_complete_unique_checkpoint_matched_history(case):
    store, episode, method, config, row, run = _reference_fixture()
    rows = [] if case == "missing" else [row, row] if case == "duplicate" else [row]
    if case == "revision":
        run["model_revision"] = "different"
    store.fetch_all = Mock(side_effect=[rows, [run]])
    with pytest.raises(ValueError):
        _load_exact_reference(store, [episode], experiment="old", method=method,
                              config=config, source_repo="source", source_revision="revision")


def test_history_uses_only_unmodified_ten_step_arm_plus_previous_three_five_step_arms():
    with patch("pnp.coarse_refinement_experiment._load_exact_reference",
               return_value=({}, "hash")) as load:
        references, provenance = load_coarse_refinement_references(
            Mock(), [], source_repo="source", source_revision="revision")
    assert len(references) == len(provenance) == 4
    stock = load.call_args_list[0].kwargs
    assert stock["method"] == Method.UNCERTAINTY
    cfg = stock["config"]
    assert not cfg.refine and cfg.uncertainty_gradient_mode is None
    assert cfg.num_samples is None and cfg.correction_lambda is None
    assert cfg.num_inference_steps is None and cfg.n_action_steps == 10
    assert [call.kwargs["method"] for call in load.call_args_list[1:]] == [
        Method.FIVE_STEP_SINGLE_QUERY, Method.FIVE_STEP_LOWEST_U20,
        Method.FIVE_STEP_LOWEST_U20_REFINE]


@pytest.mark.parametrize("resume,error", [(False, False), (True, False), (True, True)])
def test_reporting_counts_completed_three_arm_identities_and_handles_resume(resume, error, capsys):
    methods = build_coarse_refinement_methods("source@revision")
    episodes = [_episode(index) for index in range(25)]
    names = [name for name, _ in methods]
    initial, tally, done = {}, {}, set()
    store = SupabaseStore.__new__(SupabaseStore)
    if resume:
        for index, ep in enumerate(episodes):
            for name, cfg in methods[:2] if index == 24 else methods:
                initial.setdefault(_identity_key(ep), {})[name] = True
                counts = tally.setdefault((ep["suite"], name), [0, 0])
                counts[0] += 1; counts[1] += 1
                done.add(store.rollout_id("test", ep, name, cfg))
    store.existing_keys = Mock(return_value=done)
    store.start_run = Mock(); store.finish_run = Mock(); store.log_result = Mock()
    result = {"success": not error, "status": "error" if error else "completed",
              "error_msg": "sim error" if error else None}
    with (patch("pnp.libero_env.make_env", return_value=Mock()),
          patch("pnp.rollout.run_episode_batch", return_value=[result])):
        _run_collection(
            store=store, policy=None, preprocess=None, postprocess=None, device="cpu",
            experiment="test", episodes=episodes, methods=methods, cohort="test",
            shard_count=2, shard_index=0, report_every=0, report_every_identities=25,
            initial_tally=tally, initial_identity_outcomes=initial,
            matched_reference_outcomes={"hist stock": {_identity_key(ep): False for ep in episodes}},
            rollout_batch_size=1, resume_completed_only=True)
    text = capsys.readouterr().out
    store.existing_keys.assert_called_once_with("test", status="completed")
    assert store.log_result.call_count == (1 if resume else 75)
    if error:
        assert "after 25 identities" not in text
        assert "24 identities complete in this shard" in text
        assert "100% (24/24)" in text and "0% (0/24)" in text
    else:
        assert "after 25 identities completed in this shard" in text
        assert f"{1 if resume else 75} new rollouts logged" in text
        assert "100% (25/25)" in text and "0% (0/25)" in text


@pytest.mark.parametrize("shard", [0, 1])
def test_driver_keeps_two_frozen_shards_and_passes_matched_reporting(shard):
    episodes = [_episode(task, f"suite_{suite}") for suite in range(11)
                for task in range(10) for _ in range(2)]
    for index, ep in enumerate(episodes):
        ep["ep_idx"] = 10 + index % 2
        ep["init_state_hash"] += f"-{ep['ep_idx']}"
    store = SupabaseStore.__new__(SupabaseStore)
    store.fetch_all = Mock(return_value=[])
    references = {"hist stock": {_identity_key(ep): True for ep in episodes}}
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, num_inference_steps=10))
    with (patch("pnp.store.SupabaseStore", return_value=store),
          patch("pnp.diversity.source_checkpoint_model_source", return_value=("source", "revision")),
          patch("pnp.experiments.expanded_pro_suites", return_value=[f"suite_{s}" for s in range(11)]),
          patch("pnp.experiments._prepare_libero_pro_expanded_episodes", return_value=episodes),
          patch("pnp.coarse_refinement_experiment.load_coarse_refinement_references",
                return_value=(references, {})),
          patch("pnp.models.load_pi05", return_value=(policy, None, None)),
          patch("pnp.models.default_device", return_value="cpu"),
          patch("pnp.store.gather_provenance", return_value={}),
          patch("pnp.experiments._run_collection") as collect):
        run_coarse_refinement_worker(shard_index=shard, manifest_hash="manifest",
                                      source_model_revision="revision")
    call = collect.call_args.kwargs
    assert len(call["episodes"]) == 110 and len(call["methods"]) == 3
    assert len(call["matched_reference_outcomes"]["hist stock"]) == 110
    assert call["report_every_identities"] == 25 and call["report_every"] == 0
    assert call["resume_completed_only"] and call["rollout_batch_size"] == 1
    assert len(call["run_metadata"]["frozen_identity_manifest"]) == 220


def test_generated_workers_are_clean_thin_launchers():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from build_coarse_refinement_notebooks import worker_notebook
        for shard in range(2):
            path = ROOT / "notebooks" / "workers" / (
                f"62_coarse_single_refinement_pro220_worker_{shard}.ipynb")
            document = json.loads(path.read_text(encoding="utf-8"))
            assert document == worker_notebook(shard)
            source = "\n".join("".join(cell["source"]) for cell in document["cells"])
            assert f"SHARD_INDEX = {shard}" in source
            assert "330 new rollouts" in source and "25/50/75/100" in source
            assert "EPISODE_LIMIT = None" in source
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    assert cell["execution_count"] is None and cell["outputs"] == []
                    ast.parse("".join(cell["source"]))
    finally:
        sys.path.pop(0)
