import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pnp.config import RolloutConfig
from pnp.diversity import (DIVERSITY_EXPERIMENT_PREFIX, DIVERSITY_V2_EXPERIMENT_PREFIX,
                           aggregation_gate_signal_window_sweep,
                           analyze_checkpoint_refinement,
                           analyze_diversity_selective_refinement, analyze_diversity_signal,
                           analyze_source_member_ensembles,
                           bootstrap_manifest_summary,
                           bootstrap_sampler_class, build_bootstrap_manifest,
                           diversity_baseline_cohort, diversity_experiment,
                           diversity_model_source,
                           diversity_selective_refinement_figures,
                           validate_bootstrap_manifest)


def _training_module():
    path = Path(__file__).parents[1] / "scripts" / "train_pi05_bootstrap.py"
    spec = importlib.util.spec_from_file_location("train_pi05_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode_rows():
    return [
        {"episode_index": 0, "tasks": ["task a"]},
        {"episode_index": 1, "tasks": ["task a"]},
        {"episode_index": 2, "tasks": ["task b"]},
        {"episode_index": 3, "tasks": ["task b"]},
        {"episode_index": 4, "tasks": ["task b"]},
    ]


def test_diversity_experiment_prefixes_keep_v1_and_v2_isolated():
    assert diversity_experiment(0) == f"{DIVERSITY_EXPERIMENT_PREFIX}-m0"
    assert diversity_experiment(
        1, DIVERSITY_V2_EXPERIMENT_PREFIX) == f"{DIVERSITY_V2_EXPERIMENT_PREFIX}-m1"
    with np.testing.assert_raises_regex(ValueError, "member_index"):
        diversity_experiment(2, DIVERSITY_V2_EXPERIMENT_PREFIX)
    with np.testing.assert_raises_regex(ValueError, "cannot be empty"):
        diversity_experiment(0, "")


def test_task_stratified_bootstrap_is_deterministic_and_preserves_every_task():
    first = build_bootstrap_manifest(_episode_rows(), seed=159)
    second = build_bootstrap_manifest(_episode_rows(), seed=159)
    assert first == second
    validate_bootstrap_manifest(first)
    assert first["n_tasks"] == 2
    assert first["n_source_episodes"] == 5
    assert first["source_model"] == "lerobot/pi05_base"
    for member in first["members"]:
        assert len(member["task_draws"]["task a"]) == 2
        assert len(member["task_draws"]["task b"]) == 3
        assert set(member["task_draws"]["task a"]) <= {0, 1}
        assert set(member["task_draws"]["task b"]) <= {2, 3, 4}
    summary = bootstrap_manifest_summary(first)
    assert len(summary) == 2
    assert (summary.draws == 5).all()

    tampered = copy.deepcopy(first)
    tampered["source_model"] = "lerobot/pi05_libero"
    with np.testing.assert_raises_regex(ValueError, "manifest hash"):
        validate_bootstrap_manifest(tampered)


def test_bootstrap_sampler_repeats_whole_episode_frame_ranges():
    manifest = build_bootstrap_manifest(_episode_rows(), seed=7)
    sampler_type = bootstrap_sampler_class(manifest, 0)
    starts = [0, 4, 9, 12, 18]
    ends = [4, 9, 12, 18, 23]
    sampler = sampler_type(starts, ends, drop_n_last_frames=1, shuffle=False)
    counts = {int(key): int(value)
              for key, value in manifest["members"][0]["multiplicities"].items()}
    expected = sum(count * (ends[index] - starts[index] - 1)
                   for index, count in counts.items())
    assert len(sampler) == expected
    assert list(iter(sampler)) == sampler.indices


def test_training_maps_two_libero_cameras_to_raw_pi05_base_names():
    module = _training_module()
    args = SimpleNamespace(
        resume=False, output_dir=Path("unused-output"), policy_repo_id="user/model",
        member=0, compile_model=False, expert_only=False, steps=10, batch_size=2,
        num_workers=1, save_freq=5, log_freq=1, wandb=False, no_checkpoints=False)
    manifest = {
        "dataset_repo_id": "HuggingFaceVLA/libero", "dataset_revision": "dataset-sha",
        "source_model": "lerobot/pi05_base", "members": [{"seed": 159}, {"seed": 160}],
    }
    cli = module.build_lerobot_args(
        args, manifest, source_model_path="/cache/pi05_base")
    rename_arg = next(value for value in cli if value.startswith("--rename_map="))
    assert '"observation.images.image":"observation.images.base_0_rgb"' in rename_arg
    assert '"observation.images.image2":"observation.images.left_wrist_0_rgb"' in rename_arg
    assert "right_wrist_0_rgb" not in rename_arg
    assert "--save_checkpoint=true" in cli


def test_finetuned_v2_keeps_native_libero_features_and_overrides_schedule():
    module = _training_module()
    args = SimpleNamespace(
        resume=False, output_dir=Path("unused-v2-output"), policy_repo_id="user/model-v2",
        member=1, compile_model=False, expert_only=False, steps=6000, batch_size=32,
        num_workers=4, save_freq=10000, log_freq=20, wandb=False, no_checkpoints=True,
        learning_rate=5e-6, scheduler_warmup_steps=500,
        scheduler_decay_steps=6000, scheduler_decay_lr=5e-7)
    manifest = {
        "dataset_repo_id": "HuggingFaceVLA/libero", "dataset_revision": "dataset-sha",
        "source_model": "lerobot/pi05_libero_finetuned",
        "members": [{"seed": 159}, {"seed": 160}],
    }

    cli = module.build_lerobot_args(
        args, manifest, source_model_path="/cache/pi05_libero_finetuned")

    assert not any(value.startswith("--rename_map=") for value in cli)
    assert "--policy.path=/cache/pi05_libero_finetuned" in cli
    assert "--policy.optimizer_lr=5e-06" in cli
    assert "--policy.scheduler_warmup_steps=500" in cli
    assert "--policy.scheduler_decay_steps=6000" in cli
    assert "--policy.scheduler_decay_lr=5e-07" in cli
    assert "--steps=6000" in cli
    assert "--batch_size=32" in cli
    assert "--save_checkpoint=false" in cli


def test_resume_overrides_saved_checkpoint_frequency_and_can_disable_saves(tmp_path):
    module = _training_module()
    output = tmp_path / "output"
    config = output / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    args = SimpleNamespace(
        resume=True, output_dir=output, save_freq=10000, no_checkpoints=True)

    cli = module.build_lerobot_args(args, {})

    assert cli == [
        f"--config_path={config}",
        "--resume=true",
        "--save_freq=10000",
        "--save_checkpoint=false",
    ]


def test_training_rebuilds_stale_raw_base_processors_and_applies_rename_map():
    module = _training_module()

    class RenameObservationsProcessorStep:
        def __init__(self):
            self.rename_map = {}

    preprocessor = SimpleNamespace(steps=[RenameObservationsProcessorStep()])
    postprocessor = object()
    calls = []

    def make_processors(policy_cfg, pretrained_path=None, **kwargs):
        calls.append((pretrained_path, kwargs))
        assert pretrained_path is None
        return preprocessor, postprocessor

    trainer = SimpleNamespace(make_pre_post_processors=make_processors)
    mapping = {"observation.images.image": "observation.images.base_0_rgb"}
    module.install_fresh_pi05_processors(trainer, mapping)
    actual = trainer.make_pre_post_processors(
        object(), pretrained_path="/cache/pi05_base", dataset_stats={"action": {}})
    assert actual == (preprocessor, postprocessor)
    assert calls == [(None, {"dataset_stats": {"action": {}}})]
    assert preprocessor.steps[0].rename_map == mapping


def test_final_drive_export_survives_hub_upload_failure(tmp_path):
    module = _training_module()
    final_dir = tmp_path / "drive" / "final_model_m0"
    recovery_dir = tmp_path / "drive" / "recovery_m0"
    recovery_dir.mkdir(parents=True)
    (recovery_dir / "old-model.safetensors").write_text("old")
    calls = []

    class Savable:
        def __init__(self, filename, content="{}"):
            self.filename = filename
            self.content = content

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / self.filename).write_text(self.content)

    class Policy(Savable):
        def __init__(self):
            super().__init__("model.safetensors", "weights")

        def save_pretrained(self, path):
            super().save_pretrained(path)
            (Path(path) / "config.json").write_text("{}")

        def push_model_to_hub(self, cfg):
            calls.append(("hub", cfg))
            raise PermissionError("simulated 403")

    policy = Policy()
    cfg = Savable("train_config.json")
    preprocessor = Savable("policy_preprocessor.json")
    postprocessor = Savable("policy_postprocessor.json")
    trainer = SimpleNamespace(
        make_policy=lambda: policy,
        make_pre_post_processors=lambda: (preprocessor, postprocessor),
    )
    metadata = {"training_completed_steps": 3000, "member_index": 0}
    module.install_final_drive_export(
        trainer, final_dir, metadata, cleanup_recovery_dir=recovery_dir)

    actual_policy = trainer.make_policy()
    assert trainer.make_pre_post_processors() == (preprocessor, postprocessor)
    with np.testing.assert_raises_regex(PermissionError, "simulated 403"):
        actual_policy.push_model_to_hub(cfg)

    assert calls == [("hub", cfg)]
    assert (final_dir / "model.safetensors").read_text() == "weights"
    assert (final_dir / "config.json").is_file()
    assert (final_dir / "train_config.json").is_file()
    assert (final_dir / "policy_preprocessor.json").is_file()
    assert (final_dir / "policy_postprocessor.json").is_file()
    assert json.loads((final_dir / "final_export.json").read_text()) == metadata
    assert not recovery_dir.exists()
    assert not final_dir.with_name(".final_model_m0.staging").exists()
    assert not final_dir.with_name(".final_model_m0.previous").exists()


def test_model_only_recovery_keeps_one_valid_bundle_and_reloads_it(tmp_path):
    module = _training_module()

    class Savable:
        def __init__(self, filename, content="{}"):
            self.filename = filename
            self.content = content

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / self.filename).write_text(self.content)

    class Policy(Savable):
        def __init__(self):
            super().__init__("model.safetensors", "weights")

        def save_pretrained(self, path):
            super().save_pretrained(path)
            (Path(path) / "config.json").write_text("{}")

    class Accelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(policy):
            return policy

    policy = Policy()
    preprocessor = Savable("policy_preprocessor.json")
    postprocessor = Savable("policy_postprocessor.json")
    trainer = SimpleNamespace(
        make_pre_post_processors=lambda: (preprocessor, postprocessor),
        update_policy=lambda *args, **kwargs: "updated",
    )
    recovery = tmp_path / "recovery"
    metadata = {"manifest_hash": "manifest", "member_index": 1}
    module.install_model_only_recovery(
        trainer, recovery, every_steps=2, start_step=0, target_steps=6,
        metadata=metadata)
    trainer.make_pre_post_processors()

    for _ in range(4):
        assert trainer.update_policy(
            None, policy, None, None, None, Accelerator()) == "updated"

    latest = recovery / "latest"
    saved = json.loads((latest / "recovery_export.json").read_text())
    assert saved["completed_steps"] == 4
    assert saved["target_steps"] == 6
    assert not list(latest.glob("*optimizer*"))
    assert not latest.with_name(".latest.previous").exists()
    assert not latest.with_name(".latest.staging").exists()
    assert module.load_model_only_recovery(
        recovery, manifest_hash="manifest", member_index=1,
        target_steps=6) == (latest, 4)

    with np.testing.assert_raises_regex(RuntimeError, "manifest_hash mismatch"):
        module.load_model_only_recovery(
            recovery, manifest_hash="different", member_index=1,
            target_steps=6)


def _fake_checkpoint(path: Path, step: int):
    (path / "pretrained_model").mkdir(parents=True)
    (path / "training_state").mkdir()
    (path / "pretrained_model" / "train_config.json").write_text("{}")
    (path / "pretrained_model" / "config.json").write_text("{}")
    (path / "training_state" / "training_step.json").write_text(str(step))
    (path / "pretrained_model" / "model.safetensors").write_bytes(b"weights")
    (path / "training_state" / "optimizer_state.safetensors").write_bytes(b"optimizer")
    (path / "training_state" / "scheduler_state.json").write_text("{}")


def test_checkpoint_mirror_keeps_only_newest_complete_copy(tmp_path):
    module = _training_module()
    checkpoints = tmp_path / "local" / "checkpoints"
    old = checkpoints / "000500"
    new = checkpoints / "001000"
    _fake_checkpoint(old, 500)
    _fake_checkpoint(new, 1000)
    mirror = tmp_path / "drive"
    _fake_checkpoint(mirror / "000500", 500)

    destination = module.mirror_latest_checkpoint(new, mirror)

    assert destination == mirror / "001000"
    assert module._is_complete_checkpoint(destination)
    assert not old.exists()
    assert not new.exists()
    assert not (mirror / "000500").exists()


def test_incomplete_checkpoint_never_prunes_recoverable_copy(tmp_path):
    module = _training_module()
    checkpoints = tmp_path / "local" / "checkpoints"
    old = checkpoints / "000500"
    incomplete = checkpoints / "001000"
    _fake_checkpoint(old, 500)
    (incomplete / "pretrained_model").mkdir(parents=True)
    mirror = tmp_path / "drive"
    _fake_checkpoint(mirror / "000500", 500)

    with np.testing.assert_raises_regex(RuntimeError, "incomplete"):
        module.mirror_latest_checkpoint(incomplete, mirror)

    assert old.exists()
    assert (mirror / "000500").exists()


def test_checkpoint_mirror_restores_resume_layout_in_fresh_runtime(tmp_path):
    module = _training_module()
    mirror = tmp_path / "drive"
    _fake_checkpoint(mirror / "001500", 1500)
    _fake_checkpoint(mirror / "002000", 2000)
    output = tmp_path / "fresh-local"
    partial = output / "checkpoints" / "002500"
    (partial / "pretrained_model").mkdir(parents=True)

    last = module.restore_latest_mirrored_checkpoint(output, mirror)

    assert last == output / "checkpoints" / "last"
    assert last.is_symlink()
    assert last.resolve().name == "002000"
    assert module._is_complete_checkpoint(last)
    assert not partial.exists()


def test_resume_prefers_complete_drive_checkpoint_over_newer_local(tmp_path):
    module = _training_module()
    mirror = tmp_path / "drive"
    _fake_checkpoint(mirror / "002000", 2000)
    output = tmp_path / "local"
    local = output / "checkpoints" / "002500"
    _fake_checkpoint(local, 2500)
    partial = output / "checkpoints" / "003000"
    (partial / "pretrained_model").mkdir(parents=True)

    last = module.restore_latest_mirrored_checkpoint(output, mirror)

    assert last.resolve().name == "002000"
    assert module._is_complete_checkpoint(last)
    assert module._is_complete_checkpoint(mirror / "002000")
    assert not local.exists()
    assert not partial.exists()


def test_resume_uses_local_only_when_drive_has_no_complete_checkpoint(tmp_path):
    module = _training_module()
    mirror = tmp_path / "drive"
    (mirror / "002000" / "pretrained_model").mkdir(parents=True)
    output = tmp_path / "local"
    local = output / "checkpoints" / "002500"
    _fake_checkpoint(local, 2500)

    last = module.restore_latest_mirrored_checkpoint(output, mirror)

    assert last.resolve() == local.resolve()
    assert module._is_complete_checkpoint(last)
    assert module._is_complete_checkpoint(mirror / "002500")
    assert not (mirror / "002000").exists()


def test_resume_releases_restored_local_checkpoint_after_state_load(tmp_path):
    module = _training_module()
    mirror = tmp_path / "drive"
    mirrored = mirror / "002000"
    _fake_checkpoint(mirrored, 2000)
    local = tmp_path / "local" / "checkpoints" / "002000"
    _fake_checkpoint(local, 2000)
    calls = []

    def load_training_state(checkpoint_dir, optimizer, scheduler):
        calls.append(Path(checkpoint_dir))
        return 2000, optimizer, scheduler

    trainer = SimpleNamespace(
        update_last_checkpoint=lambda checkpoint_dir: checkpoint_dir,
        load_training_state=load_training_state,
    )
    module.install_checkpoint_mirroring(trainer, mirror)
    result = trainer.load_training_state(local, object(), object())

    assert result[0] == 2000
    assert calls == [local]
    assert not local.exists()


def test_diversity_signal_reports_oracle_and_first_chunk_selector():
    identities = [
        {"suite": "libero_goal_swap", "task_idx": 0, "episode_idx": index,
         "init_state_hash": str(index)} for index in range(4)
    ]
    outcomes = {0: [True, False, True, False], 1: [True, True, False, False]}
    first_u = {0: [.1, .3, .1, .2], 1: [.2, .1, .3, .2]}
    rows, steps = [], []
    for member in (0, 1):
        for index, identity in enumerate(identities):
            rollout_id = f"m{member}-{index}"
            rows.append({
                **identity, "member_index": member, "rollout_id": rollout_id,
                "success": outcomes[member][index], "status": "completed",
                "u_mean_episode": first_u[member][index],
                "generated_chunks_path": f"chunks/{rollout_id}.npz", "n_steps": 100,
            })
            for euler_step in (3, 4):
                steps.append({"member_index": member, "rollout_id": rollout_id,
                              "chunk_idx": 0, "euler_step": euler_step,
                              "u_mean": first_u[member][index]})
    tables = analyze_diversity_signal(pd.DataFrame(rows), pd.DataFrame(steps))
    summary = tables["diversity_signal_overall"].iloc[0]
    assert summary.n_pairs == 4
    assert summary.n_discordant == 2
    assert np.isclose(summary.best_member_sr, .5)
    assert np.isclose(summary.oracle_either_success_sr, .75)
    assert np.isclose(summary.lower_first_chunk_u_sr, .75)
    assert np.isclose(summary.lower_first_chunk_u_accuracy_discordant, 1.)
    assert np.isclose(summary.lower_first_chunk_u_win_auc, 1.)


def test_diversity_model_source_requires_one_immutable_baseline_revision():
    class Store:
        @staticmethod
        def fetch_all(*args, **kwargs):
            return [
                {"model_repo_id": "user/member-0", "model_revision": "abc"},
                {"model_repo_id": "user/member-0", "model_revision": "abc"},
            ]

    assert diversity_model_source(Store(), member_index=0) == ("user/member-0", "abc")

    class MixedStore:
        @staticmethod
        def fetch_all(*args, **kwargs):
            return [
                {"model_repo_id": "user/member-0", "model_revision": "abc"},
                {"model_repo_id": "user/member-0", "model_revision": "changed"},
            ]

    with np.testing.assert_raises_regex(ValueError, "exactly one recorded model source"):
        diversity_model_source(MixedStore(), member_index=0)


def test_refinement_backfill_reuses_the_existing_baseline_logical_config():
    original = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, save_trajectory=True,
        save_generated_chunks=True, skip_unused_renders=True, render_lead=2)
    backfill_baseline = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, save_trajectory=True,
        save_generated_chunks=True, skip_unused_renders=True, render_lead=2)
    refinement = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, save_trajectory=True,
        skip_unused_renders=True, render_lead=2, refine=True)

    assert original.logical_dict() == backfill_baseline.logical_dict()
    assert original.logical_dict() != refinement.logical_dict()


def test_refinement_cohort_is_exactly_ten_per_suite_and_matched_between_members():
    suites = [f"libero_suite_{index}" for index in range(13)]
    rows = [{
        "rollout_id": f"{suite}-{episode_idx}", "suite": suite,
        "task_idx": episode_idx, "episode_idx": 0,
        "init_state_hash": f"{suite}-{episode_idx}", "status": "completed",
        "success": episode_idx % 2 == 0, "method": "pnp_uncertainty_only",
        "pnp_k": 5, "pnp_step_indices": [3, 4],
    } for suite in suites for episode_idx in range(10)]

    class Store:
        calls = 0

        def fetch_all(self, *args, **kwargs):
            self.calls += 1
            return copy.deepcopy(rows)

    cohort = diversity_baseline_cohort(Store(), expected_per_suite=10)
    assert len(cohort) == 260
    assert set(cohort.groupby(["member_index", "suite"]).size()) == {10}

    class ShortStore(Store):
        def fetch_all(self, *args, **kwargs):
            result = super().fetch_all(*args, **kwargs)
            return result[:-1] if self.calls == 2 else result

    with np.testing.assert_raises_regex(ValueError, "must contain 10 baselines"):
        diversity_baseline_cohort(ShortStore(), expected_per_suite=10)


def test_selective_refinement_uses_lower_u_member_and_fixed_threshold(tmp_path):
    identities = [
        {"suite": "libero_goal_swap" if index % 2 == 0 else "libero_object_swap",
         "task_idx": 0, "episode_idx": index, "init_state_hash": str(index)}
        for index in range(4)
    ]
    baseline = {0: [True, False, True, False], 1: [True, True, False, False]}
    refined = {0: [True, True, True, False], 1: [True, False, False, True]}
    uncertainty = {0: [.02, .04, .025, .05], 1: [.03, .02, .04, .04]}
    rows, steps = [], []
    for member in (0, 1):
        for method, outcomes in (("pnp_uncertainty_only", baseline[member]),
                                 ("pnp_refinement", refined[member])):
            for index, identity in enumerate(identities):
                rollout_id = f"m{member}-{method}-{index}"
                rows.append({
                    **identity, "member_index": member, "rollout_id": rollout_id,
                    "method": method, "success": outcomes[index], "status": "completed",
                    "u_mean_episode": uncertainty[member][index] + .0035,
                })
                for chunk_idx in range(8):
                    for euler_step in (3, 4):
                        steps.append({
                            "member_index": member, "rollout_id": rollout_id,
                            "chunk_idx": chunk_idx, "euler_step": euler_step,
                            "u_mean": uncertainty[member][index] + .001 * chunk_idx,
                        })
    rollout_frame, step_frame = pd.DataFrame(rows), pd.DataFrame(steps)
    tables = analyze_diversity_selective_refinement(
        rollout_frame, step_frame, grid_size=5, min_window=1)
    overall = tables["selective_refinement_overall"].set_index("score_name")
    first = overall.loc["first_chunk"]
    assert first.n_pairs == 4
    assert np.isclose(first.lower_u_baseline_sr, .75)
    assert np.isclose(first.fixed_threshold_sr, 1.)
    assert np.isclose(first.fixed_delta_vs_best_fixed_pp, 50.)
    assert first.n_refined_fixed == 1
    assert first.fixed_F_to_S == 1
    assert first.fixed_S_to_F == 0
    assert set(["first_chunk", "prefix_2_chunks", "prefix_4_chunks",
                "prefix_8_chunks", "full_episode"]) <= set(overall.index)
    member_overall = tables["member_refinement_overall"]
    assert set(member_overall.member_index) == {0, 1}
    assert {"baseline_sr", "refinement_sr", "delta_pp", "failure_auc"} <= set(
        member_overall.columns)
    assert set(tables["member_refinement_top_windows"].member_index) == {0, 1}
    assert len(tables["selective_refinement_loso_folds"].held_out_suite.unique()) == 2
    assert "delta_vs_best_fixed_pp" in tables["selective_refinement_best_windows"]
    assert "delta_vs_best_fixed_pp" in tables["selective_refinement_loso_summary"]
    source_analysis = analyze_checkpoint_refinement(
        rollout_frame[rollout_frame.member_index == 0].drop(columns="member_index"),
        step_frame[step_frame.member_index == 0].drop(columns="member_index"),
        grid_size=5, min_window=1)
    assert set(source_analysis["member_refinement_overall"].member_index) == {
        "source_checkpoint"}
    assert not source_analysis["member_refinement_window_sweep"].empty
    for name, frame in source_analysis.items():
        tables[name.replace(
            "member_refinement", "source_checkpoint_refinement")] = frame
    tables["source_checkpoint_by_suite"] = pd.DataFrame([
        {"suite": suite, "source_baseline_sr": .25,
         "model0_baseline_sr": .5, "model1_baseline_sr": .5}
        for suite in ("libero_goal_swap", "libero_object_swap")])
    horizons = ["first_chunk", "prefix_2_chunks", "prefix_4_chunks",
                "prefix_8_chunks", "full_episode"]
    aggregate_best = tables["selective_refinement_best_windows"].set_index("score_name")
    source_best = source_analysis["member_refinement_top_windows"]
    source_best = source_best[source_best["rank"] == 1].set_index("score_name")
    tables["source_vs_aggregate_best_windows"] = pd.DataFrame([
        {"score_name": score_name,
         "source_window_delta_pp": source_best.loc[score_name].delta_pp,
         "aggregate_window_delta_pp": aggregate_best.loc[score_name].delta_pp,
         "source_window_sr": source_best.loc[score_name].selective_sr,
         "aggregate_window_sr": aggregate_best.loc[score_name].selective_sr}
        for score_name in horizons])
    gate_sweep, gate_best = aggregation_gate_signal_window_sweep(
        tables["selective_refinement_policy_pairs"], grid_size=5, min_window=1)
    assert set(gate_best.gate_signal) == {
        "minimum_u", "mean_u", "maximum_u", "absolute_u_gap"}
    assert len(gate_best) == 4 * len(set(tables[
        "selective_refinement_policy_pairs"].score_name))
    source_delta = source_best.delta_pp.rename("source_window_delta_pp")
    source_sr = source_best.selective_sr.rename("source_window_sr")
    alternative_best = gate_best.merge(
        pd.concat([source_delta, source_sr], axis=1),
        left_on="score_name", right_index=True, validate="many_to_one")
    alternative_best["delta_advantage_vs_source_pp"] = (
        alternative_best.delta_pp - alternative_best.source_window_delta_pp)
    tables["alternative_aggregate_gate_signal_sweep"] = gate_sweep
    tables["alternative_aggregate_gate_signal_best_windows"] = alternative_best
    source_member = analyze_source_member_ensembles(
        tables["member_refinement_pairs"],
        source_analysis["member_refinement_pairs"], grid_size=5, min_window=1)
    assert set(source_member["source_member_best_windows"].ensemble) == {
        "source_plus_m0", "source_plus_m1", "source_plus_m0_m1"}
    assert len(source_member["source_member_best_windows"]) == 3 * len(set(
        tables["member_refinement_pairs"].score_name))
    model1_only = tables["member_refinement_pairs"][
        tables["member_refinement_pairs"].member_index.eq(1)]
    source_m1 = analyze_source_member_ensembles(
        model1_only, source_analysis["member_refinement_pairs"],
        grid_size=5, min_window=1)
    assert set(source_m1["source_member_best_windows"].ensemble) == {"source_plus_m1"}
    tables.update(source_member)
    source_member_comparison = source_member["source_member_best_windows"].merge(
        pd.concat([source_delta, source_sr], axis=1),
        left_on="score_name", right_index=True, validate="many_to_one")
    source_member_comparison["delta_advantage_vs_source_pp"] = (
        source_member_comparison.delta_pp -
        source_member_comparison.source_window_delta_pp)
    tables["source_member_best_window_comparison"] = source_member_comparison
    tables["oracle_opportunity_summary"] = pd.DataFrame([
        {"policy": policy, "success_rate": success_rate}
        for policy, success_rate in [
            ("source baseline", .5), ("source refinement", .75),
            ("model 0 baseline", .5), ("model 1 baseline", .5),
            ("2-member baseline oracle", .75), ("4-arm member oracle", 1.),
            ("source + members baseline oracle", .75), ("all 6-arm oracle", 1.)]])
    figures = diversity_selective_refinement_figures(tables, tmp_path)
    expected = {
        "member_0_refinement_delta_by_suite.png",
        "member_0_delta_and_failure_auc.png",
        "member_0_refinement_by_uncertainty.png",
        "member_0_window_first_chunk.png",
        "member_0_window_full_episode.png",
        "member_1_refinement_delta_by_suite.png",
        "member_1_delta_and_failure_auc.png",
        "member_1_refinement_by_uncertainty.png",
        "member_1_window_first_chunk.png",
        "member_1_window_full_episode.png",
        "aggregate_selector_delta_and_auc.png",
        "aggregate_window_delta_vs_best_by_horizon.png",
        "alternative_aggregate_gate_signals.png",
        "oracle_opportunity.png",
        "source_member_best_windows.png",
        "members_vs_source_checkpoint_by_suite.png",
        "source_vs_aggregate_best_windows.png",
        "selective_refinement_by_horizon.png",
        "selective_refinement_by_chunk.png",
        "selective_refinement_first_chunk_by_suite.png",
    }
    expected |= {f"source_checkpoint_window_{score_name}.png" for score_name in horizons}
    expected |= {f"selective_refinement_window_{score_name}.png" for score_name in horizons}
    expected |= {
        f"source_vs_aggregate_window_sweep_{score_name}.png" for score_name in horizons}
    assert expected == {path.name for path in figures}


def test_prefix_scores_use_whole_episode_u_for_short_trajectories():
    identities = [
        {"suite": "libero_goal_swap", "task_idx": 0, "episode_idx": index,
         "init_state_hash": str(index)}
        for index in range(2)]
    rows, steps = [], []
    for index, identity in enumerate(identities):
        observed_id = f"observed-{index}"
        rows.extend([
            {**identity, "rollout_id": observed_id, "method": "pnp_uncertainty_only",
             "status": "completed", "success": False,
             "u_mean_episode": .05 if index == 0 else .025},
            {**identity, "rollout_id": f"refined-{index}", "method": "pnp_refinement",
             "status": "completed", "success": index == 0,
             "u_mean_episode": .05 if index == 0 else .025},
        ])
        chunk_count = 1 if index == 0 else 4
        for chunk_idx in range(chunk_count):
            for euler_step in (3, 4):
                steps.append({"rollout_id": observed_id, "chunk_idx": chunk_idx,
                              "euler_step": euler_step, "u_mean": .01 + .01 * chunk_idx})

    analysis = analyze_checkpoint_refinement(
        pd.DataFrame(rows), pd.DataFrame(steps), grid_size=5, min_window=1)
    pairs = analysis["member_refinement_pairs"]
    prefix4 = pairs[pairs.score_name == "prefix_4_chunks"].sort_values("episode_idx")
    assert len(prefix4) == 2
    assert np.isclose(prefix4.iloc[0].u, .05)  # whole-episode fallback, not chunk-0 U
    assert np.isclose(prefix4.iloc[1].u, .025)  # mean of chunks 0..3
    sweep = analysis["member_refinement_window_sweep"]
    full_window = sweep[
        (sweep.score_name == "prefix_4_chunks") & sweep.lower.eq(0.) & sweep.upper.eq(.08)]
    assert len(full_window) == 1
    assert full_window.iloc[0].n_refined == 2
    assert np.isclose(full_window.iloc[0].delta_pp, 50.)  # one net fix / both episodes
