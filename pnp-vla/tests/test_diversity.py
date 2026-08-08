import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pnp.diversity import (analyze_diversity_signal, bootstrap_manifest_summary,
                           bootstrap_sampler_class, build_bootstrap_manifest,
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
    module.install_final_drive_export(trainer, final_dir, metadata)

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
    assert not final_dir.with_name(".final_model_m0.staging").exists()
    assert not final_dir.with_name(".final_model_m0.previous").exists()


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
