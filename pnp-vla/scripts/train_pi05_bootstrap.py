"""Launch the pinned LeRobot pi0.5 trainer with a task-stratified episode bootstrap."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


LIBERO_TO_PI05_BASE_RENAME_MAP = {
    "observation.images.image": "observation.images.base_0_rgb",
    "observation.images.image2": "observation.images.left_wrist_0_rgb",
}
SUPPORTED_SOURCE_MODELS = {
    "lerobot/pi05_base",
    "lerobot/pi05_libero_finetuned",
}


def _is_complete_checkpoint(path: Path) -> bool:
    required = (
        "pretrained_model/train_config.json",
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "training_state/training_step.json",
        "training_state/optimizer_state.safetensors",
        "training_state/scheduler_state.json",
    )
    return all((path / relative).is_file() for relative in required)


def mirror_latest_checkpoint(checkpoint_dir: Path, mirror_dir: Path,
                             *, keep_local: bool = False) -> Path:
    """Durably mirror one complete checkpoint, then prune superseded copies."""
    checkpoint_dir = Path(checkpoint_dir)
    mirror_dir = Path(mirror_dir)
    if not checkpoint_dir.name.isdigit() or checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(f"refusing to prune unexpected checkpoint path: {checkpoint_dir}")
    if not _is_complete_checkpoint(checkpoint_dir):
        raise RuntimeError(f"checkpoint is incomplete: {checkpoint_dir}")

    mirror_dir.mkdir(parents=True, exist_ok=True)
    newer = sorted(
        (path for path in mirror_dir.iterdir()
         if path.is_dir() and path.name.isdigit()
         and int(path.name) > int(checkpoint_dir.name)
         and _is_complete_checkpoint(path)),
        key=lambda path: int(path.name),
    )
    if newer:
        raise RuntimeError(
            f"refusing to replace newer Drive checkpoint {newer[-1].name} "
            f"with older checkpoint {checkpoint_dir.name}")
    destination = mirror_dir / checkpoint_dir.name
    staging = mirror_dir / f".{checkpoint_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    if not destination.exists():
        shutil.copytree(checkpoint_dir, staging)
        if not _is_complete_checkpoint(staging):
            raise RuntimeError(f"mirrored checkpoint is incomplete: {staging}")
        staging.rename(destination)
    elif not _is_complete_checkpoint(destination):
        raise RuntimeError(f"existing mirrored checkpoint is incomplete: {destination}")

    # The old checkpoint remains recoverable until the new Drive copy is complete.
    for candidate in mirror_dir.iterdir():
        if candidate != destination and candidate.is_dir() and candidate.name.isdigit():
            shutil.rmtree(candidate)
    for candidate in checkpoint_dir.parent.iterdir():
        if (keep_local and candidate == checkpoint_dir) or not candidate.name.isdigit():
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate)
    local_note = "kept latest local copy" if keep_local else "removed local checkpoint copies"
    print(f"Mirrored checkpoint {checkpoint_dir.name} to {destination}; {local_note}.")
    return destination


def restore_latest_mirrored_checkpoint(output_dir: Path, mirror_dir: Path) -> Path | None:
    """Use the newest complete Drive checkpoint, falling back to local content."""
    output_dir, mirror_dir = Path(output_dir), Path(mirror_dir)
    local_checkpoints = output_dir / "checkpoints"
    local_last = output_dir / "checkpoints" / "last"
    drive_candidates = sorted(
        (path for path in mirror_dir.glob("*")
         if path.is_dir() and path.name.isdigit() and _is_complete_checkpoint(path)),
        key=lambda path: int(path.name),
    ) if mirror_dir.exists() else []
    if drive_candidates:
        source = drive_candidates[-1]
        local_checkpoints.mkdir(parents=True, exist_ok=True)
        # Drive is authoritative. Remove partial or stale local saves before restoring it.
        for candidate in local_checkpoints.iterdir():
            if candidate.is_dir() and candidate.name.isdigit():
                shutil.rmtree(candidate)
        if local_last.is_symlink():
            local_last.unlink()
        elif local_last.exists():
            shutil.rmtree(local_last)
        destination = local_checkpoints / source.name
        staging = local_checkpoints / f".{source.name}.restore"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if not _is_complete_checkpoint(staging):
            raise RuntimeError(f"restored checkpoint is incomplete: {staging}")
        staging.rename(destination)
        local_last.symlink_to(destination.name)
        print(f"Restored preferred Drive checkpoint {source.name} to local disk.")
        return local_last

    local_candidates = sorted(
        (path for path in local_checkpoints.glob("*")
         if path.is_dir() and path.name.isdigit() and _is_complete_checkpoint(path)),
        key=lambda path: int(path.name),
    ) if local_checkpoints.exists() else []
    if not local_candidates:
        return None
    source = local_candidates[-1]
    # Drive has no complete checkpoint, so preserve the local fallback before loading it.
    mirror_latest_checkpoint(source, mirror_dir, keep_local=True)
    if local_last.is_symlink():
        local_last.unlink()
    elif local_last.exists():
        shutil.rmtree(local_last)
    local_last.symlink_to(source.name)
    print(f"Using local fallback checkpoint {source.name}; Drive mirror initialized.")
    return local_last


def install_checkpoint_mirroring(trainer, mirror_dir: Path, *, keep_local: bool = False):
    """Make Drive authoritative and release local checkpoint disk after loading or saving."""
    original = trainer.update_last_checkpoint
    original_load = trainer.load_training_state

    def update_mirror_and_prune(checkpoint_dir):
        result = original(checkpoint_dir)
        mirror_latest_checkpoint(
            Path(checkpoint_dir), Path(mirror_dir), keep_local=keep_local)
        return result

    def load_and_release_local(checkpoint_dir, optimizer, scheduler):
        result = original_load(checkpoint_dir, optimizer, scheduler)
        local_checkpoint = Path(checkpoint_dir).resolve()
        mirror_checkpoint = Path(mirror_dir) / local_checkpoint.name
        if (not keep_local and local_checkpoint.parent.name == "checkpoints"
                and _is_complete_checkpoint(mirror_checkpoint)):
            shutil.rmtree(local_checkpoint)
            print(f"Loaded checkpoint {local_checkpoint.name}; released its local disk copy.")
        return result

    trainer.update_last_checkpoint = update_mirror_and_prune
    trainer.load_training_state = load_and_release_local
    return trainer


def export_inference_bundle(policy, preprocessor, postprocessor, destination: Path,
                            metadata: dict, *, train_cfg=None,
                            metadata_filename: str = "export_metadata.json") -> Path:
    """Atomically replace one model-only bundle without ever writing optimizer state."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging")
    previous = destination.with_name(f".{destination.name}.previous")
    for path in (staging, previous):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir(parents=True)
    try:
        policy.save_pretrained(staging)
        if train_cfg is not None:
            train_cfg.save_pretrained(staging)
        preprocessor.save_pretrained(staging)
        postprocessor.save_pretrained(staging)
        (staging / metadata_filename).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if not (staging / "config.json").is_file():
            raise RuntimeError("model-only Drive export is missing config.json")
        if not list(staging.glob("*.safetensors")):
            raise RuntimeError("model-only Drive export is missing model weights")
        names = {path.name for path in staging.iterdir()}
        if not any("preprocessor" in name for name in names):
            raise RuntimeError("model-only Drive export is missing the preprocessor")
        if not any("postprocessor" in name for name in names):
            raise RuntimeError("model-only Drive export is missing the postprocessor")

        if destination.exists():
            destination.rename(previous)
        try:
            staging.rename(destination)
        except Exception:
            if previous.exists() and not destination.exists():
                previous.rename(destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def load_model_only_recovery(recovery_dir: Path, *, manifest_hash: str,
                             member_index: int, target_steps: int) -> tuple[Path, int] | None:
    """Return a validated model-only recovery bundle, if one exists."""
    latest = Path(recovery_dir) / "latest"
    metadata_path = latest / "recovery_export.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "manifest_hash": manifest_hash,
        "member_index": int(member_index),
        "target_steps": int(target_steps),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"recovery snapshot {key} mismatch: {metadata.get(key)!r} != {value!r}")
    completed = int(metadata.get("completed_steps", -1))
    if not 0 < completed < int(target_steps):
        raise RuntimeError(
            f"recovery completed_steps must be between 1 and {target_steps - 1}, got {completed}")
    if not (latest / "config.json").is_file() or not list(latest.glob("*.safetensors")):
        raise RuntimeError(f"incomplete model-only recovery snapshot: {latest}")
    print(f"Using model-only recovery snapshot at global step {completed}: {latest}")
    return latest, completed


def install_model_only_recovery(trainer, recovery_dir: Path, *, every_steps: int,
                                start_step: int, target_steps: int, metadata: dict):
    """Save one rolling inference bundle after optimizer updates; never save optimizer state."""
    if every_steps <= 0:
        raise ValueError("model-only recovery frequency must be positive")
    recovery_dir = Path(recovery_dir)
    captured = {}
    local_updates = 0
    original_make_processors = trainer.make_pre_post_processors
    original_update_policy = trainer.update_policy

    def make_processors_and_capture(*args, **kwargs):
        preprocessor, postprocessor = original_make_processors(*args, **kwargs)
        captured["preprocessor"] = preprocessor
        captured["postprocessor"] = postprocessor
        return preprocessor, postprocessor

    def update_policy_and_maybe_export(*args, **kwargs):
        nonlocal local_updates
        result = original_update_policy(*args, **kwargs)
        local_updates += 1
        completed = int(start_step) + local_updates
        if completed >= int(target_steps) or completed % int(every_steps):
            return result
        accelerator = kwargs.get("accelerator")
        if accelerator is None and len(args) > 5:
            accelerator = args[5]
        if accelerator is not None and not accelerator.is_main_process:
            return result
        preprocessor = captured.get("preprocessor")
        postprocessor = captured.get("postprocessor")
        if preprocessor is None or postprocessor is None:
            raise RuntimeError("cannot save model-only recovery before processors are created")
        policy = kwargs.get("policy") if "policy" in kwargs else args[1]
        unwrapped = accelerator.unwrap_model(policy) if accelerator is not None else policy
        snapshot_metadata = {
            **metadata,
            "completed_steps": completed,
            "target_steps": int(target_steps),
            "recovery_kind": "model_only_fresh_optimizer_on_restart",
        }
        destination = export_inference_bundle(
            unwrapped, preprocessor, postprocessor, recovery_dir / "latest",
            snapshot_metadata, metadata_filename="recovery_export.json")
        print(f"Model-only recovery snapshot saved at global step {completed}: {destination}")
        return result

    trainer.make_pre_post_processors = make_processors_and_capture
    trainer.update_policy = update_policy_and_maybe_export
    return trainer


def install_final_drive_export(trainer, final_dir: Path, metadata: dict,
                               *, cleanup_recovery_dir: Path | None = None):
    """Export inference-ready weights and processors to Drive before the Hub upload."""
    final_dir = Path(final_dir)
    captured = {}
    original_make_policy = trainer.make_policy
    original_make_processors = trainer.make_pre_post_processors

    def make_policy_and_wrap_push(*args, **kwargs):
        policy = original_make_policy(*args, **kwargs)
        original_push = policy.push_model_to_hub

        def export_then_push(cfg, *push_args, **push_kwargs):
            preprocessor = captured.get("preprocessor")
            postprocessor = captured.get("postprocessor")
            if preprocessor is None or postprocessor is None:
                raise RuntimeError("cannot export final model before processors are created")
            export_inference_bundle(
                policy, preprocessor, postprocessor, final_dir, metadata,
                train_cfg=cfg, metadata_filename="final_export.json")
            print(f"Final trained model exported to Drive: {final_dir}")
            if cleanup_recovery_dir is not None and Path(cleanup_recovery_dir).exists():
                shutil.rmtree(cleanup_recovery_dir)
                print(f"Removed superseded model-only recovery: {cleanup_recovery_dir}")
            return original_push(cfg, *push_args, **push_kwargs)

        policy.push_model_to_hub = export_then_push
        return policy

    def make_processors_and_capture(*args, **kwargs):
        preprocessor, postprocessor = original_make_processors(*args, **kwargs)
        captured["preprocessor"] = preprocessor
        captured["postprocessor"] = postprocessor
        return preprocessor, postprocessor

    trainer.make_policy = make_policy_and_wrap_push
    trainer.make_pre_post_processors = make_processors_and_capture
    return trainer


def install_fresh_pi05_processors(trainer, rename_map: dict[str, str]):
    """Build processors with pinned LeRobot instead of loading incompatible Hub metadata.

    The raw checkpoint's processor JSON names ``relative_actions_processor``, while the pinned
    LeRobot registry uses the current delta/absolute action processor names. The model weights are
    compatible; only the serialized pipeline is stale. Rebuilding also installs current LIBERO
    statistics, then we apply the source-to-policy camera rename explicitly.
    """
    original = trainer.make_pre_post_processors

    def make_compatible(policy_cfg, pretrained_path=None, **kwargs):
        if pretrained_path is None:
            return original(policy_cfg, pretrained_path=None, **kwargs)
        fresh_kwargs = {}
        if "dataset_stats" in kwargs:
            fresh_kwargs["dataset_stats"] = kwargs["dataset_stats"]
        preprocessor, postprocessor = original(
            policy_cfg, pretrained_path=None, **fresh_kwargs)
        rename_steps = [
            step for step in getattr(preprocessor, "steps", [])
            if step.__class__.__name__ == "RenameObservationsProcessorStep"
        ]
        if len(rename_steps) != 1:
            raise RuntimeError(
                f"expected one rename-observations processor, found {len(rename_steps)}")
        rename_steps[0].rename_map = dict(rename_map)
        print("Built fresh pinned-LeRobot processors with LIBERO camera mapping.")
        return preprocessor, postprocessor

    trainer.make_pre_post_processors = make_compatible
    return trainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--member", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-repo-id", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-mirror-dir", type=Path)
    parser.add_argument("--final-drive-dir", type=Path)
    parser.add_argument("--model-only-recovery-dir", type=Path)
    parser.add_argument("--model-only-recovery-freq", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--scheduler-warmup-steps", type=int)
    parser.add_argument("--scheduler-decay-steps", type=int)
    parser.add_argument("--scheduler-decay-lr", type=float)
    return parser


def build_lerobot_args(args, manifest: dict, *, source_model_path: str | None = None,
                       session_steps: int | None = None) -> list[str]:
    if args.resume:
        config = args.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not config.exists():
            raise FileNotFoundError(f"cannot resume; missing {config}")
        return [
            f"--config_path={config}",
            "--resume=true",
            f"--save_freq={args.save_freq}",
            f"--save_checkpoint={str(not args.no_checkpoints).lower()}",
        ]
    if args.output_dir.exists():
        raise FileExistsError(
            f"{args.output_dir} already exists. Set --resume if it contains a valid checkpoint, "
            "or choose a new output directory.")
    member_seed = manifest["members"][args.member]["seed"]
    source_model = source_model_path or manifest["source_model"]
    values = [
        f"--dataset.repo_id={manifest['dataset_repo_id']}",
        f"--dataset.revision={manifest['dataset_revision']}",
        f"--policy.path={source_model}",
        f"--policy.repo_id={args.policy_repo_id}",
        "--policy.push_to_hub=true",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        f"--policy.compile_model={str(args.compile_model).lower()}",
        "--policy.freeze_vision_encoder=false",
        f"--policy.train_expert_only={str(args.expert_only).lower()}",
        "--policy.n_action_steps=10",
        f"--output_dir={args.output_dir}",
        f"--job_name=pi05_diverse_m{args.member}",
        f"--seed={member_seed}",
        f"--steps={session_steps if session_steps is not None else args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        "--eval_freq=0",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--save_checkpoint={str(not args.no_checkpoints).lower()}",
        f"--wandb.enable={str(args.wandb).lower()}",
    ]
    if manifest["source_model"] == "lerobot/pi05_base":
        values.insert(2, "--rename_map=" + json.dumps(
            LIBERO_TO_PI05_BASE_RENAME_MAP, separators=(",", ":")))
    if getattr(args, "learning_rate", None) is not None:
        values.append(f"--policy.optimizer_lr={args.learning_rate}")
    if getattr(args, "scheduler_warmup_steps", None) is not None:
        values.append(f"--policy.scheduler_warmup_steps={args.scheduler_warmup_steps}")
    if getattr(args, "scheduler_decay_steps", None) is not None:
        values.append(f"--policy.scheduler_decay_steps={args.scheduler_decay_steps}")
    if getattr(args, "scheduler_decay_lr", None) is not None:
        values.append(f"--policy.scheduler_decay_lr={args.scheduler_decay_lr}")
    return values


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    # LeRobot otherwise uses a second Hub cache and redownloads the ~35 GB LIBERO dataset.
    from huggingface_hub.constants import HF_HOME

    os.environ.setdefault("HF_LEROBOT_HOME", str(HF_HOME))
    from pnp.diversity import (bootstrap_manifest_summary, install_bootstrap_sampler,
                               load_bootstrap_manifest, require_full_finetune_gpu)

    manifest = load_bootstrap_manifest(args.manifest)
    if manifest["source_model"] not in SUPPORTED_SOURCE_MODELS:
        raise ValueError(
            f"unsupported diversity source model {manifest['source_model']!r}; expected one of "
            f"{sorted(SUPPORTED_SOURCE_MODELS)}")
    if not manifest.get("dataset_revision") or not manifest.get("source_model_revision"):
        raise ValueError("real training requires pinned dataset and source-model revisions")
    if args.resume and args.model_only_recovery_dir is not None:
        raise ValueError("full optimizer resume and model-only recovery are mutually exclusive")
    if not args.expert_only:
        print("Full-model fine-tune preflight:", require_full_finetune_gpu())
    else:
        print("EXPLICIT FALLBACK: training action expert/projections only (not the planned full fine-tune).")
    print(bootstrap_manifest_summary(manifest).to_string(index=False))
    print({"member": args.member, "manifest_hash": manifest["manifest_hash"],
           "source_model": manifest["source_model"],
           "source_model_revision": manifest["source_model_revision"],
           "dataset_revision": manifest["dataset_revision"],
           "policy_repo_id": args.policy_repo_id,
           "save_freq_requested": args.save_freq,
           "checkpoints_enabled": not args.no_checkpoints})

    start_step = 0
    recovered = None
    if args.model_only_recovery_dir is not None:
        recovered = load_model_only_recovery(
            args.model_only_recovery_dir,
            manifest_hash=manifest["manifest_hash"], member_index=args.member,
            target_steps=args.steps)
    if recovered is None:
        from huggingface_hub import snapshot_download
        source_model_path = snapshot_download(
            repo_id=manifest["source_model"], revision=manifest["source_model_revision"])
    else:
        source_model_path, start_step = recovered
    session_steps = int(args.steps) - int(start_step)
    if session_steps <= 0:
        raise RuntimeError(
            f"nothing to train: recovery already records {start_step}/{args.steps} steps")
    print({"global_start_step": start_step, "session_steps": session_steps,
           "global_target_steps": args.steps, "training_source_path": str(source_model_path)})

    trainer = install_bootstrap_sampler(manifest, args.member)
    if args.checkpoint_mirror_dir is not None:
        if args.resume:
            restored = restore_latest_mirrored_checkpoint(
                args.output_dir, args.checkpoint_mirror_dir)
            if restored is None and not (
                    args.output_dir / "checkpoints" / "last" / "pretrained_model" /
                    "train_config.json").is_file():
                raise FileNotFoundError(
                    f"no resumable checkpoint in {args.output_dir} or "
                    f"{args.checkpoint_mirror_dir}")
        trainer = install_checkpoint_mirroring(
            trainer, args.checkpoint_mirror_dir, keep_local=args.wandb)
    if (not args.resume and start_step == 0
            and manifest["source_model"] == "lerobot/pi05_base"):
        trainer = install_fresh_pi05_processors(
            trainer, LIBERO_TO_PI05_BASE_RENAME_MAP)
    shared_metadata = {
        "member_index": args.member,
        "manifest_hash": manifest["manifest_hash"],
        "source_model": manifest["source_model"],
        "source_model_revision": manifest["source_model_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "policy_repo_id": args.policy_repo_id,
    }
    if args.model_only_recovery_dir is not None:
        trainer = install_model_only_recovery(
            trainer, args.model_only_recovery_dir,
            every_steps=args.model_only_recovery_freq,
            start_step=start_step, target_steps=args.steps,
            metadata=shared_metadata)
    if args.final_drive_dir is not None:
        trainer = install_final_drive_export(
            trainer,
            args.final_drive_dir,
            metadata={
                **shared_metadata,
                "training_completed_steps": args.steps,
                "session_start_step": start_step,
                "session_training_steps": session_steps,
            },
            cleanup_recovery_dir=args.model_only_recovery_dir,
        )
    lerobot_args = build_lerobot_args(
        args, manifest, source_model_path=str(source_model_path),
        session_steps=session_steps)
    print("LeRobot training arguments:")
    print("\n".join(f"  {value}" for value in lerobot_args))
    sys.argv = ["lerobot-train", *lerobot_args]
    trainer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
