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


def _is_complete_checkpoint(path: Path) -> bool:
    return (
        (path / "pretrained_model" / "train_config.json").is_file()
        and (path / "training_state" / "training_step.json").is_file()
    )


def mirror_latest_checkpoint(checkpoint_dir: Path, mirror_dir: Path) -> Path:
    """Durably mirror one complete checkpoint, then prune older local and mirrored copies."""
    checkpoint_dir = Path(checkpoint_dir)
    mirror_dir = Path(mirror_dir)
    if not checkpoint_dir.name.isdigit() or checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(f"refusing to prune unexpected checkpoint path: {checkpoint_dir}")
    if not _is_complete_checkpoint(checkpoint_dir):
        raise RuntimeError(f"checkpoint is incomplete: {checkpoint_dir}")

    mirror_dir.mkdir(parents=True, exist_ok=True)
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
        if candidate != checkpoint_dir and candidate.is_dir() and candidate.name.isdigit():
            shutil.rmtree(candidate)
    print(f"Mirrored checkpoint {checkpoint_dir.name} to {destination}; pruned older checkpoints.")
    return destination


def restore_latest_mirrored_checkpoint(output_dir: Path, mirror_dir: Path) -> Path | None:
    """Restore the newest complete Drive mirror into a fresh local Colab runtime."""
    output_dir, mirror_dir = Path(output_dir), Path(mirror_dir)
    local_last = output_dir / "checkpoints" / "last"
    if (local_last / "pretrained_model" / "train_config.json").is_file():
        return local_last
    candidates = sorted(
        (path for path in mirror_dir.glob("*")
         if path.is_dir() and path.name.isdigit() and _is_complete_checkpoint(path)),
        key=lambda path: int(path.name),
    ) if mirror_dir.exists() else []
    if not candidates:
        return None
    source = candidates[-1]
    local_checkpoints = output_dir / "checkpoints"
    local_checkpoints.mkdir(parents=True, exist_ok=True)
    destination = local_checkpoints / source.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    if local_last.is_symlink():
        local_last.unlink()
    elif local_last.exists():
        raise RuntimeError(f"expected a symlink or absent path at {local_last}")
    local_last.symlink_to(destination.name)
    print(f"Restored checkpoint {source.name} from {source} to local disk.")
    return local_last


def install_checkpoint_mirroring(trainer, mirror_dir: Path):
    """Mirror only completed checkpoints and retain only the latest numbered directory."""
    original = trainer.update_last_checkpoint

    def update_mirror_and_prune(checkpoint_dir):
        result = original(checkpoint_dir)
        mirror_latest_checkpoint(Path(checkpoint_dir), Path(mirror_dir))
        return result

    trainer.update_last_checkpoint = update_mirror_and_prune
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
    parser.add_argument("--checkpoint-mirror-dir", type=Path)
    return parser


def build_lerobot_args(args, manifest: dict, *, source_model_path: str | None = None) -> list[str]:
    if args.resume:
        config = args.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not config.exists():
            raise FileNotFoundError(f"cannot resume; missing {config}")
        return [f"--config_path={config}", "--resume=true"]
    if args.output_dir.exists():
        raise FileExistsError(
            f"{args.output_dir} already exists. Set --resume if it contains a valid checkpoint, "
            "or choose a new output directory.")
    source_model = source_model_path or manifest["source_model"]
    member_seed = manifest["members"][args.member]["seed"]
    return [
        f"--dataset.repo_id={manifest['dataset_repo_id']}",
        f"--dataset.revision={manifest['dataset_revision']}",
        "--rename_map=" + json.dumps(
            LIBERO_TO_PI05_BASE_RENAME_MAP, separators=(",", ":")),
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
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        "--eval_freq=0",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        "--save_checkpoint=true",
        f"--wandb.enable={str(args.wandb).lower()}",
    ]


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    # LeRobot otherwise uses a second Hub cache and redownloads the ~35 GB LIBERO dataset.
    from huggingface_hub.constants import HF_HOME

    os.environ.setdefault("HF_LEROBOT_HOME", str(HF_HOME))
    from pnp.diversity import (DIVERSITY_SOURCE_MODEL, bootstrap_manifest_summary,
                               install_bootstrap_sampler, load_bootstrap_manifest,
                               require_full_finetune_gpu)

    manifest = load_bootstrap_manifest(args.manifest)
    if manifest["source_model"] != DIVERSITY_SOURCE_MODEL:
        raise ValueError(
            f"This experiment must start from raw {DIVERSITY_SOURCE_MODEL}; manifest requests "
            f"{manifest['source_model']}. Build a fresh shared manifest.")
    if not manifest.get("dataset_revision") or not manifest.get("source_model_revision"):
        raise ValueError("real training requires pinned dataset and source-model revisions")
    if not args.expert_only:
        print("Full-model fine-tune preflight:", require_full_finetune_gpu())
    else:
        print("EXPLICIT FALLBACK: training action expert/projections only (not the planned full fine-tune).")
    print(bootstrap_manifest_summary(manifest).to_string(index=False))
    print({"member": args.member, "manifest_hash": manifest["manifest_hash"],
           "source_model": manifest["source_model"],
           "source_model_revision": manifest["source_model_revision"],
           "dataset_revision": manifest["dataset_revision"],
           "policy_repo_id": args.policy_repo_id})

    from huggingface_hub import snapshot_download
    source_model_path = snapshot_download(
        repo_id=manifest["source_model"], revision=manifest["source_model_revision"])
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
        trainer = install_checkpoint_mirroring(trainer, args.checkpoint_mirror_dir)
    if not args.resume:
        trainer = install_fresh_pi05_processors(
            trainer, LIBERO_TO_PI05_BASE_RENAME_MAP)
    lerobot_args = build_lerobot_args(
        args, manifest, source_model_path=source_model_path)
    print("LeRobot training arguments:")
    print("\n".join(f"  {value}" for value in lerobot_args))
    sys.argv = ["lerobot-train", *lerobot_args]
    trainer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
