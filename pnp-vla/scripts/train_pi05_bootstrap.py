"""Launch the pinned LeRobot pi0.5 trainer with a task-stratified episode bootstrap."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


LIBERO_TO_PI05_BASE_RENAME_MAP = {
    "observation.images.image": "observation.images.base_0_rgb",
    "observation.images.image2": "observation.images.left_wrist_0_rgb",
}


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
    lerobot_args = build_lerobot_args(
        args, manifest, source_model_path=source_model_path)
    print("LeRobot training arguments:")
    print("\n".join(f"  {value}" for value in lerobot_args))
    sys.argv = ["lerobot-train", *lerobot_args]
    trainer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
