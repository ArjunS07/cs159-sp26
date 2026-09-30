"""Pretrain SmolVLA Q10 on trajectories, then fine-tune same-root ranking on MPS.

The frozen combined snapshot defines the split. MLflow data and checkpoints are
stored under ~/pnp-vla-runs; the local UI is served separately on port 5000.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.train_smolvla_combined_local import _load_local_credentials  # noqa: E402
from pnp.qplanning_critic.config import QPlanningModelConfig  # noqa: E402
from pnp.qplanning_critic.model import QPlanningCritic  # noqa: E402
from pnp.smolvla_combined_success import (  # noqa: E402
    SOURCE_EXPERIMENTS, load_or_create_combined_snapshot,
    prepare_combined_root_cache,
)
from pnp.smolvla_success_critic import (  # noqa: E402
    TimedRoots, action_statistics, evaluate_roots,
)
from pnp.smolvla_tree_bellman_finetune import _root_batch, _root_scores, _to  # noqa: E402
from pnp.smolvla_two_stage import (  # noqa: E402
    PRETRAIN_KINDS, TrajectorySampler, pairwise_root_loss,
    prepare_trajectory_cache, score_single, single_batch,
)
from pnp.store import SupabaseStore  # noqa: E402


EXPERIMENT = "smolvla-q10-trajectory-pretrain-root-rank-v1"
SEED = 42
BASELINE = (Path.home() / "pnp-vla-runs/checkpoints/combined_1600_root_mc_v1"
            / "671b5b211099997fc83d1277/root_mc/checkpoint_step_002000.pt")


def _learning_rate(step: int, total: int, peak: float) -> float:
    if step <= 100:
        return peak * step / 100
    return peak * 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, total - 100)))


def _set_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _audit(snapshot: dict, root_cache: dict, trajectory_cache: dict) -> dict:
    groups = {group["candidate_group_id"]: group for group in snapshot["groups"]}
    result = {"snapshot_digest": snapshot["snapshot_digest"],
              "pretraining_kinds": PRETRAIN_KINDS,
              "train_groups": len(root_cache["train_group_ids"]),
              "validation_groups": len(root_cache["validation_group_ids"]),
              "pretraining_windows": sum(x["windows"] for x in trajectory_cache["entries"]),
              "pretraining_trajectories": sum(x["windows"] > 0
                                               for x in trajectory_cache["entries"]),
              "empty_branch_continuations": sum(x["windows"] == 0
                                                 for x in trajectory_cache["entries"]),
              "pretraining_by_kind": {
                  kind: {
                      "trajectories": sum(x["windows"] > 0 for x in trajectory_cache["entries"]
                                          if x["kind"] == kind),
                      "windows": sum(x["windows"] for x in trajectory_cache["entries"]
                                     if x["kind"] == kind),
                      "successful_trajectories": sum(
                          x["windows"] > 0 and x["success"]
                          for x in trajectory_cache["entries"] if x["kind"] == kind),
                  } for kind in PRETRAIN_KINDS
              }}
    for name, ids in (("train", root_cache["train_group_ids"]),
                      ("validation", root_cache["validation_group_ids"])):
        labels = np.asarray([[bool(c["success"]) for c in groups[gid]["candidates"]]
                             for gid in ids], bool)
        result[name] = {
            "roots": len(ids), "stock_successes": int(labels[:, 0].sum()),
            "oracle_successes": int(labels.any(1).sum()),
            "mixed_roots": int((labels.any(1) & ~labels.all(1)).sum()),
            "rescueable_roots": int((~labels[:, 0] & labels.any(1)).sum()),
        }
    return result


@torch.no_grad()
def _score_spread(model, dataset: TimedRoots, device: torch.device) -> dict:
    model.eval()
    scores = []
    for start in range(0, len(dataset), 8):
        batch = _to(_root_batch([dataset[i] for i in
                                range(start, min(len(dataset), start + 8))]), device)
        scores.append(_root_scores(model, batch).cpu().numpy())
    values = np.concatenate(scores)
    return {"within_root_score_std": float(values.std(1).mean()),
            "between_root_mean_std": float(values.mean(1).std()),
            "fraction_nonstock_argmax": float((values.argmax(1) != 0).mean())}


def _save_checkpoint(path: Path, model, *, stage: str, update: int,
                     digest: str, audit: dict, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format": "smolvla_two_stage_success_q10_v1", "stage": stage,
        "update": update, "snapshot_digest": digest,
        "architecture": model.architecture_config(), "model": model.state_dict(),
        "data_audit": audit, "config": config,
    }, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-updates", type=int, default=2000)
    parser.add_argument("--finetune-updates", type=int, default=1000)
    parser.add_argument("--trajectory-batch", type=int, default=32)
    parser.add_argument("--root-batch", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="control: same fine-tuning and replay, random initialization")
    parser.add_argument("--model-scale", choices=("original", "small", "large"),
                        default="original",
                        help="small: 408k parameters; large: about 6M; data and losses stay fixed")
    parser.add_argument("--run-root", type=Path, default=Path.home() / "pnp-vla-runs")
    args = parser.parse_args()
    if (min(args.pretrain_updates, args.finetune_updates, args.trajectory_batch,
            args.root_batch) < 1 or args.root_batch % 2):
        parser.error("positive updates/batches and even root-batch required")
    if not torch.backends.mps.is_available():
        raise RuntimeError("this local experiment requires the Mac Metal GPU")
    _load_local_credentials()
    device = torch.device("mps")
    root = args.run_root.expanduser()
    store = SupabaseStore()
    snapshot = load_or_create_combined_snapshot(store)
    root_cache = prepare_combined_root_cache(
        snapshot=snapshot, cache_root=root / "cache", store=store)
    trajectory_cache = prepare_trajectory_cache(
        snapshot=snapshot, root_cache=root_cache, cache_root=root / "cache",
        store=store, workers=args.cache_workers)
    audit = _audit(snapshot, root_cache, trajectory_cache)
    print({"data_audit": audit}, flush=True)
    if args.prepare_only:
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    groups = {row["candidate_group_id"]: row for row in snapshot["groups"]}
    train_roots = TimedRoots(root_cache, root_cache["train_group_ids"], groups)
    validation = TimedRoots(root_cache, root_cache["validation_group_ids"], groups)
    first = train_roots[0]
    architectures = {
        "small": {"width": 128, "n_layers": 1, "n_heads": 4,
                  "ffn_width": 512},
        "original": {"width": 256, "n_layers": 3, "n_heads": 8,
                     "ffn_width": 1024},
        "large": {"width": 296, "n_layers": 4, "n_heads": 8,
                  "ffn_width": 1184},
    }
    cfg = QPlanningModelConfig(
        action_horizon=10, action_dim=7, dropout=.20,
        prefix_pool_tokens=128, **architectures[args.model_scale])
    model = QPlanningCritic(
        prefix_dim=first["prefix"].shape[-1],
        robot_dim=len(first["robot"]), proprio_dim=len(first["proprio"]),
        config=cfg)
    mean, std = action_statistics(train_roots)
    model.set_action_statistics(mean, std)
    model.to(device)
    sampler = TrajectorySampler(trajectory_cache)
    mixed_indices = [i for i, entry in enumerate(train_roots.entries)
                     if entry["mixed"]]
    if len(mixed_indices) != audit["train"]["mixed_roots"]:
        raise ValueError("mixed-root cache disagrees with frozen outcomes")

    config = {
        "seed": SEED, "device": "mps", "pretrain_updates": args.pretrain_updates,
        "model_scale": args.model_scale,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "skip_pretrain": args.skip_pretrain,
        "finetune_updates": args.finetune_updates,
        "trajectory_batch": args.trajectory_batch, "root_batch": args.root_batch,
        "pretrain_lr": 1e-4, "finetune_lr": 5e-5,
        "pairwise_weight": 0.2, "pairwise_temperature": 0.1,
        "trajectory_replay_weight": 0.25,
        "root_sampling": "half mixed, half uniform",
        "trajectory_sampling": "uniform root, uniform among 3 trajectories, uniform window",
        "objective": "binary MC success, then BCE plus within-root logistic ranking",
        "snapshot_digest": snapshot["snapshot_digest"],
    }
    experiment_name = {
        "original": EXPERIMENT,
        "small": "smolvla-q10-small-trajectory-pretrain-root-rank-v1",
        "large": "smolvla-q10-large-trajectory-pretrain-root-rank-v1",
    }[args.model_scale]
    checkpoint_dir = root / "checkpoints" / experiment_name / snapshot["snapshot_digest"]
    checkpoint_dir = checkpoint_dir / ("no_pretrain_control" if args.skip_pretrain
                                       else "trajectory_pretrained")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = root / "mlflow"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    db_path = tracking_dir / "tracking.db"
    mlflow.set_tracking_uri("sqlite:///" + str(db_path))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=(tracking_dir / "artifacts").as_uri())
    else:
        experiment_id = experiment.experiment_id
    baseline_metrics = None
    if BASELINE.exists():
        saved = torch.load(BASELINE, map_location="cpu", weights_only=False)
        if saved.get("snapshot_digest") != snapshot["snapshot_digest"]:
            raise ValueError("baseline checkpoint snapshot differs")
        baseline_architecture = dict(saved["architecture"])
        baseline_config = QPlanningModelConfig(**{
            key: value for key, value in baseline_architecture.items()
            if key not in ("prefix_dim", "robot_dim", "proprio_dim")})
        baseline_model = QPlanningCritic(
            prefix_dim=first["prefix"].shape[-1],
            robot_dim=len(first["robot"]), proprio_dim=len(first["proprio"]),
            config=baseline_config)
        baseline_model.load_state_dict(saved["model"])
        baseline_model.to(device)
        baseline_metrics = evaluate_roots(baseline_model, validation, device)
        baseline_metrics.update(_score_spread(baseline_model, validation, device))
        del baseline_model

    run_name = ("no-pretrain-root-ranking-control" if args.skip_pretrain
                else "trajectory-pretrain-then-root-ranking")
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_url = (f"http://127.0.0.1:5000/#/experiments/{experiment_id}"
                   f"/runs/{run.info.run_id}")
        print({"mlflow_run": run_url, "run_id": run.info.run_id}, flush=True)
        mlflow.log_params(config)
        mlflow.set_tags({"data_snapshot": snapshot["snapshot_digest"],
                         "training_host": "local-mac-mps",
                         "split": "frozen-tree-group-v1"})
        mlflow.log_dict(audit, "data_audit.json")
        if baseline_metrics is not None:
            mlflow.log_metrics({f"baseline/{key}": float(value)
                                for key, value in baseline_metrics.items()})
        started = time.perf_counter()
        stage_offset = 0 if args.skip_pretrain else args.pretrain_updates
        stage1_path = None
        if not args.skip_pretrain:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            for step in range(1, args.pretrain_updates + 1):
                model.train()
                _set_lr(optimizer, _learning_rate(step, args.pretrain_updates, 1e-4))
                rng = np.random.default_rng(SEED * 1_000_003 + step)
                batch = _to(single_batch(
                    sampler.sample(rng, args.trajectory_batch)), device)
                optimizer.zero_grad(set_to_none=True)
                scores = score_single(model, batch)
                loss = F.binary_cross_entropy(scores, batch["success"])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % 100 == 0 or step == args.pretrain_updates:
                    metrics = {"pretrain/loss": float(loss.detach()),
                               "pretrain/grad_norm": float(norm.detach()),
                               "pretrain/lr": optimizer.param_groups[0]["lr"]}
                    mlflow.log_metrics(metrics, step=step)
                    print({"stage": "pretrain", "update": step,
                           **metrics, "elapsed_minutes": round((time.perf_counter()-started)/60, 1)},
                          flush=True)
            stage1_path = checkpoint_dir / "pretrained.pt"
            _save_checkpoint(stage1_path, model, stage="pretrain",
                             update=args.pretrain_updates,
                             digest=snapshot["snapshot_digest"], audit=audit, config=config)
        stage1_metrics = evaluate_roots(model, validation, device)
        stage1_metrics.update(_score_spread(model, validation, device))
        prefix = "initial_val" if args.skip_pretrain else "stage1_val"
        mlflow.log_metrics({f"{prefix}/{key}": float(value)
                            for key, value in stage1_metrics.items()},
                           step=stage_offset)
        print({"stage1_validation": stage1_metrics}, flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        for step in range(1, args.finetune_updates + 1):
            model.train()
            _set_lr(optimizer, _learning_rate(step, args.finetune_updates, 5e-5))
            rng = np.random.default_rng(SEED * 2_000_003 + step)
            half = args.root_batch // 2
            indices = np.concatenate([
                rng.choice(mixed_indices, size=half, replace=len(mixed_indices) < half),
                rng.choice(len(train_roots), size=half, replace=len(train_roots) < half),
            ])
            root_batch = _to(_root_batch([train_roots[int(i)] for i in indices]), device)
            replay_batch = _to(single_batch(sampler.sample(rng, 8)), device)
            optimizer.zero_grad(set_to_none=True)
            scores = _root_scores(model, root_batch).float().clamp(1e-5, 1 - 1e-5)
            root_bce = F.binary_cross_entropy(scores, root_batch["success"].float())
            pair = pairwise_root_loss(scores, root_batch["success"], temperature=0.1)
            replay = F.binary_cross_entropy(
                score_single(model, replay_batch), replay_batch["success"])
            loss = root_bce + 0.2 * pair + 0.25 * replay
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 100 == 0 or step == args.finetune_updates:
                metrics = {
                    "finetune/loss": float(loss.detach()),
                    "finetune/root_bce": float(root_bce.detach()),
                    "finetune/pairwise": float(pair.detach()),
                    "finetune/replay_bce": float(replay.detach()),
                    "finetune/grad_norm": float(norm.detach()),
                    "finetune/lr": optimizer.param_groups[0]["lr"],
                }
                mlflow.log_metrics(metrics, step=stage_offset + step)
                print({"stage": "finetune", "update": step, **metrics,
                       "elapsed_minutes": round((time.perf_counter()-started)/60, 1)},
                      flush=True)
            if step % 250 == 0 or step == args.finetune_updates:
                metrics = evaluate_roots(model, validation, device)
                metrics.update(_score_spread(model, validation, device))
                mlflow.log_metrics({f"val/{key}": float(value)
                                    for key, value in metrics.items()},
                                   step=stage_offset + step)
                print({"validation_update": step, "metrics": metrics}, flush=True)
                # Subsequent runs keep a weight snapshot at each validation
                # point so score trajectories can be inspected, not just plots.
                _save_checkpoint(
                    checkpoint_dir / f"finetune_step_{step:04d}.pt", model,
                    stage="finetune", update=step,
                    digest=snapshot["snapshot_digest"], audit=audit, config=config)
        final_path = checkpoint_dir / "finetuned.pt"
        _save_checkpoint(final_path, model, stage="finetune",
                         update=args.finetune_updates,
                         digest=snapshot["snapshot_digest"], audit=audit, config=config)
        final_metrics = evaluate_roots(model, validation, device)
        final_metrics.update(_score_spread(model, validation, device))
        by_cohort = {}
        for cohort in SOURCE_EXPERIMENTS:
            ids = [gid for gid in root_cache["validation_group_ids"]
                   if groups[gid]["source_experiment"] == cohort]
            cohort_roots = TimedRoots(root_cache, ids, groups)
            by_cohort[cohort] = evaluate_roots(model, cohort_roots, device)
            by_cohort[cohort].update(_score_spread(model, cohort_roots, device))
            cohort_label = "idx10_29" if "idx30-49" not in cohort else "idx30_49"
            mlflow.log_metrics({
                f"cohort_{cohort_label}/{key}": float(value)
                for key, value in by_cohort[cohort].items()
            }, step=stage_offset + args.finetune_updates)
        report = {"run_url": run_url, "snapshot_digest": snapshot["snapshot_digest"],
                  "baseline": baseline_metrics, "stage1": stage1_metrics,
                  "stage2": final_metrics, "stage2_by_cohort": by_cohort,
                  "data_audit": audit,
                  "pretrained_checkpoint": str(stage1_path) if stage1_path else None,
                  "finetuned_checkpoint": str(final_path)}
        report_path = checkpoint_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(final_path), artifact_path="checkpoints")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
