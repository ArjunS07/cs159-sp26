"""Run matched scalar MC and fixed-policy one-step TD on the Mac GPU."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_smolvla_combined_local import _load_local_credentials
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_combined_success import load_or_create_combined_snapshot, prepare_combined_root_cache
from pnp.smolvla_scalar_returns import FORMAT, ScalarCritic, LateFusionScalarCritic, ReturnSampler, prepare_cache, logits
from pnp.smolvla_success_critic import TimedRoots, action_statistics, evaluate_roots
from pnp.smolvla_tree_bellman_finetune import _to, _root_batch
from pnp.store import SupabaseStore


def _contract(config):
    defaults = {"root_fraction": 0., "td_steps": 1, "prefix_dropout": 0., "architecture": "decoder", "reset_optimizer": False}
    return {k: v for k, v in {**defaults, **config}.items()
            if k not in {"workers", "arms", "prepare_only", "eval_interval", "checkpoint_interval", "init_from", "experiment_name"}}


def save_recovery(path, model, target, optimizer, *, arm, step, digest, config, history, run_id):
    temporary = path.with_suffix(".tmp")
    payload = {"format": FORMAT, "arm": arm, "update": step,
               "architecture": model.architecture_config(), "scalar_head": True,
               "snapshot_digest": digest,
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "target": {k: v.detach().cpu() for k, v in target.state_dict().items()},
               "optimizer": optimizer.state_dict(), "config": config, "history": history,
               "mlflow_run_id": run_id, "evaluator_version": "scalar_expected_value_v1",
               "torch_rng": torch.get_rng_state(),
               "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_recovery(path, model, target, optimizer, *, arm, digest, config):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["format"] != FORMAT or saved["arm"] != arm
            or saved["snapshot_digest"] != digest
            or saved["architecture"] != model.architecture_config()
            or _contract(saved["config"]) != _contract(config)):
        raise ValueError("resume checkpoint/data/training configuration mismatch")
    model.load_state_dict(saved["model"])
    target.load_state_dict(saved["target"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    if saved.get("mps_rng") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(saved["mps_rng"])
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-rate", type=float, default=.005)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=("mc", "td"), default=["mc", "td"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--experiment-name", default="smolvla-q10-scalar-mc-td-v1-preaction")
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--root-fraction", type=float, default=0.)
    parser.add_argument("--td-steps", type=int, choices=(1, 5, 10), default=1)
    parser.add_argument("--prefix-dropout", type=float, default=0.)
    parser.add_argument("--architecture", choices=("decoder", "late_fusion"), default="decoder")
    args = parser.parse_args()
    if "/" in args.experiment_name or not args.experiment_name or not 0 <= args.root_fraction <= 1 or not 0 <= args.prefix_dropout < 1:
        parser.error("invalid experiment name or sampling/augmentation fraction")
    if args.root_fraction and "td" in args.arms:
        parser.error("all-nine root targets apply only to MC")
    if min(args.updates, args.batch_size, args.eval_interval, args.checkpoint_interval, args.workers) < 1 or not 0 < args.target_rate <= 1 or args.lr <= 0:
        parser.error("positive budgets and valid target rate required")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(4)
    _load_local_credentials()
    root = Path.home() / "pnp-vla-runs"
    store = SupabaseStore()
    snapshot = load_or_create_combined_snapshot(store)
    cache = prepare_combined_root_cache(snapshot=snapshot, cache_root=root / "cache", store=store)
    groups = {g["candidate_group_id"]: g for g in snapshot["groups"]}
    train = TimedRoots(cache, cache["train_group_ids"], groups)
    validation = TimedRoots(cache, cache["validation_group_ids"], groups)
    mlflow.set_tracking_uri("sqlite:///" + str(root / "mlflow/tracking.db"))
    client = MlflowClient()
    name = args.experiment_name
    experiment = client.get_experiment_by_name(name)
    exp_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(root / "mlflow/artifacts").as_uri())
    output = root / "checkpoints" / name / snapshot["snapshot_digest"]
    output.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(experiment_id=exp_id, run_name="prepare-matched-return-data") as prep:
        print({"mlflow_data_run": f"http://127.0.0.1:5001/#/experiments/{exp_id}/runs/{prep.info.run_id}"}, flush=True)
        mlflow.log_params({"snapshot_digest": snapshot["snapshot_digest"], "cache_format": FORMAT,
                           "train_roots": len(train), "validation_roots": len(validation)})
        index = prepare_cache(cache, train, root / "cache", store, args.workers)
        audit = {"trajectories": len(index["entries"]), "windows": sum(e["windows"] for e in index["entries"]),
                 "kinds": sorted({e["kind"] for e in index["entries"]}),
                 "sampling": "trajectory part: uniform root/trajectory/window; optional complete nine-candidate root batches",
                 "root_examples_per_batch": 9 * min(args.batch_size // 9, int(round(args.root_fraction * args.batch_size / 9))),
                 "inputs": "full proposed action; mask only known remaining budget",
                 "split": "1280 training roots / 320 held-out roots; no validation trajectories"}
        (output / "data_audit.json").write_text(json.dumps(audit, indent=2))
        mlflow.log_artifact(str(output / "data_audit.json"))
        mlflow.log_metrics({"data/windows": audit["windows"], "data/trajectories": audit["trajectories"]})
        print({"data_audit": audit}, flush=True)
    if args.prepare_only:
        return
    mean, std = action_statistics(train)
    first = train[0]
    config = QPlanningModelConfig(action_horizon=10, action_dim=7, width=128, n_layers=2,
                                  n_heads=4, ffn_width=512, dropout=0, n_bins=2, prefix_pool_tokens=128)
    device = torch.device("mps")
    for arm in args.arms:
        torch.manual_seed(42)
        model_class = LateFusionScalarCritic if args.architecture == "late_fusion" else ScalarCritic
        model = model_class(prefix_dim=first["prefix"].shape[-1], robot_dim=len(first["robot"]),
                             proprio_dim=len(first["proprio"]), config=config)
        model.set_action_statistics(mean, std)
        model.to(device)
        target = copy.deepcopy(model).eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        if args.init_from:
            initial = torch.load(args.init_from, map_location="cpu", weights_only=False)
            if initial["snapshot_digest"] != snapshot["snapshot_digest"] or initial["arm"] != arm:
                raise ValueError("initializer snapshot/objective mismatch")
            missing, unexpected = model.load_state_dict(initial["model"], strict=False)
            if unexpected or any(not k.startswith(("late_action.", "late_head.")) for k in missing):
                raise ValueError("incompatible initializer")
            target.load_state_dict(model.state_dict())
            if args.architecture == "decoder":
                target.load_state_dict(initial["target"])
            if not args.reset_optimizer:
                optimizer.load_state_dict(initial["optimizer"])
        sampler = ReturnSampler(index)
        directory = output / arm
        directory.mkdir(exist_ok=True)
        history = []
        latest = directory / "latest.pt"
        saved = load_recovery(latest, model, target, optimizer, arm=arm,
                              digest=snapshot["snapshot_digest"], config=vars(args)) if latest.exists() else None
        if saved:
            history = saved["history"]
            print({"arm": arm, "resumed_update": saved["update"]}, flush=True)
            if saved["update"] == args.updates:
                print({"arm": arm, "status": "already complete; skipped"}, flush=True)
                continue
        started = time.monotonic()
        with mlflow.start_run(experiment_id=exp_id, run_name=f"scalar-{arm}-seed42",
                              run_id=saved["mlflow_run_id"] if saved else None) as run:
            url = f"http://127.0.0.1:5001/#/experiments/{exp_id}/runs/{run.info.run_id}"
            if not saved:
                mlflow.log_params({**vars(args), "arm": arm, "seed": 42, "gamma": 1,
                               "loss": "BCE with MC binary / TD detached soft probability target",
                               "model_description": "2-layer width128 scalar sigmoid, normalized proposals; optional direct late fusion",
                               "trainable_parameters": sum(p.numel() for p in model.parameters()),
                               "cache_format": FORMAT, "snapshot_digest": snapshot["snapshot_digest"],
                               "pairwise_loss": False, "td_backup": "recorded next proposed action, no max"})
            print({"arm": arm, "mlflow_run": url}, flush=True)
            if saved and saved.get("evaluator_version") != "scalar_expected_value_v1":
                evaluator = copy.deepcopy(model).eval()
                for row in history:
                    path = directory / f"checkpoint_step_{row['update']:06d}.pt"
                    previous = torch.load(path, map_location="cpu", weights_only=False)
                    evaluator.load_state_dict(previous["model"])
                    with torch.no_grad():
                        row["validation"] = evaluate_roots(evaluator, validation, device)
                    mlflow.log_metrics({"val/" + k: v for k, v in row["validation"].items()}, step=row["update"])
                    print({"arm": arm, "corrected_validation": row}, flush=True)
                mlflow.set_tag("validation_repaired", "scalar sigmoid replaces categorical decoding; previous metrics superseded")
                save_recovery(latest, model, target, optimizer, arm=arm, step=saved["update"],
                              digest=snapshot["snapshot_digest"], config=vars(args), history=history,
                              run_id=run.info.run_id)
                (directory / "report.json").write_text(json.dumps(
                    {"mlflow_run": url, "arm": arm, "history": history, "data_audit": audit}, indent=2))
            for step in range(saved["update"] + 1 if saved else 0, args.updates + 1):
                if step:
                    model.train()
                    # Exactly matched minibatches across arms, independent of evaluation.
                    rng = np.random.default_rng(42 * 4000003 + step)
                    root_groups = min(args.batch_size // 9, int(round(args.root_fraction * args.batch_size / 9)))
                    root_examples = 9 * root_groups
                    batch = _to(sampler.batch(rng, args.batch_size - root_examples, args.td_steps), device) if args.batch_size > root_examples else None
                    root_batch = _to(_root_batch([train[int(i)] for i in rng.integers(0, len(train), size=root_groups)]), device) if root_groups else None
                    if args.prefix_dropout:
                        for current in (batch, root_batch):
                            if current is not None:
                                kept = torch.rand_like(current["pad"].float()) >= args.prefix_dropout
                                kept[:, 0] = True
                                current["pad"] = current["pad"] & kept
                    if arm == "mc":
                        y = batch["success"].float() if batch is not None else None
                    else:
                        with torch.no_grad():
                            y = batch["reward"] + batch["discount"] * logits(target, batch, "next_").sigmoid()
                    warmup = min(100, args.updates)
                    lr = args.lr * (step / warmup if step <= warmup else
                                    .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.updates - warmup))))
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(logits(model, batch), y) * (args.batch_size - root_examples) / args.batch_size if batch is not None else torch.zeros((), device=device)
                    if root_batch is not None:
                        repeat = lambda x: x.repeat_interleave(9, 0)
                        flat = {key: repeat(root_batch[key]) for key in ("prefix", "pad", "robot", "proprio")}
                        flat.update(action=root_batch["action"].flatten(0, 1), action_valid=root_batch["action_valid"].flatten(0, 1))
                        loss = loss + F.binary_cross_entropy_with_logits(logits(model, flat), root_batch["success"].float().flatten()) * root_examples / args.batch_size
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    with torch.no_grad():
                        for tp, p in zip(target.parameters(), model.parameters()):
                            tp.lerp_(p, args.target_rate)
                    if step % 50 == 0:
                        combined_target_mean = ((float(y.sum()) if y is not None else 0.) + (float(root_batch["success"].float().sum()) if root_batch is not None else 0.)) / args.batch_size
                        mlflow.log_metrics({"train/loss": float(loss.detach()), "train/target_mean": combined_target_mean,
                                            "train/gradient_norm": float(grad_norm), "train/lr": lr}, step=step)
                        print({"arm": arm, "update": step, "loss": float(loss.detach()),
                               "elapsed_minutes": round((time.monotonic() - started) / 60, 1)}, flush=True)
                if step % args.eval_interval == 0 or step == args.updates:
                    with torch.no_grad():
                        metrics = evaluate_roots(model, validation, device)
                    history.append({"update": step, "validation": metrics})
                    mlflow.log_metrics({"val/" + k: v for k, v in metrics.items()}, step=step)
                    save_recovery(directory / f"checkpoint_step_{step:06d}.pt", model, target, optimizer,
                                  arm=arm, step=step, digest=snapshot["snapshot_digest"],
                                  config=vars(args), history=history, run_id=run.info.run_id)
                    report = {"mlflow_run": url, "arm": arm, "history": history, "data_audit": audit}
                    (directory / "report.json").write_text(json.dumps(report, indent=2))
                    print({"arm": arm, "update": step, "validation": metrics}, flush=True)
                if step % args.checkpoint_interval == 0 or step == args.updates:
                    save_recovery(latest, model, target, optimizer, arm=arm, step=step,
                                  digest=snapshot["snapshot_digest"], config=vars(args),
                                  history=history, run_id=run.info.run_id)
            mlflow.log_artifact(str(directory / "report.json"))


if __name__ == "__main__":
    main()
