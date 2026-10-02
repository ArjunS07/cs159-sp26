"""Authorized bounded queue: anchored CNN MC +8k, then recorded-policy TD(n=5) +8k.

The jobs use different architectures and data; they are not a matched MC/TD comparison.
Only existing local data/checkpoints are read. No remote writes or new data collection.
"""
from __future__ import annotations

import argparse
import fcntl
import copy
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_smolvla_anchored_local import (
    DIGEST, cpu_tree, gradient_diagnostics, load_data, log_metrics,
    pair_accuracy, root_positions, root_scores, scoring_metrics,
)
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_anchored_critic import AnchoredCritic
from pnp.smolvla_scalar_returns import FORMAT as TRAJECTORY_FORMAT, ScalarCritic, ReturnSampler, logits
from pnp.smolvla_tree_bellman_finetune import _atomic_json, _root_batch, _to

FORMAT = "smolvla_local_continuation_v1"
JOBS = ("mc_anchored_cnn", "td_n5")
EXTENSION_UPDATES = 8000


def extension_lr(step):
    if not 0 <= step <= EXTENSION_UPDATES:
        raise ValueError("extension step outside the fixed budget")
    return 1e-5 + (1e-4 - 1e-5) * .5 * (1 + math.cos(math.pi * step / EXTENSION_UPDATES))


def scalar_from_architecture(architecture):
    arch = dict(architecture)
    dims = {k: arch.pop(k) for k in ("prefix_dim", "robot_dim", "proprio_dim")}
    return ScalarCritic(**dims, config=QPlanningModelConfig(**arch))


def scalar_root_scores(model, batch):
    repeat = lambda x: x.repeat_interleave(9, 0)
    return model(repeat(batch["prefix"]), repeat(batch["pad"]), repeat(batch["robot"]),
                 repeat(batch["proprio"]), batch["action"].flatten(0, 1),
                 batch["action_valid"].flatten(0, 1)).reshape(-1, 9)


@torch.no_grad()
def evaluate(model, roots, device, anchored):
    model.eval()
    values, labels = [], []
    for start in range(0, len(roots), 4):
        batch = _to(_root_batch([roots[i] for i in range(start, min(start + 4, len(roots)))]), device)
        values.extend((root_scores(model, batch) if anchored else scalar_root_scores(model, batch)).cpu().numpy())
        labels.extend(batch["success"].cpu().numpy())
    metrics = scoring_metrics(values, labels)
    # For scalar TD the stock term can change; only anchored CNN has a frozen stock component.
    metrics["stock_bce_component"] = metrics.pop("shared_stock_bce_component")
    return metrics


def scalar_gradients(model, roots, device):
    model.eval()
    rows = []
    for start in range(0, len(roots), 4):
        batch = _to(_root_batch([roots[i] for i in range(start, min(start + 4, len(roots)))]), device)
        stock = batch["action"][:, 0].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            z = model(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"], stock,
                      batch["action_valid"][:, 0]).squeeze(-1)
            gradient, = torch.autograd.grad(z.sum(), stock)
        with torch.no_grad():
            values = scalar_root_scores(model, batch).cpu().numpy()
            scaled = gradient * model.action_std * batch["action_valid"][:, 0, :, None]
            delta = (batch["action"] - stock.detach()[:, None]) / model.action_std
            projection = (scaled[:, None] * delta).sum((-1, -2)).cpu().numpy()
            norms = scaled.square().sum((-1, -2)).sqrt().cpu().numpy()
            outcomes = batch["success"].cpu().numpy().astype(bool)
        for j, y in enumerate(outcomes):
            rescue = projection[j, 1:][y[1:]] if not y[0] else np.empty(0)
            spoil = projection[j, 1:][~y[1:]] if y[0] else np.empty(0)
            rows.append({"group_id": roots.entries[start + j]["candidate_group_id"], "labels": y.tolist(),
                         "logits": values[j].tolist(), "gradient_projection": projection[j].tolist(),
                         "normalized_gradient_norm": float(norms[j]),
                         "gradient_pair_accuracy": pair_accuracy(projection[j], y),
                         "fresh_gradient_pair_accuracy": pair_accuracy(projection[j, 1:], y[1:]),
                         "rescue_candidates": len(rescue), "rescue_positive": int((rescue > 0).sum()),
                         "rescue_zero": int((rescue == 0).sum()), "spoil_candidates": len(spoil),
                         "spoil_negative": int((spoil < 0).sum()), "spoil_zero": int((spoil == 0).sum())})
    counts = {k: sum(r[k] for r in rows) for k in ("rescue_candidates", "rescue_positive", "rescue_zero", "spoil_candidates", "spoil_negative", "spoil_zero")}
    pairs = [r["gradient_pair_accuracy"] for r in rows if r["gradient_pair_accuracy"] is not None]
    fresh = [r["fresh_gradient_pair_accuracy"] for r in rows if r["fresh_gradient_pair_accuracy"] is not None]
    summary = {**counts, "roots": len(rows), "mixed_roots": len(pairs), "fresh_mixed_roots": len(fresh),
               "gradient_macro_pair_accuracy": float(np.mean(pairs)) if pairs else None,
               "fresh_gradient_macro_pair_accuracy": float(np.mean(fresh)) if fresh else None,
               "median_normalized_gradient_norm": float(np.median([r["normalized_gradient_norm"] for r in rows])),
               "near_zero_gradient_roots": sum(r["normalized_gradient_norm"] < 1e-12 for r in rows),
               "rescue_positive_fraction": counts["rescue_positive"] / counts["rescue_candidates"] if counts["rescue_candidates"] else None,
               "spoil_negative_fraction": counts["spoil_negative"] / counts["spoil_candidates"] if counts["spoil_candidates"] else None}
    return {"summary": summary, "root_records": rows,
            "interpretation": "Original scalar candidate derivative at stock; directional outcome diagnostic, not gradient ground truth."}


def save(path, model, target, optimizer, extension_step, contract, history, run_id, device):
    temporary = path.with_suffix(".tmp")
    torch.save({"format": FORMAT, "update": contract["initializer_update"] + extension_step,
                "extension_update": extension_step, "config": contract, "snapshot_digest": DIGEST,
                "architecture": model.architecture_config(), "model": cpu_tree(model.state_dict()),
                "target": cpu_tree(target.state_dict()) if target is not None else None,
                "optimizer": cpu_tree(optimizer.state_dict()), "history": history, "mlflow_run_id": run_id,
                "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, temporary)
    temporary.replace(path)


def restore(saved, model, target, optimizer, contract, device):
    if saved["format"] != FORMAT or saved["config"] != contract or saved["snapshot_digest"] != DIGEST:
        raise ValueError("continuation recovery contract differs")
    if saved["architecture"] != model.architecture_config() or not 0 <= saved["extension_update"] <= EXTENSION_UPDATES:
        raise ValueError("continuation architecture/budget differs")
    if saved["update"] != contract["initializer_update"] + saved["extension_update"]:
        raise ValueError("continuation counter provenance differs")
    model.load_state_dict(saved["model"])
    if target is not None:
        target.load_state_dict(saved["target"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "mps":
        torch.mps.set_rng_state(saved["mps_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])


def prepare_job(job, run_root, roots, device):
    anchored = job == "mc_anchored_cnn"
    source = (run_root / "checkpoints/smolvla_anchored_cnn_v1" / DIGEST / "temporal_cnn/latest.pt" if anchored else
              run_root / "checkpoints/smolvla-overnight-20260930-td_n5" / DIGEST / "td/checkpoint_step_004000.pt")
    initial = torch.load(source, map_location="cpu", weights_only=False)
    origin = 2000 if anchored else 4000
    if initial["snapshot_digest"] != DIGEST or initial["update"] != origin:
        raise ValueError("unexpected continuation initializer snapshot/update")
    if anchored:
        arch = initial["architecture"]
        if arch["model_family"] != "anchored_temporal_cnn" or arch["delta_scale"] != 100 or initial["config"]["family"] != "temporal_cnn":
            raise ValueError("unexpected anchored initializer architecture")
        model = AnchoredCritic(scalar_from_architecture(arch["base_architecture"]), "temporal_cnn",
                               width=arch["width"], delta_scale=arch["delta_scale"]).to(device)
        target, sampler, trajectory_sha = None, None, None
    else:
        old_config = initial["config"]
        if initial["arm"] != "td" or old_config["td_steps"] != 5 or old_config["root_fraction"] != 0 or old_config["architecture"] != "decoder" or old_config["prefix_dropout"] != 0:
            raise ValueError("expected unchanged recorded-policy n=5 scalar TD initializer")
        model = scalar_from_architecture(initial["architecture"]).to(device)
        target = copy.deepcopy(model).eval().requires_grad_(False)
        target.load_state_dict(initial["target"])
        index_path = run_root / "cache" / TRAJECTORY_FORMAT / DIGEST / "cache_index.json"
        index = json.loads(index_path.read_text())
        index["cache_dir"] = str(index_path.parent)
        train_ids = {e["candidate_group_id"] for e in roots["train"].entries}
        if index["format"] != TRAJECTORY_FORMAT or index["snapshot_digest"] != DIGEST or set(index["train_group_ids"]) != train_ids:
            raise ValueError("TD trajectory cache contract differs")
        if any(e["group_id"] not in train_ids or e["windows"] < 1 or not (index_path.parent / e["file"]).is_file() for e in index["entries"]):
            raise ValueError("TD cache is incomplete or includes held-out trajectories")
        if {e["group_id"] for e in index["entries"]} != train_ids:
            raise ValueError("TD cache omits training roots")
        sampler = ReturnSampler(index)
        trajectory_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    optimizer.load_state_dict(initial["optimizer"])
    contract = {"job": job, "extension_updates": EXTENSION_UPDATES, "initializer": str(source),
                "initializer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "initializer_update": origin, "fixed_final_parent_counter": origin + EXTENSION_UPDATES,
                "original_scalar_updates_before_overnight": 0 if anchored else 2000,
                "total_learned_residual_updates" if anchored else "total_scalar_updates_including_original": 10000 if anchored else 14000,
                "snapshot_digest": DIGEST, "architecture": model.architecture_config(), "optimizer": "restored AdamW moments",
                "lr_start": 1e-4, "lr_end": 1e-5, "lr_schedule": "cosine over 8000 new updates; no warmup",
                "weight_decay": 1e-4, "clip_norm": 1., "seed": 42,
                "batch_rng_formula": "42 * 4000003 + initializer_update + extension_update",
                "objective": "ordinary all-nine root MC BCE" if anchored else "recorded-policy five-step soft-target TD BCE",
                "sampling": "uniform root without replacement, all nine candidates" if anchored else "uniform root/trajectory/window",
                "root_batch": 4 if anchored else None, "candidate_batch": 36 if anchored else 32,
                "gamma": 1, "td_steps": None if anchored else 5, "target_rate": None if anchored else .005,
                "trajectory_index_sha256": trajectory_sha, "checkpoint_interval": 100 if anchored else 50,
                "evaluation_interval": 1000, "device": device.type,
                "mean_additional_training_root_presentations": 8000 * (4 if anchored else 32) / 1280,
                "train_root_ids": [e["candidate_group_id"] for e in roots["train"].entries],
                "validation_root_ids": [e["candidate_group_id"] for e in roots["validation"].entries],
                "comparison_caveat": "different architectures and training data; not a matched MC-versus-TD comparison"}
    return model, target, optimizer, sampler, contract


def run_job(job, run_root, output, roots, device, experiment_id):
    anchored = job == "mc_anchored_cnn"
    model, target, optimizer, sampler, contract = prepare_job(job, run_root, roots, device)
    directory = output / job
    directory.mkdir(exist_ok=True)
    contract_path = directory / "experiment_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("existing continuation output contract differs")
    _atomic_json(contract_path, contract)
    latest = directory / "latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    history, start = (saved["history"], saved["extension_update"]) if saved else ([], 0)
    if saved:
        restore(saved, model, target, optimizer, contract, device)
    started = time.monotonic()
    with mlflow.start_run(experiment_id=experiment_id, run_name=job, run_id=saved["mlflow_run_id"] if saved else None) as run:
        if not saved:
            mlflow.log_params({k: v for k, v in contract.items() if k not in ("architecture", "train_root_ids", "validation_root_ids")})
            mlflow.log_artifact(str(contract_path))
            for split in ("train", "validation"):
                metrics = evaluate(model, roots[split], device, anchored)
                history.append({"extension_update": 0, "update": contract["initializer_update"], "split": split, **metrics})
                log_metrics(metrics, split, contract["initializer_update"])
                print({"job": job, "extension_update": 0, "split": split, **metrics}, flush=True)
            save(latest, model, target, optimizer, 0, contract, history, run.info.run_id, device)
        print({"job": job, "resumed_extension_update": start,
               "mlflow_run": f"http://127.0.0.1:5001/#/experiments/{experiment_id}/runs/{run.info.run_id}",
               "fixed_budget": contract["fixed_final_parent_counter"]}, flush=True)
        for extension_step in range(start + 1, EXTENSION_UPDATES + 1):
            counter = contract["initializer_update"] + extension_step
            torch.manual_seed(42 * 4000003 + counter)
            rng = np.random.default_rng(42 * 4000003 + counter)
            model.train()
            if anchored:
                positions = root_positions(counter, len(roots["train"]))
                batch = _to(_root_batch([roots["train"][int(i)] for i in positions]), device)
                prediction, y = root_scores(model, batch), batch["success"].float()
            else:
                batch = _to(sampler.batch(rng, 32, nstep=5), device)
                with torch.no_grad():
                    y = batch["reward"] + batch["discount"] * logits(target, batch, "next_").sigmoid()
                if not torch.isfinite(y).all() or torch.any((y < 0) | (y > 1)):
                    raise ValueError("TD target outside success-probability range")
                prediction = logits(model, batch)
            lr = extension_lr(extension_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite continuation loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
            optimizer.step()
            if target is not None:
                with torch.no_grad():
                    for tp, p in zip(target.parameters(), model.parameters()):
                        tp.lerp_(p, .005)
            if extension_step % 100 == 0:
                log_metrics({"loss": float(loss.detach()), "lr": lr, "gradient_norm": float(grad_norm)}, "minibatch", counter)
                print({"job": job, "extension_update": extension_step, "parent_counter": counter,
                       "loss": float(loss.detach()), "lr": lr, "elapsed_minutes": round((time.monotonic() - started) / 60, 2)}, flush=True)
            if extension_step % 1000 == 0:
                for split in ("train", "validation"):
                    metrics = evaluate(model, roots[split], device, anchored)
                    history.append({"extension_update": extension_step, "update": counter, "split": split, **metrics})
                    log_metrics(metrics, split, counter)
                    print({"job": job, "extension_update": extension_step, "split": split, **metrics}, flush=True)
            if extension_step % contract["checkpoint_interval"] == 0:
                save(latest, model, target, optimizer, extension_step, contract, history, run.info.run_id, device)
                _atomic_json(directory / "history.json", history)
        report = {"config": contract, "extension_update": 8000, "update": contract["fixed_final_parent_counter"],
                  "mlflow_run_id": run.info.run_id, "splits": {}}
        for split in ("train", "validation"):
            metrics = evaluate(model, roots[split], device, anchored)
            diagnostics = (gradient_diagnostics(model, roots[split], device, model.action_std) if anchored else
                           scalar_gradients(model, roots[split], device))
            report["splits"][split] = {"metrics": metrics, "gradients": diagnostics}
            log_metrics(metrics, "final/" + split, report["update"])
            log_metrics(diagnostics["summary"], "gradient/" + split, report["update"])
            print({"job": job, "fixed_final_extension_update": 8000, "split": split,
                   "metrics": metrics, "gradients": diagnostics["summary"]}, flush=True)
        _atomic_json(directory / "final_report.json", report)
        mlflow.log_artifact(str(directory / "final_report.json"))
        print({"job": job, "complete": str(directory / "final_report.json")}, flush=True)
    return directory / "final_report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path.home() / "pnp-vla-runs")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--job", choices=(*JOBS, "both"), default="both")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(4)
    run_root = args.run_root.expanduser().resolve()
    (run_root / "diagnostics").mkdir(parents=True, exist_ok=True)
    # Retain the handle through both jobs. A duplicate invocation must not train
    # another Metal worker or overwrite the same checkpoint/MLflow run.
    lock = (run_root / "diagnostics/local_continuation_v1.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("This continuation queue is already running") from error
    output = run_root / "checkpoints" / FORMAT / DIGEST
    output.mkdir(parents=True, exist_ok=True)
    # Existing local manifest is mandatory: this continuation does not download data.
    manifest_dir = run_root / "checkpoints/smolvla_anchored_cnn_v1" / DIGEST
    if not (manifest_dir / "snapshot.json").is_file():
        raise FileNotFoundError("existing local combined manifest is required")
    roots = load_data(run_root, manifest_dir)
    jobs = JOBS if args.job == "both" else (args.job,)
    if args.prepare_only:
        for job in jobs:
            _, _, _, _, contract = prepare_job(job, run_root, roots, device)
            print({"prepared": job, "contract": contract}, flush=True)
        return
    mlflow.set_tracking_uri("sqlite:///" + str(run_root / "mlflow/tracking.db"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(FORMAT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(FORMAT, artifact_location=(run_root / "mlflow/artifacts").as_uri())
    status = []
    for job in jobs:
        print({"starting": job, "fixed_new_updates": 8000, "queue_order": jobs}, flush=True)
        try:
            report = run_job(job, run_root, output, roots, device, experiment_id)
        except Exception as error:
            status.append({"job": job, "status": "failed", "error": repr(error)})
            _atomic_json(output / "queue_status.json", status)
            raise
        status.append({"job": job, "status": "finished", "report": str(report)})
        _atomic_json(output / "queue_status.json", status)
    print({"queue_complete": status}, flush=True)


if __name__ == "__main__":
    main()
