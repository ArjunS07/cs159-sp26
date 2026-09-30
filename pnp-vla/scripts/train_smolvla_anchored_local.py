"""Matched anchored residual transformer/CNN root-MC runs on the frozen local cache.

Two fixed 2,000-update arms; no new data, ranking loss, or policy intervention.
Only the persisted combined manifest may be downloaded. Supabase is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_smolvla_combined_local import _load_local_credentials
from pnp.pcp_critic.resumable_snapshot import _download_with_retry
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_anchored_critic import AnchoredCritic
from pnp.smolvla_combined_success import COMBINED_SNAPSHOT_KEY, ROOT_CACHE_FORMAT
from pnp.smolvla_scalar_returns import ScalarCritic
from pnp.smolvla_success_critic import TimedRoots
from pnp.smolvla_tree_bellman_finetune import _atomic_json, _json_digest, _root_batch, _to
from pnp.store import SupabaseStore

DIGEST = "671b5b211099997fc83d1277"
FORMAT = "smolvla_anchored_cnn_v1"
FAMILIES = ("transformer", "temporal_cnn")


def root_positions(update, count):
    return np.random.default_rng(42 * 4000003 + update).choice(count, 4, replace=False)


def root_scores(model, batch):
    repeat = lambda value: value.repeat_interleave(9, 0)
    reference = batch["action"][:, 0].detach()
    valid = batch["action_valid"]
    if not torch.equal(valid, valid[:, :1].expand_as(valid)):
        raise ValueError("candidate masks must share the known root action budget")
    features, baseline = model.reference_features(batch["prefix"], batch["pad"],
                                                  batch["robot"], batch["proprio"],
                                                  reference, valid[:, 0])
    return model.logits_from_features(repeat(features), repeat(baseline),
                                     batch["action"].flatten(0, 1), valid.flatten(0, 1),
                                     repeat(reference)).reshape(-1, 9)


def pair_accuracy(values, outcomes):
    positive, negative = values[outcomes], values[~outcomes]
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None]
    return float(((delta > 0).sum() + .5 * (delta == 0).sum()) / delta.size)


def scoring_metrics(logits, labels):
    logits = np.asarray(logits, np.float64)
    labels = np.asarray(labels, bool)
    probabilities = torch.as_tensor(logits).sigmoid().numpy()
    stock = labels[:, 0]
    selected = labels[np.arange(len(labels)), logits.argmax(1)]
    macro = [pair_accuracy(z, y) for z, y in zip(logits, labels)]
    fresh = [pair_accuracy(z[1:], y[1:]) for z, y in zip(logits, labels)]
    macro = [v for v in macro if v is not None]
    fresh = [v for v in fresh if v is not None]
    return {"bce": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
            "nonstock_bce": float(np.mean(np.logaddexp(0, logits[:, 1:]) - labels[:, 1:] * logits[:, 1:])),
            "shared_stock_bce_component": float(np.mean(np.logaddexp(0, logits[:, 0]) - labels[:, 0] * logits[:, 0]) / 9),
            "brier": float(np.square(probabilities - labels).mean()),
            "stock_anchor_repeated_bce": float(np.mean(np.logaddexp(0, logits[:, :1]) - labels * logits[:, :1])),
            "roots": len(labels), "mixed_roots": len(macro), "fresh_mixed_roots": len(fresh),
            "stock_successes": int(stock.sum()), "oracle_successes": int(labels.any(1).sum()),
            "selected_successes": int(selected.sum()), "rescues": int((~stock & selected).sum()),
            "spoils": int((stock & ~selected).sum()),
            "selected_minus_stock_pp": float(100 * (selected.mean() - stock.mean())),
            "macro_pair_accuracy": float(np.mean(macro)) if macro else None,
            "fresh_macro_pair_accuracy": float(np.mean(fresh)) if fresh else None,
            "within_root_logit_std": float(logits.std(1).mean())}


@torch.no_grad()
def evaluate(model, roots, device):
    model.eval()
    logits, labels = [], []
    for start in range(0, len(roots), 4):
        batch = _to(_root_batch([roots[i] for i in range(start, min(start + 4, len(roots)))]), device)
        logits.extend(root_scores(model, batch).cpu().numpy())
        labels.extend(batch["success"].cpu().numpy())
    return scoring_metrics(logits, labels)


def gradient_diagnostics(model, roots, device, action_std):
    """Candidate derivative at stock with the reference held fixed, then g dot delta."""
    model.eval()
    rows = []
    for start in range(0, len(roots), 4):
        batch = _to(_root_batch([roots[i] for i in range(start, min(start + 4, len(roots)))]), device)
        stock = batch["action"][:, 0].detach().clone().requires_grad_(True)
        reference = stock.detach().clone()
        with torch.enable_grad():
            z = model(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
                      stock, batch["action_valid"][:, 0], reference).squeeze(-1)
            gradient, = torch.autograd.grad(z.sum(), stock)
        with torch.no_grad():
            logits = root_scores(model, batch).cpu().numpy()
            mask = batch["action_valid"][:, 0, :, None]
            normalized_gradient = gradient * action_std * mask
            delta = (batch["action"] - reference[:, None]) / action_std
            projection = (normalized_gradient[:, None] * delta).sum((-1, -2)).cpu().numpy()
            norms = normalized_gradient.square().sum((-1, -2)).sqrt().cpu().numpy()
            outcomes = batch["success"].cpu().numpy().astype(bool)
        for j, labels in enumerate(outcomes):
            rescue = projection[j, 1:][labels[1:]] if not labels[0] else np.empty(0)
            spoil = projection[j, 1:][~labels[1:]] if labels[0] else np.empty(0)
            rows.append({"group_id": roots.entries[start + j]["candidate_group_id"],
                         "labels": labels.tolist(), "logits": logits[j].tolist(),
                         "gradient_projection": projection[j].tolist(),
                         "normalized_gradient_norm": float(norms[j]),
                         "gradient_pair_accuracy": pair_accuracy(projection[j], labels),
                         "fresh_gradient_pair_accuracy": pair_accuracy(projection[j, 1:], labels[1:]),
                         "rescue_candidates": int(len(rescue)), "rescue_positive": int((rescue > 0).sum()),
                         "rescue_zero": int((rescue == 0).sum()),
                         "spoil_candidates": int(len(spoil)), "spoil_negative": int((spoil < 0).sum()),
                         "spoil_zero": int((spoil == 0).sum())})
    counts = {k: sum(row[k] for row in rows) for k in
              ("rescue_candidates", "rescue_positive", "rescue_zero", "spoil_candidates", "spoil_negative", "spoil_zero")}
    pair = [r["gradient_pair_accuracy"] for r in rows if r["gradient_pair_accuracy"] is not None]
    fresh = [r["fresh_gradient_pair_accuracy"] for r in rows if r["fresh_gradient_pair_accuracy"] is not None]
    norms = [r["normalized_gradient_norm"] for r in rows]
    summary = {**counts, "roots": len(rows), "mixed_roots": len(pair), "fresh_mixed_roots": len(fresh),
               "gradient_macro_pair_accuracy": float(np.mean(pair)) if pair else None,
               "fresh_gradient_macro_pair_accuracy": float(np.mean(fresh)) if fresh else None,
               "median_normalized_gradient_norm": float(np.median(norms)),
               "near_zero_gradient_roots": int(sum(v < 1e-12 for v in norms)),
               "rescue_positive_fraction": counts["rescue_positive"] / counts["rescue_candidates"] if counts["rescue_candidates"] else None,
               "spoil_negative_fraction": counts["spoil_negative"] / counts["spoil_candidates"] if counts["spoil_candidates"] else None}
    return {"summary": summary, "root_records": rows,
            "interpretation": "Stock candidate gradient with detached fixed reference; projection against recorded outcomes is a directional diagnostic, not gradient ground truth."}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, model, optimizer, update, config, history, run_id, device):
    temporary = path.with_suffix(".tmp")
    torch.save({"format": FORMAT, "update": update, "config": config,
                "snapshot_digest": DIGEST, "architecture": model.architecture_config(),
                "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                "history": history, "mlflow_run_id": run_id,
                "torch_rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, temporary)
    temporary.replace(path)


def restore_checkpoint(saved, model, optimizer, config, device):
    if saved["format"] != FORMAT or saved["config"] != config or saved["snapshot_digest"] != DIGEST:
        raise ValueError("resume checkpoint does not match the complete experiment contract")
    if saved["architecture"] != model.architecture_config():
        raise ValueError("resume architecture differs")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "mps":
        torch.mps.set_rng_state(saved["mps_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])


def load_data(run_root, output):
    snapshot_path = output / "snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
    else:
        _load_local_credentials()
        snapshot = json.loads(_download_with_retry(SupabaseStore(), COMBINED_SNAPSHOT_KEY))
    payload = {k: v for k, v in snapshot.items() if k != "snapshot_digest"}
    if snapshot["snapshot_digest"] != DIGEST or _json_digest(payload) != DIGEST:
        raise ValueError("persisted snapshot digest differs from the approved dataset")
    cache_dir = run_root / "cache" / ROOT_CACHE_FORMAT / DIGEST
    cache = json.loads((cache_dir / "cache_index.json").read_text())
    cache["cache_dir"] = str(cache_dir)
    if cache["format"] != ROOT_CACHE_FORMAT or cache["snapshot_digest"] != DIGEST:
        raise ValueError("root cache contract differs")
    train, validation = cache["train_group_ids"], cache["validation_group_ids"]
    groups = {g["candidate_group_id"]: g for g in snapshot["groups"]}
    entries = {g["candidate_group_id"] for g in cache["group_entries"]}
    if len(train) != 1280 or len(set(train)) != 1280 or len(validation) != 320 or len(set(validation)) != 320:
        raise ValueError("expected unique 1280/320 root split")
    if set(train) & set(validation) or set(train) | set(validation) != set(groups) or entries != set(groups):
        raise ValueError("root splits/cache/snapshot are not disjoint and complete")
    if len(cache["group_entries"]) != 1600 or any(not (cache_dir / e["path"]).is_file() for e in cache["group_entries"]):
        raise ValueError("existing root cache is incomplete; no new data may be downloaded")
    _atomic_json(snapshot_path, snapshot)
    return {"train": TimedRoots(cache, train, groups), "validation": TimedRoots(cache, validation, groups)}


def log_metrics(metrics, prefix, update):
    mlflow.log_metrics({prefix + "/" + k: v for k, v in metrics.items()
                        if isinstance(v, (int, float)) and np.isfinite(v)}, step=update)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path.home() / "pnp-vla-runs")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--family", choices=(*FAMILIES, "both"), default="both")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    run_root = args.run_root.expanduser().resolve()
    output = run_root / "checkpoints" / FORMAT / DIGEST
    output.mkdir(parents=True, exist_ok=True)
    roots = load_data(run_root, output)
    checkpoint = run_root / "checkpoints/smolvla-q10-scalar-mc-td-v1-preaction" / DIGEST / "mc/latest.pt"
    initial = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if initial["snapshot_digest"] != DIGEST or initial["update"] != 2000 or initial["arm"] != "mc":
        raise ValueError("expected the common original MC update-2000 baseline")
    arch = dict(initial["architecture"])
    dims = {k: arch.pop(k) for k in ("prefix_dim", "robot_dim", "proprio_dim")}
    base = ScalarCritic(**dims, config=QPlanningModelConfig(**arch))
    base.load_state_dict(initial["model"])
    config = {"objective": "ordinary per-candidate root MC BCE", "updates": 2000,
              "lr": 3e-4, "weight_decay": 1e-4, "clip_norm": 1., "roots_per_batch": 4,
              "candidates_per_root": 9, "batch_rng_formula": "42 * 4000003 + update",
              "seed": 42, "validation_interval": 250, "checkpoint_interval": 100,
              "baseline_checkpoint": str(checkpoint), "baseline_update": 2000,
              "baseline_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "snapshot_digest": DIGEST, "train_root_ids": [e["candidate_group_id"] for e in roots["train"].entries],
              "validation_root_ids": [e["candidate_group_id"] for e in roots["validation"].entries],
              "mean_training_root_presentations": 2000 * 4 / 1280,
              "comparison": "fixed update budget; shared frozen 128-dimensional stock readout",
              "device": device.type, "delta_scale": 100, "reference_detached": True,
              "h_zero_parameter_gradient_detached": False, "frozen_baseline": True}
    contract_path = output / "experiment_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != config:
        raise ValueError("existing output experiment contract differs")
    _atomic_json(contract_path, config)
    print({"prepared": str(output), "train_roots": len(roots["train"]), "validation_roots": len(roots["validation"]),
           "objective": config["objective"], "updates_each": 2000, "families": FAMILIES}, flush=True)
    if args.prepare_only:
        return
    mlflow.set_tracking_uri("sqlite:///" + str(run_root / "mlflow/tracking.db"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(FORMAT)
    exp_id = experiment.experiment_id if experiment else client.create_experiment(FORMAT, artifact_location=(run_root / "mlflow/artifacts").as_uri())
    families = FAMILIES if args.family == "both" else (args.family,)
    for family in families:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        model = AnchoredCritic(base, family=family).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
        family_config = {**config, "family": family, "architecture": model.architecture_config()}
        arm_dir = output / family
        arm_dir.mkdir(exist_ok=True)
        latest = arm_dir / "latest.pt"
        saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
        history, start = (saved["history"], saved["update"]) if saved else ([], 0)
        if saved:
            restore_checkpoint(saved, model, optimizer, family_config, device)
        else:
            model.eval()
            test_batch = _to(_root_batch([roots["train"][i] for i in range(4)]), device)
            with torch.no_grad():
                base_device = ScalarCritic(**dims, config=QPlanningModelConfig(**arch)).to(device)
                base_device.load_state_dict(initial["model"])
                stock_logit = base_device(test_batch["prefix"], test_batch["pad"], test_batch["robot"],
                                          test_batch["proprio"], test_batch["action"][:, 0], test_batch["action_valid"][:, 0])
                torch.testing.assert_close(root_scores(model, test_batch), stock_logit.expand(-1, 9), rtol=1e-4, atol=1e-5)
                del base_device
        with mlflow.start_run(experiment_id=exp_id, run_name=family, run_id=saved["mlflow_run_id"] if saved else None) as run:
            run_id = run.info.run_id
            if not saved:
                mlflow.log_params({k: v for k, v in family_config.items() if k not in ("train_root_ids", "validation_root_ids", "architecture")})
                mlflow.log_artifact(str(output / "experiment_contract.json"))
                for split in ("train", "validation"):
                    metrics = evaluate(model, roots[split], device)
                    expected_stock = 832 if split == "train" else 209
                    if metrics["stock_successes"] != expected_stock or metrics["selected_successes"] != expected_stock:
                        raise ValueError("update-zero predictions must reproduce the recorded stock selector")
                    history.append({"update": 0, "split": split, **metrics})
                    log_metrics(metrics, split, 0)
                    print({"family": family, "update": 0, "split": split, **metrics}, flush=True)
                save_checkpoint(latest, model, optimizer, 0, family_config, history, run_id, device)
            print({"family": family, "resumed_update": start,
                   "mlflow_run": f"http://127.0.0.1:5001/#/experiments/{exp_id}/runs/{run_id}"}, flush=True)
            for update in range(start + 1, 2001):
                torch.manual_seed(42 * 4000003 + update)
                positions = root_positions(update, len(roots["train"]))
                batch = _to(_root_batch([roots["train"][int(i)] for i in positions]), device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(root_scores(model, batch), batch["success"].float())
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite MC/BCE minibatch loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
                if update % 100 == 0:
                    mlflow.log_metric("train/minibatch_bce", float(loss.detach()), step=update)
                    print({"family": family, "update": update, "minibatch_bce": float(loss.detach())}, flush=True)
                if update % 250 == 0:
                    metrics = evaluate(model, roots["validation"], device)
                    history.append({"update": update, "split": "validation", **metrics})
                    log_metrics(metrics, "validation", update)
                    print({"family": family, "update": update, "split": "validation", **metrics}, flush=True)
                if update % 100 == 0:
                    save_checkpoint(latest, model, optimizer, update, family_config, history, run_id, device)
                    _atomic_json(arm_dir / "history.json", history)
            report = {"config": family_config, "update": 2000, "mlflow_run_id": run_id, "splits": {}}
            for split in ("train", "validation"):
                metrics = evaluate(model, roots[split], device)
                diagnostics = gradient_diagnostics(model, roots[split], device, base.action_std.to(device))
                report["splits"][split] = {"metrics": metrics, "gradients": diagnostics}
                log_metrics(metrics, "final/" + split, 2000)
                log_metrics(diagnostics["summary"], "gradient/" + split, 2000)
                print({"family": family, "fixed_final_update": 2000, "split": split,
                       "metrics": metrics, "gradients": diagnostics["summary"]}, flush=True)
            _atomic_json(arm_dir / "final_report.json", report)
            mlflow.log_artifact(str(arm_dir / "final_report.json"))
            mlflow.log_artifact(str(arm_dir / "history.json"))
            print({"family": family, "complete": str(arm_dir / "final_report.json")}, flush=True)


if __name__ == "__main__":
    main()
