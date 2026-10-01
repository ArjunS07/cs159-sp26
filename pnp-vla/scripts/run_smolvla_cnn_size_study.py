"""Matched local CNN capacity and chronological early-stopping study.

Nine MC/BCE runs: widths 32/64/128 x initialization seeds 42/123/314.
Frozen original stock features are shared; every run receives identical batches.
The existing continuation GPU lock is acquired before any MPS computation.
"""
from __future__ import annotations

import argparse
import fcntl
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
    DIGEST, cpu_tree, load_data, log_metrics, pair_accuracy, root_positions,
    root_scores, scoring_metrics,
)
from scripts.run_smolvla_local_continuation import scalar_from_architecture
from pnp.smolvla_anchored_critic import AnchoredCritic
from pnp.smolvla_tree_bellman_finetune import _atomic_json, _root_batch, _to

FORMAT = "smolvla_cnn_size_study_v1"
WIDTHS = (32, 64, 128)
SEEDS = (42, 123, 314)
UPDATES = 8000
INTERVAL = 250
PATIENCE = 8
MIN_DELTA = .001


def early_state(initial_bce):
    return {"best_bce": float(initial_bce), "selected_update": 0,
            "misses": 0, "would_stop_update": None}


def advance_early(state, update, bce):
    """Only chronological improvements before the first stop can replace best."""
    if state["would_stop_update"] is not None:
        return False
    if bce < state["best_bce"] - MIN_DELTA:
        state.update(best_bce=float(bce), selected_update=update, misses=0)
        return True
    state["misses"] += 1
    if state["misses"] >= PATIENCE:
        state["would_stop_update"] = update
    return False


def learning_rate(update):
    return 3e-5 + (3e-4 - 3e-5) * .5 * (1 + math.cos(math.pi * update / UPDATES))


def selected_path(directory, update):
    # Immutable selected versions keep resume valid if interruption occurs
    # between writing an improvement checkpoint and committing latest history.
    return directory / f"early_step_{update:06d}.pt"


def cached_scores(model, batch):
    n, count = batch["action"].shape[:2]
    if count != 9 or not torch.equal(batch["valid"], batch["valid"][:, :1].expand_as(batch["valid"])):
        raise ValueError("expected nine full proposals with shared known-budget masks")
    repeat = lambda x: x.repeat_interleave(9, 0)
    return model.logits_from_features(repeat(batch["features"]), repeat(batch["baseline"]),
        batch["action"].flatten(0, 1), batch["valid"].flatten(0, 1),
        repeat(batch["action"][:, 0])).reshape(n, 9)


def take(data, positions):
    return {k: v[positions] for k, v in data.items()}


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    values = []
    for start in range(0, len(data["labels"]), 32):
        values.append(cached_scores(model, take(data, slice(start, start + 32))).cpu().numpy())
    return scoring_metrics(np.concatenate(values), data["labels"].cpu().numpy())


def cached_gradients(model, data, ids):
    model.eval()
    rows = []
    for start in range(0, len(ids), 32):
        batch = take(data, slice(start, start + 32))
        reference = batch["action"][:, 0].detach()
        stock = reference.clone().requires_grad_(True)
        z = model.logits_from_features(batch["features"], batch["baseline"],
                                     stock, batch["valid"][:, 0], reference)
        gradient, = torch.autograd.grad(z.sum(), stock)
        with torch.no_grad():
            scaled = gradient * model.action_std * batch["valid"][:, 0, :, None]
            delta = (batch["action"] - reference[:, None]) / model.action_std
            projection = (scaled[:, None] * delta).sum((-1, -2)).cpu().numpy()
            norms = scaled.square().sum((-1, -2)).sqrt().cpu().numpy()
            labels = batch["labels"].cpu().numpy().astype(bool)
        for j, y in enumerate(labels):
            rows.append({"group_id": ids[start + j], "labels": y.tolist(),
                         "gradient_projection": projection[j].tolist(),
                         "normalized_gradient_norm": float(norms[j]),
                         "gradient_pair_accuracy": pair_accuracy(projection[j], y),
                         "fresh_gradient_pair_accuracy": pair_accuracy(projection[j, 1:], y[1:])})
    pairs = [r["gradient_pair_accuracy"] for r in rows if r["gradient_pair_accuracy"] is not None]
    fresh = [r["fresh_gradient_pair_accuracy"] for r in rows if r["fresh_gradient_pair_accuracy"] is not None]
    summary = {"roots": len(rows), "mixed_roots": len(pairs), "fresh_mixed_roots": len(fresh),
               "gradient_macro_pair_accuracy": float(np.mean(pairs)) if pairs else None,
               "fresh_gradient_macro_pair_accuracy": float(np.mean(fresh)) if fresh else None,
               "median_normalized_gradient_norm": float(np.median([r["normalized_gradient_norm"] for r in rows]))}
    return {"summary": summary, "root_records": rows,
            "interpretation": "Fixed-reference candidate gradient; projection against outcomes is not gradient ground truth."}


def save(path, model, optimizer, update, config, history, early, run_id, device):
    temp = path.with_suffix(".tmp")
    torch.save({"format": FORMAT, "snapshot_digest": DIGEST, "update": update,
                "architecture": model.architecture_config(), "config": config,
                "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                "history": history, "early": dict(early), "mlflow_run_id": run_id,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None}, temp)
    temp.replace(path)


def restore(saved, model, optimizer, config, device):
    if (saved["format"] != FORMAT or saved["snapshot_digest"] != DIGEST or
            saved["config"] != config or saved["architecture"] != model.architecture_config() or
            not 0 <= saved["update"] <= UPDATES):
        raise ValueError("size-study resume contract differs")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    if device.type == "mps":
        torch.mps.set_rng_state(saved["mps_rng"])


def prepare_features(base, roots, output, base_sha, device):
    path = output / "frozen_features.pt"
    ids = {split: [e["candidate_group_id"] for e in rs.entries] for split, rs in roots.items()}
    contract = {"snapshot_digest": DIGEST, "base_sha256": base_sha, "group_ids": ids,
                "source": "detached original stock decoder readout, full proposed actions, known-budget masks"}
    model = AnchoredCritic(base, "temporal_cnn", width=32).to(device).eval()
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache["contract"] != contract:
            raise ValueError("frozen features do not match source/splits/baseline")
        data = cache["data"]
    else:
        data = {}
        for split, rs in roots.items():
            pieces = {k: [] for k in ("features", "baseline", "action", "valid", "labels")}
            with torch.no_grad():
                for start in range(0, len(rs), 4):
                    batch = _to(_root_batch([rs[i] for i in range(start, min(start + 4, len(rs)))]), device)
                    features, baseline = model.reference_features(batch["prefix"], batch["pad"],
                        batch["robot"], batch["proprio"], batch["action"][:, 0], batch["action_valid"][:, 0])
                    for key, value in zip(pieces, (features, baseline, batch["action"], batch["action_valid"], batch["success"])):
                        pieces[key].append(value.detach().cpu())
            data[split] = {k: torch.cat(v) for k, v in pieces.items()}
            print({"features_cached": split, "roots": len(rs)}, flush=True)
        temp = path.with_suffix(".tmp")
        torch.save({"contract": contract, "data": data}, temp)
        temp.replace(path)
    # Check both splits against live baseline evaluation, including nonzero residual.
    with torch.no_grad():
        model.readout[-1].weight.normal_(std=.01)
        for split, rs in roots.items():
            if (len(data[split]["labels"]) != len(rs) or
                    not all(torch.isfinite(data[split][k]).all() for k in ("features", "baseline", "action", "labels"))):
                raise ValueError("invalid feature cache")
            for index in (0, len(rs) - 1):
                direct = _to(_root_batch([rs[index]]), device)
                cached = _to(take(data[split], slice(index, index + 1)), device)
                torch.testing.assert_close(cached["action"], direct["action"], rtol=0, atol=0)
                torch.testing.assert_close(cached["valid"], direct["action_valid"], rtol=0, atol=0)
                torch.testing.assert_close(cached["labels"], direct["success"], rtol=0, atol=0)
                torch.testing.assert_close(cached_scores(model, cached), root_scores(model, direct), rtol=1e-4, atol=1e-5)
    del model
    return {split: _to(values, device) for split, values in data.items()}, ids


def run_arm(base, data, ids, output, width, seed, study, device, experiment_id):
    directory = output / f"width{width}_seed{seed}"
    directory.mkdir(exist_ok=True)
    report_path = directory / "final_report.json"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = AnchoredCritic(base, "temporal_cnn", width=width).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    config = {**study, "width": width, "seed": seed, "architecture": model.architecture_config(),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["config"] != config or report["fixed_final_update"] != UPDATES:
            raise ValueError("completed study arm contract differs")
        return report
    _atomic_json(directory / "experiment_contract.json", config)
    latest = directory / "latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    history, start, early = (saved["history"], saved["update"], saved["early"]) if saved else ([], 0, None)
    if saved:
        restore(saved, model, optimizer, config, device)
        chosen = torch.load(selected_path(directory, early["selected_update"]), map_location="cpu", weights_only=False)
        if chosen["config"] != config or chosen["update"] != early["selected_update"]:
            raise ValueError("early-selected checkpoint/history differs")
    started = time.monotonic()
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"width{width}_seed{seed}",
                          run_id=saved["mlflow_run_id"] if saved else None) as run:
        run_id = run.info.run_id
        if not saved:
            mlflow.log_params({k: v for k, v in config.items() if k not in ("group_ids", "architecture")})
            mlflow.log_artifact(str(directory / "experiment_contract.json"))
            for split in ("train", "validation"):
                metrics = evaluate(model, data[split])
                if metrics["selected_successes"] != metrics["stock_successes"]:
                    raise ValueError("zero residual must select stock")
                history.append({"update": 0, "split": split, **metrics})
                log_metrics(metrics, split, 0)
            early = early_state(history[-1]["nonstock_bce"])
            save(selected_path(directory, 0), model, optimizer, 0, config, history, early, run_id, device)
            save(latest, model, optimizer, 0, config, history, early, run_id, device)
        print({"arm": directory.name, "resumed_update": start,
               "trainable_parameters": config["trainable_parameters"],
               "mlflow_run": f"http://127.0.0.1:5001/#/experiments/{experiment_id}/runs/{run_id}"}, flush=True)
        for update in range(start + 1, UPDATES + 1):
            # Initialization seeds cannot alter the common root-index stream.
            positions = torch.as_tensor(root_positions(update, len(data["train"]["labels"])), device=device)
            batch = take(data["train"], positions)
            model.train(); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(update)
            loss = F.binary_cross_entropy_with_logits(cached_scores(model, batch), batch["labels"].float())
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite study MC loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            if update % 100 == 0:
                log_metrics({"bce": float(loss.detach()), "gradient_norm": float(norm),
                             "lr": learning_rate(update)}, "minibatch", update)
            if update % INTERVAL == 0:
                for split in ("train", "validation"):
                    metrics = evaluate(model, data[split])
                    history.append({"update": update, "split": split, **metrics})
                    log_metrics(metrics, split, update)
                if advance_early(early, update, history[-1]["nonstock_bce"]):
                    save(selected_path(directory, update), model, optimizer, update, config, history, early, run_id, device)
                print({"arm": directory.name, "update": update,
                       "train": history[-2], "validation": history[-1], "early": dict(early),
                       "elapsed_minutes": round((time.monotonic() - started) / 60, 2)}, flush=True)
            if update % 100 == 0 or update % INTERVAL == 0:
                save(latest, model, optimizer, update, config, history, early, run_id, device)
                _atomic_json(directory / "history.json", history)
        final_path = directory / "fixed_final.pt"
        save(final_path, model, optimizer, UPDATES, config, history, early, run_id, device)
        report = {"config": config, "fixed_final_update": UPDATES, "early": dict(early),
                  "mlflow_run_id": run_id, "fixed_final": {}, "early_selected": {}}
        for endpoint in ("fixed_final", "early_selected"):
            if endpoint == "early_selected":
                chosen = torch.load(selected_path(directory, early["selected_update"]), map_location="cpu", weights_only=False)
                model.load_state_dict(chosen["model"])
                report["early_checkpoint_update"] = chosen["update"]
            for split in ("train", "validation"):
                metrics = evaluate(model, data[split])
                gradients = cached_gradients(model, data[split], ids[split])
                report[endpoint][split] = {"metrics": metrics, "gradients": gradients}
                log_metrics(metrics, endpoint + "/" + split, UPDATES if endpoint == "fixed_final" else chosen["update"])
            print({"arm": directory.name, "endpoint": endpoint,
                   "validation": report[endpoint]["validation"]["metrics"],
                   "gradient": report[endpoint]["validation"]["gradients"]["summary"]}, flush=True)
        _atomic_json(report_path, report)
        mlflow.log_artifact(str(report_path)); mlflow.log_artifact(str(directory / "history.json"))
    return report


def write_summary(reports, output):
    rows = []
    for r in reports:
        for endpoint in ("fixed_final", "early_selected"):
            row = {"width": r["config"]["width"], "seed": r["config"]["seed"],
                   "trainable_parameters": r["config"]["trainable_parameters"], "endpoint": endpoint,
                   "selected_update": UPDATES if endpoint == "fixed_final" else r["early_checkpoint_update"],
                   "would_stop_update": r["early"]["would_stop_update"], "mlflow_run_id": r["mlflow_run_id"]}
            for split in ("train", "validation"):
                row[split] = r[endpoint][split]["metrics"]
                row[split + "_gradients"] = r[endpoint][split]["gradients"]["summary"]
            rows.append(row)
    _atomic_json(output / "comparison.json", {"rows": rows, "completed_arms": len(reports),
        "planned_arms": 9, "interpretation": "Exploratory repeatedly used validation; chronological rule, not independent early-stopping performance."})
    lines = ["# CNN size and stopping study", "", "Validation stock: 209/320. Each arm: 8,000 MC updates, 25 average root presentations.", "",
             "| Width | Seed | Endpoint | Selected update | Stop update | Train fresh pair | Val fresh pair | Val successes | Rescues | Spoils | Val Brier |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, v = r["train"], r["validation"]
        lines.append(f"| {r['width']} | {r['seed']} | {r['endpoint']} | {r['selected_update']} | {r['would_stop_update']} | {t['fresh_macro_pair_accuracy']:.3f} | {v['fresh_macro_pair_accuracy']:.3f} | {v['selected_successes']}/320 | {v['rescues']} | {v['spoils']} | {v['brier']:.4f} |")
    lines.extend(["", "Early selection is frozen after eight consecutive misses of a .001 improvement in nonstock validation BCE. Update zero is eligible. All arms continue to the same final exposure. These validation comparisons are exploratory."])
    (output / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--run-root", type=Path, default=Path.home() / "pnp-vla-runs")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run_root = args.run_root.expanduser().resolve()
    output = run_root / "checkpoints" / FORMAT / DIGEST
    output.mkdir(parents=True, exist_ok=True)
    source_dir = run_root / "checkpoints/smolvla_anchored_cnn_v1" / DIGEST
    base_path = run_root / "checkpoints/smolvla-q10-scalar-mc-td-v1-preaction" / DIGEST / "mc/latest.pt"
    initial = torch.load(base_path, map_location="cpu", weights_only=False)
    if initial["snapshot_digest"] != DIGEST or initial["update"] != 2000 or initial["arm"] != "mc":
        raise ValueError("expected original frozen MC update2000 baseline")
    base = scalar_from_architecture(initial["architecture"])
    base.load_state_dict(initial["model"])
    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if args.prepare_only:
        roots = load_data(run_root, source_dir)
        counts = {w: sum(p.numel() for p in AnchoredCritic(base, "temporal_cnn", width=w).parameters() if p.requires_grad) for w in WIDTHS}
        print({"prepared": FORMAT, "train_roots": len(roots["train"]), "validation_roots": len(roots["validation"]),
               "width_parameters": counts, "seeds": SEEDS, "updates_each": UPDATES, "baseline_sha256": base_sha}, flush=True)
        return
    own_lock = (run_root / "diagnostics/cnn_size_study_v1.lock").open("a+")
    try:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("CNN size study is already queued/running") from error
    gpu_lock = (run_root / "diagnostics/local_continuation_v1.lock").open("a+")
    print({"status": "waiting_for_exclusive_GPU", "queue": "after existing local continuation"}, flush=True)
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    print({"status": "exclusive_GPU_acquired"}, flush=True)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    roots = load_data(run_root, source_dir)
    data, ids = prepare_features(base, roots, output, base_sha, device)
    stream = np.stack([root_positions(u, 1280) for u in range(1, UPDATES + 1)])
    exposure = np.bincount(stream.ravel(), minlength=1280)
    study = {"format": FORMAT, "snapshot_digest": DIGEST, "baseline_sha256": base_sha,
             "group_ids": ids, "widths": list(WIDTHS), "seeds": list(SEEDS),
             "updates": UPDATES, "roots_per_batch": 4, "candidates_per_root": 9,
             "batch_rng_formula": "42 * 4000003 + update; independent of initialization seed",
             "mean_root_presentations": 25, "min_root_presentations": int(exposure.min()),
             "max_root_presentations": int(exposure.max()),
             "root_stream_sha256": hashlib.sha256(stream.tobytes()).hexdigest(),
             "objective": "ordinary all-nine root MC BCE", "optimizer": "fresh AdamW",
             "lr_peak": 3e-4, "lr_final": 3e-5, "schedule": "cosine over8000updates",
             "weight_decay": 1e-4, "clip_norm": 1., "delta_scale": 100.,
             "eval_interval": INTERVAL, "checkpoint_interval": 100, "early_patience": PATIENCE,
             "early_min_delta": MIN_DELTA, "early_criterion": "nonstock validation BCE",
             "early_policy": "chronological, update0 eligible, selected checkpoint frozen at first stop",
             "frozen_features": True, "device": device.type,
             "comparison_caveat": "reused validation, exploratory; earlier10k CNN is not matched control"}
    contract_path = output / "study_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != study:
        raise ValueError("study matrix contract differs")
    _atomic_json(contract_path, study)
    mlflow.set_tracking_uri("sqlite:///" + str(run_root / "mlflow/tracking.db"))
    client = MlflowClient(); experiment = client.get_experiment_by_name(FORMAT)
    exp_id = experiment.experiment_id if experiment else client.create_experiment(FORMAT, artifact_location=(run_root / "mlflow/artifacts").as_uri())
    _atomic_json(output / "mlflow.json", {"experiment_id": exp_id,
        "url": f"http://127.0.0.1:5001/#/experiments/{exp_id}"})
    reports = []
    # Interleave sizes within seeds so a stopped machine has a comparable first seed.
    for seed in SEEDS:
        for width in WIDTHS:
            report = run_arm(base, data, ids, output, width, seed, study, device, exp_id)
            reports.append(report); write_summary(reports, output)
    print({"study_complete": len(reports), "summary": str(output / "results.md")}, flush=True)


if __name__ == "__main__":
    main()
