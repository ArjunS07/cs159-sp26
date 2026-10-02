"""Offline MC/TD scoring and gradient diagnostics; no policy intervention."""
import argparse
import json
from pathlib import Path
import sys

import mlflow
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_smolvla_combined_local import _load_local_credentials
from scripts.analyze_smolvla_q10_gradients import summarize, _pair_accuracy
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_combined_success import load_or_create_combined_snapshot, prepare_combined_root_cache
from pnp.smolvla_scalar_returns import ScalarCritic, LateFusionScalarCritic
from pnp.smolvla_success_critic import TimedRoots, evaluate_roots
from pnp.smolvla_tree_bellman_finetune import _root_batch, _root_scores, _to
from pnp.store import SupabaseStore


def diagnose(model, roots, device):
    model.eval().requires_grad_(False)
    rows, spreads, logit_spreads, shuffled_change, state_briers = [], [], [], [], []
    for start in range(0, len(roots), 8):
        batch = _to(_root_batch([roots[i] for i in range(start, min(start + 8, len(roots)))]), device)
        action = batch["action"][:, 0].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            z = model(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
                      action, batch["action_valid"][:, 0]).squeeze(-1)
            gradient, = torch.autograd.grad(z.sum(), action)
        with torch.no_grad():
            count = batch["action"].shape[1]
            repeat = lambda x: x.repeat_interleave(count, 0)
            all_logits = model(repeat(batch["prefix"]), repeat(batch["pad"]), repeat(batch["robot"]),
                               repeat(batch["proprio"]), batch["action"].flatten(0, 1),
                               batch["action_valid"].flatten(0, 1)).reshape(-1, count)
            scores = all_logits.sigmoid().cpu().numpy()
            raw_logits = all_logits.cpu().numpy()
            shuffled = dict(batch)
            # Replace action proposals across states; no labels are assigned to these edits.
            shuffled["action"] = batch["action"].roll(1, 0)
            ablated = _root_scores(model, shuffled).cpu().numpy()
        std = model.action_std
        g_standard = gradient * std * batch["action_valid"][:, 0, :, None]
        displacement = (batch["action"] - action.detach()[:, None]) / std
        projection = (g_standard[:, None] * displacement).sum((-1, -2)).detach().cpu().numpy()
        norms = g_standard.square().sum((-1, -2)).sqrt().detach().cpu().numpy()
        dimensions = g_standard.abs().mean(1).detach().cpu().numpy()
        labels = batch["success"].cpu().numpy().astype(bool)
        spreads.extend(scores.std(1).tolist())
        logit_spreads.extend(raw_logits.std(1).tolist())
        state_briers.extend(np.square(scores[:, :1] - labels).mean(1).tolist())
        shuffled_change.extend(np.abs(ablated - scores).mean(1).tolist())
        for j, outcome in enumerate(labels):
            stock_success = bool(outcome[0])
            rows.append({"group_id": roots.entries[start + j]["candidate_group_id"],
                         "stock_success": stock_success, "candidate_successes": int(outcome.sum()),
                         "candidate_count": len(outcome),
                         "gradient_pair_accuracy": _pair_accuracy(projection[j], outcome),
                         "critic_pair_accuracy": _pair_accuracy(scores[j], outcome),
                         "fresh_pair_accuracy": _pair_accuracy(scores[j, 1:], outcome[1:]),
                         "labels": outcome.tolist(), "scores": scores[j].tolist(),
                         "logits": raw_logits[j].tolist(), "gradient_projection": projection[j].tolist(),
                         "selected_success": bool(outcome[int(scores[j].argmax())]),
                         "rescue_positive": (projection[j, 1:][outcome[1:]] > 0).tolist() if not stock_success else [],
                         "spoil_negative": (projection[j, 1:][~outcome[1:]] < 0).tolist() if stock_success else [],
                         "gradient_norm": float(norms[j]), "gradient_by_dim": dimensions[j].tolist(),
                         "probability_gradient_norm": float(norms[j] * scores[j, 0] * (1 - scores[j, 0]))})
    rng = np.random.default_rng(42)
    deltas = np.asarray([int(r["selected_success"]) - int(r["stock_success"]) for r in rows])
    bootstrap = deltas[rng.integers(0, len(rows), size=(2000, len(rows)))].mean(1)
    fresh = [r["fresh_pair_accuracy"] for r in rows if r["fresh_pair_accuracy"] is not None]
    return {**summarize(rows), "within_root_score_std": float(np.mean(spreads)),
            "within_root_logit_std": float(np.mean(logit_spreads)),
            "stock_score_repeated_brier": float(np.mean(state_briers)),
            "fresh_macro_pair_accuracy": float(np.mean(fresh)) if fresh else None,
            "fresh_mixed_roots": len(fresh),
            "selection_delta_vs_stock_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "cross_state_action_shuffle_mean_score_change": float(np.mean(shuffled_change)),
            "median_probability_gradient_norm": float(np.median([r["probability_gradient_norm"] for r in rows]))}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("mps")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = dict(saved["architecture"])
    family = arch.pop("model_family", "scalar_decoder")
    dims = {k: arch.pop(k) for k in ("prefix_dim", "robot_dim", "proprio_dim")}
    model = (LateFusionScalarCritic if family == "late_fusion_scalar" else ScalarCritic)(**dims, config=QPlanningModelConfig(**arch)).to(device)
    model.load_state_dict(saved["model"])
    _load_local_credentials()
    store = SupabaseStore()
    snapshot = load_or_create_combined_snapshot(store)
    if snapshot["snapshot_digest"] != saved["snapshot_digest"]:
        raise ValueError("checkpoint/snapshot mismatch")
    cache = prepare_combined_root_cache(snapshot=snapshot, cache_root=Path.home() / "pnp-vla-runs/cache", store=store)
    groups = {g["candidate_group_id"]: g for g in snapshot["groups"]}
    report = {"checkpoint": str(args.checkpoint), "update": saved["update"], "splits": {},
              "interpretation": "gradient projection against recorded outcomes is a directional diagnostic, not gradient ground truth"}
    mlflow.set_tracking_uri("sqlite:///" + str(Path.home() / "pnp-vla-runs/mlflow/tracking.db"))
    for split in ("train", "validation"):
        roots = TimedRoots(cache, cache["train_group_ids" if split == "train" else "validation_group_ids"], groups)
        with torch.no_grad():
            metrics = evaluate_roots(model, roots, device)
        diagnostics, rows = diagnose(model, roots, device)
        report["splits"][split] = {"selection": metrics, "diagnostics": diagnostics, "root_records": rows}
        print({"split": split, "selection": metrics, "diagnostics": diagnostics}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    with mlflow.start_run(run_id=saved["mlflow_run_id"]):
        mlflow.log_artifact(str(args.output))
        for split, values in report["splits"].items():
            mlflow.log_metrics({"diagnostic/" + split + "/" + k: v for k, v in values["diagnostics"].items()
                                if isinstance(v, (float, int))}, step=saved["update"])


if __name__ == "__main__":
    main()
