"""Measure held-out Q10 action gradients for reranking and PCP diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.train_smolvla_combined_local import _load_local_credentials  # noqa: E402
from pnp.qplanning_critic.config import QPlanningModelConfig  # noqa: E402
from pnp.qplanning_critic.model import QPlanningCritic  # noqa: E402
from pnp.smolvla_combined_success import (  # noqa: E402
    load_or_create_combined_snapshot, prepare_combined_root_cache,
)
from pnp.smolvla_success_critic import TimedRoots  # noqa: E402
from pnp.smolvla_tree_bellman_finetune import _root_batch, _root_scores, _to  # noqa: E402
from pnp.store import SupabaseStore  # noqa: E402


DIGEST = "671b5b211099997fc83d1277"
ROOT = Path.home() / "pnp-vla-runs"
PRIMARY_DIR = ROOT / "checkpoints/smolvla-q10-trajectory-pretrain-root-rank-v1" / DIGEST
SMALL_DIR = ROOT / "checkpoints/smolvla-q10-small-trajectory-pretrain-root-rank-v1" / DIGEST
CHECKPOINTS = {
    "root_only_bce": ROOT / "checkpoints/combined_1600_root_mc_v1" / DIGEST
                     / "root_mc/checkpoint_step_002000.pt",
    "trajectory_pretrained": PRIMARY_DIR / "trajectory_pretrained/pretrained.pt",
    "trajectory_ranked": PRIMARY_DIR / "trajectory_pretrained/finetuned.pt",
    "no_pretrain_ranked": PRIMARY_DIR / "no_pretrain_control/finetuned.pt",
    "control_step_0250": PRIMARY_DIR / "no_pretrain_control/finetune_step_0250.pt",
    "control_step_0500": PRIMARY_DIR / "no_pretrain_control/finetune_step_0500.pt",
    "control_step_0750": PRIMARY_DIR / "no_pretrain_control/finetune_step_0750.pt",
}


def _load_model(path: Path, device: torch.device) -> QPlanningCritic:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("snapshot_digest") != DIGEST:
        raise ValueError(f"checkpoint has wrong snapshot: {path}")
    architecture = dict(saved["architecture"])
    config = QPlanningModelConfig(**{
        key: value for key, value in architecture.items()
        if key not in ("prefix_dim", "robot_dim", "proprio_dim")})
    model = QPlanningCritic(
        prefix_dim=architecture["prefix_dim"],
        robot_dim=architecture["robot_dim"],
        proprio_dim=architecture["proprio_dim"], config=config)
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _direction(grad: np.ndarray, source: np.ndarray, target: np.ndarray,
               source_valid: np.ndarray, target_valid: np.ndarray,
               std: np.ndarray) -> tuple[float, float, float] | None:
    joint = source_valid & target_valid
    if not joint.any():
        return None
    scaled_gradient = (grad * std)[joint].reshape(-1)
    scaled_displacement = ((target - source) / std)[joint].reshape(-1)
    gradient_norm = float(np.linalg.norm(scaled_gradient))
    distance = float(np.linalg.norm(scaled_displacement))
    if gradient_norm < 1e-12 or distance < 1e-12:
        return None
    derivative = float(np.dot(scaled_gradient, scaled_displacement))
    return derivative / (gradient_norm * distance), derivative, distance


def _nearest(index: int, alternatives: np.ndarray, actions: np.ndarray,
             valid: np.ndarray, std: np.ndarray) -> int | None:
    distances = []
    for alternative in alternatives:
        joint = valid[index] & valid[alternative]
        if not joint.any():
            continue
        delta = ((actions[alternative] - actions[index]) / std)[joint]
        distances.append((float(np.linalg.norm(delta)), int(alternative)))
    return min(distances)[1] if distances else None


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"roots": 0}
    keys = ("positive_grad_norm", "negative_grad_norm",
            "negative_to_success_cosine", "negative_to_success_derivative",
            "negative_to_success_positive_fraction")
    result = {"roots": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        result[key] = float(np.mean(values)) if values else float("nan")
    return result


def diagnose(model: QPlanningCritic, dataset: TimedRoots,
             device: torch.device) -> dict:
    std = model.action_std.detach().cpu().numpy().reshape(1, 7)
    mixed_rows, rescue_rows, stock_positive_rows = [], [], []
    positive_step_norms, negative_step_norms = [], []
    indistinguishable_pairs = 0
    for start in range(0, len(dataset), 8):
        batch = _to(_root_batch([dataset[i] for i in
                                range(start, min(len(dataset), start + 8))]), device)
        actions = batch["action"].detach().requires_grad_(True)
        batch["action"] = actions
        scores = _root_scores(model, batch)
        gradient = torch.autograd.grad(scores.sum(), actions)[0]
        q = scores.detach().cpu().numpy()
        g = gradient.detach().cpu().numpy()
        a = actions.detach().cpu().numpy()
        mask = batch["action_valid"].cpu().numpy()
        labels = batch["success"].cpu().numpy().astype(bool)
        for local in range(len(q)):
            positive, negative = np.flatnonzero(labels[local]), np.flatnonzero(~labels[local])
            if not len(positive) or not len(negative):
                continue
            valid = mask[local]
            standardized = g[local] * std
            norms = np.linalg.norm(
                (standardized * valid[:, :, None]).reshape(len(q[local]), -1), axis=1)
            row = {
                "positive_grad_norm": float(norms[positive].mean()),
                "negative_grad_norm": float(norms[negative].mean()),
            }
            positive_step_norms.extend(
                np.linalg.norm(standardized[positive], axis=-1).tolist())
            negative_step_norms.extend(
                np.linalg.norm(standardized[negative], axis=-1).tolist())
            directions = []
            for bad in negative:
                good = _nearest(bad, positive, a[local], valid, std)
                if good is None:
                    continue
                direction = _direction(
                    g[local, bad], a[local, bad], a[local, good],
                    valid[bad], valid[good], std)
                if direction is None:
                    indistinguishable_pairs += 1
                else:
                    directions.append(direction)
            if directions:
                row["negative_to_success_cosine"] = float(np.mean([x[0] for x in directions]))
                row["negative_to_success_derivative"] = float(np.mean([x[1] for x in directions]))
                row["negative_to_success_positive_fraction"] = float(
                    np.mean([x[1] > 0 for x in directions]))
            mixed_rows.append(row)
            if not labels[local, 0]:
                good = _nearest(0, positive, a[local], valid, std)
                if good is not None:
                    direction = _direction(
                        g[local, 0], a[local, 0], a[local, good],
                        valid[0], valid[good], std)
                    if direction is not None:
                        rescue_rows.append(direction)
            else:
                bad = _nearest(0, negative, a[local], valid, std)
                if bad is not None:
                    direction = _direction(
                        g[local, 0], a[local, 0], a[local, bad],
                        valid[0], valid[bad], std)
                    if direction is not None:
                        stock_positive_rows.append(direction)
    result = _summarize(mixed_rows)
    result["rescueable_stock_roots"] = len(rescue_rows)
    result["stock_failure_toward_success_cosine"] = (
        float(np.mean([x[0] for x in rescue_rows])) if rescue_rows else float("nan"))
    result["stock_failure_toward_success_positive_fraction"] = (
        float(np.mean([x[1] > 0 for x in rescue_rows])) if rescue_rows else float("nan"))
    result["stock_success_toward_failure_negative_fraction"] = (
        float(np.mean([x[1] < 0 for x in stock_positive_rows]))
        if stock_positive_rows else float("nan"))
    result["indistinguishable_action_pairs"] = indistinguishable_pairs
    result["positive_step_grad_norm"] = np.mean(positive_step_norms, axis=0).tolist()
    result["negative_step_grad_norm"] = np.mean(negative_step_norms, axis=0).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-only", action="store_true",
                        help="diagnose and log only the smaller model checkpoints")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("Mac Metal GPU required")
    _load_local_credentials()
    store = SupabaseStore()
    snapshot = load_or_create_combined_snapshot(store)
    if snapshot["snapshot_digest"] != DIGEST:
        raise ValueError("frozen snapshot changed")
    cache = prepare_combined_root_cache(
        snapshot=snapshot, cache_root=ROOT / "cache", store=store)
    groups = {row["candidate_group_id"]: row for row in snapshot["groups"]}
    validation = TimedRoots(cache, cache["validation_group_ids"], groups)
    device = torch.device("mps")
    report = {"snapshot_digest": DIGEST, "validation_roots": len(validation),
              "models": {}}
    checkpoints = ({
        "small_trajectory_pretrained": SMALL_DIR / "trajectory_pretrained/pretrained.pt",
        "small_trajectory_ranked": SMALL_DIR / "trajectory_pretrained/finetuned.pt",
        "small_no_pretrain_ranked": SMALL_DIR / "no_pretrain_control/finetuned.pt",
    } if args.small_only else CHECKPOINTS)
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = _load_model(checkpoint, device)
        metrics = diagnose(model, validation, device)
        report["models"][name] = {"checkpoint": str(checkpoint), **metrics}
        print({"model": name, "metrics": {key: value for key, value in metrics.items()
                                         if not isinstance(value, list)}}, flush=True)
        del model
    destination = ROOT / "diagnostics" / DIGEST
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / ("action_gradients_small.json" if args.small_only
                          else "action_gradients.json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow/tracking.db"))
    client = MlflowClient()
    primary_id = ("177ba6a9d27f44c5a37427762cd858d9" if args.small_only
                  else "ff8b8bfa127849b7aea82671b5769bf9")
    control_id = ("297bc79c6dac46b3b8a27344b498d690" if args.small_only
                  else "19fdcc892b304baf816bb7d2ea4059c0")
    for name, row in report["models"].items():
        run_id = (control_id if name.startswith(("no_pretrain", "control_step",
                                                 "small_no_pretrain")) else primary_id)
        for key, value in row.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                client.log_metric(run_id, f"grad_{name}/{key}", float(value))
    client.log_artifact(primary_id, str(path))
    client.log_artifact(control_id, str(path))
    print({"report": str(path),
           "mlflow_primary": f"http://127.0.0.1:5000/#/experiments/{2 if args.small_only else 1}/runs/{primary_id}",
           "mlflow_control": f"http://127.0.0.1:5000/#/experiments/{2 if args.small_only else 1}/runs/{control_id}"},
          flush=True)


if __name__ == "__main__":
    main()
