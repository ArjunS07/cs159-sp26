"""Ask whether a scalar Q critic can memorize 32 leak-safe mixed roots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.train_smolvla_combined_local import _load_local_credentials  # noqa: E402
from pnp.qplanning_critic.config import QPlanningModelConfig  # noqa: E402
from pnp.qplanning_critic.model import QPlanningCritic  # noqa: E402
from pnp.smolvla_combined_success import (  # noqa: E402
    ROOT_CACHE_FORMAT, load_or_create_combined_snapshot, prepare_combined_root_cache,
)
from pnp.smolvla_success_critic import TimedRoots, action_statistics  # noqa: E402
from pnp.smolvla_tree_bellman_finetune import _root_batch, _to  # noqa: E402
from pnp.smolvla_two_stage import pairwise_root_loss  # noqa: E402
from pnp.store import SupabaseStore  # noqa: E402


SEED = 42
ROOT = Path.home() / "pnp-vla-runs"
EXPERIMENT = "smolvla-q10-action-memorization-v2-preaction"


def _score(model: QPlanningCritic, batch: dict[str, torch.Tensor],
           delta_scale: float) -> torch.Tensor:
    roots, candidates = batch["action"].shape[:2]
    repeat = lambda x: x.repeat_interleave(candidates, 0)
    actions = batch["action"]
    if delta_scale:
        actions = (actions - actions[:, :1]) * delta_scale
    return model(
        repeat(batch["prefix"]), repeat(batch["pad"]),
        repeat(batch["robot"]), repeat(batch["proprio"]),
        actions.reshape(roots * candidates, 10, 7),
        batch["action_valid"].reshape(roots * candidates, 10),
    ).reshape(roots, candidates)


@torch.no_grad()
def _evaluate(model: QPlanningCritic, selected: list[dict],
              device: torch.device, delta_scale: float) -> dict:
    model.eval()
    correct = total = perfect = 0
    margins = []
    zeroed_spread = []
    for start in range(0, len(selected), 8):
        batch = _to(_root_batch(selected[start:start + 8]), device)
        scores = _score(model, batch, delta_scale).cpu().numpy()
        labels = batch["success"].cpu().numpy().astype(bool)
        for values, success in zip(scores, labels):
            differences = values[success, None] - values[None, ~success]
            wins = int((differences > 0).sum())
            correct += wins
            total += differences.size
            perfect += int(wins == differences.size)
            margins.append(float(differences.mean()))
        # Replacing each candidate with the same stock action must collapse
        # all within-root differences if the score actually depends on action.
        batch["action"] = batch["action"][:, :1].expand_as(batch["action"]).clone()
        collapsed = _score(model, batch, delta_scale).cpu().numpy()
        zeroed_spread.extend(collapsed.std(axis=1).tolist())
    return {
        "pair_accuracy": correct / total,
        "perfect_roots": perfect,
        "mean_success_minus_failure_logit": float(np.mean(margins)),
        "same_action_score_std": float(np.mean(zeroed_spread)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-scale", type=float, default=0,
                        help="diagnostic stock-relative action multiplier; zero uses absolute actions")
    args = parser.parse_args()
    if args.delta_scale < 0:
        parser.error("delta-scale must be nonnegative")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Mac Metal GPU required")
    _load_local_credentials()
    store = SupabaseStore()
    snapshot = load_or_create_combined_snapshot(store)
    cache = prepare_combined_root_cache(snapshot=snapshot, cache_root=ROOT / "cache", store=store)
    if cache["format"] != ROOT_CACHE_FORMAT:
        raise ValueError("memorization test requires the pre-action root cache")
    groups = {g["candidate_group_id"]: g for g in snapshot["groups"]}
    train = TimedRoots(cache, cache["train_group_ids"], groups)
    rescue = [i for i, row in enumerate(train.entries)
              if row["mixed"] and not row["stock_success"]]
    other = [i for i, row in enumerate(train.entries)
             if row["mixed"] and row["stock_success"]]
    ordered = lambda indices: sorted(indices, key=lambda i: hashlib.sha256(
        ("memorize-v2|" + train.entries[i]["candidate_group_id"]).encode()).hexdigest())
    if len(rescue) < 16 or len(other) < 16:
        raise ValueError("need 16 rescueable and 16 other mixed training roots")
    indices = ordered(rescue)[:16] + ordered(other)[:16]
    selected = [train[i] for i in indices]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    first = selected[0]
    cfg = QPlanningModelConfig(
        action_horizon=10, action_dim=7, width=128, n_layers=2,
        n_heads=4, ffn_width=512, dropout=0, n_bins=2,
        prefix_pool_tokens=128)
    model = QPlanningCritic(
        prefix_dim=first["prefix"].shape[-1], robot_dim=len(first["robot"]),
        proprio_dim=len(first["proprio"]), config=cfg)
    model.value_head = nn.Linear(cfg.width, 1)
    mean, std = (action_statistics(train) if not args.delta_scale else
                 (np.zeros(7, np.float32), np.ones(7, np.float32)))
    model.set_action_statistics(mean, std)
    device = torch.device("mps")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    tracking = ROOT / "mlflow"
    mlflow.set_tracking_uri("sqlite:///" + str(tracking / "tracking.db"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (experiment.experiment_id if experiment else
                     client.create_experiment(EXPERIMENT,
                         artifact_location=(tracking / "artifacts").as_uri()))
    suffix = "absolute" if not args.delta_scale else f"delta_{args.delta_scale:g}"
    output = ROOT / "diagnostics" / snapshot["snapshot_digest"] / f"memorization_v2_{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"scalar-pair-only-32-roots-{suffix}") as run:
        url = f"http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run.info.run_id}"
        print({"mlflow_run": url, "selected_roots": len(indices),
               "trainable_parameters": sum(p.numel() for p in model.parameters())}, flush=True)
        mlflow.log_params({
            "seed": SEED, "roots": 32, "rescueable_stock_failure_roots": 16,
            "other_mixed_roots": 16, "layers": 2, "width": 128,
            "dropout": 0, "head": "scalar_logit", "loss": "pairwise_softplus",
            "replay_weight": 0, "lr": 3e-4, "updates": 2000,
            "delta_scale": args.delta_scale,
            "root_cache_format": ROOT_CACHE_FORMAT,
        })
        initial = _evaluate(model, selected, device, args.delta_scale)
        print({"update": 0, **initial}, flush=True)
        mlflow.log_metrics(initial, step=0)
        for step in range(1, 2001):
            model.train()
            rng = np.random.default_rng(SEED * 3_000_001 + step)
            chosen = rng.choice(len(selected), size=8, replace=False)
            batch = _to(_root_batch([selected[int(i)] for i in chosen]), device)
            optimizer.zero_grad(set_to_none=True)
            scores = _score(model, batch, args.delta_scale)
            loss = pairwise_root_loss(scores, batch["success"], temperature=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if step % 100 == 0:
                metrics = _evaluate(model, selected, device, args.delta_scale)
                row = {"update": step, "train_pair_loss": float(loss.detach()), **metrics}
                history.append(row)
                mlflow.log_metrics({k: v for k, v in row.items() if k != "update"}, step=step)
                print(row, flush=True)
                if metrics["pair_accuracy"] >= .99:
                    break
        checkpoint = output / "scalar_pair_only.pt"
        torch.save({"snapshot_digest": snapshot["snapshot_digest"],
                    "root_ids": [train.entries[i]["candidate_group_id"] for i in indices],
                    "architecture": model.architecture_config(),
                    "model": model.state_dict(), "history": history}, checkpoint)
        report = {"mlflow_run": url, "initial": initial, "history": history,
                  "checkpoint": str(checkpoint)}
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(path))
        print({"report": str(path), "final": history[-1]}, flush=True)


if __name__ == "__main__":
    main()
