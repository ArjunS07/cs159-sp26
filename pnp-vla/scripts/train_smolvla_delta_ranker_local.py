"""Train a stock-relative action ranker on leak-safe SmolVLA Q10 roots."""
from __future__ import annotations

import json
import math
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


ROOT = Path.home() / "pnp-vla-runs"
EXPERIMENT = "smolvla-q10-delta-ranker-v1-preaction"
SEED = 42
DELTA_SCALE = 100.0
UPDATES = 2000


def _scores(model: QPlanningCritic, batch: dict[str, torch.Tensor],
            action_mean: torch.Tensor, action_std: torch.Tensor) -> torch.Tensor:
    roots, candidates = batch["action"].shape[:2]
    stock = batch["action"][:, :1]
    absolute_stock = (stock - action_mean) / action_std
    delta = (batch["action"] - stock) / action_std * DELTA_SCALE
    action_features = torch.cat([
        absolute_stock.expand_as(batch["action"]), delta,
    ], dim=-1)
    repeat = lambda x: x.repeat_interleave(candidates, 0)
    logits = model(
        repeat(batch["prefix"]), repeat(batch["pad"]),
        repeat(batch["robot"]), repeat(batch["proprio"]),
        action_features.reshape(roots * candidates, 10, 14),
        batch["action_valid"].reshape(roots * candidates, 10),
    ).reshape(roots, candidates)
    return logits - logits[:, :1]


@torch.no_grad()
def _evaluate(model: QPlanningCritic, roots: TimedRoots, device: torch.device,
              action_mean: torch.Tensor, action_std: torch.Tensor) -> dict:
    model.eval()
    selected = stock = rescues = spoils = pair_wins = pair_count = 0
    macro_pairs = []
    rescue_pairs = []
    score_spread = []
    for start in range(0, len(roots), 8):
        batch = _to(_root_batch([roots[i] for i in
                                range(start, min(start + 8, len(roots)))]), device)
        scores = _scores(model, batch, action_mean, action_std).cpu().numpy()
        labels = batch["success"].cpu().numpy().astype(bool)
        for values, outcome in zip(scores, labels):
            stock_success = bool(outcome[0])
            chosen_success = bool(outcome[int(values.argmax())])
            stock += stock_success
            selected += chosen_success
            rescues += int(not stock_success and chosen_success)
            spoils += int(stock_success and not chosen_success)
            score_spread.append(float(values.std()))
            good, bad = values[outcome], values[~outcome]
            if len(good) and len(bad):
                differences = good[:, None] - bad[None, :]
                wins = float((differences > 0).sum() + .5 * (differences == 0).sum())
                pair_wins += wins
                pair_count += differences.size
                accuracy = wins / differences.size
                macro_pairs.append(accuracy)
                if not stock_success:
                    rescue_pairs.append(accuracy)
    return {
        "roots": len(roots), "stock_successes": stock,
        "selected_successes": selected, "rescues": rescues, "spoils": spoils,
        "pair_accuracy": pair_wins / pair_count,
        "macro_mixed_root_pair_accuracy": float(np.mean(macro_pairs)),
        "rescueable_root_pair_accuracy": float(np.mean(rescue_pairs)),
        "within_root_score_std": float(np.mean(score_spread)),
    }


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("Mac Metal GPU required")
    _load_local_credentials()
    store = SupabaseStore()
    snapshot = load_or_create_combined_snapshot(store)
    cache = prepare_combined_root_cache(snapshot=snapshot, cache_root=ROOT / "cache", store=store)
    if cache["format"] != ROOT_CACHE_FORMAT:
        raise ValueError("delta ranker requires the pre-action root cache")
    groups = {g["candidate_group_id"]: g for g in snapshot["groups"]}
    train = TimedRoots(cache, cache["train_group_ids"], groups)
    validation = TimedRoots(cache, cache["validation_group_ids"], groups)
    rescue_indices = [i for i, row in enumerate(train.entries)
                      if row["mixed"] and not row["stock_success"]]
    other_indices = [i for i, row in enumerate(train.entries)
                     if row["mixed"] and row["stock_success"]]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    first = train[0]
    cfg = QPlanningModelConfig(
        action_horizon=10, action_dim=14, width=128, n_layers=2,
        n_heads=4, ffn_width=512, dropout=0, n_bins=2,
        prefix_pool_tokens=128)
    model = QPlanningCritic(
        prefix_dim=first["prefix"].shape[-1], robot_dim=len(first["robot"]),
        proprio_dim=len(first["proprio"]), config=cfg)
    model.value_head = nn.Linear(cfg.width, 1)
    device = torch.device("mps")
    model.to(device)
    mean, std = action_statistics(train)
    action_mean = torch.from_numpy(mean).to(device)
    action_std = torch.from_numpy(std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tracking = ROOT / "mlflow"
    mlflow.set_tracking_uri("sqlite:///" + str(tracking / "tracking.db"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (experiment.experiment_id if experiment else
                     client.create_experiment(EXPERIMENT,
                         artifact_location=(tracking / "artifacts").as_uri()))
    output = ROOT / "checkpoints" / EXPERIMENT / snapshot["snapshot_digest"]
    output.mkdir(parents=True, exist_ok=True)
    history = []
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name="stock-relative-scalar-pair-ranker") as run:
        url = f"http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run.info.run_id}"
        mlflow.log_params({
            "seed": SEED, "root_cache_format": ROOT_CACHE_FORMAT,
            "train_mixed_roots": len(rescue_indices) + len(other_indices),
            "train_rescueable_roots": len(rescue_indices),
            "delta_scale": DELTA_SCALE, "updates": UPDATES,
            "batch": 8, "lr": 3e-4, "dropout": 0,
            "model": "two-layer-scalar-stock-relative-action-ranker",
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        })
        print({"mlflow_run": url, "train_mixed_roots": len(rescue_indices)
               + len(other_indices)}, flush=True)
        initial = _evaluate(model, validation, device, action_mean, action_std)
        print({"update": 0, "validation": initial}, flush=True)
        mlflow.log_metrics({f"val/{k}": v for k, v in initial.items()}, step=0)
        for step in range(1, UPDATES + 1):
            model.train()
            rng = np.random.default_rng(SEED * 4_000_003 + step)
            indices = np.r_[rng.choice(rescue_indices, size=4, replace=False),
                            rng.choice(other_indices, size=4, replace=False)]
            batch = _to(_root_batch([train[int(i)] for i in indices]), device)
            optimizer.zero_grad(set_to_none=True)
            logits = _scores(model, batch, action_mean, action_std)
            loss = pairwise_root_loss(logits, batch["success"], temperature=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr = 3e-4 * .5 * (1 + math.cos(math.pi * step / UPDATES))
            for group in optimizer.param_groups:
                group["lr"] = lr
            if step % 250 == 0:
                train_metrics = _evaluate(model, train, device, action_mean, action_std)
                val_metrics = _evaluate(model, validation, device, action_mean, action_std)
                row = {"update": step, "train_loss": float(loss.detach()),
                       "train": train_metrics, "validation": val_metrics}
                history.append(row)
                mlflow.log_metrics({"train/loss": row["train_loss"],
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()}}, step=step)
                print(row, flush=True)
        checkpoint = output / "final.pt"
        torch.save({"snapshot_digest": snapshot["snapshot_digest"],
                    "root_cache_format": ROOT_CACHE_FORMAT,
                    "architecture": model.architecture_config(),
                    "action_mean": mean, "action_std": std,
                    "delta_scale": DELTA_SCALE, "model": model.state_dict()}, checkpoint)
        report = {"mlflow_run": url, "history": history,
                  "checkpoint": str(checkpoint)}
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(path))
        print({"report": str(path), "final": history[-1]}, flush=True)


if __name__ == "__main__":
    main()
