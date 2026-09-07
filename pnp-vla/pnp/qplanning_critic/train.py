"""Fixed-schedule offline training for the Q10/Q50 comparison."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import copy
import hashlib
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .config import QPlanningTrainConfig
from .data import QPlanningWindowDataset, collate_windows
from .model import QPlanningCritic


class RolloutBatchSampler(Sampler[list[int]]):
    """Shuffle rollouts and offsets while keeping disk reads locally grouped."""

    def __init__(self, dataset: QPlanningWindowDataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = defaultdict(list)
        for index, (entry, _) in enumerate(self.dataset.indices):
            groups[entry["rollout_id"]].append(index)
        group_ids = list(groups)
        generator.shuffle(group_ids)
        pending = []
        for group_id in group_ids:
            values = groups[group_id]
            generator.shuffle(values)
            pending.extend(values)
            while len(pending) >= self.batch_size:
                yield pending[:self.batch_size]
                pending = pending[self.batch_size:]
        if pending:
            yield pending


def _loader(dataset, batch_size: int, *, shuffle: bool, seed: int = 0) -> DataLoader:
    if shuffle:
        return DataLoader(dataset, batch_sampler=RolloutBatchSampler(dataset, batch_size, seed),
                          num_workers=0, collate_fn=collate_windows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_windows)


def _to(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _ema(target: QPlanningCritic, online: QPlanningCritic, rate: float) -> None:
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), online.parameters()):
            target_parameter.lerp_(parameter, rate)
        for target_buffer, buffer in zip(target.buffers(), online.buffers()):
            if target_buffer.dtype.is_floating_point:
                target_buffer.lerp_(buffer, rate)
            else:
                target_buffer.copy_(buffer)


def _target_value(target: QPlanningCritic, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        next_value = target.expected_value(
            batch["next_prefix"], batch["next_pad"], batch["next_robot"],
            batch["next_proprio"], batch["next_action"], batch["next_action_valid"])
        return (batch["reward"] + batch["discount"] * next_value).clamp(
            target.config.value_min, target.config.value_max)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


@torch.no_grad()
def evaluate_qplanning_critic(model: QPlanningCritic, target: QPlanningCritic,
                              dataset: QPlanningWindowDataset, device,
                              *, batch_size: int) -> dict:
    model.eval(); target.eval()
    losses, predictions, bellman, monte_carlo, successes = [], [], [], [], []
    for raw in _loader(dataset, batch_size, shuffle=False):
        batch = _to(raw, device)
        target_value = _target_value(target, batch)
        logits = model(
            batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
            batch["action"], batch["action_valid"])
        losses.append(float(model.categorical_loss(logits, target_value).cpu()))
        predictions.extend((logits.softmax(-1) * model.value_bins).sum(-1).float().cpu().tolist())
        bellman.extend(target_value.float().cpu().tolist())
        monte_carlo.extend(batch["mc_return"].float().cpu().tolist())
        successes.extend(batch["success"].cpu().tolist())
    prediction = np.asarray(predictions)
    td = np.asarray(bellman)
    mc = np.asarray(monte_carlo)
    success = np.asarray(successes, bool)
    return {
        "hl_gauss_ce": float(np.mean(losses)),
        "bellman_mae": float(np.mean(np.abs(prediction - td))),
        "return_mae": float(np.mean(np.abs(prediction - mc))),
        "q_success": float(prediction[success].mean()) if success.any() else float("nan"),
        "q_failure": float(prediction[~success].mean()) if (~success).any() else float("nan"),
        "failure_auc": _auc(~success, -prediction),
        "n_transitions": int(len(prediction)),
    }


def _cpu_state(module) -> dict:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _save_checkpoint(path: Path, *, model: QPlanningCritic, target: QPlanningCritic,
                     optimizer, update: int, snapshot_id: str, cache_digest: str,
                     source_policy: dict,
                     train_config: QPlanningTrainConfig, history: list[dict]) -> None:
    payload = {
        "format": "qplanning_critic_v1", "update": update,
        "snapshot_id": snapshot_id, "cache_digest": cache_digest,
        "source_policy": source_policy,
        "architecture": model.architecture_config(), "train_config": train_config.to_dict(),
        "model": _cpu_state(model), "target": _cpu_state(target),
        "optimizer": optimizer.state_dict(), "history": history,
        "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, *, model, target, optimizer,
                     snapshot_id: str, cache_digest: str) -> tuple[int, list[dict]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "qplanning_critic_v1":
        raise ValueError(f"unsupported checkpoint format in {path}")
    if payload.get("snapshot_id") != snapshot_id or payload.get("cache_digest") != cache_digest:
        raise ValueError("checkpoint data contract differs from the requested snapshot/cache")
    if payload.get("architecture") != model.architecture_config():
        raise ValueError("checkpoint architecture differs from the requested model")
    model.load_state_dict(payload["model"])
    target.load_state_dict(payload["target"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return int(payload["update"]), list(payload.get("history", []))


def _latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = list(directory.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def train_qplanning_critic(model: QPlanningCritic,
                           train_dataset: QPlanningWindowDataset,
                           val_dataset: QPlanningWindowDataset, device, *,
                           snapshot_id: str, cache_digest: str,
                           source_policy: dict,
                           output_dir: str | Path,
                           config: QPlanningTrainConfig,
                           resume: bool = True) -> tuple[QPlanningCritic, QPlanningCritic, dict]:
    if not len(train_dataset):
        raise ValueError("training dataset is empty")
    torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)
    device = torch.device(device)
    model = model.to(device)
    target = copy.deepcopy(model).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir = Path(output_dir)
    start_update, history = 0, []
    latest = _latest_checkpoint(output_dir) if resume else None
    if latest is not None:
        start_update, history = _load_checkpoint(
            latest, model=model, target=target, optimizer=optimizer,
            snapshot_id=snapshot_id, cache_digest=cache_digest)
        model.to(device); target.to(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        print(f"[qplanning] resumed {latest.name} at step {start_update}")
    loader = _loader(train_dataset, config.micro_batch_size, shuffle=True, seed=config.seed)
    iterator = iter(loader)
    use_amp = bool(config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    started = time.perf_counter()
    rolling = defaultdict(float)
    rolling_count = 0
    for update in range(start_update + 1, config.updates + 1):
        learning_rate = config.learning_rate_at(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.accumulation_steps):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = _to(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                target_value = _target_value(target, batch)
                logits = model(
                    batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
                    batch["action"], batch["action_valid"])
                loss = model.categorical_loss(logits, target_value) / config.accumulation_steps
            loss.backward()
            prediction = (logits.detach().float().softmax(-1) * model.value_bins).sum(-1)
            rolling["ce"] += float(loss.detach()) * config.accumulation_steps
            rolling["q"] += float(prediction.mean())
            rolling["target"] += float(target_value.mean())
            rolling_count += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        _ema(target, model, config.target_rate)
        if update % config.print_interval == 0 or update == config.updates:
            elapsed = time.perf_counter() - started
            completed = update - start_update
            eta = elapsed / max(1, completed) * (config.updates - update)
            gpu = (torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0)
            print(
                f"Q{model.config.action_horizon} step {update}/{config.updates} | "
                f"train_ce {rolling['ce']/rolling_count:.4f} | "
                f"q_mean {rolling['q']/rolling_count:.4f} | "
                f"target {rolling['target']/rolling_count:.4f} | grad {float(grad_norm):.3f} | "
                f"lr {learning_rate:.2e} | GPU {gpu:.1f} GB | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")
            rolling.clear(); rolling_count = 0
        if update % config.eval_interval == 0 or update == config.updates:
            validation = evaluate_qplanning_critic(
                model, target, val_dataset, device, batch_size=config.micro_batch_size)
            record = {"update": update, "validation": validation}
            history.append(record)
            print(
                "validation | HL-Gauss CE {hl_gauss_ce:.4f} | Q MAE {return_mae:.4f} | "
                "Q(success) {q_success:.4f} | Q(failure) {q_failure:.4f} | "
                "failure AUC {failure_auc:.3f} | n {n_transitions}".format(**validation))
        if update % config.checkpoint_interval == 0 or update == config.updates:
            path = output_dir / f"checkpoint_step_{update:06d}.pt"
            _save_checkpoint(
                path, model=model, target=target, optimizer=optimizer, update=update,
                snapshot_id=snapshot_id, cache_digest=cache_digest,
                source_policy=source_policy,
                train_config=config, history=history)
            print(f"[qplanning] saved {path}")
    final_validation = evaluate_qplanning_critic(
        model, target, val_dataset, device, batch_size=config.micro_batch_size)
    report = {
        "snapshot_id": snapshot_id, "cache_digest": cache_digest,
        "horizon": model.config.action_horizon, "updates": config.updates,
        "train_windows": len(train_dataset), "validation_windows": len(val_dataset),
        "validation": final_validation, "history": history,
        "final_checkpoint": str(output_dir / f"checkpoint_step_{config.updates:06d}.pt"),
    }
    return model, target, report
