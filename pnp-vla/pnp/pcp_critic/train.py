"""Offline PCP critic training and deterministic evaluation."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import io
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import PCPCriticTrainConfig
from .data import PCPCriticTransition
from .model import PCPCritic
from .objectives import calibrated_conservative_loss, expectile_loss, td_target


class TransitionDataset(Dataset):
    def __init__(self, transitions: Iterable[PCPCriticTransition]):
        self.transitions = list(transitions)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, index):
        item = self.transitions[index]
        return item


def _pad_prefix(items, attr: str, mask_attr: str) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = [np.asarray(getattr(item, attr), np.float32) for item in items]
    masks = [np.asarray(getattr(item, mask_attr), bool) for item in items]
    width = {value.shape[-1] for value in arrays}
    if len(width) != 1:
        raise ValueError("mixed frozen-prefix widths require separate snapshots")
    length = max(len(value) for value in arrays)
    out = torch.zeros(len(items), length, next(iter(width)), dtype=torch.float32)
    pad = torch.zeros(len(items), length, dtype=torch.bool)
    for index, (value, mask) in enumerate(zip(arrays, masks)):
        out[index, :len(value)] = torch.from_numpy(value)
        pad[index, :len(mask)] = torch.from_numpy(mask)
    return out, pad


def collate_transitions(items: list[PCPCriticTransition]) -> dict[str, torch.Tensor]:
    prefix, pad = _pad_prefix(items, "prefix_embeddings", "prefix_pad_mask")
    next_prefix, next_pad = _pad_prefix(items, "next_prefix_embeddings", "next_prefix_pad_mask")
    return {
        "prefix": prefix, "pad": pad,
        "robot": torch.from_numpy(np.stack([item.robot_state for item in items]).astype(np.float32)),
        "proprio": torch.from_numpy(np.stack([item.proprio for item in items]).astype(np.float32)),
        "action": torch.from_numpy(np.stack([item.action for item in items]).astype(np.float32)),
        "next_prefix": next_prefix, "next_pad": next_pad,
        "next_robot": torch.from_numpy(np.stack([item.next_robot_state for item in items]).astype(np.float32)),
        "next_proprio": torch.from_numpy(np.stack([item.next_proprio for item in items]).astype(np.float32)),
        "next_action": torch.from_numpy(np.stack([item.next_action for item in items]).astype(np.float32)),
        "reward": torch.tensor([item.reward for item in items], dtype=torch.float32),
        "discount": torch.tensor([item.discount for item in items], dtype=torch.float32),
        "mc_return": torch.tensor([item.mc_return for item in items], dtype=torch.float32),
    }


def _loader(transitions, cfg, *, shuffle: bool) -> DataLoader:
    return DataLoader(TransitionDataset(transitions), batch_size=cfg.batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0, collate_fn=collate_transitions)


def _to(batch: dict, device):
    return {key: value.to(device) for key, value in batch.items()}


def _ema(target, source, rate: float) -> None:
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(parameter, rate)


def critic_loss(model, target, batch, cfg: PCPCriticTrainConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_q = td_target(target, batch, use_iql_value=cfg.objective == "iql")
    q = model.q_values(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"], batch["action"])
    td = sum(F.smooth_l1_loss(member, target_q) for member in q)
    metrics = {"td_loss": td.detach(), "q_mean": q.amin(0).mean().detach(),
               "target_mean": target_q.mean().detach()}
    if cfg.objective == "iql":
        value = model.state_value(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"])
        value_loss = expectile_loss(q.detach().amin(0) - value, cfg.expectile)
        metrics["value_loss"] = value_loss.detach()
        return td + value_loss, metrics
    conservative = calibrated_conservative_loss(
        model, batch, n_local=cfg.n_local_actions, n_broad=cfg.n_broad_actions,
        local_std=cfg.local_action_std, weight=cfg.conservative_weight)
    metrics["conservative_loss"] = conservative.detach()
    return td + conservative, metrics


@torch.no_grad()
def evaluate_critic(model, target, transitions, device, *, config: PCPCriticTrainConfig) -> dict:
    if not transitions:
        return {"loss": float("nan"), "n_transitions": 0}
    model.eval(); target.eval()
    values, targets, losses = [], [], []
    for raw in _loader(transitions, config, shuffle=False):
        batch = _to(raw, device)
        target_q = td_target(target, batch, use_iql_value=config.objective == "iql")
        q = model.q_values(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"], batch["action"]).amin(0)
        values.extend(q.cpu().tolist()); targets.extend(target_q.cpu().tolist())
        losses.append(float(F.smooth_l1_loss(q, target_q).cpu()))
    values, targets = np.asarray(values), np.asarray(targets)
    return {"loss": float(np.mean(losses)), "n_transitions": len(values),
            "q_mean": float(values.mean()), "target_mean": float(targets.mean()),
            "td_rmse": float(np.sqrt(np.mean((values - targets) ** 2))),
            "q_target_correlation": (float(np.corrcoef(values, targets)[0, 1])
                                     if len(values) > 1 and np.std(values) and np.std(targets) else float("nan"))}


def train_critic(model: PCPCritic, train_transitions, val_transitions, device, *,
                 config: PCPCriticTrainConfig | None = None) -> tuple[PCPCritic, dict]:
    cfg = config or PCPCriticTrainConfig()
    if not train_transitions:
        raise ValueError("cannot train a PCP critic with zero transitions")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    model = model.to(device)
    target = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loader = _loader(train_transitions, cfg, shuffle=True)
    iterator = iter(loader)
    best_loss, best_state, best_update, bad, history = float("inf"), None, 0, 0, []
    for update in range(1, cfg.updates + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader); raw = next(iterator)
        batch = _to(raw, device)
        model.train()
        loss, metrics = critic_loss(model, target, batch, cfg)
        optimizer.zero_grad(); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step(); _ema(target, model, cfg.target_rate)
        if update % cfg.eval_interval == 0 or update == cfg.updates:
            validation = evaluate_critic(model, target, val_transitions, device, config=cfg)
            record = {"update": update, "train_loss": float(loss.detach()),
                      "grad_norm": float(grad_norm), **{key: float(value) for key, value in metrics.items()},
                      "validation": validation}
            history.append(record)
            # With a one-group smoke snapshot validation may be empty. Preserve the final model.
            score = validation["loss"] if np.isfinite(validation["loss"]) else float(loss.detach())
            if score < best_loss:
                best_loss, best_state, best_update, bad = score, copy.deepcopy(model.state_dict()), update, 0
            else:
                bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate_critic(model, target, val_transitions, device, config=cfg)
    return model, {"objective": cfg.objective, "updates_ran": update, "best_update": best_update,
                   "best_validation_loss": best_loss, "validation": final, "history": history}


def transition_hash(transitions: Iterable[PCPCriticTransition]) -> str:
    digest = hashlib.sha256()
    for item in sorted(transitions, key=lambda value: (value.rollout_id, value.chunk_idx)):
        digest.update(f"{item.rollout_id}|{item.chunk_idx}|{item.reward}|{item.discount}".encode())
        digest.update(np.asarray(item.action, np.float32).tobytes())
    return digest.hexdigest()[:24]


def checkpoint_bytes(model: PCPCritic, snapshot_id: str, train_config: PCPCriticTrainConfig,
                     metrics: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save({"format": "pcp_critic_v1", "architecture": model.architecture_config(),
                "state_dict": model.cpu().state_dict(), "snapshot_id": snapshot_id,
                "train_config": asdict(train_config), "metrics": metrics}, buffer)
    return buffer.getvalue()
