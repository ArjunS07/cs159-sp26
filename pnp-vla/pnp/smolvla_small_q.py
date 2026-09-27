"""Small sigmoid Q10 controls and paired diagnostics for the frozen fresh8 roots."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .smolvla_q_selection import (
    FRESH8_EXPERIMENT, FRESH8_KINDS, fresh8_snapshot_key,
    load_success_q10, paired_counts, score_root_trees,
)
from .smolvla_success_critic import TimedRoots, action_statistics
from .smolvla_tree_bellman_finetune import (
    _root_batch, _to, load_or_create_tree_snapshot, prepare_tree_bellman_cache,
)
from .store import SupabaseStore


FORMAT = "smolvla_small_sigmoid_q10_v1"


def load_fresh8_roots(*, store=None, cache_root: str | Path,
                      tree_limit: int = 800):
    store = store or SupabaseStore()
    snapshot = load_or_create_tree_snapshot(
        store=store, snapshot_key=fresh8_snapshot_key(tree_limit),
        tree_limit=tree_limit, experiment=FRESH8_EXPERIMENT,
        candidate_kinds=FRESH8_KINDS)
    cache = prepare_tree_bellman_cache(
        snapshot=snapshot, cache_root=cache_root, gamma=1.0,
        success_reward=True, store=store)
    groups = {row["candidate_group_id"]: row for row in snapshot["groups"]}
    train = TimedRoots(cache, cache["train_group_ids"], groups)
    validation = TimedRoots(cache, cache["validation_group_ids"], groups)
    return snapshot, cache, train, validation


class SmallSuccessQ(nn.Module):
    """Masked-mean frozen prefix, robot/proprio, flattened actions, scalar logit."""

    def __init__(self, prefix_dim: int, robot_dim: int, proprio_dim: int,
                 *, width: int = 128, state_only: bool = False,
                 dropout: float = .10):
        super().__init__()
        self.prefix_dim = int(prefix_dim)
        self.robot_dim = int(robot_dim)
        self.proprio_dim = int(proprio_dim)
        self.width = int(width)
        self.state_only = bool(state_only)
        self.dropout = float(dropout)
        self.prefix = nn.Sequential(nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, width), nn.GELU())
        self.robot = nn.Sequential(nn.LayerNorm(robot_dim), nn.Linear(robot_dim, width // 2), nn.GELU())
        self.proprio = nn.Sequential(nn.LayerNorm(proprio_dim), nn.Linear(proprio_dim, width // 2), nn.GELU())
        self.action = None if state_only else nn.Sequential(
            nn.Linear(70, width), nn.LayerNorm(width), nn.GELU())
        fused = 2 * width + (0 if state_only else width)
        self.head = nn.Sequential(nn.Linear(fused, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, 1))
        self.register_buffer("action_mean", torch.zeros(7))
        self.register_buffer("action_std", torch.ones(7))

    def architecture(self):
        return dict(prefix_dim=self.prefix_dim, robot_dim=self.robot_dim,
                    proprio_dim=self.proprio_dim, width=self.width,
                    state_only=self.state_only, dropout=self.dropout)

    def forward(self, prefix, pad, robot, proprio, action, action_valid):
        if action.shape[-2:] != (10, 7) or action_valid.shape != action.shape[:2]:
            raise ValueError("small Q requires ten seven-dimensional actions")
        mask = pad.float().unsqueeze(-1)
        pooled = (prefix.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        parts = [self.prefix(pooled), self.robot(robot.float()),
                 self.proprio(proprio.float())]
        if self.action is not None:
            normalized = ((action.float() - self.action_mean) / self.action_std)
            normalized = normalized * action_valid.float().unsqueeze(-1)
            parts.append(self.action(normalized.flatten(1)))
        return self.head(torch.cat(parts, 1)).squeeze(-1)


def small_root_scores(model: SmallSuccessQ, batch: dict) -> torch.Tensor:
    roots, candidates = batch["action"].shape[:2]
    repeat = lambda value: value.repeat_interleave(candidates, 0)
    logits = model(
        repeat(batch["prefix"]), repeat(batch["pad"]),
        repeat(batch["robot"]), repeat(batch["proprio"]),
        batch["action"].reshape(roots * candidates, 10, 7),
        batch["action_valid"].reshape(roots * candidates, 10))
    return logits.reshape(roots, candidates)


@torch.no_grad()
def score_small_roots(model, roots, *, device="cpu"):
    model.eval()
    scores, outcomes = [], []
    device = torch.device(device)
    for start in range(0, len(roots), 16):
        batch = _to(_root_batch([roots[i] for i in
                                range(start, min(start + 16, len(roots)))]), device)
        scores.append(small_root_scores(model, batch).sigmoid().cpu().numpy())
        outcomes.append(batch["success"].cpu().numpy().astype(bool))
    return np.concatenate(scores), np.concatenate(outcomes)


def root_diagnostics(scores: np.ndarray, outcomes: np.ndarray) -> dict:
    """Paired diagnostics; root-level averages avoid treating branches as IID."""
    q = np.asarray(scores, np.float64)
    y = np.asarray(outcomes, bool)
    if q.shape != y.shape or q.ndim != 2 or q.shape[1] != 9 or not np.isfinite(q).all():
        raise ValueError("need finite [roots,9] scores/outcomes")
    mixed = y.any(1) & (~y).any(1)
    pair_acc = []
    for values, labels in zip(q[mixed], y[mixed]):
        differences = values[labels, None] - values[None, ~labels]
        pair_acc.append(float(((differences > 0) + .5 * (differences == 0)).mean()))
    selected = q.argmax(1)
    margin = q[:, 1:].max(1) - q[:, 0]
    gate = {}
    for threshold in (0.0, .01, .02, .05, .10, .20):
        choice = np.where(margin > threshold, q[:, 1:].argmax(1) + 1, 0)
        gate[str(threshold)] = paired_counts(y, choice)
    return {
        "roots": len(q), "mixed_roots": int(mixed.sum()),
        "oracle_successes": int(y.any(1).sum()),
        "original_failures_with_successful_alternative": int((~y[:, 0] & y[:, 1:].any(1)).sum()),
        "argmax": paired_counts(y, selected),
        "mean_root_pair_accuracy": float(np.mean(pair_acc)) if pair_acc else float("nan"),
        "within_root_score_std_mean": float(q.std(1).mean()),
        "between_root_mean_score_std": float(q.mean(1).std()),
        "brier": float(np.square(q - y).mean()),
        "constant_prevalence_brier": float(np.square(y.mean() - y).mean()),
        "gate_exploratory": gate,
    }


def analyze_existing_checkpoint(*, checkpoint: str | Path,
                                validation: TimedRoots, snapshot: dict,
                                device="cuda") -> dict:
    model, payload = load_success_q10(
        checkpoint, expected_snapshot=snapshot["snapshot_digest"], device=device)
    scores, outcomes = score_root_trees(model, validation, device=device)
    report = root_diagnostics(scores, outcomes)
    report.update({"checkpoint": str(checkpoint), "update": payload["update"],
                   "snapshot_digest": snapshot["snapshot_digest"]})
    return report


@dataclass(frozen=True)
class SmallTrainConfig:
    updates: int = 2000
    batch_roots: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    seed: int = 42
    eval_interval: int = 250


def train_small_q(*, train: TimedRoots, validation: TimedRoots,
                  snapshot_digest: str, output_path: str | Path,
                  state_only: bool = False, device="cuda",
                  config: SmallTrainConfig = SmallTrainConfig()):
    """Same frozen root split and Bernoulli labels as the larger root-MC arm."""
    torch.manual_seed(config.seed)
    device = torch.device(device)
    first = train[0]
    model = SmallSuccessQ(
        first["prefix"].shape[-1], len(first["robot"]), len(first["proprio"]),
        state_only=state_only)
    mean, std = action_statistics(train)
    model.action_mean.copy_(torch.as_tensor(mean))
    model.action_std.copy_(torch.as_tensor(std))
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    history = []
    for update in range(1, config.updates + 1):
        model.train()
        indices = rng.choice(len(train), size=config.batch_roots,
                             replace=len(train) < config.batch_roots)
        batch = _to(_root_batch([train[int(i)] for i in indices]), device)
        logits = small_root_scores(model, batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["success"].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % config.eval_interval == 0 or update == config.updates:
            scores, labels = score_small_roots(model, validation, device=device)
            metrics = root_diagnostics(scores, labels)
            record = {"update": update, "train_loss": float(loss.detach()),
                      "validation": {key: metrics[key] for key in (
                          "mixed_roots", "oracle_successes", "argmax",
                          "mean_root_pair_accuracy", "brier")}}
            history.append(record)
            print(record, flush=True)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save({"format": FORMAT, "objective": "success_probability_after_one_intervention",
                "continuation_policy": "smolvla_pnp_steps123_k311",
                "snapshot_digest": snapshot_digest,
                "representation": "frozen_prefill_masked_mean_v1",
                "architecture": model.architecture(),
                "state_only": state_only, "parameters": parameter_count,
                "updates": config.updates, "history": history,
                "model": {key: value.detach().cpu()
                          for key, value in model.state_dict().items()}}, temporary)
    os.replace(temporary, output)
    return {"checkpoint": str(output), "parameters": parameter_count,
            "final_validation": history[-1]["validation"], "history": history}
