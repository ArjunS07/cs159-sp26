"""Training utilities for the two-stage hybrid chunk critic."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import io

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import ChunkTransitionExample
from .train import AdvantageTrainConfig, evaluate_candidate_ranker


@dataclass
class HybridCriticTrainConfig:
    seed: int = 42
    gamma: float = .999
    expectile: float = .7
    distill_expectile: float = .8
    rank_weight: float = 1.0
    mc_weight: float = 1.0
    distill_weight: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    long_updates: int = 20_000
    eval_interval: int = 250
    long_patience: int = 8
    short_epochs: int = 50
    short_patience: int = 7
    target_rate: float = .005
    prefix_length: int = 10
    zero_context: bool = False


class TransitionDataset(Dataset):
    def __init__(self, examples):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        return {
            "obs": torch.from_numpy(e.obs_enc),
            "next_obs": torch.from_numpy(e.next_obs_enc),
            "actions": torch.from_numpy(e.actions),
            "mask": torch.from_numpy(e.action_mask),
            "position": torch.tensor(e.chunk_position, dtype=torch.float32),
            "next_position": torch.tensor(e.next_chunk_position, dtype=torch.float32),
            "reward": torch.tensor(e.reward, dtype=torch.float32),
            "discount": torch.tensor(e.discount, dtype=torch.float32),
            "return_target": torch.tensor(e.return_target, dtype=torch.float32),
        }


def transition_dataset_hash(examples):
    digest = hashlib.sha256()
    for e in sorted(examples, key=lambda x: (x.rollout_id, x.chunk_idx)):
        digest.update(f"{e.rollout_id}|{e.chunk_idx}|{e.reward}|{e.discount}".encode())
        digest.update(np.asarray(e.actions, dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def _expectile_loss(residual, expectile):
    weight = torch.where(residual >= 0, expectile, 1 - expectile)
    return (weight * residual.square()).mean()


def _loader(examples, config, *, shuffle):
    return DataLoader(TransitionDataset(examples), batch_size=config.batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=False)


def _to(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _ema(target, source, rate):
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(parameter, rate)


@torch.no_grad()
def evaluate_long_critic(model, examples, device, config):
    model.eval()
    losses = []
    for raw in _loader(examples, config, shuffle=False):
        b = _to(raw, device)
        q = model.long_values(b["obs"], b["position"], b["actions"], b["mask"])
        target = b["reward"] + b["discount"] * model.state_value(
            b["next_obs"], b["next_position"])
        q_loss = sum(F.smooth_l1_loss(member, target) for member in q)
        minimum = q.amin(0)
        value_loss = _expectile_loss(
            minimum - model.state_value(b["obs"], b["position"]), config.expectile)
        losses.append(float((q_loss + value_loss).cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_long_critic(model, train_examples, val_examples, device, *, config=None):
    """Fit twin Qc and V with IQL-style expectile value regression."""
    config = config or HybridCriticTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(device)
    target = copy.deepcopy(model).to(device).eval()
    parameters = [*model.long_critics.parameters(), *model.value_network.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    loader = _loader(train_examples, config, shuffle=True)
    iterator = iter(loader)
    best_loss, best_state, best_update, bad = float("inf"), None, 0, 0
    history = []
    for update in range(1, config.long_updates + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        b = _to(raw, device)
        with torch.no_grad():
            td_target = b["reward"] + b["discount"] * target.state_value(
                b["next_obs"], b["next_position"])
        q = model.long_values(b["obs"], b["position"], b["actions"], b["mask"])
        q_loss = sum(F.smooth_l1_loss(member, td_target) for member in q)
        with torch.no_grad():
            q_target = target.long_values(
                b["obs"], b["position"], b["actions"], b["mask"]).amin(0)
        value_loss = _expectile_loss(
            q_target - model.state_value(b["obs"], b["position"]), config.expectile)
        loss = q_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        _ema(target, model, config.target_rate)
        if update % config.eval_interval == 0 or update == config.long_updates:
            val_loss = evaluate_long_critic(model, val_examples, device, config)
            history.append({"update": update, "train_loss": float(loss.detach()),
                            "validation_loss": val_loss})
            if val_loss < best_loss:
                best_loss, best_update, bad = val_loss, update, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if bad >= config.long_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_validation_loss": best_loss, "best_update": best_update,
                   "updates_ran": update, "history": history}


def _candidate_groups(examples):
    groups = defaultdict(list)
    for example in examples:
        if example.candidate_group_id:
            groups[example.candidate_group_id].append(example)
    return list(groups.values())


def _candidate_loss(model, members, device, config):
    first = members[0]
    obs = torch.from_numpy(first.obs_enc).unsqueeze(0).to(device)
    position = torch.tensor([first.chunk_position], dtype=torch.float32, device=device)
    actions = torch.stack([torch.from_numpy(e.actions[:config.prefix_length])
                           for e in members]).to(device)
    masks = torch.stack([torch.from_numpy(e.action_mask[:config.prefix_length])
                         for e in members]).to(device)
    obs = obs.expand(len(members), -1)
    position = position.expand(len(members))
    q = model.short_values(obs, position, actions, masks,
                           zero_context=config.zero_context)
    target = torch.tensor([
        e.return_target if e.return_target is not None else float(e.success)
        for e in members], dtype=torch.float32, device=device)
    mc = sum(F.smooth_l1_loss(member, target) for member in q)
    labels = torch.tensor([e.success for e in members], dtype=torch.bool, device=device)
    ranking = torch.zeros((), device=device)
    comparisons = 0
    for member in q:
        positive, negative = member[labels], member[~labels]
        if len(positive) and len(negative):
            ranking = ranking + F.softplus(
                -(positive[:, None] - negative[None, :])).mean()
            comparisons += 1
    return config.mc_weight * mc + config.rank_weight * ranking, comparisons


def _distillation_loss(model, batch, device, config):
    b = _to(batch, device)
    with torch.no_grad():
        target = model.long_values(
            b["obs"], b["position"], b["actions"], b["mask"]).amin(0)
    short_actions = b["actions"][:, :config.prefix_length]
    short_mask = b["mask"][:, :config.prefix_length]
    short = model.short_values(b["obs"], b["position"], short_actions, short_mask,
                               zero_context=config.zero_context)
    return sum(_expectile_loss(target - member, config.distill_expectile)
               for member in short)


def train_short_critic(model, historical, candidate_train, candidate_val, device, *,
                       config=None):
    """Distill Qc into Qa while fitting same-state causal outcomes and rankings."""
    config = config or HybridCriticTrainConfig()
    for parameter in (*model.long_critics.parameters(), *model.value_network.parameters()):
        parameter.requires_grad_(False)
    parameters = list(model.short_critics.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    historical_loader = _loader(historical, config, shuffle=True)
    groups = _candidate_groups(candidate_train)
    eval_config = AdvantageTrainConfig(
        seed=config.seed, prefix_length=config.prefix_length,
        zero_context=config.zero_context)
    rng = np.random.default_rng(config.seed)
    best_score, best_state, best_epoch, bad = -float("inf"), None, 0, 0
    history = []
    for epoch in range(config.short_epochs):
        model.train()
        batches = iter(historical_loader)
        losses = []
        for index in rng.permutation(len(groups)):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(historical_loader)
                batch = next(batches)
            distill = _distillation_loss(model, batch, device, config)
            candidate, _ = _candidate_loss(model, groups[index], device, config)
            loss = config.distill_weight * distill + candidate
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_candidate_ranker(
            model, candidate_val, device, config=eval_config, n_bootstrap=1000)
        score = metrics["group_macro_ranking_accuracy"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "ranking": score})
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= config.short_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_ranking": best_score, "best_epoch": best_epoch,
                   "epochs_ran": epoch + 1, "history": history}


def hybrid_checkpoint_bytes(model, *, config, metadata=None):
    buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict(),
                "architecture": model.architecture_config(),
                "train_config": asdict(config), "metadata": metadata or {}}, buffer)
    return buffer.getvalue()
