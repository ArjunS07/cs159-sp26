"""Training, evaluation, and calibration for clean-chunk verifiers."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import CleanChunkExample


@dataclass
class VerifierTrainConfig:
    seed: int = 42
    prefix_length: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_rollouts: int = 32
    epochs: int = 100
    patience: int = 15
    state_loss_weight: float = 0.25
    ranking_loss_weight: float = 0.50
    score_head: str = "joint"                 # joint | state
    zero_observation: bool = False             # action-only diagnostic


class CleanChunkDataset(Dataset):
    def __init__(self, examples):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        return {
            "obs": torch.from_numpy(e.obs_enc), "actions": torch.from_numpy(e.actions),
            "mask": torch.from_numpy(e.action_mask),
            "position": torch.tensor(e.chunk_position, dtype=torch.float32),
            "label": torch.tensor(e.success, dtype=torch.float32),
            "rollout_id": e.rollout_id, "task_key": "|".join(map(str, e.task_key)),
            "suite": e.suite,
            "candidate_group_id": e.candidate_group_id or "",
        }


class DiscordantPairDataset(Dataset):
    def __init__(self, examples):
        groups = defaultdict(list)
        for example in examples or []:
            # Both modes hold the scored observation fixed across candidates.
            # ``snapshot`` branches mid-rollout; ``paired_full_episode`` is the
            # conservative fallback that branches from the identical episode
            # initial state when simulator snapshots cannot be restored.
            if (example.candidate_group_id and
                    example.pairing_mode in {"snapshot", "paired_full_episode"}):
                groups[example.candidate_group_id].append(example)
        self.pairs = []
        for members in groups.values():
            positive = [e for e in members if e.success]
            negative = [e for e in members if not e.success]
            self.pairs.extend((p, n) for p in positive for n in negative)

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _tensor(e):
        return (torch.from_numpy(e.obs_enc), torch.from_numpy(e.actions),
                torch.from_numpy(e.action_mask), torch.tensor(e.chunk_position, dtype=torch.float32))

    def __getitem__(self, index):
        positive, negative = self.pairs[index]
        po, pa, pm, pp = self._tensor(positive)
        no, na, nm, np_ = self._tensor(negative)
        return {"pos_obs": po, "pos_actions": pa, "pos_mask": pm, "pos_position": pp,
                "neg_obs": no, "neg_actions": na, "neg_mask": nm, "neg_position": np_}


def _sample_weights(examples):
    chunks_per_rollout = Counter(e.rollout_id for e in examples)
    rollout_record = {e.rollout_id: e for e in examples}
    stratum_rollouts = Counter((e.task_key, e.success) for e in rollout_record.values())
    weights = [1.0 / chunks_per_rollout[e.rollout_id] / stratum_rollouts[(e.task_key, e.success)]
               for e in examples]
    return np.asarray(weights, dtype=np.float64)


def _loader(examples, config, train=False):
    dataset = CleanChunkDataset(examples)
    sampler = None
    if train:
        weights = _sample_weights(examples)
        generator = torch.Generator().manual_seed(config.seed)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=config.batch_rollouts, sampler=sampler,
                      shuffle=False, num_workers=0)


def _average_precision(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if use.any():
            total += use.mean() * abs(y[use].mean() - p[use].mean())
    return float(total)


def _metrics(labels, probabilities, task_keys):
    from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
    y, p = np.asarray(labels), np.asarray(probabilities)
    grouped = defaultdict(list)
    for i, task in enumerate(task_keys):
        grouped[task].append(i)
    task_ap, task_auc = [], []
    for indices in grouped.values():
        yi, pi = y[indices], p[indices]
        if len(np.unique(yi)) > 1:
            task_ap.append(_average_precision(yi, pi))
            task_auc.append(float(roc_auc_score(yi, pi)))
    predicted_failure = p < 0.5
    actual_failure = y == 0
    return {
        "pr_auc": _average_precision(y, p),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "task_macro_pr_auc": float(np.mean(task_ap)) if task_ap else float("nan"),
        "task_macro_roc_auc": float(np.mean(task_auc)) if task_auc else float("nan"),
        "brier": float(brier_score_loss(y, p)), "ece": _ece(y, p),
        "failure_precision": float(precision_score(actual_failure, predicted_failure,
                                                    zero_division=0)),
        "failure_recall": float(recall_score(actual_failure, predicted_failure, zero_division=0)),
    }


def classification_metrics(labels, probabilities, task_keys):
    """Public metric helper for empirical-prior and other non-neural baselines."""
    return _metrics(labels, probabilities, task_keys)


def _forward(model, batch, device, config):
    obs = batch["obs"].to(device)
    if config.zero_observation:
        obs = torch.zeros_like(obs)
    return model(obs, batch["actions"].to(device), batch["mask"].to(device),
                 batch["position"].to(device), config.prefix_length)


def _chosen_logit(output, config):
    if config.score_head == "state":
        return output.state_logit
    if config.score_head != "joint":
        raise ValueError("score_head must be 'joint' or 'state'")
    return output.joint_logit


@torch.no_grad()
def evaluate_verifier(model, examples, device, *, config=None, scaler=None, paired_examples=None):
    config = config or VerifierTrainConfig()
    model.eval()
    labels, probs, tasks, suites = [], [], [], []
    for batch in _loader(examples, config):
        output = _forward(model, batch, device, config)
        logits = _chosen_logit(output, config)
        if scaler is not None:
            logits = scaler(logits)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
        tasks.extend(batch["task_key"])
        suites.extend(batch["suite"])
    metrics = _metrics(labels, probs, tasks)
    metrics["per_suite"] = {
        suite: _metrics([labels[i] for i in indices], [probs[i] for i in indices],
                        [tasks[i] for i in indices])
        for suite in sorted(set(suites))
        for indices in [[i for i, value in enumerate(suites) if value == suite]]
    }
    pair_dataset = DiscordantPairDataset(paired_examples)
    if len(pair_dataset):
        correct, total = 0, 0
        for batch in DataLoader(pair_dataset, batch_size=config.batch_rollouts):
            def score(prefix):
                output = model(batch[f"{prefix}_obs"].to(device),
                               batch[f"{prefix}_actions"].to(device),
                               batch[f"{prefix}_mask"].to(device),
                               batch[f"{prefix}_position"].to(device), config.prefix_length)
                return output.joint_logit
            correct += int((score("pos") > score("neg")).sum())
            total += len(batch["pos_obs"])
        metrics["paired_ranking_accuracy"] = correct / total
        metrics["n_discordant_pairs"] = total
    return metrics


def train_verifier(model, train_examples, val_examples, device, *, config=None, wandb_run=None,
                   paired_train_examples=None, paired_val_examples=None):
    config = config or VerifierTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                            eta_min=config.lr / 50)
    best_score, best_state, bad = -float("inf"), None, 0
    pair_dataset = DiscordantPairDataset(paired_train_examples)
    point_train_examples = list(train_examples) + list(paired_train_examples or [])
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = []
        for batch in _loader(point_train_examples, config, train=True):
            optimizer.zero_grad()
            output = _forward(model, batch, device, config)
            label = batch["label"].to(device)
            joint = nn.functional.binary_cross_entropy_with_logits(output.joint_logit, label)
            state = nn.functional.binary_cross_entropy_with_logits(output.state_logit, label)
            loss = (state if config.score_head == "state" else
                    joint + config.state_loss_weight * state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        if len(pair_dataset):
            for batch in DataLoader(pair_dataset, batch_size=config.batch_rollouts, shuffle=True):
                optimizer.zero_grad()
                def score(prefix):
                    return model(batch[f"{prefix}_obs"].to(device),
                                 batch[f"{prefix}_actions"].to(device),
                                 batch[f"{prefix}_mask"].to(device),
                                 batch[f"{prefix}_position"].to(device),
                                 config.prefix_length).joint_logit
                ranking = nn.functional.softplus(-(score("pos") - score("neg"))).mean()
                loss = config.ranking_loss_weight * ranking
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss.append(float(loss.detach()))
        scheduler.step()
        metrics = evaluate_verifier(model, val_examples, device, config=config,
                                    paired_examples=paired_val_examples)
        score = metrics["task_macro_pr_auc"]
        if not np.isfinite(score):
            score = metrics["pr_auc"]
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train/loss": float(np.mean(epoch_loss)),
                           **{f"val/{k}": v for k, v in metrics.items()}})
        if score > best_score:
            best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_task_macro_pr_auc": best_score, "epochs_ran": epoch + 1,
                   "train_config": asdict(config)}


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / self.log_temperature.exp().clamp_min(1e-4)


def calibrate_temperature(model, calibration_examples, device, *, config=None):
    config = config or VerifierTrainConfig()
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in _loader(calibration_examples, config):
            output = _forward(model, batch, device, config)
            logits.append(_chosen_logit(output, config).detach())
            labels.append(batch["label"].to(device))
    logits, labels = torch.cat(logits), torch.cat(labels)
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def dataset_hash(examples):
    keys = sorted((e.rollout_id, e.chunk_idx, e.candidate_group_id or "") for e in examples)
    return hashlib.sha256(repr(keys).encode()).hexdigest()[:16]


def verifier_checkpoint_bytes(model, scaler, metadata):
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "metadata": metadata}, buffer)
    return buffer.getvalue()


def new_verifier_id():
    return uuid.uuid4().hex[:16]
