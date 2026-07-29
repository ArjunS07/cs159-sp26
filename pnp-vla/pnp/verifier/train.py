"""Value pretraining and same-state action-advantage ranking."""
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
class AdvantageTrainConfig:
    seed: int = 42
    prefix_length: int = 10
    value_lr: float = 3e-4
    rank_lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_rollouts: int = 32
    value_epochs: int = 100
    rank_epochs: int = 150
    patience: int = 20
    candidate_bce_weight: float = 0.0
    zero_context: bool = False
    context_lr_multiplier: float = 0.1


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


def _sample_weights(examples):
    chunks_per_rollout = Counter(e.rollout_id for e in examples)
    rollout_record = {e.rollout_id: e for e in examples}
    stratum_rollouts = Counter((e.task_key, e.success) for e in rollout_record.values())
    weights = [1.0 / chunks_per_rollout[e.rollout_id] / stratum_rollouts[(e.task_key, e.success)]
               for e in examples]
    return np.asarray(weights, dtype=np.float64)


def _loader(examples, config, train=False, seed_offset=0):
    dataset = CleanChunkDataset(examples)
    sampler = None
    if train:
        weights = _sample_weights(examples)
        generator = torch.Generator().manual_seed(config.seed + seed_offset)
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


def _value_parameters(model):
    for module in (model.obs_encoder, model.position_encoder, model.state_head):
        yield from module.parameters()


@torch.no_grad()
def evaluate_value(model, examples, device, config=None):
    config = config or AdvantageTrainConfig()
    model.eval()
    labels, probabilities, tasks = [], [], []
    for batch in _loader(examples, config):
        context = model.encode_context(
            batch["obs"].to(device), batch["position"].to(device))
        probabilities.extend(torch.sigmoid(model.value(context)).cpu().tolist())
        labels.extend(batch["label"].tolist())
        tasks.extend(batch["task_key"])
    return _metrics(labels, probabilities, tasks)


def pretrain_value(model, train_examples, val_examples, device, *, config=None,
                   wandb_run=None):
    """Pretrain only V(s) and its context encoders on historical rollouts."""
    config = config or AdvantageTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        _value_parameters(model), lr=config.value_lr, weight_decay=config.weight_decay)
    best_score, best_state, best_epoch, bad = -float("inf"), None, None, 0
    for epoch in range(config.value_epochs):
        model.train()
        losses = []
        for batch in _loader(train_examples, config, train=True, seed_offset=epoch):
            optimizer.zero_grad()
            context = model.encode_context(
                batch["obs"].to(device), batch["position"].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                model.value(context), batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_value_parameters(model)), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_value(model, val_examples, device, config)
        score = metrics["task_macro_pr_auc"]
        if not np.isfinite(score):
            score = metrics["pr_auc"]
        if wandb_run is not None:
            wandb_run.log({
                "value/epoch": epoch, "value/train_loss": float(np.mean(losses)),
                **{f"value/val_{key}": value for key, value in metrics.items()},
            })
        if score > best_score:
            best_score, best_state, best_epoch, bad = (
                score, copy.deepcopy(model.state_dict()), epoch, 0)
        else:
            bad += 1
        if bad >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_value_task_macro_pr_auc": best_score,
        "best_value_epoch": best_epoch,
        "value_epochs_ran": epoch + 1,
    }


def _group_examples(examples):
    groups = defaultdict(list)
    for example in examples or []:
        if example.candidate_group_id:
            groups[example.candidate_group_id].append(example)
    return groups


def _score_group(model, members, device, config):
    first = members[0]
    obs = torch.from_numpy(first.obs_enc).unsqueeze(0).to(device)
    position = torch.tensor([first.chunk_position], dtype=torch.float32, device=device)
    actions = torch.stack([
        torch.from_numpy(member.actions) for member in members
    ]).unsqueeze(0).to(device)
    masks = torch.stack([
        torch.from_numpy(member.action_mask) for member in members
    ]).unsqueeze(0).to(device)
    context = model.encode_context(obs, position)
    advantage = model.rank_candidates(
        context, actions, masks, config.prefix_length, config.zero_context).squeeze(0)
    labels = torch.tensor(
        [member.success for member in members], dtype=torch.float32, device=device)
    return context, advantage, labels


def _group_rank_loss(advantage, labels):
    positive = advantage[labels.bool()]
    negative = advantage[~labels.bool()]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return nn.functional.softplus(-differences).mean()


def _bootstrap_interval(values, seed=42, n_bootstrap=10_000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


@torch.no_grad()
def candidate_rank_records(model, examples, device, *, config=None):
    """Return one ranking record per independent candidate state."""
    config = config or AdvantageTrainConfig()
    model.eval()
    records = []
    for gid, members in sorted(_group_examples(examples).items()):
        _, advantage, labels = _score_group(model, members, device, config)
        scores = advantage.cpu().numpy()
        outcomes = labels.cpu().numpy()
        positive, negative = scores[outcomes == 1], scores[outcomes == 0]
        pair_accuracy = margin = float("nan")
        comparisons = 0
        if len(positive) and len(negative):
            differences = positive[:, None] - negative[None, :]
            pair_accuracy = float(
                ((differences > 0) + .5 * (differences == 0)).mean())
            margin = float(differences.mean())
            comparisons = int(len(positive) * len(negative))
        chosen = int(np.argmax(scores))
        default = next((
            index for index, member in enumerate(members)
            if member.candidate_kind == "default"), 0)
        first = members[0]
        records.append({
            "group_id": gid, "benchmark": first.benchmark,
            "uncertainty_stratum": first.uncertainty_stratum or "unknown",
            "pair_accuracy": pair_accuracy, "margin": margin,
            "comparisons": comparisons, "top1": float(outcomes[chosen]),
            "default": float(outcomes[default]), "random": float(outcomes.mean()),
            "oracle": float(outcomes.max()),
        })
    return records


def summarize_candidate_records(records, *, seed=42, n_bootstrap=10_000):
    """Aggregate candidate records without treating correlated pairs as independent."""
    discordant = [row for row in records if np.isfinite(row["pair_accuracy"])]
    pair_values = [row["pair_accuracy"] for row in discordant]
    top1 = [row["top1"] for row in records]
    default = [row["default"] for row in records]
    random = [row["random"] for row in records]
    metrics = {
        "group_macro_ranking_accuracy": float(np.mean(pair_values)) if pair_values else float("nan"),
        "mean_score_margin": float(np.mean([row["margin"] for row in discordant]))
        if discordant else float("nan"),
        "n_groups": len(records), "n_discordant_groups": len(discordant),
        "n_pairwise_comparisons": sum(row["comparisons"] for row in discordant),
        "top1_success": float(np.mean(top1)) if top1 else float("nan"),
        "default_success": float(np.mean(default)) if default else float("nan"),
        "random_success": float(np.mean(random)) if random else float("nan"),
        "oracle_success": float(np.mean([row["oracle"] for row in records]))
        if records else float("nan"),
        "top1_uplift_default": float(np.mean(np.asarray(top1) - np.asarray(default)))
        if records else float("nan"),
        "top1_uplift_random": float(np.mean(np.asarray(top1) - np.asarray(random)))
        if records else float("nan"),
        "ranking_accuracy_ci95": _bootstrap_interval(
            pair_values, seed, n_bootstrap),
        "top1_uplift_default_ci95": _bootstrap_interval(
            np.asarray(top1) - np.asarray(default), seed + 1, n_bootstrap),
        "top1_uplift_random_ci95": _bootstrap_interval(
            np.asarray(top1) - np.asarray(random), seed + 2, n_bootstrap),
    }
    for field in ("benchmark", "uncertainty_stratum"):
        metrics[f"by_{field}"] = {}
        for value in sorted({row[field] for row in records}):
            subset = [row for row in records if row[field] == value]
            paired = [row["pair_accuracy"] for row in subset
                      if np.isfinite(row["pair_accuracy"])]
            metrics[f"by_{field}"][value] = {
                "groups": len(subset), "discordant_groups": len(paired),
                "ranking_accuracy": float(np.mean(paired)) if paired else float("nan"),
                "top1_success": float(np.mean([row["top1"] for row in subset])),
                "default_success": float(np.mean([row["default"] for row in subset])),
                "random_success": float(np.mean([row["random"] for row in subset])),
            }
    return metrics


def aggregate_candidate_records(records):
    """Average repeated seed/fold predictions before cluster-level inference."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["group_id"]].append(row)
    output = []
    numeric = ("pair_accuracy", "margin", "top1", "default", "random", "oracle")
    for group_id, members in sorted(grouped.items()):
        first = members[0]
        row = {"group_id": group_id, "benchmark": first["benchmark"],
               "uncertainty_stratum": first["uncertainty_stratum"]}
        for key in numeric:
            values = np.asarray([member[key] for member in members], dtype=np.float64)
            row[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        row["comparisons"] = int(max(member["comparisons"] for member in members))
        output.append(row)
    return output


def paired_candidate_comparison(conditioned_records, control_records, *, seed=42,
                                n_bootstrap=10_000):
    """Cluster-bootstrap conditioned-minus-control differences by state group."""
    control = {row["group_id"]: row for row in control_records}
    paired = [(row, control[row["group_id"]]) for row in conditioned_records
              if row["group_id"] in control]
    rank_differences = [a["pair_accuracy"] - b["pair_accuracy"] for a, b in paired
                        if np.isfinite(a["pair_accuracy"]) and
                        np.isfinite(b["pair_accuracy"])]
    top1_differences = [a["top1"] - b["top1"] for a, b in paired]
    return {
        "n_paired_groups": len(paired),
        "ranking_gap": float(np.mean(rank_differences)) if rank_differences else float("nan"),
        "ranking_gap_ci95": _bootstrap_interval(
            rank_differences, seed, n_bootstrap),
        "top1_gap": float(np.mean(top1_differences)) if top1_differences else float("nan"),
        "top1_gap_ci95": _bootstrap_interval(
            top1_differences, seed + 1, n_bootstrap),
    }


def verifier_registration_eligibility(metrics, action_control_comparison,
                                      shuffled_control_comparison):
    """Apply the predeclared V2 causal-selection registration gate."""
    checks = {
        "ranking_above_chance": metrics["ranking_accuracy_ci95"][0] > .5,
        "positive_default_uplift": metrics["top1_uplift_default_ci95"][0] > 0,
        "beats_action_only": action_control_comparison["ranking_gap_ci95"][0] > 0,
        "beats_shuffled_actions": shuffled_control_comparison["ranking_gap_ci95"][0] > 0,
    }
    return {"eligible": all(checks.values()), "checks": checks}


def evaluate_candidate_ranker(model, examples, device, *, config=None,
                              n_bootstrap=10_000, return_records=False):
    """Evaluate independent state groups; correlated pairs are averaged within group."""
    config = config or AdvantageTrainConfig()
    records = candidate_rank_records(model, examples, device, config=config)
    metrics = summarize_candidate_records(
        records, seed=config.seed, n_bootstrap=n_bootstrap)
    return (metrics, records) if return_records else metrics


def train_advantage(model, train_examples, val_examples, device, *, config=None,
                    wandb_run=None):
    """Freeze V(s), then optimize one equally weighted loss per candidate state."""
    config = config or AdvantageTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(device)
    model.freeze_value_pathway()
    context_parameters = list(model.context_rank.parameters())
    context_ids = {id(parameter) for parameter in context_parameters}
    action_parameters = [parameter for parameter in model.parameters()
                         if parameter.requires_grad and id(parameter) not in context_ids]
    parameters = action_parameters + context_parameters
    optimizer = torch.optim.AdamW(
        [{"params": action_parameters, "lr": config.rank_lr},
         {"params": context_parameters,
          "lr": config.rank_lr * config.context_lr_multiplier}],
        weight_decay=config.weight_decay)
    groups = _group_examples(train_examples)
    discordant_groups = sum(
        len({member.success for member in members}) > 1
        for members in groups.values())
    if not discordant_groups and config.candidate_bce_weight <= 0:
        raise ValueError("rank-only training requires at least one discordant candidate group")
    use_validation = bool(val_examples)
    best_score, best_state, best_epoch, bad = -float("inf"), None, None, 0
    history = []
    rng = np.random.default_rng(config.seed)
    for epoch in range(config.rank_epochs):
        model.train()
        model.freeze_value_pathway()
        group_ids = list(groups)
        rng.shuffle(group_ids)
        losses = []
        for gid in group_ids:
            members = groups[gid]
            context, advantage, labels = _score_group(model, members, device, config)
            ranking = _group_rank_loss(advantage, labels)
            if ranking is None and config.candidate_bce_weight <= 0:
                continue
            loss = ranking if ranking is not None else torch.zeros((), device=device)
            if config.candidate_bce_weight > 0:
                logits = model.value(context)[:, None] + advantage[None]
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(0), labels)
                loss = loss + config.candidate_bce_weight * bce
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = (evaluate_candidate_ranker(
            model, val_examples, device, config=config, n_bootstrap=500)
            if use_validation else {})
        score = metrics.get("group_macro_ranking_accuracy", float("nan"))
        if wandb_run is not None:
            wandb_run.log({
                "rank/epoch": epoch,
                "rank/train_loss": float(np.mean(losses)) if losses else float("nan"),
                **{f"rank/val_{key}": value for key, value in metrics.items()
                   if not isinstance(value, dict)},
            })
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "validation_ranking": score,
            "gradient_norm": float(grad_norm) if losses else float("nan"),
        })
        if use_validation:
            if np.isfinite(score) and score > best_score:
                best_score, best_state, best_epoch, bad = (
                    score, copy.deepcopy(model.state_dict()), epoch, 0)
            else:
                bad += 1
            if bad >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_group_macro_ranking_accuracy": (
            best_score if use_validation else None),
        "best_rank_epoch": best_epoch,
        "rank_epochs_ran": epoch + 1,
        "rank_history": history,
        "train_config": asdict(config),
    }


def dataset_hash(examples):
    """Hash identities, labels, masks, observations, and actions deterministically."""
    digest = hashlib.sha256()
    ordered = sorted(examples, key=lambda example: (
        example.rollout_id, example.chunk_idx, example.candidate_group_id or ""))
    for example in ordered:
        metadata = (
            example.rollout_id, example.experiment, example.benchmark,
            example.suite, example.task_idx, example.episode_idx,
            example.chunk_idx, example.chunk_position, example.success,
            example.candidate_group_id or "", example.candidate_kind or "",
            example.pairing_mode or "", example.uncertainty_stratum or "",
            example.trajectory_seed, example.collection_split or "",
        )
        digest.update(repr(metadata).encode())
        for array in (example.obs_enc, example.actions, example.action_mask):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(repr(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


def verifier_checkpoint_bytes(model, scaler, metadata):
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "metadata": metadata}, buffer)
    return buffer.getvalue()


def new_verifier_id():
    return uuid.uuid4().hex[:16]
