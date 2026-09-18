"""Compact from-scratch SmolVLA tree critics for action-ranking diagnosis."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .qplanning_critic.config import QPlanningModelConfig
from .qplanning_critic.model import QPlanningCritic
from .smolvla_tree_critic import (
    TREE_Q10_ACTION_DIM,
    TREE_Q10_EXPECTED_TREES,
    TREE_Q10_GAMMA,
    TREE_Q10_HORIZON,
    TREE_Q10_KINDS,
    SmolVLATreeDataset,
    _collate_groups,
    _forward_groups,
    _latest_checkpoint,
    _selection_batch_indices,
    _selection_outcome_pools,
    _state_digest,
    _to,
    evaluate_smolvla_tree_q10,
    mixed_tree_mask,
    pairwise_success_ranking_loss,
    prepare_smolvla_tree_q10_cache,
)
from .store import SupabaseStore


@dataclass(frozen=True)
class CompactTreeTrainConfig:
    seed: int = 42
    updates: int = 750
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_updates: int = 75
    effective_tree_batch: int = 8
    micro_tree_batch: int = 2
    absolute_weight: float = 0.0
    pairwise_weight: float = 1.0
    consistency_weight: float = 1.0
    ranking_temperature: float = 0.10
    ranking_margin: float = 0.05
    width: int = 256
    n_layers: int = 3
    n_heads: int = 8
    ffn_width: int = 1024
    dropout: float = 0.20
    print_interval: int = 50
    eval_interval: int = 100
    checkpoint_interval: int = 250
    grad_clip: float = 1.0
    use_bf16: bool = True

    def __post_init__(self):
        if self.updates < 1 or not 0 <= self.warmup_updates <= self.updates:
            raise ValueError("invalid update schedule")
        if self.effective_tree_batch != 8:
            raise ValueError("the exact 65% schedule requires effective_tree_batch=8")
        if self.micro_tree_batch < 1 or self.effective_tree_batch % self.micro_tree_batch:
            raise ValueError("micro_tree_batch must divide effective_tree_batch")
        if min(self.absolute_weight, self.pairwise_weight, self.consistency_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.ranking_temperature <= 0 or self.ranking_margin < 0:
            raise ValueError("invalid ranking temperature or margin")
        if self.width % self.n_heads:
            raise ValueError("width must be divisible by n_heads")

    @property
    def accumulation_steps(self) -> int:
        return self.effective_tree_batch // self.micro_tree_batch

    def learning_rate_at(self, update: int) -> float:
        if self.warmup_updates and update <= self.warmup_updates:
            return self.learning_rate * update / self.warmup_updates
        span = max(1, self.updates - self.warmup_updates)
        progress = min(1.0, max(0.0, (update - self.warmup_updates) / span))
        return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def nonmixed_score_consistency_loss(score: torch.Tensor, success: torch.Tensor, *,
                                    temperature: float = 0.10) -> torch.Tensor:
    """Suppress unsupported candidate distinctions when all tree outcomes agree."""
    if score.shape != success.shape or score.ndim != 2:
        raise ValueError("score and success must have matching [trees,candidates] shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    keep = ~mixed_tree_mask(success)
    if not bool(keep.any()):
        return score.sum() * 0.0
    values = score[keep] / temperature
    return ((values - values.mean(1, keepdim=True)) ** 2).mean()


def _save_compact_checkpoint(path: Path, *, model, optimizer, update: int,
                             cache: dict, config: CompactTreeTrainConfig,
                             history: list[dict], run_name: str):
    payload = {
        "format": "smolvla_tree_q10_compact_critic_v1", "update": update,
        "run_name": run_name, "dataset_digest": cache["dataset_digest"],
        "train_group_ids": cache["train_group_ids"],
        "validation_group_ids": cache["validation_group_ids"],
        "architecture": model.architecture_config(), "train_config": asdict(config),
        "source_policy": {"model": "HuggingFaceVLA/smolvla_libero",
                          "executed_actions": 10, "generated_actions": 50},
        "selection_score": "expected_hl_gauss_return",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    # Keep storage bounded exactly as in the preceding training workers.
    for candidate in path.parent.glob("checkpoint_step_*.pt"):
        if candidate != path:
            candidate.unlink()


def run_compact_smolvla_tree_training(
        *, run_name: str, absolute_weight: float,
        expected_trees: int = TREE_Q10_EXPECTED_TREES, updates: int = 750,
        cache_root: str | Path = "/content/smolvla_tree_q10_cache",
        output_root: str | Path = "/content/drive/MyDrive/pnp_smolvla_tree_q10_compact",
        micro_tree_batch: int = 2, download_workers: int = 8,
        device: str | None = None, resume: bool = True, store=None):
    """Train one compact arm; ``absolute_weight`` is declared as either zero or 0.05."""
    if float(absolute_weight) not in (0.0, 0.05):
        raise ValueError("the declared compact ablation requires absolute_weight 0 or 0.05")
    store = store or SupabaseStore()
    config = CompactTreeTrainConfig(
        updates=updates, micro_tree_batch=micro_tree_batch,
        absolute_weight=float(absolute_weight))
    cache = prepare_smolvla_tree_q10_cache(
        store=store, cache_root=cache_root, expected_trees=expected_trees,
        gamma=TREE_Q10_GAMMA, download_workers=download_workers)
    train = SmolVLATreeDataset(cache, cache["train_group_ids"])
    validation = SmolVLATreeDataset(cache, cache["validation_group_ids"])
    mixed_pool, nonmixed_pool = _selection_outcome_pools(train)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = QPlanningCritic(
        prefix_dim=int(cache["prefix_dim"]), robot_dim=int(cache["robot_dim"]),
        proprio_dim=int(cache["proprio_dim"]),
        config=QPlanningModelConfig(
            action_horizon=TREE_Q10_HORIZON, action_dim=int(cache["action_dim"]),
            width=config.width, n_layers=config.n_layers, n_heads=config.n_heads,
            ffn_width=config.ffn_width, dropout=config.dropout))
    model.set_action_statistics(cache["action_mean"], cache["action_std"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_digest = _state_digest(model)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir = Path(output_root).expanduser() / cache["dataset_digest"] / run_name
    start_update, history = 0, []
    latest = _latest_checkpoint(output_dir) if resume else None
    if latest is not None:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if (payload.get("format") != "smolvla_tree_q10_compact_critic_v1"
                or payload.get("dataset_digest") != cache["dataset_digest"]
                or payload.get("train_config") != asdict(config)):
            raise ValueError("existing checkpoint does not match this exact compact contract")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        model.to(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_update = int(payload["update"])
        history = list(payload.get("history", []))
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        print(f"[smolvla-tree-compact] resumed {latest.name} at step {start_update}", flush=True)

    print("Training contract", flush=True)
    print({
        "run_name": run_name, "parameter_count": parameter_count,
        "width": config.width, "layers": config.n_layers, "heads": config.n_heads,
        "ffn_width": config.ffn_width, "dropout": config.dropout,
        "absolute_weight": config.absolute_weight,
        "pairwise_weight": config.pairwise_weight,
        "nonmixed_consistency_weight": config.consistency_weight,
        "weight_decay": config.weight_decay,
        "mixed_sampling": "65% exactly over each 5-update block",
        "mixed_train_trees": len(mixed_pool), "nonmixed_train_trees": len(nonmixed_pool),
        "updates": config.updates, "effective_tree_batch": config.effective_tree_batch,
        "effective_candidate_rows": config.effective_tree_batch * len(TREE_Q10_KINDS),
        "dataset_digest": cache["dataset_digest"], "initial_model_digest": initial_digest,
        "train_trees": len(train), "validation_trees": len(validation),
        "device": str(device), "output_dir": str(output_dir),
    }, flush=True)
    if start_update == 0:
        initial = evaluate_smolvla_tree_q10(
            model, validation, device, micro_tree_batch=micro_tree_batch)
        history.append({"update": 0, "validation": initial})
        print(f"validation step 0 | {initial}", flush=True)

    use_amp = bool(config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    started = time.perf_counter()
    rolling = defaultdict(float)
    rolling_micro = rolling_mixed = rolling_nonmixed = 0
    for update in range(start_update + 1, config.updates + 1):
        learning_rate = config.learning_rate_at(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        indices = _selection_batch_indices(
            mixed=mixed_pool, nonmixed=nonmixed_pool, update=update, seed=config.seed)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        total_mixed = 6 if update % 5 == 0 else 5
        total_nonmixed = config.effective_tree_batch - total_mixed
        for offset in range(0, len(indices), config.micro_tree_batch):
            raw = _collate_groups([
                train[int(i)] for i in indices[offset:offset + config.micro_tree_batch]])
            batch = _to(raw, device)
            micro_mixed = int(mixed_tree_mask(batch["success"]).sum().item())
            micro_nonmixed = len(batch["success"]) - micro_mixed
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits, expected = _forward_groups(model, batch)
                absolute = model.categorical_loss(logits, batch["target"].reshape(-1))
                pairwise = pairwise_success_ranking_loss(
                    expected, batch["success"],
                    temperature=config.ranking_temperature, margin=config.ranking_margin)
                consistency = nonmixed_score_consistency_loss(
                    expected, batch["success"], temperature=config.ranking_temperature)
                loss = (config.absolute_weight * absolute / config.accumulation_steps
                        + config.pairwise_weight * pairwise * micro_mixed / total_mixed
                        + config.consistency_weight * consistency
                        * micro_nonmixed / total_nonmixed)
            loss.backward()
            rolling["absolute"] += float(absolute.detach())
            rolling["pairwise"] += float(pairwise.detach()) * micro_mixed
            rolling["consistency"] += float(consistency.detach()) * micro_nonmixed
            rolling["q"] += float(expected.detach().mean())
            rolling["target"] += float(batch["target"].mean())
            rolling_micro += 1
            rolling_mixed += micro_mixed
            rolling_nonmixed += micro_nonmixed
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if update % config.print_interval == 0 or update == config.updates:
            elapsed = time.perf_counter() - started
            completed = update - start_update
            eta = elapsed / max(completed, 1) * (config.updates - update)
            gpu = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
            print(
                f"compact-Q10 step {update}/{config.updates} | "
                f"abs_ce {rolling['absolute']/max(rolling_micro, 1):.4f} | "
                f"pair_rank {rolling['pairwise']/max(rolling_mixed, 1):.4f} | "
                f"same_outcome_var {rolling['consistency']/max(rolling_nonmixed, 1):.4f} | "
                f"q {rolling['q']/max(rolling_micro, 1):.4f} | "
                f"target {rolling['target']/max(rolling_micro, 1):.4f} | "
                f"mixed slots {rolling_mixed}/{rolling_micro*config.micro_tree_batch} | "
                f"grad {float(grad_norm):.3f} | lr {learning_rate:.2e} | GPU {gpu:.1f} GB | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m", flush=True)
            rolling.clear()
            rolling_micro = rolling_mixed = rolling_nonmixed = 0
        if update % config.eval_interval == 0 or update == config.updates:
            metrics = evaluate_smolvla_tree_q10(
                model, validation, device, micro_tree_batch=micro_tree_batch)
            history.append({"update": update, "validation": metrics})
            print(
                "validation | CE {hl_gauss_ce:.4f} | return MAE {return_mae:.4f} | "
                "gap MAE {difference_mae:.4f} | failure AUC {failure_auc:.3f} | "
                "within-tree {within_tree_pairwise_accuracy:.3f} | "
                "selected-stock {selected_minus_stock_pp:+.1f} pp | "
                "F->S {failure_to_success} | S->F {success_to_failure} | "
                "stock chosen {stock_selected_pct:.1f}% | n {trees}".format(**metrics),
                flush=True)
        if update % config.checkpoint_interval == 0 or update == config.updates:
            path = output_dir / f"checkpoint_step_{update:06d}.pt"
            _save_compact_checkpoint(
                path, model=model, optimizer=optimizer, update=update, cache=cache,
                config=config, history=history, run_name=run_name)
            print(f"[smolvla-tree-compact] saved {path}", flush=True)

    final = evaluate_smolvla_tree_q10(
        model, validation, device, micro_tree_batch=micro_tree_batch)
    return {
        "run_name": run_name, "absolute_weight": config.absolute_weight,
        "parameter_count": parameter_count, "dataset_digest": cache["dataset_digest"],
        "initial_model_digest": initial_digest, "updates": config.updates,
        "validation": final, "history": history,
        "final_checkpoint": str(output_dir / f"checkpoint_step_{config.updates:06d}.pt"),
    }
