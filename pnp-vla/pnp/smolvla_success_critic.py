"""Success-probability Q10 controls for one SmolVLA intervention then frozen-policy continuation.

Both arms use an immutable tree snapshot. The TD arm evaluates the recorded SmolVLA
continuation policy with gamma=1; the root-MC arm uses the observed terminal
success of each same-state alternative. The old demo/TD checkpoints remain
separate historical controls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import resolve_max_steps
from .qplanning_critic.config import QPlanningModelConfig
from .qplanning_critic.data import collate_windows
from .qplanning_critic.model import QPlanningCritic
from .smolvla_tree_bellman_finetune import (
    DEFAULT_SNAPSHOT_KEY, RootTreeDataset, TreeWindowDataset, _ema,
    _per_sample_hl_loss, _root_batch, _root_scores, _target_value, _to,
    load_or_create_tree_snapshot, prepare_tree_bellman_cache,
)
from .smolvla_tree_collection import SMOLVLA_TREE_EXPERIMENT
from .smolvla_tree_critic import TREE_Q10_KINDS
from .store import SupabaseStore


FORMAT = "smolvla_success_q10_v1"
HORIZON = 10
ARMS = ("root_mc", "tree_td")


def remaining_fraction(suite: str, root_chunk_idx: int, local_step: int) -> float:
    """Remaining environment budget at a tree boundary, normalized to [0,1]."""
    maximum = resolve_max_steps(suite)
    absolute = HORIZON * int(root_chunk_idx) + int(local_step)
    if absolute < 0 or absolute > maximum:
        raise ValueError(f"boundary step {absolute} is outside {suite}'s {maximum}-step budget")
    return (maximum - absolute) / maximum


class TimedRoots:
    def __init__(self, cache: dict, group_ids, groups: dict[str, dict],
                 root_token_dir: str | Path | None = None):
        self.base = RootTreeDataset(cache, group_ids)
        self.entries = self.base.entries
        self.groups = groups
        self.root_token_dir = Path(root_token_dir) if root_token_dir is not None else None
        self.snapshot_digest = cache["snapshot_digest"]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        entry = self.entries[index]
        group = self.groups[entry["candidate_group_id"]]
        time_left = remaining_fraction(group["suite"], group["chunk_idx"], 0)
        item["robot"] = np.r_[item["robot"], np.float32(time_left)].astype(np.float32)
        if self.root_token_dir is not None:
            gid = entry["candidate_group_id"]
            path = self.root_token_dir / (hashlib.sha256(gid.encode()).hexdigest()[:24] + ".npz")
            with np.load(path, allow_pickle=False) as feature:
                if (str(feature["group_id"]) != gid or
                        str(feature["snapshot_digest"]) != self.snapshot_digest):
                    raise ValueError(f"RL Token root feature mismatch: {path}")
                item["prefix"] = np.asarray(feature["token"], np.float32)[None]
                item["pad"] = np.ones(1, bool)
        return item


class TimedWindows:
    def __init__(self, cache: dict, group_ids, groups: dict[str, dict]):
        self.base = TreeWindowDataset(cache, group_ids)
        self.groups = groups

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        entry, _ = self.base.indices[index]
        group = self.groups[entry["candidate_group_id"]]
        item = self.base[index]
        # start_step is absolute for the stored source but branch-local for
        # counterfactuals. fork_offset is consistently zero-based after the
        # intervention root for both artifact families.
        local_step = int(item["fork_offset"]) * HORIZON
        current = remaining_fraction(group["suite"], group["chunk_idx"], local_step)
        # A terminal row has zero bootstrap weight, so its next time is only a
        # shape-compatible placeholder. Nonterminal rows advance one Q10 boundary.
        next_local = local_step + HORIZON if float(item["discount"]) > 0 else local_step
        next_value = remaining_fraction(group["suite"], group["chunk_idx"], next_local)
        item["robot"] = np.r_[item["robot"], np.float32(current)].astype(np.float32)
        item["next_robot"] = np.r_[item["next_robot"], np.float32(next_value)].astype(np.float32)
        return item


def action_statistics(roots: TimedRoots) -> tuple[np.ndarray, np.ndarray]:
    """Fit normalization only on training-root candidate actions."""
    total = np.zeros(7, np.float64)
    squared = np.zeros(7, np.float64)
    count = 0
    for index in range(len(roots)):
        item = roots[index]
        actions = np.asarray(item["actions"], np.float64)
        valid = np.asarray(item["action_valid"], bool)
        values = actions[valid]
        total += values.sum(0)
        squared += np.square(values).sum(0)
        count += len(values)
    if not count:
        raise ValueError("no training-root actions for normalization")
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def select_training_roots(group_ids, limit: int | None) -> list[str]:
    """Choose nested training subsets without changing the fixed validation split."""
    ids = list(group_ids)
    if limit is None:
        return ids
    if limit < 1 or limit > len(ids):
        raise ValueError(f"requested {limit} training roots, found {len(ids)}")
    return sorted(ids, key=lambda gid: hashlib.sha256(
        f"success-q-scale-v1|{gid}".encode()).hexdigest())[:limit]


def sample_tree_td_indices(windows: TimedWindows, rng: np.random.Generator,
                           branches: int, per_branch: int = 8,
                           root_fraction: float = 0.0) -> np.ndarray:
    """Sample uniform-marginal windows while opening few branch archives.

    Pick a branch in proportion to its number of windows, then sample windows
    within that branch. Each resulting window has the same marginal probability
    as sampling directly from the entire dataset, but adjacent items share one
    decompressed archive in TreeWindowDataset's LRU cache.
    """
    lengths = np.fromiter(
        (int(entry["n_windows"]) for entry in windows.base.entries),
        dtype=np.int64)
    if not len(lengths) or np.any(lengths <= 0):
        raise ValueError("TD sampler needs nonempty candidate branches")
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    chosen = rng.choice(len(lengths), size=branches,
                        p=lengths / lengths.sum(), replace=True)
    sampled = np.concatenate([
        starts[index] + rng.integers(0, lengths[index], size=per_branch)
        for index in chosen
    ])
    if root_fraction:
        root_count = int(round(len(sampled) * root_fraction))
        positions = rng.choice(len(sampled), size=root_count, replace=False)
        sampled[positions] = rng.choice(starts, size=root_count, replace=True)
    return sampled


@dataclass(frozen=True)
class SuccessTrainConfig:
    arm: str
    train_root_limit: int | None = None
    seed: int = 42
    updates: int = 2_000
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_updates: int = 100
    eval_interval: int = 250
    checkpoint_interval: int = 500
    target_rate: float = .005
    grad_clip: float = 1.0
    td_root_fraction: float = 0.0

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if self.train_root_limit is not None and self.train_root_limit < 1:
            raise ValueError("train_root_limit must be positive or None")
        if not 0 <= self.td_root_fraction <= 1:
            raise ValueError("td_root_fraction must be in [0, 1]")
        if self.arm != "tree_td" and self.td_root_fraction:
            raise ValueError("td_root_fraction applies only to tree_td")
        if self.learning_rate <= 0 or self.target_rate <= 0 or self.target_rate > 1:
            raise ValueError("learning_rate must be positive and target_rate in (0, 1]")
        if self.warmup_updates < 0 or self.eval_interval < 1 or self.checkpoint_interval < 1:
            raise ValueError("warmup_updates must be nonnegative; intervals must be positive")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")

    def lr_at(self, update: int) -> float:
        if update <= self.warmup_updates:
            return self.learning_rate * update / max(1, self.warmup_updates)
        progress = min(1.0, (update - self.warmup_updates)
                       / max(1, self.updates - self.warmup_updates))
        return self.learning_rate * .5 * (1 + math.cos(math.pi * progress))


def _config_contract(config: SuccessTrainConfig) -> dict:
    """Keep historical full-data checkpoints resumable after adding scaling."""
    payload = asdict(config)
    if payload["train_root_limit"] is None:
        payload.pop("train_root_limit")
    if payload["td_root_fraction"] == 0:
        payload.pop("td_root_fraction")
    return payload


def _resume_compatible(saved: dict, requested: dict) -> bool:
    """Allow a longer run and reporting cadence changes, but preserve training semantics."""
    mutable = {"updates", "eval_interval", "checkpoint_interval"}
    return ({k: v for k, v in saved.items() if k not in mutable}
            == {k: v for k, v in requested.items() if k not in mutable})


def _extension_from_checkpoint(saved: dict, config: SuccessTrainConfig,
                               requested_rate: float | None) -> tuple[int, dict | None]:
    """Plan a constant-LR extension without silently changing an existing phase."""
    start_update = int(saved["update"])
    if config.updates < start_update:
        raise ValueError("updates cannot be less than the saved checkpoint update")
    saved_planned_updates = int(saved["config"]["updates"])
    extension = saved.get("extension")
    if config.updates > saved_planned_updates:
        if requested_rate is None:
            raise ValueError(
                "extending a cosine-decay run requires extension_learning_rate; "
                "the old learning rate reached zero at its planned update count")
        extension = {"start_update": start_update,
                     "learning_rate": float(requested_rate)}
    elif (requested_rate is not None and
          (extension is None or float(requested_rate) != extension["learning_rate"])):
        raise ValueError("extension_learning_rate differs from the saved extension")
    return start_update, extension


@torch.no_grad()
def evaluate_roots(model, dataset: TimedRoots, device) -> dict:
    """Report observable selection quality at the held-out fork roots."""
    model.eval()
    scores, labels = [], []
    for start in range(0, len(dataset), 8):
        batch = _to(_root_batch([dataset[i] for i in
                                range(start, min(len(dataset), start + 8))]), device)
        scores.append(_root_scores(model, batch).cpu().numpy())
        labels.append(batch["success"].cpu().numpy())
    q = np.concatenate(scores)
    success = np.concatenate(labels).astype(bool)
    rows = np.arange(len(q))
    selected = q.argmax(1)
    stock = success[:, 0]
    selected_success = success[rows, selected]
    correct = total = 0.0
    for values, outcomes in zip(q, success):
        positive, negative = values[outcomes], values[~outcomes]
        if len(positive) and len(negative):
            delta = positive[:, None] - negative[None]
            correct += (delta > 0).sum() + .5 * (delta == 0).sum()
            total += delta.size
    probability = np.clip(q, 1e-6, 1 - 1e-6)
    return {
        "trees": len(q), "mixed_trees": int(np.any(success, 1).sum()
                                          - np.all(success, 1).sum()),
        "stock_successes": int(stock.sum()),
        "selected_successes": int(selected_success.sum()),
        "rescues": int((~stock & selected_success).sum()),
        "spoils": int((stock & ~selected_success).sum()),
        "selected_minus_stock_pp": float(100 * (selected_success.mean() - stock.mean())),
        "within_tree_pair_accuracy": float(correct / total) if total else float("nan"),
        "brier": float(np.square(q - success).mean()),
        "binary_ce": float(-(success * np.log(probability)
                             + ~success * np.log(1 - probability)).mean()),
    }


def _save(path: Path, *, model, target, optimizer, config, snapshot_digest,
          representation, extension,
          update: int, history: list[dict]):
    payload = {
        "format": FORMAT, "objective": "success_probability_after_one_intervention",
        "continuation_policy": "smolvla_pnp_steps123_k311",
        "gamma": 1.0, "reward": "step_success",
        "time_feature": "remaining_actions_over_suite_limit",
        "snapshot_digest": snapshot_digest, "config": _config_contract(config),
        "extension": extension,
        "representation": representation,
        "architecture": model.architecture_config(), "update": update,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "target": {k: v.detach().cpu() for k, v in target.state_dict().items()},
        "optimizer": optimizer.state_dict(), "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    for previous in path.parent.glob("checkpoint_step_*.pt"):
        if previous != path:
            previous.unlink()


def train_smolvla_success_q10(*, arm: str, output_root: str | Path,
                              cache_root: str | Path, tree_limit: int = 480,
                              train_root_limit: int | None = None,
                              snapshot_key: str = DEFAULT_SNAPSHOT_KEY,
                              experiment: str = SMOLVLA_TREE_EXPERIMENT,
                              candidate_kinds: tuple[str, ...] = TREE_Q10_KINDS,
                              updates: int = 2_000, resume: bool = True,
                              checkpoint_interval: int = 500,
                              batch_size: int = 8, learning_rate: float = 1e-4,
                              warmup_updates: int = 100,
                              eval_interval: int = 250,
                              target_rate: float = .005,
                              grad_clip: float = 1.0,
                              td_root_fraction: float = 0.0,
                              extension_learning_rate: float | None = None,
                              root_token_dir: str | Path | None = None,
                              root_cache_only: bool = False,
                              device=None, store=None) -> dict:
    """Train one success-Q arm on an immutable tree snapshot and fixed root split."""
    store = store or SupabaseStore()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = SuccessTrainConfig(
        arm=arm, train_root_limit=train_root_limit, updates=updates,
        checkpoint_interval=checkpoint_interval, batch_size=batch_size,
        learning_rate=learning_rate, warmup_updates=warmup_updates,
        eval_interval=eval_interval, target_rate=target_rate,
        grad_clip=grad_clip, td_root_fraction=td_root_fraction)
    if extension_learning_rate is not None and extension_learning_rate <= 0:
        raise ValueError("extension_learning_rate must be positive")
    if root_token_dir is not None and arm != "root_mc":
        raise ValueError("contextual RL Token features currently support root-MC only; TD branch features were not retained")
    if root_cache_only and arm != "root_mc":
        raise ValueError("root-only cache supports root-MC training, not tree TD")
    snapshot = load_or_create_tree_snapshot(
        store=store, snapshot_key=snapshot_key, tree_limit=tree_limit,
        experiment=experiment, candidate_kinds=candidate_kinds)
    if root_cache_only:
        from .smolvla_combined_success import prepare_combined_root_cache
        cache = prepare_combined_root_cache(
            snapshot=snapshot, cache_root=cache_root, store=store)
    else:
        cache = prepare_tree_bellman_cache(
            snapshot=snapshot, cache_root=cache_root, gamma=1.0,
            success_reward=True, store=store)
    token_digest = None
    if root_token_dir is not None:
        manifest = json.loads((Path(root_token_dir) / "manifest.json").read_text())
        if (manifest.get("format") != "smolvla_rl_token_root_features_v1" or
                manifest.get("snapshot_digest") != snapshot["snapshot_digest"] or
                int(manifest.get("train", -1)) != len(cache["train_group_ids"]) or
                int(manifest.get("validation", -1)) != len(cache["validation_group_ids"])):
            raise ValueError("RL Token root feature manifest differs from frozen Q snapshot")
        token_digest = str(manifest["rlt_checkpoint_digest"])
    groups = {row["candidate_group_id"]: row for row in snapshot["groups"]}
    train_ids = select_training_roots(cache["train_group_ids"], train_root_limit)
    train_roots = TimedRoots(cache, train_ids, groups, root_token_dir)
    val_roots = TimedRoots(cache, cache["validation_group_ids"], groups, root_token_dir)
    train_windows = None if root_cache_only else TimedWindows(cache, train_ids, groups)
    if not len(train_roots) or not len(val_roots) or (arm == "tree_td" and not len(train_windows)):
        raise ValueError("tree training/validation split is empty")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    first = train_roots[0]
    architecture = QPlanningModelConfig(
        action_horizon=HORIZON, action_dim=7, width=256, n_layers=3,
        n_heads=8, ffn_width=1024, dropout=.20,
        prefix_pool_tokens=1 if root_token_dir is not None else 128)
    model = QPlanningCritic(
        prefix_dim=first["prefix"].shape[-1], robot_dim=len(first["robot"]),
        proprio_dim=len(first["proprio"]), config=architecture)
    mean, std = action_statistics(train_roots)
    model.set_action_statistics(mean, std)
    model.to(device)
    target = copy.deepcopy(model).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate, weight_decay=1e-4)

    directory = Path(output_root).expanduser() / snapshot["snapshot_digest"]
    representation = (f"contextual_rl_token_reconstruction_v1:{token_digest}"
                      if root_token_dir is not None else "frozen_prefill_prefix_v1")
    if root_token_dir is not None:
        directory = directory / f"rl_token_{token_digest}"
    if train_root_limit is not None:
        directory = directory / f"train_roots_{train_root_limit}"
    directory = directory / arm
    existing = list(directory.glob("checkpoint_step_*.pt"))
    if existing and not resume:
        raise ValueError(
            f"checkpoints already exist in {directory}; use resume=True or a new output_root")
    if not resume:
        existing = []
    history: list[dict] = []
    start_update = 0
    extension = None
    if existing:
        path = max(existing, key=lambda value: int(value.stem.rsplit("_", 1)[-1]))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("format") != FORMAT or
            saved.get("snapshot_digest") != snapshot["snapshot_digest"] or
            saved.get("representation", "frozen_prefill_prefix_v1") != representation or
            not _resume_compatible(saved.get("config", {}), _config_contract(config)) or
            saved.get("architecture") != model.architecture_config()):
            raise ValueError("checkpoint differs from requested success-Q contract")
        start_update, extension = _extension_from_checkpoint(
            saved, config, extension_learning_rate)
        model.load_state_dict(saved["model"])
        target.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if saved.get("mps_rng") is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(saved["mps_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        history = list(saved["history"])
    elif extension_learning_rate is not None:
        raise ValueError("extension_learning_rate requires an existing checkpoint")

    print({"arm": arm, "snapshot": snapshot["snapshot_digest"],
           "train_root_limit": train_root_limit,
           "train_roots": len(train_roots), "validation_roots": len(val_roots),
           "train_windows": 0 if train_windows is None else len(train_windows), "gamma": 1.0,
           "reward": "step_success", "demo_replay_fraction": 0,
           "time_feature": "remaining_actions_over_suite_limit",
           "start_update": start_update, "planned_updates": config.updates,
           "td_root_fraction": config.td_root_fraction,
           "extension": extension}, flush=True)
    began = time.perf_counter()
    # is_bf16_supported() may report emulated support on a T4. Use BF16 only
    # on Ampere-or-newer hardware; the T4 runs this compact critic in FP32.
    use_amp = (device.type == "cuda" and
               torch.cuda.get_device_capability(device)[0] >= 8 and
               torch.cuda.is_bf16_supported())
    for update in range(start_update + 1, config.updates + 1):
        model.train()
        target.eval()
        for group in optimizer.param_groups:
            group["lr"] = (extension["learning_rate"]
                           if extension is not None and update > extension["start_update"]
                           else config.lr_at(update))
        rng = np.random.default_rng(config.seed * 1_000_003 + update)
        if arm == "root_mc":
            count = config.batch_size
            indices = rng.choice(len(train_roots), size=count,
                                 replace=len(train_roots) < count)
        else:
            indices = sample_tree_td_indices(
                train_windows, rng, config.batch_size,
                root_fraction=config.td_root_fraction)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            if arm == "root_mc":
                batch = _to(_root_batch([train_roots[int(i)] for i in indices]), device)
                prediction = _root_scores(model, batch).float().clamp(1e-5, 1 - 1e-5)
            else:
                batch = _to(collate_windows([train_windows[int(i)] for i in indices]), device)
                value = _target_value(target, batch)
                logits = model(batch["prefix"], batch["pad"], batch["robot"],
                               batch["proprio"], batch["action"], batch["action_valid"])
                loss = _per_sample_hl_loss(model, logits, value).mean()
        if arm == "root_mc":
            # CUDA autocast disallows probability-space BCE, even when its
            # input was converted to float32. Compute the proper scoring loss
            # outside autocast while retaining gradients through the model.
            loss = F.binary_cross_entropy(prediction, batch["success"].float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if arm == "tree_td":
            _ema(target, model, config.target_rate)
        if update % 100 == 0 or update == config.updates:
            print(f"[success-q] {arm} {update}/{config.updates} "
                  f"loss={float(loss):.4f} grad={float(grad_norm):.3f} "
                  f"elapsed={(time.perf_counter()-began)/60:.1f}m", flush=True)
        if update % config.eval_interval == 0 or update == config.updates:
            metrics = evaluate_roots(model, val_roots, device)
            history.append({"update": update, "validation": metrics})
            print({"update": update, "validation": metrics}, flush=True)
        if update % config.checkpoint_interval == 0 or update == config.updates:
            path = directory / f"checkpoint_step_{update:06d}.pt"
            _save(path, model=model, target=target, optimizer=optimizer,
                  config=config, snapshot_digest=snapshot["snapshot_digest"],
                  representation=representation, extension=extension,
                  update=update, history=history)
            print(f"[success-q] saved {path}", flush=True)
    return {
        "arm": arm, "snapshot_digest": snapshot["snapshot_digest"],
        "final_checkpoint": str(directory / f"checkpoint_step_{config.updates:06d}.pt"),
        "validation": evaluate_roots(model, val_roots, device),
        "history": history,
    }
