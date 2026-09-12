"""From-scratch Q50 training with outcome- or U20-prioritized replay."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Sampler

from ..pcp_critic.data import eligible_rollout_rows
from ..pcp_critic.registry import PCPCriticRegistry
from ..store import SupabaseStore
from .config import QPlanningModelConfig, QPlanningTrainConfig
from .data import QPlanningWindowDataset, prepare_qplanning_streaming_cache
from .model import QPlanningCritic
from .train import train_qplanning_critic
from .uncertainty import U20LabelCache, prepare_u20_label_cache
from .workflow import _print_preflight


PRIORITY_STRATEGIES = (
    "failure",
    "episode_u20",
    "u20_4chunk",
    "u20_8chunk",
)
PRIORITY_FRACTION = 0.5


@dataclass(frozen=True)
class ReplayPriorityPlan:
    """Indices eligible for the prioritized half of each training batch."""

    strategy: str
    priority_by_rollout: dict[str, tuple[tuple[int, ...], ...]]
    summary: dict

    @property
    def digest(self) -> str:
        groups = {
            rollout_id: [list(group) for group in rollout_groups]
            for rollout_id, rollout_groups in sorted(self.priority_by_rollout.items())
        }
        payload = {
            "strategy": self.strategy,
            "priority_fraction": PRIORITY_FRACTION,
            "groups": groups,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]


def _indices_by_rollout(dataset: QPlanningWindowDataset):
    grouped = defaultdict(list)
    suites = {}
    for dataset_index, (entry, offset) in enumerate(dataset.indices):
        rollout_id = entry["rollout_id"]
        grouped[rollout_id].append((int(offset), dataset_index))
        suites[rollout_id] = str(entry.get("suite") or entry.get("benchmark") or "")
    for rollout_id, values in grouped.items():
        values.sort()
        offsets = [offset for offset, _ in values]
        if offsets != list(range(len(offsets))):
            raise ValueError(f"non-contiguous Q windows for rollout {rollout_id}")
    return dict(grouped), suites


def _load_train_u20(labels: U20LabelCache, rollout_ids) -> dict[str, np.ndarray]:
    paths = {entry["rollout_id"]: entry["path"] for entry in labels.rollouts}
    root = Path(labels.cache_dir)
    result = {}
    for rollout_id in rollout_ids:
        if rollout_id not in paths:
            raise ValueError(f"missing U20 labels for train rollout {rollout_id}")
        with np.load(root / paths[rollout_id], allow_pickle=False) as archive:
            current = np.asarray(archive["current_u20"], np.float64)
        if not len(current) or not np.isfinite(current).all() or np.any(current <= 0):
            raise ValueError(f"invalid U20 labels for rollout {rollout_id}")
        result[rollout_id] = current
    return result


def _top_quartile(items, *, bucket_of, score_of):
    buckets = defaultdict(list)
    for item in items:
        buckets[bucket_of(item)].append(item)
    selected, thresholds = [], {}
    for bucket, values in buckets.items():
        scores = np.asarray([score_of(value) for value in values], np.float64)
        threshold = float(np.quantile(scores, 0.75))
        thresholds[str(bucket)] = threshold
        selected.extend(
            value for value in values if float(score_of(value)) >= threshold)
    return selected, thresholds


def build_replay_priority_plan(
        dataset: QPlanningWindowDataset, *, strategy: str,
        outcomes: dict[str, bool],
        labels: U20LabelCache | None = None) -> ReplayPriorityPlan:
    """Build one immutable priority pool without changing Q targets or losses."""
    if strategy not in PRIORITY_STRATEGIES:
        raise ValueError(f"unknown replay strategy {strategy!r}")
    by_rollout, suites = _indices_by_rollout(dataset)
    rollout_ids = sorted(by_rollout)
    if set(outcomes) != set(rollout_ids):
        missing = sorted(set(rollout_ids) - set(outcomes))
        extra = sorted(set(outcomes) - set(rollout_ids))
        raise ValueError(
            f"outcome IDs do not match train split; missing={missing[:3]}, extra={extra[:3]}")

    thresholds = {}
    if strategy == "failure":
        selected_ids = [rollout_id for rollout_id in rollout_ids if not outcomes[rollout_id]]
        priority = {
            rollout_id: (tuple(index for _, index in by_rollout[rollout_id]),)
            for rollout_id in selected_ids
        }
    else:
        if labels is None:
            raise ValueError(f"{strategy} requires the U20 label cache")
        u20 = _load_train_u20(labels, rollout_ids)
        for rollout_id in rollout_ids:
            if len(u20[rollout_id]) != len(by_rollout[rollout_id]):
                raise ValueError(
                    f"U20/Q-window count mismatch for rollout {rollout_id}: "
                    f"{len(u20[rollout_id])} vs {len(by_rollout[rollout_id])}")

        if strategy == "episode_u20":
            records = [
                (rollout_id, float(u20[rollout_id].mean()))
                for rollout_id in rollout_ids
            ]
            selected, thresholds = _top_quartile(
                records, bucket_of=lambda value: suites[value[0]],
                score_of=lambda value: value[1])
            priority = {
                rollout_id: (tuple(index for _, index in by_rollout[rollout_id]),)
                for rollout_id, _ in selected
            }
        else:
            block_size = 4 if strategy == "u20_4chunk" else 8
            blocks = []
            for rollout_id in rollout_ids:
                windows = by_rollout[rollout_id]
                for block_index, start in enumerate(range(0, len(windows), block_size)):
                    block = windows[start:start + block_size]
                    offsets = [offset for offset, _ in block]
                    blocks.append({
                        "rollout_id": rollout_id,
                        "suite": suites[rollout_id],
                        "block_index": block_index,
                        "indices": tuple(index for _, index in block),
                        "score": float(u20[rollout_id][offsets].mean()),
                    })
            selected, thresholds = _top_quartile(
                blocks,
                bucket_of=lambda value: (value["suite"], value["block_index"]),
                score_of=lambda value: value["score"])
            grouped = defaultdict(list)
            for block in selected:
                grouped[block["rollout_id"]].append(block["indices"])
            priority = {
                rollout_id: tuple(groups)
                for rollout_id, groups in sorted(grouped.items())
            }

    if not priority:
        raise ValueError(f"{strategy} produced an empty priority pool")
    priority_indices = {
        index for groups in priority.values() for group in groups for index in group}
    summary = {
        "strategy": strategy,
        "priority_fraction": PRIORITY_FRACTION,
        "train_rollouts": len(rollout_ids),
        "train_windows": len(dataset),
        "failed_rollouts": sum(not outcomes[rollout_id] for rollout_id in rollout_ids),
        "priority_rollouts": len(priority),
        "priority_groups": sum(len(groups) for groups in priority.values()),
        "unique_priority_windows": len(priority_indices),
        "threshold_bucket_count": len(thresholds),
    }
    return ReplayPriorityPlan(
        strategy=strategy, priority_by_rollout=priority, summary=summary)


class PrioritizedReplayBatchSampler(Sampler[list[int]]):
    """Draw 50% ordinary windows and 50% priority windows.

    The ordinary half preserves the baseline's window-uniform marginal. The
    priority half chooses a trajectory first and then a priority block, so a
    long trajectory cannot dominate merely because it has more boundaries.
    A small number of trajectory slots keeps streaming-cache I/O practical.
    """

    def __init__(self, dataset: QPlanningWindowDataset, plan: ReplayPriorityPlan,
                 batch_size: int, seed: int):
        if batch_size < 2:
            raise ValueError("prioritized replay requires batch_size >= 2")
        self.dataset = dataset
        self.plan = plan
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        grouped, _ = _indices_by_rollout(dataset)
        self.uniform = {
            rollout_id: tuple(index for _, index in windows)
            for rollout_id, windows in grouped.items()
        }
        self.uniform_ids = tuple(sorted(self.uniform))
        self.uniform_weights = tuple(len(self.uniform[value]) for value in self.uniform_ids)
        self.priority_ids = tuple(sorted(plan.priority_by_rollout))
        valid = set(range(len(dataset)))
        used = {
            index for groups in plan.priority_by_rollout.values()
            for group in groups for index in group}
        if not used <= valid:
            raise ValueError("priority plan contains indices outside its dataset")

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    @staticmethod
    def _draw_windows(generator, candidates, count):
        candidates = tuple(dict.fromkeys(candidates))
        if count <= len(candidates):
            return generator.sample(candidates, count)
        return list(candidates) + generator.choices(candidates, k=count - len(candidates))

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        n_priority = int(round(self.batch_size * PRIORITY_FRACTION))
        n_uniform = self.batch_size - n_priority
        # At batch=64 this opens about 12 rollouts: four ordinary slots carrying
        # eight windows each and eight priority slots carrying four each.
        uniform_slots = min(n_uniform, max(1, n_uniform // 8))
        priority_slots = min(n_priority, max(1, n_priority // 4))
        rollout_for_index = {
            index: rollout_id
            for rollout_id, indices in self.uniform.items() for index in indices}
        offset_for_index = {
            index: int(offset)
            for index, (_, offset) in enumerate(self.dataset.indices)}

        for _ in range(len(self)):
            ordinary_ids = generator.choices(
                self.uniform_ids, weights=self.uniform_weights, k=uniform_slots)
            priority_ids = generator.choices(self.priority_ids, k=priority_slots)
            batch = []
            for slot, rollout_id in enumerate(ordinary_ids):
                quotient, remainder = divmod(n_uniform, len(ordinary_ids))
                count = quotient + int(slot < remainder)
                batch.extend(self._draw_windows(
                    generator, self.uniform[rollout_id], count))
            for slot, rollout_id in enumerate(priority_ids):
                quotient, remainder = divmod(n_priority, len(priority_ids))
                count = quotient + int(slot < remainder)
                candidates = tuple(
                    index for group in self.plan.priority_by_rollout[rollout_id]
                    for index in group)
                batch.extend(self._draw_windows(generator, candidates, count))
            generator.shuffle(batch)
            # Group disk reads while retaining the randomized batch contents.
            batch.sort(key=lambda index: (
                rollout_for_index[index], offset_for_index[index]))
            yield batch


def _state_digest(model: QPlanningCritic) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def run_q50_priority_worker(
        *, snapshot_id: str, strategies,
        cache_root: str | Path = "/content/qplanning_cache",
        output_root: str | Path = "/content/qplanning_priority",
        micro_batch_size: int = 64,
        cache_download_workers: int = 8,
        label_download_workers: int = 8,
        device=None, resume: bool = True,
        store: SupabaseStore | None = None) -> dict:
    """Train the requested ordinary Q50 critics from step zero.

    Each strategy gets a distinct checkpoint directory. Resume only restores a
    checkpoint from that exact strategy; it never loads the prior uniform Q50.
    """
    strategies = tuple(strategies)
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("strategies must be a non-empty sequence without duplicates")
    unknown = sorted(set(strategies) - set(PRIORITY_STRATEGIES))
    if unknown:
        raise ValueError(f"unknown replay strategies: {unknown}")
    if not snapshot_id.startswith("pcpcds-"):
        raise ValueError("paste the immutable pcpcds-* snapshot ID produced by notebook 56")

    store = store or SupabaseStore()
    snapshot = PCPCriticRegistry(store).load_snapshot(snapshot_id)
    cache = prepare_qplanning_streaming_cache(
        store, snapshot, horizon=50, gamma=.99, cache_root=cache_root,
        download_workers=cache_download_workers)
    _print_preflight(snapshot, cache)
    train = QPlanningWindowDataset(cache, snapshot.train_rollout_ids)
    validation = QPlanningWindowDataset(
        cache, snapshot.val_rollout_ids, max_windows=4096)

    rows = eligible_rollout_rows(store, rollout_ids=snapshot.train_rollout_ids)
    outcomes = {}
    for row in rows:
        if row.get("success") not in (True, False, 0, 1):
            raise ValueError(f"rollout {row['rollout_id']} has no Boolean success label")
        outcomes[row["rollout_id"]] = bool(row["success"])

    needs_u20 = any(strategy != "failure" for strategy in strategies)
    labels = None
    if needs_u20:
        labels = prepare_u20_label_cache(
            store, snapshot, cache, cache_root=cache_root,
            download_workers=label_download_workers)

    train_config = QPlanningTrainConfig(
        effective_batch_size=64, micro_batch_size=micro_batch_size,
        updates=8_000, warmup_updates=500, print_interval=100,
        eval_interval=500, checkpoint_interval=1_000,
        max_validation_transitions=4096)
    model_config = QPlanningModelConfig(action_horizon=50, action_dim=cache.action_dim)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reports = {}
    for strategy in strategies:
        plan = build_replay_priority_plan(
            train, strategy=strategy, outcomes=outcomes, labels=labels)
        print("\nReplay contract")
        for key, value in plan.summary.items():
            print(f"  {key}: {value}")
        print("  batch composition: 50% ordinary window-marginal + "
              "50% trajectory-first priority")
        print("  critic/loss/targets: unchanged ordinary Q50")

        # Reset before constructing every arm: all fresh runs start from the
        # exact same seed and initialization, independent of arm order.
        torch.manual_seed(train_config.seed)
        np.random.seed(train_config.seed)
        random.seed(train_config.seed)
        model = QPlanningCritic(
            prefix_dim=cache.prefix_dim, robot_dim=cache.robot_dim,
            proprio_dim=cache.proprio_dim, config=model_config)
        model.set_action_statistics(cache.action_mean, cache.action_std)
        initial_digest = _state_digest(model)
        sampler = PrioritizedReplayBatchSampler(
            train, plan, train_config.micro_batch_size, train_config.seed)
        contract_digest = hashlib.sha256(
            f"{cache.digest}|priority-v1|{plan.digest}".encode()).hexdigest()[:24]
        output_dir = (
            Path(output_root) / snapshot_id / f"q50_priority_{strategy}_full")
        print("\nTraining contract")
        print(f"  strategy={strategy}; fresh initialization={initial_digest}")
        print(f"  device={device}; updates={train_config.updates}")
        print(f"  effective batch={train_config.effective_batch_size}; "
              f"microbatch={train_config.micro_batch_size}; "
              f"accumulation={train_config.accumulation_steps}")
        print(f"  checkpoint directory: {output_dir}")
        _, _, report = train_qplanning_critic(
            model, train, validation, device,
            snapshot_id=snapshot_id, cache_digest=contract_digest,
            source_policy={
                "repo_id": snapshot.policy_repo_id,
                "revision": snapshot.policy_revision,
            },
            output_dir=output_dir, config=train_config,
            train_batch_sampler=sampler, resume=resume)
        report.update({
            "strategy": strategy,
            "priority_plan": plan.summary,
            "priority_plan_digest": plan.digest,
            "initial_state_digest": initial_digest,
        })
        reports[strategy] = report
        del model, sampler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports
