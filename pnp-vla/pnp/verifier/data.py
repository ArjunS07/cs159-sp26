"""Dataset construction for clean t=1 action-chunk verification.

The observed rollouts persist decision-point observation encodings in ``pcp_chunks`` and the
postprocessed actions actually sent to LIBERO in ``trajectory``.  This module aligns both blobs
by chunk index, pads only terminal partial chunks, and keeps a validity mask so no invented action
is treated as data.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import io
from typing import Iterable, Sequence

import numpy as np


@dataclass
class CleanChunkExample:
    rollout_id: str
    experiment: str
    benchmark: str
    suite: str
    task_idx: int
    episode_idx: int
    chunk_idx: int
    chunk_position: float
    obs_enc: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    success: int
    candidate_group_id: str | None = None
    candidate_kind: str | None = None
    pairing_mode: str | None = None

    @property
    def task_key(self) -> tuple[str, str, int]:
        return self.benchmark, self.suite, self.task_idx


ROLLOUT_COLUMNS = (
    "rollout_id,experiment,benchmark,suite,task_idx,episode_idx,success,"
    "pcp_chunks_path,trajectory_path,status"
)


def _paged_rollouts(store, experiments: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    page_size = 1000
    for experiment in experiments:
        start = 0
        while True:
            q = (store.client.table("rollouts").select(ROLLOUT_COLUMNS)
                 .eq("experiment", experiment).eq("status", "completed")
                 .not_.is_("pcp_chunks_path", "null").not_.is_("trajectory_path", "null")
                 .range(start, start + page_size - 1))
            batch = q.execute().data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
    return rows


def build_clean_chunk_examples(row: dict, pcp_frame, trajectory: dict, *,
                               horizon: int = 50, action_dim: int = 7) -> list[CleanChunkExample]:
    """Align one rollout's observation encodings with its executed environment-space actions."""
    actions = np.asarray(trajectory["actions"], dtype=np.float32)[:, :action_dim]
    out: list[CleanChunkExample] = []
    for chunk_idx, group in pcp_frame.groupby("chunk_idx", sort=True):
        chunk_idx = int(chunk_idx)
        start = chunk_idx * horizon
        stop = min(start + horizon, len(actions))
        if start >= stop:
            continue
        n = stop - start
        padded = np.zeros((horizon, action_dim), dtype=np.float32)
        mask = np.zeros(horizon, dtype=np.bool_)
        padded[:n] = actions[start:stop]
        mask[:n] = True
        first = group.iloc[0]
        out.append(CleanChunkExample(
            rollout_id=row["rollout_id"], experiment=row["experiment"],
            benchmark=row["benchmark"], suite=row["suite"], task_idx=int(row["task_idx"]),
            episode_idx=int(row["episode_idx"]), chunk_idx=chunk_idx,
            chunk_position=float(first["chunk_pos"]),
            obs_enc=np.asarray(first["obs_enc"], dtype=np.float32).reshape(-1),
            actions=padded, action_mask=mask, success=int(bool(row["success"])),
        ))
    return out


def load_clean_chunk_examples(store, experiments: Sequence[str], *, horizon: int = 50,
                              action_dim: int = 7, progress=None) -> list[CleanChunkExample]:
    """Download and reconstruct clean examples for one or more rollout experiments."""
    import pandas as pd

    rows = _paged_rollouts(store, tuple(experiments))
    iterator = progress(rows) if progress else rows
    examples: list[CleanChunkExample] = []
    for row in iterator:
        pcp = pd.read_parquet(io.BytesIO(store._download(row["pcp_chunks_path"])))
        with np.load(io.BytesIO(store._download(row["trajectory_path"]))) as trajectory:
            examples.extend(build_clean_chunk_examples(
                row, pcp, trajectory, horizon=horizon, action_dim=action_dim))
    return examples


def load_candidate_examples(store, experiment: str = "verifier-clean-pairs-v1"):
    """Load exact/fallback candidate artifacts into the common verifier example type."""
    groups = (store.client.table("verifier_candidate_groups").select("*")
              .eq("experiment", experiment).execute().data or [])
    group_by_id = {row["candidate_group_id"]: row for row in groups}
    candidates = store.client.table("verifier_candidates").select("*").execute().data or []
    examples = []
    for row in candidates:
        group = group_by_id.get(row["candidate_group_id"])
        if group is None:
            continue
        with np.load(io.BytesIO(store._download(row["env_chunk_path"]))) as artifact:
            actions = np.asarray(artifact["actions"], dtype=np.float32)[:, :7]
            mask = (np.asarray(artifact["mask"], dtype=np.bool_) if "mask" in artifact
                    else np.ones(len(actions), dtype=np.bool_))
        with np.load(io.BytesIO(store._download(row["observation_path"]))) as artifact:
            obs_enc = np.asarray(artifact["obs_enc"], dtype=np.float32).reshape(-1)
        padded = np.zeros((50, 7), dtype=np.float32)
        padded_mask = np.zeros(50, dtype=np.bool_)
        n = min(50, len(actions))
        padded[:n], padded_mask[:n] = actions[:n], mask[:n]
        examples.append(CleanChunkExample(
            rollout_id=row["candidate_id"], experiment=experiment,
            benchmark=group["benchmark"], suite=group["suite"],
            task_idx=int(group["task_idx"]), episode_idx=int(group["episode_idx"]),
            chunk_idx=int(group["chunk_idx"]),
            chunk_position=float((group.get("metadata_json") or {}).get(
                "chunk_position", group["chunk_idx"])),
            obs_enc=obs_enc, actions=padded, action_mask=padded_mask,
            success=int(bool(row["success"])), candidate_group_id=row["candidate_group_id"],
            candidate_kind=row["candidate_kind"], pairing_mode=group["pairing_mode"],
        ))
    return examples


def _rollout_records(examples: Iterable[CleanChunkExample]) -> dict[str, CleanChunkExample]:
    return {example.rollout_id: example for example in examples}


def hard_task_keys(examples: Iterable[CleanChunkExample], lo: float = 0.10,
                   hi: float = 0.90) -> set[tuple[str, str, int]]:
    """Return mixed-outcome tasks, computing rates once per rollout rather than per chunk."""
    outcomes: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for example in _rollout_records(examples).values():
        outcomes[example.task_key].append(example.success)
    return {key for key, values in outcomes.items() if lo < np.mean(values) < hi}


def action_statistics(examples: Sequence[CleanChunkExample]):
    """Training-only per-dimension statistics over valid environment actions."""
    valid = [example.actions[example.action_mask] for example in examples
             if example.action_mask.any()]
    if not valid:
        raise ValueError("cannot compute action statistics from an empty dataset")
    actions = np.concatenate(valid, axis=0).astype(np.float64)
    return actions.mean(0).astype(np.float32), actions.std(0).clip(min=1e-6).astype(np.float32)


def _stable_order(values, seed: int):
    def key(value):
        raw = f"{seed}|{value}".encode()
        return hashlib.sha256(raw).hexdigest()
    return sorted(values, key=key)


def _interleaved_rollouts(records: Sequence[CleanChunkExample], seed: int) -> list[str]:
    by_label = {0: [], 1: []}
    for record in records:
        by_label[record.success].append(record.rollout_id)
    queues = {label: _stable_order(ids, seed + label) for label, ids in by_label.items()}
    result: list[str] = []
    while queues[0] or queues[1]:
        label = 0 if len(queues[0]) >= len(queues[1]) else 1
        if queues[label]:
            result.append(queues[label].pop(0))
        other = 1 - label
        if queues[other]:
            result.append(queues[other].pop(0))
    return result


def known_task_split(examples: Sequence[CleanChunkExample], seed: int = 42) -> dict[str, list[str]]:
    """Deterministic 6/1/1/2 rollout split within every ten-episode task."""
    grouped: dict[tuple[str, str, int], list[CleanChunkExample]] = defaultdict(list)
    for record in _rollout_records(examples).values():
        grouped[record.task_key].append(record)
    split = {name: [] for name in ("train", "val", "cal", "test")}
    for task, records in sorted(grouped.items()):
        ids = _interleaved_rollouts(records, seed + int(hashlib.md5(str(task).encode()).hexdigest()[:6], 16))
        n = len(ids)
        if n >= 10:
            cuts = (6, 7, 8)
        else:
            cuts = (max(1, round(.6 * n)), max(2, round(.7 * n)), max(3, round(.8 * n)))
        split["train"].extend(ids[:cuts[0]])
        split["val"].extend(ids[cuts[0]:cuts[1]])
        split["cal"].extend(ids[cuts[1]:cuts[2]])
        split["test"].extend(ids[cuts[2]:])
    return split


def heldout_task_split(examples: Sequence[CleanChunkExample], fold: int = 0,
                       n_folds: int = 5, seed: int = 42) -> dict[str, list[str]]:
    """Task-disjoint 24/4/4/8 split for the canonical 40-task hard cohort."""
    records = _rollout_records(examples)
    strata: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    task_to_ids: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for record in records.values():
        task_to_ids[record.task_key].append(record.rollout_id)
    for task in task_to_ids:
        strata[task[:2]].append(task)
    test_tasks: set[tuple[str, str, int]] = set()
    remaining: list[tuple[str, str, int]] = []
    for stratum, tasks in sorted(strata.items()):
        ordered = _stable_order(tasks, seed + int(hashlib.md5(str(stratum).encode()).hexdigest()[:6], 16))
        test_tasks.update(task for i, task in enumerate(ordered) if i % n_folds == fold)
        remaining.extend(task for i, task in enumerate(ordered) if i % n_folds != fold)
    remaining = _stable_order(remaining, seed + 1000 + fold)
    n_aux = max(1, len(test_tasks) // 2)
    val_tasks, cal_tasks = set(remaining[:n_aux]), set(remaining[n_aux:2 * n_aux])
    train_tasks = set(remaining[2 * n_aux:])
    split_tasks = {"train": train_tasks, "val": val_tasks, "cal": cal_tasks, "test": test_tasks}
    return {name: [rid for task in sorted(tasks) for rid in sorted(task_to_ids[task])]
            for name, tasks in split_tasks.items()}


def candidate_group_split(examples: Sequence[CleanChunkExample], seed: int = 42):
    """Deterministic 60/10/10/20 split that never separates a matched candidate group."""
    grouped, group_task = defaultdict(list), {}
    for example in examples:
        if example.candidate_group_id:
            grouped[example.candidate_group_id].append(example.rollout_id)
            group_task[example.candidate_group_id] = example.task_key
    ordered = _stable_order(grouped, seed)
    n = len(ordered)
    a, b, c = round(.6 * n), round(.7 * n), round(.8 * n)
    assignments = {"train": ordered[:a], "val": ordered[a:b],
                   "cal": ordered[b:c], "test": ordered[c:]}
    output = {name: [] for name in ("train", "val", "cal", "test")}
    for name, ids in assignments.items():
        output[name].extend(candidate_id for gid in ids for candidate_id in grouped[gid])
    return output


def select_examples(examples: Sequence[CleanChunkExample], rollout_ids: Iterable[str]):
    wanted = set(rollout_ids)
    return [example for example in examples if example.rollout_id in wanted]


def shuffle_actions_within_task(examples: Sequence[CleanChunkExample], seed: int = 42):
    """Return a shortcut-control copy with actions permuted within task and chunk index."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for index, example in enumerate(examples):
        groups[(example.task_key, example.chunk_idx)].append(index)
    donors = list(range(len(examples)))
    for indices in groups.values():
        permuted = list(indices)
        if len(permuted) > 1:
            while any(a == b for a, b in zip(indices, permuted)):
                rng.shuffle(permuted)
        for target, donor in zip(indices, permuted):
            donors[target] = donor
    return [replace(example, actions=examples[donor].actions.copy(),
                    action_mask=examples[donor].action_mask.copy())
            for example, donor in zip(examples, donors)]


def zero_actions(examples: Sequence[CleanChunkExample]):
    """Return an action-ablated copy while preserving masks and every identity field."""
    return [replace(example, actions=np.zeros_like(example.actions)) for example in examples]
