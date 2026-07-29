"""Dataset construction for clean t=1 action-chunk verification.

The observed rollouts persist decision-point observation encodings in ``pcp_chunks`` and the
postprocessed actions actually sent to LIBERO in ``trajectory``.  This module aligns both blobs
by chunk index, pads only terminal partial chunks, and keeps a validity mask so no invented action
is treated as data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import pickle
import time
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
    uncertainty_stratum: str | None = None
    trajectory_seed: int | None = None
    collection_split: str | None = None
    n_steps: int | None = None
    replay_action_count: int | None = None
    return_target: float | None = None

    @property
    def task_key(self) -> tuple[str, str, int]:
        return self.benchmark, self.suite, self.task_idx


ROLLOUT_COLUMNS = (
    "rollout_id,experiment,benchmark,suite,task_idx,episode_idx,success,"
    "n_steps,pcp_chunks_path,trajectory_path,status"
)


@dataclass
class ChunkTransitionExample:
    """One policy-decision transition with matching long and short actions."""

    rollout_id: str
    experiment: str
    benchmark: str
    suite: str
    task_idx: int
    episode_idx: int
    chunk_idx: int
    chunk_position: float
    next_chunk_position: float
    obs_enc: np.ndarray
    next_obs_enc: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    reward: float
    discount: float
    terminal: bool
    return_target: float

    @property
    def task_key(self) -> tuple[str, str, int]:
        return self.benchmark, self.suite, self.task_idx


def _download_with_retry(store, path: str, attempts: int = 5):
    """Retry transient Supabase/httpx transport failures with bounded backoff."""
    import httpx

    for attempt in range(attempts):
        try:
            return store._download(path)
        except (httpx.TransportError, httpx.TimeoutException, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


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


def build_chunk_transitions(row: dict, pcp_frame, trajectory: dict, *,
                            horizon: int = 50, action_dim: int = 7,
                            gamma: float = 0.999) -> list[ChunkTransitionExample]:
    """Align consecutive decision embeddings with the actions executed between them."""
    actions = np.asarray(trajectory["actions"], dtype=np.float32)[:, :action_dim]
    observations = {}
    positions = {}
    for chunk_idx, group in pcp_frame.groupby("chunk_idx", sort=True):
        first = group.iloc[0]
        observations[int(chunk_idx)] = np.asarray(
            first["obs_enc"], dtype=np.float32).reshape(-1)
        positions[int(chunk_idx)] = float(first["chunk_pos"])
    ordered = sorted(observations)
    out = []
    for offset, chunk_idx in enumerate(ordered):
        start = chunk_idx * horizon
        stop = min(start + horizon, len(actions))
        if start >= stop:
            continue
        is_last = offset == len(ordered) - 1
        next_idx = ordered[offset + 1] if not is_last else None
        # A nonterminal transition must span exactly one policy decision.
        if next_idx is not None and next_idx != chunk_idx + 1:
            continue
        n_actions = stop - start
        padded = np.zeros((horizon, action_dim), dtype=np.float32)
        mask = np.zeros(horizon, dtype=np.bool_)
        padded[:n_actions] = actions[start:stop]
        mask[:n_actions] = True
        terminal = next_idx is None
        success = bool(row["success"])
        remaining = max(len(actions) - start, 1)
        out.append(ChunkTransitionExample(
            rollout_id=row["rollout_id"], experiment=row["experiment"],
            benchmark=row["benchmark"], suite=row["suite"],
            task_idx=int(row["task_idx"]), episode_idx=int(row["episode_idx"]),
            chunk_idx=chunk_idx, chunk_position=positions[chunk_idx],
            next_chunk_position=(positions[next_idx] if next_idx is not None
                                 else positions[chunk_idx]),
            obs_enc=observations[chunk_idx],
            next_obs_enc=(observations[next_idx] if next_idx is not None
                          else np.zeros_like(observations[chunk_idx])),
            actions=padded, action_mask=mask,
            reward=float(success and terminal),
            discount=0.0 if terminal else float(gamma ** n_actions),
            terminal=terminal,
            return_target=float(success) * float(gamma ** (remaining - 1)),
        ))
    return out


def load_clean_chunk_examples(store, experiments: Sequence[str], *, horizon: int = 50,
                              action_dim: int = 7, progress=None,
                              cache_dir: str | Path | None = None
                              ) -> list[CleanChunkExample]:
    """Download and reconstruct clean examples for one or more rollout experiments."""
    import pandas as pd

    rows = _paged_rollouts(store, tuple(experiments))
    iterator = progress(rows) if progress else rows
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    examples: list[CleanChunkExample] = []
    for row in iterator:
        cache_path = (
            cache_dir / f"{row['rollout_id']}.pkl" if cache_dir is not None else None)
        if cache_path is not None and cache_path.exists():
            with cache_path.open("rb") as handle:
                examples.extend(pickle.load(handle))
            continue
        pcp = pd.read_parquet(io.BytesIO(
            _download_with_retry(store, row["pcp_chunks_path"])))
        with np.load(io.BytesIO(
                _download_with_retry(store, row["trajectory_path"]))) as trajectory:
            rollout_examples = build_clean_chunk_examples(
                row, pcp, trajectory, horizon=horizon, action_dim=action_dim)
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(rollout_examples, handle)
            temporary.replace(cache_path)
        examples.extend(rollout_examples)
    return examples


def load_chunk_transitions(store, experiments: Sequence[str], *, horizon: int = 50,
                           action_dim: int = 7, gamma: float = 0.999,
                           progress=None, cache_dir: str | Path | None = None
                           ) -> list[ChunkTransitionExample]:
    """Download historical artifacts and reconstruct policy-decision transitions."""
    import pandas as pd

    rows = _paged_rollouts(store, tuple(experiments))
    iterator = progress(rows) if progress else rows
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in iterator:
        cache_path = (cache_dir / f"{row['rollout_id']}.pkl"
                      if cache_dir is not None else None)
        if cache_path is not None and cache_path.exists():
            with cache_path.open("rb") as handle:
                examples.extend(pickle.load(handle))
            continue
        pcp = pd.read_parquet(io.BytesIO(
            _download_with_retry(store, row["pcp_chunks_path"])))
        with np.load(io.BytesIO(
                _download_with_retry(store, row["trajectory_path"]))) as trajectory:
            rollout_examples = build_chunk_transitions(
                row, pcp, trajectory, horizon=horizon, action_dim=action_dim,
                gamma=gamma)
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(rollout_examples, handle)
            temporary.replace(cache_path)
        examples.extend(rollout_examples)
    return examples


def load_candidate_examples(store, experiment="verifier-clean-pairs-v1", *,
                            cache_dir: str | Path | None = None):
    """Load exact/fallback candidate artifacts into the common verifier example type."""
    experiments = [experiment] if isinstance(experiment, str) else list(experiment)
    def pages(table, configure):
        rows, start = [], 0
        while True:
            batch = configure(store.client.table(table).select("*")).range(
                start, start + 999).execute().data or []
            rows.extend(batch)
            if len(batch) < 1000:
                return rows
            start += 1000

    groups = pages("verifier_candidate_groups", lambda query: (
        query.eq("experiment", experiments[0]) if len(experiments) == 1
        else query.in_("experiment", experiments)))
    group_ids = [group["candidate_group_id"] for group in groups]
    candidates = []
    for start in range(0, len(group_ids), 100):
        batch_ids = group_ids[start:start + 100]
        candidates.extend(pages(
            "verifier_candidates",
            lambda query, ids=batch_ids: query.in_("candidate_group_id", ids)))
    candidate_counts = Counter(row["candidate_group_id"] for row in candidates)
    # Old fallback IDs included a requested mid-rollout chunk even though the
    # actual branch was always episode-initial. Keep only the richest group per
    # true causal identity so repeated fallbacks cannot inflate pair counts.
    fallback = defaultdict(list)
    retained = []
    for group in groups:
        if group["pairing_mode"] == "paired_full_episode":
            key = (group["benchmark"], group["suite"], group["task_idx"], group["episode_idx"])
            fallback[key].append(group)
        else:
            retained.append(group)
    for members in fallback.values():
        retained.append(max(members, key=lambda g: (
            candidate_counts[g["candidate_group_id"]],
            experiments.index(g["experiment"]), g["candidate_group_id"])))
    groups = retained
    candidates_by_group = defaultdict(list)
    for candidate in candidates:
        candidates_by_group[candidate["candidate_group_id"]].append(candidate)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for group in groups:
        group_id = group["candidate_group_id"]
        rows = candidates_by_group[group_id]
        candidate_hash = hashlib.sha256("|".join(sorted(
            row["candidate_id"] for row in rows)).encode()).hexdigest()[:8]
        cache_path = (
            cache_dir / f"{group_id}-{candidate_hash}.pkl"
            if cache_dir is not None else None)
        if cache_path is not None and cache_path.exists():
            with cache_path.open("rb") as handle:
                examples.extend(pickle.load(handle))
            continue
        if not rows:
            continue
        with np.load(io.BytesIO(
                _download_with_retry(store, rows[0]["observation_path"]))) as artifact:
            obs_enc = np.asarray(artifact["obs_enc"], dtype=np.float32).reshape(-1)
        group_examples = []
        for row in rows:
            with np.load(io.BytesIO(
                    _download_with_retry(store, row["env_chunk_path"]))) as artifact:
                actions = np.asarray(artifact["actions"], dtype=np.float32)[:, :7]
                mask = (np.asarray(artifact["mask"], dtype=np.bool_) if "mask" in artifact
                        else np.ones(len(actions), dtype=np.bool_))
            padded = np.zeros((50, 7), dtype=np.float32)
            padded_mask = np.zeros(50, dtype=np.bool_)
            n = min(50, len(actions))
            padded[:n], padded_mask[:n] = actions[:n], mask[:n]
            group_examples.append(CleanChunkExample(
                rollout_id=row["candidate_id"], experiment=group["experiment"],
                benchmark=group["benchmark"], suite=group["suite"],
                task_idx=int(group["task_idx"]), episode_idx=int(group["episode_idx"]),
                chunk_idx=int(group["chunk_idx"]),
                chunk_position=float((group.get("metadata_json") or {}).get(
                    "chunk_position", group["chunk_idx"])),
                obs_enc=obs_enc, actions=padded, action_mask=padded_mask,
                success=int(bool(row["success"])), candidate_group_id=group_id,
                candidate_kind=row["candidate_kind"], pairing_mode=group["pairing_mode"],
                uncertainty_stratum=group.get("uncertainty_stratum"),
                trajectory_seed=(group.get("trajectory_seed") if group.get("trajectory_seed")
                                 is not None else (group.get("metadata_json") or {}).get(
                                     "trajectory_seed")),
                collection_split=(group.get("collection_split") or
                                  (group.get("metadata_json") or {}).get(
                                      "collection_split")),
                n_steps=(int(row["n_steps"]) if row.get("n_steps") is not None else None),
                replay_action_count=int((group.get("metadata_json") or {}).get(
                    "replay_action_count", 0)),
                return_target=(
                    float(bool(row["success"])) * .999 ** max(
                        int(row["n_steps"]) - int((group.get("metadata_json") or {}).get(
                            "replay_action_count", 0)) - 1, 0)
                    if row.get("n_steps") is not None else float(bool(row["success"]))),
            ))
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(group_examples, handle)
            temporary.replace(cache_path)
        examples.extend(group_examples)
    return examples


def _rollout_records(examples: Iterable[CleanChunkExample]) -> dict[str, CleanChunkExample]:
    return {example.rollout_id: example for example in examples}


def _stable_order(values, seed: int):
    def key(value):
        raw = f"{seed}|{value}".encode()
        return hashlib.sha256(raw).hexdigest()
    return sorted(values, key=key)


def _partition_sizes(n: int) -> dict[str, int]:
    if n >= 10:
        return {"train": n - 4, "val": 1, "cal": 1, "test": 2}
    raw = {"train": .6 * n, "val": .1 * n, "cal": .1 * n, "test": .2 * n}
    sizes = {name: int(value) for name, value in raw.items()}
    for name in sorted(raw, key=lambda key: raw[key] - sizes[key], reverse=True)[:n - sum(sizes.values())]:
        sizes[name] += 1
    return sizes


def _stratified_task_assignment(records: Sequence[CleanChunkExample], seed: int):
    """Fill fixed split capacities while spreading every outcome proportionally.

    In particular, two minority examples allocate to train and test under a 6/1/1/2 split,
    rather than both being consumed by the leading training slice.
    """
    names = ("train", "val", "cal", "test")
    capacities = _partition_sizes(len(records))
    assigned = {name: [] for name in names}
    by_label = defaultdict(list)
    for record in records:
        by_label[record.success].append(record.rollout_id)
    label_order = sorted(by_label, key=lambda label: (len(by_label[label]), label))
    for label in label_order:
        ids = _stable_order(by_label[label], seed + int(label))
        ideal = {name: len(ids) * capacities[name] / max(len(records), 1) for name in names}
        label_counts = {name: 0 for name in names}
        tie_order = list(names)
        shift = (seed + int(label)) % len(names)
        tie_order = tie_order[shift:] + tie_order[:shift]
        for rollout_id in ids:
            available = [name for name in names if len(assigned[name]) < capacities[name]]
            chosen = max(available, key=lambda name: (
                ideal[name] - label_counts[name], -tie_order.index(name)))
            assigned[chosen].append(rollout_id)
            label_counts[chosen] += 1
    return assigned


def known_task_split(examples: Sequence[CleanChunkExample], seed: int = 42) -> dict[str, list[str]]:
    """Deterministic 6/1/1/2 rollout split within every ten-episode task."""
    grouped: dict[tuple[str, str, int], list[CleanChunkExample]] = defaultdict(list)
    for record in _rollout_records(examples).values():
        grouped[record.task_key].append(record)
    split = {name: [] for name in ("train", "val", "cal", "test")}
    for task, records in sorted(grouped.items()):
        task_seed = seed + int(hashlib.md5(str(task).encode()).hexdigest()[:6], 16)
        assignment = _stratified_task_assignment(records, task_seed)
        for name, ids in assignment.items():
            split[name].extend(ids)
    return split


def _candidate_groups(examples: Sequence[CleanChunkExample]):
    groups: dict[str, list[CleanChunkExample]] = defaultdict(list)
    for example in examples:
        if example.candidate_group_id:
            groups[example.candidate_group_id].append(example)
    return groups


def validate_candidate_groups(examples: Sequence[CleanChunkExample], *,
                              expected_candidates: int | None = None):
    """Reject partial, duplicated, or non-same-state candidate groups."""
    groups = _candidate_groups(examples)
    if not groups:
        raise ValueError("candidate dataset contains no groups")
    errors = []
    for gid, members in groups.items():
        if expected_candidates is not None and len(members) != expected_candidates:
            errors.append(f"{gid}: expected {expected_candidates} candidates, got {len(members)}")
        ids = [member.rollout_id for member in members]
        if len(ids) != len(set(ids)):
            errors.append(f"{gid}: duplicate candidate IDs")
        if sum(member.candidate_kind == "default" for member in members) != 1:
            errors.append(f"{gid}: expected exactly one default candidate")
        first = members[0]
        state_key = (
            first.benchmark, first.suite, first.task_idx, first.episode_idx,
            first.chunk_idx, first.chunk_position,
        )
        for member in members[1:]:
            member_key = (
                member.benchmark, member.suite, member.task_idx, member.episode_idx,
                member.chunk_idx, member.chunk_position,
            )
            if member_key != state_key or not np.array_equal(member.obs_enc, first.obs_enc):
                errors.append(f"{gid}: candidates do not share one observation/state")
                break
    if errors:
        raise ValueError("invalid candidate groups:\n" + "\n".join(errors[:20]))
    return {
        "groups": len(groups),
        "candidates": sum(map(len, groups.values())),
        "discordant_groups": sum(
            len({member.success for member in members}) > 1
            for members in groups.values()),
    }


def _candidate_ids(groups, group_ids):
    return [example.rollout_id for gid in group_ids for example in groups[gid]]


def _balanced_group_order(groups, group_ids, seed):
    """Stable interleave across benchmark/uncertainty strata."""
    buckets = defaultdict(list)
    for gid in group_ids:
        first = groups[gid][0]
        buckets[(first.benchmark, first.uncertainty_stratum or "unknown")].append(gid)
    for key in buckets:
        buckets[key] = _stable_order(
            buckets[key], seed + int(hashlib.md5(str(key).encode()).hexdigest()[:6], 16))
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return ordered


def locked_candidate_split(examples: Sequence[CleanChunkExample], test_fraction: float = .20,
                           seed: int = 42):
    """Lock a group-safe test set while preserving discordance and coarse cohort balance."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    groups = _candidate_groups(examples)
    discordant = {
        gid: len({example.success for example in members}) > 1
        for gid, members in groups.items()
    }
    development, test = [], []
    for status in (False, True):
        ids = [gid for gid in groups if discordant[gid] == status]
        ordered = _balanced_group_order(groups, ids, seed + int(status))
        n_test = min(len(ordered), max(1, round(test_fraction * len(ordered))))
        test.extend(ordered[:n_test])
        development.extend(ordered[n_test:])
    return {
        "development": _candidate_ids(groups, development),
        "test": _candidate_ids(groups, test),
    }


def candidate_cv_splits(examples: Sequence[CleanChunkExample],
                        development_ids: Iterable[str], n_folds: int = 4,
                        seed: int = 42):
    """Return deterministic episode-safe folds over a development partition."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    development = select_examples(examples, development_ids)
    groups = _candidate_groups(development)
    group_discordant = {
        gid: len({example.success for example in members}) > 1
        for gid, members in groups.items()
    }
    if sum(group_discordant.values()) < n_folds:
        raise ValueError(
            f"need at least {n_folds} discordant development groups for {n_folds}-fold CV")

    episode_groups = defaultdict(list)
    for gid, members in groups.items():
        first = members[0]
        episode_groups[(
            first.benchmark, first.suite, first.task_idx, first.episode_idx
        )].append(gid)
    episode_discordant = {
        identity: any(group_discordant[gid] for gid in gids)
        for identity, gids in episode_groups.items()
    }
    fold_episodes = [[] for _ in range(n_folds)]
    for status in (False, True):
        identities = [
            identity for identity in episode_groups
            if episode_discordant[identity] == status]
        ordered = _stable_order(identities, seed + int(status))
        for index, identity in enumerate(ordered):
            fold_episodes[index % n_folds].append(identity)
    all_episodes = set(episode_groups)
    return [{
        "train": _candidate_ids(
            groups, sorted(gid for identity in all_episodes - set(validation)
                           for gid in episode_groups[identity])),
        "val": _candidate_ids(
            groups, sorted(gid for identity in validation
                           for gid in episode_groups[identity])),
    } for validation in fold_episodes]


def candidate_episode_identities(examples: Sequence[CleanChunkExample]):
    return {
        (example.benchmark, example.suite, example.task_idx, example.episode_idx)
        for example in examples
    }


def exclude_candidate_identities(historical: Sequence[CleanChunkExample],
                                 candidates: Sequence[CleanChunkExample]):
    return exclude_episode_identities(
        historical, candidate_episode_identities(candidates))


def exclude_episode_identities(
        examples: Sequence[CleanChunkExample],
        held_out: Iterable[tuple[str, str, int, int]]):
    held_out = set(held_out)
    return [example for example in examples if (
        example.benchmark, example.suite, example.task_idx, example.episode_idx
    ) not in held_out]


def prefix_action_statistics(examples: Sequence[CleanChunkExample], prefix_length: int = 10):
    valid = []
    for example in examples:
        mask = example.action_mask.copy()
        mask[prefix_length:] = False
        if mask.any():
            valid.append(example.actions[mask])
    if not valid:
        raise ValueError("cannot compute prefix action statistics from an empty dataset")
    actions = np.concatenate(valid).astype(np.float64)
    return actions.mean(0).astype(np.float32), actions.std(0).clip(min=1e-6).astype(np.float32)


def shuffle_candidate_actions_within_group(examples: Sequence[CleanChunkExample],
                                           seed: int = 42):
    """Apply a uniformly sampled derangement within each candidate state."""
    rng = np.random.default_rng(seed)
    output = list(examples)
    index_groups = defaultdict(list)
    for index, example in enumerate(examples):
        if example.candidate_group_id:
            index_groups[example.candidate_group_id].append(index)
    for indices in index_groups.values():
        donors = np.asarray(indices)
        if len(donors) > 1:
            for _ in range(100):
                candidate = rng.permutation(donors)
                if np.all(candidate != donors):
                    donors = candidate
                    break
            else:
                donors = np.roll(donors, 1)
        for target, donor in zip(indices, donors):
            output[target] = replace(
                examples[target], actions=examples[donor].actions.copy(),
                action_mask=examples[donor].action_mask.copy())
    return output


def select_examples(examples: Sequence[CleanChunkExample], rollout_ids: Iterable[str]):
    wanted = set(rollout_ids)
    return [example for example in examples if example.rollout_id in wanted]
