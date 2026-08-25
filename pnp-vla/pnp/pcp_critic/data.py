"""Immutable, lossless PCP-critic datasets from PCP-search rollout artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from ..pcp_search.data import ACTION_HORIZON, validate_training_artifact


@dataclass(frozen=True)
class PCPCriticTransition:
    rollout_id: str
    group_id: str
    benchmark: str
    suite: str
    task_idx: int
    chunk_idx: int
    prefix_embeddings: np.ndarray
    prefix_pad_mask: np.ndarray
    robot_state: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    next_prefix_embeddings: np.ndarray
    next_prefix_pad_mask: np.ndarray
    next_robot_state: np.ndarray
    next_proprio: np.ndarray
    next_action: np.ndarray
    reward: float
    discount: float
    mc_return: float
    terminal: bool


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    rollout_ids: tuple[str, ...]
    train_rollout_ids: tuple[str, ...]
    val_rollout_ids: tuple[str, ...]
    policy_repo_id: str
    policy_revision: str
    artifact_schema_version: int
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    provenance: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "DatasetSnapshot":
        return cls(**{
            **value,
            "rollout_ids": tuple(value["rollout_ids"]),
            "train_rollout_ids": tuple(value["train_rollout_ids"]),
            "val_rollout_ids": tuple(value["val_rollout_ids"]),
            "action_mean": tuple(value["action_mean"]),
            "action_std": tuple(value["action_std"]),
        })


@dataclass(frozen=True)
class CompactCacheIndex:
    """Local, bounded-memory representation of a snapshot for Colab training."""
    snapshot_id: str
    cache_dir: str
    rollouts: tuple[dict, ...]
    prefix_dim: int
    robot_dim: int
    proprio_dim: int
    n_transitions: int

    @property
    def digest(self) -> str:
        payload = json.dumps({"snapshot_id": self.snapshot_id, "rollouts": self.rollouts},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


_ROLLOUT_FIELDS = (
    "rollout_id,benchmark,suite,task_idx,episode_idx,init_state_hash,training_data_path,"
    "training_data_schema_version,run_id,"
    "pcp_train_eligible,training_ready"
)


def _policy_contract(row: dict) -> tuple[str, str]:
    return (str(row.get("policy_repo_id") or row.get("model_repo_id") or ""),
            str(row.get("policy_revision") or row.get("model_revision") or ""))


def eligible_rollout_rows(store, *, rollout_ids: Iterable[str] | None = None) -> list[dict]:
    """Return only train-eligible, artifact-complete PCP-search rows.

    The explicit eligibility filter is the hard boundary protecting held-out PRO
    suites from training and normalization.
    """
    wanted = set(rollout_ids or ())
    rows = store.fetch_all(
        "rollouts", _ROLLOUT_FIELDS,
        configure=lambda q: q.eq("training_ready", True).eq("pcp_train_eligible", True)
        .not_.is_("training_data_path", "null"),
        order_by=("rollout_id",))
    if wanted:
        rows = [row for row in rows if row["rollout_id"] in wanted]
        missing = wanted - {row["rollout_id"] for row in rows}
        if missing:
            raise ValueError(f"requested rollouts are not eligible: {sorted(missing)[:3]}")
    run_ids = sorted({row.get("run_id") for row in rows if row.get("run_id")})
    runs = []
    for start in range(0, len(run_ids), 100):
        runs.extend(store.fetch_all(
            "experiment_runs", "run_id,model_repo_id,model_revision",
            configure=lambda q, batch=run_ids[start:start + 100]: q.in_("run_id", batch),
            order_by=("run_id",)))
    by_run = {row["run_id"]: row for row in runs}
    for row in rows:
        run = by_run.get(row.get("run_id"))
        if not run:
            raise ValueError(f"eligible rollout {row['rollout_id']} has no provenance run")
        row["policy_repo_id"] = run.get("model_repo_id") or ""
        row["policy_revision"] = run.get("model_revision") or ""
    return rows


def _squeeze_prefix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    # Boundary artifacts have a leading boundary axis. Individual prefix values
    # should be (tokens, width); tolerate a singleton lane axis from old batches.
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"expected prefix embeddings [tokens,width], got {value.shape}")
    return value.astype(np.float32, copy=False)


def _squeeze_mask(value: np.ndarray, tokens: int) -> np.ndarray:
    value = np.asarray(value)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    value = value.reshape(-1).astype(bool, copy=False)
    if len(value) != tokens:
        raise ValueError(f"prefix mask width {len(value)} does not match tokens {tokens}")
    return value


def transitions_from_artifact(row: dict, arrays: dict[str, np.ndarray], *, gamma: float) -> list[PCPCriticTransition]:
    """Build exact H=10 semi-MDP transitions without using simulator state."""
    validate_training_artifact(arrays)
    actions = np.asarray(arrays["bellman/action"], np.float32)
    next_actions = np.asarray(arrays["bellman/next_action"], np.float32)
    if actions.shape[1:] != next_actions.shape[1:] or actions.shape[1] < ACTION_HORIZON:
        raise ValueError("incompatible PCP critic action chunk shapes")
    rewards = np.asarray(arrays["bellman/rewards"], np.float32)
    valid = np.asarray(arrays["bellman/validity_mask"], bool)
    terminated = np.asarray(arrays["bellman/terminated"], bool)
    truncated = np.asarray(arrays["bellman/truncated"], bool)
    prefix = arrays["prefix/prefix_embeddings"]
    pad = arrays["prefix/prefix_pad_masks"]
    robot = np.asarray(arrays["boundary/raw_robot_state"], np.float32)
    proprio = np.asarray(arrays["boundary/policy_proprio"], np.float32)
    n = len(actions)
    if not (len(prefix) == len(pad) == len(robot) == len(proprio) == n + 1):
        raise ValueError("decision-boundary tensors do not align with Bellman transitions")
    returns = np.zeros(n, np.float32)
    continuation = 0.0
    for i in range(n - 1, -1, -1):
        width = int(valid[i].sum())
        discounted = sum((gamma ** j) * float(rewards[i, j]) for j in range(width))
        terminal = bool(np.any((terminated[i] | truncated[i]) & valid[i]))
        returns[i] = discounted + (0.0 if terminal else (gamma ** width) * continuation)
        continuation = returns[i]
    group = "|".join(str(row.get(key) or "") for key in ("benchmark", "suite", "task_idx", "init_state_hash"))
    result = []
    for i in range(n):
        width = int(valid[i].sum())
        terminal = bool(np.any((terminated[i] | truncated[i]) & valid[i]))
        state_prefix = _squeeze_prefix(prefix[i])
        next_prefix = _squeeze_prefix(prefix[i + 1])
        result.append(PCPCriticTransition(
            rollout_id=row["rollout_id"], group_id=group,
            benchmark=str(row.get("benchmark") or ""), suite=str(row.get("suite") or ""),
            task_idx=int(row.get("task_idx") or 0), chunk_idx=i,
            prefix_embeddings=state_prefix, prefix_pad_mask=_squeeze_mask(pad[i], len(state_prefix)),
            robot_state=np.asarray(robot[i], np.float32), proprio=np.asarray(proprio[i], np.float32),
            action=actions[i], next_prefix_embeddings=next_prefix,
            next_prefix_pad_mask=_squeeze_mask(pad[i + 1], len(next_prefix)),
            next_robot_state=np.asarray(robot[i + 1], np.float32),
            next_proprio=np.asarray(proprio[i + 1], np.float32), next_action=next_actions[i],
            reward=float(sum((gamma ** j) * float(rewards[i, j]) for j in range(width))),
            discount=0.0 if terminal else float(gamma ** width), mc_return=float(returns[i]),
            terminal=terminal))
    return result


def load_transitions(store, rows: Iterable[dict], *, gamma: float = .99) -> list[PCPCriticTransition]:
    transitions = []
    for row in rows:
        arrays = store.load_training_data(row["training_data_path"])
        transitions.extend(transitions_from_artifact(row, arrays, gamma=gamma))
    return transitions


def _stream_action_statistics(store, rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate artifacts and calculate action moments without retaining rollouts.

    Snapshot creation is deliberately metadata-heavy: one artifact may be in RAM,
    but no prefix/image tensor survives the next iteration.
    """
    count = 0
    mean = m2 = None
    n_transitions = 0
    for row in rows:
        arrays = store.load_training_data(row["training_data_path"])
        validation = validate_training_artifact(arrays)
        actions = np.asarray(arrays["bellman/action"], np.float64).reshape(-1, arrays["bellman/action"].shape[-1])
        batch_count = len(actions)
        batch_mean = actions.mean(axis=0)
        batch_m2 = ((actions - batch_mean) ** 2).sum(axis=0)
        if mean is None:
            count, mean, m2 = batch_count, batch_mean, batch_m2
        else:
            delta = batch_mean - mean
            total = count + batch_count
            m2 = m2 + batch_m2 + delta ** 2 * count * batch_count / total
            mean = mean + delta * batch_count / total
            count = total
        n_transitions += int(validation["n_transitions"])
        del arrays, actions
    if not count:
        raise ValueError("eligible artifacts contain no actions")
    return mean.astype(np.float32), np.sqrt(m2 / count).astype(np.float32).clip(1e-6), n_transitions


def _cache_rollout_path(cache_dir: Path, rollout_id: str) -> Path:
    return cache_dir / f"{rollout_id}.npz"


def _write_compact_rollout(path: Path, row: dict, arrays: dict[str, np.ndarray], *, gamma: float) -> dict:
    """Persist only critic inputs/targets, never images or simulator state."""
    transitions = transitions_from_artifact(row, arrays, gamma=gamma)
    if not transitions:
        raise ValueError(f"rollout {row['rollout_id']} contains no critic transitions")
    first = transitions[0]
    packed = {
        "prefix": np.stack([item.prefix_embeddings for item in transitions]).astype(np.float16),
        "pad": np.stack([item.prefix_pad_mask for item in transitions]).astype(bool),
        "robot": np.stack([item.robot_state for item in transitions]).astype(np.float32),
        "proprio": np.stack([item.proprio for item in transitions]).astype(np.float32),
        "action": np.stack([item.action for item in transitions]).astype(np.float32),
        "next_prefix": np.stack([item.next_prefix_embeddings for item in transitions]).astype(np.float16),
        "next_pad": np.stack([item.next_prefix_pad_mask for item in transitions]).astype(bool),
        "next_robot": np.stack([item.next_robot_state for item in transitions]).astype(np.float32),
        "next_proprio": np.stack([item.next_proprio for item in transitions]).astype(np.float32),
        "next_action": np.stack([item.next_action for item in transitions]).astype(np.float32),
        "reward": np.asarray([item.reward for item in transitions], np.float32),
        "discount": np.asarray([item.discount for item in transitions], np.float32),
        "mc_return": np.asarray([item.mc_return for item in transitions], np.float32),
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(temporary, path)
    return {"rollout_id": row["rollout_id"], "path": path.name, "n_transitions": len(transitions),
            "group_id": first.group_id, "benchmark": first.benchmark, "suite": first.suite,
            "task_idx": first.task_idx, "prefix_dim": int(first.prefix_embeddings.shape[-1]),
            "robot_dim": len(first.robot_state), "proprio_dim": len(first.proprio)}


def prepare_compact_cache(store, snapshot: DatasetSnapshot, *, gamma: float,
                          cache_root: str | Path) -> CompactCacheIndex:
    """Create/resume a local compact cache one rollout at a time.

    This is intentionally a local Colab cache. It makes training restartable while
    keeping raw observations and simulator state out of both RAM and the cache.
    """
    cache_dir = Path(cache_root).expanduser() / snapshot.snapshot_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = eligible_rollout_rows(store, rollout_ids=snapshot.rollout_ids)
    manifest_path = cache_dir / "index.json"
    known = {}
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text())
            if old.get("snapshot_id") == snapshot.snapshot_id:
                known = {item["rollout_id"]: item for item in old.get("rollouts", [])}
        except (OSError, json.JSONDecodeError):
            pass
    entries = []
    for index, row in enumerate(rows, start=1):
        path = _cache_rollout_path(cache_dir, row["rollout_id"])
        entry = known.get(row["rollout_id"])
        if entry and path.exists():
            entries.append(entry)
            continue
        arrays = store.load_training_data(row["training_data_path"])
        try:
            entry = _write_compact_rollout(path, row, arrays, gamma=gamma)
        finally:
            del arrays
        entries.append(entry)
        if index % 25 == 0 or index == len(rows):
            print(f"[pcp-critic] compact cache: {index}/{len(rows)} rollouts")
    dims = {(item["prefix_dim"], item["robot_dim"], item["proprio_dim"]) for item in entries}
    if len(dims) != 1:
        raise ValueError(f"mixed critic input dimensions require separate snapshots: {dims}")
    prefix_dim, robot_dim, proprio_dim = next(iter(dims))
    payload = {"cache_schema_version": 1, "snapshot_id": snapshot.snapshot_id,
               "rollouts": entries, "prefix_dim": prefix_dim, "robot_dim": robot_dim,
               "proprio_dim": proprio_dim, "n_transitions": sum(item["n_transitions"] for item in entries)}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temporary, manifest_path)
    return CompactCacheIndex(snapshot_id=snapshot.snapshot_id, cache_dir=str(cache_dir),
                             rollouts=tuple(entries), prefix_dim=prefix_dim, robot_dim=robot_dim,
                             proprio_dim=proprio_dim, n_transitions=payload["n_transitions"])


def _partition(rows: list[dict], *, seed: int, train_fraction: float) -> tuple[list[str], list[str]]:
    groups: dict[str, list[str]] = {}
    for row in rows:
        key = "|".join(str(row.get(name) or "") for name in
                       ("benchmark", "suite", "task_idx", "init_state_hash"))
        groups.setdefault(key, []).append(row["rollout_id"])
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
    cutoff = max(1, min(len(ordered) - 1, round(len(ordered) * train_fraction))) if len(ordered) > 1 else 1
    train_groups = set(ordered[:cutoff])
    train = sorted(rid for group, ids in groups.items() if group in train_groups for rid in ids)
    val = sorted(rid for group, ids in groups.items() if group not in train_groups for rid in ids)
    return train, val


def create_snapshot(store, *, name: str, seed: int = 42, train_fraction: float = .8,
                    gamma: float = .99, rollout_ids: Iterable[str] | None = None) -> DatasetSnapshot:
    """Create a content-addressed immutable training snapshot and persist it."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    rows = eligible_rollout_rows(store, rollout_ids=rollout_ids)
    if not rows:
        raise ValueError("no eligible PCP-search training rows")
    contracts = {_policy_contract(row) for row in rows}
    if len(contracts) != 1:
        raise ValueError(f"mixed policy contracts require separate snapshots: {contracts}")
    versions = {int(row.get("training_data_schema_version") or 0) for row in rows}
    if len(versions) != 1:
        raise ValueError(f"mixed training artifact versions require separate snapshots: {versions}")
    # Validate and stream statistics one artifact at a time.  Do not materialize
    # frozen prefixes for an entire dataset in the preflight notebook.
    mean, std, n_transitions = _stream_action_statistics(store, rows)
    train_ids, val_ids = _partition(rows, seed=seed, train_fraction=train_fraction)
    payload = {
        "name": name, "rollout_ids": sorted(row["rollout_id"] for row in rows),
        "policy_contract": _policy_contract(rows[0]), "artifact_schema_version": next(iter(versions)),
        "seed": seed, "train_fraction": train_fraction, "gamma": gamma,
        "train_rollout_ids": train_ids, "val_rollout_ids": val_ids,
        "action_mean": mean.tolist(), "action_std": std.tolist(),
    }
    snapshot_id = "pcpcds-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return DatasetSnapshot(
        snapshot_id=snapshot_id, rollout_ids=tuple(payload["rollout_ids"]),
        train_rollout_ids=tuple(train_ids), val_rollout_ids=tuple(val_ids),
        policy_repo_id=payload["policy_contract"][0], policy_revision=payload["policy_contract"][1],
        artifact_schema_version=next(iter(versions)), action_mean=tuple(float(x) for x in mean),
        action_std=tuple(float(x) for x in std), provenance={**payload, "n_transitions": n_transitions} )
