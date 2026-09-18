"""Demonstration pretraining for the compact SmolVLA Q10 critic.

The official LIBERO demonstrations are successful trajectories.  We cache one
frozen SmolVLA prefix every ten actions and train with the same EMA Bellman
target used by Q-Planning and by the v3 counterfactual-branch artifacts:

    r[t:t+10] + gamma**10 * Q_target(s[t+10], a[t+10:t+20]).

Only compact prefix embeddings, physical state, and normalized action windows
are persisted.  The policy is never updated.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .config import SMOLVLA_REPO_ID
from .diversity import DIVERSITY_DATASET_REPO, _task_key
from .models import load_smolvla
from .qplanning_critic.config import QPlanningModelConfig, QPlanningTrainConfig
from .qplanning_critic.data import QPlanningCacheIndex, QPlanningWindowDataset
from .qplanning_critic.model import QPlanningCritic, pool_prefix_tokens
from .qplanning_critic.train import train_qplanning_critic


SMOLVLA_DEMO_DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
SMOLVLA_DEMO_SCHEMA_VERSION = 1
SMOLVLA_DEMO_SNAPSHOT_ID = "smolvla-libero-demo-q10-v1"
SMOLVLA_DEMO_HORIZON = 10
SMOLVLA_DEMO_PREFIX_TOKENS = 128
SMOLVLA_DEMO_EPISODES_PER_TASK = 10
SMOLVLA_DEMO_TRAIN_EPISODES_PER_TASK = 8
SMOLVLA_DEMO_EXPECTED_TASKS = 40


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temporary, path)


def _manifest_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def _episode_rows(metadata) -> list[dict]:
    return [dict(metadata.episodes[index]) for index in range(len(metadata.episodes))]


def select_demo_episodes(episode_rows: Iterable[dict], *, seed: int = 42,
                         episodes_per_task: int = SMOLVLA_DEMO_EPISODES_PER_TASK,
                         train_episodes_per_task: int = SMOLVLA_DEMO_TRAIN_EPISODES_PER_TASK
                         ) -> dict:
    """Choose an equal deterministic train/validation count from every task."""
    if not 0 < train_episodes_per_task < episodes_per_task:
        raise ValueError("train_episodes_per_task must lie inside episodes_per_task")
    by_task = defaultdict(list)
    rows_by_episode = {}
    for raw in episode_rows:
        row = dict(raw)
        episode = int(row["episode_index"])
        task = _task_key(row)
        by_task[task].append(episode)
        rows_by_episode[episode] = row
    if len(by_task) != SMOLVLA_DEMO_EXPECTED_TASKS:
        raise ValueError(
            f"expected {SMOLVLA_DEMO_EXPECTED_TASKS} LIBERO tasks, found {len(by_task)}")
    selected = []
    for task, episodes in sorted(by_task.items()):
        ordered = sorted(episodes, key=lambda episode: hashlib.sha256(
            f"{seed}|{task}|{episode}".encode()).hexdigest())
        if len(ordered) < episodes_per_task:
            raise ValueError(
                f"task {task!r} has {len(ordered)} demonstrations; need {episodes_per_task}")
        for rank, episode in enumerate(ordered[:episodes_per_task]):
            selected.append({
                "episode_index": int(episode), "task": task,
                "split": "train" if rank < train_episodes_per_task else "validation",
                "length": int(rows_by_episode[episode].get("length", 0)),
            })
    contract = {
        "schema_version": SMOLVLA_DEMO_SCHEMA_VERSION,
        "dataset_repo_id": DIVERSITY_DATASET_REPO,
        "dataset_revision": SMOLVLA_DEMO_DATASET_REVISION,
        "seed": int(seed), "episodes_per_task": int(episodes_per_task),
        "train_episodes_per_task": int(train_episodes_per_task),
        "episodes": selected,
    }
    contract["manifest_digest"] = _manifest_digest(contract)
    return contract


def _axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert LIBERO's xyz axis-angle representation to xyzw quaternion."""
    value = np.asarray(axis_angle, np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], np.float32)
    xyz = value / angle * math.sin(angle / 2.0)
    return np.asarray([*xyz, math.cos(angle / 2.0)], np.float32)


def _raw_robot_from_demo_state(state: np.ndarray) -> np.ndarray:
    """Map demonstration xyz+axis-angle+gripper(8D) to rollout xyz+quat+gripper(9D)."""
    state = np.asarray(state, np.float32).reshape(-1)
    if len(state) != 8:
        raise ValueError(f"expected 8D LIBERO observation.state, got {state.shape}")
    return np.concatenate([
        state[:3], _axis_angle_to_quaternion(state[3:6]), state[6:8]
    ]).astype(np.float32)


def _pad_action_window(actions: np.ndarray, start: int, horizon: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, np.float32)
    width = min(horizon, len(actions) - start)
    if width <= 0:
        raise ValueError("action window begins after the demonstration")
    # Repeat a real action through preprocessing, then zero invalid normalized rows.
    result = np.repeat(actions[min(start + width - 1, len(actions) - 1)][None], horizon, 0)
    result[:width] = actions[start:start + width]
    valid = np.zeros(horizon, bool); valid[:width] = True
    return result, valid


def _demo_windows(*, prefixes: np.ndarray, pads: np.ndarray,
                  raw_robots: np.ndarray, proprios: np.ndarray,
                  normalized_actions: np.ndarray, action_valid: np.ndarray,
                  starts: np.ndarray, episode_length: int,
                  gamma: float = .99) -> dict[str, np.ndarray]:
    """Create exact Q10 windows for one successful demonstration."""
    starts = np.asarray(starts, np.int32)
    if not (len(prefixes) == len(pads) == len(raw_robots) == len(proprios)
            == len(normalized_actions) == len(action_valid) == len(starts)):
        raise ValueError("demonstration boundary arrays do not align")
    records = []
    for index, start_value in enumerate(starts):
        start = int(start_value)
        valid = np.asarray(action_valid[index], bool)
        width = int(valid.sum())
        terminal = start + width >= int(episode_length)
        may_bootstrap = width == SMOLVLA_DEMO_HORIZON and not terminal and index + 1 < len(starts)
        next_index = index + 1 if may_bootstrap else index
        reward = np.float32(gamma ** (width - 1) if terminal else 0.0)
        records.append({
            "prefix": np.asarray(prefixes[index], np.float16),
            "pad": np.asarray(pads[index], bool),
            "robot": np.asarray(raw_robots[index], np.float32),
            "proprio": np.asarray(proprios[index], np.float32),
            "action": np.asarray(normalized_actions[index], np.float32),
            "action_valid": valid,
            "next_prefix": np.asarray(prefixes[next_index], np.float16),
            "next_pad": np.asarray(pads[next_index], bool),
            "next_robot": np.asarray(raw_robots[next_index], np.float32),
            "next_proprio": np.asarray(proprios[next_index], np.float32),
            "next_action": np.asarray(normalized_actions[next_index], np.float32),
            "next_action_valid": np.asarray(action_valid[next_index], bool),
            "reward": reward,
            "discount": np.float32(gamma ** SMOLVLA_DEMO_HORIZON if may_bootstrap else 0.0),
            "mc_return": np.float32(gamma ** (int(episode_length) - start - 1)),
            "success": np.bool_(True),
            "start_step": np.int32(start),
        })
    return {key: np.stack([row[key] for row in records]) for key in records[0]}


def _stack_samples(samples: list[dict], action_windows: np.ndarray,
                   tasks: list[str]) -> dict:
    batch = {"task": list(tasks), "action": torch.from_numpy(action_windows).float()}
    keys = ("observation.images.image", "observation.images.image2", "observation.state")
    for key in keys:
        values = [torch.as_tensor(sample[key]) for sample in samples]
        batch[key] = torch.stack(values)
    return batch


@torch.no_grad()
def _encode_episode(dataset, local_indices: list[int], actions: np.ndarray,
                    task: str, policy, preprocess, device, *, batch_size: int,
                    gamma: float) -> dict[str, np.ndarray]:
    frame_rows = [dataset.get_raw_item(index) for index in local_indices]
    frame_indices = np.asarray([int(row["frame_index"]) for row in frame_rows], np.int32)
    order = np.argsort(frame_indices)
    local_indices = [local_indices[int(i)] for i in order]
    actions = np.asarray(actions, np.float32)[order]
    frame_indices = frame_indices[order]
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"episode frame indices are not contiguous: {frame_indices[:5]}")
    starts = np.arange(0, len(actions), SMOLVLA_DEMO_HORIZON, dtype=np.int32)
    action_windows, validity = zip(*[
        _pad_action_window(actions, int(start), SMOLVLA_DEMO_HORIZON) for start in starts
    ])
    action_windows = np.stack(action_windows)
    validity = np.stack(validity)
    prefixes, pads, robots, proprios, normalized = [], [], [], [], []
    from lerobot.utils.constants import (
        ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE)
    for offset in range(0, len(starts), batch_size):
        sl = slice(offset, offset + batch_size)
        positions = starts[sl]
        samples = [dataset[local_indices[int(position)]] for position in positions]
        raw_states = np.stack([
            np.asarray(sample["observation.state"], np.float32) for sample in samples])
        batch = _stack_samples(samples, action_windows[sl], [task] * len(samples))
        prepared = preprocess(batch)
        images, image_masks = policy.prepare_images(prepared)
        padded_state = policy.prepare_state(prepared)
        embedded, valid, _ = policy.model.embed_prefix(
            images, image_masks, prepared[OBS_LANGUAGE_TOKENS],
            prepared[OBS_LANGUAGE_ATTENTION_MASK], state=padded_state)
        embedded, valid = pool_prefix_tokens(
            embedded, valid.bool(), SMOLVLA_DEMO_PREFIX_TOKENS)
        normalized_action = prepared[ACTION].detach().float().cpu().numpy()
        local_valid = validity[sl]
        normalized_action[~local_valid] = 0.0
        prefixes.extend(embedded.detach().to("cpu", torch.float16).numpy())
        pads.extend(valid.detach().cpu().numpy())
        robots.extend(_raw_robot_from_demo_state(state) for state in raw_states)
        proprios.extend(prepared[OBS_STATE].detach().float().cpu().numpy())
        normalized.extend(normalized_action)
    return _demo_windows(
        prefixes=np.asarray(prefixes), pads=np.asarray(pads),
        raw_robots=np.asarray(robots), proprios=np.asarray(proprios),
        normalized_actions=np.asarray(normalized), action_valid=validity,
        starts=starts, episode_length=len(actions), gamma=gamma)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def prepare_smolvla_demo_q10_cache(*, cache_root: str | Path,
                                   source_root: str | Path | None = None,
                                   seed: int = 42,
                                   episodes_per_task: int = SMOLVLA_DEMO_EPISODES_PER_TASK,
                                   train_episodes_per_task: int = SMOLVLA_DEMO_TRAIN_EPISODES_PER_TASK,
                                   encode_batch_size: int = 16,
                                   gamma: float = .99,
                                   device=None) -> tuple[QPlanningCacheIndex, tuple[str, ...], tuple[str, ...]]:
    """Download selected demonstrations and build a resumable compact Drive cache."""
    if encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(
        DIVERSITY_DATASET_REPO, revision=SMOLVLA_DEMO_DATASET_REVISION)
    manifest = select_demo_episodes(
        _episode_rows(metadata), seed=seed, episodes_per_task=episodes_per_task,
        train_episodes_per_task=train_episodes_per_task)
    root = Path(cache_root).expanduser() / manifest["manifest_digest"]
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "selection.json", manifest)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, preprocess, _ = load_smolvla(device=device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    entries = {}
    for row in manifest["episodes"]:
        path = root / f"episode_{row['episode_index']:06d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                entries[row["episode_index"]] = {
                    "rollout_id": f"demo-{row['episode_index']}", "path": path.name,
                    "n_windows": int(len(archive["reward"])), "split": row["split"],
                    "task": row["task"], "episode_index": row["episode_index"],
                }
    grouped = defaultdict(list)
    for row in manifest["episodes"]:
        if row["episode_index"] not in entries:
            grouped[row["task"]].append(row)
    done = len(entries)
    total = len(manifest["episodes"])
    print(f"[smolvla-demo-q10] compact cache {done}/{total} episodes ready", flush=True)
    for task, task_rows in sorted(grouped.items()):
        episode_ids = [int(row["episode_index"]) for row in task_rows]
        task_root = None
        if source_root is not None:
            task_root = Path(source_root).expanduser() / manifest["manifest_digest"]
        dataset = LeRobotDataset(
            DIVERSITY_DATASET_REPO, root=task_root, episodes=episode_ids,
            revision=SMOLVLA_DEMO_DATASET_REVISION)
        columns = dataset.hf_dataset.select_columns(["episode_index", "action"])
        episode_values = np.asarray(columns["episode_index"], np.int64)
        action_values = np.asarray(columns["action"], np.float32)
        local_by_episode = {
            episode: np.flatnonzero(episode_values == episode).astype(int).tolist()
            for episode in episode_ids}
        for row in task_rows:
            episode = int(row["episode_index"])
            local = local_by_episode[episode]
            arrays = _encode_episode(
                dataset, local, action_values[local], task, policy, preprocess, device,
                batch_size=encode_batch_size, gamma=gamma)
            path = root / f"episode_{episode:06d}.npz"
            _write_npz(path, arrays)
            entries[episode] = {
                "rollout_id": f"demo-{episode}", "path": path.name,
                "n_windows": int(len(arrays["reward"])), "split": row["split"],
                "task": task, "episode_index": episode,
            }
            done += 1
            print(
                f"[smolvla-demo-q10] {done}/{total} episodes | {task} ep={episode} | "
                f"{len(arrays['reward'])} Q10 windows", flush=True)
        del dataset, columns, episode_values, action_values
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ordered = [entries[int(row["episode_index"])] for row in manifest["episodes"]]
    first_path = root / ordered[0]["path"]
    with np.load(first_path, allow_pickle=False) as first:
        prefix_dim = int(first["prefix"].shape[-1])
        robot_dim = int(first["robot"].shape[-1])
        proprio_dim = int(first["proprio"].shape[-1])
        action_dim = int(first["action"].shape[-1])
    action_sum = np.zeros(action_dim, np.float64)
    action_sumsq = np.zeros(action_dim, np.float64)
    action_count = 0
    for entry in ordered:
        if entry["split"] != "train":
            continue
        with np.load(root / entry["path"], allow_pickle=False) as archive:
            actions = np.asarray(archive["action"], np.float64)
            valid = np.asarray(archive["action_valid"], bool)
            values = actions[valid]
            action_sum += values.sum(0); action_sumsq += np.square(values).sum(0)
            action_count += len(values)
    mean = action_sum / action_count
    variance = np.maximum(action_sumsq / action_count - np.square(mean), 1e-12)
    train_ids = tuple(row["rollout_id"] for row in ordered if row["split"] == "train")
    val_ids = tuple(row["rollout_id"] for row in ordered if row["split"] == "validation")
    cache = QPlanningCacheIndex(
        snapshot_id=SMOLVLA_DEMO_SNAPSHOT_ID, horizon=SMOLVLA_DEMO_HORIZON,
        cache_dir=str(root), rollouts=tuple(ordered), prefix_dim=prefix_dim,
        robot_dim=robot_dim, proprio_dim=proprio_dim, action_dim=action_dim,
        action_mean=tuple(float(x) for x in mean),
        action_std=tuple(float(x) for x in np.sqrt(variance)),
        n_windows=sum(row["n_windows"] for row in ordered),
        n_train_windows=sum(row["n_windows"] for row in ordered if row["split"] == "train"),
        n_val_windows=sum(row["n_windows"] for row in ordered if row["split"] == "validation"),
        n_bootstrap_windows=0, generated_executed_first10_mae=float("nan"),
        generated_executed_first10_max_abs=float("nan"), storage_mode="materialized",
        gamma=gamma)
    _atomic_json(root / "cache_index.json", {
        "manifest": manifest, "cache_digest": cache.digest,
        "entries": ordered, "dimensions": {
            "prefix": prefix_dim, "robot": robot_dim,
            "proprio": proprio_dim, "action": action_dim},
        "windows": {"train": cache.n_train_windows, "validation": cache.n_val_windows},
    })
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache, train_ids, val_ids


def run_smolvla_demo_q10_pretraining(*,
                                     cache_root: str | Path,
                                     output_root: str | Path,
                                     source_root: str | Path | None = None,
                                     updates: int = 4_000,
                                     micro_batch_size: int = 64,
                                     encode_batch_size: int = 16,
                                     seed: int = 42,
                                     resume: bool = True,
                                     device=None) -> dict:
    """Build the demo cache and train one resumable compact Q10 critic."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache, train_ids, val_ids = prepare_smolvla_demo_q10_cache(
        cache_root=cache_root, source_root=source_root, seed=seed,
        encode_batch_size=encode_batch_size, device=device)
    train = QPlanningWindowDataset(cache, train_ids, max_open_rollouts=8)
    validation = QPlanningWindowDataset(cache, val_ids, max_open_rollouts=8)
    model_config = QPlanningModelConfig(
        action_horizon=10, action_dim=cache.action_dim, width=256,
        n_layers=3, n_heads=8, ffn_width=1024, dropout=.20,
        prefix_pool_tokens=SMOLVLA_DEMO_PREFIX_TOKENS)
    model = QPlanningCritic(
        prefix_dim=cache.prefix_dim, robot_dim=cache.robot_dim,
        proprio_dim=cache.proprio_dim, config=model_config)
    model.set_action_statistics(cache.action_mean, cache.action_std)
    train_config = QPlanningTrainConfig(
        seed=seed, gamma=cache.gamma, learning_rate=3e-4, weight_decay=1e-4,
        effective_batch_size=64, micro_batch_size=micro_batch_size,
        updates=updates, warmup_updates=min(250, updates // 10),
        print_interval=100, eval_interval=500, checkpoint_interval=500,
        target_rate=.005, grad_clip=1.0, use_bf16=True)
    output = Path(output_root).expanduser() / cache.digest / "demo_q10_pretrain"
    print("Training contract", flush=True)
    print(
        f"  official successful LIBERO demonstrations: "
        f"{len(train_ids)} train / {len(val_ids)} validation episodes", flush=True)
    print(
        f"  Q10 windows: {len(train)} train / {len(validation)} validation; "
        f"EMA Bellman target; frozen SmolVLA; pooled prefix=128", flush=True)
    print(
        f"  compact critic: width=256, layers=3; updates={updates}; "
        f"effective batch=64; microbatch={micro_batch_size}", flush=True)
    print(f"  checkpoint directory: {output}", flush=True)
    _, _, report = train_qplanning_critic(
        model, train, validation, device,
        snapshot_id=SMOLVLA_DEMO_SNAPSHOT_ID, cache_digest=cache.digest,
        source_policy={
            "repo_id": SMOLVLA_REPO_ID,
            "dataset_repo_id": DIVERSITY_DATASET_REPO,
            "dataset_revision": SMOLVLA_DEMO_DATASET_REVISION,
            "purpose": "successful-demonstration initialization before v3 branch TD training",
        },
        output_dir=output, config=train_config, resume=resume)
    report["note"] = (
        "Demo validation is success-only; failure AUC is intentionally undefined. "
        "Use Bellman/return MAE here and evaluate failure discrimination after tree training.")
    return report

