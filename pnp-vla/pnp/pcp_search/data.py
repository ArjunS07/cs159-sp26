"""Lossless Bellman/RL-token artifact construction and validation."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


TRAINING_ARTIFACT_SCHEMA_VERSION = 1
ACTION_HORIZON = 10


def _stack(values: Iterable[Any], *, name: str) -> np.ndarray:
    values = [np.asarray(value) for value in values]
    if not values:
        return np.empty((0,), dtype=np.float32)
    try:
        return np.stack(values)
    except ValueError as error:
        shapes = [value.shape for value in values]
        raise ValueError(f"cannot stack {name}; boundary shapes differ: {shapes}") from error


def _flatten_prefixes(prefixes: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def visit(path: str, values: list[Any]) -> None:
        first = values[0]
        if isinstance(first, dict):
            keys = set(first)
            if any(set(value) != keys for value in values):
                raise ValueError(f"prefix dictionaries differ at {path}")
            for key in sorted(keys):
                visit(f"{path}/{key}", [value[key] for value in values])
            return
        if isinstance(first, list):
            width = len(first)
            if any(len(value) != width for value in values):
                raise ValueError(f"prefix list widths differ at {path}")
            for index in range(width):
                visit(f"{path}/{index}", [value[index] for value in values])
            return
        arrays[path] = _stack(values, name=path)

    visit("prefix", prefixes)
    return arrays


def build_training_artifact(*, decisions: list[dict], prefixes: list[dict],
                            generated_chunks: np.ndarray,
                            normalized_actions: np.ndarray, env_actions: np.ndarray,
                            rewards: np.ndarray, terminated: np.ndarray,
                            truncated: np.ndarray, step_success: np.ndarray,
                            robot_states: np.ndarray, sim_states: np.ndarray,
                            chunk_start_steps: list[int], chunk_noise_seeds: list[int],
                            episode_seed: int, perturb_seed: int,
                            initial_state: np.ndarray) -> dict[str, np.ndarray]:
    """Materialize H=10 transitions and every decision-boundary prefix.

    ``generated_chunks`` contains one action at every decision boundary, including the final
    terminal/truncated boundary.  The last action is never executed; it is retained as ``A_next``
    and masked out by terminal/truncation flags.
    """
    n_transitions = len(chunk_start_steps)
    if len(decisions) != n_transitions + 1 or len(prefixes) != n_transitions + 1:
        raise ValueError(
            "training artifact needs one more decision/prefix than executed chunks: "
            f"decisions={len(decisions)}, prefixes={len(prefixes)}, chunks={n_transitions}")
    generated_chunks = np.asarray(generated_chunks, dtype=np.float32)
    if generated_chunks.ndim != 3 or len(generated_chunks) != n_transitions + 1:
        raise ValueError(
            f"expected generated chunks (C+1,H,A), got {generated_chunks.shape}")
    if generated_chunks.shape[1] < ACTION_HORIZON:
        raise ValueError("generated action chunk is shorter than the 10-action Bellman horizon")

    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
    env_actions = np.asarray(env_actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    step_success = np.asarray(step_success, dtype=bool)
    n_steps = len(env_actions)
    for name, value in (("normalized_actions", normalized_actions), ("rewards", rewards),
                        ("terminated", terminated), ("truncated", truncated),
                        ("step_success", step_success)):
        if len(value) != n_steps:
            raise ValueError(f"{name} has {len(value)} rows; expected {n_steps}")
    if len(robot_states) != n_steps + 1 or len(sim_states) != n_steps + 1:
        raise ValueError("step state arrays must have T+1 rows")

    transition_rewards = np.zeros((n_transitions, ACTION_HORIZON), np.float32)
    transition_terminated = np.zeros((n_transitions, ACTION_HORIZON), bool)
    transition_truncated = np.zeros((n_transitions, ACTION_HORIZON), bool)
    transition_success = np.zeros((n_transitions, ACTION_HORIZON), bool)
    transition_valid = np.zeros((n_transitions, ACTION_HORIZON), bool)
    transition_normalized = np.zeros(
        (n_transitions, ACTION_HORIZON, normalized_actions.shape[-1]), np.float32)
    transition_env = np.zeros(
        (n_transitions, ACTION_HORIZON, env_actions.shape[-1]), np.float32)
    next_boundary_steps = []
    for chunk_idx, start in enumerate(chunk_start_steps):
        end = (chunk_start_steps[chunk_idx + 1]
               if chunk_idx + 1 < n_transitions else n_steps)
        width = end - start
        if not 0 < width <= ACTION_HORIZON:
            raise ValueError(f"chunk {chunk_idx} covers invalid step interval [{start}, {end})")
        transition_rewards[chunk_idx, :width] = rewards[start:end]
        transition_terminated[chunk_idx, :width] = terminated[start:end]
        transition_truncated[chunk_idx, :width] = truncated[start:end]
        transition_success[chunk_idx, :width] = step_success[start:end]
        transition_valid[chunk_idx, :width] = True
        transition_normalized[chunk_idx, :width] = normalized_actions[start:end]
        transition_env[chunk_idx, :width] = env_actions[start:end]
        next_boundary_steps.append(end)

    arrays = {
        "schema_version": np.asarray(TRAINING_ARTIFACT_SCHEMA_VERSION, np.int16),
        "action_horizon": np.asarray(ACTION_HORIZON, np.int16),
        "episode_seed": np.asarray(episode_seed, np.int64),
        "perturb_seed": np.asarray(perturb_seed, np.int64),
        "initial_state": np.asarray(initial_state).copy(),
        "chunk_start_steps": np.asarray(chunk_start_steps, np.int32),
        "next_boundary_steps": np.asarray(next_boundary_steps, np.int32),
        "chunk_noise_seeds": np.asarray(chunk_noise_seeds, np.int64),
        "actions_normalized": normalized_actions,
        "actions_env": env_actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "step_success": step_success,
        "robot_state_t_plus_1": np.asarray(robot_states, dtype=np.float32),
        "sim_state_t_plus_1": np.asarray(sim_states),
        "bellman/action": generated_chunks[:-1],
        "bellman/next_action": generated_chunks[1:],
        "bellman/executed_normalized": transition_normalized,
        "bellman/executed_env": transition_env,
        "bellman/rewards": transition_rewards,
        "bellman/terminated": transition_terminated,
        "bellman/truncated": transition_truncated,
        "bellman/success": transition_success,
        "bellman/validity_mask": transition_valid,
        "boundary/step": np.asarray([row["step"] for row in decisions], np.int32),
        "boundary/raw_agentview": _stack(
            [row["raw_agentview"] for row in decisions], name="raw_agentview"),
        "boundary/raw_wrist": _stack(
            [row["raw_wrist"] for row in decisions], name="raw_wrist"),
        "boundary/raw_robot_state": _stack(
            [row["raw_robot_state"] for row in decisions], name="raw_robot_state"),
        "boundary/policy_proprio": _stack(
            [row["policy_proprio"] for row in decisions], name="policy_proprio"),
        "boundary/sim_state": _stack(
            [row["sim_state"] for row in decisions], name="boundary_sim_state"),
        "boundary/instruction": np.asarray(
            [row["instruction"] for row in decisions], dtype=np.str_),
        **_flatten_prefixes(prefixes),
    }
    validate_training_artifact(arrays)
    return arrays


def validate_training_artifact(arrays: dict[str, np.ndarray]) -> dict[str, int | bool]:
    required = {
        "bellman/action", "bellman/next_action", "bellman/rewards",
        "bellman/terminated", "bellman/truncated", "bellman/validity_mask",
        "boundary/raw_agentview", "boundary/raw_wrist", "boundary/raw_robot_state",
        "boundary/policy_proprio", "robot_state_t_plus_1", "sim_state_t_plus_1",
        "prefix/prefix_embeddings", "prefix/token_ids", "prefix/token_masks",
        "prefix/prefix_pad_masks", "prefix/prefix_attention_masks",
        "prefix/prefix_attention_2d_masks", "prefix/prefix_position_ids",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"training artifact is missing {missing}")
    for prefix in ("prefix/processed_images/", "prefix/processed_image_masks/"):
        if not any(key.startswith(prefix) for key in arrays):
            raise ValueError(f"training artifact has no {prefix.rstrip('/')} tensors")
    n = len(arrays["bellman/action"])
    boundaries = n + 1
    if len(arrays["bellman/next_action"]) != n:
        raise ValueError("Bellman action and next_action counts differ")
    if arrays["bellman/rewards"].shape != (n, ACTION_HORIZON):
        raise ValueError("Bellman rewards have the wrong H=10 shape")
    for key in required:
        if key.startswith("boundary/") or key.startswith("prefix/"):
            if len(arrays[key]) != boundaries:
                raise ValueError(f"{key} has {len(arrays[key])}, expected {boundaries}")
    valid = arrays["bellman/validity_mask"]
    if valid.dtype != bool or not valid.any(axis=1).all():
        raise ValueError("each Bellman transition needs at least one valid step")
    if not all(np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all()
               for key in ("bellman/action", "bellman/next_action", "bellman/rewards",
                           "boundary/raw_robot_state", "boundary/policy_proprio")):
        raise ValueError("training artifact contains non-finite numeric values")
    return {
        "training_ready": True,
        "n_transitions": n,
        "n_steps": len(arrays["actions_env"]),
        "n_boundaries": boundaries,
    }
