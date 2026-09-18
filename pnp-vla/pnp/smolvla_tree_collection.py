"""Depth-1 SmolVLA tree collection from the immutable notebook-87 source cohort.

Each tree contains the exact stored source continuation plus eight counterfactual branches:
four rerun the same P&P policy with fresh initial flow noise, and four hold the source initial
noise fixed while changing only the P&P perturbation stream. The stored source is never decoded
again: SmolVLA's P&P path is measurably GPU-batch-shape-sensitive, so all eight counterfactuals
are generated in one fixed eight-lane batch. Roots are fixed before collection:
65% choose the largest pair-weighted U10 boundary and 35% choose a deterministic uniform
boundary, balanced across suite and source outcome. Version 3 additionally stores every
counterfactual's sequential Q10 current/next-boundary tensors, so training can use the same
EMA-bootstrapped Bellman objective as Q-Planning rather than a root-level Monte Carlo label.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import io
import json
import os
import subprocess
import time

import numpy as np
import torch

from .config import RolloutConfig, SMOLVLA_REPO_ID
from .pcp_critic.resumable_snapshot import _download_with_retry
from .pnp import PnPRecorder, _pnp_gen
from .qplanning_fork_pilot import (
    _SOURCE_FIDELITY_ARRAYS,
    _load_selected_training_arrays,
    _source_boundary,
    _source_policy_observation,
    _trajectory_actions_from_payload,
)
from .libero_env import obs_to_policy, refresh_camera_observation, set_camera_observables
from .pcp_search.data import ACTION_HORIZON
from .rollout import (
    _draw_chunk_noise,
    _raw_robot_state,
    _sim_state,
    _stack_policy_batches,
    _training_decision,
    chunk_noise_seed,
)
from .sampler import _temp_strategy
from .smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
)
from .smolvla_tree_source_experiment import (
    SMOLVLA_TREE_PRIORITY_FRACTION,
    SMOLVLA_TREE_SOURCE_EXPERIMENT,
    SMOLVLA_TREE_SOURCE_IDENTITIES,
    prepare_smolvla_tree_source_episodes,
)
from .store import SupabaseStore
from .tap import BatchedRolloutTap, PrefixCaptureTap, RolloutTap
from .verifier.collection import (
    _content_digest,
    _reset_and_replay_actions,
    _unwrap_sim,
    candidate_group_id,
    postprocess_chunk,
)


SMOLVLA_TREE_COLLECTION_VERSION = 3
# v3 is deliberately disjoint from the old root-only v2 trees.  Every
# counterfactual now persists an executed branch artifact with the current and
# next decision-boundary inputs required by an EMA Bellman target.
SMOLVLA_TREE_LEGACY_EXPERIMENT = "smolvla-libero-depth1-hybrid-trees-v2-egl"
SMOLVLA_TREE_EXPERIMENT = "smolvla-libero-depth1-hybrid-trees-v3-bellman"
SMOLVLA_TREE_MANIFEST_PATH = (
    "smolvla_trees/manifests/idx10_29_depth1_hybrid_v3_bellman.json")
SMOLVLA_TREE_COUNT = 800
SMOLVLA_TREE_SHARDS = 2
SMOLVLA_TREE_CANDIDATES = 9
SMOLVLA_TREE_FRESH_CANDIDATES = 4
SMOLVLA_TREE_PERTURB_CANDIDATES = 4
SMOLVLA_TREE_ACTIONS = 10
SMOLVLA_TREE_INTEGRATION_STEPS = 10
SMOLVLA_TREE_MIN_BOUNDARIES = 3
SMOLVLA_TREE_PRINT_EVERY = 5
SMOLVLA_BRANCH_PREFIX_TOKENS = 128
_U_TIME_KEY = __import__("re").compile(r"^c(?P<chunk>\d+)_s(?P<step>\d+)_u_time$")
_SMOLVLA_SOURCE_ARRAYS = tuple(_SOURCE_FIDELITY_ARRAYS) + (
    "perturb_seed", "boundary/instruction",
    "prefix/prefix_embeddings", "prefix/prefix_pad_masks")


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()[:20]


def _rank(namespace: str, *values) -> str:
    return hashlib.sha256("|".join(map(str, (namespace, *values))).encode()).hexdigest()


def _seed(namespace: str, *values) -> int:
    return int(_rank(namespace, *values)[:16], 16) & ((1 << 63) - 1)


def _source_rows(store) -> list[dict]:
    rows = store.fetch_all(
        "rollouts",
        "rollout_id,benchmark,suite,task_idx,episode_idx,init_state_hash,success,n_steps,"
        "n_chunks,max_steps,episode_seed,config_hash,ahats_path,trajectory_path,"
        "training_data_path,status",
        configure=lambda query: query.eq(
            "experiment", SMOLVLA_TREE_SOURCE_EXPERIMENT).eq(
            "method", "smolvla_pnp_steps123_k311").eq("status", "completed"),
        order_by=("rollout_id",),
    )
    identities = {
        (row["suite"], int(row["task_idx"]), int(row["episode_idx"]), row["init_state_hash"])
        for row in rows
    }
    if len(rows) != SMOLVLA_TREE_SOURCE_IDENTITIES or len(identities) != len(rows):
        raise ValueError(
            f"expected {SMOLVLA_TREE_SOURCE_IDENTITIES} unique notebook-87 v2 sources; "
            f"found {len(rows)} rows/{len(identities)} identities")
    if len({row["config_hash"] for row in rows}) != 1:
        raise ValueError("source collection contains mixed behavior configurations")
    required = ("ahats_path", "trajectory_path", "training_data_path")
    missing = [row["rollout_id"] for row in rows
               if any(not row.get(field) for field in required)]
    if missing:
        raise ValueError(f"source rows lack required artifacts: {missing[:3]}")
    return rows


def weighted_u10_profile(payload: bytes) -> tuple[float, ...]:
    """Decode the notebook-86 (3,1,1)-pair-weighted U10 score per source chunk."""
    by_chunk: dict[int, dict[int, float]] = defaultdict(dict)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        for key in archive.files:
            match = _U_TIME_KEY.match(key)
            if match is None:
                continue
            profile = np.asarray(archive[key], np.float32).reshape(-1)
            if len(profile) < 10:
                raise ValueError(f"{key} has fewer than 10 action positions")
            by_chunk[int(match.group("chunk"))][int(match.group("step"))] = float(
                profile[:10].mean())
    if not by_chunk:
        raise ValueError("source uncertainty artifact contains no U-time profiles")
    expected_chunks = list(range(max(by_chunk) + 1))
    if sorted(by_chunk) != expected_chunks:
        raise ValueError("source uncertainty chunks are not contiguous")
    weights = dict(zip(SMOLVLA_SCHEDULE_STEPS, SMOLVLA_SCHEDULE_K_BY_STEP))
    result = []
    for chunk in expected_chunks:
        if set(by_chunk[chunk]) != set(weights):
            raise ValueError(
                f"chunk {chunk} has Euler steps {sorted(by_chunk[chunk])}, "
                f"expected {sorted(weights)}")
        result.append(sum(weights[step] * by_chunk[chunk][step] for step in weights)
                      / sum(weights.values()))
    if not np.isfinite(result).all():
        raise ValueError("source U10 profile contains non-finite values")
    return tuple(map(float, result))


def _balanced_priority_ids(rows: list[dict], count: int) -> set[str]:
    buckets: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["suite"]), bool(row["success"]))].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: _rank("priority-episode", row["rollout_id"]))
    selected = []
    keys = sorted(buckets)
    while len(selected) < count:
        advanced = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(str(buckets[key].pop(0)["rollout_id"]))
                advanced = True
        if not advanced:
            break
    if len(selected) != count:
        raise ValueError(f"could select only {len(selected)}/{count} priority episodes")
    return set(selected)


def _eligible_roots(profile: tuple[float, ...]) -> tuple[list[int], bool]:
    # Prefer a mid-trajectory root with itself plus two complete source boundaries remaining.
    preferred = list(range(1, len(profile) - SMOLVLA_TREE_MIN_BOUNDARIES + 1))
    if preferred:
        return preferred, False
    # Very short successful episodes still contribute one valid, explicitly flagged root.
    fallback = list(range(max(1, len(profile) - 1)))
    return fallback, True


def build_smolvla_tree_manifest(store=None) -> dict:
    store = store or SupabaseStore()
    rows = _source_rows(store)
    priority_count = int(round(SMOLVLA_TREE_PRIORITY_FRACTION * len(rows)))
    priority_ids = _balanced_priority_ids(rows, priority_count)
    items = []
    for index, row in enumerate(rows, 1):
        profile = weighted_u10_profile(
            _download_with_retry(store, str(row["ahats_path"])))
        if len(profile) != int(row["n_chunks"]):
            raise ValueError(
                f"source {row['rollout_id']} has {len(profile)} U10 chunks but "
                f"n_chunks={row['n_chunks']}")
        eligible, fallback = _eligible_roots(profile)
        if not eligible:
            raise ValueError(f"source {row['rollout_id']} has no branchable boundary")
        strategy = "u10_priority" if str(row["rollout_id"]) in priority_ids else "uniform"
        if strategy == "u10_priority":
            root = min(eligible, key=lambda chunk: (
                -profile[chunk], _rank("u10-tie", row["rollout_id"], chunk)))
        else:
            root = eligible[int(_rank("uniform-root", row["rollout_id"]), 16) % len(eligible)]
        items.append({
            "source_rollout_id": str(row["rollout_id"]),
            "suite": str(row["suite"]), "task_idx": int(row["task_idx"]),
            "episode_idx": int(row["episode_idx"]),
            "init_state_hash": str(row["init_state_hash"]),
            "source_success": bool(row["success"]), "source_n_steps": int(row["n_steps"]),
            "source_n_chunks": int(row["n_chunks"]), "source_episode_seed": int(row["episode_seed"]),
            "source_max_steps": int(row["max_steps"]),
            "selection_strategy": strategy, "chunk_idx": int(root),
            "source_root_u10": float(profile[root]),
            "source_u10_profile": list(profile),
            "short_episode_root_fallback": bool(fallback),
            "selection_tiebreak": _rank("tree-order", row["rollout_id"]),
        })
        if index % 25 == 0 or index == len(rows):
            print(f"[smolvla-tree] uncertainty manifests: {index}/{len(rows)}", flush=True)
    items.sort(key=lambda item: item["selection_tiebreak"])
    for ordinal, item in enumerate(items):
        item["ordinal"] = ordinal
        item["shard_index"] = ordinal % SMOLVLA_TREE_SHARDS
    payload = {
        "version": SMOLVLA_TREE_COLLECTION_VERSION,
        "source_experiment": SMOLVLA_TREE_SOURCE_EXPERIMENT,
        "experiment": SMOLVLA_TREE_EXPERIMENT,
        "trees": SMOLVLA_TREE_COUNT, "shards": SMOLVLA_TREE_SHARDS,
        "candidate_count": SMOLVLA_TREE_CANDIDATES,
        "candidate_families": {"stored_source": 1, "fresh_initial_noise": 4,
                               "fixed_initial_noise_new_pnp_perturbation": 4},
        "priority_fraction": SMOLVLA_TREE_PRIORITY_FRACTION,
        "uniform_fraction": 1.0 - SMOLVLA_TREE_PRIORITY_FRACTION,
        "root_uncertainty": "(3,1,1)-pair-weighted U10",
        "minimum_preferred_boundaries_remaining": SMOLVLA_TREE_MIN_BOUNDARIES,
        "integration_steps": SMOLVLA_TREE_INTEGRATION_STEPS,
        "executed_actions": SMOLVLA_TREE_ACTIONS,
        "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
        "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
        "branch_artifact": "sequential_q10_bellman_v1",
        "branch_prefix_pool_tokens": SMOLVLA_BRANCH_PREFIX_TOKENS,
        "critic_target": "EMA Bellman r_0:9 + gamma^10 Qbar(next)",
        "items": items,
    }
    if len(items) != SMOLVLA_TREE_COUNT:
        raise AssertionError(f"manifest contains {len(items)} trees")
    document = {"manifest_hash": _digest(payload), "payload": payload}
    store._upload(SMOLVLA_TREE_MANIFEST_PATH, _canonical_json(document))
    print({
        "manifest_path": SMOLVLA_TREE_MANIFEST_PATH,
        "manifest_hash": document["manifest_hash"],
        "trees": len(items),
        "u10_priority": sum(item["selection_strategy"] == "u10_priority" for item in items),
        "uniform": sum(item["selection_strategy"] == "uniform" for item in items),
        "short_episode_fallbacks": sum(item["short_episode_root_fallback"] for item in items),
    }, flush=True)
    return document


def load_or_build_smolvla_tree_manifest(store=None) -> dict:
    """Reuse the immutable manifest after the first worker creates it."""
    store = store or SupabaseStore()
    try:
        document = json.loads(_download_with_retry(store, SMOLVLA_TREE_MANIFEST_PATH))
    except Exception as error:
        message = str(error).lower()
        if "404" not in message and "not found" not in message and "does not exist" not in message:
            raise
        return build_smolvla_tree_manifest(store)
    payload = document.get("payload")
    if (not isinstance(payload, dict)
            or int(payload.get("version", -1)) != SMOLVLA_TREE_COLLECTION_VERSION
            or payload.get("source_experiment") != SMOLVLA_TREE_SOURCE_EXPERIMENT
            or payload.get("experiment") != SMOLVLA_TREE_EXPERIMENT
            or _digest(payload) != document.get("manifest_hash")):
        raise ValueError("persisted SmolVLA tree manifest failed validation")
    if len(payload.get("items", ())) != SMOLVLA_TREE_COUNT:
        raise ValueError("persisted SmolVLA tree manifest has the wrong tree count")
    print({
        "manifest_path": SMOLVLA_TREE_MANIFEST_PATH,
        "manifest_hash": document["manifest_hash"],
        "trees": len(payload["items"]),
        "status": "loaded existing immutable manifest",
    }, flush=True)
    return document


def _load_source_bundle(store, row: dict) -> dict:
    actions = _trajectory_actions_from_payload(
        _download_with_retry(store, str(row["trajectory_path"])))
    arrays = _load_selected_training_arrays(
        store, str(row["training_data_path"]), names=_SMOLVLA_SOURCE_ARRAYS)
    if not np.array_equal(actions, np.asarray(arrays["actions_env"], np.float32)):
        raise AssertionError(f"source {row['rollout_id']} trajectory/training actions differ")
    return {"row": row, "arrays": arrays, "actions": actions}


def _candidate_u10(recorder: PnPRecorder) -> float:
    weights = dict(zip(SMOLVLA_SCHEDULE_STEPS, SMOLVLA_SCHEDULE_K_BY_STEP))
    values = {}
    for chunk in recorder.current_chunks():
        for record in chunk.get("steps", []):
            values[int(record["step"])] = float(
                np.asarray(record["u_time"], np.float32)[:10].mean())
    if set(values) != set(weights):
        raise RuntimeError(f"candidate P&P trace has steps {sorted(values)}")
    return float(sum(weights[step] * values[step] for step in weights) / sum(weights.values()))


def _pnp_config(*, save_features=False, save_training_data=False) -> RolloutConfig:
    return RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True, n_action_steps=SMOLVLA_TREE_ACTIONS,
        save_pcp_features=bool(save_features),
        save_generated_chunks=bool(save_training_data),
        save_training_data=bool(save_training_data),
    )


def _source_perturb_generator(policy, device, *, perturb_seed: int,
                              completed_chunks: int) -> torch.Generator:
    """Reconstruct notebook 87's lane-local P&P RNG at one source boundary."""
    generator = torch.Generator(device=torch.device(device))
    # training_data/perturb_seed is already the actual generator seed after the XOR mask.
    generator.manual_seed(int(perturb_seed))
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    draws_per_chunk = sum(map(int, SMOLVLA_SCHEDULE_K_BY_STEP))
    for _ in range(int(completed_chunks) * draws_per_chunk):
        torch.empty(shape, device=device).normal_(generator=generator)
    return generator


def _clone_generator(generator: torch.Generator, device) -> torch.Generator:
    clone = torch.Generator(device=torch.device(device))
    clone.set_state(generator.get_state())
    return clone


def _generate_alternatives(policy, batch, device, *, item: dict, source: dict):
    model = policy.model
    kinds = ([f"fresh_seed_{index}" for index in range(1, 5)]
             + [f"pnp_perturb_{index}" for index in range(1, 5)])
    fresh_initial_seeds = [
        _seed("fresh-initial", item["source_rollout_id"], item["chunk_idx"], index)
        for index in range(1, 5)
    ]
    pnp_perturb_seeds = [
        _seed("candidate-perturb", item["source_rollout_id"], item["chunk_idx"], kind)
        for kind in kinds[4:]
    ]
    source_noise_seed = int(source["noise_seed"])
    # Four lanes vary only initial flow noise; four hold the source initial noise fixed and vary
    # only the P&P perturbation stream. The exact source is read from its immutable artifact and
    # deliberately does not enter this batch: calibration showed that decoding it again can be
    # batch-shape-sensitive even with the exact stored input and RNG streams.
    initial_seeds = [*fresh_initial_seeds, *([source_noise_seed] * 4)]
    source_generator = _source_perturb_generator(
        policy, device, perturb_seed=int(np.asarray(source["perturb_seed"])),
        completed_chunks=int(item["chunk_idx"]))
    generators = [_clone_generator(source_generator, device) for _ in range(4)]
    for seed in pnp_perturb_seeds:
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(int(seed))
        generators.append(generator)

    noises = torch.cat([
        _draw_chunk_noise(policy, device, seed) for seed in initial_seeds], dim=0)
    batches = _stack_policy_batches([batch for _ in initial_seeds])
    recorders = [PnPRecorder() for _ in initial_seeds]
    for recorder in recorders:
        recorder.new_episode()
    config = _pnp_config(save_features=True)
    tap = BatchedRolloutTap(
        config, recorders, generators, device, model._pnp.action_dim)
    previous_steps, previous_position = model._pnp.num_steps, model._pnp.chunk_pos
    model._pnp.num_steps = SMOLVLA_TREE_INTEGRATION_STEPS
    # Notebook 87 executes ten actions per decision but conditions chunk position using the
    # generated 50-action width, as run_episode_batch does.
    estimated_chunks = max(1, round(
        item["source_max_steps"] / int(policy.config.chunk_size)))
    model._pnp.chunk_pos = [
        min(item["chunk_idx"] / estimated_chunks, 1.0) for _ in initial_seeds]
    try:
        with _temp_strategy(model, tap), torch.no_grad():
            chunks = policy.predict_action_chunk(batches, noise=noises)
    finally:
        model._pnp.num_steps = previous_steps
        model._pnp.chunk_pos = previous_position
    arrays = chunks.detach().cpu().numpy().astype(np.float32)
    if not tap.pcp_chunks or not tap.pcp_chunks[0]:
        raise RuntimeError("candidate generation captured no SmolVLA observation embedding")
    obs_enc = np.asarray(tap.pcp_chunks[0][0]["obs_enc"], np.float32)
    metadata = {
        kind: {
            "candidate_family": ("fresh_initial_noise" if kind.startswith("fresh")
                                 else "fixed_initial_noise_new_pnp_perturbation"),
            "initial_noise_seed": int(initial_seeds[index]),
            "perturbation_seed": int(
                np.asarray(source["perturb_seed"]) if index < 4
                else pnp_perturb_seeds[index - 4]),
            "pnp_u10": _candidate_u10(recorders[index]),
        }
        for index, kind in enumerate(kinds)
    }
    # Every root P&P path consumes the same fixed five perturbation tensors, independent of its
    # initial latent. Therefore the first fresh-noise lane ends at the exact source-stream RNG
    # position immediately after this root and can initialize all common-future continuations.
    return (kinds, arrays, metadata, obs_enc,
            tap.generators[0].get_state().clone())


def _root_training_prefix(arrays: dict, boundary_index: int) -> dict:
    """Return the exact frozen SmolVLA prefix at the persisted source root."""
    return _compact_training_prefix({
        "prefix_embeddings": np.asarray(
            arrays["prefix/prefix_embeddings"][boundary_index]).copy(),
        "prefix_pad_masks": np.asarray(
            arrays["prefix/prefix_pad_masks"][boundary_index]).copy(),
    })


def _compact_training_prefix(value: dict) -> dict:
    """Keep and deterministically pool only the frozen critic prefix tensors."""
    required = ("prefix_embeddings", "prefix_pad_masks")
    missing = [name for name in required if name not in value]
    if missing:
        raise RuntimeError(f"captured SmolVLA prefix is missing {missing}")
    embeddings = np.asarray(value["prefix_embeddings"])
    masks = np.asarray(value["prefix_pad_masks"], bool)
    while embeddings.ndim > 2 and embeddings.shape[0] == 1:
        embeddings = embeddings[0]
    while masks.ndim > 1 and masks.shape[0] == 1:
        masks = masks[0]
    masks = masks.reshape(-1)
    if embeddings.ndim != 2 or len(embeddings) != len(masks):
        raise ValueError(
            f"invalid prefix shapes {embeddings.shape}/{masks.shape}")
    embeddings = embeddings[masks]
    if not len(embeddings):
        raise ValueError("captured prefix has no valid tokens")
    target = SMOLVLA_BRANCH_PREFIX_TOKENS
    tensor = torch.as_tensor(embeddings, dtype=torch.float32).T.unsqueeze(0)
    if len(embeddings) > target:
        tensor = torch.nn.functional.adaptive_avg_pool1d(tensor, target)
    pooled = tensor.squeeze(0).T.numpy().astype(np.float16)
    result = np.zeros((target, pooled.shape[-1]), np.float16)
    width = min(target, len(pooled))
    result[:width] = pooled[:width]
    valid = np.zeros(target, bool); valid[:width] = True
    # Preserve the historical singleton lane axis used by source artifacts.
    return {"prefix_embeddings": result[None], "prefix_pad_masks": valid[None]}


def _pack_branch_training_data(*, decisions: list[dict], prefixes: list[dict],
                               generated_chunks: list[np.ndarray],
                               normalized_actions: list[np.ndarray],
                               env_actions: list[np.ndarray], rewards: list[float],
                               terminated: list[bool], truncated: list[bool],
                               step_success: list[bool], robot_states: list[np.ndarray],
                               sim_states: list[np.ndarray], chunk_start_steps: list[int],
                               chunk_noise_seeds: list[int], episode_seed: int,
                               initial_state: np.ndarray) -> dict[str, np.ndarray]:
    """Build the compact, lossless Q10 artifact used by EMA Bellman training.

    Unlike the old v2 tree rows, this contains each executed branch transition
    and both sides of every nonterminal bootstrap.  Raw camera pixels and the
    duplicate tokenization tensors are intentionally omitted; the exact frozen
    prefix embeddings needed by the critic are retained at every boundary.
    """
    transitions = len(chunk_start_steps)
    if not transitions or len(decisions) != transitions + 1 or len(prefixes) != transitions + 1:
        raise ValueError(
            "branch Bellman artifact needs C chunks and C+1 boundaries: "
            f"chunks={transitions}, decisions={len(decisions)}, prefixes={len(prefixes)}")
    n_steps = len(env_actions)
    if not (len(normalized_actions) == len(rewards) == len(terminated)
            == len(truncated) == len(step_success) == n_steps):
        raise ValueError("branch step arrays do not align")
    if len(robot_states) != n_steps + 1 or len(sim_states) != n_steps + 1:
        raise ValueError("branch physical-state arrays need T+1 rows")

    normalized = np.asarray(normalized_actions, np.float32)
    environment = np.asarray(env_actions, np.float32)
    action_dim = int(normalized.shape[-1])
    executed = np.zeros((transitions, ACTION_HORIZON, action_dim), np.float32)
    valid = np.zeros((transitions, ACTION_HORIZON), bool)
    for index, start in enumerate(chunk_start_steps):
        stop = (chunk_start_steps[index + 1]
                if index + 1 < transitions else n_steps)
        width = int(stop - start)
        if not 0 < width <= ACTION_HORIZON:
            raise ValueError(f"branch chunk {index} has invalid interval [{start},{stop})")
        executed[index, :width] = normalized[start:stop]
        valid[index, :width] = True

    prefixes = [_compact_training_prefix(value) for value in prefixes]
    embeddings = np.stack([
        np.asarray(value["prefix_embeddings"]) for value in prefixes]).astype(np.float16)
    masks = np.stack([
        np.asarray(value["prefix_pad_masks"]) for value in prefixes]).astype(bool)
    generated = np.asarray(generated_chunks, np.float32)
    if len(generated) != transitions or generated.shape[1] < ACTION_HORIZON:
        raise ValueError("one generated action chunk is required per branch transition")
    generated = generated[:, :, :action_dim]
    boundary_steps = np.asarray([row["step"] for row in decisions], np.int32)
    if boundary_steps[0] != 0 or boundary_steps[-1] != n_steps:
        raise ValueError(
            f"branch-local boundaries must span [0,{n_steps}], got "
            f"[{boundary_steps[0]},{boundary_steps[-1]}]")

    artifact = {
        "branch_training_schema_version": np.asarray(1, np.int16),
        "action_horizon": np.asarray(ACTION_HORIZON, np.int16),
        "episode_seed": np.asarray(episode_seed, np.int64),
        "initial_state": np.asarray(initial_state).copy(),
        "chunk_start_steps": np.asarray(chunk_start_steps, np.int32),
        "chunk_noise_seeds": np.asarray(chunk_noise_seeds, np.int64),
        "actions_normalized": normalized,
        "actions_env": environment,
        "rewards": np.asarray(rewards, np.float32),
        "terminated": np.asarray(terminated, bool),
        "truncated": np.asarray(truncated, bool),
        "step_success": np.asarray(step_success, bool),
        "robot_state_t_plus_1": np.asarray(robot_states, np.float32),
        "sim_state_t_plus_1": np.asarray(sim_states),
        "bellman/action": generated,
        "bellman/executed_normalized": executed,
        "bellman/validity_mask": valid,
        "boundary/step": boundary_steps,
        "boundary/raw_robot_state": np.stack([
            np.asarray(row["raw_robot_state"], np.float32) for row in decisions]),
        "boundary/policy_proprio": np.stack([
            np.asarray(row["policy_proprio"], np.float32) for row in decisions]),
        "boundary/sim_state": np.stack([
            np.asarray(row["sim_state"]) for row in decisions]),
        "boundary/instruction": np.asarray([
            str(row["instruction"]) for row in decisions], dtype=np.str_),
        "prefix/prefix_embeddings": embeddings,
        "prefix/prefix_pad_masks": masks,
    }
    # Validate with the exact downstream EMA-TD window builder before anything
    # is uploaded.  This catches a missing next boundary or an off-by-one mask
    # during collection rather than hours later in the training notebook.
    from .qplanning_critic.data import (
        _validate_qplanning_fields, qplanning_windows_from_artifact)
    _validate_qplanning_fields(artifact)
    windows = qplanning_windows_from_artifact(
        {"rollout_id": "branch-preflight"}, artifact,
        horizon=ACTION_HORIZON, gamma=.99)
    if len(windows["reward"]) != transitions:
        raise AssertionError("branch artifact/window transition counts differ")
    return artifact


def _run_training_continuation(
        env, obs, ep, policy, preprocess, postprocess, device, *,
        source: dict, source_arrays: dict, policy_chunk: np.ndarray,
        env_chunk: np.ndarray, branch_seed: int, source_next_perturb_state,
        root_noise_seed: int, root_step: int,
        root_sim_state: np.ndarray) -> tuple[bool, int, dict]:
    """Execute one branch and persist genuine Q10 current/next transitions."""
    config = _pnp_config(save_training_data=True)
    recorder = PnPRecorder(); recorder.new_episode()
    tap = RolloutTap(
        config, recorder, device, policy.model._pnp.action_dim,
        action_postprocess=postprocess)
    local_steps = 0
    absolute_steps = int(root_step)
    replan = int(source["boundary_index"]) + 1
    position_stride = int(policy.config.chunk_size)
    estimated_chunks = max(1, round(int(ep["max_steps"]) / position_stride))
    lead = 2
    skipping = bool(set_camera_observables(env, True))

    root_decision = {
        "step": 0,
        "raw_robot_state": np.asarray(source["raw_robot"], np.float32).copy(),
        "policy_proprio": np.asarray(source["policy_proprio"], np.float32).copy(),
        "sim_state": np.asarray(root_sim_state).copy(),
        "instruction": str(ep["task_desc"]),
    }
    decisions = [root_decision]
    prefixes = [_root_training_prefix(source_arrays, source["boundary_index"])]
    generated_chunks = [np.asarray(policy_chunk, np.float32).copy()]
    chunk_start_steps = [0]
    noise_seeds = [int(root_noise_seed)]
    normalized_actions: list[np.ndarray] = []
    environment_actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    success_flags: list[bool] = []
    robot_states = [np.asarray(source["raw_robot"], np.float32).copy()]
    sim_states = [np.asarray(root_sim_state).copy()]
    queue_policy = list(np.asarray(policy_chunk, np.float32)[:SMOLVLA_TREE_ACTIONS])
    queue_env = list(np.asarray(env_chunk, np.float32)[:SMOLVLA_TREE_ACTIONS])
    success = False

    def render_next(needed: bool) -> None:
        if skipping:
            set_camera_observables(env, needed)

    def execute_queue() -> bool:
        nonlocal obs, local_steps, absolute_steps, success
        while queue_policy:
            render_next(len(queue_policy) <= lead)
            normalized = np.asarray(queue_policy.pop(0), np.float32).reshape(-1)[:7]
            action = np.asarray(queue_env.pop(0), np.float32).reshape(-1)[:7]
            obs, reward, done, _ = env.step(action)
            local_steps += 1; absolute_steps += 1
            step_success = bool(env.check_success())
            terminal = bool(done or step_success)
            truncation = bool(absolute_steps >= int(ep["max_steps"]) and not terminal)
            normalized_actions.append(normalized.copy())
            environment_actions.append(action.copy())
            rewards.append(float(reward))
            terminated_flags.append(terminal)
            truncated_flags.append(truncation)
            success_flags.append(step_success)
            robot_states.append(_raw_robot_state(obs))
            state = _sim_state(env)
            if state is None:
                raise RuntimeError("branch Bellman collection requires simulator state")
            sim_states.append(state)
            if step_success:
                success = True
            if terminal or truncation:
                return True
        return False

    _pnp_gen(device).set_state(source_next_perturb_state)
    previous_steps = policy.model._pnp.num_steps
    policy.model._pnp.num_steps = SMOLVLA_TREE_INTEGRATION_STEPS
    try:
        with _temp_strategy(policy.model, tap):
            finished = execute_queue()
            while not finished and absolute_steps < int(ep["max_steps"]):
                policy.model._pnp.chunk_pos = min(
                    (absolute_steps // SMOLVLA_TREE_ACTIONS) / estimated_chunks, 1.0)
                policy_observation = obs_to_policy(obs, ep["task_desc"])
                decisions.append(_training_decision(
                    obs, env, ep["task_desc"], local_steps, policy_observation))
                seed = chunk_noise_seed(branch_seed, replan)
                noise = _draw_chunk_noise(policy, device, seed)
                with torch.no_grad():
                    chunk = policy.predict_action_chunk(preprocess(policy_observation), noise=noise)
                generated = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if len(tap.training_prefixes) != len(decisions) - 1:
                    raise RuntimeError("branch continuation prefix capture fell out of alignment")
                prefixes.append(_compact_training_prefix(tap.training_prefixes[-1]))
                generated_chunks.append(generated.copy())
                chunk_start_steps.append(local_steps)
                noise_seeds.append(int(seed))
                queue_policy[:] = list(generated[:SMOLVLA_TREE_ACTIONS])
                queue_env[:] = list(postprocess_chunk(
                    generated[:SMOLVLA_TREE_ACTIONS], postprocess, device))
                replan += 1
                finished = execute_queue()

        # Every transition needs an actual next decision-boundary prefix.  A
        # terminal/truncated observation may have skipped cameras, so refresh it
        # at the same simulator state before the capture-only model call.
        obs = refresh_camera_observation(env, obs)
        terminal_observation = obs_to_policy(obs, ep["task_desc"])
        decisions.append(_training_decision(
            obs, env, ep["task_desc"], local_steps, terminal_observation))
        terminal_tap = PrefixCaptureTap()
        terminal_seed = chunk_noise_seed(branch_seed, replan)
        terminal_noise = _draw_chunk_noise(policy, device, terminal_seed)
        policy.model._pnp.chunk_pos = min(
            (absolute_steps // SMOLVLA_TREE_ACTIONS) / estimated_chunks, 1.0)
        with _temp_strategy(policy.model, terminal_tap), torch.no_grad():
            policy.predict_action_chunk(preprocess(terminal_observation), noise=terminal_noise)
        prefixes.append(_compact_training_prefix(terminal_tap.training_prefixes[0]))
    finally:
        policy.model._pnp.num_steps = previous_steps
        if skipping:
            set_camera_observables(env, True)

    training_data = _pack_branch_training_data(
        decisions=decisions, prefixes=prefixes, generated_chunks=generated_chunks,
        normalized_actions=normalized_actions, env_actions=environment_actions,
        rewards=rewards, terminated=terminated_flags, truncated=truncated_flags,
        step_success=success_flags, robot_states=robot_states, sim_states=sim_states,
        chunk_start_steps=chunk_start_steps, chunk_noise_seeds=noise_seeds,
        episode_seed=branch_seed, initial_state=np.asarray(root_sim_state))
    return success, absolute_steps, training_data


def collect_smolvla_depth1_tree(env, ep, policy, preprocess, postprocess, device, *,
                                item: dict, bundle: dict, manifest_hash: str) -> tuple[dict, list]:
    source = _source_boundary(bundle, item)
    source["perturb_seed"] = np.asarray(bundle["arrays"]["perturb_seed"])
    root_step = int(source["root_step"])
    parent_actions = np.asarray(bundle["actions"][:root_step], np.float32)
    obs, terminal_events = _reset_and_replay_actions(
        env, ep, policy, parent_actions, skip_unused_renders=True, render_lead=2)
    _, sim = _unwrap_sim(env)
    replay_state = np.asarray(sim.get_state().flatten()).copy()
    setter = getattr(sim, "set_state_from_flattened", None)
    if not callable(setter):
        raise RuntimeError("MuJoCo simulator has no flat-state restoration")
    setter(np.asarray(source["sim_state"]).copy()); sim.forward()
    canonical_sim_state = copy.deepcopy(sim.get_state())
    canonical_state = np.asarray(canonical_sim_state.flatten()).copy()
    state_error = float(np.max(np.abs(canonical_state - np.asarray(source["sim_state"]))))
    if state_error > 1e-6:
        raise RuntimeError(f"source root restoration failed (max_abs={state_error:.3g})")

    estimated_chunks = max(1, round(
        int(ep["max_steps"]) / int(policy.config.chunk_size)))
    policy.model._pnp.chunk_pos = min(item["chunk_idx"] / estimated_chunks, 1.0)
    policy_observation = _source_policy_observation(
        bundle["arrays"], source["boundary_index"], ep["task_desc"])
    batch = preprocess(policy_observation)
    (kinds, alternatives, alternative_meta, obs_enc,
     source_next_perturb_state) = _generate_alternatives(
        policy, batch, device, item=item, source=source)

    policy_chunks = {"stored_source": np.asarray(source["policy_chunk"], np.float32)}
    policy_chunks.update({kind: alternatives[index] for index, kind in enumerate(kinds)})
    env_chunks = {
        kind: postprocess_chunk(chunk, postprocess, device)
        for kind, chunk in policy_chunks.items()
    }
    # Candidate zero is the exact already-observed edge, not a decoded approximation.
    source_width = min(SMOLVLA_TREE_ACTIONS, len(bundle["actions"]) - root_step)
    env_chunks["stored_source"][:source_width] = np.asarray(
        bundle["actions"][root_step:root_step + source_width], np.float32)

    group_id = candidate_group_id(
        "libero", item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=SMOLVLA_TREE_EXPERIMENT,
        trajectory_seed=item["source_episode_seed"])
    candidates = []
    for kind in policy_chunks:
        branch_training_data = None
        if kind == "stored_source":
            success, n_steps = bool(item["source_success"]), int(item["source_n_steps"])
            branch_meta = {
                "candidate_family": "stored_source",
                "reused_historical_suffix": True,
                "initial_noise_seed": int(source["noise_seed"]),
                "bellman_data_mode": "source_artifact_suffix",
                "training_data_path": str(bundle["row"]["training_data_path"]),
                "training_data_start_boundary": int(source["boundary_index"]),
            }
        else:
            branch_obs, branch_events = _reset_and_replay_actions(
                env, ep, policy, parent_actions,
                skip_unused_renders=True, render_lead=2)
            _, branch_sim = _unwrap_sim(env)
            branch_sim.set_state(copy.deepcopy(canonical_sim_state)); branch_sim.forward()
            corrected = np.asarray(branch_sim.get_state().flatten())
            correction_error = float(np.max(np.abs(corrected - canonical_state)))
            if correction_error > 1e-6:
                raise RuntimeError(
                    f"branch root restoration failed (max_abs={correction_error:.3g})")
            # Restore the source stream immediately after its root chunk. Every branch then gets
            # the same source future initial-noise and P&P randomness; only intervention differs.
            success, n_steps, branch_training_data = _run_training_continuation(
                env, branch_obs, ep, policy, preprocess, postprocess, device,
                source=source, source_arrays=bundle["arrays"],
                policy_chunk=policy_chunks[kind], env_chunk=env_chunks[kind],
                branch_seed=int(item["source_episode_seed"]),
                source_next_perturb_state=source_next_perturb_state,
                root_noise_seed=int(alternative_meta[kind]["initial_noise_seed"]),
                root_step=root_step, root_sim_state=canonical_state)
            branch_meta = {
                **alternative_meta[kind],
                "reused_historical_suffix": False,
                "common_continuation_seed": int(item["source_episode_seed"]),
                "parent_replay_terminal_events": branch_events,
                "root_restore_max_abs": correction_error,
                "bellman_data_mode": "persisted_counterfactual_branch",
                "bellman_transitions": int(len(
                    branch_training_data["bellman/action"])),
            }
        candidate_id = hashlib.sha256(f"{group_id}|{kind}".encode()).hexdigest()[:24]
        candidates.append({
            "candidate_id": candidate_id, "candidate_kind": kind,
            "success": bool(success), "n_steps": int(n_steps),
            "rollout_id": (item["source_rollout_id"] if kind == "stored_source" else None),
            "metadata_json": {
                **branch_meta, "source_rollout_id": item["source_rollout_id"],
                "chunk_idx": int(item["chunk_idx"]),
                "executed_prefix_length": SMOLVLA_TREE_ACTIONS,
                "policy_chunk_sha256": _content_digest(policy_chunks[kind]),
            },
            "blobs": {
                "policy_chunk": {"actions": policy_chunks[kind]},
                "env_chunk": {"actions": env_chunks[kind],
                              "mask": np.ones(len(env_chunks[kind]), dtype=np.bool_)},
                "observation": {
                    "obs_enc": obs_enc,
                    "policy_proprio": np.asarray(source["policy_proprio"], np.float32),
                },
                **({"training_data": branch_training_data}
                   if branch_training_data is not None else {}),
            },
        })
    group = {
        "candidate_group_id": group_id, "experiment": SMOLVLA_TREE_EXPERIMENT,
        "benchmark": "libero", "suite": item["suite"],
        "task_idx": int(item["task_idx"]), "episode_idx": int(item["episode_idx"]),
        "chunk_idx": int(item["chunk_idx"]),
        "uncertainty_stratum": item["selection_strategy"],
        "pairing_mode": "exact_source_root_hybrid_candidates",
        "prefix_length": SMOLVLA_TREE_ACTIONS, "snapshot_validated": True,
        "trajectory_seed": int(item["source_episode_seed"]),
        "collection_split": "smolvla_tree_train",
        "manifest_hash": manifest_hash, "model_revision": SMOLVLA_REPO_ID,
        "metadata_json": {
            "source_rollout_id": item["source_rollout_id"],
            "source_training_data_path": bundle["row"]["training_data_path"],
            "source_boundary_index": int(source["boundary_index"]),
            "source_success": bool(item["source_success"]),
            "source_root_u10": float(item["source_root_u10"]),
            "source_u10_profile": item["source_u10_profile"],
            "root_selection_strategy": item["selection_strategy"],
            "short_episode_root_fallback": bool(item["short_episode_root_fallback"]),
            "candidate_families": {"stored_source": 1, "fresh_initial_noise": 4,
                                   "fixed_initial_noise_new_pnp_perturbation": 4},
            "candidate_count": SMOLVLA_TREE_CANDIDATES,
            "integration_steps": SMOLVLA_TREE_INTEGRATION_STEPS,
            "n_action_steps": SMOLVLA_TREE_ACTIONS,
            "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "parent_replay_terminal_events": terminal_events,
            "replay_root_max_abs_vs_source": float(np.max(np.abs(
                replay_state - np.asarray(source["sim_state"])))),
            "source_state_set_max_abs": state_error,
            "root_sim_state_sha256": _content_digest(canonical_state),
            "root_policy_input_sha256": _content_digest(batch),
            "common_continuation_seed": int(item["source_episode_seed"]),
            "common_continuation_replan_start": int(item["chunk_idx"]) + 1,
            "chunk_position_stride": int(policy.config.chunk_size),
            "source_candidate_mode": "exact_stored_artifact_no_redecode",
            "counterfactual_generation_batch_size": 8,
            "counterfactual_artifact": (
                "Q10 current/next frozen prefixes, physical states, generated/executed "
                "actions, rewards, and terminal masks"),
            "branch_training_schema_version": 1,
            "critic_target": "EMA Bellman r_0:9 + gamma^10 Qbar(next)",
            "videos": False,
        },
    }
    return group, candidates


def _complete_groups(store, items: list[dict]) -> tuple[set[str], dict[str, list[dict]]]:
    rows = store.fetch_all(
        "verifier_candidate_groups", "candidate_group_id,metadata_json",
        configure=lambda query: query.eq("experiment", SMOLVLA_TREE_EXPERIMENT),
        order_by=("candidate_group_id",))
    ids = [row["candidate_group_id"] for row in rows]
    candidates = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        candidates.extend(store.fetch_all(
            "verifier_candidates",
            "candidate_id,candidate_group_id,candidate_kind,success,n_steps,metadata_json",
            configure=lambda query, batch=batch: query.in_("candidate_group_id", batch),
            order_by=("candidate_group_id", "candidate_id")))
    by_group: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_group[str(candidate["candidate_group_id"])].append(candidate)
    group_metadata = {
        str(row["candidate_group_id"]): dict(row.get("metadata_json") or {})
        for row in rows}

    def bellman_complete(group_id: str, values: list[dict]) -> bool:
        if len(values) != SMOLVLA_TREE_CANDIDATES:
            return False
        if group_metadata.get(group_id, {}).get("branch_training_schema_version") != 1:
            return False
        return all(bool((row.get("metadata_json") or {}).get("training_data_path"))
                   for row in values)

    complete = {group_id for group_id, values in by_group.items()
                if bellman_complete(group_id, values)}
    return complete, by_group


def _print_progress(store, items: list[dict]) -> tuple[set[str], str]:
    complete, by_group = _complete_groups(store, items)
    allowed = {
        candidate_group_id(
            "libero", item["suite"], item["task_idx"], item["episode_idx"],
            item["chunk_idx"], namespace=SMOLVLA_TREE_EXPERIMENT,
            trajectory_seed=item["source_episode_seed"])
        for item in items
    }
    complete &= allowed
    trees = [by_group[group_id] for group_id in sorted(complete)]
    if not trees:
        table = "No complete trees in this shard yet."
        return complete, table

    def candidate(tree, kind):
        return next(row for row in tree if row["candidate_kind"] == kind)

    stock = [bool(candidate(tree, "stored_source")["success"]) for tree in trees]
    fresh_any = [any(bool(row["success"]) for row in tree
                     if str(row["candidate_kind"]).startswith("fresh_seed_")) for tree in trees]
    perturb_any = [any(bool(row["success"]) for row in tree
                       if str(row["candidate_kind"]).startswith("pnp_perturb_")) for tree in trees]
    outcomes = [[bool(row["success"]) for row in tree] for tree in trees]
    branch_successes = sum(sum(values) for values in outcomes)
    mixed = sum(len(set(values)) > 1 for values in outcomes)
    any_success = [any(values) for values in outcomes]
    failures = max(1, sum(not value for value in stock))
    pct = lambda value, denominator=len(trees): 100 * value / max(denominator, 1)
    table = (
        "Exact depth-1 SmolVLA trees in THIS shard; partial trees excluded.\n"
        f"trees={len(trees)}/{len(items)} | branches={len(trees) * SMOLVLA_TREE_CANDIDATES} | "
        f"branch SR={pct(branch_successes, len(trees) * SMOLVLA_TREE_CANDIDATES):.1f}% | "
        f"mixed={pct(mixed):.1f}%\n"
        f"stock SR={pct(sum(stock)):.1f}% | any-success={pct(sum(any_success)):.1f}% | "
        f"oracle gain={pct(sum(any_success)) - pct(sum(stock)):+.1f} pp\n"
        f"among stock failures: fresh-seed any-success={pct(sum(
            fresh and not base for fresh, base in zip(fresh_any, stock)), failures):.1f}% | "
        f"P&P-perturb any-success={pct(sum(
            perturb and not base for perturb, base in zip(perturb_any, stock)), failures):.1f}%"
    )
    return complete, table


def run_smolvla_tree_worker(*, shard_count: int = SMOLVLA_TREE_SHARDS,
                            shard_index: int = 0,
                            tree_limit: int | None = None,
                            store=None) -> dict:
    from . import libero_env, models

    if shard_count != SMOLVLA_TREE_SHARDS or shard_index not in range(shard_count):
        raise ValueError("the fixed tree manifest requires shard_count=2 and shard_index 0 or 1")
    store = store or SupabaseStore()
    document = load_or_build_smolvla_tree_manifest(store)
    payload = document["payload"]
    items = [dict(item) for item in payload["items"]
             if int(item["shard_index"]) == int(shard_index)]
    if tree_limit is not None:
        if int(tree_limit) < 1:
            raise ValueError("tree_limit must be positive or None")
        items = items[:int(tree_limit)]
    if tree_limit is None and len(items) != SMOLVLA_TREE_COUNT // SMOLVLA_TREE_SHARDS:
        raise AssertionError(f"worker has {len(items)} trees, expected 400")

    complete, table = _print_progress(store, items)
    print(table, flush=True)
    pending = []
    for item in items:
        group_id = candidate_group_id(
            "libero", item["suite"], item["task_idx"], item["episode_idx"],
            item["chunk_idx"], namespace=SMOLVLA_TREE_EXPERIMENT,
            trajectory_seed=item["source_episode_seed"])
        if group_id not in complete:
            pending.append(item)
    print(
        f"[smolvla-tree] namespace={SMOLVLA_TREE_EXPERIMENT} | "
        f"schema-complete={len(complete)}/{len(items)} | pending={len(pending)}",
        flush=True)
    if not pending:
        return {"new_trees": 0, "complete_trees": len(complete), "requested_trees": len(items)}

    if os.name == "posix" and torch.cuda.is_available():
        egl = subprocess.run(
            "ldconfig -p | grep -q libEGL_nvidia", shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if egl.returncode != 0:
            raise RuntimeError(
                "NVIDIA EGL is unavailable. Run the notebook's GPU-renderer package cell, "
                "restart the runtime if MuJoCo was already imported, and rerun from the top.")

    source_rows = {str(row["rollout_id"]): row for row in _source_rows(store)}
    episodes = prepare_smolvla_tree_source_episodes()
    lookup = {(ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"])): ep for ep in episodes}
    policy, preprocess, postprocess = models.load_smolvla()
    device = models.default_device()
    store.start_run(
        "smolvla_depth1_tree_collection", "libero", SMOLVLA_TREE_EXPERIMENT,
        config={
            "manifest_path": SMOLVLA_TREE_MANIFEST_PATH,
            "manifest_hash": document["manifest_hash"],
            "shard_count": shard_count, "shard_index": shard_index,
            "trees": len(items), "candidate_count": SMOLVLA_TREE_CANDIDATES,
            "candidate_families": payload["candidate_families"],
            "priority_fraction": SMOLVLA_TREE_PRIORITY_FRACTION,
            "integration_steps": SMOLVLA_TREE_INTEGRATION_STEPS,
            "n_action_steps": SMOLVLA_TREE_ACTIONS, "videos": False,
            "branch_artifact": "sequential_q10_bellman_v1",
            "branch_prefix_pool_tokens": SMOLVLA_BRANCH_PREFIX_TOKENS,
            "critic_target": "EMA Bellman r_0:9 + gamma^10 Qbar(next)",
        })
    new_trees = 0
    started = time.monotonic()
    try:
        for item in pending:
            tree_started = time.monotonic()
            print(
                f"[smolvla-tree] starting {new_trees + 1}/{len(pending)} | "
                f"{item['selection_strategy']} | {item['suite']} task {item['task_idx']} "
                f"ep {item['episode_idx']} chunk {item['chunk_idx']} | "
                f"U10={item['source_root_u10']:.5f}", flush=True)
            row = source_rows[item["source_rollout_id"]]
            bundle = _load_source_bundle(store, row)
            ep = lookup[(item["suite"], int(item["task_idx"]), int(item["episode_idx"]))]
            env = libero_env.make_env(ep["bddl_path"])
            try:
                group, candidates = collect_smolvla_depth1_tree(
                    env, ep, policy, preprocess, postprocess, device,
                    item=item, bundle=bundle, manifest_hash=document["manifest_hash"])
            finally:
                env.close()
            store.register_candidate_group(group, candidates)
            new_trees += 1
            outcomes = [bool(candidate["success"]) for candidate in candidates]
            families = defaultdict(list)
            for candidate in candidates:
                families[candidate["metadata_json"]["candidate_family"]].append(
                    bool(candidate["success"]))
            elapsed = time.monotonic() - started
            eta = elapsed / new_trees * (len(pending) - new_trees)
            print(
                f"[smolvla-tree] completed {new_trees}/{len(pending)} | "
                f"branches={sum(outcomes)}/{len(outcomes)} success | "
                f"stock={'S' if outcomes[0] else 'F'} | mixed={len(set(outcomes)) > 1} | "
                f"fresh={sum(families['fresh_initial_noise'])}/4 | "
                f"perturb={sum(families['fixed_initial_noise_new_pnp_perturbation'])}/4 | "
                f"tree={time.monotonic() - tree_started:.1f}s | ETA={eta / 60:.1f}m",
                flush=True)
            if new_trees % SMOLVLA_TREE_PRINT_EVERY == 0:
                _, table = _print_progress(store, items)
                print(table, flush=True)
    finally:
        store.finish_run(n_rollouts=new_trees * SMOLVLA_TREE_CANDIDATES)
    complete, table = _print_progress(store, items)
    print(table, flush=True)
    return {
        "manifest_hash": document["manifest_hash"], "shard_index": shard_index,
        "requested_trees": len(items), "new_trees": new_trees,
        "complete_trees": len(complete),
        "new_candidate_rows": new_trees * SMOLVLA_TREE_CANDIDATES,
    }
