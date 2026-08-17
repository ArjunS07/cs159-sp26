"""Same-observation candidate collection for an uncertainty-gradient critic.

The executed trajectory is always candidate slot 0, whose seed is the ordinary
baseline chunk seed. Extra slots are measurement-only and never touch the
environment. Each artifact therefore supplies within-state action variation
without changing the corrected 10-action LIBERO evaluation protocol.
"""
from __future__ import annotations

import datetime as dt
import math
import time

import numpy as np
import torch

from .config import ADIM, LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT, PI05_REPO_ID
from .libero_env import obs_to_policy, set_camera_observables
from .pnp import _pnp_seed_perturb
from .rollout import (
    _draw_chunk_noise, candidate_chunk_noise_seed, chunk_noise_seed,
    compute_instability, episode_seed, iter_task_envs)
from .sampler import enable_encoding_cache, measure_chunk_uncertainty, set_strategy
from .store import SupabaseStore


EXPERIMENT = "libero-u20-same-state-candidates-v1"
METHOD = "uncertainty_candidate_dataset"
COLLECTION_EPISODE_INDICES = tuple(range(20, 40))
EPISODES_PER_TASK = len(COLLECTION_EPISODE_INDICES)
CANDIDATE_COUNT = 6
TARGET_CHUNKS = (0, 2, 4, 8)
PROBE_STEPS = (3, 4)
PNP_K = 5
N_ACTION_STEPS = 10
SHARD_COUNT = 4
REPORT_EVERY = 10
SOURCE_MODEL_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
TRAIN_EPISODE_INDICES = tuple(range(20, 36))
VALIDATION_EPISODE_INDICES = tuple(range(36, 40))


def logical_config() -> dict:
    return {
        "method": METHOD,
        "source_model_repo_id": PI05_REPO_ID,
        "source_model_revision": SOURCE_MODEL_REVISION,
        "primary_critic_input": "ordinary_pre_refinement_action_chunk",
        "candidate_count": CANDIDATE_COUNT,
        "target_chunks": list(TARGET_CHUNKS),
        "probe_steps": list(PROBE_STEPS),
        "pnp_k": PNP_K,
        "n_action_steps": N_ACTION_STEPS,
        "candidate_zero_executed": True,
        "candidate_noise_scheme": "slot0_baseline_seed;slots1+independent",
        "candidate_labels": ["u_time", "u_iter_time"],
        "collection_episode_indices": list(COLLECTION_EPISODE_INDICES),
        "critic_data_split": {
            "train_episode_indices": list(TRAIN_EPISODE_INDICES),
            "validation_episode_indices": list(VALIDATION_EPISODE_INDICES),
        },
    }


def prepare_episodes(episode_indices=COLLECTION_EPISODE_INDICES):
    """Build all four standard suites with a larger, deterministic init-state slice."""
    from . import libero_env
    from .experiments import LIBERO_SUITES

    benchmark_dict = libero_env.init_libero_benchmark()
    tasks = [
        (suite, task_idx)
        for suite in LIBERO_SUITES
        for task_idx in range(benchmark_dict[suite]().n_tasks)
    ]
    episodes = libero_env.build_final_episodes(
        benchmark_dict, episode_idxs=episode_indices, tasks=tasks)
    expected = len(tasks) * len(episode_indices)
    if len(episodes) != expected:
        raise ValueError(
            f"expected {expected} standard-LIBERO identities "
            f"({len(episode_indices)}/task), found {len(episodes)}")
    return episodes


def _empty_artifact(obs_dim=0):
    steps = len(PROBE_STEPS)
    pairs = PNP_K - 1
    return {
        "group_chunk_idx": np.empty((0,), dtype=np.int16),
        "group_chunk_pos": np.empty((0,), dtype=np.float32),
        "candidate_noise_seeds": np.empty((0, CANDIDATE_COUNT), dtype=np.int64),
        "candidate_initial_action_chunk": np.empty(
            (0, CANDIDATE_COUNT, 50, ADIM), dtype=np.float32),
        "candidate_z_hat": np.empty(
            (0, CANDIDATE_COUNT, steps, 50, ADIM), dtype=np.float32),
        "candidate_first_a_hat": np.empty(
            (0, CANDIDATE_COUNT, steps, 50, ADIM), dtype=np.float32),
        "candidate_last_a_hat": np.empty(
            (0, CANDIDATE_COUNT, steps, 50, ADIM), dtype=np.float32),
        "candidate_u_time": np.empty(
            (0, CANDIDATE_COUNT, steps, 50), dtype=np.float32),
        "candidate_u_iter_time": np.empty(
            (0, CANDIDATE_COUNT, steps, pairs, 50), dtype=np.float32),
        "obs_enc": np.empty((0, obs_dim), dtype=np.float32),
        "step_indices": np.asarray(PROBE_STEPS, dtype=np.int16),
    }


def _stack_artifact(groups):
    if not groups:
        return _empty_artifact()
    keys = (
        "group_chunk_idx", "group_chunk_pos", "candidate_noise_seeds",
        "candidate_initial_action_chunk", "candidate_z_hat",
        "candidate_first_a_hat", "candidate_last_a_hat", "candidate_u_time",
        "candidate_u_iter_time", "obs_enc")
    artifact = {
        key: np.stack([group[key] for group in groups])
        for key in keys
    }
    artifact["step_indices"] = np.asarray(PROBE_STEPS, dtype=np.int16)
    return artifact


def _prefix_mean(array, width):
    return float(np.asarray(array)[..., :width].mean())


def collect_episode(env, ep, policy, preprocess, postprocess, device):
    """Run candidate-0 baseline and label extra same-observation candidates."""
    model = policy.model
    set_strategy(model, None)
    model._pnp.vf_evals = 0
    ep_seed = episode_seed(ep["init_state"], ep.get("ep_idx", 0))
    torch.manual_seed(ep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ep_seed)
    target_chunks = set(TARGET_CHUNKS)
    max_steps = int(ep["max_steps"])
    est_chunks = max(1, round(max_steps / policy.config.chunk_size))
    queue = []
    groups = []
    executed_actions = []
    chunk_boundary_actions = []
    success = False
    terminated_reason = "max_steps"
    step = -1
    chunk_idx = 0
    inference_ms = 0.0
    nan_count = 0
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.perf_counter()

    skipping = set_camera_observables(env, True)

    def render_next(needed):
        if skipping:
            set_camera_observables(env, bool(needed))

    try:
        env.reset()
        policy.reset()
        obs = env.set_init_state(ep["init_state"])
        for wait_step in range(NUM_STEPS_WAIT):
            render_next(wait_step >= NUM_STEPS_WAIT - 2)
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        for step in range(max_steps):
            if not queue:
                model._pnp.chunk_pos = min(chunk_idx / est_chunks, 1.0)
                batch = preprocess(obs_to_policy(obs, ep["task_desc"]))
                inference_start = time.perf_counter()
                if chunk_idx in target_chunks:
                    actions, features, seeds = [], [], []
                    for candidate_idx in range(CANDIDATE_COUNT):
                        seed = candidate_chunk_noise_seed(
                            ep_seed, chunk_idx, candidate_idx)
                        noise = _draw_chunk_noise(policy, device, seed)
                        _pnp_seed_perturb(seed)
                        action, _, _, feature = measure_chunk_uncertainty(
                            policy, batch, noise,
                            probe_steps=PROBE_STEPS, num_iterations=PNP_K,
                            uncertainty_horizon=20, return_details=True,
                            return_features=True)
                        actions.append(
                            action.squeeze(0).detach().float().cpu().numpy()[:, :ADIM])
                        features.append(feature)
                        seeds.append(seed)
                    reference_steps = features[0]["step_indices"]
                    reference_obs = features[0]["obs_enc"]
                    for feature in features[1:]:
                        if not np.array_equal(feature["step_indices"], reference_steps):
                            raise RuntimeError("candidate probe-step ordering drifted")
                        if not np.allclose(feature["obs_enc"], reference_obs, atol=1e-5, rtol=0):
                            raise RuntimeError("same-observation candidate encodings differ")
                    groups.append({
                        "group_chunk_idx": np.asarray(chunk_idx, dtype=np.int16),
                        "group_chunk_pos": np.asarray(
                            model._pnp.chunk_pos, dtype=np.float32),
                        "candidate_noise_seeds": np.asarray(seeds, dtype=np.int64),
                        # The first ordinary model query, before any P&P correction.
                        # This is the primary deployable critic input; z_hat/a_hat are
                        # retained only so the representation can be ablated later.
                        "candidate_initial_action_chunk":
                            np.stack(actions).astype(np.float32),
                        "candidate_z_hat": np.stack(
                            [feature["z_hat"] for feature in features]).astype(np.float32),
                        "candidate_first_a_hat": np.stack(
                            [feature["first_a_hat"] for feature in features]).astype(
                                np.float32),
                        "candidate_last_a_hat": np.stack(
                            [feature["last_a_hat"] for feature in features]).astype(
                                np.float32),
                        "candidate_u_time": np.stack(
                            [feature["u_time"] for feature in features]).astype(np.float32),
                        "candidate_u_iter_time": np.stack(
                            [feature["u_iter_time"] for feature in features]).astype(np.float32),
                        "obs_enc": reference_obs.astype(np.float32),
                    })
                    chunk = actions[0]
                else:
                    seed = chunk_noise_seed(ep_seed, chunk_idx)
                    noise = _draw_chunk_noise(policy, device, seed)
                    with torch.no_grad():
                        chunk = (policy.predict_action_chunk(batch, noise=noise)
                                 .squeeze(0).detach().float().cpu().numpy()[:, :ADIM])
                inference_ms += (time.perf_counter() - inference_start) * 1000
                queue = [action.copy() for action in chunk[:N_ACTION_STEPS]]
                chunk_boundary_actions.append(queue[0].copy())
                chunk_idx += 1

            action = queue.pop(0)
            if not np.all(np.isfinite(action)):
                nan_count += 1
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            policy_action = torch.as_tensor(action, device=device).unsqueeze(0)
            env_action = postprocess(policy_action)
            if isinstance(env_action, torch.Tensor):
                env_action = env_action.squeeze(0).detach().cpu().numpy()
            else:
                env_action = np.asarray(env_action).squeeze(0)
            env_action = np.asarray(env_action).reshape(-1)[:ADIM]
            executed_actions.append(env_action.copy())
            render_next(len(queue) < 2)
            obs, _, done, _ = env.step(env_action)
            if env.check_success():
                success = True
                terminated_reason = "success"
                break
            if done:
                terminated_reason = "env_done"
                break
    finally:
        if skipping:
            set_camera_observables(env, True)

    artifact = _stack_artifact(groups)
    u10 = _prefix_mean(artifact["candidate_u_time"], 10) if groups else None
    u20 = _prefix_mean(artifact["candidate_u_time"], 20) if groups else None
    u50 = _prefix_mean(artifact["candidate_u_time"], 50) if groups else None
    return {
        "success": success,
        "n_steps": step + 1,
        "n_chunks": chunk_idx,
        "n_candidate_groups": len(groups),
        "n_candidates": len(groups) * CANDIDATE_COUNT,
        "u10": u10, "u20": u20, "u50": u50,
        "artifact": artifact,
        "nan_action_count": nan_count,
        "elapsed_s": time.perf_counter() - started,
        "inference_ms_total": inference_ms,
        "n_vf_evals": int(model._pnp.vf_evals),
        "terminated_reason": terminated_reason,
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "instability": compute_instability(
            executed_actions, chunk_boundary_actions),
        "episode_seed": ep_seed,
    }


def _progress(tally, diagnostic):
    lines = [
        "n = completed baseline trajectories in THIS shard",
        f"{'suite':<20}{'SR':>15}{'groups':>12}{'candidates':>14}",
    ]
    for suite in sorted(tally):
        n, wins, groups, candidates = tally[suite]
        lines.append(
            f"{suite.removeprefix('libero_'):<20}"
            f"{wins / max(n, 1):>14.0%} {groups:>11}{candidates:>14}")
    if diagnostic["count"]:
        lines.append(
            "candidate-label means: "
            f"U10={diagnostic['u10'] / diagnostic['count']:.5f}  "
            f"U20={diagnostic['u20'] / diagnostic['count']:.5f}  "
            f"U50={diagnostic['u50'] / diagnostic['count']:.5f}")
    return "\n".join(lines)


def run_worker(shard_index: int, shard_count: int = SHARD_COUNT,
               experiment: str = EXPERIMENT, episode_limit: int | None = None):
    """Collect one resume-safe shard of the standard-LIBERO candidate dataset."""
    from collections import defaultdict
    from tqdm.auto import tqdm

    from . import models
    from .experiments import identity_shard

    if shard_count != SHARD_COUNT:
        raise ValueError(f"this fixed plan requires shard_count={SHARD_COUNT}")
    episodes = identity_shard(prepare_episodes(), shard_count, shard_index)
    if episode_limit is not None:
        if int(episode_limit) <= 0:
            raise ValueError("episode_limit must be positive or None")
        episodes = episodes[:int(episode_limit)]
    store = SupabaseStore()
    config = logical_config()
    config_hash = store.config_hash(config)
    completed_rows = store.fetch_all(
        "rollouts", "rollout_id",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "method", METHOD).eq("config_hash", config_hash).eq(
            "status", "completed"),
        order_by=("rollout_id",))
    done = {row["rollout_id"] for row in completed_rows}
    pending = []
    for ep in episodes:
        identity = {key: ep.get(key) for key in (
            "benchmark", "suite", "task_idx", "ep_idx", "init_state_hash")}
        rid = store.make_rollout_id(experiment, identity, config)
        if rid not in done:
            pending.append((ep, rid))

    print({
        "experiment": experiment,
        "standard_libero_development_identities_total": EPISODES_PER_TASK * 40,
        "identities_in_shard": len(episodes),
        "pending_in_shard": len(pending),
        "candidates_per_state": CANDIDATE_COUNT,
        "episode_limit": episode_limit,
        "target_chunk_indices": list(TARGET_CHUNKS),
        "maximum_candidate_examples_in_shard":
            len(episodes) * len(TARGET_CHUNKS) * CANDIDATE_COUNT,
        "probe_steps": list(PROBE_STEPS), "pnp_k": PNP_K,
        "executed_candidate": 0, "n_action_steps": N_ACTION_STEPS,
        "source_model_revision": SOURCE_MODEL_REVISION,
    })

    policy, preprocess, postprocess = models.load_pi05(revision=SOURCE_MODEL_REVISION)
    device = models.default_device()
    enable_encoding_cache(policy.model, size=4, model_revision=SOURCE_MODEL_REVISION)
    store.start_run(
        driver="libero_uncertainty_candidate_dataset", benchmark="libero",
        experiment=experiment, config={
            **config, "shard_count": shard_count, "shard_index": shard_index})

    tally = defaultdict(lambda: [0, 0, 0, 0])
    diagnostic = {"u10": 0.0, "u20": 0.0, "u50": 0.0, "count": 0}
    completed = 0
    try:
        by_identity = {(
            ep["suite"], ep["task_idx"], ep["ep_idx"], ep["init_state_hash"]): rid
            for ep, rid in pending}
        pending_episodes = [ep for ep, _ in pending]
        with tqdm(total=len(pending), desc=f"u-critic[{shard_index}]",
                  unit="episode", dynamic_ncols=True) as progress:
            for env, task_episodes in iter_task_envs(pending_episodes):
                for ep in task_episodes:
                    key = (ep["suite"], ep["task_idx"], ep["ep_idx"],
                           ep["init_state_hash"])
                    rid = by_identity[key]
                    try:
                        result = collect_episode(
                            env, ep, policy, preprocess, postprocess, device)
                        row = {
                            "rollout_id": rid, "benchmark": "libero",
                            "suite": ep["suite"], "task_idx": ep["task_idx"],
                            "task_desc": ep["task_desc"], "episode_idx": ep["ep_idx"],
                            "init_state_hash": ep["init_state_hash"],
                            "bddl_sha256": ep["bddl_sha256"],
                            "suite_family": ep["suite_family"],
                            "max_steps": ep["max_steps"], "chunk_size": 50,
                            "n_chunks": result["n_chunks"], "method": METHOD,
                            "pnp_enabled": True,
                            "pnp_step_indices": list(PROBE_STEPS), "pnp_k": PNP_K,
                            "action_dim": ADIM, "num_samples": CANDIDATE_COUNT,
                            "episode_seed": result["episode_seed"],
                            "perturb_seed": result["episode_seed"],
                            "config_json": config, "config_hash": config_hash,
                            "success": result["success"], "n_steps": result["n_steps"],
                            "elapsed_s": result["elapsed_s"],
                            "terminated_reason": result["terminated_reason"],
                            "status": "completed", "error_msg": None,
                            "nan_action_count": result["nan_action_count"],
                            "started_at": result["started_at"],
                            "finished_at": result["finished_at"],
                            "n_vf_evals": result["n_vf_evals"],
                            "inference_ms_total": result["inference_ms_total"],
                            "n_pnp_activations":
                                result["n_candidates"] * len(PROBE_STEPS),
                            "ms_candidate_u": {
                                "dataset": "same_observation_uncertainty_critic",
                                "candidate_count": CANDIDATE_COUNT,
                                "target_chunks": list(TARGET_CHUNKS),
                                "groups_collected": result["n_candidate_groups"],
                                "critic_split": (
                                    "train" if ep["ep_idx"] in TRAIN_EPISODE_INDICES
                                    else "validation"),
                                "candidates_collected": result["n_candidates"],
                                "u10_mean": result["u10"],
                                "u20_mean": result["u20"],
                                "u50_mean": result["u50"],
                            },
                            **result["instability"],
                        }
                        store.log_episode(
                            row, blobs={"generated_chunks": result["artifact"]})
                    except Exception as error:
                        now = dt.datetime.now(dt.timezone.utc).isoformat()
                        store.log_episode({
                            "rollout_id": rid, "benchmark": "libero",
                            "suite": ep["suite"], "task_idx": ep["task_idx"],
                            "task_desc": ep["task_desc"], "episode_idx": ep["ep_idx"],
                            "init_state_hash": ep["init_state_hash"],
                            "bddl_sha256": ep["bddl_sha256"],
                            "max_steps": ep["max_steps"], "method": METHOD,
                            "pnp_enabled": True,
                            "pnp_step_indices": list(PROBE_STEPS), "pnp_k": PNP_K,
                            "action_dim": ADIM, "num_samples": CANDIDATE_COUNT,
                            "config_json": config, "config_hash": config_hash,
                            "success": None, "n_steps": 0, "elapsed_s": 0,
                            "terminated_reason": "error", "status": "errored",
                            "error_msg": f"{type(error).__name__}: {error}",
                            "nan_action_count": 0,
                            "started_at": now, "finished_at": now,
                        })
                        raise

                    completed += 1
                    counts = tally[ep["suite"]]
                    counts[0] += 1
                    counts[1] += int(result["success"])
                    counts[2] += result["n_candidate_groups"]
                    counts[3] += result["n_candidates"]
                    if result["u10"] is not None:
                        diagnostic["u10"] += result["u10"]
                        diagnostic["u20"] += result["u20"]
                        diagnostic["u50"] += result["u50"]
                        diagnostic["count"] += 1
                    progress.update()
                    progress.set_postfix_str(
                        f"{ep['suite'].removeprefix('libero_')} "
                        f"groups={result['n_candidate_groups']} "
                        f"U20={result['u20']:.4f}", refresh=False)
                    if completed % REPORT_EVERY == 0:
                        tqdm.write(f"\n--- u-critic[{shard_index}] after "
                                   f"{completed}/{len(pending)} episodes ---")
                        tqdm.write(_progress(tally, diagnostic))
    except BaseException:
        store.finish_run(status="failed", n_rollouts=completed)
        raise
    store.finish_run(n_rollouts=completed)
    print(f"u-critic[{shard_index}]: logged {completed} new baseline trajectories")
    if tally:
        print(_progress(tally, diagnostic))
    return completed
