"""Rollout PRIMITIVES (Dedup #2). No experiment/driver loops — those live in the notebooks.

`run_episode(env, ep, policy, preprocess, device, config)` runs ONE episode: it builds the
`RolloutTap` from `config`, owns the per-(episode,chunk) noise generator so chunk i's initial
noise is byte-identical across every method, and returns a plain result dict whose contents
depend on which sinks were enabled. `iter_task_envs(episodes)` yields `(env, task_eps)`
handling the OffScreenRenderEnv lifecycle. Nothing here writes to the store — the notebook
loop calls `store.log_result(...)` with the result.
"""
from __future__ import annotations

import hashlib
import time
from itertools import groupby

import numpy as np
import torch

from .config import ADIM, LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT, RolloutConfig
from .libero_env import make_env, obs_to_policy
from .pnp import PnPRecorder, _pnp_seed_perturb, multi_sample_select
from .tap import RolloutTap
from . import sampler as _sampler


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────
def episode_seed(init_state, episode_idx) -> int:
    b = hashlib.md5(np.asarray(init_state).tobytes() + str(episode_idx or 0).encode()).digest()
    return int.from_bytes(b[:4], "big")


def chunk_noise_seed(ep_seed: int, chunk_idx: int) -> int:
    """Deterministic per-(episode,chunk) noise seed — the same across every method."""
    b = hashlib.md5(f"{int(ep_seed)}:{int(chunk_idx)}".encode()).digest()
    return int.from_bytes(b[:4], "big")


def _draw_chunk_noise(policy, device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(int(seed))
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    return torch.empty(shape, device=device).normal_(generator=gen)


# ─────────────────────────────────────────────────────────────────────────────
# Instability metrics (ported from smolvla_eval_core.py)
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_disagreement(chunk_boundary_actions):
    if len(chunk_boundary_actions) < 2:
        return None
    d = [float(np.linalg.norm(chunk_boundary_actions[i + 1] - chunk_boundary_actions[i]))
         for i in range(len(chunk_boundary_actions) - 1)]
    return float(np.mean(d))


def compute_instability(executed_actions, chunk_boundary_actions=None, gripper_dim=6):
    if not executed_actions:
        return dict(action_delta_l2_mean=0.0, action_delta_l2_max=0.0, action_var_mean=0.0,
                    gripper_flip_count=0, gripper_flip_rate=0.0, chunk_disagreement_mean=None)
    arr = np.stack([np.asarray(a).flatten()[:ADIM] for a in executed_actions])
    if len(arr) >= 2:
        deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        d_mean, d_max = float(np.mean(deltas)), float(np.max(deltas))
    else:
        d_mean = d_max = 0.0
    signs = (arr[:, gripper_dim] > 0.0).astype(int)
    flips = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
    return dict(
        action_delta_l2_mean=d_mean, action_delta_l2_max=d_max,
        action_var_mean=float(np.var(arr, axis=0).mean()),
        gripper_flip_count=flips, gripper_flip_rate=flips / max(len(arr) - 1, 1),
        chunk_disagreement_mean=_chunk_disagreement(chunk_boundary_actions or []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config -> tap (the notebook never constructs the tap directly)
# ─────────────────────────────────────────────────────────────────────────────
def build_tap(config: RolloutConfig, recorder: PnPRecorder, device, adim: int):
    """A tap exists iff the rollout has a probe. Vanilla / extra_steps / multi-sample (which
    probes at the chunk level) run with no tap installed."""
    if not config.has_probe:
        return None
    return RolloutTap(config, recorder, device, adim)


# ─────────────────────────────────────────────────────────────────────────────
# The one rollout primitive.
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(env, ep, policy, preprocess, device, config: RolloutConfig | None = None):
    """Run one episode under `config`. Returns a result dict (outcome, metrics, trajectory,
    recorder episode, and sink outputs — pcp_chunks / pcp_telemetry / ms_selections).

    Store/DB writes are the notebook's job (via store.log_result)."""
    config = config or RolloutConfig()
    model = policy.model
    adim = getattr(model, "_pnp_action_dim", ADIM)
    task_desc = ep["task_desc"]
    max_steps = ep["max_steps"]
    chunk_size = policy.config.chunk_size
    multisample = config.num_samples is not None

    recorder = PnPRecorder()
    tap = build_tap(config, recorder, device, adim)
    model._pnp_num_steps = config.num_inference_steps   # extra_steps override (None = default)

    ep_seed = episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
    torch.manual_seed(ep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ep_seed)
    _pnp_seed_perturb(ep_seed)                           # isolated perturbation stream

    _sampler.set_strategy(model, tap)
    model._pnp_vf_evals = 0
    recorder.new_episode(meta={k: ep.get(k) for k in ("suite", "task_idx", "ep_idx")})

    est_chunks = max(1, round(max_steps / chunk_size))
    queue, ci = [], 0
    executed_actions, robot_states, chunk_boundary_actions, chunk_noise_seeds = [], [], [], []
    ms_selections = [] if multisample else None
    frames = [] if config.save_observations else None
    nan_count = 0
    success = False
    status, error_msg = "completed", None
    t0 = time.time()
    step = 0
    try:
        env.reset()
        policy.reset()
        obs = env.set_init_state(ep["init_state"])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        for step in range(max_steps):
            if not queue:
                model._pnp_chunk_pos = min(ci / est_chunks, 1.0)
                cns = chunk_noise_seed(ep_seed, ci)
                chunk_noise_seeds.append(cns)
                noise = _draw_chunk_noise(policy, device, cns)
                batch = preprocess(obs_to_policy(obs, task_desc))
                if multisample:
                    def _noise_of(si, _ci=ci):
                        return _draw_chunk_noise(policy, device,
                                                 chunk_noise_seed(ep_seed, _ci * 1000 + si))
                    chunk, chosen, cand_u = multi_sample_select(
                        policy, batch, ep_seed, ci, config.num_samples,
                        tuple(config.ms_probe_steps), _noise_of)
                    ms_selections.append({"chunk_idx": ci, "chosen": int(chosen), "cand_u": cand_u})
                else:
                    with torch.no_grad():
                        chunk = policy.predict_action_chunk(batch, noise=noise)
                arr = chunk.squeeze(0).detach().cpu().numpy()
                queue = [arr[i].copy() for i in range(arr.shape[0])]
                chunk_boundary_actions.append(np.asarray(queue[0]).flatten()[:ADIM].copy())
                ci += 1
            a = queue.pop(0)
            if not np.all(np.isfinite(a)):
                nan_count += 1
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            executed_actions.append(np.asarray(a).flatten()[:ADIM].copy())
            if frames is not None:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            robot_states.append(np.concatenate([
                obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]).copy())
            obs, _, done, _ = env.step(a)
            if env.check_success():
                success = True
                break
            if done:
                break
    except Exception as e:                              # log errored episodes, don't drop them
        status, error_msg = "errored", f"{type(e).__name__}: {e}"

    elapsed = time.time() - t0
    n_steps = step + 1
    recorder.close_episode(success, n_steps)
    inst = compute_instability(executed_actions, chunk_boundary_actions)

    vf_evals = int(getattr(model, "_pnp_vf_evals", 0))
    if vf_evals == 0:                                   # vanilla path (orig sampler doesn't count)
        vf_evals = (config.num_inference_steps or policy.config.num_inference_steps) * max(ci, 1)

    result = dict(
        success=success, n_steps=n_steps, elapsed_s=elapsed, status=status, error_msg=error_msg,
        nan_action_count=nan_count, n_chunks=ci, n_vf_evals=vf_evals, chunk_size=chunk_size,
        episode_seed=ep_seed, chunk_noise_seeds=chunk_noise_seeds, instability=inst,
        recorder_episode=recorder.episodes[-1] if recorder.episodes else None,
        trajectory=dict(
            actions=np.asarray(executed_actions, dtype=np.float32),
            robot_state=np.asarray(robot_states, dtype=np.float32),
        ) if config.save_trajectory else None,
        obs_frames=frames,
    )
    if tap is not None:
        if tap.save_pcp:
            result["pcp_chunks"] = tap.pcp_chunks
        if tap.pcp_telemetry is not None:
            result["pcp_telemetry"] = tap.pcp_telemetry
    if ms_selections is not None:
        result["ms_selections"] = ms_selections
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Env-lifecycle helper (the one loop convenience the repo provides)
# ─────────────────────────────────────────────────────────────────────────────
def iter_task_envs(episodes):
    """Yield (env, task_episodes) grouped by (suite, task_idx); closes each env in finally."""
    key = lambda e: (e["suite"], e["task_idx"])
    for _, grp in groupby(sorted(episodes, key=key), key=key):
        grp = list(grp)
        env = make_env(grp[0]["bddl_path"])
        try:
            yield env, grp
        finally:
            env.close()
