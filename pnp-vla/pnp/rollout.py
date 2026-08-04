"""Rollout PRIMITIVES (Dedup #2). No experiment/driver loops — those live in the notebooks.

`run_episode(env, ep, policy, preprocess, postprocess, device, config)` runs ONE episode: it builds the
`RolloutTap` from `config`, owns the per-(episode,chunk) noise generator so chunk i's initial
noise is byte-identical across every method, and returns a plain result dict whose contents
depend on which sinks were enabled. `iter_task_envs(episodes)` yields `(env, task_eps)`
handling the OffScreenRenderEnv lifecycle. Nothing here writes to the store — the notebook
loop calls `store.log_result(...)` with the result.
"""
from __future__ import annotations

import hashlib
import datetime as dt
import time
from itertools import groupby

import numpy as np
import torch

from .config import (ADIM, LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT, PERTURB_SEED_MASK,
                     VIDEO_FPS, RolloutConfig)
from .libero_env import make_env, obs_to_policy
from .pnp import PnPRecorder, _pnp_seed_perturb, multi_sample_select
from .tap import RolloutTap, BatchedRolloutTap
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


def _encode_mp4(frames, fps: int = VIDEO_FPS) -> bytes:
    """Encode a list of HxWx3 uint8 frames to mp4 bytes (imageio-ffmpeg)."""
    import os
    import tempfile
    import imageio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        imageio.mimwrite(path, [np.ascontiguousarray(x) for x in frames], fps=fps,
                         macro_block_size=1)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.remove(path)


# ─────────────────────────────────────────────────────────────────────────────
# Instability metrics
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
def _run_episode_serial(env, ep, policy, preprocess, postprocess, device,
                        config: RolloutConfig | None = None):
    """Run one episode under `config`. Returns a result dict (outcome, metrics, trajectory,
    recorder episode, and sink outputs — pcp_chunks / pcp_telemetry / ms_selections).

    Store/DB writes are the notebook's job (via store.log_result)."""
    config = config or RolloutConfig()
    model = policy.model
    adim = model._pnp.action_dim
    task_desc = ep["task_desc"]
    max_steps = ep["max_steps"]
    chunk_size = policy.config.chunk_size
    multisample = config.num_samples is not None

    recorder = PnPRecorder()
    tap = build_tap(config, recorder, device, adim)
    model._pnp.num_steps = config.num_inference_steps   # extra_steps override (None = default)

    ep_seed = episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
    torch.manual_seed(ep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ep_seed)
    _pnp_seed_perturb(ep_seed)                           # isolated perturbation stream

    _sampler.set_strategy(model, tap)
    model._pnp.vf_evals = 0
    recorder.new_episode(meta={k: ep.get(k) for k in ("suite", "task_idx", "ep_idx")})

    est_chunks = max(1, round(max_steps / chunk_size))
    queue, ci = [], 0
    executed_actions, robot_states, chunk_boundary_actions, chunk_noise_seeds = [], [], [], []
    generated_chunks = [] if config.save_generated_chunks else None
    ms_selections = [] if multisample else None
    # capture agentview frames when either sink wants them (obs_frames OR a video)
    frames = [] if (config.save_observations or config.video != "off") else None
    nan_count = 0
    success = False
    status, error_msg = "completed", None
    terminated_reason = "max_steps"
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    inference_ms_total = 0.0
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
                model._pnp.chunk_pos = min(ci / est_chunks, 1.0)
                cns = chunk_noise_seed(ep_seed, ci)
                chunk_noise_seeds.append(cns)
                noise = _draw_chunk_noise(policy, device, cns)
                batch = preprocess(obs_to_policy(obs, task_desc))
                inference_t0 = time.perf_counter()
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
                if generated_chunks is not None:
                    generated_chunks.append(arr.copy())
                inference_ms_total += (time.perf_counter() - inference_t0) * 1000.0
                queue = [arr[i].copy() for i in range(arr.shape[0])]
                chunk_boundary_actions.append(np.asarray(queue[0]).flatten()[:ADIM].copy())
                ci += 1
            a = queue.pop(0)
            if not np.all(np.isfinite(a)):
                nan_count += 1
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            # predict_action_chunk returns normalized policy-space actions. LIBERO must receive
            # the official postprocessor's environment-space action, exactly as in LeRobot's
            # select_action evaluation path and the pre-refactor notebooks.
            action = torch.as_tensor(a, device=device).unsqueeze(0)
            action = postprocess(action)
            if isinstance(action, torch.Tensor):
                a = action.squeeze(0).detach().cpu().numpy()
            else:
                a = np.asarray(action).squeeze(0)
            executed_actions.append(np.asarray(a).flatten()[:ADIM].copy())
            if frames is not None:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            robot_states.append(np.concatenate([
                obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]).copy())
            obs, _, done, _ = env.step(a)
            if env.check_success():
                success = True
                terminated_reason = "success"
                break
            if done:
                terminated_reason = "env_done"
                break
    except Exception as e:                              # log errored episodes, don't drop them
        status, error_msg = "errored", f"{type(e).__name__}: {e}"
        terminated_reason = "error"

    elapsed = time.time() - t0
    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    n_steps = step + 1
    recorder.close_episode(success, n_steps)
    inst = compute_instability(executed_actions, chunk_boundary_actions)

    vf_evals = int(model._pnp.vf_evals)
    if vf_evals == 0:                                   # vanilla path (orig sampler doesn't count)
        vf_evals = (config.num_inference_steps or policy.config.num_inference_steps) * max(ci, 1)

    # video sink: encode iff configured for this outcome ('all', or 'failures_only' on failure)
    video_bytes = None
    if config.video != "off" and frames:
        if config.video == "all" or (config.video == "failures_only" and not success):
            video_bytes = _encode_mp4(frames)

    result = dict(
        success=success, n_steps=n_steps, elapsed_s=elapsed, status=status, error_msg=error_msg,
        terminated_reason=terminated_reason, started_at=started_at, finished_at=finished_at,
        inference_ms_total=inference_ms_total,
        nan_action_count=nan_count, n_chunks=ci, n_vf_evals=vf_evals, chunk_size=chunk_size,
        episode_seed=ep_seed, perturb_seed=ep_seed,
        chunk_noise_seeds=chunk_noise_seeds, instability=inst,
        recorder_episode=recorder.episodes[-1] if recorder.episodes else None,
        trajectory=dict(
            actions=np.asarray(executed_actions, dtype=np.float32),
            robot_state=np.asarray(robot_states, dtype=np.float32),
        ) if config.save_trajectory else None,
        generated_chunks=(dict(chunks=np.asarray(generated_chunks, dtype=np.float32))
                          if generated_chunks is not None else None),
        obs_frames=frames if config.save_observations else None,
        video=video_bytes,
    )
    if tap is not None:
        if tap.save_pcp:
            result["pcp_chunks"] = tap.pcp_chunks
        if tap.pcp_telemetry is not None:
            result["pcp_telemetry"] = tap.pcp_telemetry
    if ms_selections is not None:
        result["ms_selections"] = ms_selections
    return result


def _stack_batches(items):
    """Recursively concatenate preprocessor outputs along their batch dimension."""
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(items, dim=0)
    if isinstance(first, dict):
        return {key: _stack_batches([item[key] for item in items]) for key in first}
    if isinstance(first, tuple):
        return tuple(_stack_batches([item[i] for item in items]) for i in range(len(first)))
    if isinstance(first, list):
        return [_stack_batches([item[i] for item in items]) for i in range(len(first))]
    raise TypeError(f"cannot batch preprocessor output of type {type(first).__name__}")


def run_episode_batch(envs, episodes, policy, preprocess, postprocess, device,
                      config: RolloutConfig | None = None):
    """Run independent environments with one shared, truly batched policy invocation.

    Results preserve input order. Multi-sample and learned-PCP selection intentionally use the
    established serial path until candidate/state branching is migrated.
    """
    config = config or RolloutConfig()
    if len(envs) != len(episodes) or not episodes:
        raise ValueError("envs and episodes must have the same non-zero length")
    if len(episodes) == 1:
        return [_run_episode_serial(envs[0], episodes[0], policy, preprocess,
                                    postprocess, device, config)]
    if config.num_samples is not None or config.correction_lambda is not None:
        reason = ("multi-sample selection" if config.num_samples is not None
                  else "learned PCP correction")
        print(f"[rollout] {reason} is not batch-enabled; using serial runner for this collection")
        return [_run_episode_serial(env, ep, policy, preprocess, postprocess, device, config)
                for env, ep in zip(envs, episodes)]

    model = policy.model
    adim, chunk_size = model._pnp.action_dim, policy.config.chunk_size
    model._pnp.num_steps = config.num_inference_steps
    seeds = [episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
             for ep in episodes]
    recorders = [PnPRecorder() for _ in episodes]
    for recorder, ep in zip(recorders, episodes):
        recorder.new_episode(meta={k: ep.get(k) for k in ("suite", "task_idx", "ep_idx")})
    states = []
    now = lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    for env, ep, seed in zip(envs, episodes, seeds):
        perturb_gen = torch.Generator(device=torch.device(device))
        perturb_gen.manual_seed(int(seed) ^ PERTURB_SEED_MASK)
        state = dict(obs=None, queue=[], ci=0, step=0, success=False, status="completed",
                     error_msg=None, terminated_reason="max_steps", actions=[], robot_states=[],
                     boundaries=[], noise_seeds=[],
                     generated=[] if config.save_generated_chunks else None,
                     frames=[] if (config.save_observations or config.video != "off") else None,
                     nan_count=0, inference_ms=0.0, vf_evals=0, perturb_gen=perturb_gen,
                     started_at=now(), t0=time.time(), done=False)
        try:
            env.reset()
            state["obs"] = env.set_init_state(ep["init_state"])
            for _ in range(NUM_STEPS_WAIT):
                state["obs"], _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        except Exception as exc:
            state.update(status="errored", error_msg=f"{type(exc).__name__}: {exc}",
                         terminated_reason="error", done=True)
        states.append(state)
    policy.reset()

    while any(not state["done"] for state in states):
        infer_ids = [i for i, state in enumerate(states)
                     if not state["done"] and not state["queue"]]
        if infer_ids:
            batches, noises, positions = [], [], []
            for i in infer_ids:
                ep, state = episodes[i], states[i]
                cseed = chunk_noise_seed(seeds[i], state["ci"])
                state["noise_seeds"].append(cseed)
                noises.append(_draw_chunk_noise(policy, device, cseed))
                batches.append(preprocess(obs_to_policy(state["obs"], ep["task_desc"])))
                positions.append(min(state["ci"] / max(1, round(ep["max_steps"] / chunk_size)), 1.0))
            tap = (BatchedRolloutTap(config, [recorders[i] for i in infer_ids],
                                     [states[i]["perturb_gen"] for i in infer_ids], device, adim)
                   if config.has_probe else None)
            _sampler.set_strategy(model, tap)
            model._pnp.chunk_pos = positions
            before_vf = model._pnp.vf_evals
            started = time.perf_counter()
            try:
                with torch.no_grad():
                    chunks = policy.predict_action_chunk(
                        _stack_batches(batches), noise=torch.cat(noises, dim=0))
                infer_ms = (time.perf_counter() - started) * 1000.0
                vf_delta = model._pnp.vf_evals - before_vf
                arrays = chunks.detach().cpu().numpy()
                for lane, i in enumerate(infer_ids):
                    state = states[i]; arr = arrays[lane]
                    state["inference_ms"] += infer_ms
                    state["vf_evals"] += vf_delta or (
                        config.num_inference_steps or policy.config.num_inference_steps)
                    state["queue"] = [row.copy() for row in arr]
                    state["boundaries"].append(arr[0, :adim].copy())
                    if state["generated"] is not None: state["generated"].append(arr.copy())
                    state["ci"] += 1
                    if tap is not None and tap.save_pcp:
                        # Tap instances are inference-call-local; transfer their lane output.
                        target = state.setdefault("pcp_chunks", [])
                        for chunk in tap.pcp_chunks[lane]:
                            chunk["chunk_idx"] = len(target)
                            target.append(chunk)
            except Exception as exc:
                for i in infer_ids:
                    states[i].update(status="errored", error_msg=f"{type(exc).__name__}: {exc}",
                                     terminated_reason="error", done=True)
                continue

        step_ids = [i for i, state in enumerate(states) if not state["done"]]
        raw = []
        for i in step_ids:
            a = states[i]["queue"].pop(0)
            if not np.all(np.isfinite(a)):
                states[i]["nan_count"] += 1
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            raw.append(torch.as_tensor(a, device=device).unsqueeze(0))
        processed = postprocess(torch.cat(raw, dim=0))
        if isinstance(processed, torch.Tensor): processed = processed.detach().cpu().numpy()
        processed = np.asarray(processed)
        for lane, i in enumerate(step_ids):
            env, ep, state = envs[i], episodes[i], states[i]
            a, obs = np.asarray(processed[lane]), state["obs"]
            try:
                state["actions"].append(a.flatten()[:adim].copy())
                if state["frames"] is not None:
                    state["frames"].append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                state["robot_states"].append(np.concatenate([
                    obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]).copy())
                state["step"] += 1
                state["obs"], _, env_done, _ = env.step(a)
                if env.check_success():
                    state.update(success=True, terminated_reason="success", done=True)
                elif env_done:
                    state.update(terminated_reason="env_done", done=True)
                elif state["step"] >= ep["max_steps"]:
                    state["done"] = True
            except Exception as exc:
                state.update(status="errored", error_msg=f"{type(exc).__name__}: {exc}",
                             terminated_reason="error", done=True)

    results = []
    for i, (ep, state, recorder) in enumerate(zip(episodes, states, recorders)):
        recorder.close_episode(state["success"], state["step"])
        frames = state["frames"]
        video = (_encode_mp4(frames) if frames and config.video != "off" and
                 (config.video == "all" or not state["success"]) else None)
        result = dict(success=state["success"], n_steps=state["step"],
            elapsed_s=time.time() - state["t0"], status=state["status"], error_msg=state["error_msg"],
            terminated_reason=state["terminated_reason"], started_at=state["started_at"],
            finished_at=now(), inference_ms_total=state["inference_ms"],
            nan_action_count=state["nan_count"], n_chunks=state["ci"],
            n_vf_evals=state["vf_evals"], chunk_size=chunk_size, episode_seed=seeds[i],
            perturb_seed=seeds[i], chunk_noise_seeds=state["noise_seeds"],
            instability=compute_instability(state["actions"], state["boundaries"]),
            recorder_episode=recorder.episodes[-1] if recorder.episodes else None,
            trajectory=dict(actions=np.asarray(state["actions"], dtype=np.float32),
                            robot_state=np.asarray(state["robot_states"], dtype=np.float32))
                       if config.save_trajectory else None,
            generated_chunks=(dict(chunks=np.asarray(state["generated"], dtype=np.float32))
                              if state["generated"] is not None else None),
            obs_frames=frames if config.save_observations else None, video=video)
        if config.save_pcp_features: result["pcp_chunks"] = state.get("pcp_chunks", [])
        results.append(result)
    return results


def run_episode(env, ep, policy, preprocess, postprocess, device,
                config: RolloutConfig | None = None):
    """Singleton compatibility wrapper around :func:`run_episode_batch`."""
    return run_episode_batch([env], [ep], policy, preprocess, postprocess, device, config)[0]


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
