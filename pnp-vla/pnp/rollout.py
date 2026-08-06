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

from .config import ADIM, LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT, VIDEO_FPS, RolloutConfig
from .libero_env import make_env, obs_to_policy, set_camera_observables
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
def run_episode(env, ep, policy, preprocess, postprocess, device,
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
    # Camera rendering is ~90% of a LIBERO step, and the policy reads an observation only at
    # chunk boundaries. Skip the rest -- but only when no sink needs every frame, and only if the
    # installed robosuite actually supports the toggle (otherwise the policy would get no image).
    needs_every_frame = frames is not None
    skipping = config.skip_unused_renders and not needs_every_frame
    if skipping:
        skipping = set_camera_observables(env, True)

    def _render_next(needed: bool) -> None:
        """Arm/disarm rendering for the NEXT env.step, whose obs is consumed if `needed`."""
        if skipping:
            set_camera_observables(env, needed)

    try:
        env.reset()
        policy.reset()
        obs = env.set_init_state(ep["init_state"])
        for wait_step in range(NUM_STEPS_WAIT):
            # Only the final settling observation feeds the first policy call.
            _render_next(wait_step == NUM_STEPS_WAIT - 1)
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
            # The queue just emptied => the next iteration re-plans from the obs this step
            # returns, so that one render is required. Every other render is discarded.
            _render_next(not queue)
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
    finally:
        # Leave the env as we found it; iter_task_envs reuses one env across rollouts.
        if skipping:
            set_camera_observables(env, True)

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


# ─────────────────────────────────────────────────────────────────────────────
# Env-lifecycle helper (the one loop convenience the repo provides)
# ─────────────────────────────────────────────────────────────────────────────
def assert_render_skip_equivalent(env, ep, policy, preprocess, postprocess, device,
                                  config: RolloutConfig | None = None, raise_on_fail=True):
    """Verify on the REAL simulator that skip_unused_renders changes nothing the policy sees.

    Unit tests cover the step-selection logic against a fake env, but they cannot cover the part
    that actually risks silence: robosuite caches observables, so re-enabling a disabled camera
    might return a stale frame. If it did, the policy would receive a one-chunk-old image and the
    success rate would quietly collapse -- no exception, no log line.

    Runs `ep` twice under identical seeds, once with skipping off and once on, recording exactly
    the tensors handed to the policy at each decision point (by wrapping `preprocess`, so
    run_episode is untouched). Images must be bit-identical; actions are reported but compared
    loosely because bf16 flow matching is not bit-reproducible across runs.
    """
    base = config or RolloutConfig()
    if base.save_observations or base.video != "off":
        raise ValueError("frame sinks force every render; test with them off")

    def _record(sink):
        def wrapped(raw):
            for key in ("observation.images.image", "observation.images.image2"):
                if key in raw:
                    sink.append(np.asarray(raw[key].detach().cpu()).copy())
            return preprocess(raw)
        return wrapped

    from dataclasses import replace
    runs = {}
    for label, skip in (("always", False), ("skipped", True)):
        frames: list = []
        result = run_episode(env, ep, policy, _record(frames), postprocess, device,
                             replace(base, skip_unused_renders=skip))
        runs[label] = (frames, result)

    always_frames, always = runs["always"]
    skipped_frames, skipped = runs["skipped"]
    n = min(len(always_frames), len(skipped_frames))
    identical = (len(always_frames) == len(skipped_frames)
                 and all(np.array_equal(always_frames[i], skipped_frames[i]) for i in range(n)))
    first_bad = next((i for i in range(n)
                      if not np.array_equal(always_frames[i], skipped_frames[i])), None)
    action_gap = float(np.abs(
        always["trajectory"]["actions"][:min(always["n_steps"], skipped["n_steps"])]
        - skipped["trajectory"]["actions"][:min(always["n_steps"], skipped["n_steps"])]
    ).max()) if always.get("trajectory") and skipped.get("trajectory") else float("nan")

    msg = (f"assert_render_skip_equivalent: decisions={len(always_frames)}/{len(skipped_frames)} "
           f"images_identical={identical}"
           + (f" first_mismatch_at={first_bad}" if first_bad is not None else "")
           + f"  success {always['success']}->{skipped['success']}"
           f"  n_steps {always['n_steps']}->{skipped['n_steps']}"
           f"  max|action delta|={action_gap:.2e}"
           f"  elapsed {always['elapsed_s']:.1f}s->{skipped['elapsed_s']:.1f}s "
           f"({always['elapsed_s'] / max(skipped['elapsed_s'], 1e-9):.2f}x)"
           f"  -> {'OK' if identical else 'FAIL'}")
    if not identical and raise_on_fail:
        raise AssertionError(msg)
    print(msg)
    return identical


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
