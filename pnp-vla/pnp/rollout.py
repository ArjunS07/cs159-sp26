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
from .pnp import (PnPRecorder, _pnp_seed_perturb, multi_policy_select, multi_sample_select,
                  summarize_probe_diagnostics)
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


def candidate_chunk_noise_seed(ep_seed: int, chunk_idx: int, candidate_idx: int) -> int:
    """Stable candidate seed; slot 0 retains the ordinary rollout's chunk seed."""
    if int(candidate_idx) == 0:
        return chunk_noise_seed(ep_seed, chunk_idx)
    b = hashlib.md5(
        f"{int(ep_seed)}:{int(chunk_idx)}:candidate:{int(candidate_idx)}".encode()).digest()
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


def candidate_action_disagreement(actions, action_dim: int = ADIM,
                                  gripper_dim: int = 6) -> dict:
    """Pairwise diversity metrics for clean, non-perturbed candidate action chunks.

    Two-candidate outputs preserve the original definitions exactly. With three or more
    candidates, scalar metrics average over every unordered candidate pair; maxima cover all
    pairs. Raw candidate chunks are stored separately when that sink is enabled.
    """
    if len(actions) < 2:
        raise ValueError("candidate disagreement requires at least two actions")
    arrays = [action.squeeze(0).detach().float().cpu().numpy() for action in actions]
    horizon = min(len(array) for array in arrays)
    arrays = [array[:horizon, :action_dim] for array in arrays]
    pair_metrics = []
    for left in range(len(arrays)):
        for right in range(left + 1, len(arrays)):
            a0, a1 = arrays[left], arrays[right]
            delta = a0 - a1
            per_action_l2 = np.linalg.norm(delta, axis=1)
            scale = np.mean((np.linalg.norm(a0, axis=1) + np.linalg.norm(a1, axis=1)) / 2)
            flat0, flat1 = a0.reshape(-1), a1.reshape(-1)
            cosine_denominator = np.linalg.norm(flat0) * np.linalg.norm(flat1)
            cosine = (float(np.dot(flat0, flat1) / cosine_denominator)
                      if cosine_denominator > 0 else None)
            gripper_disagreement = None
            if gripper_dim < action_dim:
                gripper_disagreement = float(np.mean(
                    (a0[:, gripper_dim] > 0) != (a1[:, gripper_dim] > 0)))
            pair_metrics.append({
                "action_l2_mean": float(per_action_l2.mean()),
                "action_l2_max": float(per_action_l2.max()),
                "first_action_l2": float(per_action_l2[0]),
                "action_abs_mean": float(np.abs(delta).mean()),
                "action_l2_normalized": float(per_action_l2.mean() / max(scale, 1e-12)),
                "action_cosine": cosine,
                "gripper_sign_disagreement": gripper_disagreement,
            })

    def _mean(name):
        values = [item[name] for item in pair_metrics if item[name] is not None]
        return float(np.mean(values)) if values else None

    return {
        "n_candidates": len(arrays),
        "n_candidate_pairs": len(pair_metrics),
        "action_l2_mean": _mean("action_l2_mean"),
        "action_l2_max": float(max(item["action_l2_max"] for item in pair_metrics)),
        "first_action_l2": _mean("first_action_l2"),
        "action_abs_mean": _mean("action_abs_mean"),
        "action_l2_normalized": _mean("action_l2_normalized"),
        "action_cosine": _mean("action_cosine"),
        "gripper_sign_disagreement": _mean("gripper_sign_disagreement"),
    }


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
                config: RolloutConfig | None = None, *, candidate_bundles=None):
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
    if candidate_bundles is not None:
        candidate_bundles = list(candidate_bundles)
        if not multisample or len(candidate_bundles) != int(config.num_samples):
            raise ValueError(
                "candidate_bundles requires num_samples equal to the number of candidates")
        if any(len(bundle) != 4 for bundle in candidate_bundles):
            raise ValueError(
                "each candidate bundle must be (label, policy, preprocess, postprocess)")

    recorder = PnPRecorder()
    tap = build_tap(config, recorder, device, adim)
    runtime_policies = ([bundle[1] for bundle in candidate_bundles]
                        if candidate_bundles is not None else [policy])
    unique_policies = list({id(item): item for item in runtime_policies}.values())
    original_n_action_steps = {
        id(runtime_policy): (
            hasattr(runtime_policy.config, "n_action_steps"),
            getattr(runtime_policy.config, "n_action_steps", None))
        for runtime_policy in unique_policies}
    if config.n_action_steps is not None:
        if config.n_action_steps > chunk_size:
            raise ValueError(
                f"n_action_steps={config.n_action_steps} exceeds chunk_size={chunk_size}")
        for runtime_policy in unique_policies:
            runtime_policy.config.n_action_steps = int(config.n_action_steps)
    for runtime_policy in unique_policies:
        runtime_policy.model._pnp.num_steps = config.num_inference_steps

    ep_seed = episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
    torch.manual_seed(ep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ep_seed)
    _pnp_seed_perturb(ep_seed)                           # isolated perturbation stream

    _sampler.set_strategy(model, tap)
    for runtime_policy in unique_policies:
        runtime_policy.model._pnp.vf_evals = 0
    recorder.new_episode(meta={k: ep.get(k) for k in ("suite", "task_idx", "ep_idx")})

    est_chunks = max(1, round(max_steps / chunk_size))
    queue, ci = [], 0
    queue_postprocess = postprocess
    executed_actions, robot_states, chunk_boundary_actions, chunk_noise_seeds = [], [], [], []
    generated_chunks = [] if config.save_generated_chunks else None
    candidate_generated_chunks = (
        [] if config.save_generated_chunks and candidate_bundles is not None else None)
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
    lead = max(1, int(config.render_lead))

    def _render_next(needed: bool) -> None:
        """Arm/disarm rendering for the NEXT env.step, whose obs is consumed if `needed`."""
        if skipping:
            set_camera_observables(env, needed)

    try:
        env.reset()
        for runtime_policy in unique_policies:
            runtime_policy.reset()
        obs = env.set_init_state(ep["init_state"])
        for wait_step in range(NUM_STEPS_WAIT):
            # Only the final settling observation feeds the first policy call, but the cameras
            # need `lead` steps of warm-up before they return a freshly rendered frame.
            _render_next(wait_step >= NUM_STEPS_WAIT - lead)
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        for step in range(max_steps):
            if not queue:
                for runtime_policy in unique_policies:
                    runtime_policy.model._pnp.chunk_pos = min(ci / est_chunks, 1.0)
                cns = chunk_noise_seed(ep_seed, ci)
                chunk_noise_seeds.append(cns)
                noise = _draw_chunk_noise(policy, device, cns)
                inference_t0 = time.perf_counter()
                if candidate_bundles is not None:
                    candidate_inputs, candidate_noises, perturb_seeds = [], [], []
                    for candidate_index, (_, candidate_policy, candidate_preprocess,
                                          _) in enumerate(candidate_bundles):
                        candidate_inputs.append((
                            candidate_policy,
                            candidate_preprocess(obs_to_policy(obs, task_desc))))
                        candidate_seed = candidate_chunk_noise_seed(
                            ep_seed, ci, candidate_index)
                        candidate_noises.append(
                            _draw_chunk_noise(candidate_policy, device, candidate_seed))
                        perturb_seeds.append(candidate_seed)
                    detailed_selection = config.selection_uncertainty_horizon is not None
                    selection = multi_policy_select(
                        candidate_inputs, candidate_noises, tuple(config.ms_probe_steps),
                        num_iterations=config.pnp_k, perturb_seeds=perturb_seeds,
                        uncertainty_horizon=config.selection_uncertainty_horizon,
                        return_details=detailed_selection)
                    chunk, chosen, cand_u, candidate_actions = selection[:4]
                    candidate_profiles = selection[4] if detailed_selection else None
                    labels = [str(bundle[0]) for bundle in candidate_bundles]
                    ms_selections.append({
                        "chunk_idx": ci, "chosen": int(chosen), "cand_u": cand_u,
                        "labels": labels, "chosen_label": labels[chosen],
                        "action_disagreement": candidate_action_disagreement(
                            candidate_actions, action_dim=adim),
                        **({"candidate_profiles": candidate_profiles,
                            "selection_uncertainty_horizon":
                                int(config.selection_uncertainty_horizon)}
                           if candidate_profiles is not None else {})})
                    if candidate_generated_chunks is not None:
                        candidate_generated_chunks.append(np.stack([
                            action.squeeze(0).detach().float().cpu().numpy()
                            for action in candidate_actions]))
                    queue_postprocess = candidate_bundles[chosen][3]
                elif multisample:
                    batch = preprocess(obs_to_policy(obs, task_desc))
                    def _noise_of(si, _ci=ci):
                        return _draw_chunk_noise(policy, device,
                                                 chunk_noise_seed(ep_seed, _ci * 1000 + si))
                    detailed_selection = config.selection_uncertainty_horizon is not None
                    selection = multi_sample_select(
                        policy, batch, ep_seed, ci, config.num_samples,
                        tuple(config.ms_probe_steps), _noise_of,
                        num_iterations=config.pnp_k,
                        uncertainty_horizon=config.selection_uncertainty_horizon,
                        return_details=detailed_selection)
                    chunk, chosen, cand_u = selection[:3]
                    candidate_profiles = selection[3] if detailed_selection else None
                    ms_selections.append({
                        "chunk_idx": ci, "chosen": int(chosen), "cand_u": cand_u,
                        **({"candidate_profiles": candidate_profiles,
                            "selection_uncertainty_horizon":
                                int(config.selection_uncertainty_horizon)}
                           if candidate_profiles is not None else {})})
                    queue_postprocess = postprocess
                else:
                    batch = preprocess(obs_to_policy(obs, task_desc))
                    with torch.no_grad():
                        chunk = policy.predict_action_chunk(batch, noise=noise)
                    queue_postprocess = postprocess
                full_arr = chunk.squeeze(0).detach().cpu().numpy()
                if generated_chunks is not None:
                    generated_chunks.append(full_arr.copy())
                # Do not silently consult policy.config here: historical rollout IDs with an
                # omitted horizon mean "execute the full generated chunk". Corrected/new drivers
                # must set n_action_steps explicitly so resume cannot mix old and new protocols.
                execution_horizon = config.n_action_steps
                arr = (full_arr if execution_horizon is None
                       else full_arr[:int(execution_horizon)])
                if execution_horizon is not None and len(arr) != execution_horizon:
                    raise ValueError(
                        f"policy returned only {len(full_arr)} actions, cannot execute "
                        f"n_action_steps={execution_horizon}")
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
            action = queue_postprocess(action)
            if isinstance(action, torch.Tensor):
                a = action.squeeze(0).detach().cpu().numpy()
            else:
                a = np.asarray(action).squeeze(0)
            executed_actions.append(np.asarray(a).flatten()[:ADIM].copy())
            if frames is not None:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            robot_states.append(np.concatenate([
                obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]).copy())
            # The queue empties in `len(queue)` more steps, and the step that empties it returns
            # the obs the next re-plan consumes. Arm the cameras `lead` steps ahead of it.
            _render_next(len(queue) < lead)
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
        for runtime_policy in unique_policies:
            existed, original = original_n_action_steps[id(runtime_policy)]
            if existed:
                runtime_policy.config.n_action_steps = original
            elif hasattr(runtime_policy.config, "n_action_steps"):
                delattr(runtime_policy.config, "n_action_steps")

    elapsed = time.time() - t0
    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    n_steps = step + 1
    recorder.close_episode(success, n_steps)
    inst = compute_instability(executed_actions, chunk_boundary_actions)

    vf_evals = sum(int(runtime_policy.model._pnp.vf_evals)
                   for runtime_policy in unique_policies)
    if vf_evals == 0:                                   # vanilla path (orig sampler doesn't count)
        vf_evals = (config.num_inference_steps or policy.config.num_inference_steps) * max(ci, 1)

    # video sink: encode iff configured for this outcome ('all', or 'failures_only' on failure)
    video_bytes = None
    if config.video != "off" and frames:
        if config.video == "all" or (config.video == "failures_only" and not success):
            video_bytes = _encode_mp4(frames)

    probe_diagnostics = summarize_probe_diagnostics(
        recorder.episodes[-1] if recorder.episodes else None)
    if ms_selections and ms_selections[0].get("candidate_profiles"):
        chosen_profiles = [
            selection["candidate_profiles"][selection["chosen"]]
            for selection in ms_selections]
        for source, destination in (
                ("u10", "u_first10"), ("u20", "u_first20"),
                ("u_full", "u_full"), ("contraction10", "contraction_first10"),
                ("contraction20", "contraction_first20"),
                ("contraction_full", "contraction_full")):
            values = [profile.get(source) for profile in chosen_profiles]
            values = [float(value) for value in values
                      if value is not None and np.isfinite(value)]
            if values:
                probe_diagnostics[destination] = float(np.mean(values))

    result = dict(
        success=success, n_steps=n_steps, elapsed_s=elapsed, status=status, error_msg=error_msg,
        terminated_reason=terminated_reason, started_at=started_at, finished_at=finished_at,
        inference_ms_total=inference_ms_total,
        nan_action_count=nan_count, n_chunks=ci, n_vf_evals=vf_evals, chunk_size=chunk_size,
        episode_seed=ep_seed, perturb_seed=ep_seed,
        chunk_noise_seeds=chunk_noise_seeds, instability=inst,
        recorder_episode=recorder.episodes[-1] if recorder.episodes else None,
        probe_diagnostics=probe_diagnostics,
        trajectory=dict(
            actions=np.asarray(executed_actions, dtype=np.float32),
            robot_state=np.asarray(robot_states, dtype=np.float32),
        ) if config.save_trajectory else None,
        generated_chunks=(dict(
            chunks=np.asarray(generated_chunks, dtype=np.float32),
            **({"candidate_chunks": np.asarray(
                candidate_generated_chunks, dtype=np.float32)}
               if candidate_generated_chunks is not None else {}))
            if generated_chunks is not None else None),
        obs_frames=frames if config.save_observations else None,
        video=video_bytes,
    )
    if tap is not None:
        if tap.save_pcp:
            result["pcp_chunks"] = tap.pcp_chunks
        if tap.pcp_telemetry is not None:
            result["pcp_telemetry"] = tap.pcp_telemetry
        if tap.refinement_gate_telemetry is not None:
            result["refinement_gate_telemetry"] = tap.refinement_gate_telemetry
    if ms_selections is not None:
        result["ms_selections"] = ms_selections
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Env-lifecycle helper (the one loop convenience the repo provides)
# ─────────────────────────────────────────────────────────────────────────────
def _sim_state(env):
    """Flattened MuJoCo state of a (possibly wrapped) robosuite env, or None."""
    target = getattr(env, "env", env)
    sim = getattr(target, "sim", None) or getattr(env, "sim", None)
    if sim is None:
        return None
    try:
        return np.asarray(sim.get_state().flatten()).copy()
    except Exception:
        return None


def measure_render_timing(env, ep, n_steps=8, settle=4):
    """Decide fresh-vs-stale by COST, not by pixels.

    This stack renders a different (internally consistent) image on every reset even though
    physics is bit-identical, so no cross-run pixel comparison can act as an oracle. Timing can:
    a cached observable returns in microseconds, a real render of two 360x360 cameras costs
    ~230 ms on Mesa software EGL. So if the step on which the cameras are re-enabled costs about
    as much as an always-rendered step, it rendered; if it costs about as much as a
    cameras-off step, it served a cache and skipping is unsafe at that lead.
    """
    def timed_steps(count):
        out = []
        for _ in range(count):
            start = time.perf_counter()
            env.step(LIBERO_DUMMY_ACTION)
            out.append((time.perf_counter() - start) * 1000.0)
        return out

    set_camera_observables(env, True)
    env.reset()
    env.set_init_state(ep["init_state"])
    timed_steps(settle)
    on = timed_steps(n_steps)

    set_camera_observables(env, False)
    timed_steps(settle)
    off = timed_steps(n_steps)

    # Re-enable, then time the very next step (lead 1) and the one after (lead 2).
    set_camera_observables(env, True)
    after_enable = timed_steps(2)
    set_camera_observables(env, True)

    on_ms, off_ms = float(np.median(on)), float(np.median(off))
    midpoint = (on_ms + off_ms) / 2
    print(f"cameras on   (median): {on_ms:7.1f} ms/step")
    print(f"cameras off  (median): {off_ms:7.1f} ms/step")
    print(f"render cost          : {on_ms - off_ms:7.1f} ms/step "
          f"({100 * (on_ms - off_ms) / max(on_ms, 1e-9):.0f}% of a step)")
    verdicts = []
    for offset, value in enumerate(after_enable, start=1):
        rendered = value > midpoint
        verdicts.append(rendered)
        print(f"step {offset} after re-enable: {value:7.1f} ms -> "
              f"{'RENDERED' if rendered else 'served cache'}")
    lead = next((i for i, rendered in enumerate(verdicts, start=1) if rendered), None)
    if lead is None:
        print("=> no step after re-enabling paid for a render; skipping is unsafe here")
    else:
        print(f"=> a re-enabled camera renders on step {lead}, so render_lead = {lead}")
    return {"on_ms": on_ms, "off_ms": off_ms, "after_enable_ms": after_enable, "lead": lead}


def diagnose_frame_mismatch(env, ep, n_steps=6, camera="agentview_image", n_replays=3):
    """Locate WHY two replays render differently when their physics is identical.

    A constant, large, structured pixel difference is not rasteriser noise. This distinguishes the
    candidate explanations:

      * only the FIRST replay differs -> cold render context / uninitialised framebuffer after
        make_env; benign, and fixed by discarding one warm-up replay.
      * a frame matches another replay at a DIFFERENT step -> staleness or an off-by-one.
      * a frame matches a flipped copy -> orientation, i.e. the [::-1, ::-1] convention.
      * none of the above -> genuinely different scene content (materials, object variants).
    """
    def replay():
        set_camera_observables(env, True)
        env.reset()
        env.set_init_state(ep["init_state"])
        out = []
        for _ in range(n_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            value = obs.get(camera)
            out.append(None if value is None else np.asarray(value, dtype=np.int16).copy())
        return out

    runs = [replay() for _ in range(n_replays)]
    d = lambda a, b: (None if a is None or b is None else int(np.abs(a - b).max()))

    print("pairwise max |dpixel| at each step (replay i vs replay j)")
    header = "".join(f"{f'{i}v{j}':>8}" for i in range(n_replays) for j in range(i + 1, n_replays))
    print(f"{'step':>5}{header}")
    for step in range(n_steps):
        cells = "".join(f"{str(d(runs[i][step], runs[j][step])):>8}"
                        for i in range(n_replays) for j in range(i + 1, n_replays))
        print(f"{step:>5}{cells}")

    later_agree = all(d(runs[i][s], runs[j][s]) == 0
                      for s in range(n_steps) for i in range(1, n_replays)
                      for j in range(i + 1, n_replays))
    print(f"\nreplays after the first agree exactly: {later_agree}")
    if later_agree and n_replays > 2 and any(
            d(runs[0][s], runs[1][s]) not in (0, None) for s in range(n_steps)):
        print("=> only the FIRST replay differs: cold render context. Discard one warm-up replay "
              "and rendering is reproducible.")

    first, second = runs[0][n_steps - 1], runs[1][n_steps - 1]
    if first is not None and second is not None:
        best = min(((d(first, runs[1][s]), s) for s in range(n_steps)), key=lambda t: t[0])
        print(f"closest match for replay0 step {n_steps - 1} is replay1 step {best[1]} "
              f"(|dpixel|max={best[0]})")
        print(f"vs vertical flip  |dpixel|max={d(first, second[::-1])}")
        print(f"vs 180 deg flip   |dpixel|max={d(first, second[::-1, ::-1])}")
        print(f"mean pixel value  replay0={first.mean():.2f}  replay1={second.mean():.2f}")
    set_camera_observables(env, True)
    return runs


def measure_replay_determinism(env, ep, n_steps=12, camera="agentview_image"):
    """Separate PHYSICS drift from RENDER drift across two identical replays.

    A bit-exact frame comparison is only a valid oracle if replaying an episode reproduces itself.
    When it does not, the question is which layer moved:

      * physics identical, pixels differ  -> the rasteriser is nondeterministic (Mesa software EGL
        does this). Harmless for behaviour; compare frames with a tolerance, not equality.
      * physics differs                   -> reset()/set_init_state() is not restoring state, which
        undermines any paired-arm comparison at the same identity, not just this optimisation.

    Returns {'state_max', 'image_max', 'image_frac', 'first_state_drift', 'first_image_drift'}.
    """
    def replay():
        set_camera_observables(env, True)
        env.reset()
        env.set_init_state(ep["init_state"])
        states, images = [], []
        for _ in range(n_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            states.append(_sim_state(env))
            value = obs.get(camera)
            images.append(None if value is None else np.asarray(value, dtype=np.int16).copy())
        return states, images

    states_a, images_a = replay()
    states_b, images_b = replay()

    state_max, image_max, image_frac = 0.0, 0, 0.0
    first_state, first_image = None, None
    print(f"{'step':>5}{'|dstate|max':>14}{'|dpixel|max':>13}{'pixels differing':>18}")
    for step in range(n_steps):
        sa, sb = states_a[step], states_b[step]
        ds = float(np.abs(sa - sb).max()) if sa is not None and sb is not None else float("nan")
        ia, ib = images_a[step], images_b[step]
        if ia is None or ib is None:
            di, frac = -1, float("nan")
        else:
            diff = np.abs(ia - ib)
            di, frac = int(diff.max()), float((diff > 0).mean())
        print(f"{step:>5}{ds:>14.3e}{di:>13}{frac:>17.2%}")
        if ds == ds and ds > 0 and first_state is None:
            first_state = step
        if di > 0 and first_image is None:
            first_image = step
        state_max = max(state_max, 0.0 if ds != ds else ds)
        image_max, image_frac = max(image_max, di), max(image_frac, 0.0 if frac != frac else frac)

    print(f"\nphysics: max |dstate| = {state_max:.3e}"
          + (f", first drift at step {first_state}" if first_state is not None else " (identical)"))
    print(f"render : max |dpixel| = {image_max}, up to {image_frac:.2%} of pixels"
          + (f", first drift at step {first_image}" if first_image is not None else " (identical)"))
    if state_max == 0.0 and image_max > 0:
        print("=> physics is deterministic, the rasteriser is not. Compare frames with a "
              "tolerance; behaviour is unaffected.")
    elif state_max > 0.0:
        print("=> the SIMULATOR diverges between replays. This affects paired comparisons at the "
              "same identity, not just render skipping.")
    return {"state_max": state_max, "image_max": image_max, "image_frac": image_frac,
            "first_state_drift": first_state, "first_image_drift": first_image}


def measure_render_lag(env, ep, off_at=1, on_at=8, n_steps=12, camera="agentview_image"):
    """Measure how many steps a re-enabled camera needs before it renders fresh again.

    Robosuite serves the last cached value for a disabled observable, so re-enabling does not
    guarantee a current frame. Replays one fixed dummy-action sequence twice -- always rendering,
    then disabled over [off_at, on_at) -- and reports the first step after `on_at` whose image
    matches the always-rendered reference. That count is `RolloutConfig.render_lead`.

    Needs no policy, so it costs seconds. Returns the measured lead, or None if it never matches.
    """
    def replay(off=None, on=None):
        set_camera_observables(env, True)
        env.reset()
        env.set_init_state(ep["init_state"])
        out = []
        for step in range(n_steps):
            if off is not None and step == off:
                set_camera_observables(env, False)
            if on is not None and step == on:
                set_camera_observables(env, True)
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            value = obs.get(camera)
            out.append(None if value is None else np.asarray(value).copy())
        return out

    # CONTROL FIRST. Everything below compares two separate replays, so it is meaningless unless
    # an untouched replay reproduces itself frame for frame. Without this check a non-reproducible
    # env looks exactly like a stale-frame bug.
    reference = replay()
    control = replay()
    def _delta(a, b):
        if a is None or b is None:
            return None
        return int(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max())

    # The control establishes the NOISE FLOOR. Mesa's rasteriser is not bit-deterministic, so
    # bit-equality is the wrong oracle -- "fresh" means within the floor, exactly as
    # assert_pnp_noop measures a vanilla-vs-vanilla floor before judging the probe.
    floor = max((d for d in (_delta(a, b) for a, b in zip(reference, control))
                 if d is not None), default=0)
    tolerance = max(3 * floor, 2)
    print(f"control: max |dpixel| between two untouched replays = {floor} "
          f"-> fresh means within {tolerance}")

    probed = replay(off=off_at, on=on_at)
    set_camera_observables(env, True)

    lead = None
    print(f"\n{'step':>5}{'state':>10}{'d(reference)':>14}{'d(pre-disable)':>16}{'verdict':>10}")
    # The probed run's OWN last rendered frame before the disable window.
    stale = probed[off_at - 1] if off_at >= 1 else None
    for step in range(on_at, n_steps):
        image = probed[step]
        d_ref, d_stale = _delta(image, reference[step]), _delta(image, stale)
        fresh = d_ref is not None and d_ref <= tolerance
        print(f"{step:>5}{'missing' if image is None else 'present':>10}"
              f"{'-' if d_ref is None else d_ref:>14}{'-' if d_stale is None else d_stale:>16}"
              f"{'fresh' if fresh else 'STALE':>10}")
        if fresh and lead is None:
            lead = step - on_at + 1
    if lead is None:
        print("\nnever came within the noise floor -- rendering cannot be skipped this way")
    else:
        print(f"\nmeasured render_lead = {lead}")
    return lead


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
    runs = []
    # Two always-rendering runs first: their disagreement is the nondeterminism FLOOR. Neither the
    # rasteriser nor bf16 flow matching is bit-reproducible, so bit-equality would fail for
    # reasons that have nothing to do with skipping. Same shape as assert_pnp_noop.
    for skip in (False, False, True):
        frames: list = []
        result = run_episode(env, ep, policy, _record(frames), postprocess, device,
                             replace(base, skip_unused_renders=skip))
        runs.append((frames, result))
    (base_frames, base_result), (ctrl_frames, _), (skip_frames, skip_result) = runs

    def _gap(left, right):
        n = min(len(left), len(right))
        if n == 0:
            return float("nan"), None
        deltas = [float(np.abs(np.asarray(left[i], dtype=np.float64)
                               - np.asarray(right[i], dtype=np.float64)).max())
                  for i in range(n)]
        worst = max(deltas)
        return worst, (deltas.index(worst) if worst > 0 else None)

    floor, _ = _gap(base_frames, ctrl_frames)
    gap, first_bad = _gap(base_frames, skip_frames)
    tolerance = max(3 * floor, 1e-3) if floor == floor else float("nan")
    same_count = len(base_frames) == len(skip_frames)
    ok = same_count and gap == gap and gap <= tolerance

    msg = (f"assert_render_skip_equivalent: decisions={len(base_frames)}/{len(skip_frames)}"
           f"  image_gap={gap:.4g} (floor={floor:.4g}, tol={tolerance:.4g})"
           + (f" worst_at={first_bad}" if first_bad is not None else "")
           + f"  success {base_result['success']}->{skip_result['success']}"
           f"  n_steps {base_result['n_steps']}->{skip_result['n_steps']}"
           f"  elapsed {base_result['elapsed_s']:.1f}s->{skip_result['elapsed_s']:.1f}s "
           f"({base_result['elapsed_s'] / max(skip_result['elapsed_s'], 1e-9):.2f}x)"
           f"  -> {'OK' if ok else 'FAIL'}")
    if not ok and raise_on_fail:
        raise AssertionError(msg)
    print(msg)
    return ok


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
