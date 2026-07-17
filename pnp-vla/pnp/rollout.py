"""ONE per-episode rollout engine (Dedup #2) + the explicit per-(episode,chunk) noise contract.

`run_episode` unifies run_episode_pnp / collect_episode / eval_episode. It OWNS the initial
diffusion noise: each chunk's noise is drawn from a dedicated per-episode generator keyed by
(episode_seed, chunk_idx) and passed explicitly to predict_action_chunk(noise=...), so chunk
i's noise is byte-identical across every method regardless of ambient global-RNG state or
trajectory divergence. The isolated P&P perturbation generator is seeded separately and never
touches this stream.

Drivers (run_controlled_slice, run_pro, pcp_collect, pcp_eval) live here too and are added in
phase 5; they build the strategy + experiment_runs row and call run_episode.

NOTE (verification item): all paths execute the RAW predict_action_chunk output (matching the
PCP collect/eval path). If the pi0.5 postprocessor turns out to be non-identity for actions,
the controlled-slice parity check will flag it.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from itertools import groupby

from .config import ADIM, DIM_NAMES, LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT, PCPConfig, RunConfig
from .libero_env import make_env, obs_to_policy
from .pnp import PNP_CONFIG, PNP_RECORDER, RecordStrategy, _pnp_seed_perturb
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
# The one rollout engine.
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(env, ep, policy, preprocess, device, *, strategy=None,
                run_cfg: RunConfig | None = None, num_inference_steps=None):
    """Run one episode. Returns a result dict (outcome + metrics + trajectory + recorder data).

    `strategy` is a duck-typed sampler strategy (RecordStrategy / Collect / Correct) or None
    for vanilla. `PNP_CONFIG` must already be set by the caller for the pass. Storage/DB writes
    are the driver's job — this stays store-agnostic.
    """
    run_cfg = run_cfg or RunConfig()
    model = policy.model
    task_desc = ep["task_desc"]
    max_steps = ep["max_steps"]
    chunk_size = policy.config.chunk_size

    ep_seed = episode_seed(ep["init_state"], ep.get("ep_idx", ep.get("episode_idx", 0)))
    torch.manual_seed(ep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ep_seed)
    _pnp_seed_perturb(ep_seed)                       # isolated perturbation stream, per episode

    _sampler.set_strategy(model, strategy)
    model._pnp_vf_evals = 0
    PNP_RECORDER.new_episode(meta={k: ep.get(k) for k in ("suite", "task_idx", "ep_idx")})

    est_chunks = max(1, round(max_steps / chunk_size))
    queue, ci = [], 0
    executed_actions, robot_states, chunk_boundary_actions = [], [], []
    chunk_noise_seeds = []
    frames = [] if run_cfg.record_obs_frames else None
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
    except Exception as e:                            # log errored episodes, don't drop them
        status, error_msg = "errored", f"{type(e).__name__}: {e}"

    elapsed = time.time() - t0
    n_steps = step + 1
    PNP_RECORDER.close_episode(success, n_steps)
    inst = compute_instability(executed_actions, chunk_boundary_actions)

    vf_evals = int(getattr(model, "_pnp_vf_evals", 0))
    if vf_evals == 0:                                 # vanilla path (orig sampler doesn't count)
        vf_evals = (num_inference_steps or policy.config.num_inference_steps) * max(ci, 1)

    return dict(
        success=success, n_steps=n_steps, elapsed_s=elapsed, status=status, error_msg=error_msg,
        nan_action_count=nan_count, n_chunks=ci, n_vf_evals=vf_evals, chunk_size=chunk_size,
        episode_seed=ep_seed, chunk_noise_seeds=chunk_noise_seeds,
        instability=inst,
        recorder_episode=PNP_RECORDER.episodes[-1] if PNP_RECORDER.episodes else None,
        trajectory=dict(
            actions=np.asarray(executed_actions, dtype=np.float32),
            robot_state=np.asarray(robot_states, dtype=np.float32),
        ) if run_cfg.record_trajectory else None,
        obs_frames=frames,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Recorder -> DB rows
# ═════════════════════════════════════════════════════════════════════════════
def _dvec(v, prefix):
    v = list(v or [])
    return {f"{prefix}{i}": (float(v[i]) if i < len(v) else None) for i in range(ADIM)}


def recorder_to_rows(rec_ep, chunk_noise_seeds):
    """Map a PnPRecorder episode -> (euler_steps rows, action_vectors rows, summary metrics)."""
    euler, vecs = [], []
    u_means, u_vecs, mm_bc = [], [], []
    if rec_ep:
        for c in rec_ep.get("chunks", []):
            ci = c["chunk_idx"]
            cns = chunk_noise_seeds[ci] if ci < len(chunk_noise_seeds) else None
            for st in c.get("steps", []):
                u_means.append(st["u_mean"])
                u_vecs.append(np.asarray(st.get("u_vec", [])))
                euler.append({
                    "chunk_idx": ci, "chunk_noise_seed": cns, "euler_step": st["step"],
                    "s": st["s"], "u_mean": st["u_mean"], "u_max": st["u_max"],
                    "a_std_mean": st.get("a_std_mean"),
                    **_dvec(st.get("u_vec"), "u_d"), **_dvec(st.get("a_std_vec"), "a_std_d"),
                })
                v = {"chunk_idx": ci, "euler_step": st["step"],
                     "a_mean_vec": list(map(float, st.get("a_mean_vec", []))),
                     "a_std_vec": list(map(float, st.get("a_std_vec", [])))}
                if "bc_vec" in st:
                    v["bc_vec"] = list(map(float, st["bc_vec"]))
                    v["mm_pc1_frac"] = st.get("mm_pc1_frac")
                    v["mm_bc_pc1"] = st.get("mm_bc_pc1")
                    mm_bc.append(st.get("mm_bc_pc1"))
                vecs.append(v)
    summary = {
        "u_mean_episode": float(np.mean(u_means)) if u_means else None,
        "u_max_episode": float(np.max(u_means)) if u_means else None,
        "n_pnp_activations": len(euler),
        "mm_bc_pc1_episode": float(np.nanmean(mm_bc)) if mm_bc else None,
    }
    if u_vecs:
        mean_uv = np.nanmean(np.stack([np.pad(u, (0, max(0, ADIM - len(u))))[:ADIM] for u in u_vecs]), axis=0)
        summary.update({f"u_mean_d{i}": float(mean_uv[i]) for i in range(ADIM)})
    return euler, vecs, summary


# ═════════════════════════════════════════════════════════════════════════════
# Method configuration (sets PNP_CONFIG + returns strategy + denormalized config)
# ═════════════════════════════════════════════════════════════════════════════
def configure_method(method, *, step_indices, pnp_k, base_steps, extra_steps,
                     record_per_iteration=False, compute_multimodal=False):
    """Set PNP_CONFIG for `method`; return (strategy, num_inference_steps, denorm_config)."""
    PNP_CONFIG.record_per_iteration = record_per_iteration
    PNP_CONFIG.compute_multimodal = compute_multimodal
    PNP_CONFIG.time_min = None
    is_pnp = method in ("pnp_uncertainty_only", "pnp_refinement", "pnp_refinement_avg")
    PNP_CONFIG.enabled = is_pnp
    if is_pnp:
        PNP_CONFIG.step_indices = tuple(step_indices)
        PNP_CONFIG.num_iterations = pnp_k
        PNP_CONFIG.mode = "uncertainty" if method == "pnp_uncertainty_only" else "both"
        PNP_CONFIG.refine_average = method == "pnp_refinement_avg"
        strategy = RecordStrategy(PNP_CONFIG)
        nis = None
    else:
        strategy = None
        nis = extra_steps if method == "extra_steps" else base_steps
    denorm = {
        "method": method, "pnp_enabled": is_pnp,
        "pnp_step_indices": list(step_indices) if is_pnp else None,
        "pnp_k": pnp_k if is_pnp else None,
        "refine_average": PNP_CONFIG.refine_average if is_pnp else None,
        "num_inference_steps": nis,
    }
    return strategy, nis, denorm


# ═════════════════════════════════════════════════════════════════════════════
# Generic experiment driver (used by slice + PRO)
# ═════════════════════════════════════════════════════════════════════════════
def run_methods(store, policy, preprocess, device, episodes, methods, *, driver, benchmark,
                step_indices=(2, 3), pnp_k=3, base_steps=10, extra_steps=16,
                experiment=None, run_cfg=None, record_per_iteration=False,
                compute_multimodal=False):
    """Run every method over every episode (grouped by env), logging to Supabase. Resumable."""
    run_cfg = run_cfg or RunConfig(experiment=experiment)
    store.start_run(driver=driver, benchmark=benchmark, experiment=run_cfg.experiment,
                    notes=run_cfg.notes)
    done = store.existing_keys(store.experiment)
    n = 0
    eps_sorted = sorted(episodes, key=lambda e: (e["suite"], e["task_idx"]))
    for (suite, task_idx), grp in groupby(eps_sorted, key=lambda e: (e["suite"], e["task_idx"])):
        grp = list(grp)
        env = make_env(grp[0]["bddl_path"])
        try:
            for method in methods:
                strat, nis, denorm = configure_method(
                    method, step_indices=step_indices, pnp_k=pnp_k, base_steps=base_steps,
                    extra_steps=extra_steps, record_per_iteration=record_per_iteration,
                    compute_multimodal=compute_multimodal)
                for ep in grp:
                    identity = {k: ep[k] for k in ("benchmark", "suite", "task_idx",
                                                   "episode_idx", "ep_idx", "init_state_hash")
                                if k in ep}
                    rid = store.make_rollout_id(store.experiment, identity, denorm)
                    if rid in done:
                        continue
                    res = run_episode(env, ep, policy, preprocess, device, strategy=strat,
                                      run_cfg=run_cfg, num_inference_steps=nis)
                    _log_rollout(store, rid, ep, denorm, res, benchmark, run_cfg)
                    n += 1
        finally:
            env.close()
    store.finish_run(status="completed", n_rollouts=n)
    return n


def _log_rollout(store, rid, ep, denorm, res, benchmark, run_cfg):
    euler, vecs, summary = recorder_to_rows(res.get("recorder_episode"),
                                            res.get("chunk_noise_seeds", []))
    row = {
        "rollout_id": rid, "benchmark": ep.get("benchmark", benchmark),
        "suite": ep["suite"], "task_idx": ep["task_idx"], "task_desc": ep.get("task_desc"),
        "episode_idx": ep.get("ep_idx", ep.get("episode_idx")),
        "init_state_hash": ep.get("init_state_hash"),
        "suite_family": ep.get("suite_family"), "perturb_axis": ep.get("perturb_axis"),
        "perturb_strength": ep.get("perturb_strength"), "distractor_object": ep.get("distractor_object"),
        "max_steps": ep.get("max_steps"), "chunk_size": res.get("chunk_size"),
        "n_chunks": res["n_chunks"], "action_dim": ADIM,
        "episode_seed": res["episode_seed"], "config_hash": store.config_hash(denorm),
        "config_json": denorm,
        "success": res["success"], "n_steps": res["n_steps"], "elapsed_s": res["elapsed_s"],
        "terminated_reason": "success" if res["success"] else res["status"],
        "status": res["status"], "error_msg": res["error_msg"],
        "nan_action_count": res["nan_action_count"], "n_vf_evals": res["n_vf_evals"],
        **denorm, **summary, **res["instability"],
    }
    blobs = {}
    if res.get("trajectory"):
        blobs["trajectory"] = res["trajectory"]
    if res.get("obs_frames"):
        blobs["obs_frames"] = res["obs_frames"]
    if PNP_CONFIG.record_per_iteration and res.get("recorder_episode"):
        ah = {}
        for c in res["recorder_episode"].get("chunks", []):
            for st in c.get("steps", []):
                if "a_hats" in st:
                    ah[f"c{c['chunk_idx']}_s{st['step']}"] = st["a_hats"]
        if ah:
            blobs["ahats"] = ah
    store.log_episode(row, euler_steps=euler, action_vectors=vecs, blobs=blobs or None)


# ═════════════════════════════════════════════════════════════════════════════
# Concrete drivers
# ═════════════════════════════════════════════════════════════════════════════
SLICE_METHODS = ["vanilla", "extra_steps", "pnp_uncertainty_only", "pnp_refinement"]
PRO_METHODS = ["pnp_uncertainty_only", "pnp_refinement", "pnp_refinement_avg"]


def run_controlled_slice(store, policy, preprocess, device, episodes, *, methods=None,
                         step_indices=(2, 3), pnp_k=3, base_steps=10, extra_steps=16,
                         experiment=None, run_cfg=None):
    return run_methods(store, policy, preprocess, device, episodes, methods or SLICE_METHODS,
                       driver="run_controlled_slice", benchmark="libero",
                       step_indices=step_indices, pnp_k=pnp_k, base_steps=base_steps,
                       extra_steps=extra_steps, experiment=experiment, run_cfg=run_cfg)


def run_pro(store, policy, preprocess, device, episodes, *, methods=None,
            step_indices=(3, 4), pnp_k=10, experiment=None, run_cfg=None,
            compute_multimodal=True):
    return run_methods(store, policy, preprocess, device, episodes, methods or PRO_METHODS,
                       driver="run_pro", benchmark="libero_pro", step_indices=step_indices,
                       pnp_k=pnp_k, experiment=experiment, run_cfg=run_cfg,
                       compute_multimodal=compute_multimodal)


# ── PCP collection ───────────────────────────────────────────────────────────
def pcp_collect(store, policy, preprocess, device, episodes, *, cfg=None, experiment=None,
                run_cfg=None):
    """Collect labeled (z_hat, obs_enc) chunks on the given episodes -> qc_rollouts."""
    import pandas as pd
    from .pcp import CollectStrategy
    cfg = cfg or PCPConfig()
    run_cfg = run_cfg or RunConfig(experiment=experiment)
    store.start_run(driver="pcp_collect", benchmark="libero_pro", experiment=run_cfg.experiment)
    eps_sorted = sorted(episodes, key=lambda e: (e["suite"], e["task_idx"]))
    n = 0
    for (suite, task_idx), grp in groupby(eps_sorted, key=lambda e: (e["suite"], e["task_idx"])):
        grp = list(grp)
        env = make_env(grp[0]["bddl_path"])
        try:
            for ep in grp:
                rid = f"{ep['suite']}:{ep['task_idx']}:{ep.get('ep_idx')}:{ep['init_state_hash']}"
                strat = CollectStrategy(cfg, adim=ADIM)
                PNP_CONFIG.enabled = True
                PNP_CONFIG.mode = "uncertainty"
                res = run_episode(env, ep, policy, preprocess, device, strategy=strat,
                                  run_cfg=run_cfg)
                # flatten per-chunk records into a parquet blob
                rows = []
                for c in strat.chunks:
                    for st in c["steps"]:
                        rows.append({"chunk_idx": c["chunk_idx"], "chunk_pos": c["chunk_pos"],
                                     "obs_enc": c["obs_enc"].tolist(), "step_idx": st["step_idx"],
                                     "s": st["s"], "z_hat": st["z_hat"].reshape(-1).tolist()})
                store.log_qc_rollout(
                    {"rollout_id": rid, "suite": ep["suite"], "task_idx": ep["task_idx"],
                     "episode_idx": ep.get("ep_idx"), "init_state_hash": ep["init_state_hash"],
                     "success": res["success"], "n_chunks": len(strat.chunks)},
                    chunks_df=pd.DataFrame(rows) if rows else None)
                n += 1
        finally:
            env.close()
    store.finish_run(n_rollouts=n)
    return n


# ── PCP 3-way eval ───────────────────────────────────────────────────────────
def pcp_eval(store, policy, preprocess, device, episodes, q_ckpt_id, *, cfg=None,
             experiment=None, run_cfg=None):
    """vanilla / pnp-only / pcp 3-way eval using a registered Q-corrector -> qc_eval + rollouts."""
    from .pcp import QCorrector, TemperatureScaler, CorrectStrategy
    cfg = cfg or PCPConfig()
    run_cfg = run_cfg or RunConfig(experiment=experiment)
    ckpt, qrow = store.load_q_corrector(q_ckpt_id)
    q_model = QCorrector(ckpt["action_dim"], ckpt["obs_dim"]).to(device)
    q_model.load_state_dict(ckpt["model"]); q_model.eval()
    q_scaler = TemperatureScaler().to(device); q_scaler.load_state_dict(ckpt["scaler"])
    store.start_run(driver="pcp_eval", benchmark="libero_pro", experiment=run_cfg.experiment)

    passes = [("vanilla", None), ("pnp_only", 0.0), ("pcp", cfg.lambda_pcp)]
    eps_sorted = sorted(episodes, key=lambda e: (e["suite"], e["task_idx"]))
    n = 0
    for (suite, task_idx), grp in groupby(eps_sorted, key=lambda e: (e["suite"], e["task_idx"])):
        grp = list(grp)
        env = make_env(grp[0]["bddl_path"])
        try:
            for name, lam in passes:
                lab = -1.0 if lam is None else lam
                for ep in grp:
                    rid = f"{ep['suite']}:{ep['task_idx']}:{ep.get('ep_idx')}:{ep['init_state_hash']}"
                    if lam is None:
                        PNP_CONFIG.enabled = False
                        strat = None
                    else:
                        PNP_CONFIG.enabled = True
                        PNP_CONFIG.mode = "both"
                        strat = CorrectStrategy(cfg, q_model, q_scaler, lam, device, adim=ADIM)
                    res = run_episode(env, ep, policy, preprocess, device, strategy=strat,
                                      run_cfg=run_cfg)
                    store.log_qc_eval({"rollout_id": rid, "lambda": lab, "suite": ep["suite"],
                                       "task_idx": ep["task_idx"], "episode_idx": ep.get("ep_idx"),
                                       "success": res["success"]})
                    n += 1
        finally:
            env.close()
    store.finish_run(n_rollouts=n)
    return n
