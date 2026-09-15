"""Current-boundary U20-gated latent Q-guidance on frozen held-out PRO160."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Method, RolloutConfig
from .five_step_diversity_experiment import _identity_key, identity_manifest_hash
from .qplanning_critic import load_qplanning_scorer
from .qplanning_eval_experiment import (
    QPLANNING_HELDOUT_EXPERIMENT, QPLANNING_HELDOUT_IDENTITIES,
    QPLANNING_HELDOUT_REPORT_EVERY, _completed_existing_methods,
    _qplanning_heldout_episodes, build_qplanning_stock_method)
from .qplanning_gradient_eval_experiment import (
    QGUIDE_ACTION_STEPS, QGUIDE_DENOISE_STEPS, QGUIDE_EULER_STEP,
    QGUIDE_UPDATE_RMS)


QGUIDE_U20_GATE_EXPERIMENT = "pi05-q50-u20-gated-latent-guidance-pro-heldout160-v1"
QGUIDE_U20_GATE_SHARDS = 4
QGUIDE_U20_GATE_THRESHOLD = 0.0225
QGUIDE_U20_GATE_HORIZON = 20
QGUIDE_U20_GATE_PNP_K = 5
QGUIDE_U20_GATE_PROBE_STEPS = (3, 4)
QGUIDE_U20_GATE_STRATEGIES = ("original", "u20_8chunk")
QGUIDE_U20_GATE_METHODS = {
    "original": Method.QGUIDE_U20_GATE_Q50_ORIGINAL,
    "u20_8chunk": Method.QGUIDE_U20_GATE_Q50_PRIORITY_U20_8CHUNK,
}
QGUIDE_U20_GATE_REQUIRED_UPDATES = {"original": 8_000, "u20_8chunk": 6_000}


def build_qguide_u20_gate_method(strategy: str, source_id: str, scorer, *,
                                  threshold: float = QGUIDE_U20_GATE_THRESHOLD,
                                  update_rms: float = QGUIDE_UPDATE_RMS):
    """Build one exact-stock-fallback, current-boundary-U20-gated Q guide."""
    if strategy not in QGUIDE_U20_GATE_METHODS:
        raise ValueError(f"unsupported gated Q-guidance critic {strategy!r}")
    return QGUIDE_U20_GATE_METHODS[strategy], RolloutConfig(
        policy_source_id=source_id,
        q_guidance_ckpt_id=scorer.checkpoint_id,
        q_guidance_step=QGUIDE_EULER_STEP,
        q_guidance_step_size=float(update_rms),
        q_guidance_scorer=scorer,
        q_guidance_gate_threshold=float(threshold),
        q_guidance_gate_horizon=QGUIDE_U20_GATE_HORIZON,
        q_guidance_gate_pnp_k=QGUIDE_U20_GATE_PNP_K,
        q_guidance_gate_probe_steps=QGUIDE_U20_GATE_PROBE_STEPS,
        num_inference_steps=QGUIDE_DENOISE_STEPS,
        n_action_steps=QGUIDE_ACTION_STEPS,
        save_uncertainty=False,
        save_observations=False,
        save_generated_chunks=False,
        video="off",
        skip_unused_renders=True,
        render_lead=2,
    )


def run_qplanning_u20_gated_gradient_heldout160(
        *, original_checkpoint_path: str | Path,
        u20_8chunk_checkpoint_path: str | Path,
        shard_count: int = QGUIDE_U20_GATE_SHARDS, shard_index: int,
        episode_limit: int | None = None,
        threshold: float = QGUIDE_U20_GATE_THRESHOLD,
        update_rms: float = QGUIDE_UPDATE_RMS,
        experiment: str = QGUIDE_U20_GATE_EXPERIMENT):
    """Evaluate both gated guides and exact matched historical stock on one shard."""
    from . import models, sampler
    from .experiments import _run_collection
    from .pcp_search.collection import manifest_shard
    from .store import SupabaseStore, gather_provenance

    if int(shard_count) != QGUIDE_U20_GATE_SHARDS:
        raise ValueError(f"gated Q-guidance requires shard_count={QGUIDE_U20_GATE_SHARDS}")
    shard_count = int(shard_count)
    if (isinstance(shard_index, bool) or int(shard_index) != shard_index
            or not 0 <= int(shard_index) < shard_count):
        raise ValueError(f"shard_index must lie in [0, {shard_count})")
    shard_index = int(shard_index)
    if episode_limit is not None:
        if (isinstance(episode_limit, bool) or int(episode_limit) != episode_limit
                or int(episode_limit) < 1):
            raise ValueError("episode_limit must be a positive integer or None")
        episode_limit = int(episode_limit)
    if float(threshold) != QGUIDE_U20_GATE_THRESHOLD:
        raise ValueError(
            f"this experiment is frozen to U20 threshold {QGUIDE_U20_GATE_THRESHOLD}")
    if float(update_rms) != QGUIDE_UPDATE_RMS:
        raise ValueError(f"this experiment is frozen to latent RMS {QGUIDE_UPDATE_RMS}")

    frozen, all_episodes = _qplanning_heldout_episodes()
    by_key = {
        (ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"])): ep
        for ep in all_episodes}
    shard_items = manifest_shard(frozen.items, shard_count, shard_index)
    episodes = [
        by_key[(item.suite, int(item.task_idx), int(item.init_state_index))]
        for item in shard_items]
    if len(episodes) != QPLANNING_HELDOUT_IDENTITIES // shard_count:
        raise AssertionError("gated Q-guidance shard must contain exactly 40 identities")
    if episode_limit is not None:
        episodes = episodes[:episode_limit]

    checkpoint_paths = {
        "original": Path(original_checkpoint_path).expanduser(),
        "u20_8chunk": Path(u20_8chunk_checkpoint_path).expanduser(),
    }
    device = models.default_device()
    scorers = {
        strategy: load_qplanning_scorer(
            path, device=device, expected_horizon=50,
            expected_source_revision=frozen.policy_revision)
        for strategy, path in checkpoint_paths.items()}
    expected_source = {
        "repo_id": frozen.policy_repo_id, "revision": frozen.policy_revision}
    if any(scorer.source_policy != expected_source for scorer in scorers.values()):
        raise ValueError("a gated critic was trained for a different source policy")
    if len({scorer.snapshot_id for scorer in scorers.values()}) != 1:
        raise ValueError("both gated critics must use the same immutable dataset snapshot")
    found_updates = {strategy: scorer.update for strategy, scorer in scorers.items()}
    if found_updates != QGUIDE_U20_GATE_REQUIRED_UPDATES:
        raise ValueError(
            f"gated Q-guidance requires {QGUIDE_U20_GATE_REQUIRED_UPDATES}; "
            f"found {found_updates}")

    source_id = f"{frozen.policy_repo_id}@{frozen.policy_revision}"
    methods = [
        build_qguide_u20_gate_method(
            strategy, source_id, scorers[strategy], threshold=threshold,
            update_rms=update_rms)
        for strategy in QGUIDE_U20_GATE_STRATEGIES]
    store = SupabaseStore()
    config_hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    identity_keys = {_identity_key(episode) for episode in episodes}
    initial_tally, initial_outcomes = _completed_existing_methods(
        store, experiment=experiment, method_config_hashes=config_hashes,
        identity_keys=identity_keys)

    stock_method, stock_config = build_qplanning_stock_method(source_id)
    stock_hash = store.config_hash(store._logical_key(stock_method, stock_config))
    _, historical_rows = _completed_existing_methods(
        store, experiment=QPLANNING_HELDOUT_EXPERIMENT,
        method_config_hashes={stock_method: stock_hash}, identity_keys=identity_keys)
    historical_stock = {
        key: outcomes[stock_method] for key, outcomes in historical_rows.items()
        if stock_method in outcomes}
    if len(historical_stock) != len(episodes):
        raise ValueError(
            f"matched historical stock is incomplete: "
            f"{len(historical_stock)}/{len(episodes)} identities")

    metadata = {
        "source_model_repo_id": frozen.policy_repo_id,
        "source_model_revision": frozen.policy_revision,
        "frozen_collection_manifest_id": frozen.manifest_id,
        "frozen_identity_manifest_hash": identity_manifest_hash(all_episodes),
        "data_split": "heldout",
        "heldout_category": "position_perturb",
        "target_identities": QPLANNING_HELDOUT_IDENTITIES,
        "evaluated_identities": len(episodes),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "checkpoint_ids": {
            strategy: scorer.checkpoint_id for strategy, scorer in scorers.items()},
        "critic_updates": found_updates,
        "guidance": "one normalized Q-ascent update, gated by current-boundary stock U20",
        "guidance_euler_step_zero_based": QGUIDE_EULER_STEP,
        "latent_update_rms": float(update_rms),
        "u20_gate_threshold": float(threshold),
        "u20_gate_horizon": QGUIDE_U20_GATE_HORIZON,
        "u20_gate_pnp_k": QGUIDE_U20_GATE_PNP_K,
        "u20_gate_probe_steps": list(QGUIDE_U20_GATE_PROBE_STEPS),
        "num_inference_steps": QGUIDE_DENOISE_STEPS,
        "generated_chunk_size": 50,
        "n_action_steps": QGUIDE_ACTION_STEPS,
        "historical_stock_experiment": QPLANNING_HELDOUT_EXPERIMENT,
        "historical_stock_config_hash": stock_hash,
        "config_hashes": config_hashes,
        "video": "off",
        "online_learning": False,
    }
    print({
        "experiment": experiment,
        "split": "same frozen position-perturbation PRO160 as notebook 68",
        "identities": len(episodes),
        "new_rollouts": len(episodes) * len(methods),
        "shard": f"{shard_index}/{shard_count}",
        "arms": ["U20-gated original Q50", "U20-gated 8-chunk-priority Q50"],
        "gate": (
            f"current-boundary K={QGUIDE_U20_GATE_PNP_K} U20 at Euler steps "
            f"{QGUIDE_U20_GATE_PROBE_STEPS} >= {float(threshold):g}"),
        "guidance": (
            f"one Q-ascent update at zero-based Euler step {QGUIDE_EULER_STEP}; "
            f"latent RMS={float(update_rms):g}"),
        "fallback": "non-firing boundary executes exact stock chunk from gate pass",
        "decode": "ordinary 10 Euler steps; first 10 of 50 actions execute",
        "historical_reference": (
            f"exact matched stock outcomes from {QPLANNING_HELDOUT_EXPERIMENT}; not rerun"),
        "telemetry": "per-boundary U10/U20/U50, gate decisions, Q change, action displacement",
        "video_frames_generated_chunks": "off",
        "report_every_complete_identities": QPLANNING_HELDOUT_REPORT_EVERY,
    })

    policy, preprocess, postprocess = models.load_pi05(
        repo_id=frozen.policy_repo_id, revision=frozen.policy_revision)
    if int(policy.config.chunk_size) != 50:
        raise ValueError(f"expected 50-action PI chunks, found {policy.config.chunk_size}")
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)
    sampler.enable_encoding_cache(
        policy.model, size=4, model_revision=frozen.policy_revision)
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=device, experiment=experiment, episodes=episodes, methods=methods,
        cohort="q50_u20_gated_gradient_pro_heldout160",
        shard_count=shard_count, shard_index=shard_index, benchmark="libero_pro",
        driver="pi05_q50_u20_gated_gradient_pro_heldout160",
        run_metadata=metadata,
        provenance=gather_provenance(
            model_repo_id=frozen.policy_repo_id,
            model_revision=frozen.policy_revision),
        report_every=0,
        report_every_identities=QPLANNING_HELDOUT_REPORT_EVERY,
        initial_tally=initial_tally,
        initial_identity_outcomes=initial_outcomes,
        matched_reference_outcomes={"historic stock VLA": historical_stock},
        progress_include_overall=True,
        rollout_batch_size=1,
        resume_completed_only=True)
    return {
        "experiment": experiment,
        "manifest_id": frozen.manifest_id,
        "identities_requested": len(episodes),
        "rollouts_requested": len(episodes) * len(methods),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "checkpoint_ids": {
            strategy: scorer.checkpoint_id for strategy, scorer in scorers.items()},
        "config_hashes": config_hashes,
        "threshold": float(threshold),
        "update_rms": float(update_rms),
    }


def validate_qplanning_u20_gated_gradient_sentinel(
        *, checkpoint_ids: dict,
        experiment: str = QGUIDE_U20_GATE_EXPERIMENT, store=None) -> dict:
    """Audit persisted gate settings and at least one completed row per requested arm."""
    from .store import SupabaseStore

    store = store or SupabaseStore()
    rows = store.fetch_all(
        "rollouts",
        "rollout_id,status,method,config_json,ms_candidate_u,video_path,obs_frames_path",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "status", "completed"), order_by=("rollout_id",))
    audits = {}
    for strategy, checkpoint_id in checkpoint_ids.items():
        if strategy not in QGUIDE_U20_GATE_METHODS:
            raise ValueError(f"unknown gated Q-guidance strategy: {strategy}")
        method = QGUIDE_U20_GATE_METHODS[strategy]
        matched = []
        for row in rows:
            if row.get("method") != method:
                continue
            config = row.get("config_json")
            if isinstance(config, str):
                config = json.loads(config)
            if (config or {}).get("q_guidance_ckpt_id") == checkpoint_id:
                matched.append((row, config))
        if not matched:
            raise ValueError(f"no completed gated Q-guidance rollout for {strategy}")
        row, config = matched[0]
        expected = {
            "q_guidance_step": QGUIDE_EULER_STEP,
            "q_guidance_step_size": QGUIDE_UPDATE_RMS,
            "q_guidance_gate_threshold": QGUIDE_U20_GATE_THRESHOLD,
            "q_guidance_gate_horizon": QGUIDE_U20_GATE_HORIZON,
            "q_guidance_gate_pnp_k": QGUIDE_U20_GATE_PNP_K,
            "q_guidance_gate_probe_steps": list(QGUIDE_U20_GATE_PROBE_STEPS),
            "num_inference_steps": QGUIDE_DENOISE_STEPS,
            "n_action_steps": QGUIDE_ACTION_STEPS,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise AssertionError(
                    f"{strategy} {key}: expected {value}, found {config.get(key)}")
        telemetry = row.get("ms_candidate_u")
        if isinstance(telemetry, str):
            telemetry = json.loads(telemetry)
        guidance = (telemetry or {}).get("q_guidance")
        if not guidance or guidance.get("n_gate_considered", 0) < 1:
            raise AssertionError(f"{strategy} is missing U20-gate telemetry")
        if row.get("video_path") or row.get("obs_frames_path"):
            raise AssertionError(f"{strategy} unexpectedly saved video or frames")
        audits[strategy] = {
            "rollout_id": row["rollout_id"],
            "boundaries": guidance["n_gate_considered"],
            "gate_fires": guidance["n_gate_fired"],
            "gate_fire_rate": guidance["gate_fire_rate"],
            "mean_gate_u20": guidance["mean_gate_u20"],
        }
    summary = {
        "status": "passed",
        "experiment": experiment,
        "threshold": QGUIDE_U20_GATE_THRESHOLD,
        "gate": {
            "horizon": QGUIDE_U20_GATE_HORIZON,
            "pnp_k": QGUIDE_U20_GATE_PNP_K,
            "probe_steps": list(QGUIDE_U20_GATE_PROBE_STEPS),
        },
        "arms": audits,
    }
    print("U20-gated Q-guidance sentinel:", summary)
    return summary
