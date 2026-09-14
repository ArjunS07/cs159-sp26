"""One-step latent Q-guidance evaluation on the frozen PRO160 split."""
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


QGUIDE_EVAL_EXPERIMENT = "pi05-q50-latent-guidance-pro-heldout160-v1"
QGUIDE_FIVE_EVAL_EXPERIMENT = "pi05-q50-latent-guidance-five-pro-heldout160-v1"
QGUIDE_FIVE_SHARDS = 4
QGUIDE_DENOISE_STEPS = 10
QGUIDE_EULER_STEP = 3
QGUIDE_UPDATE_RMS = 0.005
QGUIDE_ACTION_STEPS = 10
QGUIDE_METHODS = {
    "original": Method.QGUIDE_Q50_ORIGINAL,
    "failure": Method.QGUIDE_Q50_PRIORITY_FAILURE,
    "episode_u20": Method.QGUIDE_Q50_PRIORITY_EPISODE_U20,
    "u20_4chunk": Method.QGUIDE_Q50_PRIORITY_U20_4CHUNK,
    "u20_8chunk": Method.QGUIDE_Q50_PRIORITY_U20_8CHUNK,
}
QGUIDE_REQUIRED_UPDATES = {
    "original": 8_000,
    "failure": 6_000,
    "episode_u20": 6_000,
    "u20_4chunk": 6_000,
    "u20_8chunk": 6_000,
}
QGUIDE_V1_STRATEGIES = ("original", "failure", "u20_4chunk")
QGUIDE_FIVE_STRATEGIES = (
    "original", "failure", "episode_u20", "u20_4chunk", "u20_8chunk")


def build_qguide_method(strategy: str, source_id: str, scorer,
                        *, update_rms: float = QGUIDE_UPDATE_RMS):
    if strategy not in QGUIDE_METHODS:
        raise ValueError(f"unsupported Q-guidance critic {strategy!r}")
    config = RolloutConfig(
        policy_source_id=source_id,
        q_guidance_ckpt_id=scorer.checkpoint_id,
        q_guidance_step=QGUIDE_EULER_STEP,
        q_guidance_step_size=float(update_rms),
        q_guidance_scorer=scorer,
        num_inference_steps=QGUIDE_DENOISE_STEPS,
        n_action_steps=QGUIDE_ACTION_STEPS,
        save_uncertainty=False,
        save_observations=False,
        save_generated_chunks=False,
        video="off",
        skip_unused_renders=True,
        render_lead=2,
    )
    return QGUIDE_METHODS[strategy], config


def run_qplanning_latent_guidance_heldout160(
        *, original_checkpoint_path: str | Path,
        failure_checkpoint_path: str | Path,
        u20_4chunk_checkpoint_path: str | Path,
        episode_u20_checkpoint_path: str | Path | None = None,
        u20_8chunk_checkpoint_path: str | Path | None = None,
        shard_count: int = 1, shard_index: int = 0,
        episode_limit: int | None = None,
        update_rms: float = QGUIDE_UPDATE_RMS,
        experiment: str = QGUIDE_EVAL_EXPERIMENT):
    """Run three or five frozen critics as single-latent Q guides."""
    from . import models, sampler
    from .experiments import _run_collection
    from .pcp_search.collection import manifest_shard
    from .store import SupabaseStore, gather_provenance

    if episode_limit is not None:
        if (isinstance(episode_limit, bool) or int(episode_limit) != episode_limit
                or int(episode_limit) < 1):
            raise ValueError("episode_limit must be a positive integer or None")
        episode_limit = int(episode_limit)
    if not 0 < float(update_rms) <= 0.05:
        raise ValueError("update_rms must lie in (0, 0.05]")
    five_critic = (
        episode_u20_checkpoint_path is not None
        or u20_8chunk_checkpoint_path is not None)
    if five_critic and (
            episode_u20_checkpoint_path is None
            or u20_8chunk_checkpoint_path is None):
        raise ValueError(
            "five-critic evaluation requires both episode-U20 and 8-chunk-U20 checkpoints")
    expected_shards = QGUIDE_FIVE_SHARDS if five_critic else 1
    if (isinstance(shard_count, bool) or int(shard_count) != shard_count
            or int(shard_count) != expected_shards):
        raise ValueError(
            f"{'five' if five_critic else 'three'}-critic evaluation requires "
            f"shard_count={expected_shards}")
    shard_count = int(shard_count)
    if (isinstance(shard_index, bool) or int(shard_index) != shard_index
            or not 0 <= int(shard_index) < shard_count):
        raise ValueError(f"shard_index must lie in [0, {shard_count})")
    shard_index = int(shard_index)
    strategies = QGUIDE_FIVE_STRATEGIES if five_critic else QGUIDE_V1_STRATEGIES

    frozen, all_episodes = _qplanning_heldout_episodes()
    if five_critic:
        by_key = {
            (episode["suite"], int(episode["task_idx"]), int(episode["ep_idx"])): episode
            for episode in all_episodes}
        shard_items = manifest_shard(frozen.items, shard_count, shard_index)
        episodes = [
            by_key[(item.suite, int(item.task_idx), int(item.init_state_index))]
            for item in shard_items]
        if len(episodes) != QPLANNING_HELDOUT_IDENTITIES // shard_count:
            raise AssertionError("five-critic shard does not contain exactly 40 identities")
    else:
        episodes = list(all_episodes)
    if episode_limit is not None:
        episodes = episodes[:episode_limit]
    if (episode_limit is None and len(episodes)
            != QPLANNING_HELDOUT_IDENTITIES // shard_count):
        raise AssertionError("Q-guidance evaluation has the wrong identity count")

    checkpoint_paths = {
        "original": Path(original_checkpoint_path).expanduser(),
        "failure": Path(failure_checkpoint_path).expanduser(),
        "u20_4chunk": Path(u20_4chunk_checkpoint_path).expanduser(),
    }
    if five_critic:
        checkpoint_paths.update({
            "episode_u20": Path(episode_u20_checkpoint_path).expanduser(),
            "u20_8chunk": Path(u20_8chunk_checkpoint_path).expanduser(),
        })
    checkpoint_paths = {
        strategy: checkpoint_paths[strategy] for strategy in strategies}
    device = models.default_device()
    scorers = {
        strategy: load_qplanning_scorer(
            path, device=device, expected_horizon=50,
            expected_source_revision=frozen.policy_revision)
        for strategy, path in checkpoint_paths.items()
    }
    expected_source = {
        "repo_id": frozen.policy_repo_id, "revision": frozen.policy_revision}
    if any(scorer.source_policy != expected_source for scorer in scorers.values()):
        raise ValueError("a Q-guidance critic was trained for a different source policy")
    if len({scorer.snapshot_id for scorer in scorers.values()}) != 1:
        raise ValueError("all critics must use the same immutable dataset snapshot")
    found_updates = {strategy: scorer.update for strategy, scorer in scorers.items()}
    required_updates = {
        strategy: QGUIDE_REQUIRED_UPDATES[strategy] for strategy in strategies}
    if found_updates != required_updates:
        raise ValueError(
            f"Q-guidance requires checkpoints {required_updates}; "
            f"found {found_updates}")

    source_id = f"{frozen.policy_repo_id}@{frozen.policy_revision}"
    methods = [
        build_qguide_method(strategy, source_id, scorers[strategy],
                            update_rms=update_rms)
        for strategy in strategies]
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
        method_config_hashes={stock_method: stock_hash},
        identity_keys=identity_keys)
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
        "checkpoint_paths": {
            strategy: str(path) for strategy, path in checkpoint_paths.items()},
        "critic_updates": found_updates,
        "guidance": "one normalized Q-ascent update on the live flow latent",
        "guidance_euler_step_zero_based": QGUIDE_EULER_STEP,
        "latent_update_rms": float(update_rms),
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
        "arms": [f"latent guidance by {strategy} Q50" for strategy in strategies],
        "guidance": (
            f"one Q-ascent update at zero-based Euler step {QGUIDE_EULER_STEP}; "
            f"latent RMS={float(update_rms):g}"),
        "decode": "ordinary 10 Euler steps; first 10 of 50 actions execute",
        "stock_pairing": (
            "same initial noise; exact stock chunk is decoded online for displacement only"),
        "historical_reference": (
            f"exact matched stock outcomes from {QPLANNING_HELDOUT_EXPERIMENT}; not rerun"),
        "telemetry": (
            "pre/post Q, Q delta, latent gradient/update RMS, exact stock-to-guided "
            "first10/full50 action displacement"),
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
        cohort=("q50_latent_guidance_five_pro_heldout160"
                if five_critic else "q50_latent_guidance_pro_heldout160"),
        shard_count=shard_count, shard_index=shard_index, benchmark="libero_pro",
        driver=("pi05_q50_latent_guidance_five_pro_heldout160"
                if five_critic else "pi05_q50_latent_guidance_pro_heldout160"),
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
        "update_rms": float(update_rms),
    }


def validate_qplanning_latent_guidance_sentinel(
        *, checkpoint_ids: dict,
        experiment: str = QGUIDE_EVAL_EXPERIMENT, store=None) -> dict:
    """Audit persisted settings and guidance telemetry for requested arms."""
    from .store import SupabaseStore

    store = store or SupabaseStore()
    rows = store.fetch_all(
        "rollouts",
        "rollout_id,status,method,config_json,ms_candidate_u,video_path,obs_frames_path",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "status", "completed"),
        order_by=("rollout_id",))
    audits = {}
    unknown = sorted(set(checkpoint_ids) - set(QGUIDE_METHODS))
    if unknown:
        raise ValueError(f"unknown Q-guidance strategies: {unknown}")
    for strategy, checkpoint_id in checkpoint_ids.items():
        method = QGUIDE_METHODS[strategy]
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
            raise ValueError(f"no completed Q-guidance rollout for {strategy}")
        row, config = matched[0]
        expected = {
            "q_guidance_step": QGUIDE_EULER_STEP,
            "q_guidance_step_size": QGUIDE_UPDATE_RMS,
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
        if not guidance or guidance.get("n_updates", 0) < 1:
            raise AssertionError(f"{strategy} is missing Q-guidance telemetry")
        if row.get("video_path") or row.get("obs_frames_path"):
            raise AssertionError(f"{strategy} unexpectedly saved video or frames")
        audits[strategy] = {
            "rollout_id": row["rollout_id"],
            "decision_boundaries": guidance["n_updates"],
            "mean_delta_q": guidance["mean_delta_q"],
            "mean_first10_env_motion_rms": guidance[
                "mean_first10_env_motion_rms"],
        }
    summary = {
        "status": "passed",
        "experiment": experiment,
        "guidance_step_zero_based": QGUIDE_EULER_STEP,
        "latent_update_rms": QGUIDE_UPDATE_RMS,
        "decode_steps": QGUIDE_DENOISE_STEPS,
        "executed_actions_per_boundary": QGUIDE_ACTION_STEPS,
        "arms": audits,
    }
    print("Q-guidance sentinel:", summary)
    return summary
