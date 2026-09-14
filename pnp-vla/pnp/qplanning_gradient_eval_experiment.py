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
QGUIDE_DENOISE_STEPS = 10
QGUIDE_EULER_STEP = 3
QGUIDE_UPDATE_RMS = 0.005
QGUIDE_ACTION_STEPS = 10
QGUIDE_METHODS = {
    "original": Method.QGUIDE_Q50_ORIGINAL,
    "failure": Method.QGUIDE_Q50_PRIORITY_FAILURE,
    "u20_4chunk": Method.QGUIDE_Q50_PRIORITY_U20_4CHUNK,
}
QGUIDE_REQUIRED_UPDATES = {
    "original": 8_000,
    "failure": 6_000,
    "u20_4chunk": 6_000,
}


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
        episode_limit: int | None = None,
        update_rms: float = QGUIDE_UPDATE_RMS,
        experiment: str = QGUIDE_EVAL_EXPERIMENT):
    """Run three frozen critics as single-latent Q guides; reuse matched stock."""
    from . import models, sampler
    from .experiments import _run_collection
    from .store import SupabaseStore, gather_provenance

    if episode_limit is not None:
        if (isinstance(episode_limit, bool) or int(episode_limit) != episode_limit
                or int(episode_limit) < 1):
            raise ValueError("episode_limit must be a positive integer or None")
        episode_limit = int(episode_limit)
    if not 0 < float(update_rms) <= 0.05:
        raise ValueError("update_rms must lie in (0, 0.05]")

    frozen, all_episodes = _qplanning_heldout_episodes()
    episodes = all_episodes if episode_limit is None else all_episodes[:episode_limit]
    if episode_limit is None and len(episodes) != QPLANNING_HELDOUT_IDENTITIES:
        raise AssertionError("Q-guidance evaluation requires all 160 held-out identities")

    checkpoint_paths = {
        "original": Path(original_checkpoint_path).expanduser(),
        "failure": Path(failure_checkpoint_path).expanduser(),
        "u20_4chunk": Path(u20_4chunk_checkpoint_path).expanduser(),
    }
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
        raise ValueError("all three critics must use the same immutable dataset snapshot")
    found_updates = {strategy: scorer.update for strategy, scorer in scorers.items()}
    if found_updates != QGUIDE_REQUIRED_UPDATES:
        raise ValueError(
            f"Q-guidance requires checkpoints {QGUIDE_REQUIRED_UPDATES}; "
            f"found {found_updates}")

    source_id = f"{frozen.policy_repo_id}@{frozen.policy_revision}"
    methods = [
        build_qguide_method(strategy, source_id, scorers[strategy],
                            update_rms=update_rms)
        for strategy in QGUIDE_METHODS]
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
        "arms": [
            "latent guidance by original Q50",
            "latent guidance by failure-priority Q50",
            "latent guidance by 4-chunk-U20-priority Q50",
        ],
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
        cohort="q50_latent_guidance_pro_heldout160",
        shard_count=1, shard_index=0, benchmark="libero_pro",
        driver="pi05_q50_latent_guidance_pro_heldout160",
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
        "checkpoint_ids": {
            strategy: scorer.checkpoint_id for strategy, scorer in scorers.items()},
        "config_hashes": config_hashes,
        "update_rms": float(update_rms),
    }


def validate_qplanning_latent_guidance_sentinel(
        *, checkpoint_ids: dict,
        experiment: str = QGUIDE_EVAL_EXPERIMENT, store=None) -> dict:
    """Audit persisted settings and guidance telemetry for all three arms."""
    from .store import SupabaseStore

    store = store or SupabaseStore()
    rows = store.fetch_all(
        "rollouts",
        "rollout_id,status,method,config_json,ms_candidate_u,video_path,obs_frames_path",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "status", "completed"),
        order_by=("rollout_id",))
    audits = {}
    for strategy, method in QGUIDE_METHODS.items():
        matched = []
        for row in rows:
            if row.get("method") != method:
                continue
            config = row.get("config_json")
            if isinstance(config, str):
                config = json.loads(config)
            if (config or {}).get("q_guidance_ckpt_id") == checkpoint_ids[strategy]:
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
