"""Evaluate two replay-prioritized Q50 critics on the frozen PRO160 split."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Method
from .five_step_diversity_experiment import _identity_key, identity_manifest_hash
from .qplanning_critic import load_qplanning_scorer
from .qplanning_eval_experiment import (
    QPLANNING_EVAL_ACTION_STEPS, QPLANNING_EVAL_CANDIDATE_BATCH_SIZE,
    QPLANNING_EVAL_DENOISE_STEPS, QPLANNING_EVAL_NUM_CANDIDATES,
    QPLANNING_EVAL_NUM_ELITES, QPLANNING_EVAL_TEMPERATURE,
    QPLANNING_HELDOUT_EXPERIMENT, QPLANNING_HELDOUT_IDENTITIES,
    QPLANNING_HELDOUT_REPORT_EVERY, _completed_existing_methods,
    _qplanning_heldout_episodes, build_qplanning_eval_method,
    build_qplanning_stock_method, validate_qplanning_eval_sentinel)


QPLANNING_PRIORITY_EVAL_EXPERIMENT = (
    "pi05-qplanning-q50-priority-pro-heldout160-v1")
QPLANNING_PRIORITY_EVAL_UPDATE = 6_000
QPLANNING_PRIORITY_METHODS = {
    "failure": Method.QPLANNING_Q50_PRIORITY_FAILURE,
    "u20_4chunk": Method.QPLANNING_Q50_PRIORITY_U20_4CHUNK,
}


def build_qplanning_priority_method(strategy: str, source_id: str, scorer, *,
                                    candidate_batch_size: int):
    """Use ordinary Q50 inference while retaining the critic's replay identity."""
    if strategy not in QPLANNING_PRIORITY_METHODS:
        raise ValueError(f"unsupported priority strategy {strategy!r}")
    _, config = build_qplanning_eval_method(
        50, source_id, scorer, candidate_batch_size=candidate_batch_size)
    return QPLANNING_PRIORITY_METHODS[strategy], config


def run_qplanning_priority_heldout160(
        *, failure_checkpoint_path: str | Path,
        u20_4chunk_checkpoint_path: str | Path,
        episode_limit: int | None = None,
        candidate_batch_size: int = QPLANNING_EVAL_CANDIDATE_BATCH_SIZE,
        experiment: str = QPLANNING_PRIORITY_EVAL_EXPERIMENT):
    """Run both priority critics on all 160 identities; reuse matched stock."""
    from . import models, sampler
    from .experiments import _run_collection
    from .store import SupabaseStore, gather_provenance

    if episode_limit is not None:
        if (isinstance(episode_limit, bool) or int(episode_limit) != episode_limit
                or int(episode_limit) < 1):
            raise ValueError("episode_limit must be a positive integer or None")
        episode_limit = int(episode_limit)
    if (isinstance(candidate_batch_size, bool)
            or int(candidate_batch_size) != candidate_batch_size
            or not 1 <= int(candidate_batch_size) <= QPLANNING_EVAL_NUM_CANDIDATES):
        raise ValueError("candidate_batch_size must lie in [1, 64]")
    candidate_batch_size = int(candidate_batch_size)

    frozen, all_episodes = _qplanning_heldout_episodes()
    episodes = all_episodes if episode_limit is None else all_episodes[:episode_limit]
    if episode_limit is None and len(episodes) != QPLANNING_HELDOUT_IDENTITIES:
        raise AssertionError("priority evaluation requires all 160 held-out identities")

    device = models.default_device()
    checkpoint_paths = {
        "failure": Path(failure_checkpoint_path).expanduser(),
        "u20_4chunk": Path(u20_4chunk_checkpoint_path).expanduser(),
    }
    scorers = {
        strategy: load_qplanning_scorer(
            path, device=device, expected_horizon=50,
            expected_source_revision=frozen.policy_revision)
        for strategy, path in checkpoint_paths.items()
    }
    expected_source = {
        "repo_id": frozen.policy_repo_id, "revision": frozen.policy_revision}
    if any(scorer.source_policy != expected_source for scorer in scorers.values()):
        raise ValueError("a priority critic was trained for a different source policy")
    if len({scorer.snapshot_id for scorer in scorers.values()}) != 1:
        raise ValueError("priority critics must use the same immutable dataset snapshot")
    if any(scorer.update != QPLANNING_PRIORITY_EVAL_UPDATE
           for scorer in scorers.values()):
        found = {strategy: scorer.update for strategy, scorer in scorers.items()}
        raise ValueError(f"priority evaluation requires step-6000 checkpoints; found {found}")

    source_id = f"{frozen.policy_repo_id}@{frozen.policy_revision}"
    methods = [
        build_qplanning_priority_method(
            strategy, source_id, scorers[strategy],
            candidate_batch_size=candidate_batch_size)
        for strategy in QPLANNING_PRIORITY_METHODS]
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
        "priority_checkpoint_ids": {
            strategy: scorer.checkpoint_id for strategy, scorer in scorers.items()},
        "priority_checkpoint_paths": {
            strategy: str(path) for strategy, path in checkpoint_paths.items()},
        "critic_update": QPLANNING_PRIORITY_EVAL_UPDATE,
        "num_candidates": QPLANNING_EVAL_NUM_CANDIDATES,
        "num_elites": QPLANNING_EVAL_NUM_ELITES,
        "temperature": QPLANNING_EVAL_TEMPERATURE,
        "planner_num_inference_steps": QPLANNING_EVAL_DENOISE_STEPS,
        "generated_chunk_size": 50,
        "n_action_steps": QPLANNING_EVAL_ACTION_STEPS,
        "candidate_batch_size": candidate_batch_size,
        "candidate_seed_scheme": "stock_slot0_v1",
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
        "arms": ["Q50 failure-priority", "Q50 4-chunk-U20-priority"],
        "planner": "64 candidates -> top 16 -> Q-softmax weighted full chunk",
        "decode": "3 Euler steps per candidate",
        "execution": "first 10 blended actions, then replan",
        "historical_reference": (
            f"exact matched stock from {QPLANNING_HELDOUT_EXPERIMENT}; not rerun"),
        "online_learning": "off; critics are frozen",
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
        cohort="qplanning_q50_priority_pro_heldout160",
        shard_count=1, shard_index=0, benchmark="libero_pro",
        driver="pi05_qplanning_q50_priority_pro_heldout160",
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
    }


def validate_qplanning_priority_sentinel(
        *, failure_checkpoint_id: str, u20_4chunk_checkpoint_id: str,
        experiment: str = QPLANNING_PRIORITY_EVAL_EXPERIMENT,
        store=None) -> dict:
    """Audit both persisted priority arms and one identity shared by them."""
    from .store import SupabaseStore

    store = store or SupabaseStore()
    checkpoint_ids = {
        Method.QPLANNING_Q50_PRIORITY_FAILURE: failure_checkpoint_id,
        Method.QPLANNING_Q50_PRIORITY_U20_4CHUNK: u20_4chunk_checkpoint_id,
    }
    planner_audits = {
        method: validate_qplanning_eval_sentinel(
            horizon=50, checkpoint_id=checkpoint_id, experiment=experiment,
            method=method, store=store)
        for method, checkpoint_id in checkpoint_ids.items()}
    rows = store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,method,config_json",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "status", "completed"),
        order_by=("rollout_id",))
    by_identity = {}
    for row in rows:
        method = row.get("method")
        if method not in checkpoint_ids:
            continue
        config = row.get("config_json")
        if isinstance(config, str):
            config = json.loads(config)
        if (config or {}).get("qplanning_ckpt_id") != checkpoint_ids[method]:
            continue
        by_identity.setdefault(_identity_key(row), set()).add(method)
    complete = [key for key, methods in by_identity.items()
                if set(checkpoint_ids).issubset(methods)]
    if not complete:
        raise ValueError("no held-out identity is complete under both priority critics")
    summary = {
        "status": "passed",
        "experiment": experiment,
        "matched_identity": sorted(complete)[0],
        "arms": ["Q50 failure-priority", "Q50 4-chunk-U20-priority"],
        "candidates_per_boundary": 64,
        "elites_per_boundary": 16,
        "decode_steps": 3,
        "executed_actions_per_boundary": 10,
        "video_and_frames_absent": all(
            audit["video_and_frames_absent"] for audit in planner_audits.values()),
    }
    print("Priority-Q50 held-out sentinel:", summary)
    return summary
