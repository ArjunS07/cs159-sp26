"""Q10/Q50 candidate-planning evaluation on frozen LIBERO-PRO cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .config import Method, RolloutConfig
from .five_step_diversity_experiment import (
    _identity_key, _load_verified_historical_baseline,
    identity_manifest_hash, identity_manifest_payload)
from .qplanning_critic import load_qplanning_scorer


QPLANNING_EVAL_EPISODE_INDICES = (10, 11)
QPLANNING_EVAL_IDENTITIES = 220
QPLANNING_EVAL_NUM_CANDIDATES = 64
QPLANNING_EVAL_NUM_ELITES = 16
QPLANNING_EVAL_TEMPERATURE = 1.0
QPLANNING_EVAL_DENOISE_STEPS = 3
QPLANNING_EVAL_ACTION_STEPS = 10
QPLANNING_EVAL_CANDIDATE_BATCH_SIZE = 8
QPLANNING_EVAL_REPORT_EVERY = 25
QPLANNING_EVAL_EXPERIMENTS = {
    10: "pi05-qplanning-q10-pro220-v1",
    50: "pi05-qplanning-q50-pro220-v1",
}

QPLANNING_HELDOUT_EXPERIMENT = "pi05-qplanning-q10-q50-pro-heldout160-v1"
QPLANNING_HELDOUT_IDENTITIES = 160
QPLANNING_HELDOUT_SHARDS = 4
QPLANNING_HELDOUT_REPORT_EVERY = 10


def build_qplanning_eval_method(
        horizon: int, source_id: str, scorer, *,
        candidate_batch_size: int = QPLANNING_EVAL_CANDIDATE_BATCH_SIZE):
    """Build the fixed paper-style sample-64/top-16/Q-softmax planning arm."""
    horizon = int(horizon)
    if horizon not in QPLANNING_EVAL_EXPERIMENTS:
        raise ValueError("horizon must be 10 or 50")
    if int(scorer.horizon) != horizon:
        raise ValueError(f"Q{horizon} worker received a Q{scorer.horizon} critic")
    method = Method.QPLANNING_Q10 if horizon == 10 else Method.QPLANNING_Q50
    config = RolloutConfig(
        num_samples=QPLANNING_EVAL_NUM_CANDIDATES,
        candidate_set_id=(
            f"{source_id}|q{horizon}|sample{QPLANNING_EVAL_NUM_CANDIDATES}|"
            f"elite{QPLANNING_EVAL_NUM_ELITES}"),
        policy_source_id=source_id,
        candidate_seed_scheme="stock_slot0_v1",
        qplanning_ckpt_id=scorer.checkpoint_id,
        qplanning_n_elites=QPLANNING_EVAL_NUM_ELITES,
        qplanning_temperature=QPLANNING_EVAL_TEMPERATURE,
        qplanning_candidate_batch_size=int(candidate_batch_size),
        qplanning_scorer=scorer,
        num_inference_steps=QPLANNING_EVAL_DENOISE_STEPS,
        n_action_steps=QPLANNING_EVAL_ACTION_STEPS,
        save_trajectory=True,
        save_generated_chunks=False,
        save_observations=False,
        video="off",
        skip_unused_renders=True,
        render_lead=2)
    return method, config


def build_qplanning_stock_method(source_id: str):
    """Build the ordinary source-policy baseline used beside both planners."""
    return Method.VANILLA, RolloutConfig(
        policy_source_id=source_id,
        num_inference_steps=10,
        n_action_steps=QPLANNING_EVAL_ACTION_STEPS,
        save_trajectory=True,
        save_generated_chunks=False,
        save_observations=False,
        video="off",
        skip_unused_renders=True,
        render_lead=2)


def _completed_existing(store, *, experiment, method, config_hash, identity_keys):
    rows = store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq("experiment", experiment).eq("method", method),
        order_by=("rollout_id",))
    initial_tally, initial_outcomes = {}, {}
    for row in rows:
        key = _identity_key(row)
        if (key not in identity_keys or row.get("status") != "completed"
                or row.get("config_hash") != config_hash):
            continue
        if row.get("success") not in (True, False, 0, 1):
            raise ValueError("completed Q-Planning rollout has an invalid success outcome")
        if key in initial_outcomes:
            raise ValueError("duplicate completed Q-Planning identity/config")
        initial_outcomes[key] = {method: bool(row["success"])}
        counts = initial_tally.setdefault((key[0], method), [0, 0])
        counts[0] += 1
        counts[1] += int(row["success"])
    return initial_tally, initial_outcomes


def _completed_existing_methods(store, *, experiment, method_config_hashes,
                                identity_keys):
    """Restore completed rows for an exact multi-arm, same-identity comparison."""
    rows = store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq("experiment", experiment),
        order_by=("rollout_id",))
    initial_tally, initial_outcomes = {}, {}
    for row in rows:
        method = row.get("method")
        key = _identity_key(row)
        if (method not in method_config_hashes or key not in identity_keys
                or row.get("status") != "completed"
                or row.get("config_hash") != method_config_hashes[method]):
            continue
        if row.get("success") not in (True, False, 0, 1):
            raise ValueError("completed held-out rollout has an invalid success outcome")
        outcomes = initial_outcomes.setdefault(key, {})
        if method in outcomes:
            raise ValueError("duplicate completed held-out identity/method/config")
        outcomes[method] = bool(row["success"])
        counts = initial_tally.setdefault((key[0], method), [0, 0])
        counts[0] += 1
        counts[1] += int(row["success"])
    return initial_tally, initial_outcomes


def _qplanning_heldout_episodes():
    """Materialize the exact category-held-out 160-row PRO manifest as episodes."""
    from .experiments import _prepare_libero_pro_expanded_episodes
    from .pcp_search.pro import PRO_HELDOUT_QUOTAS, build_pro_manifest

    frozen = build_pro_manifest(split="heldout")
    suites = list(PRO_HELDOUT_QUOTAS)
    # The 30-row suites use states 0--2/task; the 20-row suites use 0--1/task.
    installed = _prepare_libero_pro_expanded_episodes(
        suites=suites, episode_idxs=(0, 1, 2), allow_zero_sr_suites=True)
    by_key = {
        (episode["suite"], int(episode["task_idx"]), int(episode["ep_idx"])): episode
        for episode in installed}
    episodes = []
    for item in frozen.items:
        if item.behavior_seed_index != 0:
            raise AssertionError("held-out evaluation requires behavior_seed_index=0")
        key = (item.suite, int(item.task_idx), int(item.init_state_index))
        if key not in by_key:
            raise ValueError(f"installed LIBERO-PRO assets are missing held-out identity {key}")
        episodes.append(by_key[key])
    if len(episodes) != QPLANNING_HELDOUT_IDENTITIES:
        raise AssertionError(
            f"expected {QPLANNING_HELDOUT_IDENTITIES} held-out identities, "
            f"found {len(episodes)}")
    if len({_identity_key(episode) for episode in episodes}) != len(episodes):
        raise AssertionError("held-out manifest contains duplicate physical identities")
    return frozen, episodes


def run_qplanning_eval_worker(
        *, horizon: int, checkpoint_path: str | Path,
        episode_limit: int | None = None,
        candidate_batch_size: int = QPLANNING_EVAL_CANDIDATE_BATCH_SIZE,
        manifest_hash: str = "", source_model_revision: str = "",
        experiment: str | None = None):
    """Evaluate one trained critic on all 220 exact PRO pilot identities."""
    from . import models, sampler
    from .diversity import (
        SOURCE_FRACTIONAL_SHORT_SUITES, source_checkpoint_model_source)
    from .experiments import (
        _prepare_libero_pro_expanded_episodes, _run_collection, expanded_pro_suites)
    from .store import SupabaseStore, gather_provenance

    horizon = int(horizon)
    if horizon not in QPLANNING_EVAL_EXPERIMENTS:
        raise ValueError("horizon must be 10 or 50")
    if not manifest_hash or not source_model_revision:
        raise ValueError("load manifest_hash and source_model_revision from the shared v2 manifest")
    if episode_limit is not None:
        if (isinstance(episode_limit, bool) or int(episode_limit) != episode_limit
                or episode_limit < 1):
            raise ValueError("episode_limit must be a positive integer or None")
        episode_limit = int(episode_limit)
    if (isinstance(candidate_batch_size, bool)
            or int(candidate_batch_size) != candidate_batch_size
            or not 1 <= int(candidate_batch_size) <= QPLANNING_EVAL_NUM_CANDIDATES):
        raise ValueError("candidate_batch_size must lie in [1, 64]")
    candidate_batch_size = int(candidate_batch_size)
    experiment = experiment or QPLANNING_EVAL_EXPERIMENTS[horizon]

    store = SupabaseStore()
    scorer = load_qplanning_scorer(
        checkpoint_path, device=models.default_device(), expected_horizon=horizon,
        expected_source_revision=source_model_revision)
    source_repo, source_revision = source_checkpoint_model_source(
        store, expected_revision=source_model_revision)
    if (scorer.source_policy["repo_id"] != source_repo
            or scorer.source_policy["revision"] != source_revision):
        raise ValueError("critic and historical PRO cohort use different source checkpoints")

    suites = [suite for suite in expanded_pro_suites()
              if suite not in SOURCE_FRACTIONAL_SHORT_SUITES]
    manifest = _prepare_libero_pro_expanded_episodes(
        suites=suites, episode_idxs=QPLANNING_EVAL_EPISODE_INDICES)
    if len(suites) != 11 or len(manifest) != QPLANNING_EVAL_IDENTITIES:
        raise ValueError("expected the frozen 11-suite, 220-identity PRO cohort")
    historical_rows, historical_config_hash = _load_verified_historical_baseline(
        store, manifest, source_repo=source_repo, source_revision=source_revision)
    episodes = manifest if episode_limit is None else manifest[:episode_limit]
    identity_keys = {_identity_key(episode) for episode in episodes}
    reference = {
        _identity_key(row): bool(row["success"])
        for row in historical_rows.to_dict("records")
        if _identity_key(row) in identity_keys}
    if len(reference) != len(episodes):
        raise ValueError("historical stock rows do not match every requested identity")

    source_id = f"{source_repo}@{source_revision}"
    method, config = build_qplanning_eval_method(
        horizon, source_id, scorer, candidate_batch_size=candidate_batch_size)
    config_hash = store.config_hash(store._logical_key(method, config))
    initial_tally, initial_outcomes = _completed_existing(
        store, experiment=experiment, method=method, config_hash=config_hash,
        identity_keys=identity_keys)
    frozen_hash = identity_manifest_hash(manifest)
    checkpoint_path = str(Path(checkpoint_path).expanduser())
    metadata = {
        "source_model_repo_id": source_repo,
        "source_model_revision": source_revision,
        "bootstrap_manifest_hash": manifest_hash,
        "frozen_identity_manifest_hash": frozen_hash,
        "frozen_identity_manifest": identity_manifest_payload(manifest),
        "historical_experiment": "pi05-direct-u20-gradient-pro220-v1",
        "historical_config_hash": historical_config_hash,
        "episode_indices": list(QPLANNING_EVAL_EPISODE_INDICES),
        "suites": suites,
        "target_identities": QPLANNING_EVAL_IDENTITIES,
        "critic_horizon": horizon,
        "critic_checkpoint_id": scorer.checkpoint_id,
        "critic_checkpoint_path": checkpoint_path,
        "critic_snapshot_id": scorer.snapshot_id,
        "critic_update": scorer.update,
        "num_candidates": QPLANNING_EVAL_NUM_CANDIDATES,
        "num_elites": QPLANNING_EVAL_NUM_ELITES,
        "temperature": QPLANNING_EVAL_TEMPERATURE,
        "num_inference_steps": QPLANNING_EVAL_DENOISE_STEPS,
        "generated_chunk_size": 50,
        "n_action_steps": QPLANNING_EVAL_ACTION_STEPS,
        "candidate_batch_size": candidate_batch_size,
        "candidate_seed_scheme": "stock_slot0_v1",
        "config_hash": config_hash,
        "video": "off",
    }
    print({
        "experiment": experiment,
        "arm": method,
        "source": source_id,
        "critic": scorer.checkpoint_id,
        "frozen_cohort": "11 suites x 10 tasks x init indices 10,11",
        "identities_requested": len(episodes),
        "rollouts_requested": len(episodes),
        "planner": "64 candidates -> top 16 -> Q-softmax weighted full chunk",
        "decode": "3 Euler steps per candidate",
        "execution": "first 10 of the blended 50 actions, then replan",
        "historical": "exact matched stock source, 10 Euler steps / 10 executed actions",
        "video_and_frames": "off",
        "report_every_completed_identities": QPLANNING_EVAL_REPORT_EVERY,
        "config_hash": config_hash,
    })

    policy, preprocess, postprocess = models.load_pi05(
        repo_id=source_repo, revision=source_revision)
    if int(policy.config.chunk_size) != 50:
        raise ValueError(f"expected 50-action PI chunks, found {policy.config.chunk_size}")
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)
    sampler.enable_encoding_cache(
        policy.model, size=4, model_revision=source_revision)
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=[(method, config)], cohort=f"qplanning_q{horizon}_pro220",
        shard_count=1, shard_index=0, benchmark="libero_pro",
        driver=f"pi05_qplanning_q{horizon}_pro220", run_metadata=metadata,
        provenance=gather_provenance(
            model_repo_id=source_repo, model_revision=source_revision),
        report_every=0, report_every_identities=QPLANNING_EVAL_REPORT_EVERY,
        initial_tally=initial_tally,
        initial_identity_outcomes=initial_outcomes,
        matched_reference_outcomes={"hist stock 10-step": reference},
        progress_include_overall=True,
        rollout_batch_size=1, resume_completed_only=True)
    return {
        "experiment": experiment, "method": method,
        "checkpoint_id": scorer.checkpoint_id, "config_hash": config_hash,
        "identities_requested": len(episodes)}


def run_qplanning_heldout_worker(
        *, q10_checkpoint_path: str | Path, q50_checkpoint_path: str | Path,
        shard_count: int = QPLANNING_HELDOUT_SHARDS, shard_index: int,
        episode_limit: int | None = None,
        candidate_batch_size: int = QPLANNING_EVAL_CANDIDATE_BATCH_SIZE,
        experiment: str = QPLANNING_HELDOUT_EXPERIMENT):
    """Run stock, Q10, and Q50 on one shard of the untouched PRO160 split."""
    from . import models, sampler
    from .experiments import _run_collection
    from .pcp_search.collection import manifest_shard
    from .store import SupabaseStore, gather_provenance

    if (isinstance(shard_count, bool) or int(shard_count) != shard_count
            or int(shard_count) != QPLANNING_HELDOUT_SHARDS):
        raise ValueError(
            f"held-out evaluation is frozen to {QPLANNING_HELDOUT_SHARDS} shards")
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
    if (isinstance(candidate_batch_size, bool)
            or int(candidate_batch_size) != candidate_batch_size
            or not 1 <= int(candidate_batch_size) <= QPLANNING_EVAL_NUM_CANDIDATES):
        raise ValueError("candidate_batch_size must lie in [1, 64]")
    candidate_batch_size = int(candidate_batch_size)

    frozen, all_episodes = _qplanning_heldout_episodes()
    expected_source = {
        "repo_id": frozen.policy_repo_id, "revision": frozen.policy_revision}
    if expected_source != {
            "repo_id": "lerobot/pi05_libero_finetuned",
            "revision": "8e174154ef5f6c60a8da12ae99c303d8963138c1"}:
        raise ValueError("held-out manifest is not pinned to the expected source checkpoint")
    by_item_key = {
        (episode["suite"], int(episode["task_idx"]), int(episode["ep_idx"])): episode
        for episode in all_episodes}
    shard_items = manifest_shard(frozen.items, shard_count, shard_index)
    episodes = [
        by_item_key[(item.suite, int(item.task_idx), int(item.init_state_index))]
        for item in shard_items]
    if len(episodes) != QPLANNING_HELDOUT_IDENTITIES // shard_count:
        raise AssertionError("held-out shard does not contain exactly 40 identities")
    if episode_limit is not None:
        episodes = episodes[:episode_limit]

    device = models.default_device()
    q10_scorer = load_qplanning_scorer(
        q10_checkpoint_path, device=device, expected_horizon=10,
        expected_source_revision=frozen.policy_revision)
    q50_scorer = load_qplanning_scorer(
        q50_checkpoint_path, device=device, expected_horizon=50,
        expected_source_revision=frozen.policy_revision)
    if q10_scorer.source_policy != expected_source:
        raise ValueError("Q10 critic was not trained for the frozen source checkpoint")
    if q50_scorer.source_policy != expected_source:
        raise ValueError("Q50 critic was not trained for the frozen source checkpoint")
    if q10_scorer.snapshot_id != q50_scorer.snapshot_id:
        raise ValueError("Q10 and Q50 critics must use the same dataset snapshot")
    if q10_scorer.update != 8000 or q50_scorer.update != 8000:
        raise ValueError("held-out confirmation requires both final step-8000 checkpoints")

    source_id = f"{frozen.policy_repo_id}@{frozen.policy_revision}"
    methods = [
        build_qplanning_stock_method(source_id),
        build_qplanning_eval_method(
            10, source_id, q10_scorer,
            candidate_batch_size=candidate_batch_size),
        build_qplanning_eval_method(
            50, source_id, q50_scorer,
            candidate_batch_size=candidate_batch_size),
    ]
    store = SupabaseStore()
    config_hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    identity_keys = {_identity_key(episode) for episode in episodes}
    initial_tally, initial_outcomes = _completed_existing_methods(
        store, experiment=experiment, method_config_hashes=config_hashes,
        identity_keys=identity_keys)
    metadata = {
        "source_model_repo_id": frozen.policy_repo_id,
        "source_model_revision": frozen.policy_revision,
        "frozen_collection_manifest_id": frozen.manifest_id,
        "frozen_identity_manifest_hash": identity_manifest_hash(all_episodes),
        "frozen_identity_manifest": identity_manifest_payload(all_episodes),
        "data_split": "heldout",
        "heldout_category": "position_perturb",
        "target_identities": QPLANNING_HELDOUT_IDENTITIES,
        "shard_identities": len(episodes),
        "q10_checkpoint_id": q10_scorer.checkpoint_id,
        "q50_checkpoint_id": q50_scorer.checkpoint_id,
        "q10_checkpoint_path": str(Path(q10_checkpoint_path).expanduser()),
        "q50_checkpoint_path": str(Path(q50_checkpoint_path).expanduser()),
        "num_candidates": QPLANNING_EVAL_NUM_CANDIDATES,
        "num_elites": QPLANNING_EVAL_NUM_ELITES,
        "temperature": QPLANNING_EVAL_TEMPERATURE,
        "planner_num_inference_steps": QPLANNING_EVAL_DENOISE_STEPS,
        "stock_num_inference_steps": 10,
        "generated_chunk_size": 50,
        "n_action_steps": QPLANNING_EVAL_ACTION_STEPS,
        "candidate_batch_size": candidate_batch_size,
        "candidate_seed_scheme": "stock_slot0_v1",
        "config_hashes": config_hashes,
        "video": "off",
        "online_learning": False,
    }
    print({
        "experiment": experiment,
        "split": "untouched position-perturbation PRO160",
        "manifest_id": frozen.manifest_id,
        "shard": f"{shard_index}/{shard_count}",
        "identities_in_shard": len(episodes),
        "rollouts_in_shard": len(episodes) * len(methods),
        "arms": ["stock VLA (10 decode steps)",
                 "Q10 (64 -> top16 softmax, 3 decode steps)",
                 "Q50 (64 -> top16 softmax, 3 decode steps)"],
        "execution": "first 10 actions, then replan",
        "candidate_alignment": (
            "same seed schedule; exact same candidates at the shared first boundary"),
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
        cohort="qplanning_q10_q50_pro_heldout160",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_qplanning_q10_q50_pro_heldout160",
        run_metadata=metadata,
        provenance=gather_provenance(
            model_repo_id=frozen.policy_repo_id,
            model_revision=frozen.policy_revision),
        report_every=0,
        report_every_identities=QPLANNING_HELDOUT_REPORT_EVERY,
        initial_tally=initial_tally,
        initial_identity_outcomes=initial_outcomes,
        matched_reference_outcomes={},
        progress_include_overall=True,
        rollout_batch_size=1,
        resume_completed_only=True)
    return {
        "experiment": experiment,
        "manifest_id": frozen.manifest_id,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "identities_requested": len(episodes),
        "rollouts_requested": len(episodes) * len(methods),
        "q10_checkpoint_id": q10_scorer.checkpoint_id,
        "q50_checkpoint_id": q50_scorer.checkpoint_id,
        "config_hashes": config_hashes,
    }


def validate_qplanning_eval_sentinel(*, horizon: int, checkpoint_id: str,
                                     experiment: str | None = None,
                                     store=None) -> dict:
    """Audit one persisted rollout's planner settings and boundary telemetry."""
    from .store import SupabaseStore

    horizon = int(horizon)
    experiment = experiment or QPLANNING_EVAL_EXPERIMENTS[horizon]
    method = Method.QPLANNING_Q10 if horizon == 10 else Method.QPLANNING_Q50
    store = store or SupabaseStore()
    rows = store.fetch_all(
        "rollouts",
        "rollout_id,status,config_json,ms_candidate_u,video_path,obs_frames_path",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "method", method).eq("status", "completed"),
        order_by=("rollout_id",))
    matched = []
    for row in rows:
        config = row["config_json"]
        if isinstance(config, str):
            config = json.loads(config)
        if config.get("qplanning_ckpt_id") == checkpoint_id:
            matched.append((row, config))
    if not matched:
        raise ValueError("no completed rollout matches this exact critic checkpoint")
    row, config = matched[0]
    expected = {
        "num_samples": 64, "qplanning_n_elites": 16,
        "qplanning_temperature": 1.0, "num_inference_steps": 3,
        "n_action_steps": 10}
    for key, value in expected.items():
        if config.get(key) != value:
            raise AssertionError(f"{key}: expected {value}, found {config.get(key)}")
    if row.get("video_path") or row.get("obs_frames_path"):
        raise AssertionError("Q-Planning sentinel unexpectedly saved video or frames")
    telemetry = row["ms_candidate_u"]
    if isinstance(telemetry, str):
        telemetry = json.loads(telemetry)
    selections = (telemetry or {}).get("qplanning")
    if not selections:
        raise AssertionError("Q-Planning telemetry is missing")
    boundaries = len(selections["q_values"])
    if boundaries < 1:
        raise AssertionError("Q-Planning sentinel has no decision boundaries")
    fields = (
        "best_index", "elite_indices", "elite_weights", "candidate_noise_seeds",
        "n_candidates", "n_elites", "temperature", "denoise_steps",
        "first10_diversity", "full50_diversity", "inference_ms", "n_vf_evals")
    if any(len(selections.get(field, [])) != boundaries for field in fields):
        raise AssertionError("Q-Planning telemetry boundary counts disagree")
    for boundary in range(boundaries):
        if len(selections["q_values"][boundary]) != 64:
            raise AssertionError("a boundary does not contain 64 Q values")
        if len(selections["elite_indices"][boundary]) != 16:
            raise AssertionError("a boundary does not contain 16 elites")
        if abs(sum(selections["elite_weights"][boundary]) - 1.0) > 1e-5:
            raise AssertionError("elite weights do not sum to one")
        if selections["n_candidates"][boundary] != 64:
            raise AssertionError("persisted candidate count is not 64")
        if selections["denoise_steps"][boundary] != 3:
            raise AssertionError("persisted denoise step count is not 3")
    summary = {
        "status": "passed", "experiment": experiment, "horizon": horizon,
        "rollout_id": row["rollout_id"], "decision_boundaries": boundaries,
        "candidates_per_boundary": 64, "elites_per_boundary": 16,
        "decode_steps": 3, "executed_actions_per_boundary": 10,
        "video_and_frames_absent": True}
    print("Q-Planning sentinel:", summary)
    return summary


def validate_qplanning_heldout_sentinel(
        *, q10_checkpoint_id: str, q50_checkpoint_id: str,
        experiment: str = QPLANNING_HELDOUT_EXPERIMENT, store=None) -> dict:
    """Audit that one exact held-out identity completed under all three arms."""
    from .store import SupabaseStore

    store = store or SupabaseStore()
    q10 = validate_qplanning_eval_sentinel(
        horizon=10, checkpoint_id=q10_checkpoint_id,
        experiment=experiment, store=store)
    q50 = validate_qplanning_eval_sentinel(
        horizon=50, checkpoint_id=q50_checkpoint_id,
        experiment=experiment, store=store)
    rows = store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,method,config_json,"
        "video_path,obs_frames_path",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "status", "completed"),
        order_by=("rollout_id",))
    required = {Method.VANILLA, Method.QPLANNING_Q10, Method.QPLANNING_Q50}
    by_identity = {}
    for row in rows:
        if row.get("method") in required:
            by_identity.setdefault(_identity_key(row), {})[row["method"]] = row
    complete = {
        key: arms for key, arms in by_identity.items()
        if required.issubset(arms)}
    if not complete:
        raise ValueError("no held-out identity is complete under stock, Q10, and Q50")
    identity, arms = next(iter(sorted(complete.items())))
    stock = arms[Method.VANILLA]
    config = stock["config_json"]
    if isinstance(config, str):
        config = json.loads(config)
    if config.get("num_inference_steps") != 10 or config.get("n_action_steps") != 10:
        raise AssertionError("stock sentinel is not 10 decode steps / 10 executed actions")
    if config.get("num_samples") is not None:
        raise AssertionError("stock sentinel unexpectedly uses candidate selection")
    if stock.get("video_path") or stock.get("obs_frames_path"):
        raise AssertionError("stock sentinel unexpectedly saved video or frames")
    summary = {
        "status": "passed",
        "experiment": experiment,
        "matched_identity": identity,
        "arms": ["stock VLA", "Q10 planner", "Q50 planner"],
        "stock_decode_steps": 10,
        "planner_decode_steps": q10["decode_steps"],
        "executed_actions_per_boundary": 10,
        "q10_candidates": q10["candidates_per_boundary"],
        "q50_candidates": q50["candidates_per_boundary"],
        "video_and_frames_absent": True,
    }
    print("Held-out three-arm sentinel:", summary)
    return summary
