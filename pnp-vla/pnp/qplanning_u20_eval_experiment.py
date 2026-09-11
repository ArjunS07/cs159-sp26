"""Matched Q50+U20 coefficient evaluation on the frozen PRO160 identities."""
from __future__ import annotations

from pathlib import Path

from .config import Method, RolloutConfig
from .five_step_diversity_experiment import _identity_key, identity_manifest_hash
from .qplanning_critic import (
    QPlanningU20Scorer, load_qplanning_u20_scorer)
from .qplanning_eval_experiment import (
    QPLANNING_EVAL_ACTION_STEPS, QPLANNING_EVAL_CANDIDATE_BATCH_SIZE,
    QPLANNING_EVAL_DENOISE_STEPS, QPLANNING_EVAL_NUM_CANDIDATES,
    QPLANNING_EVAL_NUM_ELITES, QPLANNING_EVAL_TEMPERATURE,
    QPLANNING_HELDOUT_EXPERIMENT, QPLANNING_HELDOUT_IDENTITIES,
    QPLANNING_HELDOUT_REPORT_EVERY, _completed_existing_methods,
    _qplanning_heldout_episodes, build_qplanning_stock_method)


QPLANNING_U20_EXPERIMENT = "pi05-qplanning-q50-u20-beta-pro-heldout160-v1"
QPLANNING_U20_SHARDS = 2
QPLANNING_U20_BETAS = (0.25, 0.5, 1.0)
QPLANNING_U20_METHODS = {
    0.25: Method.QPLANNING_Q50_U025,
    0.5: Method.QPLANNING_Q50_U050,
    1.0: Method.QPLANNING_Q50_U100,
}


def build_qplanning_u20_method(source_id: str, scorer, *,
                               candidate_batch_size: int):
    beta = float(scorer.uncertainty_beta)
    if beta not in QPLANNING_U20_METHODS:
        raise ValueError(f"unsupported uncertainty coefficient {beta}")
    method = QPLANNING_U20_METHODS[beta]
    return method, RolloutConfig(
        num_samples=QPLANNING_EVAL_NUM_CANDIDATES,
        candidate_set_id=(
            f"{source_id}|q50u20|sample{QPLANNING_EVAL_NUM_CANDIDATES}|"
            f"elite{QPLANNING_EVAL_NUM_ELITES}|beta{beta:.2f}"),
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


def _scorers(checkpoint_path, *, source_revision, device):
    first = load_qplanning_u20_scorer(
        checkpoint_path, uncertainty_beta=QPLANNING_U20_BETAS[0],
        device=device, expected_source_revision=source_revision)
    scorers = [first]
    for beta in QPLANNING_U20_BETAS[1:]:
        scorers.append(QPlanningU20Scorer(
            first.model, checkpoint_id=first.base_checkpoint_id,
            checkpoint_path=first.checkpoint_path,
            snapshot_id=first.snapshot_id, update=first.update,
            source_policy=first.source_policy,
            uncertainty_beta=beta, device=device))
    return scorers


def run_qplanning_u20_heldout_worker(
        *, checkpoint_path: str | Path,
        shard_count: int = QPLANNING_U20_SHARDS, shard_index: int,
        episode_limit: int | None = None,
        candidate_batch_size: int = QPLANNING_EVAL_CANDIDATE_BATCH_SIZE,
        experiment: str = QPLANNING_U20_EXPERIMENT):
    """Run beta={.25,.5,1} and compare with already logged matched stock."""
    from . import models, sampler
    from .experiments import _run_collection
    from .pcp_search.collection import manifest_shard
    from .store import SupabaseStore, gather_provenance

    if int(shard_count) != QPLANNING_U20_SHARDS:
        raise ValueError(f"Q50+U20 evaluation is frozen to {QPLANNING_U20_SHARDS} shards")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError(f"shard_index must lie in [0, {shard_count})")
    shard_count, shard_index = int(shard_count), int(shard_index)
    if episode_limit is not None:
        if isinstance(episode_limit, bool) or int(episode_limit) < 1:
            raise ValueError("episode_limit must be a positive integer or None")
        episode_limit = int(episode_limit)
    if not 1 <= int(candidate_batch_size) <= QPLANNING_EVAL_NUM_CANDIDATES:
        raise ValueError("candidate_batch_size must lie in [1, 64]")
    candidate_batch_size = int(candidate_batch_size)

    frozen, all_episodes = _qplanning_heldout_episodes()
    by_key = {
        (episode["suite"], int(episode["task_idx"]), int(episode["ep_idx"])): episode
        for episode in all_episodes}
    shard_items = manifest_shard(frozen.items, shard_count, shard_index)
    episodes = [
        by_key[(item.suite, int(item.task_idx), int(item.init_state_index))]
        for item in shard_items]
    if len(episodes) != QPLANNING_HELDOUT_IDENTITIES // shard_count:
        raise AssertionError("Q50+U20 shard does not contain exactly 80 identities")
    if episode_limit is not None:
        episodes = episodes[:episode_limit]

    device = models.default_device()
    scorers = _scorers(
        checkpoint_path, source_revision=frozen.policy_revision, device=device)
    expected_source = {
        "repo_id": frozen.policy_repo_id, "revision": frozen.policy_revision}
    if any(scorer.source_policy != expected_source for scorer in scorers):
        raise ValueError("Q50+U20 critic was trained for a different source policy")
    if scorers[0].update != 8000:
        raise ValueError("evaluation requires the final step-8000 Q50+U20 checkpoint")

    source_id = f"{frozen.policy_repo_id}@{frozen.policy_revision}"
    methods = [
        build_qplanning_u20_method(
            source_id, scorer, candidate_batch_size=candidate_batch_size)
        for scorer in scorers]
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
        "data_split": "heldout", "heldout_category": "position_perturb",
        "target_identities": QPLANNING_HELDOUT_IDENTITIES,
        "shard_identities": len(episodes), "uncertainty_betas": list(QPLANNING_U20_BETAS),
        "q50_u20_checkpoint_id": scorers[0].base_checkpoint_id,
        "q50_u20_checkpoint_path": str(Path(checkpoint_path).expanduser()),
        "num_candidates": QPLANNING_EVAL_NUM_CANDIDATES,
        "num_elites": QPLANNING_EVAL_NUM_ELITES,
        "planner_num_inference_steps": QPLANNING_EVAL_DENOISE_STEPS,
        "current_u20_num_inference_steps": 10,
        "current_u20_probe_steps": [3, 4], "current_u20_k": 5,
        "n_action_steps": QPLANNING_EVAL_ACTION_STEPS,
        "candidate_batch_size": candidate_batch_size,
        "historical_stock_experiment": QPLANNING_HELDOUT_EXPERIMENT,
        "historical_stock_config_hash": stock_hash,
        "video": "off", "online_learning": False,
    }
    print({
        "experiment": experiment,
        "split": "same frozen position-perturbation PRO160 as notebook 68",
        "shard": f"{shard_index}/{shard_count}",
        "identities_in_shard": len(episodes),
        "new_rollouts_in_shard": len(episodes) * len(methods),
        "arms": [f"Q50+U20 beta={beta:g}" for beta in QPLANNING_U20_BETAS],
        "candidate_rule": (
            "rank by z(Q) - beta*z(predicted future U20); "
            "Q-softmax blend the selected top 16"),
        "current_u20": "live 10-step decoder, probes (3,4), K=5, first 20 actions",
        "execution": "first 10 blended actions, then replan",
        "historical_reference": (
            f"matched stock from {QPLANNING_HELDOUT_EXPERIMENT}; not rerun"),
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
        cohort="qplanning_q50_u20_beta_pro_heldout160",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_qplanning_q50_u20_beta_pro_heldout160",
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
        rollout_batch_size=1, resume_completed_only=True)
    return {
        "experiment": experiment, "manifest_id": frozen.manifest_id,
        "shard_count": shard_count, "shard_index": shard_index,
        "identities_requested": len(episodes),
        "rollouts_requested": len(episodes) * len(methods),
        "checkpoint_id": scorers[0].base_checkpoint_id,
        "config_hashes": config_hashes,
    }
