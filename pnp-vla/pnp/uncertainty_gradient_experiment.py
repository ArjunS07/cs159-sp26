"""Paired online direct-U20-gradient pilot on the established 220 PRO identities."""
from __future__ import annotations

import pandas as pd

from .config import Method, RolloutConfig


DIRECT_U20_GRADIENT_EXPERIMENT = "pi05-direct-u20-gradient-pro220-v1"
DIRECT_U20_GRADIENT_RMS = 0.01
DIRECT_U20_GRADIENT_EPISODE_INDICES = (10, 11)


def build_direct_u20_gradient_methods(step_size: float = DIRECT_U20_GRADIENT_RMS):
    """Baseline, true-gradient descent, and equal-compute/equal-RMS random control."""
    common = dict(
        pnp_steps=(3, 4), pnp_k=5, n_action_steps=10,
        save_time_uncertainty=True, skip_unused_renders=True, render_lead=2)
    return [
        (Method.UNCERTAINTY, RolloutConfig(**common)),
        (Method.U20_GRADIENT, RolloutConfig(
            **common, uncertainty_gradient_mode="descent",
            uncertainty_gradient_step_size=float(step_size),
            uncertainty_gradient_horizon=20)),
        (Method.LATENT_RANDOM_CONTROL, RolloutConfig(
            **common, uncertainty_gradient_mode="random",
            uncertainty_gradient_step_size=float(step_size),
            uncertainty_gradient_horizon=20)),
    ]


def run_direct_u20_gradient_worker(
        *, shard_count: int = 4, shard_index: int = 0,
        episode_indices=DIRECT_U20_GRADIENT_EPISODE_INDICES,
        episode_limit: int | None = None,
        step_size: float = DIRECT_U20_GRADIENT_RMS,
        manifest_hash: str = "", source_model_revision: str = "",
        experiment: str = DIRECT_U20_GRADIENT_EXPERIMENT):
    """Run the three paired arms on the same 220 identities used by notebook 35."""
    from . import models
    from .diversity import (
        DIVERSITY_PAIR_KEYS, SOURCE_FRACTIONAL_SHORT_SUITES,
        source_checkpoint_model_source)
    from .experiments import (
        _prepare_libero_pro_expanded_episodes, _run_collection,
        expanded_pro_suites, identity_shard)
    from .store import SupabaseStore, gather_provenance

    episode_indices = tuple(map(int, episode_indices))
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode_indices must be non-empty and unique")
    if any(index < 0 for index in episode_indices):
        raise ValueError("episode_indices must be non-negative")
    if not manifest_hash:
        raise ValueError("manifest_hash is required; load the shared v2 Drive manifest")
    if not source_model_revision:
        raise ValueError("source_model_revision is required from the shared v2 manifest")
    if float(step_size) <= 0:
        raise ValueError("step_size must be positive")

    store = SupabaseStore()
    source_repo, source_revision = source_checkpoint_model_source(
        store, expected_revision=source_model_revision)
    suites = [suite for suite in expanded_pro_suites()
              if suite not in SOURCE_FRACTIONAL_SHORT_SUITES]
    if len(suites) != 11:
        raise ValueError(f"expected 11 held-out PRO suites, found {len(suites)}")
    manifest = _prepare_libero_pro_expanded_episodes(
        suites=suites, episode_idxs=episode_indices)
    expected_identities = len(suites) * 10 * len(episode_indices)
    if len(manifest) != expected_identities:
        raise ValueError(
            f"expected {expected_identities} identities, found {len(manifest)}")
    if {int(ep["ep_idx"]) for ep in manifest} != set(episode_indices):
        raise ValueError("manifest contains unexpected episode indices")

    episodes = identity_shard(manifest, shard_count, shard_index)
    if episode_limit is not None:
        if isinstance(episode_limit, bool) or int(episode_limit) != episode_limit \
                or episode_limit < 1:
            raise ValueError("episode_limit must be a positive integer or None")
        episodes = episodes[:int(episode_limit)]

    methods = build_direct_u20_gradient_methods(step_size)
    shard_identities = {
        (ep["suite"], ep["task_idx"], ep["ep_idx"], ep["init_state_hash"])
        for ep in episodes}
    expected_hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    existing = pd.DataFrame(store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq("experiment", experiment),
        order_by=("rollout_id",)))
    if not existing.empty:
        expected = existing.method.map(expected_hashes)
        existing = existing[
            existing.config_hash.eq(expected)
            & existing.status.eq("completed")
            & existing[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(shard_identities)]
    initial_tally = {} if existing.empty else {
        (suite, method): [len(group), int(group.success.astype(bool).sum())]
        for (suite, method), group in existing.groupby(["suite", "method"], sort=True)}

    print({
        "experiment": experiment,
        "source": f"{source_repo}@{source_revision}",
        "cohort": "11 suites x 10 tasks x held-out init indices 10,11",
        "excluded_short_asset_suites": list(SOURCE_FRACTIONAL_SHORT_SUITES),
        "target_identities": len(manifest),
        "identities_in_shard": len(episodes),
        "rollouts_in_shard": len(episodes) * len(methods),
        "arms": [method for method, _ in methods],
        "probe": "K=5 U20 at zero-based Euler steps (3,4)",
        "gradient_update_rms": float(step_size),
        "execution": "10 of each generated 50-action chunk",
    })
    print("Periodic output: paired SR for baseline, true U20 descent, and random control.")

    policy, preprocess, postprocess = models.load_pi05(
        repo_id=source_repo, revision=source_revision)
    if int(policy.config.chunk_size) != 50:
        raise ValueError(
            f"direct-gradient design requires a 50-action chunk, found {policy.config.chunk_size}")
    # We differentiate only with respect to the live sampler latent. Freezing weights prevents
    # construction of parameter-gradient state and materially reduces online memory.
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)

    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=methods, cohort="direct_u20_gradient_pro220",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_direct_u20_gradient_pro220",
        run_metadata={
            "source_model_repo_id": source_repo,
            "source_model_revision": source_revision,
            "bootstrap_manifest_hash": manifest_hash,
            "suites": suites,
            "excluded_short_asset_suites": list(SOURCE_FRACTIONAL_SHORT_SUITES),
            "episode_indices": list(episode_indices),
            "target_identities": len(manifest),
            "pnp_k": 5, "pnp_steps": [3, 4],
            "uncertainty_horizon": 20,
            "gradient_update_rms": float(step_size),
            "n_action_steps": 10,
            "requested_methods": [method for method, _ in methods]},
        provenance=gather_provenance(
            model_repo_id=source_repo, model_revision=source_revision),
        # Ten complete three-arm identities per periodic report.
        report_every=30, initial_tally=initial_tally, historical_sr=False)
