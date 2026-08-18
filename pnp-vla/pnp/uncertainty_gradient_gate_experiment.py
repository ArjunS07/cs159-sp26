"""Online decoded-action trust-region pilot on the established 220 PRO identities."""
from __future__ import annotations

import pandas as pd

from .config import Method, RolloutConfig
from .uncertainty_gradient_experiment import (
    DIRECT_U20_GRADIENT_EPISODE_INDICES,
    DIRECT_U20_GRADIENT_EXPERIMENT,
    DIRECT_U20_GRADIENT_RMS,
    build_direct_u20_gradient_methods,
)


U20_ACTION_GATE_EXPERIMENT = "pi05-u20-gradient-action-gate-pro220-v1"
U20_ACTION_GATE_THRESHOLDS = (0.015, 0.020)


def build_u20_action_gate_methods(
        thresholds=U20_ACTION_GATE_THRESHOLDS,
        step_size: float = DIRECT_U20_GRADIENT_RMS):
    thresholds = tuple(float(value) for value in thresholds)
    if thresholds != U20_ACTION_GATE_THRESHOLDS:
        raise ValueError(
            f"pilot thresholds are predeclared as {U20_ACTION_GATE_THRESHOLDS}, "
            f"received {thresholds}")
    common = dict(
        pnp_steps=(3, 4), pnp_k=5, n_action_steps=10,
        save_time_uncertainty=True, skip_unused_renders=True, render_lead=2,
        uncertainty_gradient_mode="descent",
        uncertainty_gradient_step_size=float(step_size),
        uncertainty_gradient_horizon=20)
    return [
        (Method.U20_GRADIENT_GATE_015, RolloutConfig(
            **common, uncertainty_gradient_action_rms_max=thresholds[0])),
        (Method.U20_GRADIENT_GATE_020, RolloutConfig(
            **common, uncertainty_gradient_action_rms_max=thresholds[1])),
    ]


def _historical_comparators(store, target_identities):
    from .diversity import DIVERSITY_PAIR_KEYS

    old_methods = dict(build_direct_u20_gradient_methods(DIRECT_U20_GRADIENT_RMS))
    selected = {
        Method.UNCERTAINTY: old_methods[Method.UNCERTAINTY],
        Method.U20_GRADIENT: old_methods[Method.U20_GRADIENT],
    }
    hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in selected.items()}
    rows = pd.DataFrame(store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq(
            "experiment", DIRECT_U20_GRADIENT_EXPERIMENT),
        order_by=("rollout_id",)))
    if rows.empty:
        raise ValueError(
            f"{DIRECT_U20_GRADIENT_EXPERIMENT} has no rows; finish notebook 48 first")
    expected_hash = rows.method.map(hashes)
    rows = rows[
        rows.config_hash.eq(expected_hash)
        & rows.status.eq("completed")
        & rows[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(target_identities)]
    references = {}
    for method, label in (
            (Method.UNCERTAINTY, "stock baseline"),
            (Method.U20_GRADIENT, "ungated gradient")):
        arm = rows[rows.method.eq(method)]
        identities = set(arm[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1))
        if identities != target_identities or len(arm) != len(target_identities):
            raise ValueError(
                f"historical {label} must contain exactly {len(target_identities)} "
                f"pilot identities; found {len(arm)} rows / {len(identities)} identities")
        references[label] = (
            arm.groupby("suite").success.mean().astype(float).to_dict())
    return references


def run_u20_action_gate_worker(
        *, shard_count: int = 4, shard_index: int = 0,
        episode_indices=DIRECT_U20_GRADIENT_EPISODE_INDICES,
        episode_limit: int | None = None,
        step_size: float = DIRECT_U20_GRADIENT_RMS,
        thresholds=U20_ACTION_GATE_THRESHOLDS,
        manifest_hash: str = "", source_model_revision: str = "",
        experiment: str = U20_ACTION_GATE_EXPERIMENT):
    """Run two online displacement caps on the exact notebook-48 PRO pilot cohort."""
    from . import models
    from .diversity import (
        DIVERSITY_PAIR_KEYS, SOURCE_FRACTIONAL_SHORT_SUITES,
        source_checkpoint_model_source)
    from .experiments import (
        _prepare_libero_pro_expanded_episodes, _run_collection,
        expanded_pro_suites, identity_shard)
    from .store import SupabaseStore, gather_provenance

    episode_indices = tuple(map(int, episode_indices))
    if episode_indices != DIRECT_U20_GRADIENT_EPISODE_INDICES:
        raise ValueError(
            f"pilot episode indices are fixed at {DIRECT_U20_GRADIENT_EPISODE_INDICES}")
    if not manifest_hash:
        raise ValueError("manifest_hash is required from the shared v2 Drive manifest")
    if not source_model_revision:
        raise ValueError("source_model_revision is required from the shared v2 manifest")

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
    target_identities = {
        (ep["suite"], ep["task_idx"], ep["ep_idx"], ep["init_state_hash"])
        for ep in manifest}
    historical_sr = _historical_comparators(store, target_identities)

    episodes = identity_shard(manifest, shard_count, shard_index)
    if episode_limit is not None:
        if (isinstance(episode_limit, bool)
                or int(episode_limit) != episode_limit or episode_limit < 1):
            raise ValueError("episode_limit must be a positive integer or None")
        episodes = episodes[:int(episode_limit)]
    methods = build_u20_action_gate_methods(thresholds, step_size)
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
        "benchmark": "LIBERO-PRO",
        "source": f"{source_repo}@{source_revision}",
        "cohort": "same 220 identities as notebook 48",
        "identities_in_shard": len(episodes),
        "rollouts_in_shard": len(episodes) * len(methods),
        "online_caps": list(map(float, thresholds)),
        "metric": "RMS over first 10 postprocessed arm actions; gripper excluded",
        "execution": "accept gradient chunk at each boundary iff RMS <= cap",
    })
    print("Historical columns are exact matched stock and ungated-gradient rollouts.")

    policy, preprocess, postprocess = models.load_pi05(
        repo_id=source_repo, revision=source_revision)
    if int(policy.config.chunk_size) != 50:
        raise ValueError(
            f"action-gate design requires a 50-action chunk, found {policy.config.chunk_size}")
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)

    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=methods, cohort="u20_action_gate_pro220",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_u20_action_gate_pro220",
        run_metadata={
            "source_model_repo_id": source_repo,
            "source_model_revision": source_revision,
            "bootstrap_manifest_hash": manifest_hash,
            "episode_indices": list(episode_indices),
            "target_identities": len(manifest),
            "pnp_k": 5, "pnp_steps": [3, 4],
            "uncertainty_horizon": 20,
            "gradient_update_rms": float(step_size),
            "action_rms_thresholds": list(map(float, thresholds)),
            "n_action_steps": 10,
            "requested_methods": [method for method, _ in methods]},
        provenance=gather_provenance(
            model_repo_id=source_repo, model_revision=source_revision),
        report_every=20, initial_tally=initial_tally,
        historical_sr=historical_sr)
