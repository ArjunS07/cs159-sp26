"""Stock SmolVLA evaluation with the checkpoint's native one-action horizon."""
from __future__ import annotations

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    SMOLVLA_LIBERO_EXPERIMENT,
    _prepare_libero_episodes,
    _run_collection,
)
from .smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
    _completed_rows,
    _identity,
    _matched_historical_stock,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_A1_EXPERIMENT = "smolvla-libero-stock-a1-v1"
SMOLVLA_A1_PNP_EXPERIMENT = "smolvla-libero-a1-pnp-steps123-k311-v1"


def build_smolvla_a1_method():
    config = RolloutConfig(
        num_inference_steps=10,
        n_action_steps=1,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_STOCK_A1, config


def build_smolvla_a1_pnp_method():
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        num_inference_steps=10,
        n_action_steps=1,
        save_time_uncertainty=True,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_PNP_S123_K311_A1, config


def _existing_outcomes(store, *, experiment, method, config, episodes):
    wanted = {_identity(ep) for ep in episodes}
    outcomes = {}
    for row in _completed_rows(
            store, experiment=experiment, method=method, config=config):
        key = _identity(row)
        if key in wanted:
            outcomes.setdefault(key, {})[method] = bool(row["success"])
    return outcomes


def run_smolvla_a1_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_A1_EXPERIMENT):
    """Evaluate stock SmolVLA with one executed action on worker 83's 400 identities."""
    from . import models

    episodes = _prepare_libero_episodes()
    method, config = build_smolvla_a1_method()
    store = SupabaseStore()
    historical = _matched_historical_stock(store, episodes)
    existing = _existing_outcomes(
        store, experiment=experiment, method=method, config=config, episodes=episodes)
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-83 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "arm": "stock SmolVLA; re-query after one executed action",
        "implementation": (
            "predict_action_chunk -> execute first action -> re-query; equivalent action "
            "horizon to select_action with checkpoint n_action_steps=1"),
        "integration_steps": 10,
        "n_action_steps": 1,
        "generated_chunk_size": 50,
        "reference": f"exact-matched A10 stock from {SMOLVLA_LIBERO_EXPERIMENT}",
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: A1 stock versus exact-matched historical A10 stock every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=models.default_device(),
        experiment=experiment,
        episodes=episodes,
        methods=[(method, config)],
        cohort="smolvla_stock_a1",
        shard_count=1,
        shard_index=0,
        benchmark="libero",
        driver="smolvla_stock_a1_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "integration_steps": 10,
            "n_action_steps": 1,
            "generated_chunk_size": 50,
            "historical_reference_experiment": SMOLVLA_LIBERO_EXPERIMENT,
            "video": "off",
        },
        report_every=0,
        report_every_identities=10,
        matched_reference_outcomes={"historic stock A10": historical},
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )


def run_smolvla_a1_pnp_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_A1_PNP_EXPERIMENT):
    """Evaluate `(1,2,3)/(3,1,1)` P&P while replanning after every action."""
    from . import models

    episodes = _prepare_libero_episodes()
    method, config = build_smolvla_a1_pnp_method()
    store = SupabaseStore()
    historical_a10 = _matched_historical_stock(store, episodes)
    existing = _existing_outcomes(
        store, experiment=experiment, method=method, config=config, episodes=episodes)
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-83 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "arm": "P&P steps=(1,2,3), K=(3,1,1); re-query after one action",
        "integration_steps": 10,
        "n_action_steps": 1,
        "generated_chunk_size": 50,
        "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
        "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
        "reference": (
            "exact-matched historical A10 stock during collection; compare against "
            "notebook 98 A1 stock after both runs complete"),
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print(
        "Periodic output: A1 P&P plus exact-matched historical A10 stock every 10 identities. "
        "The causal P&P comparison is notebook 98's A1 stock arm.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=models.default_device(),
        experiment=experiment,
        episodes=episodes,
        methods=[(method, config)],
        cohort="smolvla_a1_pnp_s123_k311",
        shard_count=1,
        shard_index=0,
        benchmark="libero",
        driver="smolvla_a1_pnp_s123_k311_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "integration_steps": 10,
            "n_action_steps": 1,
            "generated_chunk_size": 50,
            "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "a1_stock_experiment": SMOLVLA_A1_EXPERIMENT,
            "historical_reference_experiment": SMOLVLA_LIBERO_EXPERIMENT,
            "video": "off",
        },
        report_every=0,
        report_every_identities=10,
        matched_reference_outcomes={"historic stock A10": historical_a10},
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )
