"""SmolVLA stock/refinement averaging followed by flow-manifold projection."""
from __future__ import annotations

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    SMOLVLA_LIBERO_EXPERIMENT,
    _prepare_libero_episodes,
    _run_collection,
)
from .smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_EXPERIMENT,
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
    _completed_rows,
    _identity,
    _matched_historical_stock,
    build_smolvla_schedule_method,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_CONSENSUS_PROJECTION_EXPERIMENT = (
    "smolvla-libero-a10-stock-refine-average-project-s05-k35-v1")
PROJECTION_STEP = 5
PROJECTION_K_VALUES = (3, 5)


def build_smolvla_consensus_projection_methods():
    """K=3/K=5 projection arms with one identical stock/refined parent pair."""
    methods = []
    names = {
        3: Method.SMOLVLA_CONSENSUS_PROJECT_K3,
        5: Method.SMOLVLA_CONSENSUS_PROJECT_K5,
    }
    for projection_k in PROJECTION_K_VALUES:
        config = RolloutConfig(
            pnp_steps=SMOLVLA_SCHEDULE_STEPS,
            pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
            pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
            refine=True,
            num_inference_steps=10,
            n_action_steps=LIBERO_ACTION_STEPS,
            consensus_projection_k=projection_k,
            consensus_projection_step=PROJECTION_STEP,
            save_time_uncertainty=True,
            save_trajectory=True,
            skip_unused_renders=LIBERO_SKIP_RENDERS,
            render_lead=LIBERO_RENDER_LEAD,
        )
        methods.append((names[projection_k], config))
    return methods


def _matched_historical_schedule(store, episodes) -> dict[tuple, bool]:
    method, config = build_smolvla_schedule_method()
    rows = _completed_rows(
        store, experiment=SMOLVLA_SCHEDULE_EXPERIMENT,
        method=method, config=config)
    wanted = {_identity(ep) for ep in episodes}
    matched = {}
    for row in rows:
        key = _identity(row)
        if key in wanted:
            if key in matched:
                raise ValueError(f"duplicate historical P&P schedule identity: {key}")
            matched[key] = bool(row["success"])
    missing = sorted(wanted - set(matched))
    if missing:
        raise ValueError(
            f"worker 85 historical schedule is missing {len(missing)} identities; "
            f"first missing={missing[0]}")
    return matched


def _existing_outcomes(store, *, experiment, methods, episodes):
    wanted = {_identity(ep) for ep in episodes}
    outcomes = {}
    for method, config in methods:
        for row in _completed_rows(
                store, experiment=experiment, method=method, config=config):
            key = _identity(row)
            if key in wanted:
                outcomes.setdefault(key, {})[method] = bool(row["success"])
    return outcomes


def run_smolvla_consensus_projection_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_CONSENSUS_PROJECTION_EXPERIMENT):
    """Evaluate K=3 and K=5 projection on worker 85's exact 400 identities."""
    from . import models

    episodes = _prepare_libero_episodes()
    methods = build_smolvla_consensus_projection_methods()
    store = SupabaseStore()
    historical_stock = _matched_historical_stock(store, episodes)
    historical_refine = _matched_historical_schedule(store, episodes)
    existing = _existing_outcomes(
        store, experiment=experiment, methods=methods, episodes=episodes)
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-85 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "new_arms": ["average + project K=3", "average + project K=5"],
        "parents": "exact stock + completed P&P steps=(1,2,3), K=(3,1,1)",
        "consensus": "50/50 arm-action average; stock gripper on sign disagreement",
        "projection": "forward-noise to s=0.5, fixed-time P&P, integrate s=0.5 to 0",
        "projection_k": list(PROJECTION_K_VALUES),
        "integration_steps": 10,
        "n_action_steps": LIBERO_ACTION_STEPS,
        "generated_chunk_size": 50,
        "historical_references": [
            SMOLVLA_LIBERO_EXPERIMENT, SMOLVLA_SCHEDULE_EXPERIMENT],
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print(
        "Periodic output: both new arms plus exact-matched historical stock and "
        "worker-85 refinement every 10 completed identities.")

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
        methods=methods,
        cohort="smolvla_consensus_projection_k3_k5",
        shard_count=1,
        shard_index=0,
        benchmark="libero",
        driver="smolvla_consensus_projection_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "parent_pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "parent_pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "consensus_weights": [0.5, 0.5],
            "projection_step": PROJECTION_STEP,
            "projection_s": 0.5,
            "projection_k_values": list(PROJECTION_K_VALUES),
            "integration_steps": 10,
            "n_action_steps": LIBERO_ACTION_STEPS,
            "generated_chunk_size": 50,
            "historical_stock_experiment": SMOLVLA_LIBERO_EXPERIMENT,
            "historical_refine_experiment": SMOLVLA_SCHEDULE_EXPERIMENT,
            "video": "off",
        },
        report_every=0,
        report_every_identities=10,
        matched_reference_outcomes={
            "historic stock VLA": historical_stock,
            "historic P&P (1,2,3)/(3,1,1)": historical_refine,
        },
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )

