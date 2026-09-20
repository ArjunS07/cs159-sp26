"""Cross-decision SmolVLA chunk-consensus projection experiment."""
from __future__ import annotations

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    _prepare_libero_episodes,
    _run_collection,
)
from .smolvla_blend_ablation_experiment import _matched_historical_projection_k3
from .smolvla_consensus_projection_experiment import PROJECTION_STEP, _matched_historical_schedule
from .smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
    _completed_rows,
    _identity,
    _matched_historical_stock,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_TEMPORAL_BLEND_EXPERIMENT = "smolvla-libero-a10-temporal-overlap-project-s05-k3-v1"


def build_smolvla_temporal_blend_methods():
    common = dict(
        temporal_overlap_consensus=True,
        consensus_projection_k=3,
        consensus_projection_step=PROJECTION_STEP,
        num_inference_steps=10,
        n_action_steps=LIBERO_ACTION_STEPS,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    stock = RolloutConfig(**common)
    refined = RolloutConfig(
        **common,
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        save_time_uncertainty=True,
    )
    return [
        (Method.SMOLVLA_TEMPORAL_STOCK_PROJECT_K3, stock),
        (Method.SMOLVLA_TEMPORAL_REFINED_PROJECT_K3, refined),
    ]


def _existing_outcomes(store, experiment, methods, episodes):
    wanted = {_identity(ep) for ep in episodes}
    outcomes = {}
    for method, config in methods:
        for row in _completed_rows(
                store, experiment=experiment, method=method, config=config):
            key = _identity(row)
            if key in wanted:
                outcomes.setdefault(key, {})[method] = bool(row["success"])
    return outcomes


def run_smolvla_temporal_blend_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_TEMPORAL_BLEND_EXPERIMENT):
    """Compare stock versus refined parents under aligned cross-decision averaging."""
    from . import models

    episodes = _prepare_libero_episodes()
    methods = build_smolvla_temporal_blend_methods()
    store = SupabaseStore()
    historical_stock = _matched_historical_stock(store, episodes)
    historical_refine = _matched_historical_schedule(store, episodes)
    historical_projected = _matched_historical_projection_k3(store, episodes)
    existing = _existing_outcomes(store, experiment, methods, episodes)

    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-85/96 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "arms": [
            "stock previous/current parents",
            "P&P-refined previous/current parents",
        ],
        "alignment": "previous[10:50] averages current[0:40]; current[40:50] unchanged",
        "stored_previous": "final projected chunk used for execution",
        "first_decision": "project current parent alone",
        "refined_parent_schedule": "steps=(1,2,3), K=(3,1,1)",
        "projection": {"step": 5, "s": 0.5, "k": 3},
        "integration_steps": 10,
        "n_action_steps": 10,
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: both temporal arms and exact-matched historical references every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=methods, cohort="smolvla_temporal_overlap_project_k3_a10",
        shard_count=1, shard_index=0, benchmark="libero",
        driver="smolvla_temporal_overlap_project_k3_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "overlap_shift": 10,
            "overlap_actions": 40,
            "current_only_tail_actions": 10,
            "stored_previous_stage": "final_projected_chunk",
            "refined_parent_pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "refined_parent_pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "projection_step": PROJECTION_STEP,
            "projection_k": 3,
            "n_action_steps": 10,
            "video": "off",
        },
        report_every=0, report_every_identities=10,
        matched_reference_outcomes={
            "historic stock A10": historical_stock,
            "historic P&P A10": historical_refine,
            "historic K3 blend A10": historical_projected,
        },
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )
