"""SmolVLA consensus ablations: raw averaging and alternative parent candidates."""
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
from .smolvla_a1_experiment import _existing_outcomes
from .smolvla_consensus_projection_experiment import (
    PROJECTION_STEP,
    SMOLVLA_CONSENSUS_PROJECTION_EXPERIMENT,
    _matched_historical_schedule,
    build_smolvla_consensus_projection_methods,
)
from .smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
    _completed_rows,
    _identity,
    _matched_historical_stock,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_BLEND_ABLATION_EXPERIMENT = "smolvla-libero-a10-blend-ablation-v1"
SMOLVLA_CONSENSUS_A1_EXPERIMENT = "smolvla-libero-a1-consensus-project-s05-k3-v1"
SMOLVLA_CONSENSUS_A20_EXPERIMENT = "smolvla-libero-a20-consensus-project-s05-k3-v1"
SMOLVLA_MIXED_PARENT_EXPERIMENT = (
    "smolvla-libero-a10-two-stock-two-refine-project-s05-k3-v1")


def build_smolvla_blend_ablation_methods():
    common = dict(
        num_inference_steps=10,
        n_action_steps=LIBERO_ACTION_STEPS,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    raw_average = RolloutConfig(
        **common,
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        consensus_average_only=True,
    )
    four_candidate_project = RolloutConfig(
        **common,
        consensus_candidate_count=4,
        consensus_candidate_inference_steps=5,
        consensus_projection_k=3,
        consensus_projection_step=PROJECTION_STEP,
    )
    return [
        (Method.SMOLVLA_STOCK_REFINE_RAW_AVERAGE, raw_average),
        (Method.SMOLVLA_FOUR_CANDIDATE_PROJECT_K3, four_candidate_project),
    ]


def build_smolvla_consensus_a1_method():
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        num_inference_steps=10,
        n_action_steps=1,
        consensus_projection_k=3,
        consensus_projection_step=PROJECTION_STEP,
        save_time_uncertainty=True,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_CONSENSUS_PROJECT_K3_A1, config


def build_smolvla_consensus_a20_method():
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        num_inference_steps=10,
        n_action_steps=20,
        consensus_projection_k=3,
        consensus_projection_step=PROJECTION_STEP,
        save_time_uncertainty=True,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_CONSENSUS_PROJECT_K3_A20, config


def build_smolvla_mixed_parent_method():
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        num_inference_steps=10,
        n_action_steps=10,
        consensus_candidate_count=4,
        consensus_candidate_inference_steps=10,
        consensus_candidate_refine_count=2,
        consensus_projection_k=3,
        consensus_projection_step=PROJECTION_STEP,
        save_time_uncertainty=True,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_TWO_STOCK_TWO_REFINE_PROJECT_K3, config


def _matched_historical_projection_k3(store, episodes):
    method, config = build_smolvla_consensus_projection_methods()[0]
    wanted = {_identity(ep) for ep in episodes}
    matched = {}
    for row in _completed_rows(
            store, experiment=SMOLVLA_CONSENSUS_PROJECTION_EXPERIMENT,
            method=method, config=config):
        key = _identity(row)
        if key in wanted:
            if key in matched:
                raise ValueError(f"duplicate historical K3 projection identity: {key}")
            matched[key] = bool(row["success"])
    missing = sorted(wanted - set(matched))
    if missing:
        raise ValueError(
            f"worker 96 K3 projection is missing {len(missing)} identities; "
            f"first missing={missing[0]}")
    return matched


def run_smolvla_blend_ablation_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_BLEND_ABLATION_EXPERIMENT):
    """Run both A10 blend ablations over the exact worker-85/96 cohort."""
    from . import models

    episodes = _prepare_libero_episodes()
    methods = build_smolvla_blend_ablation_methods()
    store = SupabaseStore()
    historical_stock = _matched_historical_stock(store, episodes)
    historical_refine = _matched_historical_schedule(store, episodes)
    existing = {}
    wanted = {_identity(ep) for ep in episodes}
    for method, config in methods:
        for row in _completed_rows(
                store, experiment=experiment, method=method, config=config):
            key = _identity(row)
            if key in wanted:
                existing.setdefault(key, {})[method] = bool(row["success"])

    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-85/96 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "arms": [
            "raw 50/50 stock + refined average (no projection)",
            "four ordinary 5-step candidates, batched average, K3 s=0.5 projection",
        ],
        "parent_refine": "P&P steps=(1,2,3), K=(3,1,1)",
        "candidate_arm": {"count": 4, "integration_steps": 5},
        "projection": {"step": 5, "s": 0.5, "k": 3},
        "execution": {"integration_steps": 10, "n_action_steps": 10},
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: both arms and exact-matched historical stock/refine every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=methods, cohort="smolvla_blend_ablation_a10", shard_count=1,
        shard_index=0, benchmark="libero", driver="smolvla_blend_ablation_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "raw_average_parents": ["stock", "P&P (1,2,3)/(3,1,1)"],
            "candidate_count": 4,
            "candidate_integration_steps": 5,
            "projection_step": PROJECTION_STEP,
            "projection_k": 3,
            "n_action_steps": 10,
            "video": "off",
        },
        report_every=0, report_every_identities=10,
        matched_reference_outcomes={
            "historic stock A10": historical_stock,
            "historic P&P A10": historical_refine,
        },
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )


def run_smolvla_consensus_a1_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_CONSENSUS_A1_EXPERIMENT):
    """Run the successful K3 stock/refine projection while replanning every action."""
    from . import models

    episodes = _prepare_libero_episodes()
    method, config = build_smolvla_consensus_a1_method()
    store = SupabaseStore()
    historical_stock = _matched_historical_stock(store, episodes)
    historical_projected_a10 = _matched_historical_projection_k3(store, episodes)
    existing = _existing_outcomes(
        store, experiment=experiment, method=method, config=config, episodes=episodes)
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-85/96 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "arm": "stock + P&P-refined average, K3 s=0.5 projection; replan every action",
        "parent_refine": "P&P steps=(1,2,3), K=(3,1,1)",
        "projection": {"step": 5, "s": 0.5, "k": 3},
        "integration_steps": 10,
        "n_action_steps": 1,
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: A1 projection plus exact-matched stock A10 and projected A10 every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=[(method, config)], cohort="smolvla_consensus_project_k3_a1",
        shard_count=1, shard_index=0, benchmark="libero",
        driver="smolvla_consensus_project_k3_a1_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "parent_pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "parent_pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "projection_step": PROJECTION_STEP,
            "projection_k": 3,
            "n_action_steps": 1,
            "video": "off",
        },
        report_every=0, report_every_identities=10,
        matched_reference_outcomes={
            "historic stock A10": historical_stock,
            "historic K3 projection A10": historical_projected_a10,
        },
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )


def run_smolvla_consensus_a20_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_CONSENSUS_A20_EXPERIMENT):
    """Run notebook-96 K3 projection while executing 20 actions per decision."""
    from . import models

    episodes = _prepare_libero_episodes()
    method, config = build_smolvla_consensus_a20_method()
    store = SupabaseStore()
    historical_stock = _matched_historical_stock(store, episodes)
    historical_projected_a10 = _matched_historical_projection_k3(store, episodes)
    existing = _existing_outcomes(
        store, experiment=experiment, method=method, config=config, episodes=episodes)
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-85/96 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "arm": "stock + refined average, K3 s=0.5 projection; execute 20 actions",
        "parent_refine": "P&P steps=(1,2,3), K=(3,1,1)",
        "projection": {"step": 5, "s": 0.5, "k": 3},
        "integration_steps": 10,
        "n_action_steps": 20,
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: A20 projection plus exact-matched stock A10 and projected A10 every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=[(method, config)], cohort="smolvla_consensus_project_k3_a20",
        shard_count=1, shard_index=0, benchmark="libero",
        driver="smolvla_consensus_project_k3_a20_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "parent_pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "parent_pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "projection_step": PROJECTION_STEP,
            "projection_k": 3,
            "n_action_steps": 20,
            "video": "off",
        },
        report_every=0, report_every_identities=10,
        matched_reference_outcomes={
            "historic stock A10": historical_stock,
            "historic K3 projection A10": historical_projected_a10,
        },
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )


def run_smolvla_mixed_parent_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_MIXED_PARENT_EXPERIMENT):
    """Average two stock and two refined 10-step parents, then project at s=0.5."""
    from . import models

    episodes = _prepare_libero_episodes()
    method, config = build_smolvla_mixed_parent_method()
    store = SupabaseStore()
    historical_stock = _matched_historical_stock(store, episodes)
    historical_projected_a10 = _matched_historical_projection_k3(store, episodes)
    existing = _existing_outcomes(
        store, experiment=experiment, method=method, config=config, episodes=episodes)
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-85/96 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "arm": "average two stock + two P&P-refined parents, then K3 projection",
        "parent_count": 4,
        "stock_parents": 2,
        "refined_parents": 2,
        "parent_integration_steps": 10,
        "parent_refine": "P&P steps=(1,2,3), K=(3,1,1)",
        "projection": {"step": 5, "s": 0.5, "k": 3},
        "n_action_steps": 10,
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: mixed-parent projection plus exact-matched stock and K3 projection every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=[(method, config)], cohort="smolvla_mixed_parent_project_k3_a10",
        shard_count=1, shard_index=0, benchmark="libero",
        driver="smolvla_mixed_parent_project_k3_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "candidate_count": 4,
            "stock_candidate_count": 2,
            "refined_candidate_count": 2,
            "candidate_integration_steps": 10,
            "candidate_pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "candidate_pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "projection_step": PROJECTION_STEP,
            "projection_k": 3,
            "n_action_steps": 10,
            "video": "off",
        },
        report_every=0, report_every_identities=10,
        matched_reference_outcomes={
            "historic stock A10": historical_stock,
            "historic K3 projection A10": historical_projected_a10,
        },
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )
