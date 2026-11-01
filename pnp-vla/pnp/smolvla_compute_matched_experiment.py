"""Compute-matched SmolVLA baselines for the refined-aggregation study.

Both arms spend 30 sample-equivalent vector-field evaluations per generated chunk:
one integrates a single stock sample for 30 Euler steps; the other integrates three
independent stock samples for 10 steps each and averages their final chunks.
"""
from __future__ import annotations

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    SMOLVLA_LIBERO_EXPERIMENT,
    _prepare_libero_episodes,
    _run_collection,
    identity_shard,
)
from .smolvla_blend_ablation_experiment import (
    SMOLVLA_BLEND_ABLATION_EXPERIMENT,
    SMOLVLA_CONSENSUS_S03_EXPERIMENT,
    build_smolvla_blend_ablation_methods,
    build_smolvla_consensus_s03_method,
)
from .smolvla_consensus_projection_experiment import _matched_historical_schedule
from .smolvla_followup_experiments import (
    _completed_rows,
    _identity,
    _matched_historical_stock,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_COMPUTE_MATCHED_EXPERIMENT = "smolvla-libero-a10-compute-matched-30nfe-v1"
SMOLVLA_COMPUTE_MATCHED_SHARDS = 2
SAMPLE_EQUIVALENT_NFES = 30


def build_smolvla_compute_matched_methods():
    """Return the 30-step stock and 3x10-step raw-average controls."""
    common = dict(
        n_action_steps=LIBERO_ACTION_STEPS,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    stock_30 = RolloutConfig(
        **common,
        num_inference_steps=30,
    )
    average_3x10 = RolloutConfig(
        **common,
        num_inference_steps=10,
        consensus_candidate_count=3,
        consensus_candidate_inference_steps=10,
    )
    return [
        (Method.SMOLVLA_STOCK_30_STEP, stock_30),
        (Method.SMOLVLA_THREE_STOCK_AVERAGE_10_STEP, average_3x10),
    ]


def _matched_outcomes(store, episodes, *, experiment, method, config, label):
    wanted = {_identity(ep) for ep in episodes}
    matched = {}
    for row in _completed_rows(
            store, experiment=experiment, method=method, config=config):
        key = _identity(row)
        if key not in wanted:
            continue
        if key in matched:
            raise ValueError(f"duplicate {label} identity: {key}")
        matched[key] = bool(row["success"])
    missing = sorted(wanted - set(matched))
    if missing:
        raise ValueError(
            f"{label} is missing {len(missing)} matched identities; "
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


def run_smolvla_compute_matched_worker(
        *, shard_count: int = SMOLVLA_COMPUTE_MATCHED_SHARDS,
        shard_index: int = 0, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_COMPUTE_MATCHED_EXPERIMENT):
    """Run one resume-safe half of both compute-matched baseline arms."""
    from . import models

    if shard_count != SMOLVLA_COMPUTE_MATCHED_SHARDS:
        raise ValueError(
            f"compute-matched evaluation is frozen to "
            f"{SMOLVLA_COMPUTE_MATCHED_SHARDS} shards")
    if shard_index not in range(shard_count):
        raise ValueError(f"invalid compute-matched shard {shard_index}/{shard_count}")

    all_episodes = _prepare_libero_episodes()
    episodes = identity_shard(all_episodes, shard_count, shard_index)
    if len(all_episodes) != 400 or len(episodes) != 200:
        raise AssertionError(
            f"expected 400 full/200 shard identities, found "
            f"{len(all_episodes)}/{len(episodes)}")
    methods = build_smolvla_compute_matched_methods()
    store = SupabaseStore()

    historical_stock = _matched_historical_stock(store, episodes)
    historical_refine = _matched_historical_schedule(store, episodes)
    raw_method, raw_config = build_smolvla_blend_ablation_methods()[0]
    historical_raw_average = _matched_outcomes(
        store, episodes, experiment=SMOLVLA_BLEND_ABLATION_EXPERIMENT,
        method=raw_method, config=raw_config, label="historic naive average")
    blend_method, blend_config = build_smolvla_consensus_s03_method()
    historical_refined_aggregation = _matched_outcomes(
        store, episodes, experiment=SMOLVLA_CONSENSUS_S03_EXPERIMENT,
        method=blend_method, config=blend_config, label="historic refined aggregation s=0.3")
    existing = _existing_outcomes(
        store, experiment=experiment, methods=methods, episodes=episodes)

    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "exact standard-LIBERO identities: indices 0-9",
        "full_identities": len(all_episodes),
        "identities_in_this_shard": len(episodes),
        "new_rollouts_in_this_shard": len(episodes) * len(methods),
        "shard": [shard_index, shard_count],
        "arms": {
            "stock_30_step": {
                "samples": 1, "integration_steps": 30,
                "sample_equivalent_nfes_per_chunk": 30,
                "batched_vfield_calls_per_chunk": 30,
            },
            "three_stock_average_10_step": {
                "samples": 3, "integration_steps_per_sample": 10,
                "sample_equivalent_nfes_per_chunk": 30,
                "batched_vfield_calls_per_chunk": 10,
                "projection": "none",
            },
        },
        "candidate_independence": (
            "candidate 0 uses the stock initial noise; candidates 1-2 use independent "
            "isolated candidate-noise draws"),
        "averaging": (
            "arithmetic mean of continuous action dimensions; candidate-0 gripper "
            "is retained only when candidate gripper signs disagree"),
        "n_action_steps": LIBERO_ACTION_STEPS,
        "generated_chunk_size": 50,
        "rollout_batch_size": rollout_batch_size,
        "latency": "stored per rollout; candidate arm batches its three samples",
        "video": "off",
    })
    print(
        "Periodic output: both new 30-NFE arms plus exact-matched historical stock, "
        "refinement, naive average, and refined aggregation every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    return _run_collection(
        store=store,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=models.default_device(),
        experiment=experiment,
        episodes=episodes,
        methods=methods,
        cohort="smolvla_compute_matched_30nfe",
        shard_count=shard_count,
        shard_index=shard_index,
        benchmark="libero",
        driver="smolvla_compute_matched_30nfe_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities_full": len(all_episodes),
            "target_identities_shard": len(episodes),
            "episode_idxs": list(range(10)),
            "sample_equivalent_nfes_per_chunk": SAMPLE_EQUIVALENT_NFES,
            "stock_30_integration_steps": 30,
            "candidate_count": 3,
            "candidate_integration_steps": 10,
            "candidate_projection": None,
            "candidate_batched_vfield_calls_per_chunk": 10,
            "n_action_steps": LIBERO_ACTION_STEPS,
            "generated_chunk_size": 50,
            "historical_stock_experiment": SMOLVLA_LIBERO_EXPERIMENT,
            "historical_naive_average_experiment": SMOLVLA_BLEND_ABLATION_EXPERIMENT,
            "historical_refined_aggregation_experiment": SMOLVLA_CONSENSUS_S03_EXPERIMENT,
            "video": "off",
        },
        report_every=0,
        report_every_identities=10,
        matched_reference_outcomes={
            "historic stock A10": historical_stock,
            "historic P&P A10": historical_refine,
            "historic naive average": historical_raw_average,
            "historic refined aggregation s03": historical_refined_aggregation,
        },
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )
