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
    _completed_rows,
    _identity,
    _matched_historical_stock,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_A1_EXPERIMENT = "smolvla-libero-stock-a1-v1"


def build_smolvla_a1_method():
    config = RolloutConfig(
        num_inference_steps=10,
        n_action_steps=1,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_STOCK_A1, config


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

