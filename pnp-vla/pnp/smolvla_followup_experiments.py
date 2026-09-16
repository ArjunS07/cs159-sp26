"""Matched SmolVLA LIBERO evaluation for an early variable-K P&P schedule."""
from __future__ import annotations

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    SMOLVLA_LIBERO_EXPERIMENT,
    _prepare_libero_episodes,
    _run_collection,
    build_smolvla_libero_methods,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_SCHEDULE_EXPERIMENT = "smolvla-libero-a10-pnp-steps123-k311-v1"
SMOLVLA_SCHEDULE_STEPS = (1, 2, 3)
SMOLVLA_SCHEDULE_K_BY_STEP = (3, 1, 1)


def _identity(ep_or_row) -> tuple:
    return (
        str(ep_or_row["suite"]),
        int(ep_or_row["task_idx"]),
        int(ep_or_row.get("ep_idx", ep_or_row.get("episode_idx", 0))),
        str(ep_or_row.get("init_state_hash") or ""),
    )


def build_smolvla_schedule_method():
    """Always-on P&P: K=3 at Euler step 1, then K=1 at steps 2 and 3."""
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        n_action_steps=LIBERO_ACTION_STEPS,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_PNP_S123_K311, config


def _completed_rows(store, *, experiment: str, method: str, config: RolloutConfig):
    config_hash = store.config_hash(store._logical_key(method, config))
    return store.fetch_all(
        "rollouts",
        "rollout_id,suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "method", method).eq("config_hash", config_hash).eq("status", "completed"),
        order_by=("rollout_id",),
    )


def _matched_historical_stock(store, episodes) -> dict[tuple, bool]:
    method, config = build_smolvla_libero_methods()[0]
    rows = _completed_rows(
        store, experiment=SMOLVLA_LIBERO_EXPERIMENT, method=method, config=config)
    wanted = {_identity(ep) for ep in episodes}
    matched = {}
    for row in rows:
        key = _identity(row)
        if key in wanted:
            if key in matched:
                raise ValueError(f"duplicate historical SmolVLA stock identity: {key}")
            matched[key] = bool(row["success"])
    missing = sorted(wanted - set(matched))
    if missing:
        raise ValueError(
            f"worker 83 historical stock is missing {len(missing)} identities; "
            f"first missing={missing[0]}")
    return matched


def _existing_outcomes(store, *, experiment, method, config, episodes):
    wanted = {_identity(ep) for ep in episodes}
    outcomes = {}
    for row in _completed_rows(
            store, experiment=experiment, method=method, config=config):
        key = _identity(row)
        if key in wanted:
            outcomes.setdefault(key, {})[method] = bool(row["success"])
    return outcomes


def run_smolvla_schedule_eval_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_SCHEDULE_EXPERIMENT):
    """Evaluate one variable-K P&P arm against exact-matched worker-83 stock rows."""
    from . import models

    episodes = _prepare_libero_episodes()
    method, config = build_smolvla_schedule_method()
    store = SupabaseStore()
    historical = _matched_historical_stock(store, episodes)
    existing = _existing_outcomes(
        store, experiment=experiment, method=method, config=config, episodes=episodes)
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "worker-83 standard LIBERO identities: indices 0-9",
        "identities": len(episodes),
        "new_arm": "always-on P&P steps=(1,2,3), K_by_step=(3,1,1)",
        "reference": f"exact-matched stock rows from {SMOLVLA_LIBERO_EXPERIMENT}",
        "integration_steps": 10,
        "n_action_steps": LIBERO_ACTION_STEPS,
        "generated_chunk_size": 50,
        "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
        "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: new schedule versus exact-matched historical stock every 10 identities.")

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
        cohort="smolvla_schedule_s123_k311",
        shard_count=1,
        shard_index=0,
        benchmark="libero",
        driver="smolvla_schedule_s123_k311_eval",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_idxs": list(range(10)),
            "integration_steps": 10,
            "n_action_steps": LIBERO_ACTION_STEPS,
            "generated_chunk_size": 50,
            "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "historical_reference_experiment": SMOLVLA_LIBERO_EXPERIMENT,
            "video": "off",
        },
        report_every=0,
        report_every_identities=10,
        matched_reference_outcomes={"historic stock VLA": historical},
        initial_identity_outcomes=existing,
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )
