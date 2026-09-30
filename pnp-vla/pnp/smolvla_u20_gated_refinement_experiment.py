"""Online per-chunk U20-Q75 gating for plain SmolVLA P&P refinement."""
from __future__ import annotations

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    LIBERO_SUITES,
    _prepare_libero_episodes,
    _run_collection,
    format_matched_progress_table,
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
from .smolvla_u20_gated_blend_experiment import SMOLVLA_U20_Q75_BY_SUITE
from .store import SupabaseStore, gather_provenance


SMOLVLA_U20_GATED_REFINEMENT_EXPERIMENT = (
    "smolvla-libero-a10-online-u20-q75-gated-plain-refine-k311-v1")


def build_smolvla_u20_gated_refinement_method(suite: str):
    """Keep notebook 108's gate fixed; replace only its blend with plain refinement."""
    if suite not in SMOLVLA_U20_Q75_BY_SUITE:
        raise ValueError(f"no frozen U20-Q75 threshold for suite {suite!r}")
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        refine_chunk_gate_threshold=float(SMOLVLA_U20_Q75_BY_SUITE[suite]),
        refine_chunk_gate_horizon=20,
        num_inference_steps=10,
        n_action_steps=LIBERO_ACTION_STEPS,
        save_time_uncertainty=True,
        save_trajectory=False,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_U20_GATED_REFINEMENT_K311, config


def _matched_historical_refinement(store, episodes):
    method, config = build_smolvla_schedule_method()
    wanted = {_identity(ep) for ep in episodes}
    matched = {}
    for row in _completed_rows(
            store, experiment=SMOLVLA_SCHEDULE_EXPERIMENT,
            method=method, config=config):
        key = _identity(row)
        if key in wanted:
            if key in matched:
                raise ValueError(f"duplicate historical refinement identity: {key}")
            matched[key] = bool(row["success"])
    if set(matched) != wanted:
        print(
            f"Historical notebook-85 plain-refinement reference is incomplete "
            f"({len(matched)}/{len(wanted)}); periodic tables will omit it.",
            flush=True,
        )
        return None
    return matched


def _existing_outcomes(store, episodes):
    wanted = {_identity(ep) for ep in episodes}
    outcomes = {}
    for suite in LIBERO_SUITES:
        method, config = build_smolvla_u20_gated_refinement_method(suite)
        for row in _completed_rows(
                store, experiment=SMOLVLA_U20_GATED_REFINEMENT_EXPERIMENT,
                method=method, config=config):
            key = _identity(row)
            if key in wanted:
                outcomes.setdefault(key, {})[method] = bool(row["success"])
    return outcomes


def _print_gate_summary(store, episodes, historical):
    method = Method.SMOLVLA_U20_GATED_REFINEMENT_K311
    outcomes = _existing_outcomes(store, episodes)
    print("\nFinal exact-matched online-gate table")
    print(format_matched_progress_table(outcomes, [method], historical))
    print("\nGate utilization by suite")
    print(f"{'suite':<20}{'episodes':>10}{'boundary fire rate':>21}{'refined chunks':>18}")
    for suite in LIBERO_SUITES:
        name, config = build_smolvla_u20_gated_refinement_method(suite)
        config_hash = store.config_hash(store._logical_key(name, config))
        rows = store.fetch_all(
            "rollouts", "ms_candidate_u,status",
            configure=lambda query, s=suite, h=config_hash: query.eq(
                "experiment", SMOLVLA_U20_GATED_REFINEMENT_EXPERIMENT).eq(
                "suite", s).eq("method", name).eq("config_hash", h).eq(
                "status", "completed"),
        )
        gates = [
            (row.get("ms_candidate_u") or {}).get("refinement_chunk_gate") or {}
            for row in rows]
        fires = sum(int(gate.get("n_fired") or 0) for gate in gates)
        considered = sum(int(gate.get("n_considered") or 0) for gate in gates)
        fire_rate = fires / considered if considered else float("nan")
        print(f"{suite:<20}{len(rows):>10}{fire_rate:>21.1%}{fires:>18}")


def run_smolvla_u20_gated_refinement_eval_worker(
        *, rollout_batch_size: int = 1,
        experiment: str = SMOLVLA_U20_GATED_REFINEMENT_EXPERIMENT):
    """Select exact stock or completed plain refinement at every decision boundary."""
    from . import models

    if experiment != SMOLVLA_U20_GATED_REFINEMENT_EXPERIMENT:
        raise ValueError("the frozen gated-refinement worker uses its own namespace")
    if rollout_batch_size != 1:
        raise ValueError("chunk-level conditional refinement is frozen to batch size 1")

    episodes = _prepare_libero_episodes()
    store = SupabaseStore()
    historical = {"historic stock A10": _matched_historical_stock(store, episodes)}
    historical_refinement = _matched_historical_refinement(store, episodes)
    if historical_refinement is not None:
        historical["historic always plain refine"] = historical_refinement

    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "standard LIBERO indices 0-9",
        "identities": len(episodes),
        "decision": (
            "at every chunk: pair-weighted stock U20 >= suite Q75 -> completed "
            "plain P&P refinement; else exact stock"),
        "thresholds": SMOLVLA_U20_Q75_BY_SUITE,
        "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
        "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
        "integration_steps": 10,
        "n_action_steps": LIBERO_ACTION_STEPS,
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print(
        "Periodic output: gated plain refinement plus exact-matched historical stock "
        "and always-refine references every 10 identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    existing = _existing_outcomes(store, episodes)

    for suite in LIBERO_SUITES:
        suite_episodes = [ep for ep in episodes if ep["suite"] == suite]
        method, config = build_smolvla_u20_gated_refinement_method(suite)
        suite_existing = {
            key: value for key, value in existing.items() if key[0] == suite}
        suite_historical = {
            label: {key: value for key, value in reference.items() if key[0] == suite}
            for label, reference in historical.items()
        }
        print(
            f"\n=== {suite}: threshold={SMOLVLA_U20_Q75_BY_SUITE[suite]:.8f} ===",
            flush=True,
        )
        _run_collection(
            store=store,
            policy=policy,
            preprocess=preprocess,
            postprocess=postprocess,
            device=models.default_device(),
            experiment=experiment,
            episodes=suite_episodes,
            methods=[(method, config)],
            cohort=f"smolvla_online_u20_q75_plain_refine_{suite}",
            shard_count=1,
            shard_index=0,
            benchmark="libero",
            driver="smolvla_online_u20_q75_plain_refine_eval",
            run_metadata={
                "model_repo_id": SMOLVLA_REPO_ID,
                "target_identities": len(episodes),
                "episode_indices": list(range(10)),
                "suite": suite,
                "pair_weighted_u20_threshold": SMOLVLA_U20_Q75_BY_SUITE[suite],
                "gate_horizon": 20,
                "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
                "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
                "intervention": "plain_refinement",
                "integration_steps": 10,
                "n_action_steps": LIBERO_ACTION_STEPS,
                "online_per_chunk": True,
                "historical_stock_experiment": "smolvla-libero-a10-pnp-k5-steps34-v1",
                "historical_refinement_experiment": SMOLVLA_SCHEDULE_EXPERIMENT,
                "video": "off",
            },
            report_every=0,
            report_every_identities=10,
            matched_reference_outcomes=suite_historical,
            initial_identity_outcomes=suite_existing,
            rollout_batch_size=rollout_batch_size,
            provenance=provenance,
            resume_completed_only=True,
        )
    _print_gate_summary(store, episodes, historical)

