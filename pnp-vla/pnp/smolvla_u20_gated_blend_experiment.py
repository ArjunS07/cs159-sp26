"""Online per-chunk U20-Q75 gating for SmolVLA s=0.3 consensus projection."""
from __future__ import annotations

from dataclasses import replace

from .config import Method, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_SUITES,
    _prepare_libero_episodes,
    _run_collection,
    format_matched_progress_table,
)
from .smolvla_blend_ablation_experiment import (
    SMOLVLA_CONSENSUS_S03_EXPERIMENT,
    build_smolvla_consensus_s03_method,
)
from .smolvla_followup_experiments import (
    _completed_rows,
    _identity,
    _matched_historical_stock,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_U20_GATED_S03_EXPERIMENT = (
    "smolvla-libero-a10-online-u20-q75-gated-consensus-s03-k3-v1")
SMOLVLA_U20_Q75_BY_SUITE = {
    "libero_10": 0.08123845383524894,
    "libero_goal": 0.09962177574634552,
    "libero_object": 0.07061833441257477,
    "libero_spatial": 0.08815010413527488,
}


def build_smolvla_u20_gated_s03_method(suite: str):
    if suite not in SMOLVLA_U20_Q75_BY_SUITE:
        raise ValueError(f"no frozen U20-Q75 threshold for suite {suite!r}")
    _, base = build_smolvla_consensus_s03_method()
    config = replace(
        base,
        consensus_projection_gate_threshold=float(SMOLVLA_U20_Q75_BY_SUITE[suite]),
        consensus_projection_gate_horizon=20,
        save_trajectory=False,
    )
    return Method.SMOLVLA_U20_GATED_CONSENSUS_S03_K3, config


def _historical_s03_if_complete(store, episodes):
    method, config = build_smolvla_consensus_s03_method()
    wanted = {_identity(ep) for ep in episodes}
    matched = {}
    for row in _completed_rows(
            store, experiment=SMOLVLA_CONSENSUS_S03_EXPERIMENT,
            method=method, config=config):
        key = _identity(row)
        if key in wanted:
            if key in matched:
                raise ValueError(f"duplicate historical s=0.3 identity: {key}")
            matched[key] = bool(row["success"])
    if set(matched) != wanted:
        print(
            f"Historical notebook-106 s=0.3 reference is incomplete "
            f"({len(matched)}/{len(wanted)}); periodic tables will omit it.",
            flush=True,
        )
        return None
    return matched


def _existing_gated_outcomes(store, episodes):
    wanted = {_identity(ep) for ep in episodes}
    outcomes = {}
    for suite in LIBERO_SUITES:
        method, config = build_smolvla_u20_gated_s03_method(suite)
        for row in _completed_rows(
                store, experiment=SMOLVLA_U20_GATED_S03_EXPERIMENT,
                method=method, config=config):
            key = _identity(row)
            if key in wanted:
                outcomes.setdefault(key, {})[method] = bool(row["success"])
    return outcomes


def _print_final_gate_summary(store, episodes, historical):
    method = Method.SMOLVLA_U20_GATED_CONSENSUS_S03_K3
    outcomes = _existing_gated_outcomes(store, episodes)
    print("\nFinal exact-matched online-gate table")
    print(format_matched_progress_table(outcomes, [method], historical))
    print("\nGate utilization by suite")
    print(f"{'suite':<20}{'episodes':>10}{'boundary fire rate':>21}{'projected chunks':>19}")
    for suite in LIBERO_SUITES:
        name, config = build_smolvla_u20_gated_s03_method(suite)
        config_hash = store.config_hash(store._logical_key(name, config))
        rows = store.fetch_all(
            "rollouts", "gate_fire_rate,n_corrections_applied,ms_candidate_u,status",
            configure=lambda query, s=suite, h=config_hash: query.eq(
                "experiment", SMOLVLA_U20_GATED_S03_EXPERIMENT).eq(
                "suite", s).eq("method", name).eq("config_hash", h).eq(
                "status", "completed"),
        )
        gates = [
            (row.get("ms_candidate_u") or {}).get("consensus_projection_gate") or {}
            for row in rows]
        fires = sum(int(gate.get("n_fired") or 0) for gate in gates)
        considered = sum(int(gate.get("n_considered") or 0) for gate in gates)
        fire_rate = fires / considered if considered else float("nan")
        print(f"{suite:<20}{len(rows):>10}{fire_rate:>21.1%}{fires:>19}")


def run_smolvla_u20_gated_s03_eval_worker(
        *, rollout_batch_size: int = 1,
        experiment: str = SMOLVLA_U20_GATED_S03_EXPERIMENT):
    """Run a true online gate, selecting stock or s=0.3 blend at every chunk."""
    from . import models

    if experiment != SMOLVLA_U20_GATED_S03_EXPERIMENT:
        raise ValueError("the frozen Q75 worker does not accept another experiment namespace")
    if rollout_batch_size != 1:
        raise ValueError(
            "online gating is frozen to batch size 1 so rejected chunks consume no "
            "projection RNG and exactly preserve the conditional rollout")
    episodes = _prepare_libero_episodes()
    store = SupabaseStore()
    historical_stock = _matched_historical_stock(store, episodes)
    historical_s03 = _historical_s03_if_complete(store, episodes)
    historical = {"historic stock A10": historical_stock}
    if historical_s03 is not None:
        historical["historic always s=0.3 blend"] = historical_s03

    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "cohort": "standard LIBERO indices 0-9",
        "identities": len(episodes),
        "decision": "at every chunk: pair-weighted stock U20 >= suite Q75 -> s=0.3 blend; else exact stock",
        "thresholds": SMOLVLA_U20_Q75_BY_SUITE,
        "pnp_steps": [1, 2, 3],
        "pnp_k_by_step": [3, 1, 1],
        "projection": {"s": 0.3, "step": 7, "k": 3},
        "integration_steps": 10,
        "n_action_steps": 10,
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print(
        "Periodic output: one gated arm plus exact-matched historical references every "
        "10 completed identities. Suites run sequentially; the final table pools all four.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    existing = _existing_gated_outcomes(store, episodes)
    for suite in LIBERO_SUITES:
        suite_episodes = [ep for ep in episodes if ep["suite"] == suite]
        method, config = build_smolvla_u20_gated_s03_method(suite)
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
            cohort=f"smolvla_online_u20_q75_s03_{suite}",
            shard_count=1,
            shard_index=0,
            benchmark="libero",
            driver="smolvla_online_u20_q75_s03_eval",
            run_metadata={
                "model_repo_id": SMOLVLA_REPO_ID,
                "target_identities": len(episodes),
                "episode_indices": list(range(10)),
                "suite": suite,
                "pair_weighted_u20_threshold": SMOLVLA_U20_Q75_BY_SUITE[suite],
                "gate_horizon": 20,
                "pnp_steps": [1, 2, 3],
                "pnp_k_by_step": [3, 1, 1],
                "projection_s": 0.3,
                "projection_step": 7,
                "projection_k": 3,
                "integration_steps": 10,
                "n_action_steps": 10,
                "online_per_chunk": True,
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
    _print_final_gate_summary(store, episodes, historical)
