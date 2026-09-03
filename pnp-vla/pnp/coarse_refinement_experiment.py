"""Single-query 5/3-step refinement follow-up on the frozen PRO220 pilot."""
from __future__ import annotations

import pandas as pd

from .config import Method, RolloutConfig
from .five_step_diversity_experiment import (
    FIVE_STEP_DIVERSITY_EPISODE_INDICES, FIVE_STEP_DIVERSITY_EXPERIMENT,
    _identity_key, build_five_step_diversity_methods,
    identity_manifest_hash, identity_manifest_payload)


COARSE_REFINEMENT_EXPERIMENT = "pi05-coarse-single-refinement-pro220-v1"
COARSE_REFINEMENT_EPISODE_INDICES = FIVE_STEP_DIVERSITY_EPISODE_INDICES
COARSE_REFINEMENT_SHARD_COUNT = 2
COARSE_REFINEMENT_REPORT_EVERY = 25


def build_coarse_refinement_methods(source_id: str):
    """Direct refine-last sampling; no best-of-N, gradients, or threshold gates."""
    common = dict(
        policy_source_id=source_id, pnp_k=5, n_action_steps=10,
        save_time_uncertainty=True, save_trajectory=True, save_generated_chunks=True,
        skip_unused_renders=True, render_lead=2)
    return [
        (Method.FIVE_STEP_SINGLE_REFINE, RolloutConfig(
            **common, num_inference_steps=5, pnp_steps=(2, 3), refine=True)),
        (Method.THREE_STEP_SINGLE_REFINE, RolloutConfig(
            **common, num_inference_steps=3, pnp_steps=(2,), refine=True)),
        (Method.THREE_STEP_SINGLE_QUERY, RolloutConfig(
            **common, num_inference_steps=3, pnp_steps=(2,))),
    ]


def coarse_refinement_arm_settings(methods):
    return [{"method": method, "num_inference_steps": config.num_inference_steps,
             "probe_steps_zero_based": list(config.pnp_steps),
             "probe_flow_times": [1 - step / config.num_inference_steps
                                  for step in config.pnp_steps],
             "refine": config.refine, "refine_average": config.refine_average,
             "num_queries": 1, "pnp_k": config.pnp_k, "n_action_steps": 10}
            for method, config in methods]


def _load_exact_reference(store, manifest, *, experiment, method, config,
                          source_repo, source_revision):
    """Require exact identity, sampler config, seed, horizon, and checkpoint matches."""
    from .rollout import episode_seed

    expected_hash = store.config_hash(store._logical_key(method, config))
    rows = store.fetch_all(
        "rollouts",
        "run_id,suite,task_idx,episode_idx,init_state_hash,status,success,method,"
        "config_hash,episode_seed,max_steps,chunk_size",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "method", method).eq("config_hash", expected_hash).eq("status", "completed"),
        order_by=("rollout_id",))
    expected = {_identity_key(episode): episode for episode in manifest}
    selected = {}
    for row in rows:
        key = _identity_key(row)
        if key not in expected:
            continue
        if key in selected:
            raise ValueError(f"{experiment}/{method}: duplicate historical identity {key}")
        if (row["method"] != method or row["config_hash"] != expected_hash
                or row["status"] != "completed"):
            raise ValueError(f"{experiment}/{method}: unexpected historical config/status")
        if row.get("success") not in (True, False, 0, 1):
            raise ValueError(f"{experiment}/{method}: invalid success outcome")
        episode = expected[key]
        seed = episode_seed(episode["init_state"], key[2],
                            episode.get("behavior_seed_index", 0))
        if (row.get("episode_seed") != seed or row.get("max_steps") != episode["max_steps"]
                or row.get("chunk_size") != 50):
            raise ValueError(f"{experiment}/{method}: seed/horizon mismatch for {key}")
        if not str(row.get("run_id") or "").strip():
            raise ValueError(f"{experiment}/{method}: missing run provenance")
        selected[key] = row
    if set(selected) != set(expected):
        raise ValueError(
            f"{experiment}/{method}: missing {len(set(expected) - set(selected))} "
            "exact historical identities; finish the previous pilot before this follow-up")
    runs = store.fetch_all(
        "experiment_runs", "run_id,model_repo_id,model_revision",
        configure=lambda query: query.eq("experiment", experiment), order_by=("run_id",))
    provenance = {str(row["run_id"]): row for row in runs}
    for row in selected.values():
        run = provenance.get(str(row["run_id"]), {})
        if (run.get("model_repo_id") != source_repo
                or run.get("model_revision") != source_revision):
            raise ValueError(f"{experiment}/{method}: historical checkpoint provenance mismatch")
    return {key: bool(row["success"]) for key, row in selected.items()}, expected_hash


def load_coarse_refinement_references(store, manifest, *, source_repo, source_revision):
    from .uncertainty_gradient_experiment import (
        DIRECT_U20_GRADIENT_EXPERIMENT, build_direct_u20_gradient_methods)

    # The historical gradient experiment's first arm is stock/no-op uncertainty ONLY.
    # Never load either its gradient-steered or random-control arm as stock.
    stock = build_direct_u20_gradient_methods()[0]
    five_step = build_five_step_diversity_methods(f"{source_repo}@{source_revision}")
    specs = [("hist 10s x1", DIRECT_U20_GRADIENT_EXPERIMENT, stock),
             ("hist 5s x1", FIVE_STEP_DIVERSITY_EXPERIMENT, five_step[0]),
             ("hist 5s x3 U20", FIVE_STEP_DIVERSITY_EXPERIMENT, five_step[1]),
             ("hist 5s x3 ref", FIVE_STEP_DIVERSITY_EXPERIMENT, five_step[2])]
    references, provenance = {}, {}
    for label, experiment, (method, config) in specs:
        references[label], config_hash = _load_exact_reference(
            store, manifest, experiment=experiment, method=method, config=config,
            source_repo=source_repo, source_revision=source_revision)
        provenance[label] = {"experiment": experiment, "method": method,
                             "config_hash": config_hash}
    return references, provenance


def run_coarse_refinement_worker(
        *, shard_index=0, shard_count=COARSE_REFINEMENT_SHARD_COUNT,
        episode_limit=None, manifest_hash="", source_model_revision="",
        experiment=COARSE_REFINEMENT_EXPERIMENT):
    """Run 110 identities x three single-query arms; resume only completed rollouts."""
    from . import models
    from .diversity import (
        SOURCE_FRACTIONAL_SHORT_SUITES, source_checkpoint_model_source)
    from .experiments import (
        _prepare_libero_pro_expanded_episodes, _run_collection,
        expanded_pro_suites, format_matched_progress_table, identity_shard)
    from .store import SupabaseStore, gather_provenance

    if shard_count != COARSE_REFINEMENT_SHARD_COUNT or shard_index not in (0, 1):
        raise ValueError("frozen PRO220 follow-up requires shard_count=2 and shard_index=0 or 1")
    if not manifest_hash or not source_model_revision:
        raise ValueError("load manifest_hash and source_model_revision from the shared v2 manifest")
    if episode_limit is not None:
        if (isinstance(episode_limit, bool) or int(episode_limit) != episode_limit
                or episode_limit < 1):
            raise ValueError("episode_limit must be a positive integer or None")
        episode_limit = int(episode_limit)

    store = SupabaseStore()
    source_repo, source_revision = source_checkpoint_model_source(
        store, expected_revision=source_model_revision)
    suites = [suite for suite in expanded_pro_suites()
              if suite not in SOURCE_FRACTIONAL_SHORT_SUITES]
    manifest = _prepare_libero_pro_expanded_episodes(
        suites=suites, episode_idxs=COARSE_REFINEMENT_EPISODE_INDICES)
    if len(suites) != 11 or len(manifest) != 220 or len(identity_manifest_payload(manifest)) != 220:
        raise ValueError("expected the frozen 11-suite, 220-identity PRO cohort")
    references, historical_provenance = load_coarse_refinement_references(
        store, manifest, source_repo=source_repo, source_revision=source_revision)
    full_shard = identity_shard(manifest, shard_count, shard_index)
    if len(full_shard) != 110:
        raise ValueError(f"expected 110 identities in shard {shard_index}")
    episodes = full_shard if episode_limit is None else full_shard[:episode_limit]
    shard_keys = {_identity_key(episode) for episode in episodes}
    references = {label: {key: value for key, value in outcomes.items() if key in shard_keys}
                  for label, outcomes in references.items()}
    methods = build_coarse_refinement_methods(f"{source_repo}@{source_revision}")
    expected_hashes = {method: store.config_hash(store._logical_key(method, config))
                       for method, config in methods}
    existing = store.fetch_all(
        "rollouts", "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq("experiment", experiment), order_by=("rollout_id",))
    initial_tally, initial_outcomes = {}, {}
    for row in existing:
        key, method = _identity_key(row), row["method"]
        if (key not in shard_keys or method not in expected_hashes
                or row["config_hash"] != expected_hashes[method] or row["status"] != "completed"):
            continue
        if row.get("success") not in (True, False, 0, 1):
            raise ValueError("completed rollout has an invalid success outcome")
        outcomes = initial_outcomes.setdefault(key, {})
        if method in outcomes:
            raise ValueError("duplicate completed coarse-refinement identity/method")
        outcomes[method] = bool(row["success"])
        counts = initial_tally.setdefault((key[0], method), [0, 0])
        counts[0] += 1
        counts[1] += int(row["success"])

    arm_settings = coarse_refinement_arm_settings(methods)
    metadata = {
        "source_model_repo_id": source_repo, "source_model_revision": source_revision,
        "bootstrap_manifest_hash": manifest_hash,
        "frozen_identity_manifest_hash": identity_manifest_hash(manifest),
        "frozen_identity_manifest": identity_manifest_payload(manifest),
        "historical_references": historical_provenance,
        "episode_indices": list(COARSE_REFINEMENT_EPISODE_INDICES), "suites": suites,
        "target_identities": 220, "generated_chunk_size": 50, "n_action_steps": 10,
        "num_queries": 1, "pnp_k": 5, "uncertainty_horizons": [10, 20, 50],
        "arms": arm_settings, "requested_methods": [method for method, _ in methods],
        "config_hashes": expected_hashes, "absolute_threshold": None,
        "perturb_seed_scheme": "ordinary_episode_stream_v1",
    }
    print({"experiment": experiment, "source": f"{source_repo}@{source_revision}",
           "identities_in_full_shard": 110, "new_rollouts_in_full_shard": 330,
           "identities_requested": len(episodes), "new_rollouts_requested": len(episodes) * 3,
           "report_every_completed_identities": COARSE_REFINEMENT_REPORT_EVERY,
           "frozen_identity_manifest_hash": metadata["frozen_identity_manifest_hash"],
           "new_config_hashes": expected_hashes, "historical_references": historical_provenance})
    print(pd.DataFrame(arm_settings).to_string(index=False))
    print("Progress bar counts ROLLOUTS. Comparison tables print at 25/50/75/100 completed "
          "three-arm IDENTITIES and at the end (normally every 75 new rollouts).")
    print("Historical stock has no refinement, selection, or gradient steering. "
          "The 3-step no-refinement arm uses measurement-only PnP; executed actions stay stock.")
    print("Historical columns: 10s x1 = stock; 5s x1 = coarse stock; "
          "5s x3 U20 = lowest-U20 selection; 5s x3 ref = select then refine.")
    if initial_outcomes:
        print("Resumed completed results:")
        print(format_matched_progress_table(
            initial_outcomes, [method for method, _ in methods], references))

    policy, preprocess, postprocess = models.load_pi05(repo_id=source_repo, revision=source_revision)
    if int(policy.config.chunk_size) != 50 or int(policy.config.num_inference_steps) != 10:
        raise ValueError("expected the historical stock model's 50-action / 10-step defaults")
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=methods, cohort="coarse_single_refinement_pro220",
        shard_count=shard_count, shard_index=shard_index, benchmark="libero_pro",
        driver="pi05_coarse_single_refinement_pro220", run_metadata=metadata,
        provenance=gather_provenance(model_repo_id=source_repo, model_revision=source_revision),
        report_every=0, report_every_identities=COARSE_REFINEMENT_REPORT_EVERY,
        initial_tally=initial_tally, initial_identity_outcomes=initial_outcomes,
        matched_reference_outcomes=references, progress_include_overall=True,
        rollout_batch_size=1, resume_completed_only=True)
