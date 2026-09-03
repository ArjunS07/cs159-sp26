"""Five-integration-step diversity pilot on the frozen 220-identity PRO cohort."""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd

from .config import Method, RolloutConfig


FIVE_STEP_DIVERSITY_EXPERIMENT = "pi05-five-step-diversity-pro220-v1"
FIVE_STEP_DIVERSITY_EPISODE_INDICES = (10, 11)
FIVE_STEP_DIVERSITY_PROBE_STEPS = (2, 3)
FIVE_STEP_DIVERSITY_PROBE_TIMES = (0.6, 0.4)
FIVE_STEP_DIVERSITY_NUM_INFERENCE_STEPS = 5
FIVE_STEP_DIVERSITY_NUM_QUERIES = 3
FIVE_STEP_DIVERSITY_PNP_K = 5
FIVE_STEP_DIVERSITY_SELECTION_HORIZON = 20
FIVE_STEP_DIVERSITY_ACTION_STEPS = 10
FIVE_STEP_DIVERSITY_SHARD_COUNT = 2
FIVE_STEP_DIVERSITY_IDENTITIES = 220


def _identity_key(row) -> tuple[str, int, int, str]:
    return (str(row["suite"]), int(row["task_idx"]),
            int(row.get("ep_idx", row.get("episode_idx", 0))),
            str(row.get("init_state_hash", "")))


def identity_manifest_payload(episodes) -> list[dict]:
    """Canonical, exact identity list stored in every experiment-run record."""
    return [
        {"suite": suite, "task_idx": task_idx, "episode_idx": episode_idx,
         "init_state_hash": init_state_hash}
        for suite, task_idx, episode_idx, init_state_hash in sorted(
            {_identity_key(episode) for episode in episodes})
    ]


def identity_manifest_hash(episodes) -> str:
    payload = json.dumps(
        identity_manifest_payload(episodes), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_five_step_diversity_methods(source_id: str):
    """Return the coarse x1, coarse x3, and coarse x3-then-refine arms."""
    common = dict(
        pnp_k=FIVE_STEP_DIVERSITY_PNP_K,
        policy_source_id=source_id,
        num_inference_steps=FIVE_STEP_DIVERSITY_NUM_INFERENCE_STEPS,
        n_action_steps=FIVE_STEP_DIVERSITY_ACTION_STEPS,
        save_trajectory=True, save_generated_chunks=True,
        skip_unused_renders=True, render_lead=2)
    candidate_common = dict(
        **common, num_samples=FIVE_STEP_DIVERSITY_NUM_QUERIES,
        ms_probe_steps=FIVE_STEP_DIVERSITY_PROBE_STEPS,
        selection_uncertainty_horizon=FIVE_STEP_DIVERSITY_SELECTION_HORIZON,
        candidate_set_id="|".join([source_id] * FIVE_STEP_DIVERSITY_NUM_QUERIES),
        candidate_seed_scheme="stock_slot0_v1")
    methods = [
        (Method.FIVE_STEP_SINGLE_QUERY, RolloutConfig(
            **common, pnp_steps=FIVE_STEP_DIVERSITY_PROBE_STEPS,
            save_time_uncertainty=True)),
        (Method.FIVE_STEP_LOWEST_U20, RolloutConfig(**candidate_common)),
        (Method.FIVE_STEP_LOWEST_U20_REFINE, RolloutConfig(
            **candidate_common, multi_sample_refine_selected=True)),
    ]
    if len({method for method, _ in methods}) != 3:
        raise AssertionError("five-step diversity arms require unique method labels")
    return methods


def _load_verified_historical_baseline(store, manifest, *, source_repo, source_revision):
    """Load the exact 10-step x1 rows and prove identity/config/model provenance."""
    from .diversity import DIVERSITY_PAIR_KEYS
    from .uncertainty_gradient_experiment import (
        DIRECT_U20_GRADIENT_EXPERIMENT, build_direct_u20_gradient_methods)

    historical_method, historical_config = build_direct_u20_gradient_methods()[0]
    historical_hash = store.config_hash(
        store._logical_key(historical_method, historical_config))
    rows = pd.DataFrame(store.fetch_all(
        "rollouts",
        "run_id,suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq(
            "experiment", DIRECT_U20_GRADIENT_EXPERIMENT).eq(
            "method", historical_method).eq("config_hash", historical_hash),
        order_by=("suite", "task_idx", "episode_idx", "init_state_hash")))
    if rows.empty:
        raise ValueError(
            f"no exact 10-step baseline rows found in {DIRECT_U20_GRADIENT_EXPERIMENT}")
    manifest_keys = {_identity_key(episode) for episode in manifest}
    rows = rows[
        rows.status.eq("completed")
        & rows[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(manifest_keys)].copy()
    if rows.duplicated(DIVERSITY_PAIR_KEYS).any():
        raise ValueError("duplicate exact 10-step baseline identities")
    row_keys = {_identity_key(row) for row in rows.to_dict("records")}
    if row_keys != manifest_keys:
        missing = sorted(manifest_keys - row_keys)
        extra = sorted(row_keys - manifest_keys)
        raise ValueError(
            f"10-step baseline does not exactly match the frozen cohort: "
            f"missing={len(missing)}, extra={len(extra)}")

    if rows.run_id.isna().any() or rows.run_id.astype(str).str.strip().eq("").any():
        raise ValueError("historical 10-step baseline has rows without run_id provenance")
    run_ids = {str(value) for value in rows.run_id}
    runs = store.fetch_all(
        "experiment_runs", "run_id,model_repo_id,model_revision",
        configure=lambda query: query.eq("experiment", DIRECT_U20_GRADIENT_EXPERIMENT),
        order_by=("run_id",))
    provenance = {str(row["run_id"]): row for row in runs}
    mismatched = [
        run_id for run_id in run_ids
        if run_id not in provenance
        or provenance[run_id].get("model_repo_id") != source_repo
        or provenance[run_id].get("model_revision") != source_revision
    ]
    if not run_ids or mismatched:
        raise ValueError(
            "historical 10-step baseline lacks exact source-checkpoint provenance: "
            f"{sorted(mismatched)}")
    return rows, historical_hash


def _load_npz_artifact(store, path: str) -> dict[str, np.ndarray]:
    if not path:
        raise ValueError("missing generated-chunks artifact")
    with np.load(io.BytesIO(store._download(path)), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_five_step_diversity_sentinel(
        *, shard_index: int, source_id: str,
        experiment: str = FIVE_STEP_DIVERSITY_EXPERIMENT, store=None) -> dict:
    """Validate one completed three-arm identity from this worker's shard.

    This checks persisted artifacts, not just in-memory configuration: all candidates are
    present, candidate slot 0 retains the ordinary policy seed at every overlapping boundary,
    the selected seed is the one recorded for execution, and refinement reuses that seed.
    """
    from .diversity import SOURCE_FRACTIONAL_SHORT_SUITES
    from .experiments import (
        _prepare_libero_pro_expanded_episodes, expanded_pro_suites, identity_shard)
    from .store import SupabaseStore

    if shard_index not in (0, 1):
        raise ValueError("sentinel shard_index must be 0 or 1")
    store = store or SupabaseStore()
    suites = [suite for suite in expanded_pro_suites()
              if suite not in SOURCE_FRACTIONAL_SHORT_SUITES]
    manifest = _prepare_libero_pro_expanded_episodes(
        suites=suites, episode_idxs=FIVE_STEP_DIVERSITY_EPISODE_INDICES)
    shard_keys = {
        _identity_key(episode)
        for episode in identity_shard(manifest, FIVE_STEP_DIVERSITY_SHARD_COUNT, shard_index)}
    methods = build_five_step_diversity_methods(source_id)
    method_names = [method for method, _ in methods]
    expected_hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    rows = pd.DataFrame(store.fetch_all(
        "rollouts",
        "rollout_id,suite,task_idx,episode_idx,init_state_hash,method,status,"
        "config_hash,config_json,ms_candidate_u,generated_chunks_path,"
        "video_path,obs_frames_path",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "status", "completed"),
        order_by=("suite", "task_idx", "episode_idx", "init_state_hash", "method")))
    if rows.empty:
        raise ValueError(f"no completed sentinel rows found for {experiment}")
    rows["_identity"] = rows.apply(lambda row: _identity_key(row), axis=1)
    rows = rows[
        rows["_identity"].isin(shard_keys)
        & rows.apply(
            lambda row: row["method"] in expected_hashes
            and row["config_hash"] == expected_hashes[row["method"]], axis=1)].copy()
    complete_identity = None
    for identity, group in rows.groupby("_identity", sort=True):
        if set(group.method) == set(method_names) and len(group) == len(method_names):
            complete_identity = identity
            rows = group.copy()
            break
    if complete_identity is None:
        raise ValueError(
            f"no identity in shard {shard_index} has all three exact completed arms")

    by_method = {row["method"]: row for row in rows.to_dict("records")}
    artifacts = {}
    for method in method_names:
        row = by_method[method]
        config_json = row["config_json"]
        if isinstance(config_json, str):
            config_json = json.loads(config_json)
        if config_json.get("policy_source_id") != source_id:
            raise AssertionError(f"{method}: policy source is not revision-bound")
        if config_json.get("num_inference_steps") != 5:
            raise AssertionError(f"{method}: expected five inference steps")
        if config_json.get("n_action_steps") != 10:
            raise AssertionError(f"{method}: expected ten executed actions per chunk")
        if row.get("video_path") or row.get("obs_frames_path"):
            raise AssertionError(f"{method}: sentinel unexpectedly saved video/frames")
        artifacts[method] = _load_npz_artifact(store, row["generated_chunks_path"])

    single = artifacts[Method.FIVE_STEP_SINGLE_QUERY]
    if single["chunks"].ndim != 3 or single["chunks"].shape[1] != 50:
        raise AssertionError("single-query artifact does not contain 50-action chunks")
    if len(single["chunk_noise_seeds"]) != len(single["chunks"]):
        raise AssertionError("single-query chunk/seed counts disagree")

    checked_boundaries = {}
    for method in (Method.FIVE_STEP_LOWEST_U20,
                   Method.FIVE_STEP_LOWEST_U20_REFINE):
        row = by_method[method]
        artifact = artifacts[method]
        candidates = artifact.get("candidate_chunks")
        if candidates is None or candidates.ndim != 4 or candidates.shape[1:3] != (3, 50):
            raise AssertionError(f"{method}: expected every 3 x 50 candidate set")
        if len(artifact["chunks"]) != len(candidates):
            raise AssertionError(f"{method}: selected/candidate boundary counts disagree")
        telemetry = row["ms_candidate_u"]
        if isinstance(telemetry, str):
            telemetry = json.loads(telemetry)
        required = (
            "chosen", "u", "candidate_profiles", "candidate_noise_seeds",
            "selected_noise_seed", "selected_perturb_seed",
            "executed_prefix_disagreement", "inference_ms", "n_vf_evals")
        missing = [name for name in required if name not in telemetry]
        if missing:
            raise AssertionError(f"{method}: missing persisted telemetry {missing}")
        boundary_count = len(candidates)
        if any(len(telemetry[name]) != boundary_count for name in required):
            raise AssertionError(f"{method}: telemetry boundary counts disagree")
        overlap = min(boundary_count, len(single["chunk_noise_seeds"]))
        for boundary in range(boundary_count):
            candidate_seeds = telemetry["candidate_noise_seeds"][boundary]
            chosen = int(telemetry["chosen"][boundary])
            selected_seed = int(telemetry["selected_noise_seed"][boundary])
            if len(candidate_seeds) != 3:
                raise AssertionError(f"{method}: boundary {boundary} lacks three seeds")
            if selected_seed != int(candidate_seeds[chosen]):
                raise AssertionError(f"{method}: selected seed does not match chosen candidate")
            if int(telemetry["selected_perturb_seed"][boundary]) != selected_seed:
                raise AssertionError(f"{method}: measurement perturb seed differs from noise seed")
            if int(artifact["chunk_noise_seeds"][boundary]) != selected_seed:
                raise AssertionError(f"{method}: artifact records the wrong executed noise seed")
            profiles = telemetry["candidate_profiles"][boundary]
            if len(profiles) != 3 or any(
                    not {"u10", "u20", "u_full"}.issubset(profile) for profile in profiles):
                raise AssertionError(f"{method}: U10/U20/U50 candidate profiles are incomplete")
            if method == Method.FIVE_STEP_LOWEST_U20:
                np.testing.assert_array_equal(
                    artifact["chunks"][boundary], candidates[boundary, chosen])
            else:
                refinement = telemetry.get("selected_refinement", [])[boundary]
                if (int(refinement["initial_noise_seed"]) != selected_seed
                        or int(refinement["perturb_seed"]) != selected_seed):
                    raise AssertionError(
                        f"{method}: refinement did not reuse the selected candidate seed")
        for boundary in range(overlap):
            if (int(telemetry["candidate_noise_seeds"][boundary][0])
                    != int(single["chunk_noise_seeds"][boundary])):
                raise AssertionError(
                    f"{method}: candidate 0 does not match the stock seed at boundary {boundary}")
        checked_boundaries[method] = boundary_count

    summary = {
        "status": "passed", "experiment": experiment, "shard_index": shard_index,
        "identity": {
            "suite": complete_identity[0], "task_idx": complete_identity[1],
            "episode_idx": complete_identity[2],
            "init_state_hash": complete_identity[3]},
        "candidate_seed_scheme": "stock_slot0_v1",
        "single_query_boundaries": int(len(single["chunks"])),
        "candidate_boundaries_checked": checked_boundaries,
        "all_candidates_persisted": True, "video_and_frames_absent": True,
    }
    print("five-step diversity sentinel:", summary)
    return summary


def run_five_step_diversity_worker(
        *, shard_count: int = FIVE_STEP_DIVERSITY_SHARD_COUNT, shard_index: int = 0,
        episode_indices=FIVE_STEP_DIVERSITY_EPISODE_INDICES,
        episode_limit: int | None = None, manifest_hash: str = "",
        source_model_revision: str = "",
        experiment: str = FIVE_STEP_DIVERSITY_EXPERIMENT):
    """Run all three arms on one of two exact, resume-safe 110-identity shards."""
    from . import models
    from .diversity import (
        DIVERSITY_PAIR_KEYS, SOURCE_FRACTIONAL_SHORT_SUITES,
        source_checkpoint_model_source)
    from .experiments import (
        _prepare_libero_pro_expanded_episodes, _run_collection,
        expanded_pro_suites, identity_shard)
    from .store import SupabaseStore, gather_provenance

    episode_indices = tuple(map(int, episode_indices))
    if episode_indices != FIVE_STEP_DIVERSITY_EPISODE_INDICES:
        raise ValueError(
            f"frozen five-step cohort requires episode indices "
            f"{FIVE_STEP_DIVERSITY_EPISODE_INDICES}")
    if shard_count != FIVE_STEP_DIVERSITY_SHARD_COUNT:
        raise ValueError(
            f"frozen five-step pilot requires {FIVE_STEP_DIVERSITY_SHARD_COUNT} shards")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    if not manifest_hash:
        raise ValueError("manifest_hash is required from the shared v2 manifest")
    if not source_model_revision:
        raise ValueError("source_model_revision is required from the shared v2 manifest")
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
    if len(suites) != 11:
        raise ValueError(f"expected 11 established PRO pilot suites, found {len(suites)}")
    manifest = _prepare_libero_pro_expanded_episodes(
        suites=suites, episode_idxs=episode_indices)
    if len(manifest) != FIVE_STEP_DIVERSITY_IDENTITIES:
        raise ValueError(
            f"expected {FIVE_STEP_DIVERSITY_IDENTITIES} frozen identities, found {len(manifest)}")
    manifest_payload = identity_manifest_payload(manifest)
    frozen_hash = identity_manifest_hash(manifest)
    historical_rows, historical_config_hash = _load_verified_historical_baseline(
        store, manifest, source_repo=source_repo, source_revision=source_revision)
    full_shard = identity_shard(manifest, shard_count, shard_index)
    if len(full_shard) != FIVE_STEP_DIVERSITY_IDENTITIES // shard_count:
        raise ValueError(
            f"expected 110 identities in shard {shard_index}, found {len(full_shard)}")
    episodes = full_shard if episode_limit is None else full_shard[:episode_limit]
    shard_keys = {_identity_key(episode) for episode in episodes}
    matched_historical = historical_rows[
        historical_rows[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(shard_keys)].copy()
    if len(matched_historical) != len(episodes):
        raise ValueError(
            f"expected {len(episodes)} shard-matched historical rows, "
            f"found {len(matched_historical)}")
    historical_sr = (
        matched_historical.groupby("suite").success.mean().astype(float).to_dict())
    source_id = f"{source_repo}@{source_revision}"
    methods = build_five_step_diversity_methods(source_id)
    if any(config.n_action_steps != 10 or config.num_inference_steps != 5
           for _, config in methods):
        raise AssertionError("five-step pilot must decode with 5 steps and execute 10 actions")

    expected_hashes = {
        method: store.config_hash(store._logical_key(method, config))
        for method, config in methods}
    existing = pd.DataFrame(store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq("experiment", experiment),
        order_by=("rollout_id",)))
    if not existing.empty:
        expected = existing.method.map(expected_hashes)
        existing = existing[
            existing.config_hash.eq(expected)
            & existing.status.eq("completed")
            & existing[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(shard_keys)].copy()
        if existing.duplicated(DIVERSITY_PAIR_KEYS + ["method"]).any():
            raise ValueError("duplicate five-step rollout identity/method rows")
    initial_tally = {} if existing.empty else {
        (suite, method): [len(group), int(group.success.astype(bool).sum())]
        for (suite, method), group in existing.groupby(["suite", "method"], sort=True)}
    initial_identity_methods = [] if existing.empty else [
        (*_identity_key(row), row["method"]) for row in existing.to_dict("records")]

    print({
        "experiment": experiment, "source": source_id,
        "frozen_identity_manifest_hash": frozen_hash,
        "historical_10_step_config_hash": historical_config_hash,
        "new_config_hashes": expected_hashes,
        "target_identities": len(manifest), "identities_in_shard": len(episodes),
        "new_rollouts_in_full_shard": len(full_shard) * len(methods),
        "new_rollouts_requested": len(episodes) * len(methods),
        "arms": [method for method, _ in methods],
        "num_inference_steps": 5, "generated_chunk_size": 50,
        "n_action_steps": 10, "num_queries": 3, "pnp_k": 5,
        "probe_steps_zero_based": list(FIVE_STEP_DIVERSITY_PROBE_STEPS),
        "probe_flow_times": list(FIVE_STEP_DIVERSITY_PROBE_TIMES),
        "selection_horizon": 20, "absolute_threshold": None,
    })
    print("Periodic tables are grouped by complete three-arm identities, not rollout count.")

    policy, preprocess, postprocess = models.load_pi05(
        repo_id=source_repo, revision=source_revision)
    if int(policy.config.chunk_size) != 50:
        raise ValueError(
            f"five-step design requires a 50-action model output, found {policy.config.chunk_size}")
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=methods, cohort="five_step_diversity_pro220",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_five_step_diversity_pro220",
        run_metadata={
            "source_model_repo_id": source_repo,
            "source_model_revision": source_revision,
            "bootstrap_manifest_hash": manifest_hash,
            "frozen_identity_manifest_hash": frozen_hash,
            "frozen_identity_manifest": manifest_payload,
            "historical_experiment": "pi05-direct-u20-gradient-pro220-v1",
            "historical_config_hash": historical_config_hash,
            "episode_indices": list(episode_indices), "suites": suites,
            "target_identities": len(manifest),
            "num_inference_steps": 5, "generated_chunk_size": 50,
            "n_action_steps": 10, "num_queries": 3, "pnp_k": 5,
            "pnp_steps": list(FIVE_STEP_DIVERSITY_PROBE_STEPS),
            "probe_flow_times": list(FIVE_STEP_DIVERSITY_PROBE_TIMES),
            "uncertainty_horizons": [10, 20, 50],
            "selection_uncertainty_horizon": 20,
            "candidate_seed_scheme": "stock_slot0_v1",
            "absolute_threshold": None,
            "requested_methods": [method for method, _ in methods],
            "config_hashes": expected_hashes,
        },
        provenance=gather_provenance(
            model_repo_id=source_repo, model_revision=source_revision),
        report_every=0, report_every_identities=25,
        initial_tally=initial_tally,
        initial_identity_methods=initial_identity_methods,
        historical_sr={"10-step x1 matched": historical_sr},
        progress_include_overall=True,
        progress_count_label=(
            "n = identities with this arm completed in THIS shard; each identity has one "
            "rollout per arm"),
        rollout_batch_size=1, resume_completed_only=True)
