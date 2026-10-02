"""Outcome-unfiltered, one-intervention-per-episode real flow PCP evaluation.

Initial states 0--4 have prior benchmark use, but are disjoint from the frozen
critic's tree cohorts. A new behavior stream is collected; this is not a claim
of unseen tasks or untouched initial states. Scheduled roots are fixed using
the environment horizon BEFORE observing episode length, uncertainty or return.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import smolvla_tree_collection as tree
from .experiments import LIBERO_SUITES, _run_collection
from .five_step_diversity_experiment import identity_manifest_payload
from .pcp_critic.resumable_snapshot import _download_with_retry
from .qplanning_critic.model import pool_prefix_tokens
from .smolvla_combined_success import COMBINED_SNAPSHOT_KEY
from .smolvla_focused_pcp_eval import EXPECTED_CHECKPOINT_SHA256
from .smolvla_flow_pcp_pilot import generate_flow_controls
from .smolvla_pcp_pilot import KINDS, complete_candidate_contract, load_scalar_checkpoint
from .smolvla_tree_bellman_finetune import _root_mask, _root_prefix, _json_digest
from .smolvla_tree_source_experiment import build_smolvla_tree_source_method
from .store import SupabaseStore, gather_provenance
from .verifier.collection import candidate_group_id

SOURCE_EXPERIMENT = "smolvla-libero-population-pcp-source-idx0-4-behavior1-v1"
EXPERIMENT = "smolvla-libero-population-flow-pcp-200-v1"
STAGES = {"early": 0.0, "middle": 1 / 3, "late": 2 / 3}
IDENTITIES = 200


def episode_key(ep):
    return ep["suite"], int(ep["task_idx"]), int(ep.get("ep_idx", ep.get("episode_idx")))


def population_manifest(episodes, snapshot):
    """Select identities and scheduled times without episode outcomes or U scores."""
    expected = {(suite, task, index) for suite in LIBERO_SUITES
                for task in range(10) for index in range(5)}
    if len(episodes) != IDENTITIES or {episode_key(e) for e in episodes} != expected:
        raise ValueError("require exactly all 40 tasks x five initial states 0--4")
    used = {(g["suite"], int(g["task_idx"]), int(g["episode_idx"])) for g in snapshot["groups"]}
    if used & expected:
        raise ValueError("evaluation identities overlap frozen critic tree data")
    # Stage allocation rotates across task and episode so every task gets every
    # stage and global counts are balanced, independently of success/length.
    items = []
    for suite_index, suite in enumerate(LIBERO_SUITES):
        for task in range(10):
            for index in range(5):
                ep = next(e for e in episodes if episode_key(e) == (suite, task, index))
                stage = tuple(STAGES)[(suite_index * 10 + task + index) % 3]
                chunk = int(STAGES[stage] * int(ep["max_steps"]) // 10)
                items.append({"suite": suite, "task_idx": task, "episode_idx": index,
                    "init_state_hash": ep["init_state_hash"], "behavior_seed_index": 1,
                    "stage": stage, "chunk_idx": chunk, "scheduled_step": chunk * 10,
                    "max_steps": int(ep["max_steps"])})
    items.sort(key=lambda i: tree._rank(EXPERIMENT, i["suite"], i["task_idx"], i["episode_idx"]))
    for ordinal, item in enumerate(items):
        item["ordinal"] = ordinal
    payload = {"experiment": EXPERIMENT, "source_experiment": SOURCE_EXPERIMENT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "critic_snapshot_digest": snapshot["snapshot_digest"],
        "items": items, "stage_fractions_of_environment_budget": STAGES,
        "selection": "all_40_tasks_x_indices0_4; no outcome or uncertainty filtering",
        "behavior_seed_index": 1, "initial_state_provenance": "previous policy benchmarks",
        "correction_step": 3, "correction_s": .7, "clean_rms": [.02, .06],
        "controls": sorted(KINDS), "root_only_intervention": True,
        "early_termination": "retain in full episode denominator; all arms equal source outcome",
        "primary_comparison": "q_plus_large versus random_large",
        "secondary_comparisons": "all predefined live controls versus zero; same-radius descent",
        "uncertainty_analysis": "posthoc descriptive baseline-U10 quartiles, no gating or tuning",
        "experimental_unit": "episode; repeated task dependence also reported"}
    return {**payload, "manifest_hash": tree._digest(payload)}


def reached_schedule(source_row, item):
    # A terminal boundary has no action proposal. Never replace the scheduled
    # time with the last surviving root: that would condition sampling on return.
    return int(source_row["n_steps"]) > item["scheduled_step"] and int(source_row["n_chunks"]) > item["chunk_idx"]


def prepare_population_episodes():
    from . import libero_env
    bd = libero_env.init_libero_benchmark()
    tasks = [(suite, task) for suite in LIBERO_SUITES for task in range(bd[suite]().n_tasks)]
    episodes = libero_env.build_final_episodes(bd, episode_idxs=list(range(5)), tasks=tasks)
    return [{**e, "behavior_seed_index": 1} for e in episodes]


def preaction_row(bundle, item):
    source = tree._source_boundary(bundle, item)
    arrays, boundary = bundle["arrays"], source["boundary_index"]
    prefix = _root_prefix(arrays["prefix/prefix_embeddings"][boundary])
    pad = _root_mask(arrays["prefix/prefix_pad_masks"][boundary], len(prefix))
    pooled, valid = pool_prefix_tokens(torch.from_numpy(prefix.astype(np.float32))[None],
                                      torch.from_numpy(pad)[None], 128)
    remaining = item["source_max_steps"] - source["root_step"]
    if remaining < 1:
        raise ValueError("scheduled root has no preaction budget")
    return {"prefix": pooled[0].numpy().astype(np.float16), "pad": valid[0].numpy(),
        "robot": np.r_[np.asarray(source["raw_robot"], np.float32).reshape(-1),
                        np.float32(remaining / item["source_max_steps"])],
        "proprio": np.asarray(source["policy_proprio"], np.float32).reshape(-1),
        "action_valid": (np.arange(10) < min(10, remaining))[None]}


def _source_rows(store):
    return store.fetch_all("rollouts", "*", configure=lambda q:q.eq(
        "experiment", SOURCE_EXPERIMENT).eq("status", "completed"), order_by=("rollout_id",))


def run_population_pcp_eval(*, checkpoint_path, shard_index=0, shard_count=3,
                            episode_limit=None, run_intervention=False, store=None,
                            output_path="population_pcp_report.json", device="cuda"):
    """A shard collects its source episodes then evaluates their scheduled roots.

    Completed nine-candidate groups resume safely. Partial roots repeat; errors
    are never dropped or replaced. Paid dispatch belongs to the explicit launcher.
    """
    if shard_count < 1 or shard_index not in range(shard_count) or (episode_limit is not None and episode_limit < 1):
        raise ValueError("invalid sharding/limit")
    store = store or SupabaseStore()
    snapshot = json.loads(_download_with_retry(store, COMBINED_SNAPSHOT_KEY))
    if _json_digest({k:v for k,v in snapshot.items() if k != "snapshot_digest"}) != snapshot["snapshot_digest"]:
        raise ValueError("frozen critic snapshot digest mismatch")
    episodes = prepare_population_episodes()
    manifest = population_manifest(episodes, snapshot)
    model, metadata = load_scalar_checkpoint(checkpoint_path, snapshot, device)
    if metadata["critic_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("predeclared PCP checkpoint changed")
    key = f"smolvla_pcp_pilot/manifests/{EXPERIMENT}/{manifest['manifest_hash']}.json"
    selected = [i for i in manifest["items"] if i["ordinal"] % shard_count == shard_index][:episode_limit]
    result = {"experiment": EXPERIMENT, "cohort_manifest": manifest, **metadata,
        "interpretation": "critic-tree-disjoint episode evaluation; previously benchmarked initial states",
        "shard_index": shard_index, "shard_count": shard_count,
        "planned_episodes": len(selected), "run_intervention": run_intervention, "outcomes": []}
    Path(output_path).write_text(json.dumps(result, indent=2))
    if not run_intervention:
        return result
    try:
        old = json.loads(_download_with_retry(store, key))
    except Exception as error:
        if not any(t in str(error).lower() for t in ("404", "not found", "does not exist")):
            raise
        store._upload(key, json.dumps(manifest, sort_keys=True).encode())
        old = json.loads(_download_with_retry(store, key))
    if old != manifest:
        raise ValueError("immutable population manifest changed")
    from . import models, libero_env
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA required for simulator collection")
    eps = {episode_key(e):e for e in episodes}
    # Preflight checkpoint, split and manifest BEFORE starting any simulator run.
    policy, preprocess, postprocess = models.load_smolvla()
    method, config = build_smolvla_tree_source_method()
    provenance = gather_provenance(model_repo_id="HuggingFaceVLA/smolvla_libero")
    provenance["policy_model"] = "smolvla"
    _run_collection(store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=SOURCE_EXPERIMENT,
        episodes=[eps[episode_key(i)] for i in selected], methods=[(method, config)],
        cohort=SOURCE_EXPERIMENT, shard_count=shard_count, shard_index=shard_index,
        benchmark="libero", driver="smolvla_population_pcp_source",
        run_metadata={"cohort_manifest": manifest, "frozen_identity_manifest": identity_manifest_payload(episodes)},
        report_every=0, report_every_identities=10, rollout_batch_size=1,
        provenance=provenance, resume_completed_only=True)
    source_rows = {}
    for row in _source_rows(store):
        identity = episode_key(row)
        if identity in source_rows:
            raise ValueError("duplicate population source identity")
        source_rows[identity] = row
    store.start_run("smolvla_population_flow_pcp", "libero", EXPERIMENT, config=result)
    status, new_roots = "failed", 0
    started = time.monotonic()
    report_key = f"smolvla_pcp_pilot/reports/{EXPERIMENT}/shard_{shard_index}.json"
    try:
        for plan in selected:
            source_row = source_rows[episode_key(plan)]
            if (source_row["init_state_hash"] != plan["init_state_hash"] or
                    int((source_row.get("metadata_json") or {}).get("behavior_seed_index", 1)) != 1):
                raise ValueError("source initial state/behavior stream mismatch")
            record = {"identity": list(episode_key(plan)), "stage": plan["stage"],
                "scheduled_step": plan["scheduled_step"], "source_rollout_id": source_row["rollout_id"],
                "reached_schedule": reached_schedule(source_row, plan)}
            if not record["reached_schedule"]:
                record.update(success_by_control={k:bool(source_row["success"]) for k in KINDS},
                              no_intervention_reason="episode terminated before scheduled decision")
            else:
                item = {**plan, "source_rollout_id": source_row["rollout_id"],
                    "source_success": bool(source_row["success"]), "source_n_steps": int(source_row["n_steps"]),
                    "source_n_chunks": int(source_row["n_chunks"]), "source_episode_seed": int(source_row["episode_seed"]),
                    "source_max_steps": int(source_row["max_steps"]), "selection_strategy": "fixed_episode_budget_stage"}
                gid = candidate_group_id("libero", item["suite"], item["task_idx"], item["episode_idx"],
                    item["chunk_idx"], namespace=EXPERIMENT, trajectory_seed=item["source_episode_seed"])
                existing = store.fetch_all("verifier_candidate_groups", "manifest_hash", configure=lambda q:q.eq(
                    "candidate_group_id", gid), order_by=("candidate_group_id",))
                if existing and any(r["manifest_hash"] != manifest["manifest_hash"] for r in existing):
                    raise ValueError("population candidate contract changed")
                candidates = store.fetch_all("verifier_candidates", "candidate_kind,success,n_steps,metadata_json",
                    configure=lambda q:q.eq("candidate_group_id", gid), order_by=("candidate_kind",)) if existing else []
                if not complete_candidate_contract(candidates, expected_critic_sha=EXPECTED_CHECKPOINT_SHA256):
                    bundle = tree._load_source_bundle(store, source_row)
                    source = tree._source_boundary(bundle, item)
                    row = preaction_row(bundle, item)
                    ep = eps[episode_key(plan)]
                    extras = generate_flow_controls(policy=policy, preprocess=preprocess, item=item,
                        bundle=bundle, source=source, row=row, critic=model, valid=row["action_valid"][0],
                        random_seed=tree._seed(EXPERIMENT, source_row["rollout_id"], item["chunk_idx"]),
                        task_description=ep["task_desc"])
                    env = libero_env.make_env(ep["bddl_path"])
                    try:
                        group, candidates = tree.collect_smolvla_depth1_tree(env, ep, policy, preprocess,
                            postprocess, models.default_device(), item=item, bundle=bundle,
                            manifest_hash=manifest["manifest_hash"], experiment=EXPERIMENT,
                            fresh_count=1, perturb_count=0, extra_policy_chunks=extras)
                    finally:
                        env.close()
                    if group["candidate_group_id"] != gid or {c["candidate_kind"] for c in candidates} != KINDS:
                        raise RuntimeError("population group/candidate contract mismatch")
                    profile = tree.weighted_u10_profile(_download_with_retry(store, source_row["ahats_path"]))
                    group["metadata_json"].update(stage=plan["stage"], baseline_u10=profile[item["chunk_idx"]],
                        critic_sha256=EXPECTED_CHECKPOINT_SHA256, original_split="evaluation",
                        initial_state_provenance="previous benchmark; new behavior stream")
                    for candidate in candidates:
                        candidate["metadata_json"].update(critic_sha256=EXPECTED_CHECKPOINT_SHA256,
                            original_split="evaluation", stage=plan["stage"])
                    store.register_candidate_group(group, candidates)
                    new_roots += 1
                record.update(candidate_group_id=gid, success_by_control={c["candidate_kind"]:bool(c["success"]) for c in candidates})
            result["outcomes"].append(record)
            Path(output_path).write_text(json.dumps(result, indent=2))
            store._upload(report_key, json.dumps(result, sort_keys=True).encode())
            print({"completed_episodes": len(result["outcomes"]), "planned_episodes": len(selected),
                   "new_intervention_roots": new_roots, "elapsed_minutes": round((time.monotonic()-started)/60, 1)}, flush=True)
        status = "completed"
    finally:
        store.finish_run(status=status, n_rollouts=new_roots * 8)
    return result
