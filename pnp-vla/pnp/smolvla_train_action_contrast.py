"""Training-only controlled contrasts selected from the frozen old 800 roots.

Selection uses historical labels, never current critic scores or new outcomes.
The artifacts are a separate dataset; this module does not ingest training data.
"""
from __future__ import annotations

from collections import Counter
import json
import time

import torch

from . import smolvla_tree_collection as tree
from .pcp_critic.resumable_snapshot import _download_with_retry
from .smolvla_action_contrast_pilot import AXES, KINDS, SCALES, _controlled_chunks
from .smolvla_combined_success import COMBINED_SNAPSHOT_KEY
from .smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS
from .smolvla_tree_bellman_finetune import _json_digest, _split_groups
from .smolvla_tree_source_experiment import prepare_smolvla_tree_source_episodes
from .store import SupabaseStore
from .verifier.collection import candidate_group_id


EXPERIMENT = "smolvla-libero-q10-training-action-contrast-v1"
ROOT_COUNT = 12
SHARD_COUNT = 3
EXPECTED_SNAPSHOT_DIGEST = "671b5b211099997fc83d1277"
QUOTAS = {"mixed": 6, "all_failure": 3, "all_success": 3}


def _key(row):
    return (row["suite"], int(row["task_idx"]), int(row["episode_idx"]),
            int(row["chunk_idx"]))


def select_manifest(snapshot, source_document):
    """Pure deterministic selection, preserving the original cohort split."""
    if (snapshot.get("snapshot_digest") != EXPECTED_SNAPSHOT_DIGEST
            or len(snapshot["groups"]) != 1600):
        raise ValueError("requires the original frozen 1600-root training snapshot")
    cohort = [g for g in snapshot["groups"]
              if g["source_experiment"] == FRESH8_EXPERIMENT]
    if len(cohort) != 800 or len({_key(g) for g in cohort}) != 800:
        raise ValueError("requires exactly 800 distinct original-cohort roots")
    if len({g["candidate_group_id"] for g in cohort}) != 800:
        raise ValueError("duplicate original candidate group IDs")
    sources = {_key(i): i for i in source_document["payload"]["items"]}
    if len(sources) != 800:
        raise ValueError("requires 800 distinct immutable source roots")
    train, validation = _split_groups(cohort)
    train, validation = set(train), set(validation)
    buckets = {name: [] for name in QUOTAS}
    for group in cohort:
        if group["candidate_group_id"] not in train:
            continue
        candidates = group["candidates"]
        if (len(candidates) != 9
                or {c["candidate_kind"] for c in candidates} != set(FRESH8_KINDS)
                or candidates[0]["candidate_kind"] != "stored_source"):
            raise ValueError("historical training candidate contract differs")
        outcomes = [bool(c["success"]) for c in candidates]
        stratum = ("mixed" if any(outcomes) and not all(outcomes)
                   else "all_success" if all(outcomes) else "all_failure")
        item = sources.get(_key(group))
        if item is None or not 10 <= int(item["episode_idx"]) <= 29:
            raise ValueError("training root absent from original idx10-29 sources")
        if bool(item["source_success"]) != outcomes[0]:
            raise ValueError("stored-source outcome differs from frozen snapshot")
        buckets[stratum].append((item, group))
    eligible = {name: len(rows) for name, rows in buckets.items()}
    selected, suites, tasks = [], Counter(), Counter()
    for stratum, quota in QUOTAS.items():
        remaining = list(buckets[stratum])
        if len(remaining) < quota:
            raise ValueError(f"need {quota} training roots in {stratum}, found {len(remaining)}")
        for _ in range(quota):
            item, group = min(remaining, key=lambda pair: (
                suites[pair[0]["suite"]],
                tasks[(pair[0]["suite"], int(pair[0]["task_idx"]))],
                tree._rank(EXPERIMENT, pair[1]["candidate_group_id"])))
            remaining.remove((item, group))
            selected.append({**{k: v for k, v in item.items() if k != "shard_index"},
                "original_group_id": group["candidate_group_id"],
                "original_split": "train", "historical_stratum": stratum,
                "historical_successes": sum(bool(c["success"]) for c in group["candidates"]),
                "source_training_data_path": group["source_training_data_path"]})
            suites[item["suite"]] += 1
            tasks[(item["suite"], int(item["task_idx"]))] += 1
    selected.sort(key=lambda i: tree._rank(EXPERIMENT + "-order", i["original_group_id"]))
    items = [{**item, "ordinal": index} for index, item in enumerate(selected)]
    if (len(items) != ROOT_COUNT
            or len({i["source_rollout_id"] for i in items}) != ROOT_COUNT
            or any(i["original_group_id"] in validation for i in items)):
        raise ValueError("training selection is not distinct or crosses validation")
    protocol = {"version": 1, "experiment": EXPERIMENT,
        "snapshot_digest": snapshot["snapshot_digest"],
        "source_manifest_hash": source_document["manifest_hash"],
        "source_experiment": FRESH8_EXPERIMENT, "episode_indices": [10, 29],
        "original_split": "train", "root_count": ROOT_COUNT, "quotas": QUOTAS,
        "selection": "historical_outcome_strata_then_suite_task_diversity_then_stable_hash",
        "candidate_kinds": sorted(KINDS), "scales": list(SCALES), "axes": list(AXES),
        "time_profile": "sin(pi*(arange(10)+0.5)/10)", "normalized_action_clip": [-1, 1],
        "root_only_intervention": True, "n_action_steps": 10,
        "preserve_full_proposal_and_tail": True, "input_mask": "preaction_deadline_only",
        "continuation": "frozen_source_PnP_with_paired_source_future_noise_and_perturb_streams",
        "continuation_seeds_per_root": 1, "integration_steps": 10,
        "pnp_steps": [1, 2, 3], "pnp_k_by_step": [3, 1, 1],
        "automatic_training_ingestion": False, "independent_evaluation": False}
    audit = {"old_cohort_roots": 800, "original_train_roots": len(train),
        "original_validation_roots": len(validation), "eligible_by_stratum": eligible,
        "selected_by_stratum": dict(Counter(i["historical_stratum"] for i in items)),
        "selected_by_suite": dict(suites),
        "selected_task_count": len(tasks), "validation_overlap": 0}
    payload = {"protocol": protocol, "audit": audit, "items": items}
    return {**payload, "manifest_hash": tree._digest(payload)}


def shard_items(document, shard_index, shard_count=SHARD_COUNT, tree_limit=None):
    if (shard_count < 1 or shard_index not in range(shard_count)
            or (tree_limit is not None and tree_limit < 1)):
        raise ValueError("invalid shard or tree limit")
    return [i for i in document["items"] if i["ordinal"] % shard_count == shard_index][:tree_limit]


def _group_id(item):
    return candidate_group_id("libero", item["suite"], item["task_idx"],
        item["episode_idx"], item["chunk_idx"], namespace=EXPERIMENT,
        trajectory_seed=item["source_episode_seed"])


def persist_manifest(store, document):
    """Content-addressed immutable protocol; fail closed on conflicting content."""
    key = f"smolvla_trees/manifests/training_action_contrast_v1/{document['manifest_hash']}.json"
    try:
        saved = json.loads(_download_with_retry(store, key))
    except Exception as error:
        if not any(token in str(error).lower() for token in ("404", "not found", "does not exist")):
            raise
        store._upload(key, json.dumps(document, sort_keys=True).encode())
        saved = json.loads(_download_with_retry(store, key))
    if saved != document:
        raise ValueError("persisted training action-contrast manifest differs")
    return key


def _completed(store, items, manifest_hash):
    allowed = {_group_id(item): item for item in items}
    groups = store.fetch_all("verifier_candidate_groups", "candidate_group_id,manifest_hash",
        configure=lambda q: q.eq("experiment", EXPERIMENT), order_by=("candidate_group_id",))
    complete = set()
    for group in groups:
        gid = group["candidate_group_id"]
        if gid not in allowed:
            continue
        if group["manifest_hash"] != manifest_hash:
            raise ValueError("existing training contrast group has a different manifest")
        rows = store.fetch_all("verifier_candidates", "candidate_kind,metadata_json",
            configure=lambda q, gid=gid: q.eq("candidate_group_id", gid),
            order_by=("candidate_kind",))
        if (len(rows) == len(KINDS) and {r["candidate_kind"] for r in rows} == KINDS
                and all((r.get("metadata_json") or {}).get("training_data_path")
                    and r["metadata_json"].get("original_split") == "train"
                    and r["metadata_json"].get("original_group_id") == allowed[gid]["original_group_id"]
                    and r["metadata_json"].get("collection_manifest_hash") == manifest_hash
                    for r in rows)):
            complete.add(gid)
    return complete


def run_training_action_contrast(*, shard_index=0, shard_count=SHARD_COUNT,
                                tree_limit=None, run_collection=False, store=None):
    """Preview by default; collection and persistence require explicit opt-in."""
    # Validate before reading storage or importing simulator/model dependencies.
    shard_items({"items": []}, shard_index, shard_count, tree_limit)
    store = store or SupabaseStore()
    snapshot = json.loads(_download_with_retry(store, COMBINED_SNAPSHOT_KEY))
    if _json_digest({k: v for k, v in snapshot.items() if k != "snapshot_digest"}) != snapshot["snapshot_digest"]:
        raise ValueError("frozen combined snapshot digest mismatch")
    source_document = tree.load_or_build_smolvla_tree_manifest(store)
    document = select_manifest(snapshot, source_document)
    items = shard_items(document, shard_index, shard_count, tree_limit)
    complete = _completed(store, items, document["manifest_hash"])
    pending = [i for i in items if _group_id(i) not in complete]
    result = {"experiment": EXPERIMENT, "manifest": document,
        "shard_index": shard_index, "shard_count": shard_count,
        "requested_roots": len(items), "complete_trees": len(complete), "new_trees": 0,
        "pending_source_ids": [i["source_rollout_id"] for i in pending],
        "candidates_per_root": len(KINDS), "run_collection": bool(run_collection)}
    print({k: v for k, v in result.items() if k != "manifest"}, flush=True)
    if not run_collection or not pending:
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("select a CUDA runtime before opting into collection")
    egl = tree.subprocess.run("ldconfig -p | grep -q libEGL_nvidia", shell=True,
        stdout=tree.subprocess.DEVNULL, stderr=tree.subprocess.DEVNULL)
    if egl.returncode:
        raise RuntimeError("NVIDIA EGL is unavailable")
    from . import libero_env, models

    manifest_path = persist_manifest(store, document)
    rows = {r["rollout_id"]: r for r in tree._source_rows(store)}
    for item in pending:
        if rows[item["source_rollout_id"]]["training_data_path"] != item["source_training_data_path"]:
            raise ValueError("source artifact no longer matches frozen training snapshot")
    episodes = prepare_smolvla_tree_source_episodes()
    lookup = {(e["suite"], int(e["task_idx"]), int(e["ep_idx"])): e for e in episodes}
    policy, preprocess, postprocess = models.load_smolvla()
    store.start_run("smolvla_training_action_contrast", "libero", EXPERIMENT,
        config={**document["protocol"], "manifest_hash": document["manifest_hash"],
            "manifest_path": manifest_path, "shard_index": shard_index,
            "shard_count": shard_count, "tree_limit": tree_limit})
    status, count, began = "failed", 0, time.monotonic()
    try:
        for item in pending:
            bundle = tree._load_source_bundle(store, rows[item["source_rollout_id"]])
            source = tree._source_boundary(bundle, item)
            ep = lookup[(item["suite"], int(item["task_idx"]), int(item["episode_idx"]))]
            env = libero_env.make_env(ep["bddl_path"])
            try:
                group, candidates = tree.collect_smolvla_depth1_tree(
                    env, ep, policy, preprocess, postprocess, models.default_device(),
                    item=item, bundle=bundle, manifest_hash=document["manifest_hash"],
                    experiment=EXPERIMENT, fresh_count=1, perturb_count=0,
                    extra_policy_chunks=_controlled_chunks(source["policy_chunk"]))
            finally:
                env.close()
            if (group["candidate_group_id"] != _group_id(item)
                    or len(candidates) != len(KINDS)
                    or {c["candidate_kind"] for c in candidates} != KINDS):
                raise RuntimeError("training action-contrast candidate contract mismatch")
            provenance = {"original_split": "train", "original_group_id": item["original_group_id"],
                "historical_stratum": item["historical_stratum"],
                "collection_manifest_hash": document["manifest_hash"],
                "collection_manifest_path": manifest_path,
                "snapshot_digest": snapshot["snapshot_digest"],
                "automatic_training_ingestion": False}
            group["metadata_json"].update(provenance)
            for candidate in candidates:
                candidate["metadata_json"].update(provenance)
            store.register_candidate_group(group, candidates)
            count += 1
            print({"completed_this_run": count, "pending_at_start": len(pending),
                "elapsed_minutes": round((time.monotonic() - began) / 60, 1)}, flush=True)
        status = "completed"
    finally:
        store.finish_run(status=status, n_rollouts=(len(KINDS) - 1) * count)
    result.update(new_trees=count, complete_trees=len(complete) + count,
                  collection_manifest_path=manifest_path)
    return result
