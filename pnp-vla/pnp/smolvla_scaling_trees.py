"""Second, disjoint 800-root SmolVLA fresh8 tree cohort (indices 30--49)."""
from __future__ import annotations

import json
import time

import torch

from . import smolvla_tree_collection as tree
from .smolvla_scaling_source import (
    SOURCE_EXPERIMENT, SOURCE_IDENTITIES, prepare_scaling_source_episodes,
)
from .smolvla_tree_source_experiment import SMOLVLA_TREE_PRIORITY_FRACTION
from .store import SupabaseStore
from .verifier.collection import candidate_group_id


TREE_EXPERIMENT = "smolvla-libero-depth1-fresh8-trees-idx30-49-v1-bellman"
MANIFEST_PATH = "smolvla_trees/manifests/idx30_49_depth1_fresh8_v1_bellman.json"
# This value is part of the already-persisted manifest hash. Do not change it.
TREE_SHARDS = 2
KINDS = {"stored_source", *(f"fresh_seed_{index}" for index in range(1, 9))}


def scaling_source_rows(store) -> list[dict]:
    rows = store.fetch_all(
        "rollouts",
        "rollout_id,benchmark,suite,task_idx,episode_idx,init_state_hash,success,n_steps,"
        "n_chunks,max_steps,episode_seed,config_hash,ahats_path,trajectory_path,"
        "training_data_path,status",
        configure=lambda query: query.eq("experiment", SOURCE_EXPERIMENT).eq(
            "method", "smolvla_pnp_steps123_k311").eq("status", "completed"),
        order_by=("rollout_id",),
    )
    identities = {(r["suite"], int(r["task_idx"]), int(r["episode_idx"]),
                   r["init_state_hash"]) for r in rows}
    if len(rows) != SOURCE_IDENTITIES or len(identities) != len(rows):
        raise ValueError(f"need {SOURCE_IDENTITIES} complete distinct new source episodes; found {len(rows)}")
    if len({r["config_hash"] for r in rows}) != 1:
        raise ValueError("new source cohort has mixed policy configurations")
    if any(not all(r.get(field) for field in (
            "ahats_path", "trajectory_path", "training_data_path")) for r in rows):
        raise ValueError("new source cohort has missing rich artifacts")
    return rows


def _group_id(item):
    return candidate_group_id(
        "libero", item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=TREE_EXPERIMENT,
        trajectory_seed=item["source_episode_seed"])


def _build_manifest(store):
    rows = scaling_source_rows(store)
    priority_count = round(SMOLVLA_TREE_PRIORITY_FRACTION * len(rows))
    priority_ids = tree._balanced_priority_ids(rows, priority_count)
    items = []
    for index, row in enumerate(rows, 1):
        profile = tree.weighted_u10_profile(tree._download_with_retry(
            store, str(row["ahats_path"])))
        if len(profile) != int(row["n_chunks"]):
            raise ValueError(f"U10 profile/chunk mismatch for {row['rollout_id']}")
        eligible, fallback = tree._eligible_roots(profile)
        if not eligible:
            raise ValueError(f"no branchable boundary for {row['rollout_id']}")
        strategy = ("u10_priority" if row["rollout_id"] in priority_ids else "uniform")
        if strategy == "u10_priority":
            root = min(eligible, key=lambda chunk: (
                -profile[chunk], tree._rank("u10-tie", row["rollout_id"], chunk)))
        else:
            root = eligible[int(tree._rank("uniform-root", row["rollout_id"]), 16)
                            % len(eligible)]
        items.append({
            "source_rollout_id": str(row["rollout_id"]),
            "suite": str(row["suite"]), "task_idx": int(row["task_idx"]),
            "episode_idx": int(row["episode_idx"]),
            "init_state_hash": str(row["init_state_hash"]),
            "source_success": bool(row["success"]),
            "source_n_steps": int(row["n_steps"]),
            "source_n_chunks": int(row["n_chunks"]),
            "source_episode_seed": int(row["episode_seed"]),
            "source_max_steps": int(row["max_steps"]),
            "selection_strategy": strategy, "chunk_idx": int(root),
            "source_root_u10": float(profile[root]),
            "source_u10_profile": list(profile),
            "short_episode_root_fallback": bool(fallback),
            "selection_tiebreak": tree._rank("tree-order", row["rollout_id"]),
        })
        if index % 100 == 0:
            print({"new_root_profiles": index, "total": SOURCE_IDENTITIES}, flush=True)
    items.sort(key=lambda item: item["selection_tiebreak"])
    for ordinal, item in enumerate(items):
        item["ordinal"] = ordinal
        item["shard_index"] = ordinal % TREE_SHARDS
    payload = {
        "version": 1, "source_experiment": SOURCE_EXPERIMENT,
        "experiment": TREE_EXPERIMENT, "trees": SOURCE_IDENTITIES,
        "shards": TREE_SHARDS, "candidate_count": 9,
        "candidate_families": {"stored_source": 1, "fresh_initial_noise": 8,
                               "fixed_initial_noise_new_pnp_perturbation": 0},
        "priority_fraction": SMOLVLA_TREE_PRIORITY_FRACTION,
        "uniform_fraction": 1.0 - SMOLVLA_TREE_PRIORITY_FRACTION,
        "root_uncertainty": "(3,1,1)-pair-weighted U10",
        "minimum_preferred_boundaries_remaining": tree.SMOLVLA_TREE_MIN_BOUNDARIES,
        "integration_steps": 10, "executed_actions": 10,
        "pnp_steps": list(tree.SMOLVLA_SCHEDULE_STEPS),
        "pnp_k_by_step": list(tree.SMOLVLA_SCHEDULE_K_BY_STEP),
        "branch_artifact": "sequential_q10_bellman_v1",
        "items": items,
    }
    document = {"manifest_hash": tree._digest(payload), "payload": payload}
    store._upload(MANIFEST_PATH, tree._canonical_json(document))
    return document


def load_or_build_scaling_manifest(store=None):
    store = store or SupabaseStore()
    try:
        document = json.loads(tree._download_with_retry(store, MANIFEST_PATH))
    except Exception as error:
        message = str(error).lower()
        if not any(term in message for term in ("404", "not found", "does not exist")):
            raise
        _build_manifest(store)
        document = json.loads(tree._download_with_retry(store, MANIFEST_PATH))
    payload = document.get("payload")
    if (not isinstance(payload, dict) or payload.get("version") != 1
            or payload.get("source_experiment") != SOURCE_EXPERIMENT
            or payload.get("experiment") != TREE_EXPERIMENT
            or len(payload.get("items", ())) != SOURCE_IDENTITIES
            or tree._digest(payload) != document.get("manifest_hash")):
        raise ValueError("new scaling-tree manifest is incompatible")
    return document


def _complete_groups(store, items, manifest_hash):
    allowed = {_group_id(item) for item in items}
    rows = store.fetch_all(
        "verifier_candidate_groups", "candidate_group_id,manifest_hash,metadata_json",
        configure=lambda q: q.eq("experiment", TREE_EXPERIMENT),
        order_by=("candidate_group_id",))
    groups = {row["candidate_group_id"]: row for row in rows
              if row["candidate_group_id"] in allowed}
    by_group = {gid: [] for gid in groups}
    ids = list(groups)
    for start in range(0, len(ids), 75):
        batch = ids[start:start + 75]
        candidates = store.fetch_all(
            "verifier_candidates", "candidate_group_id,candidate_kind,metadata_json",
            configure=lambda q, batch=batch: q.in_("candidate_group_id", batch),
            order_by=("candidate_group_id", "candidate_kind"))
        for candidate in candidates:
            by_group[candidate["candidate_group_id"]].append(candidate)
    return {gid for gid, group in groups.items()
            if group["manifest_hash"] == manifest_hash
            and len(by_group[gid]) == 9
            and {c["candidate_kind"] for c in by_group[gid]} == KINDS
            and all((c.get("metadata_json") or {}).get("training_data_path")
                    for c in by_group[gid])}


def _worker_items(manifest: dict, shard_index: int, shard_count: int) -> list[dict]:
    if not 1 <= shard_count <= SOURCE_IDENTITIES or shard_index not in range(shard_count):
        raise ValueError("shard_count must be 1..800 and shard_index must be in range")
    items = [dict(item) for item in manifest["payload"]["items"]
             if int(item["ordinal"]) % shard_count == shard_index]
    expected = sum(index % shard_count == shard_index for index in range(SOURCE_IDENTITIES))
    if len(items) != expected:
        raise ValueError("tree worker partition differs from the frozen manifest")
    return items


def run_scaling_fresh8_tree_worker(*, shard_index: int,
                                   shard_count: int,
                                   tree_limit: int | None = 1,
                                   store=None):
    """Resume-safe worker partition over the original immutable root manifest."""
    from . import libero_env, models

    if not 1 <= shard_count <= SOURCE_IDENTITIES or shard_index not in range(shard_count):
        raise ValueError("shard_count must be 1..800 and shard_index must be in range")
    if tree_limit is not None and tree_limit < 1:
        raise ValueError("tree_limit must be positive or None")
    store = store or SupabaseStore()
    manifest = load_or_build_scaling_manifest(store)
    items = _worker_items(manifest, shard_index, shard_count)
    if tree_limit is not None:
        items = items[:tree_limit]
    complete = _complete_groups(store, items, manifest["manifest_hash"])
    pending = [item for item in items if _group_id(item) not in complete]
    print({"experiment": TREE_EXPERIMENT,
           "manifest_hash": manifest["manifest_hash"],
           "shard": f"{shard_index}/{shard_count}",
           "requested_roots": len(items), "complete_roots": len(complete),
           "pending_roots": len(pending),
           "new_branches_per_root": 8}, flush=True)
    if not pending:
        return {"new_trees": 0, "complete_trees": len(complete)}
    if not torch.cuda.is_available():
        raise RuntimeError("select a CUDA Colab runtime")
    egl = tree.subprocess.run(
        "ldconfig -p | grep -q libEGL_nvidia", shell=True,
        stdout=tree.subprocess.DEVNULL, stderr=tree.subprocess.DEVNULL)
    if egl.returncode:
        raise RuntimeError("NVIDIA EGL is unavailable")
    source_rows = {r["rollout_id"]: r for r in scaling_source_rows(store)}
    episodes = prepare_scaling_source_episodes()
    lookup = {(ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"])): ep
              for ep in episodes}
    policy, preprocess, postprocess = models.load_smolvla()
    device = models.default_device()
    store.start_run(
        "smolvla_scaling_fresh8_tree_collection", "libero", TREE_EXPERIMENT,
        config={"source_manifest_hash": manifest["manifest_hash"],
                "shard_count": shard_count, "shard_index": shard_index,
                "requested_roots": len(items), "candidate_count": 9,
                "fresh_initial_noise_candidates": 8, "integration_steps": 10,
                "n_action_steps": 10, "source_episode_indices": [30, 49],
                "videos": False})
    status, completed = "failed", 0
    began = time.monotonic()
    try:
        for item in pending:
            tree_began = time.monotonic()
            print({"starting_tree": completed + 1, "pending_at_start": len(pending),
                   "ordinal": item["ordinal"], "shard": f"{shard_index}/{shard_count}"},
                  flush=True)
            row = source_rows[item["source_rollout_id"]]
            bundle = tree._load_source_bundle(store, row)
            ep = lookup[(item["suite"], int(item["task_idx"]), int(item["episode_idx"]))]
            env = libero_env.make_env(ep["bddl_path"])
            try:
                group, candidates = tree.collect_smolvla_depth1_tree(
                    env, ep, policy, preprocess, postprocess, device,
                    item=item, bundle=bundle,
                    manifest_hash=manifest["manifest_hash"],
                    experiment=TREE_EXPERIMENT,
                    fresh_count=8, perturb_count=0)
            finally:
                env.close()
            if (group["candidate_group_id"] != _group_id(item)
                    or {c["candidate_kind"] for c in candidates} != KINDS):
                raise RuntimeError("scaling tree violated fresh8 candidate contract")
            store.register_candidate_group(group, candidates)
            completed += 1
            print({"completed_this_run": completed,
                   "pending_at_start": len(pending),
                   "ordinal": item["ordinal"],
                   "tree_seconds": round(time.monotonic() - tree_began, 1),
                   "elapsed_minutes": round((time.monotonic() - began) / 60, 1)},
                  flush=True)
        status = "completed"
    finally:
        store.finish_run(status=status, n_rollouts=8 * completed)
    return {"new_trees": completed, "complete_trees": len(items)}
