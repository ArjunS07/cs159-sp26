"""Three resume-safe Colab workers for fresh-noise-only SmolVLA Q10 trees.

The v3 source manifest fixes all 800 decision roots before v4 outcomes are
observed. Complete v3 roots contribute their four existing fresh-noise branches;
only fresh seeds 5-8 are collected for them. Roots without a complete v3 tree
collect all eight. The old experiment and artifacts stay untouched.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import time

import torch

from . import smolvla_tree_collection as tree
from .smolvla_tree_source_experiment import prepare_smolvla_tree_source_episodes
from .store import SupabaseStore
from .verifier.collection import candidate_group_id


SMOLVLA_FRESH8_EXPERIMENT = "smolvla-libero-depth1-fresh8-trees-v4-bellman"
FRESH8_KINDS = {"stored_source", *(f"fresh_seed_{i}" for i in range(1, 9))}
FRESH8_SHARDS = 3


def _group_id(item: dict) -> str:
    return candidate_group_id(
        "libero", item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=SMOLVLA_FRESH8_EXPERIMENT,
        trajectory_seed=item["source_episode_seed"])


def _v3_group_id(item: dict) -> str:
    return candidate_group_id(
        "libero", item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=tree.SMOLVLA_TREE_EXPERIMENT,
        trajectory_seed=item["source_episode_seed"])


def _reusable_v3(store, items: list[dict], manifest_hash: str) -> dict[str, list[dict]]:
    """Find complete v3 trees with all four persisted fresh branch artifacts."""
    wanted = {_v3_group_id(item): item for item in items}
    if not wanted:
        return {}
    groups = store.fetch_all(
        "verifier_candidate_groups", "candidate_group_id,manifest_hash,metadata_json",
        configure=lambda q: q.eq("experiment", tree.SMOLVLA_TREE_EXPERIMENT),
        order_by=("candidate_group_id",))
    groups = {row["candidate_group_id"]: row for row in groups
              if row["candidate_group_id"] in wanted
              and row.get("manifest_hash") == manifest_hash
              and (row.get("metadata_json") or {}).get("source_rollout_id")
                  == wanted[row["candidate_group_id"]]["source_rollout_id"]}
    rows = defaultdict(list)
    for start in range(0, len(groups), 75):
        subset = list(groups)[start:start + 75]
        candidates = store.fetch_all(
            "verifier_candidates",
            "candidate_group_id,candidate_kind,success,n_steps,metadata_json,"
            "policy_chunk_path,env_chunk_path,observation_path",
            configure=lambda q, subset=subset: q.in_("candidate_group_id", subset),
            order_by=("candidate_group_id", "candidate_kind"))
        for candidate in candidates:
            rows[candidate["candidate_group_id"]].append(candidate)
    expected = {"stored_source", *(f"fresh_seed_{i}" for i in range(1, 5)),
                *(f"pnp_perturb_{i}" for i in range(1, 5))}
    return {
        group_id: [row for row in candidates
                   if row["candidate_kind"].startswith("fresh_seed_")]
        for group_id, candidates in rows.items()
        if len(candidates) == 9
        and {row["candidate_kind"] for row in candidates} == expected
        and all((row.get("metadata_json") or {}).get("training_data_path")
                and row.get("policy_chunk_path") and row.get("env_chunk_path")
                and row.get("observation_path") for row in candidates)
    }


def _link_v3_fresh(store, group_id: str, candidates: list[dict]) -> None:
    """Link immutable v3 artifacts into the v4 candidate group without GPU work."""
    for source in candidates:
        kind = source["candidate_kind"]
        candidate = {key: source[key] for key in (
            "candidate_kind", "success", "n_steps", "policy_chunk_path",
            "env_chunk_path", "observation_path")}
        candidate.update({
            "candidate_id": hashlib.sha256(f"{group_id}|{kind}".encode()).hexdigest()[:24],
            "candidate_group_id": group_id,
            "rollout_id": None,
            "metadata_json": {
                **(source.get("metadata_json") or {}),
                "reused_from_v3_group_id": source["candidate_group_id"],
            },
        })
        store.client.table("verifier_candidates").upsert(
            store._json(candidate), on_conflict="candidate_id").execute()


def _complete_groups(store, items: list[dict], manifest_hash: str) -> set[str]:
    allowed = {_group_id(item) for item in items}
    groups = store.fetch_all(
        "verifier_candidate_groups", "candidate_group_id,manifest_hash,metadata_json",
        configure=lambda q: q.eq("experiment", SMOLVLA_FRESH8_EXPERIMENT),
        order_by=("candidate_group_id",))
    groups = {row["candidate_group_id"]: row for row in groups
              if row["candidate_group_id"] in allowed}
    rows = defaultdict(list)
    ids = sorted(groups)
    for start in range(0, len(ids), 75):
        subset = ids[start:start + 75]
        candidates = store.fetch_all(
            "verifier_candidates",
            "candidate_group_id,candidate_kind,metadata_json",
            configure=lambda q, subset=subset: q.in_("candidate_group_id", subset),
            order_by=("candidate_group_id", "candidate_kind"))
        for candidate in candidates:
            rows[candidate["candidate_group_id"]].append(candidate)
    complete = set()
    for group_id, group in groups.items():
        metadata = group.get("metadata_json") or {}
        candidates = rows[group_id]
        if (group.get("manifest_hash") == manifest_hash
                and metadata.get("candidate_families") == {
                    "stored_source": 1, "fresh_initial_noise": 8,
                    "fixed_initial_noise_new_pnp_perturbation": 0}
                and len(candidates) == 9
                and {row["candidate_kind"] for row in candidates} == FRESH8_KINDS
                and all((row.get("metadata_json") or {}).get("training_data_path")
                        for row in candidates)):
            complete.add(group_id)
    return complete


def run_smolvla_fresh8_worker(*, shard_count: int = FRESH8_SHARDS,
                              shard_index: int = 0,
                              tree_limit: int | None = None,
                              store=None) -> dict:
    """Collect a fixed shard; rerun the same shard to skip completed trees."""
    from . import libero_env, models

    if shard_count != FRESH8_SHARDS or shard_index not in range(shard_count):
        raise ValueError("fresh8 requires three fixed shards, indexed 0..2")
    if tree_limit is not None and tree_limit < 1:
        raise ValueError("tree_limit must be positive or None")
    store = store or SupabaseStore()
    manifest = tree.load_or_build_smolvla_tree_manifest(store)
    payload = manifest["payload"]
    if len(payload["items"]) != 800 or payload["candidate_count"] != 9:
        raise ValueError("unexpected frozen source-root manifest")
    items = [dict(item) for item in payload["items"]
             if int(item["ordinal"]) % shard_count == shard_index]
    expected = sum(index % shard_count == shard_index for index in range(800))
    if len(items) != expected:
        raise ValueError(f"expected {expected} roots in shard, found {len(items)}")
    if tree_limit is not None:
        items = items[:tree_limit]
    complete = _complete_groups(store, items, manifest["manifest_hash"])
    pending = [item for item in items if _group_id(item) not in complete]
    reusable = _reusable_v3(store, pending, manifest["manifest_hash"])
    print({
        "experiment": SMOLVLA_FRESH8_EXPERIMENT,
        "source_manifest_hash": manifest["manifest_hash"],
        "shard": f"{shard_index}/{shard_count}",
        "requested_roots": len(items),
        "complete_roots": len(complete),
        "pending_roots": len(pending),
        "roots_reusing_four_v3_branches": len(reusable),
        "new_branches_per_root": "4 with complete v3, otherwise 8",
        "continuations": "sequential same-seed frozen P&P",
    }, flush=True)
    if not pending:
        return {"new_trees": 0, "complete_trees": len(complete),
                "requested_trees": len(items)}
    if not torch.cuda.is_available():
        raise RuntimeError("select a Colab GPU before collecting fresh8 trees")
    print({"gpu": torch.cuda.get_device_name(0),
           "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)},
          flush=True)
    egl = tree.subprocess.run(
        "ldconfig -p | grep -q libEGL_nvidia", shell=True,
        stdout=tree.subprocess.DEVNULL, stderr=tree.subprocess.DEVNULL)
    if egl.returncode != 0:
        raise RuntimeError("NVIDIA EGL unavailable; run the notebook EGL setup cell")
    source_rows = {str(row["rollout_id"]): row for row in tree._source_rows(store)}
    episodes = prepare_smolvla_tree_source_episodes()
    lookup = {(ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"])): ep
              for ep in episodes}
    policy, preprocess, postprocess = models.load_smolvla()
    device = models.default_device()
    store.start_run(
        "smolvla_fresh8_tree_collection", "libero", SMOLVLA_FRESH8_EXPERIMENT,
        config={"source_manifest_hash": manifest["manifest_hash"],
                "shard_count": shard_count, "shard_index": shard_index,
                "requested_roots": len(items), "candidate_count": 9,
                "fresh_initial_noise_candidates": 8,
                "reuse_v3_fresh_roots": len(reusable),
                "integration_steps": 10, "n_action_steps": 10,
                "source_episode_indices": [10, 29], "videos": False})
    began = time.monotonic()
    new_trees = 0
    new_branches = 0
    status = "failed"
    try:
        for item in pending:
            started = time.monotonic()
            old_fresh = reusable.get(_v3_group_id(item), [])
            fresh_start = 5 if old_fresh else 1
            fresh_count = 4 if old_fresh else 8
            row = source_rows[item["source_rollout_id"]]
            bundle = tree._load_source_bundle(store, row)
            ep = lookup[(item["suite"], int(item["task_idx"]),
                         int(item["episode_idx"]))]
            env = libero_env.make_env(ep["bddl_path"])
            try:
                group, candidates = tree.collect_smolvla_depth1_tree(
                    env, ep, policy, preprocess, postprocess, device,
                    item=item, bundle=bundle,
                    manifest_hash=manifest["manifest_hash"],
                    experiment=SMOLVLA_FRESH8_EXPERIMENT,
                    fresh_count=fresh_count, perturb_count=0,
                    fresh_start_index=fresh_start)
            finally:
                env.close()
            kinds = {candidate["candidate_kind"] for candidate in candidates}
            expected_kinds = {"stored_source", *(
                f"fresh_seed_{i}" for i in range(fresh_start, fresh_start + fresh_count))}
            if (group["candidate_group_id"] != _group_id(item)
                    or kinds != expected_kinds):
                raise RuntimeError("fresh8 tree violated its candidate contract")
            if old_fresh:
                group["metadata_json"].update({
                    "candidate_families": {
                        "stored_source": 1, "fresh_initial_noise": 8,
                        "fixed_initial_noise_new_pnp_perturbation": 0},
                    "candidate_count": 9,
                    "reused_v3_fresh_count": 4,
                    "reused_v3_group_id": _v3_group_id(item),
                })
            store.register_candidate_group(group, candidates)
            if old_fresh:
                _link_v3_fresh(store, group["candidate_group_id"], old_fresh)
            new_trees += 1
            new_branches += fresh_count
            torch.cuda.synchronize()
            peak_gib = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            elapsed = time.monotonic() - began
            print({
                "completed_this_run": new_trees,
                "pending_at_start": len(pending),
                "stock_success": bool(candidates[0]["success"]),
                "fresh_successes": sum(bool(c["success"])
                                       for c in [*candidates[1:], *old_fresh]),
                "mixed": len({bool(c["success"])
                              for c in [*candidates, *old_fresh]}) > 1,
                "new_branches": fresh_count,
                "tree_seconds": round(time.monotonic() - started, 1),
                "peak_gpu_gib": round(peak_gib, 2),
                "eta_minutes": round(elapsed / new_trees *
                                     (len(pending) - new_trees) / 60, 1),
            }, flush=True)
        status = "completed"
    finally:
        store.finish_run(status=status, n_rollouts=new_branches)
    return {"new_trees": new_trees, "complete_trees": len(items),
            "requested_trees": len(items)}
