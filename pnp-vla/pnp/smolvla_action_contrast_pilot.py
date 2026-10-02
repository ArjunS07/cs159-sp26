"""Small, resume-safe Q10 action-contrast pilot on existing SmolVLA source roots.

This tests whether controlled, smooth action deviations yield informative paired
outcomes before spending another large collection budget on near-identical
fresh-noise chunks. It is research data, not an untouched evaluation cohort.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import smolvla_tree_collection as tree
from .smolvla_tree_source_experiment import prepare_smolvla_tree_source_episodes
from .store import SupabaseStore
from .verifier.collection import candidate_group_id


EXPERIMENT = "smolvla-libero-q10-action-contrast-pilot-v1"
ROOT_COUNT = 18
SHARD_COUNT = 3
SCALES = (0.02, 0.06)
AXES = (0, 1, 2)  # Cartesian translation in the normalized policy action.
KINDS = {"stored_source", "fresh_seed_1", *(
    f"translation_{axis}_{scale_label}_{sign_label}"
    for axis in AXES for scale_label in ("small", "large")
    for sign_label in ("plus", "minus"))}


def _pilot_items(store) -> tuple[list[dict], str]:
    source = tree.load_or_build_smolvla_tree_manifest(store)
    items = source["payload"]["items"]
    # Select with source labels and stable IDs only. Existing candidate outcomes
    # are never used for this choice. Spread the small pilot over all four suites.
    chosen = []
    for suite in sorted({item["suite"] for item in items}):
        for outcome in (False, True):
            bucket = [item for item in items if item["suite"] == suite
                      and bool(item["source_success"]) == outcome]
            bucket.sort(key=lambda item: tree._rank(
                "action-contrast-pilot-v1", item["source_rollout_id"]))
            chosen.extend(bucket[:2])
    remaining = [item for item in items if item not in chosen]
    remaining.sort(key=lambda item: tree._rank(
        "action-contrast-pilot-v1-extra", item["source_rollout_id"]))
    chosen.extend(remaining[:ROOT_COUNT - len(chosen)])
    if len(chosen) != ROOT_COUNT or len({i["source_rollout_id"] for i in chosen}) != ROOT_COUNT:
        raise ValueError("could not select 18 distinct pilot roots")
    chosen.sort(key=lambda item: tree._rank(
        "action-contrast-pilot-v1-order", item["source_rollout_id"]))
    payload = {"version": 1, "experiment": EXPERIMENT,
               "source_manifest_hash": source["manifest_hash"],
               "scales": SCALES, "axes": AXES,
               "items": [{**item, "ordinal": n} for n, item in enumerate(chosen)]}
    return payload["items"], tree._digest(payload)


def _group_id(item: dict) -> str:
    return candidate_group_id(
        "libero", item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=EXPERIMENT,
        trajectory_seed=item["source_episode_seed"])


def _controlled_chunks(source_chunk: np.ndarray) -> dict[str, tuple[np.ndarray, dict]]:
    stock = np.asarray(source_chunk, np.float32)
    if stock.ndim != 2 or stock.shape[0] < 10 or stock.shape[1] < 7:
        raise ValueError(f"invalid source policy chunk shape {stock.shape}")
    time_profile = np.sin(np.pi * (np.arange(10, dtype=np.float32) + .5) / 10)
    extras = {}
    for axis in AXES:
        for scale_label, magnitude in (("small", SCALES[0]), ("large", SCALES[1])):
            for sign_label, sign in (("plus", 1), ("minus", -1)):
                candidate = stock.copy()
                candidate[:10, axis] = np.clip(
                    stock[:10, axis] + sign * magnitude * time_profile, -1, 1)
                name = f"translation_{axis}_{scale_label}_{sign_label}"
                extras[name] = (candidate, {
                    "intervention": "smooth_normalized_translation_offset",
                    "action_axis": axis, "requested_scale": sign * magnitude,
                    "realized_max_abs_delta": float(np.max(np.abs(
                        candidate[:10, axis] - stock[:10, axis]))),
                })
    return extras


def _completed(store, items: list[dict], manifest_hash: str) -> set[str]:
    allowed = {_group_id(item) for item in items}
    rows = store.fetch_all(
        "verifier_candidate_groups", "candidate_group_id,manifest_hash",
        configure=lambda q: q.eq("experiment", EXPERIMENT),
        order_by=("candidate_group_id",))
    groups = {r["candidate_group_id"] for r in rows
              if r["candidate_group_id"] in allowed
              and r["manifest_hash"] == manifest_hash}
    complete = set()
    for group_id in groups:
        candidates = store.fetch_all(
            "verifier_candidates", "candidate_kind,metadata_json",
            configure=lambda q, gid=group_id: q.eq("candidate_group_id", gid),
            order_by=("candidate_kind",))
        if (len(candidates) == len(KINDS)
                and {r["candidate_kind"] for r in candidates} == KINDS
                and all((r.get("metadata_json") or {}).get("training_data_path")
                        for r in candidates)):
            complete.add(group_id)
    return complete


def run_action_contrast_pilot(*, shard_index: int, tree_limit: int | None = 1,
                              store=None) -> dict:
    """Collect one of three fixed shards; completed groups are skipped on rerun."""
    from . import libero_env, models

    if shard_index not in range(SHARD_COUNT):
        raise ValueError("shard_index must be 0, 1, or 2")
    if tree_limit is not None and tree_limit < 1:
        raise ValueError("tree_limit must be positive or None")
    store = store or SupabaseStore()
    all_items, manifest_hash = _pilot_items(store)
    items = [item for item in all_items if item["ordinal"] % SHARD_COUNT == shard_index]
    if tree_limit is not None:
        items = items[:tree_limit]
    complete = _completed(store, items, manifest_hash)
    pending = [item for item in items if _group_id(item) not in complete]
    print({"experiment": EXPERIMENT, "manifest_hash": manifest_hash,
           "shard": f"{shard_index}/{SHARD_COUNT}",
           "requested_roots": len(items), "complete_roots": len(complete),
           "pending_roots": len(pending), "candidates_per_root": len(KINDS),
           "scales": SCALES}, flush=True)
    if not pending:
        return {"new_trees": 0, "complete_trees": len(complete)}
    if not torch.cuda.is_available():
        raise RuntimeError("select a CUDA Colab runtime")
    egl = tree.subprocess.run(
        "ldconfig -p | grep -q libEGL_nvidia", shell=True,
        stdout=tree.subprocess.DEVNULL, stderr=tree.subprocess.DEVNULL)
    if egl.returncode:
        raise RuntimeError("NVIDIA EGL is unavailable")
    rows = {r["rollout_id"]: r for r in tree._source_rows(store)}
    episodes = prepare_smolvla_tree_source_episodes()
    lookup = {(ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"])): ep
              for ep in episodes}
    policy, preprocess, postprocess = models.load_smolvla()
    store.start_run(
        "smolvla_action_contrast_pilot", "libero", EXPERIMENT,
        config={"source_manifest_hash": manifest_hash,
                "shard_index": shard_index, "shard_count": SHARD_COUNT,
                "candidate_count": len(KINDS), "scales": SCALES,
                "axes": AXES, "n_action_steps": 10})
    status, count = "failed", 0
    began = time.monotonic()
    try:
        for item in pending:
            row = rows[item["source_rollout_id"]]
            bundle = tree._load_source_bundle(store, row)
            source = tree._source_boundary(bundle, item)
            ep = lookup[(item["suite"], int(item["task_idx"]), int(item["episode_idx"]))]
            env = libero_env.make_env(ep["bddl_path"])
            try:
                group, candidates = tree.collect_smolvla_depth1_tree(
                    env, ep, policy, preprocess, postprocess, models.default_device(),
                    item=item, bundle=bundle, manifest_hash=manifest_hash,
                    experiment=EXPERIMENT, fresh_count=1, perturb_count=0,
                    extra_policy_chunks=_controlled_chunks(source["policy_chunk"]))
            finally:
                env.close()
            if group["candidate_group_id"] != _group_id(item) or {
                    c["candidate_kind"] for c in candidates} != KINDS:
                raise RuntimeError("pilot candidate contract mismatch")
            store.register_candidate_group(group, candidates)
            count += 1
            print({"completed_this_run": count, "pending_at_start": len(pending),
                   "elapsed_minutes": round((time.monotonic() - began) / 60, 1)},
                  flush=True)
        status = "completed"
    finally:
        store.finish_run(status=status, n_rollouts=(len(KINDS) - 1) * count)
    return {"new_trees": count, "complete_trees": len(complete) + count}
