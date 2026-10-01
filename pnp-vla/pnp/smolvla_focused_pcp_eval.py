"""Predeclared real-PCP evaluation enriched for historically mixed validation roots."""
from __future__ import annotations

from collections import Counter

from . import smolvla_tree_collection as tree
from .smolvla_action_contrast_pilot import _pilot_items
from .smolvla_flow_pcp_pilot import generate_flow_controls
from .smolvla_pcp_pilot import KINDS, run_pcp_pilot
from .smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS
from .smolvla_tree_bellman_finetune import _split_groups

EXPERIMENT = "smolvla-libero-focused-validation-flow-pcp-v1"
EXPECTED_CHECKPOINT_SHA256 = "9b8800bc9f27116afcb1812f93f8a9f6767ac012c7cdac790502032fbb7d0c8e"


def _key(row):
    return (row["suite"], int(row["task_idx"]), int(row["episode_idx"]), int(row["chunk_idx"]))


def focused_manifest(snapshot, source_items, excluded_source_ids, *, target_roots=48,
                     shard_count=3, checkpoint_sha=EXPECTED_CHECKPOINT_SHA256, source_manifest_hash=None):
    """Pure selection from frozen historical outcomes and original cohort split.

    No current critic scores, PCP results, or new observations enter selection.
    Shortages are reported; training roots and duplicate identities are refused.
    """
    if (target_roots < 1 or shard_count < 1 or not isinstance(checkpoint_sha, str)
            or len(checkpoint_sha) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha)):
        raise ValueError("invalid target, shard count, or checkpoint SHA")
    cohort = [g for g in snapshot["groups"] if g["source_experiment"] == FRESH8_EXPERIMENT]
    if len({_key(g) for g in cohort}) != len(cohort):
        raise ValueError("duplicate old-cohort root identities")
    _, validation = _split_groups(cohort)
    validation = set(validation)
    sources = {_key(item): item for item in source_items}
    if len(sources) != len(source_items):
        raise ValueError("duplicate immutable source root identities")
    excluded = set(excluded_source_ids)
    eligible = []
    mixed_count = 0
    for group in cohort:
        if group["candidate_group_id"] not in validation:
            continue
        candidates = group["candidates"]
        if (len(candidates) != 9 or {c["candidate_kind"] for c in candidates} != set(FRESH8_KINDS)
                or candidates[0]["candidate_kind"] != "stored_source"):
            raise ValueError("historical validation candidate contract differs")
        outcomes = [bool(c["success"]) for c in candidates]
        if len(set(outcomes)) != 2:
            continue
        mixed_count += 1
        item = sources.get(_key(group))
        if item is None:
            raise ValueError("old validation root absent from immutable idx10-29 source manifest")
        if item["source_rollout_id"] in excluded:
            continue
        if not 10 <= int(item["episode_idx"]) <= 29:
            raise ValueError("selected old-cohort source is outside episode indices 10-29")
        eligible.append((item, group, outcomes[0]))
    selected = []
    # Stable hash order within each stock-outcome stratum. Alternate strata,
    # then fill from the available one when the other has no roots remaining.
    buckets = {status: sorted([r for r in eligible if r[2] == status],
        key=lambda r: tree._rank("focused-validation-pcp-v1", r[1]["candidate_group_id"]))
        for status in (False, True)}
    while len(selected) < min(target_roots, len(eligible)):
        for status in (False, True):
            if buckets[status] and len(selected) < target_roots:
                selected.append(buckets[status].pop(0))
    items = [{**item, "ordinal": n, "original_group_id": group["candidate_group_id"],
              "original_split": "validation", "historically_mixed": True}
             for n, (item, group, _) in enumerate(selected)]
    if len({i["source_rollout_id"] for i in items}) != len(items):
        raise ValueError("selected sources are not distinct")
    protocol = {"experiment": EXPERIMENT, "version": 1,
        "checkpoint_sha256": checkpoint_sha, "snapshot_digest": snapshot["snapshot_digest"],
        "source_manifest_hash": source_manifest_hash,
        "source_experiment": FRESH8_EXPERIMENT, "episode_indices": [10, 29],
        "selection": "historically_mixed_original_validation_excluding_18_prior_pilot_sources",
        "target_roots": target_roots, "shard_count": shard_count,
        "correction_step": 3, "correction_s": .7, "probe_k": 1,
        "correction_mode": "rms", "clean_rms": [.02, .06],
        "controls": sorted(KINDS), "continuation_seeds_per_root": 1,
        "excluded_source_rollout_ids": sorted(excluded),
        "root_only_intervention": True, "independent_test": False}
    audit = {"old_cohort_roots": len(cohort), "old_validation_roots": len(validation),
             "historically_mixed_validation_roots": mixed_count,
             "eligible_after_prior_pilot_exclusion": len(eligible),
             "selected_roots": len(items), "target_shortfall": max(0, target_roots - len(items)),
             "eligible_historical_stock_outcomes": dict(Counter("success" if r[2] else "failure" for r in eligible)),
             "selected_historical_stock_outcomes": dict(Counter("success" if r[2] else "failure" for r in selected)),
             "selected_by_suite": dict(Counter(i["suite"] for i in items)),
             "roots_by_shard": {str(s): sum(i["ordinal"] % shard_count == s for i in items) for s in range(shard_count)},
             "enrichment": "selected on recorded mixed outcomes; reused validation; exploratory conditional comparison"}
    payload = {"protocol": protocol, "audit": audit, "items": items}
    return {**payload, "manifest_hash": tree._digest(payload)}


def run_focused_flow_pcp_eval(*, checkpoint_path, expected_checkpoint_sha=EXPECTED_CHECKPOINT_SHA256,
                              target_roots=48, shard_count=3, shard_index=0, root_limit=None,
                              run_intervention=False, **kwargs):
    """CPU manifest/gradient benchmark or opt-in CUDA intervention; no implicit launch."""
    if (not isinstance(expected_checkpoint_sha, str) or len(expected_checkpoint_sha) != 64
            or any(c not in "0123456789abcdef" for c in expected_checkpoint_sha)):
        raise ValueError("an exact lowercase checkpoint SHA256 is required")
    def select(snapshot, store, metadata):
        if metadata["architecture"].get("model_family") != "late_fusion_scalar":
            raise ValueError("focused evaluation requires the predeclared late-fusion scalar checkpoint")
        source = tree.load_or_build_smolvla_tree_manifest(store)
        prior, _ = _pilot_items(store)
        return focused_manifest(snapshot, source["payload"]["items"],
            [i["source_rollout_id"] for i in prior], target_roots=target_roots,
            shard_count=shard_count, checkpoint_sha=metadata["critic_sha256"],
            source_manifest_hash=source["manifest_hash"])
    return run_pcp_pilot(checkpoint_path=checkpoint_path, shard_index=shard_index,
        root_limit=root_limit, run_intervention=run_intervention, **kwargs,
        _proposal_generator=generate_flow_controls, _cohort_provider=select,
        _shard_count=shard_count, _expected_checkpoint_sha=expected_checkpoint_sha,
        _experiment_base=EXPERIMENT, _method_config={
            "method": "flow_step_clean_estimate_q_correction", "correction_step": 3,
            "correction_noise_level": .7, "correction_probe_k": 1, "correction_mode": "rms",
            "scales_rms": [.02, .06], "pnp_steps": [1, 2, 3], "pnp_k_by_step": [3, 1, 1],
            "integration_steps": 10, "root_only_intervention": True,
            "selection_protocol": "historical_mixed_original_old_cohort_validation",
            "target_roots": target_roots, "protocol_shard_count": shard_count,
            "critic_domain": "terminal_action_Q_extrapolated_to_flow_clean_estimate"})
