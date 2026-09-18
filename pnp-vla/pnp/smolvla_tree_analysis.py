"""Analysis helpers for the SmolVLA depth-1 hybrid tree collection."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .smolvla_tree_collection import (
    SMOLVLA_TREE_CANDIDATES,
    SMOLVLA_TREE_COUNT,
    SMOLVLA_TREE_EXPERIMENT,
)
from .store import SupabaseStore


EXPECTED_KINDS = (
    "stored_source",
    "fresh_seed_1", "fresh_seed_2", "fresh_seed_3", "fresh_seed_4",
    "pnp_perturb_1", "pnp_perturb_2", "pnp_perturb_3", "pnp_perturb_4",
)


def _family(candidate: dict) -> str:
    metadata = candidate.get("metadata_json") or {}
    if metadata.get("candidate_family"):
        return str(metadata["candidate_family"])
    kind = str(candidate["candidate_kind"])
    if kind == "stored_source":
        return "stored_source"
    if kind.startswith("fresh_seed_"):
        return "fresh_initial_noise"
    if kind.startswith("pnp_perturb_"):
        return "fixed_initial_noise_new_pnp_perturbation"
    return "unknown"


def _candidate_rows(store, group_ids: list[str]) -> list[dict]:
    rows = []
    for start in range(0, len(group_ids), 100):
        batch = group_ids[start:start + 100]
        rows.extend(store.fetch_all(
            "verifier_candidates",
            "candidate_id,candidate_group_id,candidate_kind,success,n_steps,metadata_json",
            configure=lambda query, ids=batch: query.in_("candidate_group_id", ids),
            order_by=("candidate_group_id", "candidate_id")))
    return rows


def load_smolvla_tree_results(*, store=None,
                              experiment: str = SMOLVLA_TREE_EXPERIMENT) -> dict[str, pd.DataFrame]:
    """Load complete trees and return tree-, candidate-, and completion-level tables."""
    store = store or SupabaseStore()
    groups = store.fetch_all(
        "verifier_candidate_groups",
        "candidate_group_id,experiment,suite,task_idx,episode_idx,chunk_idx,"
        "uncertainty_stratum,metadata_json",
        configure=lambda query: query.eq("experiment", experiment),
        order_by=("candidate_group_id",))
    group_ids = [str(row["candidate_group_id"]) for row in groups]
    candidates = _candidate_rows(store, group_ids)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_group[str(candidate["candidate_group_id"])].append(candidate)

    tree_rows, candidate_rows = [], []
    expected = set(EXPECTED_KINDS)
    for group in groups:
        group_id = str(group["candidate_group_id"])
        rows = by_group.get(group_id, [])
        kinds = [str(row["candidate_kind"]) for row in rows]
        complete = len(rows) == SMOLVLA_TREE_CANDIDATES and set(kinds) == expected
        metadata = group.get("metadata_json") or {}
        base = {
            "candidate_group_id": group_id,
            "suite": str(group["suite"]),
            "task_idx": int(group["task_idx"]),
            "episode_idx": int(group["episode_idx"]),
            "chunk_idx": int(group["chunk_idx"]),
            "selection_strategy": str(metadata.get(
                "root_selection_strategy", group.get("uncertainty_stratum") or "unknown")),
            "source_success": bool(metadata.get("source_success", False)),
            "source_root_u10": float(metadata.get("source_root_u10", np.nan)),
            "complete": bool(complete), "branches": len(rows),
        }
        if not complete:
            tree_rows.append(base)
            continue

        for row in rows:
            candidate_metadata = row.get("metadata_json") or {}
            candidate_rows.append({
                **{key: base[key] for key in (
                    "candidate_group_id", "suite", "task_idx", "episode_idx",
                    "chunk_idx", "selection_strategy", "source_success", "source_root_u10")},
                "candidate_id": str(row["candidate_id"]),
                "candidate_kind": str(row["candidate_kind"]),
                "candidate_family": _family(row),
                "success": bool(row["success"]), "n_steps": int(row["n_steps"]),
                "pnp_u10": float(candidate_metadata.get("pnp_u10", np.nan)),
            })

        stock = next(row for row in rows if row["candidate_kind"] == "stored_source")
        fresh = [row for row in rows if str(row["candidate_kind"]).startswith("fresh_seed_")]
        perturb = [row for row in rows if str(row["candidate_kind"]).startswith("pnp_perturb_")]
        counterfactuals = fresh + perturb
        all_outcomes = np.asarray([bool(row["success"]) for row in rows])
        counter_outcomes = np.asarray([bool(row["success"]) for row in counterfactuals])
        fresh_outcomes = np.asarray([bool(row["success"]) for row in fresh])
        perturb_outcomes = np.asarray([bool(row["success"]) for row in perturb])
        stock_success = bool(stock["success"])
        successful_steps = [int(row["n_steps"]) for row in rows if row["success"]]
        tree_rows.append({
            **base,
            "stock_success": stock_success,
            "stock_n_steps": int(stock["n_steps"]),
            "mixed_outcomes": len(set(all_outcomes.tolist())) > 1,
            "counterfactual_mixed": len(set(counter_outcomes.tolist())) > 1,
            "fresh_mixed": len(set(fresh_outcomes.tolist())) > 1,
            "perturb_mixed": len(set(perturb_outcomes.tolist())) > 1,
            "branch_success_fraction": float(all_outcomes.mean()),
            "counterfactual_success_fraction": float(counter_outcomes.mean()),
            "successful_branches": int(all_outcomes.sum()),
            "successful_counterfactuals": int(counter_outcomes.sum()),
            "fresh_success_fraction": float(fresh_outcomes.mean()),
            "perturb_success_fraction": float(perturb_outcomes.mean()),
            "any_success": bool(all_outcomes.any()),
            "any_counterfactual_success": bool(counter_outcomes.any()),
            "fresh_any_success": bool(fresh_outcomes.any()),
            "perturb_any_success": bool(perturb_outcomes.any()),
            "fresh_recovers_failure": bool(not stock_success and fresh_outcomes.any()),
            "perturb_recovers_failure": bool(not stock_success and perturb_outcomes.any()),
            "either_recovers_failure": bool(not stock_success and counter_outcomes.any()),
            "fresh_has_failure_from_success": bool(stock_success and not fresh_outcomes.all()),
            "perturb_has_failure_from_success": bool(stock_success and not perturb_outcomes.all()),
            "best_success_n_steps": min(successful_steps) if successful_steps else np.nan,
        })

    trees = pd.DataFrame(tree_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    completion = pd.DataFrame([{
        "experiment": experiment,
        "manifest_target_trees": SMOLVLA_TREE_COUNT,
        "logged_group_rows": len(groups),
        "complete_trees": int(trees.complete.sum()) if len(trees) else 0,
        "partial_trees": int((~trees.complete).sum()) if len(trees) else 0,
        "missing_trees": SMOLVLA_TREE_COUNT - len(groups),
        "complete_candidate_rows": len(candidate_frame),
        "all_logged_candidate_rows": len(candidates),
    }])
    return {"trees": trees, "candidates": candidate_frame, "completion": completion}


def summarize_trees(trees: pd.DataFrame, by=()) -> pd.DataFrame:
    """Summarize only complete trees, optionally grouped by one or more columns."""
    frame = trees[trees.complete].copy()
    by = [by] if isinstance(by, str) else list(by)
    groups = [((), frame)] if not by else frame.groupby(by, observed=True, dropna=False)
    records = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        stock_failures = int((~group.stock_success).sum())
        record = {column: value for column, value in zip(by, key)}
        record.update({
            "trees": len(group),
            "branches": int(group.branches.sum()),
            "stock_sr_pct": 100 * group.stock_success.mean(),
            "counterfactual_branch_sr_pct": 100 * group.counterfactual_success_fraction.mean(),
            "mixed_trees_pct": 100 * group.mixed_outcomes.mean(),
            "counterfactual_mixed_pct": 100 * group.counterfactual_mixed.mean(),
            "any_success_pct": 100 * group.any_success.mean(),
            "oracle_gain_vs_stock_pp": 100 * (
                group.any_success.astype(int) - group.stock_success.astype(int)).mean(),
            "fresh_branch_sr_pct": 100 * group.fresh_success_fraction.mean(),
            "perturb_branch_sr_pct": 100 * group.perturb_success_fraction.mean(),
            "fresh_failure_recovery_pct": (
                100 * group.fresh_recovers_failure.sum() / stock_failures
                if stock_failures else np.nan),
            "perturb_failure_recovery_pct": (
                100 * group.perturb_recovers_failure.sum() / stock_failures
                if stock_failures else np.nan),
            "either_failure_recovery_pct": (
                100 * group.either_recovers_failure.sum() / stock_failures
                if stock_failures else np.nan),
            "stock_failures": stock_failures,
            "mean_root_u10": group.source_root_u10.mean(),
        })
        records.append(record)
    return pd.DataFrame(records)


def uncertainty_ranking_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Candidate-level failure AUC and within-tree low-U success ranking diagnostics."""
    frame = candidates[candidates.candidate_family.ne("stored_source")].dropna(
        subset=["pnp_u10"]).copy()
    records = []
    for family, group in [("all_counterfactuals", frame), *list(
            frame.groupby("candidate_family", observed=True))]:
        if not len(group):
            continue
        labels = (~group.success).to_numpy(bool)
        scores = group.pnp_u10.to_numpy(float)
        positive, negative = scores[labels], scores[~labels]
        if len(positive) and len(negative):
            auc = float((positive[:, None] > negative[None, :]).mean()
                        + 0.5 * (positive[:, None] == negative[None, :]).mean())
        else:
            auc = np.nan
        correct = ties = pairs = 0
        for _, tree in group.groupby("candidate_group_id"):
            success_u = tree.loc[tree.success, "pnp_u10"].to_numpy()
            failure_u = tree.loc[~tree.success, "pnp_u10"].to_numpy()
            if not len(success_u) or not len(failure_u):
                continue
            delta = failure_u[:, None] - success_u[None, :]
            correct += int((delta > 0).sum())
            ties += int((delta == 0).sum())
            pairs += int(delta.size)
        lowest = group.loc[group.groupby("candidate_group_id").pnp_u10.idxmin()]
        records.append({
            "family": family, "candidates": len(group),
            "trees": group.candidate_group_id.nunique(),
            "candidate_failure_auc_high_u": auc,
            "within_tree_pairwise_low_u_success_accuracy": (
                (correct + 0.5 * ties) / pairs if pairs else np.nan),
            "decidable_success_failure_pairs": pairs,
            "lowest_u_candidate_sr_pct": 100 * lowest.success.mean(),
            "mean_candidate_sr_pct": 100 * group.success.mean(),
            "lowest_u_minus_mean_pp": 100 * (lowest.success.mean() - group.success.mean()),
        })
    return pd.DataFrame(records)
