"""Read-only provenance and difficulty survey for PCP-search collection control."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..libero_pro import describe_suite
from ..store import SupabaseStore
from .collection import EXPERIMENT
from .pro import pro_partition_summary


LEGACY_PRO_EXPERIMENT = "pro-16suite-k5-steps34-v1"


def _group(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "successes": 0, "steps": 0, "chunks": 0,
                 "trajectory_rows": 0, "pcp_rows": 0, "training_ready": 0})
    for row in rows:
        out = grouped[str(row.get("suite"))]
        out["n"] += 1
        out["successes"] += int(bool(row.get("success")))
        out["steps"] += int(row.get("n_steps") or 0)
        out["chunks"] += int(row.get("n_chunks") or 0)
        out["trajectory_rows"] += int(bool(row.get("trajectory_path")))
        out["pcp_rows"] += int(bool(row.get("pcp_chunks_path")))
        out["training_ready"] += int(bool(row.get("training_ready")))
    return [
        {"suite": suite, **describe_suite(suite), **values,
         "success_rate": values["successes"] / values["n"] if values["n"] else None,
         "mean_steps": values["steps"] / values["n"] if values["n"] else None,
         "mean_chunks": values["chunks"] / values["n"] if values["n"] else None}
        for suite, values in sorted(grouped.items())
    ]


def collect_initial_survey(store: SupabaseStore | None = None) -> dict[str, Any]:
    """Return the initial survey without mutating Supabase or creating a manifest."""
    store = store or SupabaseStore()
    fields = ("suite,method,status,success,n_steps,n_chunks,trajectory_path,pcp_chunks_path,"
              "training_ready,training_data_path")
    legacy = store.fetch_all(
        "rollouts", fields,
        configure=lambda query: query.eq("experiment", LEGACY_PRO_EXPERIMENT).eq(
            "method", "pnp_uncertainty_only").eq("status", "completed"),
        order_by=("suite",))
    current = store.fetch_all(
        "rollouts", fields + ",benchmark,pcp_partition_id,pcp_data_split,pcp_train_eligible",
        configure=lambda query: query.eq("experiment", EXPERIMENT).eq("status", "completed"),
        order_by=("benchmark", "suite"))
    return {
        "legacy_pro_experiment": LEGACY_PRO_EXPERIMENT,
        "legacy_stock_pnp_rows": len(legacy),
        "legacy_direct_training_eligible": 0,
        "legacy_exclusion_reason": (
            "historical 50-action execution and no complete PCP-search Bellman/RL-token artifact"),
        "legacy_suite_profile": _group(legacy),
        "current_pcp_rows": len(current),
        "current_pcp_training_ready": sum(bool(row.get("training_ready")) for row in current),
        "current_pcp_suite_profile": _group(current),
        "approved_pro_partition": pro_partition_summary(),
    }


def render_initial_survey(survey: dict[str, Any]) -> str:
    """Render a compact, notebook-friendly report without notebook-owned query logic."""
    lines = [
        "PCP-search initial collection survey",
        f"legacy stock P&P rows: {survey['legacy_stock_pnp_rows']} (training eligible: 0)",
        f"current PCP rows: {survey['current_pcp_rows']} "
        f"(training ready: {survey['current_pcp_training_ready']})",
        "",
        "Legacy PRO difficulty (stock P&P only):",
        "suite | family | n | success | mean chunks",
    ]
    for row in survey["legacy_suite_profile"]:
        lines.append(
            f"{row['suite']} | {row['suite_family']} | {row['n']} | "
            f"{row['success_rate']:.1%} | {row['mean_chunks']:.2f}")
    partition = survey["approved_pro_partition"]
    lines.extend(["", f"approved PRO partition: {partition['train_eligible_rollouts']} train, "
                  f"{partition['heldout_rollouts']} heldout"])
    return "\n".join(lines)
