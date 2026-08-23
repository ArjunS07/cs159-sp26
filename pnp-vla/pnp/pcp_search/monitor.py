"""Read-only manifest health report for the thin Colab monitor notebook."""
from __future__ import annotations

from collections import Counter

from ..store import SupabaseStore


def collect_manifest_monitor(manifest_id: str, store: SupabaseStore | None = None) -> dict:
    store = store or SupabaseStore()
    results = store.fetch_all(
        "pcp_search_manifest_results", "ordinal,rollout_id,status,reason,validation_json",
        configure=lambda query: query.eq("manifest_id", manifest_id), order_by=("ordinal",))
    ids = [row["rollout_id"] for row in results if row.get("rollout_id")]
    rollouts = []
    for start in range(0, len(ids), 100):
        rollouts.extend(store.fetch_all(
            "rollouts", "rollout_id,success,n_chunks,training_ready,pcp_data_split,"
            "pcp_train_eligible,suite",
            configure=lambda query, batch=ids[start:start + 100]: query.in_("rollout_id", batch),
            order_by=("rollout_id",)))
    by_id = {row["rollout_id"]: row for row in rollouts}
    ready = [row for row in rollouts if row.get("training_ready")]
    return {
        "manifest_id": manifest_id,
        "result_statuses": dict(Counter(row["status"] for row in results)),
        "linked_rollouts": len(rollouts),
        "training_ready": len(ready),
        "successes": sum(bool(row.get("success")) for row in ready),
        "failures": sum(not bool(row.get("success")) for row in ready),
        "bellman_transitions": sum(int(row.get("n_chunks") or 0) for row in ready),
        "by_split": dict(Counter(str(row.get("pcp_data_split") or "legacy") for row in ready)),
        "train_eligible": sum(bool(row.get("pcp_train_eligible")) for row in ready),
        "unlinked_result_ordinals": [row["ordinal"] for row in results
                                    if row.get("rollout_id") not in by_id],
    }
