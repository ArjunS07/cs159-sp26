"""Historical-data-driven task and state selection for PCP-search rollouts."""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable, Mapping

from .manifest import ManifestItem, RolloutManifest


INITIAL_ROLLOUT_BUDGET = 400
NEXT_TRANCHE_BUDGET = 200
BLOCK_SIZE = 5
PER_TASK_TRANCHE_CAP = 50
POLICY_REPO_ID = "lerobot/pi05_libero_finetuned"
POLICY_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
HISTORICAL_EXPERIMENT = "libero-u20-same-state-candidates-v1"
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ALL_TASKS = tuple((suite, task_idx) for suite in LIBERO_SUITES for task_idx in range(10))

TIER_A = {("libero_spatial", 5), ("libero_10", 8)}
TIER_B = ({("libero_10", 6)}
          | {("libero_goal", task_idx) for task_idx in (0, 2, 3, 4, 6)})
TIER_C = ({("libero_10", task_idx) for task_idx in (0, 4, 9)}
          | {("libero_goal", 9)}
          | {("libero_spatial", task_idx) for task_idx in (1, 8, 9)})
BOOSTED_LONG = {("libero_10", task_idx) for task_idx in (1, 2, 3, 7)}

# Interleaving the two untouched index bands prevents a tranche from being accidentally tied to
# one contiguous part of LIBERO's state array.
UNTOUCHED_STATE_ORDER = tuple(
    value for pair in zip(range(10, 20), range(40, 50)) for value in pair)

COLLECTION_CONFIG = {
    "n_action_steps": 10,
    "generated_chunk_size": 50,
    "pnp_steps": [3, 4],
    "pnp_k": 5,
    "refine": False,
    "num_inference_steps": 10,
    "save_training_data": True,
    "save_generated_chunks": True,
    "save_complete_pnp_trace": True,
}


def initial_task_allocation() -> dict[tuple[str, int], int]:
    """Return the approved, exactly-400 allocation over all 40 standard tasks."""
    allocation = {task: 5 for task in ALL_TASKS}
    allocation.update({task: 30 for task in TIER_A})
    allocation.update({task: 20 for task in TIER_B})
    allocation.update({task: 13 for task in TIER_C})
    allocation.update({task: 6 for task in BOOSTED_LONG})
    if len(allocation) != 40 or sum(allocation.values()) != INITIAL_ROLLOUT_BUDGET:
        raise AssertionError(
            f"invalid initial allocation: {len(allocation)} tasks, "
            f"{sum(allocation.values())} rollouts")
    return allocation


def _extract_u20(row: Mapping[str, Any]) -> float | None:
    for key in ("u20", "u20_mean", "u_first20", "mean_u20", "selection_u20"):
        value = row.get(key)
        if value is not None:
            try:
                value = float(value)
                return value if math.isfinite(value) else None
            except (TypeError, ValueError):
                pass
    telemetry = row.get("ms_candidate_u") or {}
    if isinstance(telemetry, Mapping):
        value = telemetry.get("u20_mean")
        if value is not None:
            try:
                value = float(value)
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
        profiles = telemetry.get("candidate_profiles") or []
        chosen = telemetry.get("chosen") or telemetry.get("chosen_idx") or []
        if profiles:
            if isinstance(chosen, list) and len(chosen) == len(profiles):
                values = []
                for chunk_profiles, index in zip(profiles, chosen):
                    try:
                        values.append(float(chunk_profiles[int(index)]["u20"]))
                    except (IndexError, KeyError, TypeError, ValueError):
                        continue
            else:
                values = [float(profile[0]["u20"]) for profile in profiles if profile]
            values = [value for value in values if math.isfinite(value)]
            if values:
                return sum(values) / len(values)
    return None


def _history_identity(row: Mapping[str, Any]) -> tuple[int, str]:
    index = row.get("episode_idx", row.get("ep_idx"))
    return int(index), str(row.get("init_state_hash") or "")


def _rank_replay_states(rows: Iterable[Mapping[str, Any]], task: tuple[str, int]) -> list[dict]:
    """Prior failures first, then high-U20 successes, one entry per physical state."""
    eligible = [row for row in rows
                if (row.get("suite"), int(row.get("task_idx", -1))) == task
                and row.get("status", "completed") == "completed"]
    by_state: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in eligible:
        key = _history_identity(row)
        previous = by_state.get(key)
        score = _extract_u20(row)
        previous_score = _extract_u20(previous) if previous else None
        if (previous is None
                or (score is not None and (previous_score is None or score > previous_score))):
            by_state[key] = row

    def rank(row: Mapping[str, Any]):
        failure = not bool(row.get("success"))
        uncertainty = _extract_u20(row)
        return (0 if failure else 1,
                -(uncertainty if uncertainty is not None else -math.inf),
                _history_identity(row))

    return [dict(row) for row in sorted(by_state.values(), key=rank)]


def _tier(task: tuple[str, int]) -> str:
    if task in TIER_A:
        return "A"
    if task in TIER_B:
        return "B"
    if task in TIER_C:
        return "C"
    if task in BOOSTED_LONG:
        return "long_boost"
    return "coverage"


def historical_task_profile(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], dict]:
    """Summarize the pinned task-selection cohort without treating U20 as cross-task truth."""
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        task = (str(row.get("suite")), int(row.get("task_idx", -1)))
        if task in ALL_TASKS and row.get("status", "completed") == "completed":
            grouped[task].append(row)
    profile = {}
    for task in ALL_TASKS:
        task_rows = grouped[task]
        failures = [row for row in task_rows if not bool(row.get("success"))]
        u20 = [value for value in (_extract_u20(row) for row in task_rows)
               if value is not None]
        profile[task] = {
            "n": len(task_rows),
            "successes": len(task_rows) - len(failures),
            "failures": len(failures),
            "failure_rate": len(failures) / len(task_rows) if task_rows else None,
            "mean_failure_chunks": (
                sum(float(row.get("n_chunks") or 0) for row in failures) / len(failures)
                if failures else None),
            "mean_u20": sum(u20) / len(u20) if u20 else None,
        }
    return profile


def build_initial_manifest(
        historical_rows: Iterable[Mapping[str, Any]], *,
        name: str = "pcp-search-initial-400",
        query_provenance: Mapping[str, Any] | None = None) -> RolloutManifest:
    """Build the approved initial manifest from historical task/outcome records.

    Every task first receives untouched physical states from indices 10--19 and 40--49.  Tier A
    then receives ten new behavior seeds on previously failed states, falling back to high-U20
    successful states.  Historical outcomes select states only; they never change the fixed task
    allocation.
    """
    rows = list(historical_rows)
    allocation = initial_task_allocation()
    profile = historical_task_profile(rows)
    items: list[ManifestItem] = []
    for task in ALL_TASKS:
        count = allocation[task]
        untouched_count = min(count, len(UNTOUCHED_STATE_ORDER))
        for state_index in UNTOUCHED_STATE_ORDER[:untouched_count]:
            items.append(ManifestItem(
                ordinal=len(items), suite=task[0], task_idx=task[1],
                init_state_index=state_index, behavior_seed_index=0,
                tier=_tier(task), selection_reason="untouched_state"))
        extra = count - untouched_count
        if extra:
            ranked = _rank_replay_states(rows, task)
            if len(ranked) < extra:
                raise ValueError(
                    f"{task} needs {extra} replay states but historical query supplied "
                    f"only {len(ranked)} unique completed states")
            for rank_index, source in enumerate(ranked[:extra], start=1):
                reason = ("prior_failure_new_behavior_seed" if not source.get("success")
                          else "high_u20_success_new_behavior_seed")
                items.append(ManifestItem(
                    ordinal=len(items), suite=task[0], task_idx=task[1],
                    init_state_index=int(source.get("episode_idx", source.get("ep_idx"))),
                    behavior_seed_index=rank_index,
                    tier=_tier(task), selection_reason=reason,
                    source_rollout_id=source.get("rollout_id")))
    provenance = {
        "historical_experiment": HISTORICAL_EXPERIMENT,
        "historical_rows": len(rows),
        "selection_policy": (
            "fixed task allocation; untouched 10-19/40-49; Tier-A extras failures then U20"),
        "expected_failures_from_historical_task_rates": sum(
            allocation[task] * float(profile[task]["failure_rate"] or 0.0)
            for task in ALL_TASKS),
        "historical_task_profile": {
            f"{suite}/{task_idx}": profile[(suite, task_idx)]
            for suite, task_idx in ALL_TASKS},
        **dict(query_provenance or {}),
    }
    return RolloutManifest(
        name=name, items=tuple(items), policy_repo_id=POLICY_REPO_ID,
        policy_revision=POLICY_REVISION, collection_config=dict(COLLECTION_CONFIG),
        provenance=provenance)


def _wilson_upper(failures: int, total: int, confidence: float = 0.90) -> float:
    if total <= 0:
        return 1.0
    # One-sided normal quantile; Wilson remains finite at 0/n and n/n.
    z = NormalDist().inv_cdf(confidence)
    p = failures / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return min(1.0, (centre + radius) / denominator)


@dataclass(frozen=True)
class TaskPriority:
    task: tuple[str, int]
    score: float
    failure_ucb90: float
    expected_failure_chunks: float
    ready_transition_coverage: int


def task_priorities(rows: Iterable[Mapping[str, Any]],
                    ready_transition_coverage: Mapping[tuple[str, int], int]) -> list[TaskPriority]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        task = (str(row.get("suite")), int(row.get("task_idx", -1)))
        if task in ALL_TASKS and row.get("status", "completed") == "completed":
            grouped[task].append(row)
    result = []
    for task in ALL_TASKS:
        task_rows = grouped[task]
        failures = [row for row in task_rows if not bool(row.get("success"))]
        expected_chunks = ([float(row.get("n_chunks") or 0) for row in failures]
                           or [float(row.get("n_chunks") or 0) for row in task_rows] or [1.0])
        expected_chunks = max(sum(expected_chunks) / len(expected_chunks), 1.0)
        coverage = max(int(ready_transition_coverage.get(task, 0)), 0)
        ucb = _wilson_upper(len(failures), len(task_rows))
        score = ucb * expected_chunks / math.sqrt(max(coverage, 1))
        result.append(TaskPriority(task, score, ucb, expected_chunks, coverage))
    return sorted(result, key=lambda item: (-item.score, item.task))


def build_next_tranche_manifest(
        rows: Iterable[Mapping[str, Any]],
        ready_transition_coverage: Mapping[tuple[str, int], int], *,
        parent_manifest_id: str,
        tranche_index: int,
        prior_rollout_counts: Mapping[tuple[str, int], int] | None = None) -> RolloutManifest:
    """Allocate one immutable 200-rollout adaptive tranche in five-rollout blocks."""
    if tranche_index < 1:
        raise ValueError("tranche_index must be positive")
    rows = list(rows)
    priorities = task_priorities(rows, ready_transition_coverage)
    prior = collections.Counter(prior_rollout_counts or {})
    additions = collections.Counter()
    blocks = NEXT_TRANCHE_BUDGET // BLOCK_SIZE
    # Weighted greedy allocation. Recompute the diminishing-return denominator after each block;
    # this prevents one high-score task from consuming the entire tranche while retaining the
    # approved scientific score and the 50-rollout cap.
    for _ in range(blocks):
        eligible = [priority for priority in priorities
                    if additions[priority.task] + BLOCK_SIZE <= PER_TASK_TRANCHE_CAP]
        if not eligible:
            raise ValueError("per-task caps cannot accommodate the requested tranche")
        chosen = max(
            eligible,
            key=lambda priority: (
                priority.score / math.sqrt(1 + additions[priority.task] / BLOCK_SIZE),
                -prior[priority.task], priority.task))
        additions[chosen.task] += BLOCK_SIZE

    items: list[ManifestItem] = []
    # Use behavior seeds on the approved untouched-state ring. The pair remains unique even once
    # a task receives more than twenty adaptive rollouts.
    for task in ALL_TASKS:
        for offset in range(additions[task]):
            state_index = UNTOUCHED_STATE_ORDER[offset % len(UNTOUCHED_STATE_ORDER)]
            seed_index = tranche_index * 100 + offset // len(UNTOUCHED_STATE_ORDER) + 1
            items.append(ManifestItem(
                ordinal=len(items), suite=task[0], task_idx=task[1],
                init_state_index=state_index, behavior_seed_index=seed_index,
                tier="adaptive", selection_reason="failure_ucb90_x_failure_chunks_over_coverage"))
    if len(items) != NEXT_TRANCHE_BUDGET:
        raise AssertionError(f"expected {NEXT_TRANCHE_BUDGET} items, got {len(items)}")
    return RolloutManifest(
        name=f"pcp-search-adaptive-{tranche_index:02d}", items=tuple(items),
        policy_repo_id=POLICY_REPO_ID, policy_revision=POLICY_REVISION,
        collection_config=dict(COLLECTION_CONFIG), parent_manifest_id=parent_manifest_id,
        provenance={
            "tranche_index": tranche_index,
            "block_size": BLOCK_SIZE,
            "per_task_cap": PER_TASK_TRANCHE_CAP,
            "priority_formula": "failure_ucb90 * expected_failure_chunks / sqrt(ready_coverage)",
            "allocations": {f"{suite}/{task_idx}": additions[(suite, task_idx)]
                            for suite, task_idx in ALL_TASKS},
        })


def fetch_historical_task_records(store, *, experiment: str = HISTORICAL_EXPERIMENT) -> list[dict]:
    """Package-owned Supabase query used by the manifest CLI and analysis notebooks."""
    return store.fetch_all(
        "rollouts",
        "rollout_id,suite,task_idx,episode_idx,init_state_hash,status,success,n_chunks,"
        "n_steps,u_mean_episode,ms_candidate_u,config_json,run_id",
        configure=lambda query: query.eq("experiment", experiment),
        order_by=("suite", "task_idx", "episode_idx", "rollout_id"),
    )
