"""Canonical experimental conditions and versioned LIBERO cohort membership."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from pnp.config import Method
from pnp.experiments import FULL_ABLATION_TASKS

COHORT_MANIFEST_VERSION = "standard-libero-v1"
PAIR_KEYS = ["suite", "task_idx", "episode_idx", "init_state_hash"]
OBSERVED_LABEL = "observed/no-op (steps 1-9, K=3)"


def _steps(value: Any) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    return tuple(int(x) for x in value)


def condition_label(row: Mapping[str, Any]) -> str:
    """Human label for one behavior-defining config; never a grouping key itself."""
    method = row.get("method")
    if method == Method.UNCERTAINTY:
        steps = _steps(row.get("pnp_step_indices"))
        k = row.get("pnp_k")
        if steps and k is not None and not pd.isna(k):
            schedule = ",".join(map(str, steps))
            return f"observed/no-op (steps {schedule}, K={int(k)})"
        return OBSERVED_LABEL
    if method == Method.EXTRA_STEPS:
        return f"compute control ({int(row['num_inference_steps'])} steps)"
    if method == Method.REFINEMENT:
        schedule = ",".join(map(str, _steps(row.get("pnp_step_indices"))))
        variant = "mean" if row.get("refine_average") is True else "last"
        return f"refine-{variant} ({schedule})"
    return str(method)


def assign_standard_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Assign membership from the package-owned, versioned task manifest."""
    out = df.copy()
    out["full_ablation_member"] = [
        (str(s), int(t)) in FULL_ABLATION_TASKS
        for s, t in zip(out["suite"], out["task_idx"])
    ]
    out["cohort_manifest_version"] = COHORT_MANIFEST_VERSION
    out["condition_label"] = [condition_label(r) for r in out.to_dict("records")]
    return out


def schedule_family(row: Mapping[str, Any]) -> str | None:
    if row.get("method") != Method.REFINEMENT:
        return None
    steps = _steps(row.get("pnp_step_indices"))
    return "adjacent_two_step" if len(steps) == 2 and steps[1] == steps[0] + 1 else "periodic"


@dataclass(frozen=True)
class Condition:
    config_hash: str
    label: str
    method: str
    steps: tuple[int, ...]
    inference_steps: int | None


def condition_catalog(df: pd.DataFrame) -> list[Condition]:
    rows = []
    for config_hash, group in df.groupby("config_hash", sort=True):
        first = group.iloc[0]
        rows.append(Condition(
            str(config_hash), condition_label(first), str(first["method"]),
            _steps(first.get("pnp_step_indices")),
            int(first["num_inference_steps"]) if pd.notna(first.get("num_inference_steps")) else None,
        ))
    return rows
