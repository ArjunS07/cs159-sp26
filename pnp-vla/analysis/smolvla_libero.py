"""Matched SmolVLA LIBERO outcome and uncertainty-flip analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.suffix_sensitivity import bootstrap_rank_auc
from pnp.diversity import DIVERSITY_PAIR_KEYS


TRANSITION_ORDER = ("F->F", "F->S", "S->F", "S->S")


def pair_stock_refinement(stock: pd.DataFrame, refinement: pd.DataFrame) -> pd.DataFrame:
    """Pair two completed arms on exact identities and label outcome transitions."""
    required = set(DIVERSITY_PAIR_KEYS + ["success"])
    for label, frame in (("stock", stock), ("refinement", refinement)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} rows are missing columns: {sorted(missing)}")
        if frame.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"duplicate exact identities in {label} arm")

    left = stock[DIVERSITY_PAIR_KEYS + ["success"]].rename(
        columns={"success": "baseline_success"})
    right = refinement[DIVERSITY_PAIR_KEYS + ["success"]].rename(
        columns={"success": "condition_success"})
    paired = left.merge(right, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(
            f"arms are not exactly matched: stock={len(left)}, refinement={len(right)}, "
            f"intersection={len(paired)}")
    paired["baseline_success"] = paired.baseline_success.astype(bool)
    paired["condition_success"] = paired.condition_success.astype(bool)
    paired["transition"] = np.select(
        [
            ~paired.baseline_success & ~paired.condition_success,
            ~paired.baseline_success & paired.condition_success,
            paired.baseline_success & ~paired.condition_success,
            paired.baseline_success & paired.condition_success,
        ],
        TRANSITION_ORDER,
        default="invalid",
    )
    paired["transition"] = pd.Categorical(
        paired.transition, categories=TRANSITION_ORDER, ordered=True)
    paired["any_flip"] = paired.baseline_success.ne(paired.condition_success)
    return paired.sort_values(DIVERSITY_PAIR_KEYS).reset_index(drop=True)


def flip_auc_table(paired: pd.DataFrame, score_columns, *, n_boot: int = 3000
                   ) -> pd.DataFrame:
    """AUCs for distinct flip questions without mixing baseline successes and failures.

    Larger scores are always treated as predicting the named positive event.
    """
    cohorts = (
        ("all", paired, "any outcome flip", paired.any_flip),
        ("stock failures", paired[~paired.baseline_success], "F->S rescue",
         paired.loc[~paired.baseline_success, "condition_success"]),
        ("stock successes", paired[paired.baseline_success], "S->F harm",
         ~paired.loc[paired.baseline_success, "condition_success"]),
        ("discordant only", paired[paired.any_flip], "F->S rather than S->F",
         paired.loc[paired.any_flip, "condition_success"]),
    )
    rows = []
    for cohort, frame, event, labels in cohorts:
        labels = labels.to_numpy(bool)
        for score in score_columns:
            auc, low, high = bootstrap_rank_auc(
                labels, frame[score].to_numpy(float), n_boot=n_boot)
            rows.append({
                "cohort": cohort,
                "positive_event": event,
                "score_name": score,
                "episodes": len(frame),
                "positives": int(labels.sum()),
                "auc": auc,
                "auc_ci_low": low,
                "auc_ci_high": high,
            })
    return pd.DataFrame(rows)

def uncertainty_quantile_flip_table(paired: pd.DataFrame, *, score_column: str,
                                    bins: int = 4) -> pd.DataFrame:
    """Outcome transitions and paired SR change in equal-count uncertainty bins."""
    frame = paired.copy()
    frame["uncertainty_quantile"] = pd.qcut(
        frame[score_column].rank(method="first"), bins,
        labels=np.arange(1, bins + 1)).astype(int)
    rows = []
    for quantile, group in frame.groupby("uncertainty_quantile", sort=True):
        counts = group.transition.value_counts().reindex(TRANSITION_ORDER, fill_value=0)
        rows.append({
            "score_name": score_column,
            "uncertainty_quantile": int(quantile),
            "episodes": len(group),
            "score_mean": float(group[score_column].mean()),
            "stock_sr_pct": 100 * float(group.baseline_success.mean()),
            "refinement_sr_pct": 100 * float(group.condition_success.mean()),
            "refinement_minus_stock_pp": 100 * float(
                group.condition_success.mean() - group.baseline_success.mean()),
            "flip_rate_pct": 100 * float(group.any_flip.mean()),
            "F_to_F": int(counts["F->F"]),
            "F_to_S": int(counts["F->S"]),
            "S_to_F": int(counts["S->F"]),
            "S_to_S": int(counts["S->S"]),
        })
    return pd.DataFrame(rows)
