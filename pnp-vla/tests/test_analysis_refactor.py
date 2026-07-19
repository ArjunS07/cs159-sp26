import math

import numpy as np
import pandas as pd
import pytest

from analysis.conditions import OBSERVED_LABEL, assign_standard_cohorts, condition_label
from analysis.statistics import (auc_metrics, discordant_test, holm_adjust,
                                 paired_counts, wilson_interval)
from analysis.validate import ValidationError, coverage_matrix, pair_one_to_one
from pnp.config import Method


def test_condition_labels_preserve_configuration():
    assert condition_label({"method": Method.UNCERTAINTY}) == OBSERVED_LABEL
    assert condition_label({"method": Method.EXTRA_STEPS, "num_inference_steps": 19}) == "compute control (19 steps)"
    assert condition_label({"method": Method.REFINEMENT, "pnp_step_indices": [2, 5, 8],
                            "refine_average": False}) == "refine-last (2,5,8)"


def test_cohort_assignment_uses_manifest_not_coverage():
    frame = pd.DataFrame({"suite": ["libero_goal", "libero_object"], "task_idx": [0, 0],
                          "method": [Method.UNCERTAINTY] * 2})
    out = assign_standard_cohorts(frame)
    assert out.full_ablation_member.tolist() == [True, False]


def test_unequal_method_coverage_stays_separate_by_config_hash():
    frame = pd.DataFrame({
        "config_hash": ["steps16"] * 4 + ["steps19"] * 2,
        "method": [Method.EXTRA_STEPS] * 6,
        "suite": ["s"] * 6, "task_idx": [0] * 6,
        "episode_idx": [0, 1, 2, 3, 0, 1], "init_state_hash": list("abcdef"),
        "num_inference_steps": [16] * 4 + [19] * 2,
        "full_ablation_member": [False] * 4 + [True] * 2,
    })
    matrix = coverage_matrix(frame)
    assert set(zip(matrix.config_hash, matrix.n_rollouts)) == {("steps16", 4), ("steps19", 2)}
    assert "method" not in matrix.columns or len(matrix) == 2


def test_pairing_is_strictly_one_to_one():
    base = pd.DataFrame({"suite": ["s"], "task_idx": [0], "episode_idx": [0],
                         "init_state_hash": ["h"], "success": [False]})
    condition = pd.concat([base.assign(success=True), base.assign(success=False)])
    with pytest.raises(pd.errors.MergeError):
        pair_one_to_one(base, condition)


def test_transitions_intervals_and_exact_test():
    counts = paired_counts([False, False, True, True], [True, False, False, True])
    assert counts == {"F_to_S": 1, "S_to_F": 1, "S_to_S": 1, "F_to_F": 1}
    assert discordant_test(1, 1) == 1.0
    lo, hi = wilson_interval(5, 10)
    assert lo < .5 < hi


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = np.array([.04, .001, .03, .2])
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= raw)


def test_auc_edge_cases_are_explicit_nulls():
    result = auc_metrics([0, 0, 0], [1, 2, 3])
    assert math.isnan(result["roc_auc"])
    assert result["n"] == 3
