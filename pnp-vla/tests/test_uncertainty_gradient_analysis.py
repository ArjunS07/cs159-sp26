import pandas as pd
import pytest

from analysis.uncertainty_gradient import (
    arm_success_table, first_u20_gate_sweep, gradient_telemetry_table,
    match_direct_gradient_cohort, paired_effect_table)
from pnp.config import Method


def _telemetry(delta):
    return {"uncertainty_gradient": {
        "records": [
            {"pre_u20": .03, "post_u20": .03 + delta,
             "delta_u20": delta, "update_rms": .01},
            {"pre_u20": .02, "post_u20": .02 + delta,
             "delta_u20": delta, "update_rms": .01}],
        "n_updates": 2}}


def _rows():
    outcomes = {
        Method.UNCERTAINTY: [False, False, True, True],
        Method.U20_GRADIENT: [True, False, False, True],
        Method.LATENT_RANDOM_CONTROL: [False, True, True, True],
    }
    rows = []
    for method, successes in outcomes.items():
        for index, success in enumerate(successes):
            rows.append({
                "rollout_id": f"{method}-{index}", "suite": f"suite_{index % 2}",
                "task_idx": index, "episode_idx": 10, "init_state_hash": f"h{index}",
                "method": method, "status": "completed", "success": success,
                "n_steps": 20, "u_mean_episode": .02,
                "ms_candidate_u": (
                    None if method == Method.UNCERTAINTY else
                    _telemetry(-.001 if method == Method.U20_GRADIENT else .0001)),
            })
    return pd.DataFrame(rows)


def test_match_and_summarize_three_paired_arms():
    paired, coverage = match_direct_gradient_cohort(
        _rows(), expected_identities=4, require_complete=True)
    assert len(paired) == 4
    assert coverage.matched_identities_used.eq(4).all()
    assert paired.first_pre_u20_abs_difference.eq(0).all()
    success = arm_success_table(paired)
    assert success.set_index("arm").loc["unrefined baseline", "success_rate_pct"] == 50
    effects = paired_effect_table(paired).set_index("comparison")
    gradient = effects.loc["gradient minus unrefined baseline"]
    assert gradient.condition_minus_baseline_pp == 0
    assert gradient.F_to_S == 1 and gradient.S_to_F == 1
    telemetry = gradient_telemetry_table(paired).set_index("arm")
    assert telemetry.loc["true U20 gradient", "mean_post_minus_pre_u20"] < 0
    assert telemetry.loc["random control", "mean_post_minus_pre_u20"] > 0


def test_preview_uses_only_complete_matches_and_gate_keeps_whole_denominator():
    rows = _rows()
    drop = ((rows.method == Method.LATENT_RANDOM_CONTROL) & (rows.task_idx == 3))
    paired, coverage = match_direct_gradient_cohort(rows[~drop], require_complete=False)
    assert len(paired) == 3
    assert coverage.matched_identities_used.eq(3).all()
    sweep = first_u20_gate_sweep(paired, grid_size=5)
    assert sweep.episodes_in_sr_denominator.eq(3).all()
    with pytest.raises(ValueError, match="expected 4 identical"):
        match_direct_gradient_cohort(
            rows[~drop], expected_identities=4, require_complete=True)
