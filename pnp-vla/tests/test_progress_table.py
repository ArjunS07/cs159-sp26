"""The in-flight progress table, which is how a collapsed suite gets noticed within minutes."""
from pnp import Method
from pnp.experiments import (
    HISTORICAL_PRO_BASELINE_SR,
    expanded_pro_suites,
    format_progress_table,
)

ARMS = [Method.UNCERTAINTY, Method.REFINEMENT]


def test_reports_running_rate_per_suite_and_arm():
    tally = {
        ("libero_goal_swap", Method.UNCERTAINTY): [25, 2],
        ("libero_goal_swap", Method.REFINEMENT): [25, 3],
        ("libero_spatial_with_milk", Method.UNCERTAINTY): [10, 8],
    }
    table = format_progress_table(tally, ARMS)
    assert "observed" in table and "refine" in table
    assert "8% (2/25)" in table          # rate, not just the last outcome
    assert "12% (3/25)" in table
    assert "80% (8/10)" in table
    # The historical baseline sits alongside so a collapse is obvious in place.
    assert "9%" in table and "83%" in table


def test_missing_arm_renders_as_a_placeholder_not_a_crash():
    tally = {("libero_goal_swap", Method.UNCERTAINTY): [4, 0]}
    table = format_progress_table(tally, ARMS)
    assert "0% (0/4)" in table
    assert "-" in table.splitlines()[1]


def test_unknown_suite_has_no_historical_reference():
    table = format_progress_table({("libero_made_up", Method.UNCERTAINTY): [2, 1]}, ARMS)
    assert "libero_made_up" in table
    assert "50% (1/2)" in table


def test_every_collected_suite_has_a_historical_reference():
    """Otherwise the column is blank exactly where the sanity check is needed."""
    for suite in expanded_pro_suites():
        assert suite in HISTORICAL_PRO_BASELINE_SR, suite


def test_suites_are_listed_in_a_stable_order():
    tally = {("libero_spatial_swap", Method.UNCERTAINTY): [1, 0],
             ("libero_goal_swap", Method.UNCERTAINTY): [1, 1]}
    rows = format_progress_table(tally, ARMS).splitlines()[1:]
    assert [row.split()[0] for row in rows] == ["libero_goal_swap", "libero_spatial_swap"]
