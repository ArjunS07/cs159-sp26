import numpy as np
import pandas as pd

from pnp.verifier.data import CleanChunkExample
from pnp.verifier import diagnostics as dg


def _candidate(group, kind, success, prefix_value, *, stratum="high"):
    actions = np.full((50, 7), float(prefix_value), np.float32)
    return CleanChunkExample(
        rollout_id=f"{group}-{kind}", experiment="e", benchmark="libero",
        suite="suite", task_idx=0, episode_idx=0, chunk_idx=0, chunk_position=0.0,
        obs_enc=np.zeros(8, np.float32), actions=actions,
        action_mask=np.ones(50, bool), success=success,
        candidate_group_id=group, candidate_kind=kind, uncertainty_stratum=stratum,
        return_target=float(success))


def _group(group, outcomes, *, stratum="high"):
    """One default + fresh_noise candidates; ``outcomes`` are their successes."""
    examples = [_candidate(group, "default", outcomes[0], 0.0, stratum=stratum)]
    for i, success in enumerate(outcomes[1:], start=1):
        examples.append(_candidate(group, f"fresh_noise_{i}", success, i, stratum=stratum))
    return examples


# --------------------------------------------------------------------------- #
# Pure metric helpers
# --------------------------------------------------------------------------- #
def test_group_pair_accuracy_orderings():
    assert dg.group_pair_accuracy([1.0, 0.0], [1, 0]) == 1.0   # success scored higher
    assert dg.group_pair_accuracy([0.0, 1.0], [1, 0]) == 0.0   # success scored lower
    assert dg.group_pair_accuracy([1.0, 1.0], [1, 0]) == 0.5   # tie -> 0.5 credit
    assert np.isnan(dg.group_pair_accuracy([1.0, 0.0], [1, 1]))  # no discordant pair
    assert np.isnan(dg.group_pair_accuracy([np.nan, 0.0], [1, 0]))  # unscored group


def test_default_and_oracle_uplift():
    group = pd.DataFrame({
        "is_default": [True, False, False],
        "success": [0, 1, 0],
        "mode": [0, 0, 1],
    })
    assert dg.default_success(group) == 0
    assert dg.group_oracle_uplift(group) == 1.0  # oracle picks the success, default failed
    # majority mode is mode 0 (2 members), its success rate is 0.5
    assert dg.majority_mode_success(group) == 0.5


def test_within_mode_pair_fraction():
    # 1 success 1 failure, both in mode 0 -> all BT pairs are within-mode.
    same = pd.DataFrame({"success": [1, 0], "mode": [0, 0]})
    assert dg.within_mode_pair_fraction(same) == 1.0
    # success in mode 0, failure in mode 1 -> no within-mode pairs.
    split = pd.DataFrame({"success": [1, 0], "mode": [0, 1]})
    assert dg.within_mode_pair_fraction(split) == 0.0
    # no discordant pair -> nan.
    assert np.isnan(dg.within_mode_pair_fraction(pd.DataFrame({"success": [1, 1], "mode": [0, 1]})))


def test_mode_ranking_accuracy():
    # mode 1 has higher mean score and higher success rate -> concordant -> 1.0
    good = pd.DataFrame({"mode": [0, 0, 1, 1], "score": [0.0, 0.1, 0.9, 1.0],
                         "success": [0, 0, 1, 1]})
    assert dg.mode_ranking_accuracy(good) == 1.0
    # scores inverted relative to success -> discordant -> 0.0
    bad = pd.DataFrame({"mode": [0, 0, 1, 1], "score": [0.9, 1.0, 0.0, 0.1],
                        "success": [0, 0, 1, 1]})
    assert dg.mode_ranking_accuracy(bad) == 0.0
    # unscored -> nan
    assert np.isnan(dg.mode_ranking_accuracy(
        pd.DataFrame({"mode": [0, 1], "score": [np.nan, 0.0], "success": [0, 1]})))


def test_pairwise_spread_and_assign_modes_fallback():
    assert dg.pairwise_spread([np.zeros(3)]) == 0.0            # single item
    assert dg.pairwise_spread([np.zeros(3), np.ones(3)]) > 0.0  # distinct items
    # <= 2 items always collapses to a single mode, regardless of sklearn.
    labels = dg.assign_modes([np.zeros(3), np.ones(3)])
    assert list(labels) == [0, 0]
    labels3 = dg.assign_modes([np.zeros(3), np.ones(3), np.full(3, 2.0)])
    assert len(labels3) == 3 and labels3.dtype.kind in "iu"


# --------------------------------------------------------------------------- #
# Table builders (model-free path)
# --------------------------------------------------------------------------- #
def test_build_candidate_table_model_free():
    examples = _group("g0", [0, 1, 0]) + _group("g1", [1, 0, 1])
    cand = dg.build_candidate_table(examples, model=None, prefix_length=10)
    assert len(cand) == 6
    assert set(cand.columns) >= {"group_id", "score", "score_zc", "success", "prefix",
                                 "is_default", "stratum", "kind"}
    assert cand["score"].isna().all() and cand["score_zc"].isna().all()  # no model -> NaN
    assert cand[cand.kind == "default"].is_default.all()
    assert len(cand["prefix"].iloc[0]) == 10 * 7  # flattened masked prefix


def test_add_mode_structure_columns_and_terciles():
    # three groups with increasing internal spread -> the three terciles appear.
    examples = (
        [_candidate("low", "default", 0, 0.0), _candidate("low", "fresh_noise_1", 1, 0.0)]
        + [_candidate("mid", "default", 0, 0.0), _candidate("mid", "fresh_noise_1", 1, 1.0)]
        + [_candidate("high", "default", 0, 0.0), _candidate("high", "fresh_noise_1", 1, 5.0)]
    )
    cand = dg.build_candidate_table(examples, model=None)
    dg.add_mode_structure(cand)
    assert {"mode", "n_modes", "spread_tercile"} <= set(cand.columns)
    assert set(cand["spread_tercile"]) == {"low", "mid", "high"}
    assert (cand["n_modes"] >= 1).all()


def test_stratified_and_deployment_stats_are_model_aware():
    cand = dg.build_candidate_table(_group("g0", [0, 1, 0]) + _group("g1", [1, 0, 0]), model=None)
    dg.add_mode_structure(cand)
    table = dg.stratified(cand, "stratum", n_bootstrap=50)
    assert "control_gap" in table.columns
    assert np.isnan(table["ranking"]).all()      # unscored -> ranking undefined
    assert np.isnan(table["control_gap"]).all()

    stats = dg.deployment_stratum_stats(
        cand, stratum_col="stratum", stratum_value="high", n_bootstrap=50)
    assert set(stats) == {"ranking", "ci", "control_gap", "oracle_uplift", "discordant_n"}
    assert stats["discordant_n"] == 0           # no model scores -> no usable pairs
    assert np.isnan(stats["ranking"])
    assert stats["oracle_uplift"] >= 0.0        # oracle uplift is model-free and available
