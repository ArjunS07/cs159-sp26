import numpy as np
import pandas as pd

from pnp.verifier.data import CleanChunkExample
from pnp.verifier import diagnostics as dg


def _candidate(group, kind, success, prefix_value, *, stratum="high", n_steps=None):
    actions = np.full((50, 7), float(prefix_value), np.float32)
    return CleanChunkExample(
        rollout_id=f"{group}-{kind}", experiment="e", benchmark="libero",
        suite="suite", task_idx=0, episode_idx=0, chunk_idx=0, chunk_position=0.0,
        obs_enc=np.zeros(8, np.float32), actions=actions,
        action_mask=np.ones(50, bool), success=success,
        candidate_group_id=group, candidate_kind=kind, uncertainty_stratum=stratum,
        return_target=float(success), n_steps=n_steps)


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


def test_pivot_euler_rows_averages_per_noise_level():
    rows = [
        {"rollout_id": "r0", "chunk_idx": 0, "euler_step": 1, "s": 0.9, "u_mean": 0.02},
        {"rollout_id": "r0", "chunk_idx": 0, "euler_step": 5, "s": 0.5, "u_mean": 0.04},
        {"rollout_id": "r0", "chunk_idx": 0, "euler_step": 5, "s": 0.5, "u_mean": 0.06},  # avg 0.05
        {"rollout_id": "r0", "chunk_idx": 1, "euler_step": 1, "s": 0.9, "u_mean": 0.10},
    ]
    pivot = dg._pivot_euler_rows(rows)
    assert pivot[("r0", 0)] == {"u_s90": 0.02, "u_s50": 0.05}
    assert pivot[("r0", 1)] == {"u_s90": 0.10}
    assert dg._u_step_column(0.5) == "u_s50" and dg._u_step_column(0.1) == "u_s10"


def test_prefix_distance_disagreement_rises_and_flattens():
    # prefixes at value 0 (both success) and value 5 (both failure): the only disagreeing
    # pairs are the far (0 vs 5) pairs -> P(disagree) rises with distance.
    rising = ([_candidate("g", "default", 1, 0.0), _candidate("g", "fresh_noise_1", 1, 0.0),
               _candidate("g", "fresh_noise_2", 0, 5.0), _candidate("g", "fresh_noise_3", 0, 5.0)])
    cand = dg.build_candidate_table(rising, model=None)
    table, slope = dg.prefix_distance_disagreement(cand)
    assert list(table.columns) == ["distance_bin", "dist_mid", "p_disagree", "n"]
    assert slope > 0
    # all-success group -> no disagreements anywhere -> flat (zero-slope) curve.
    flat = dg.build_candidate_table(_group("h", [1, 1, 1, 1]), model=None)
    _, flat_slope = dg.prefix_distance_disagreement(flat)
    assert flat_slope == 0.0


def test_handcrafted_features_shape_and_values():
    cand = dg.build_candidate_table(_group("g", [0, 1, 0]), model=None)  # prefixes 0, 1, 2
    feats = dg.handcrafted_features(cand)
    assert list(feats.columns) == list(dg._HANDCRAFTED_FEATURE_NAMES)
    assert feats.shape == (3, 9)
    default_row = feats.loc[cand.index[cand["is_default"]][0]]
    # constant prefix -> zero jerk/smoothness/std; default is 0 distance from itself.
    assert default_row["jerk"] == 0.0 and default_row["smoothness"] == 0.0
    assert default_row["grip_mean"] == 0.0 and default_row["dist_default"] == 0.0
    assert default_row["chunk_position"] == 0.0
    assert np.isclose(default_row["dist_centroid"], np.sqrt(70))  # ||0 - 1|| over 70 dims
    fn2_row = feats.loc[cand.index[cand["kind"] == "fresh_noise_2"][0]]
    assert fn2_row["grip_mean"] == 2.0


def test_fit_gbt_baseline_returns_comparable_ranking():
    cand = dg.build_candidate_table(
        sum((_group(f"g{i}", [0, 1, 0]) for i in range(6)), []), model=None)
    dg.add_mode_structure(cand)
    result = dg.fit_gbt_baseline(cand, n_splits=3)
    assert set(result) >= {"ranking", "n_groups", "importances"}
    if dg._HAVE_SKLEARN:  # sklearn is the [analysis] extra; absent -> graceful NaN result.
        assert np.isnan(result["ranking"]) or 0.0 <= result["ranking"] <= 1.0
        assert len(result["importances"]) == 9
    else:
        assert result["importances"] == {}


def test_selector_zoo_pick_best_and_reject_worst():
    # successes default=0, fresh_noise_1=1, fresh_noise_2=0; inject a controlled score column.
    cand = dg.build_candidate_table(_group("g", [0, 1, 0]), model=None)
    dg.add_mode_structure(cand)
    cand["score"] = cand["kind"].map(
        {"default": 0.1, "fresh_noise_1": 0.9, "fresh_noise_2": 0.5})
    zoo = dg.selector_zoo(cand, score_cols=("score",), quantile=0.25)
    pick = zoo[zoo.selector == "pick_best"].iloc[0]
    assert pick["deployed_success"] == 1.0 and pick["uplift_vs_default"] == 1.0  # picks the success
    # quantile(0.25)=0.3 -> survivors are scores 0.9 (success) and 0.5 (failure) -> mean 0.5.
    reject = zoo[zoo.selector == "reject_worst"].iloc[0]
    assert reject["deployed_success"] == 0.5
    rnd = zoo[zoo.selector == "random"].iloc[0]
    assert np.isclose(rnd["deployed_success"], 1 / 3)  # group mean success


def test_simulate_selector_uplift_monotonic_in_r_and_k():
    outcomes = [0, 1, 0, 1, 0, 0, 1, 0]  # 8 candidates, 3 successes
    cand = dg.build_candidate_table(
        sum((_group(f"g{i}", outcomes) for i in range(4)), []), model=None)
    surf = dg.simulate_selector_uplift(cand, r_grid=(0.5, 0.75, 1.0), k_grid=(1, 2, 4, 8))
    for k in (1, 2, 4, 8):  # deployed success is non-decreasing in r at every budget
        col = surf[surf.k == k].sort_values("r")["deployed_success"].to_numpy()
        assert np.all(np.diff(col) >= -1e-12)
        if k > 1:  # k=1 has no selection benefit (best-of-1 == random)
            assert col[-1] > col[0]
    oracle = surf[surf.r == 1.0].sort_values("k")["deployed_success"].to_numpy()
    assert np.all(np.diff(oracle) >= -1e-12) and oracle[-1] > oracle[0]  # best-of-k grows with k


def test_selector_zoo_paired_uplift_ci():
    # 5 groups, default always fails, fresh_noise_1 always succeeds; score picks the winner.
    cand = dg.build_candidate_table(
        sum((_group(f"g{i}", [0, 1, 0]) for i in range(5)), []), model=None)
    dg.add_mode_structure(cand)
    cand["score"] = cand["kind"].map(
        {"default": 0.1, "fresh_noise_1": 0.9, "fresh_noise_2": 0.5})
    zoo = dg.selector_zoo(cand, score_cols=("score",), n_bootstrap=500)
    assert {"uplift_ci_lo", "uplift_ci_hi"} <= set(zoo.columns)
    pick = zoo[zoo.selector == "pick_best"].iloc[0]
    assert pick["uplift_vs_default"] == 1.0 and pick["uplift_ci_lo"] > 0  # beats default

    # all-success groups -> every selector ties the default -> uplift CI includes 0.
    flat = dg.build_candidate_table(
        sum((_group(f"h{i}", [1, 1, 1]) for i in range(5)), []), model=None)
    dg.add_mode_structure(flat)
    zoo_flat = dg.selector_zoo(flat, n_bootstrap=500)
    rnd = zoo_flat[zoo_flat.selector == "random"].iloc[0]
    assert rnd["uplift_vs_default"] == 0.0 and rnd["uplift_ci_lo"] == 0.0  # does not beat default


def test_decidable_mass_binary_and_continuous():
    examples = (
        [_candidate("mixed", "default", 0, 0.0, n_steps=280),         # decidable (0 and 1);
         _candidate("mixed", "fresh_noise_1", 1, 1.0, n_steps=90),    # success finishes early,
         _candidate("mixed", "fresh_noise_2", 0, 2.0, n_steps=280)]   # failures run to the cap
        + [_candidate("allwin", "default", 1, 0.0, n_steps=100),      # unanimous success...
           _candidate("allwin", "fresh_noise_1", 1, 1.0, n_steps=80)]  # ...but faster candidate
        + [_candidate("allfail", "default", 0, 0.0, n_steps=280),     # unanimous failure,
           _candidate("allfail", "fresh_noise_1", 0, 1.0, n_steps=280)])  # identical n_steps
    cand = dg.build_candidate_table(examples, model=None)
    binary = dg.decidable_mass(cand)
    assert binary["n_groups"] == 3 and binary["binary_decidable"] == 1
    assert np.isclose(binary["binary_fraction"], 1 / 3)
    cont = dg.decidable_mass(cand, value_col="n_steps")
    # "allwin" has a within-group n_steps spread -> value-decidable on top of the binary one.
    assert cont["value_decidable"] == 2 and np.isclose(cont["value_fraction"], 2 / 3)
    assert len(dg.decidable_groups(cand).group_id.unique()) == 1  # only "mixed"


def test_prefix_distance_disagreement_continuous_rises():
    # near pairs share n_steps, far pairs differ by 100 -> continuous gap rises with distance.
    examples = [_candidate("g", "default", 1, 0.0, n_steps=100),
                _candidate("g", "fresh_noise_1", 1, 0.0, n_steps=100),
                _candidate("g", "fresh_noise_2", 1, 5.0, n_steps=200),
                _candidate("g", "fresh_noise_3", 1, 5.0, n_steps=200)]
    cand = dg.build_candidate_table(examples, model=None)
    _, slope = dg.prefix_distance_disagreement(cand, value_col="n_steps")
    assert slope > 0  # smooth structure the all-success binary label completely hides
    _, binary_slope = dg.prefix_distance_disagreement(cand)  # all success -> flat binary
    assert binary_slope == 0.0


def test_speed_selection_uplift():
    examples = (
        [_candidate("a", "default", 1, 0.0, n_steps=100),   # default succeeds in 100...
         _candidate("a", "fresh_noise_1", 1, 1.0, n_steps=80),  # ...faster success available
         _candidate("a", "fresh_noise_2", 0, 2.0, n_steps=200)]
        + [_candidate("b", "default", 1, 0.0, n_steps=50),  # default already the fastest
           _candidate("b", "fresh_noise_1", 1, 1.0, n_steps=50)])
    cand = dg.build_candidate_table(examples, model=None)
    out = dg.speed_selection_uplift(cand, n_bootstrap=500)
    assert out["n_eligible_groups"] == 2
    assert np.isclose(out["mean_step_savings"], 10.0)          # (20 + 0) / 2
    assert np.isclose(out["fraction_with_faster_option"], 0.5)  # only group "a"


def test_stratify_by_u_level_buckets_into_terciles():
    # six groups with distinct endpoint-U values -> three terciles over group values.
    cand = dg.build_candidate_table(
        sum((_group(f"g{i}", [0, 1, 0]) for i in range(6)), []), model=None)
    cand["u_s10"] = cand["group_id"].map(
        {f"g{i}": v for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])})
    table = dg.stratify_by_u_level(cand, "u_s10")
    assert "u_s10_bucket" in cand.columns
    assert set(table["u_s10_bucket"]) <= {"low", "mid", "high"}
    assert "oracle_uplift" in table.columns
