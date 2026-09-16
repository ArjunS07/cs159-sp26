import pandas as pd

from analysis.smolvla_libero import (
    flip_auc_table, pair_stock_refinement, uncertainty_quantile_flip_table)


def _arm(successes):
    return pd.DataFrame({
        "suite": ["libero_10"] * len(successes),
        "task_idx": list(range(len(successes))),
        "episode_idx": [0] * len(successes),
        "init_state_hash": [f"h{i}" for i in range(len(successes))],
        "success": successes,
    })


def test_pair_and_flip_aucs_keep_conditional_questions_separate():
    paired = pair_stock_refinement(
        _arm([False, False, True, True]),
        _arm([False, True, False, True]),
    )
    assert paired.transition.astype(str).tolist() == ["F->F", "F->S", "S->F", "S->S"]
    paired["u"] = [0.1, 0.9, 0.8, 0.2]
    result = flip_auc_table(paired, ["u"], n_boot=20).set_index("cohort")
    assert result.loc["stock failures", "auc"] == 1.0
    assert result.loc["stock successes", "auc"] == 1.0
    assert result.loc["discordant only", "auc"] == 1.0


def test_quantile_table_preserves_all_transitions():
    paired = pair_stock_refinement(
        _arm([False, False, True, True]),
        _arm([False, True, False, True]),
    )
    paired["u"] = [0.1, 0.2, 0.3, 0.4]
    table = uncertainty_quantile_flip_table(paired, score_column="u", bins=2)
    assert table.episodes.sum() == 4
    assert table[["F_to_F", "F_to_S", "S_to_F", "S_to_S"]].to_numpy().sum() == 4
