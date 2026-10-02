import numpy as np
import pytest

from pnp.experiments import LIBERO_SUITES
from pnp.smolvla_population_pcp_eval import population_manifest, reached_schedule
from scripts.validate_smolvla_q_predictions import auc, average_precision, summarize


def episodes():
    return [dict(suite=suite, task_idx=task, ep_idx=index, max_steps=300,
                 init_state_hash=f"{suite}/{task}/{index}")
            for suite in LIBERO_SUITES for task in range(10) for index in range(5)]


def test_population_selection_is_fixed_outcome_blind_and_covers_all_stages():
    eps = episodes()
    first = population_manifest(eps, {"groups": [], "snapshot_digest": "frozen"})
    second = population_manifest([{**e, "success": True, "n_steps": 1, "uncertainty": 999}
                                  for e in reversed(eps)], {"groups": [], "snapshot_digest": "frozen"})
    assert first == second
    items = first["items"]
    assert len(items) == 200
    assert {i["stage"] for i in items} == {"early", "middle", "late"}
    counts = [sum(i["stage"] == stage for i in items) for stage in ("early", "middle", "late")]
    assert max(counts)-min(counts) <= 1
    shards = [{i["ordinal"] for i in items if i["ordinal"] % 3 == shard} for shard in range(3)]
    assert not shards[0] & shards[1] and len(set.union(*shards)) == 200


def test_population_refuses_missing_or_training_overlap():
    with pytest.raises(ValueError):
        population_manifest(episodes()[:-1], {"groups": [], "snapshot_digest": "frozen"})
    with pytest.raises(ValueError, match="overlap"):
        population_manifest(episodes(), {"groups": [dict(suite=LIBERO_SUITES[0], task_idx=0, episode_idx=0)],
                                        "snapshot_digest": "frozen"})


def test_terminal_before_or_exactly_at_schedule_not_replaced_with_surviving_root():
    item = {"scheduled_step": 100, "chunk_idx": 10}
    assert not reached_schedule({"n_steps": 90, "n_chunks": 9}, item)
    assert not reached_schedule({"n_steps": 100, "n_chunks": 10}, item)
    assert reached_schedule({"n_steps": 101, "n_chunks": 11}, item)


def test_auc_and_ap_handle_ties_as_ties_not_input_order():
    assert auc([False, True], [0, 1]) == 1
    assert auc([False, True], [1, 0]) == 0
    assert auc([False, True], [.5, .5]) == .5
    assert average_precision([False, True], [.5, .5]) == .5


def test_random_selection_uses_number_of_successful_candidates():
    records = [dict(group_id="all_success", labels=[True]*9, logits=[0]*9),
               dict(group_id="all_failure", labels=[False]*9, logits=[0]*9),
               dict(group_id="one_success", labels=[False]*8+[True], logits=[0]*9)]
    report = summarize(records)
    assert report["random_selection_expected_successes"] == pytest.approx(1+1/9)
    assert report["within_root_pair_accuracy"] == .5
    assert report["selected_successes"] == 1
    assert report["repeated_stock_score_baseline"]["brier"] == .25
    assert report["mixed_roots"] == 1


def test_missing_candidates_cannot_silently_change_denominator():
    with pytest.raises(ValueError):
        summarize([dict(group_id="short", labels=[True], logits=[0])])
