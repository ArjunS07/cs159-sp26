import numpy as np
from dataclasses import replace

from pnp.verifier.data import (
    CleanChunkExample, build_chunk_transitions, build_clean_chunk_examples, candidate_cv_splits,
    exclude_candidate_identities, known_task_split, locked_candidate_split, select_examples,
    shuffle_candidate_actions_within_group, validate_candidate_groups,
)


def _example(task, episode, success, chunk=0):
    return CleanChunkExample(
        rollout_id=f"r-{task}-{episode}", experiment="e", benchmark="libero",
        suite="suite", task_idx=task, episode_idx=episode, chunk_idx=chunk,
        chunk_position=0.0, obs_enc=np.zeros(8, np.float32),
        actions=np.zeros((50, 7), np.float32), action_mask=np.ones(50, bool), success=success)


def test_build_clean_chunks_preserves_terminal_partial_mask():
    import pandas as pd

    row = {"rollout_id": "r", "experiment": "e", "benchmark": "libero", "suite": "s",
           "task_idx": 0, "episode_idx": 0, "success": True}
    frame = pd.DataFrame([
        {"chunk_idx": 0, "chunk_pos": 0.0, "obs_enc": [1.0] * 8},
        {"chunk_idx": 1, "chunk_pos": 0.5, "obs_enc": [2.0] * 8},
    ])
    trajectory = {"actions": np.arange(60 * 7, dtype=np.float32).reshape(60, 7)}
    examples = build_clean_chunk_examples(row, frame, trajectory)
    assert len(examples) == 2
    assert examples[0].action_mask.sum() == 50
    assert examples[1].action_mask.sum() == 10
    assert np.all(examples[1].actions[10:] == 0)


def test_build_chunk_transitions_has_td_and_monte_carlo_targets():
    import pandas as pd

    row = {"rollout_id": "r", "experiment": "e", "benchmark": "libero", "suite": "s",
           "task_idx": 0, "episode_idx": 0, "success": True}
    frame = pd.DataFrame([
        {"chunk_idx": 0, "chunk_pos": 0.0, "obs_enc": [1.0] * 8},
        {"chunk_idx": 1, "chunk_pos": 0.5, "obs_enc": [2.0] * 8},
    ])
    trajectory = {"actions": np.ones((60, 7), dtype=np.float32)}
    transitions = build_chunk_transitions(row, frame, trajectory, gamma=.9)
    assert len(transitions) == 2
    assert transitions[0].reward == 0
    assert np.isclose(transitions[0].discount, .9 ** 50)
    assert np.array_equal(transitions[0].next_obs_enc, np.full(8, 2, np.float32))
    assert transitions[0].next_chunk_position == .5
    assert transitions[1].terminal and transitions[1].reward == 1
    assert transitions[1].discount == 0 and transitions[1].action_mask.sum() == 10
    assert np.isclose(transitions[0].return_target, .9 ** 59)
    assert np.isclose(transitions[1].return_target, .9 ** 9)


def test_known_split_keeps_rollout_chunks_together_and_disjoint():
    examples = [_example(task, episode, episode % 2) for task in range(4) for episode in range(10)]
    split = known_task_split(examples)
    assert {name: len(ids) for name, ids in split.items()} == {
        "train": 24, "val": 4, "cal": 4, "test": 8}
    sets = [set(ids) for ids in split.values()]
    assert not any(a & b for i, a in enumerate(sets) for b in sets[i + 1:])


def test_known_split_spreads_two_failures_between_train_and_test():
    examples = [_example(0, episode, episode >= 2) for episode in range(10)]
    split = known_task_split(examples)
    label = {e.rollout_id: e.success for e in examples}
    assert sum(not label[rid] for rid in split["train"]) == 1
    assert sum(not label[rid] for rid in split["test"]) == 1
    assert {name: len(ids) for name, ids in split.items()} == {
        "train": 6, "val": 1, "cal": 1, "test": 2}


def test_known_split_keeps_a_single_failure_in_training():
    examples = [_example(0, episode, episode >= 1) for episode in range(10)]
    split = known_task_split(examples)
    label = {e.rollout_id: e.success for e in examples}
    assert sum(not label[rid] for rid in split["train"]) == 1
    assert all(label[rid] for name in ("val", "cal", "test") for rid in split[name])


def _candidate_examples(n_groups=20):
    examples = []
    for group_index in range(n_groups):
        base = replace(
            _example(group_index % 4, group_index, True),
            benchmark="libero_pro" if group_index % 2 else "libero",
            uncertainty_stratum=("high" if group_index % 3 else "medium"),
            candidate_group_id=f"g{group_index}", candidate_kind="default",
        )
        outcomes = (True, False, True, False) if group_index < 10 else (True,) * 4
        for candidate_index, success in enumerate(outcomes):
            actions = np.full((50, 7), candidate_index + group_index * 10, np.float32)
            examples.append(replace(
                base, rollout_id=f"g{group_index}-c{candidate_index}",
                candidate_kind="default" if candidate_index == 0 else "sampled",
                success=success, actions=actions))
    return examples


def test_locked_split_and_cv_are_group_safe_and_cover_development_once():
    examples = _candidate_examples()
    locked = locked_candidate_split(examples, seed=7)
    development = select_examples(examples, locked["development"])
    test = select_examples(examples, locked["test"])
    assert len({e.candidate_group_id for e in test}) == 4
    assert ({e.candidate_group_id for e in development}
            .isdisjoint({e.candidate_group_id for e in test}))
    folds = candidate_cv_splits(examples, locked["development"], seed=7)
    validation = []
    for fold in folds:
        train_groups = {e.candidate_group_id for e in select_examples(examples, fold["train"])}
        val_groups = {e.candidate_group_id for e in select_examples(examples, fold["val"])}
        assert train_groups.isdisjoint(val_groups)
        validation.extend(val_groups)
    assert sorted(validation) == sorted({e.candidate_group_id for e in development})


def test_candidate_cv_keeps_multiple_groups_from_one_episode_together():
    examples = _candidate_examples(20)
    second_group = [
        replace(
            example, rollout_id=example.rollout_id.replace("g0", "g-extra"),
            candidate_group_id="g-extra")
        for example in examples if example.candidate_group_id == "g0"
    ]
    examples += second_group
    ids = [example.rollout_id for example in examples]
    folds = candidate_cv_splits(examples, ids, seed=9)
    locations = {}
    for fold_index, fold in enumerate(folds):
        for group in {
                example.candidate_group_id
                for example in select_examples(examples, fold["val"])}:
            locations[group] = fold_index
    assert locations["g0"] == locations["g-extra"]


def test_identity_exclusion_and_within_group_shuffle():
    candidates = _candidate_examples(2)
    historical = [_example(0, 0, True), _example(9, 9, False)]
    filtered = exclude_candidate_identities(historical, candidates)
    assert [example.task_idx for example in filtered] == [9]
    shuffled = shuffle_candidate_actions_within_group(candidates, seed=4)
    for group in ("g0", "g1"):
        before = [e for e in candidates if e.candidate_group_id == group]
        after = [e for e in shuffled if e.candidate_group_id == group]
        assert all(not np.array_equal(a.actions, b.actions) for a, b in zip(before, after))
        assert sorted(float(e.actions[0, 0]) for e in before) == sorted(
            float(e.actions[0, 0]) for e in after)


def test_candidate_group_validation_rejects_partial_and_state_mismatch():
    examples = _candidate_examples(2)
    assert validate_candidate_groups(examples, expected_candidates=4) == {
        "groups": 2, "candidates": 8, "discordant_groups": 2}
    with np.testing.assert_raises_regex(ValueError, "expected 4 candidates"):
        validate_candidate_groups(examples[:-1], expected_candidates=4)
    mismatched = list(examples)
    mismatched[1] = replace(mismatched[1], chunk_idx=9)
    with np.testing.assert_raises_regex(ValueError, "do not share one observation/state"):
        validate_candidate_groups(mismatched, expected_candidates=4)
