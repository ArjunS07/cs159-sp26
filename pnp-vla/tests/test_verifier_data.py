import numpy as np
import pandas as pd

from pnp.verifier.data import (
    CleanChunkExample, build_clean_chunk_examples, hard_task_keys,
    heldout_task_split, known_task_split,
)


def _example(task, episode, success, chunk=0):
    return CleanChunkExample(
        rollout_id=f"r-{task}-{episode}", experiment="e", benchmark="libero",
        suite="suite", task_idx=task, episode_idx=episode, chunk_idx=chunk,
        chunk_position=0.0, obs_enc=np.zeros(8, np.float32),
        actions=np.zeros((50, 7), np.float32), action_mask=np.ones(50, bool), success=success)


def test_build_clean_chunks_preserves_terminal_partial_mask():
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


def test_hard_tasks_count_each_rollout_once_not_each_chunk():
    examples = []
    for episode in range(10):
        examples.extend([_example(0, episode, episode < 5, chunk=0),
                         _example(0, episode, episode < 5, chunk=1)])
        examples.append(_example(1, episode, True))
    assert hard_task_keys(examples) == {("libero", "suite", 0)}


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


def test_heldout_split_has_disjoint_tasks():
    examples = [_example(task, episode, episode % 2) for task in range(40) for episode in range(10)]
    split = heldout_task_split(examples)
    task_of = {e.rollout_id: e.task_idx for e in examples}
    task_sets = {name: {task_of[rid] for rid in ids} for name, ids in split.items()}
    assert len(task_sets["test"]) == 8
    assert len(task_sets["val"]) == len(task_sets["cal"]) == 4
    assert not any(a & b for i, a in enumerate(task_sets.values())
                   for b in list(task_sets.values())[i + 1:])
