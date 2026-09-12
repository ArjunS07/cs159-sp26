import json

import numpy as np

from pnp.qplanning_critic.data import QPlanningCacheIndex, QPlanningWindowDataset
from pnp.qplanning_critic.replay_priority import (
    PrioritizedReplayBatchSampler, build_replay_priority_plan)
from pnp.qplanning_critic.uncertainty import U20LabelCache


def _dataset_and_labels(tmp_path):
    entries, label_entries = [], []
    for rollout in range(8):
        rollout_id = f"r{rollout}"
        entries.append({
            "rollout_id": rollout_id,
            "path": f"{rollout_id}.npz",
            "n_windows": 8,
            "suite": f"suite{rollout // 4}",
        })
        label_path = tmp_path / f"{rollout_id}_u.npz"
        np.savez(
            label_path,
            current_u20=np.full(8, .01 + .01 * (rollout % 4), np.float32),
            next_current_u20=np.zeros(8, np.float32),
            future_u20=np.zeros(8, np.float32),
            future_u20_valid=np.zeros(8, bool),
        )
        label_entries.append({
            "rollout_id": rollout_id,
            "path": label_path.name,
            "n_windows": 8,
        })
    cache = QPlanningCacheIndex(
        "pcpcds-test", 50, str(tmp_path), tuple(entries), 8, 3, 2, 7,
        tuple(np.zeros(7)), tuple(np.ones(7)), 64, 64, 0, 0, 0.0, 0.0)
    labels = U20LabelCache(
        "pcpcds-test", str(tmp_path), tuple(label_entries),
        0.0, 1.0, 0.0, 1.0, 64, 0)
    return QPlanningWindowDataset(cache, {entry["rollout_id"] for entry in entries}), labels


def test_priority_plans_select_failure_and_within_suite_high_u(tmp_path):
    dataset, labels = _dataset_and_labels(tmp_path)
    outcomes = {f"r{index}": index % 2 == 0 for index in range(8)}

    failure = build_replay_priority_plan(
        dataset, strategy="failure", outcomes=outcomes)
    assert set(failure.priority_by_rollout) == {"r1", "r3", "r5", "r7"}

    episode = build_replay_priority_plan(
        dataset, strategy="episode_u20", outcomes=outcomes, labels=labels)
    assert set(episode.priority_by_rollout) == {"r3", "r7"}

    four = build_replay_priority_plan(
        dataset, strategy="u20_4chunk", outcomes=outcomes, labels=labels)
    eight = build_replay_priority_plan(
        dataset, strategy="u20_8chunk", outcomes=outcomes, labels=labels)
    assert set(four.priority_by_rollout) == {"r3", "r7"}
    assert set(eight.priority_by_rollout) == {"r3", "r7"}
    assert four.summary["priority_groups"] == 4
    assert eight.summary["priority_groups"] == 2


def test_priority_sampler_is_fixed_size_deterministic_and_valid(tmp_path):
    dataset, labels = _dataset_and_labels(tmp_path)
    outcomes = {f"r{index}": index % 2 == 0 for index in range(8)}
    plan = build_replay_priority_plan(
        dataset, strategy="u20_4chunk", outcomes=outcomes, labels=labels)
    first = next(iter(PrioritizedReplayBatchSampler(dataset, plan, 16, 42)))
    second = next(iter(PrioritizedReplayBatchSampler(dataset, plan, 16, 42)))
    assert first == second
    assert len(first) == 16
    assert all(0 <= index < len(dataset) for index in first)


def test_two_priority_notebooks_split_four_fresh_q50_runs():
    expected = {
        0: ("failure", "episode_u20"),
        1: ("u20_4chunk", "u20_8chunk"),
    }
    for worker, strategies in expected.items():
        path = (
            f"notebooks/workers/72_train_q50_priority_worker_{worker}.ipynb")
        notebook = json.loads(open(path, encoding="utf-8").read())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert f"STRATEGIES = {strategies!r}" in source
        assert "run_q50_priority_worker(" in source
        assert "'updates_per_strategy': 8000" in source
        assert "MICRO_BATCH_SIZE = 64" in source
        assert "resume=True" in source
