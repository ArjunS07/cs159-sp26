import numpy as np

from pnp.smolvla_demo_pretraining import (
    _axis_angle_to_quaternion, _demo_windows, select_demo_episodes)


def _rows():
    rows = []
    for task in range(40):
        for episode in range(12):
            rows.append({
                "episode_index": task * 100 + episode,
                "tasks": [f"task-{task}"], "length": 21})
    return rows


def test_demo_selection_is_balanced_and_disjoint():
    manifest = select_demo_episodes(_rows(), seed=7)
    selected = manifest["episodes"]
    assert len(selected) == 400
    by_task = {}
    for row in selected:
        by_task.setdefault(row["task"], []).append(row)
    assert len(by_task) == 40
    assert all(sum(row["split"] == "train" for row in rows) == 8
               for rows in by_task.values())
    assert all(sum(row["split"] == "validation" for row in rows) == 2
               for rows in by_task.values())


def test_axis_angle_to_xyzw_quaternion():
    np.testing.assert_allclose(
        _axis_angle_to_quaternion(np.zeros(3)), [0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(
        _axis_angle_to_quaternion(np.asarray([0, 0, np.pi])), [0, 0, 1, 0], atol=1e-6)


def test_demo_windows_have_terminal_reward_and_ema_bootstrap():
    count, tokens, width = 3, 4, 6
    arrays = _demo_windows(
        prefixes=np.zeros((count, tokens, width), np.float16),
        pads=np.ones((count, tokens), bool),
        raw_robots=np.zeros((count, 9), np.float32),
        proprios=np.zeros((count, 8), np.float32),
        normalized_actions=np.zeros((count, 10, 7), np.float32),
        action_valid=np.asarray(
            [[1] * 10, [1] * 10, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], bool),
        starts=np.asarray([0, 10, 20]), episode_length=21, gamma=.99)
    np.testing.assert_allclose(arrays["discount"][:2], .99 ** 10)
    assert arrays["discount"][2] == 0
    assert arrays["reward"][0] == 0 and arrays["reward"][1] == 0
    np.testing.assert_allclose(arrays["reward"][2], 1.0)
    np.testing.assert_allclose(arrays["mc_return"], [.99 ** 20, .99 ** 10, 1.0])
    assert arrays["next_action_valid"][2].sum() == 1
