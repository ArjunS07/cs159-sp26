import numpy as np
import pytest

from pnp.smolvla_success_critic import (
    SuccessTrainConfig, TimedRoots, TimedWindows, _config_contract,
    action_statistics, remaining_fraction,
    sample_tree_td_indices, select_training_roots)
from pnp.smolvla_tree_bellman_finetune import _materialize_candidate
from pnp.smolvla_tree_bellman_finetune import _fetch_tree_snapshot


def test_remaining_fraction_uses_suite_budget_and_root_offset():
    assert remaining_fraction("libero_spatial", 5, 10) == pytest.approx(160 / 220)
    assert remaining_fraction("libero_object", 0, 0) == 1.0
    with pytest.raises(ValueError):
        remaining_fraction("libero_spatial", 23, 0)


def test_timed_tree_views_keep_root_and_next_budget_aligned(tmp_path):
    np.savez_compressed(
        tmp_path / "group_g.npz",
        prefix=np.zeros((2, 4), np.float16), pad=np.ones(2, bool),
        robot=np.zeros(9, np.float32), proprio=np.zeros(8, np.float32),
        actions=np.ones((9, 10, 7), np.float32),
        action_valid=np.ones((9, 10), bool),
        success=np.zeros(9, bool),
    )
    np.savez_compressed(
        tmp_path / "candidate.npz",
        robot=np.zeros((1, 9), np.float32),
        next_robot=np.zeros((1, 9), np.float32),
        start_step=np.asarray([60], np.int32),
        fork_offset=np.asarray([1], np.int32),
        discount=np.asarray([1.0], np.float32),
    )
    cache = {
        "cache_dir": str(tmp_path),
        "group_entries": [{"candidate_group_id": "g", "path": "group_g.npz"}],
        "candidate_entries": [{"candidate_group_id": "g", "path": "candidate.npz",
                               "n_windows": 1}],
    }
    groups = {"g": {"suite": "libero_spatial", "chunk_idx": 5}}
    roots = TimedRoots(cache, ["g"], groups)
    windows = TimedWindows(cache, ["g"], groups)
    assert roots[0]["robot"][-1] == pytest.approx(170 / 220)
    assert windows[0]["robot"][-1] == pytest.approx(160 / 220)
    assert windows[0]["next_robot"][-1] == pytest.approx(150 / 220)
    mean, std = action_statistics(roots)
    np.testing.assert_allclose(mean, 1)
    assert np.all(std > 0)


def test_success_cache_uses_terminal_event_not_shaped_reward(monkeypatch, tmp_path):
    arrays = {
        "actions_normalized": np.zeros((2, 7), np.float32),
        "rewards": np.asarray([.2, .7], np.float32),
        "terminated": np.asarray([False, True]),
        "truncated": np.zeros(2, bool),
        "step_success": np.asarray([False, True]),
        "boundary/step": np.asarray([0, 2], np.int32),
        "boundary/raw_robot_state": np.zeros((2, 9), np.float32),
        "boundary/policy_proprio": np.zeros((2, 8), np.float32),
        "prefix/prefix_embeddings": np.zeros((2, 1, 2, 4), np.float32),
        "prefix/prefix_pad_masks": np.ones((2, 1, 2), bool),
        "bellman/action": np.zeros((1, 50, 7), np.float32),
        "bellman/executed_normalized": np.zeros((1, 10, 7), np.float32),
        "bellman/validity_mask": np.asarray([[True, True] + [False] * 8]),
    }
    monkeypatch.setattr(
        "pnp.smolvla_tree_bellman_finetune.load_training_fields_with_retry",
        lambda *args: arrays)
    candidate = {
        "candidate_id": "c", "candidate_kind": "fresh_seed_1",
        "training_data_path": "unused", "success": True,
        "training_data_start_boundary": 0,
    }
    _materialize_candidate(None, candidate, "g", tmp_path,
                           gamma=1.0, success_reward=True)
    with np.load(tmp_path / "c.npz") as archive:
        assert float(archive["reward"][0]) == pytest.approx(1.0)
        assert float(archive["mc_return"][0]) == pytest.approx(1.0)
        assert float(archive["discount"][0]) == 0.0


def test_td_sampler_groups_windows_without_changing_marginal_distribution():
    class Dataset:
        base = type("Base", (), {"entries": [
            {"n_windows": 1}, {"n_windows": 3}]})()

    rng = np.random.default_rng(9)
    samples = np.concatenate([
        sample_tree_td_indices(Dataset(), rng, branches=4)
        for _ in range(1000)
    ])
    counts = np.bincount(samples, minlength=4) / len(samples)
    assert len(counts) == 4
    np.testing.assert_allclose(counts, np.full(4, .25), atol=.025)


def test_fresh8_snapshot_orders_only_the_nine_requested_candidate_kinds():
    kinds = ("stored_source", *(f"fresh_seed_{i}" for i in range(1, 9)))

    class Store:
        def fetch_all(self, table, columns, *, configure=None, order_by=()):
            if table == "verifier_candidate_groups":
                return [{"candidate_group_id": "g", "suite": "libero_spatial",
                         "task_idx": 0, "episode_idx": 12, "chunk_idx": 4,
                         "metadata_json": {}}]
            return [{"candidate_id": kind, "candidate_group_id": "g",
                     "candidate_kind": kind, "success": kind == "fresh_seed_8",
                     "n_steps": 80,
                     "metadata_json": {"training_data_path": f"path/{kind}"}}
                    for kind in reversed(kinds)]

    snapshot = _fetch_tree_snapshot(
        Store(), tree_limit=1, experiment="fresh8-test",
        candidate_kinds=kinds)
    assert snapshot["experiment"] == "fresh8-test"
    assert tuple(snapshot["candidate_kinds"]) == kinds
    assert tuple(row["candidate_kind"] for row in snapshot["groups"][0]["candidates"]) == kinds


def test_existing_v3_snapshot_contract_keeps_implicit_candidate_order():
    from pnp.smolvla_tree_critic import TREE_Q10_KINDS

    class Store:
        def fetch_all(self, table, columns, *, configure=None, order_by=()):
            if table == "verifier_candidate_groups":
                return [{"candidate_group_id": "g", "suite": "libero_spatial",
                         "task_idx": 0, "episode_idx": 12, "chunk_idx": 4,
                         "metadata_json": {}}]
            return [{"candidate_id": kind, "candidate_group_id": "g",
                     "candidate_kind": kind, "success": False, "n_steps": 80,
                     "metadata_json": {"training_data_path": f"path/{kind}"}}
                    for kind in TREE_Q10_KINDS]

    snapshot = _fetch_tree_snapshot(Store(), tree_limit=1)
    assert "candidate_kinds" not in snapshot
    assert tuple(row["candidate_kind"] for row in snapshot["groups"][0]["candidates"]) == TREE_Q10_KINDS


def test_scaling_subsets_are_nested_and_independent_of_input_order():
    ids = [f"group-{i}" for i in range(640)]
    small = select_training_roots(ids, 100)
    medium = select_training_roots(list(reversed(ids)), 200)
    large = select_training_roots(ids, 400)
    assert set(small) < set(medium) < set(large)
    assert len({*small, *medium, *large}) == 400
    with pytest.raises(ValueError):
        select_training_roots(ids, 641)


def test_full_data_checkpoint_contract_remains_compatible():
    assert "train_root_limit" not in _config_contract(SuccessTrainConfig(arm="root_mc"))
    assert _config_contract(SuccessTrainConfig(
        arm="root_mc", train_root_limit=100))["train_root_limit"] == 100
