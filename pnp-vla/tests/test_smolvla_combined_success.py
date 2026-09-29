import io
from unittest.mock import patch

import numpy as np
import pytest

from pnp.smolvla_combined_success import (
    COMBINED_EXPERIMENT, _read_root_group, load_or_create_combined_snapshot,
)
from pnp.smolvla_q_selection import FRESH8_KINDS
from pnp.smolvla_tree_bellman_finetune import _json_digest


def _bytes(**arrays):
    handle = io.BytesIO()
    np.savez_compressed(handle, **arrays)
    return handle.getvalue()


def test_root_cache_matches_source_actions_and_masks_terminal_tail(tmp_path):
    source = np.arange(70, dtype=np.float32).reshape(10, 7) / 100
    paths = {"source.npz": _bytes(**{
        "prefix/prefix_embeddings": np.ones((2, 1, 4, 8), np.float16),
        "prefix/prefix_pad_masks": np.ones((2, 1, 4), bool),
        "boundary/raw_robot_state": np.ones((2, 9), np.float32),
        "boundary/policy_proprio": np.ones((2, 8), np.float32),
        "boundary/step": np.array([0, 10]),
        "bellman/action": source[None],
    })}
    candidates = []
    for index, kind in enumerate(FRESH8_KINDS):
        action = source if index == 0 else source + index / 10
        path = f"{kind}.npz"
        paths[path] = _bytes(actions=action)
        candidates.append({"candidate_id": kind, "candidate_kind": kind,
                           "policy_chunk_path": path, "n_steps": 6 if index == 1 else 10,
                           "success": index == 1})

    class Store:
        _download = paths.__getitem__

    group = {"candidate_group_id": "root", "source_training_data_path": "source.npz",
             "source_boundary_index": 0, "suite": "libero_spatial", "task_idx": 0,
             "source_experiment": "test", "candidates": candidates}
    entry = _read_root_group(Store(), group, tmp_path)
    with np.load(tmp_path / entry["path"]) as archive:
        np.testing.assert_array_equal(archive["actions"][0], source)
        np.testing.assert_array_equal(archive["actions"][1, :6], source[:6] + .1)
        assert not archive["action_valid"][1, 6:].any()
        assert archive["actions"].shape == (9, 10, 7)
        assert archive["prefix"].shape == (128, 8)
        assert archive["success"].sum() == 1


def test_combined_snapshot_freezes_disjoint_cohorts():
    from pnp.smolvla_combined_success import SOURCE_EXPERIMENTS

    snapshots = []
    for cohort in range(2):
        groups = []
        for index in range(800):
            groups.append({"candidate_group_id": f"g{cohort}-{index}",
                           "candidates": [{"candidate_id": f"c{cohort}-{index}-{kind}"}
                                          for kind in FRESH8_KINDS]})
        snapshots.append({"snapshot_digest": f"source-{cohort}", "groups": groups})

    class Store:
        def __init__(self):
            self.payload = None

        def _upload(self, _key, value):
            self.payload = value

    store = Store()

    def download(_store, _key):
        if store.payload is None:
            raise FileNotFoundError("404 not found")
        return store.payload

    with patch("pnp.smolvla_combined_success._fetch_tree_snapshot",
               side_effect=snapshots), patch(
               "pnp.smolvla_combined_success._download_with_retry", side_effect=download):
        result = load_or_create_combined_snapshot(store)
    assert result["experiment"] == COMBINED_EXPERIMENT
    assert len(result["groups"]) == 1600
    assert [row["experiment"] for row in result["source_snapshots"]] == list(SOURCE_EXPERIMENTS)
    check = dict(result)
    digest = check.pop("snapshot_digest")
    assert digest == _json_digest(check)


def test_combined_snapshot_rejects_duplicate_groups():
    same = {"snapshot_digest": "x", "groups": [
        {"candidate_group_id": f"g-{index}", "candidates": [
            {"candidate_id": f"c-{index}-{kind}"} for kind in FRESH8_KINDS]}
        for index in range(800)]}
    with patch("pnp.smolvla_combined_success._fetch_tree_snapshot", return_value=same):
        with pytest.raises(ValueError, match="distinct complete trees"):
            load_or_create_combined_snapshot(object())
