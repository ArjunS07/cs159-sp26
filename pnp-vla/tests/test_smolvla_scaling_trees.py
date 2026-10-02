import json
from pathlib import Path

import pytest

from pnp.smolvla_scaling_trees import (
    KINDS, SOURCE_IDENTITIES, TREE_SHARDS, _complete_groups, _group_id,
    _worker_items,
)


def test_three_worker_partition_preserves_all_frozen_roots():
    assert TREE_SHARDS == 2  # the persisted manifest hash keeps its original hint
    manifest = {"payload": {"items": [
        {"ordinal": ordinal, "shard_index": ordinal % 2}
        for ordinal in range(SOURCE_IDENTITIES)
    ]}}
    partitions = [_worker_items(manifest, index, 3) for index in range(3)]
    assert [len(items) for items in partitions] == [267, 267, 266]
    ordinals = [item["ordinal"] for items in partitions for item in items]
    assert sorted(ordinals) == list(range(SOURCE_IDENTITIES))
    assert len(ordinals) == len(set(ordinals))
    with pytest.raises(ValueError):
        _worker_items(manifest, 3, 3)


def test_completed_two_worker_tree_is_seen_by_new_partition():
    item = {"ordinal": 2, "shard_index": 0, "suite": "libero_spatial",
            "task_idx": 1, "episode_idx": 30, "chunk_idx": 2,
            "source_episode_seed": 47}
    manifest = {"payload": {"items": [
        {"ordinal": ordinal, "shard_index": ordinal % 2, **({} if ordinal != 2 else item)}
        for ordinal in range(SOURCE_IDENTITIES)
    ]}}
    assert item in _worker_items(manifest, 2, 3)
    group_id = _group_id(item)

    class Store:
        def fetch_all(self, table, _columns, **_kwargs):
            if table == "verifier_candidate_groups":
                return [{"candidate_group_id": group_id, "manifest_hash": "frozen"}]
            return [{"candidate_group_id": group_id, "candidate_kind": kind,
                     "metadata_json": {"training_data_path": f"data/{kind}"}}
                    for kind in KINDS]

    assert _complete_groups(Store(), [item], "frozen") == {group_id}


def test_three_117_notebooks_expose_matching_shard_settings():
    directory = Path(__file__).parents[1] / "notebooks" / "workers"
    paths = sorted(directory.glob("117_smolvla_scaling_fresh8_tree_worker_*.ipynb"))
    assert len(paths) == 3
    for index, path in enumerate(paths):
        notebook = json.loads(path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "SHARD_COUNT = 3" in source
        assert f"SHARD_INDEX = {index}" in source
        assert "shard_count=SHARD_COUNT" in source
        assert "TREE_LIMIT = 1" in source
