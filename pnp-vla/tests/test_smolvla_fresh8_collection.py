import json
from pathlib import Path

from pnp.smolvla_fresh8_collection import (
    FRESH8_KINDS, SMOLVLA_FRESH8_EXPERIMENT, _complete_groups, _group_id,
    _link_v3_fresh, _reusable_v3, _v3_group_id)


class _Store:
    def __init__(self, group, candidates):
        self.group = group
        self.candidates = candidates

    def fetch_all(self, table, columns, *, configure=None, order_by=()):
        return [self.group] if table == "verifier_candidate_groups" else self.candidates


def _item():
    return {"suite": "libero_spatial", "task_idx": 2, "episode_idx": 13,
            "chunk_idx": 4, "source_episode_seed": 42,
            "source_rollout_id": "source-42"}


def test_resume_requires_exact_nine_candidate_fresh8_contract():
    item = _item()
    group_id = _group_id(item)
    group = {"candidate_group_id": group_id, "manifest_hash": "frozen",
             "metadata_json": {"candidate_families": {
            "stored_source": 1, "fresh_initial_noise": 8,
            "fixed_initial_noise_new_pnp_perturbation": 0}}}
    candidates = [
        {"candidate_group_id": group_id, "candidate_kind": kind,
         "metadata_json": {"training_data_path": f"path/{kind}"}}
        for kind in sorted(FRESH8_KINDS)]
    assert _complete_groups(_Store(group, candidates), [item], "frozen") == {group_id}
    assert _complete_groups(_Store(group, candidates[:-1]), [item], "frozen") == set()
    assert _complete_groups(_Store(group, candidates), [item], "other") == set()


def test_v3_reuse_requires_complete_persisted_branches():
    item = _item()
    group_id = _v3_group_id(item)
    group = {"candidate_group_id": group_id, "manifest_hash": "frozen",
             "metadata_json": {"source_rollout_id": item["source_rollout_id"]}}
    kinds = ["stored_source", *(f"fresh_seed_{i}" for i in range(1, 5)),
             *(f"pnp_perturb_{i}" for i in range(1, 5))]
    candidates = [{"candidate_group_id": group_id, "candidate_kind": kind,
                   "success": False, "n_steps": 10,
                   "metadata_json": {"training_data_path": f"data/{kind}"},
                   "policy_chunk_path": f"policy/{kind}",
                   "env_chunk_path": f"env/{kind}",
                   "observation_path": f"observation/{kind}"}
                  for kind in kinds]
    assert len(_reusable_v3(_Store(group, candidates), [item], "frozen")[group_id]) == 4
    assert _reusable_v3(_Store(group, candidates[:-1]), [item], "frozen") == {}
    assert _reusable_v3(_Store(group, candidates), [item], "other") == {}


def test_v3_link_preserves_artifact_paths_and_adds_provenance():
    class Table:
        def __init__(self):
            self.rows = []

        def upsert(self, row, *, on_conflict):
            assert on_conflict == "candidate_id"
            self.rows.append(row)
            return self

        def execute(self):
            return None

    class Store:
        def __init__(self):
            self.saved = Table()
            self.client = self

        def table(self, name):
            assert name == "verifier_candidates"
            return self.saved

        def _json(self, row):
            return row

    store = Store()
    old = {"candidate_group_id": "v3", "candidate_kind": "fresh_seed_1",
           "success": True, "n_steps": 14,
           "metadata_json": {"training_data_path": "data/1"},
           "policy_chunk_path": "policy/1", "env_chunk_path": "env/1",
           "observation_path": "observation/1"}
    _link_v3_fresh(store, "v4", [old])
    row = store.saved.rows[0]
    assert row["candidate_group_id"] == "v4"
    assert row["metadata_json"]["reused_from_v3_group_id"] == "v3"
    assert row["metadata_json"]["training_data_path"] == "data/1"
    assert row["policy_chunk_path"] == "policy/1"


def test_three_colab_workers_start_with_one_tree_smoke():
    assert SMOLVLA_FRESH8_EXPERIMENT.endswith("fresh8-trees-v4-bellman")
    directory = Path(__file__).parents[1] / "notebooks" / "workers"
    paths = sorted(directory.glob("108_smolvla_fresh8_tree_worker_*.ipynb"))
    assert len(paths) == 3
    for index, path in enumerate(paths):
        notebook = json.loads(path.read_text())
        source = "\n".join("".join(cell.get("source", []))
                           for cell in notebook["cells"])
        assert "SHARD_COUNT = 3" in source
        assert f"SHARD_INDEX = {index}" in source
        assert "TREE_LIMIT = 1" in source
        assert "run_smolvla_fresh8_worker" in source
        assert "libEGL_nvidia" in source
