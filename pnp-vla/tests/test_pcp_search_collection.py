import ast
import json
from pathlib import Path

from pnp.config import Method
from pnp.pcp_search.collection import build_collection_config, manifest_shard
from pnp.pcp_search.task_selection import build_initial_manifest, ALL_TASKS
from pnp.rollout import episode_seed
from pnp.store import SupabaseStore


def _history():
    return [{
        "rollout_id": f"r-{suite}-{task_idx}-{episode_idx}",
        "suite": suite, "task_idx": task_idx, "episode_idx": episode_idx,
        "init_state_hash": str(episode_idx), "status": "completed",
        "success": episode_idx >= 30, "u20": episode_idx / 100,
    } for suite, task_idx in ALL_TASKS for episode_idx in range(20, 40)]


def test_collection_config_is_vanilla_ten_action_and_complete():
    manifest = build_initial_manifest(_history())
    config = build_collection_config(manifest)
    assert config.n_action_steps == 10
    assert tuple(config.pnp_steps) == (3, 4)
    assert config.pnp_k == 5
    assert not config.refine
    assert config.save_training_data
    assert config.save_ahats and config.save_time_uncertainty
    assert config.save_pcp_features and config.save_generated_chunks
    assert not config.skip_unused_renders


def test_behavior_seed_changes_new_seed_and_rollout_identity_without_changing_legacy_ids():
    state = [1.0, 2.0]
    assert episode_seed(state, 3) == episode_seed(state, 3, 0)
    assert episode_seed(state, 3, 1) != episode_seed(state, 3, 0)
    store = object.__new__(SupabaseStore)
    config = build_collection_config(build_initial_manifest(_history()))
    legacy = {"benchmark": "libero", "suite": "libero_goal", "task_idx": 0,
              "ep_idx": 3, "init_state_hash": "x"}
    explicit = {**legacy, "behavior_seed_index": 0}
    assert (store.rollout_id("e", legacy, Method.PCP_SEARCH_COLLECT, config)
            != store.rollout_id("e", explicit, Method.PCP_SEARCH_COLLECT, config))


def test_manifest_shards_are_disjoint_and_complete():
    items = build_initial_manifest(_history()).items
    shards = [manifest_shard(items, 4, index) for index in range(4)]
    assert [len(shard) for shard in shards] == [100, 100, 100, 100]
    assert {item.ordinal for shard in shards for item in shard} == set(range(400))


def test_generated_worker_notebooks_are_thin_and_parse():
    root = Path(__file__).parents[1]
    for index in range(4):
        path = root / "notebooks" / "workers" / f"52_pcp_search_worker_{index}.ipynb"
        document = json.loads(path.read_text())
        code_cells = ["".join(cell["source"]) for cell in document["cells"]
                      if cell["cell_type"] == "code"]
        assert len(code_cells) == 2
        assert "run_pcp_search_worker" in code_cells[-1]
        assert "ROLLOUT_BATCH_SIZE = 8" in code_cells[-1]
        assert "rollout_batch_size=ROLLOUT_BATCH_SIZE" in code_cells[-1]
        assert "SupabaseStore" not in code_cells[-1]
        for source in code_cells:
            ast.parse(source)
