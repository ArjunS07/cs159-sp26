import json
from pathlib import Path

from pnp.config import Method
from pnp.smolvla_tree_source_experiment import (
    SMOLVLA_TREE_SOURCE_EPISODE_INDICES,
    SMOLVLA_TREE_SOURCE_IDENTITIES,
    SMOLVLA_TREE_SOURCE_SHARDS,
    build_smolvla_tree_source_method,
)


def test_tree_source_config_is_restoration_and_training_ready():
    method, config = build_smolvla_tree_source_method()
    assert method == Method.SMOLVLA_PNP_S123_K311
    assert tuple(config.pnp_steps) == (1, 2, 3)
    assert tuple(config.pnp_k_by_step) == (3, 1, 1)
    assert config.refine and config.n_action_steps == 10
    assert config.save_time_uncertainty
    assert config.save_trajectory and config.save_generated_chunks
    assert config.save_training_data
    assert config.video == "off"
    assert SMOLVLA_TREE_SOURCE_EPISODE_INDICES == tuple(range(10, 30))
    assert SMOLVLA_TREE_SOURCE_IDENTITIES == 800
    assert SMOLVLA_TREE_SOURCE_SHARDS == 2


def test_two_tree_source_worker_notebooks_are_fixed():
    worker_dir = Path(__file__).parents[1] / "notebooks" / "workers"
    launchers = sorted(worker_dir.glob("87_smolvla_tree_source_idx10_29_worker_*.ipynb"))
    assert len(launchers) == 2
    for index, path in enumerate(launchers):
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "SHARD_COUNT = 2" in source
        assert f"SHARD_INDEX = {index}" in source
        assert "run_smolvla_tree_source_worker" in source
        assert "ROLLOUT_BATCH_SIZE = 8" in source
        assert "'integration_steps': 10" in source
        assert "'n_action_steps': 10" in source
        assert "65% uncertainty-prioritized / 35% uniform" in source
        assert "'video': 'off'" in source
