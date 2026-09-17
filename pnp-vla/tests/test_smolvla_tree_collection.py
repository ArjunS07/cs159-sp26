import io
import json
from pathlib import Path

import numpy as np

from pnp.smolvla_tree_collection import (
    SMOLVLA_TREE_CANDIDATES,
    SMOLVLA_TREE_COLLECTION_VERSION,
    SMOLVLA_TREE_COUNT,
    SMOLVLA_TREE_EXPERIMENT,
    SMOLVLA_TREE_FRESH_CANDIDATES,
    SMOLVLA_TREE_PERTURB_CANDIDATES,
    SMOLVLA_TREE_SHARDS,
    _balanced_priority_ids,
    _eligible_roots,
    _pnp_config,
    weighted_u10_profile,
)


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def test_weighted_u10_profile_uses_311_pair_weights():
    arrays = {}
    for chunk in range(2):
        for step, value in ((1, 1 + chunk), (2, 6 + chunk), (3, 11 + chunk)):
            arrays[f"c{chunk}_s{step}_u_time"] = np.full(50, value, np.float32)
    profile = weighted_u10_profile(_npz(**arrays))
    assert profile == ((3 * 1 + 6 + 11) / 5, (3 * 2 + 7 + 12) / 5)


def test_root_selection_prefers_three_boundaries_but_has_short_fallback():
    eligible, fallback = _eligible_roots(tuple(range(8)))
    assert eligible == [1, 2, 3, 4, 5]
    assert not fallback
    eligible, fallback = _eligible_roots((0.1, 0.2))
    assert eligible == [0]
    assert fallback


def test_priority_assignment_has_exact_requested_count_and_stratifies():
    rows = [
        {"rollout_id": f"r{index}", "suite": f"suite{index % 4}",
         "success": bool((index // 4) % 2)}
        for index in range(80)
    ]
    selected = _balanced_priority_ids(rows, 52)
    assert len(selected) == 52
    represented = {
        (row["suite"], row["success"]) for row in rows if row["rollout_id"] in selected}
    assert len(represented) == 8


def test_tree_contract_and_worker_notebooks():
    assert SMOLVLA_TREE_COLLECTION_VERSION == 2
    assert SMOLVLA_TREE_EXPERIMENT.endswith("v2-egl")
    assert SMOLVLA_TREE_COUNT == 800
    assert SMOLVLA_TREE_SHARDS == 2
    assert SMOLVLA_TREE_CANDIDATES == 9
    assert 1 + SMOLVLA_TREE_FRESH_CANDIDATES + SMOLVLA_TREE_PERTURB_CANDIDATES == 9
    config = _pnp_config()
    assert tuple(config.pnp_steps) == (1, 2, 3)
    assert tuple(config.pnp_k_by_step) == (3, 1, 1)
    assert config.refine and config.n_action_steps == 10

    worker_dir = Path(__file__).parents[1] / "notebooks" / "workers"
    notebooks = sorted(worker_dir.glob("88_smolvla_depth1_tree_worker_*.ipynb"))
    assert len(notebooks) == 2
    for index, path in enumerate(notebooks):
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "SHARD_COUNT = 2" in source
        assert f"SHARD_INDEX = {index}" in source
        assert "TREE_LIMIT = None" in source
        assert "run_smolvla_tree_worker" in source
        assert "1 stored source + 4 fresh seed + 4 P&P perturbation" in source
        assert "libEGL_nvidia" in source
