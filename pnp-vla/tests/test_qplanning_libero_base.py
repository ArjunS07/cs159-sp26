import json

import numpy as np

from pnp.pcp_critic.data import DatasetSnapshot
from pnp.qplanning_critic.libero_base import build_libero_only_snapshot


def test_libero_only_snapshot_preserves_parent_split_and_excludes_pro():
    rollout_ids = tuple(f"r{index}" for index in range(8))
    parent = DatasetSnapshot(
        snapshot_id="pcpcds-parent",
        rollout_ids=rollout_ids,
        train_rollout_ids=rollout_ids[:6],
        val_rollout_ids=rollout_ids[6:],
        policy_repo_id="repo",
        policy_revision="revision",
        artifact_schema_version=1,
        action_mean=tuple(np.zeros(7)),
        action_std=tuple(np.ones(7)),
        provenance={},
    )
    libero_ids = {"r0", "r1", "r2", "r6"}
    rows = [{
        "rollout_id": rollout_id,
        "benchmark": "libero" if rollout_id in libero_ids else "libero_pro",
    } for rollout_id in rollout_ids]
    filtered = build_libero_only_snapshot(parent, rows, expected_rollouts=4)
    assert filtered.rollout_ids == ("r0", "r1", "r2", "r6")
    assert filtered.train_rollout_ids == ("r0", "r1", "r2")
    assert filtered.val_rollout_ids == ("r6",)


def test_libero_base_notebook_has_strict_4000_update_contract():
    notebook = json.loads(
        open("notebooks/73_train_q50_libero_only_base.ipynb", encoding="utf-8").read())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_q50_libero_base_training(" in source
    assert "UPDATES = 4000" in source
    assert "EXPECTED_LIBERO_ROLLOUTS = 600" in source
    assert "'libero_pro_used': False" in source
    assert "resume=True" in source
