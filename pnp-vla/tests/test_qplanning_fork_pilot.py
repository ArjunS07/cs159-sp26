import ast
from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from pnp.pcp_search.pro import PRO_TEN_STATE_SUITES, PRO_TRAIN_QUOTAS
from pnp.qplanning_fork_pilot import (
    FORK_PILOT_CANDIDATES,
    FORK_PILOT_SHARDS,
    FORK_PILOT_STRATEGIES,
    FORK_PILOT_TRAIN_PRIORITY_FRACTION,
    FORK_PILOT_TREES_PER_STRATEGY,
    FORK_PILOT_U20_BOUNDARIES,
    _trajectory_actions_from_payload,
    _validate_manifest,
    build_fixed_fork_manifest,
    u20_profile_from_ahats,
)


ROOT = Path(__file__).parents[1]


def test_u20_profile_uses_first_twenty_action_positions_and_probe_mean():
    payload = BytesIO()
    np.savez_compressed(
        payload,
        c0_s3_u_time=np.arange(1, 51, dtype=np.float32),
        c0_s4_u_time=np.arange(3, 53, dtype=np.float32),
        c1_s3_u_time=np.full(50, 2.0, np.float32),
        c1_s4_u_time=np.full(50, 4.0, np.float32),
    )
    profile = u20_profile_from_ahats(payload.getvalue())
    assert profile == (11.5, 3.0)


def test_compact_source_trajectory_actions_are_lossless():
    payload = BytesIO()
    actions = np.arange(140, dtype=np.float32).reshape(20, 7)
    np.savez_compressed(payload, actions=actions, robot_state=np.zeros((20, 8)))
    loaded = _trajectory_actions_from_payload(payload.getvalue())
    np.testing.assert_array_equal(loaded, actions)


def _source_fixture():
    suites = sorted(set(PRO_TRAIN_QUOTAS) - set(PRO_TEN_STATE_SUITES))
    rows, profiles = [], {}
    for suite_index, suite in enumerate(suites):
        for index in range(40):
            rollout_id = f"r-{suite_index}-{index}"
            rows.append({
                "rollout_id": rollout_id,
                "benchmark": "libero_pro",
                "suite": suite,
                "task_idx": index % 10,
                "episode_idx": index // 10,
                "init_state_hash": f"h-{suite_index}-{index}",
                "success": index % 3 != 0,
            })
            profiles[rollout_id] = tuple(
                .01 + .0001 * suite_index + .0002 * step + .00001 * index
                for step in range(8))
    return rows, profiles


def test_fixed_manifest_has_equal_tree_budgets_and_balanced_shards():
    rows, profiles = _source_fixture()
    first = build_fixed_fork_manifest(rows, profiles)
    second = build_fixed_fork_manifest(rows, profiles)
    assert first == second
    _validate_manifest(first)
    payload = first["payload"]
    assert payload["candidate_count"] == FORK_PILOT_CANDIDATES == 9
    assert payload["future_training_priority_fraction"] == 0.65
    assert payload["u20_boundaries"] == FORK_PILOT_U20_BOUNDARIES == 3
    assert "exact stored source-trajectory" in payload["source_scope"]
    counts = Counter(item["strategy"] for item in payload["trees"])
    assert counts == Counter({strategy: FORK_PILOT_TREES_PER_STRATEGY
                              for strategy in FORK_PILOT_STRATEGIES})
    shard_counts = Counter((item["strategy"], item["shard_index"])
                           for item in payload["trees"])
    assert set(shard_counts.values()) == {
        FORK_PILOT_TREES_PER_STRATEGY // FORK_PILOT_SHARDS}
    assert not ({item["suite"] for item in payload["trees"]}
                & set(PRO_TEN_STATE_SUITES))
    assert all(item["chunk_idx"] + FORK_PILOT_U20_BOUNDARIES
               <= item["chunk_count"] for item in payload["trees"])


def test_fork_notebooks_are_clean_and_encode_the_frozen_contract():
    paths = [ROOT / "notebooks" / "77_qplanning_fork_pilot_preflight.ipynb"]
    paths += [
        ROOT / "notebooks" / "workers"
        / f"78_collect_qplanning_fork_pilot_worker_{index}.ipynb"
        for index in range(4)]
    paths += [ROOT / "notebooks" / "79_analyze_qplanning_fork_pilot.ipynb"]
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "50-action" not in source or "execute" not in source.lower()
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                ast.parse("".join(cell["source"]), filename=f"{path.name}:{cell_index}")
    preflight = paths[0].read_text(encoding="utf-8")
    assert "run_fork_restoration_preflight" in preflight
    for index, path in enumerate(paths[1:5]):
        source = path.read_text(encoding="utf-8")
        assert f"SHARD_INDEX = {index}" in source
        assert "TREE_LIMIT_PER_STRATEGY = None" in source
        assert "branch_outcomes_in_full_worker" in source
    analysis = paths[-1].read_text(encoding="utf-8")
    assert "mixed_outcome_trees_pct" in analysis
    assert "oracle_gain_over_stock_pp" in analysis


def test_priority_fraction_is_explicitly_sixty_five_percent():
    assert FORK_PILOT_TRAIN_PRIORITY_FRACTION == .65
