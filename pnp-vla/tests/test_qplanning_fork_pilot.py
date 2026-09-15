import ast
from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from pnp.pcp_search.pro import PRO_TEN_STATE_SUITES, PRO_TRAIN_QUOTAS
from pnp.qplanning_fork_pilot import (
    FORK_PILOT_CANDIDATES,
    FORK_PILOT_MANIFEST_PATH,
    FORK_PILOT_SHARDS,
    FORK_PILOT_SKIP_UNUSED_RENDERS,
    FORK_PILOT_STRATEGIES,
    FORK_PILOT_TRAIN_PRIORITY_FRACTION,
    FORK_PILOT_TREES_PER_STRATEGY,
    FORK_PILOT_VERSION,
    FORK_PILOT_U20_BOUNDARIES,
    FORK_PILOT_RENDER_LEAD,
    _SOURCE_FIDELITY_ARRAYS,
    _load_selected_training_arrays,
    _source_boundary,
    _trajectory_actions_from_payload,
    _validate_manifest,
    build_fixed_fork_manifest,
    u20_profile_from_ahats,
)
from pnp.store import _training_data_payloads


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


def test_source_fidelity_loader_selects_arrays_from_multipart_artifact():
    rng = np.random.default_rng(3)
    arrays = {
        "initial_state": rng.normal(size=12),
        "episode_seed": np.asarray(11, np.int64),
        "actions_env": rng.normal(size=(20, 7)).astype(np.float32),
        "sim_state_t_plus_1": rng.normal(size=(21, 18)),
        "chunk_start_steps": np.asarray([0, 10], np.int32),
        "chunk_noise_seeds": np.asarray([4, 5, 6], np.int64),
        "bellman/action": rng.normal(size=(2, 50, 7)).astype(np.float32),
        "boundary/step": np.asarray([0, 10, 20], np.int32),
        "boundary/raw_agentview": rng.integers(
            0, 256, size=(3, 32, 32, 3), dtype=np.uint8),
        "boundary/raw_wrist": rng.integers(
            0, 256, size=(3, 32, 32, 3), dtype=np.uint8),
        "boundary/raw_robot_state": rng.normal(size=(3, 9)).astype(np.float32),
        "boundary/policy_proprio": rng.normal(size=(3, 8)).astype(np.float32),
        "boundary/sim_state": rng.normal(size=(3, 18)),
    }
    manifest_path, payloads = _training_data_payloads(
        "source", arrays, max_part_bytes=16384)
    assert manifest_path.endswith("manifest.json")

    class Store:
        def __init__(self):
            self.payloads = dict(payloads)

        def _download(self, path):
            return self.payloads[path]

    loaded = _load_selected_training_arrays(Store(), manifest_path)
    assert set(loaded) == set(_SOURCE_FIDELITY_ARRAYS)
    for name in _SOURCE_FIDELITY_ARRAYS:
        np.testing.assert_array_equal(loaded[name], arrays[name])


def test_source_boundary_uses_persisted_boundary_index_and_ten_action_stride():
    arrays = {
        "chunk_start_steps": np.asarray([0, 10, 20], np.int32),
        "boundary/step": np.asarray([0, 10, 20, 27], np.int32),
        "sim_state_t_plus_1": np.arange(28 * 4).reshape(28, 4),
        "boundary/raw_agentview": np.zeros((4, 2, 2, 3), np.uint8),
        "boundary/raw_wrist": np.zeros((4, 2, 2, 3), np.uint8),
        "boundary/raw_robot_state": np.zeros((4, 9), np.float32),
        "boundary/policy_proprio": np.zeros((4, 8), np.float32),
        "boundary/sim_state": np.asarray([
            np.arange(0, 4), np.arange(40, 44),
            np.arange(80, 84), np.arange(108, 112)]),
        "bellman/action": np.zeros((4, 50, 7), np.float32),
        "chunk_noise_seeds": np.asarray([1, 2, 3, 4], np.int64),
    }
    source = _source_boundary(
        {"arrays": arrays}, {"chunk_idx": 2, "source_rollout_id": "r"})
    assert source["root_step"] == 20
    assert source["boundary_index"] == 2
    np.testing.assert_array_equal(source["sim_state"], arrays["sim_state_t_plus_1"][20])
    np.testing.assert_array_equal(source["boundary_sim_state"], source["sim_state"])


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
    assert payload["skip_unused_renders"] is FORK_PILOT_SKIP_UNUSED_RENDERS is True
    assert payload["render_lead"] == FORK_PILOT_RENDER_LEAD == 2
    assert "persisted source simulator states, policy inputs, and actions" in payload[
        "source_scope"]
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
    assert "run_fork_source_fidelity_preflight" in preflight
    assert "source_fidelity_videos" in preflight
    assert "DO NOT launch/resume fork collectors" in preflight
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


def test_corrected_pilot_uses_fresh_v5_namespace():
    assert FORK_PILOT_VERSION == 5
    assert FORK_PILOT_MANIFEST_PATH.endswith("fixed_three_priority_v5.json")


def test_worker_assigns_collector_return_directly_not_as_singleton_tuple():
    source = (ROOT / "pnp" / "qplanning_fork_pilot.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "result"
                for target in node.targets)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "collect_replay_candidate_group"
    ]
    singleton_wrapped = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "result"
                for target in node.targets)
        and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 1
        and isinstance(node.value.elts[0], ast.Call)
        and getattr(node.value.elts[0].func, "id", None)
        == "collect_replay_candidate_group"
    ]
    assert direct
    assert not singleton_wrapped
