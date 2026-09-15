import ast
from collections import Counter
import json
from pathlib import Path

from pnp.pcp_search.pro import PRO_TEN_STATE_SUITES, PRO_TRAIN_QUOTAS
from pnp.qplanning_fork_pilot import (
    FORK_PILOT_CANDIDATES,
    FORK_PILOT_SHARDS,
    FORK_PILOT_STRATEGIES,
    FORK_PILOT_TREES_PER_STRATEGY,
    build_fixed_fork_manifest,
)
from pnp.qplanning_fork20_pilot import (
    FORK20_INTERVENTION_ACTIONS,
    FORK20_REPLAN_ACTIONS,
    FORK20_REQUIRED_BOUNDARIES,
    _validate_fork20_manifest,
    build_fixed_fork20_manifest,
)


ROOT = Path(__file__).parents[1]


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


def test_fork20_manifest_is_four_boundary_and_maximally_paired():
    rows, profiles = _source_fixture()
    reference = build_fixed_fork_manifest(rows, profiles)
    document = build_fixed_fork20_manifest(
        rows, profiles, reference_trees=reference["payload"]["trees"],
        reference_manifest_hash=reference["manifest_hash"])
    _validate_fork20_manifest(document)
    payload = document["payload"]
    assert payload["candidate_intervention_actions"] == FORK20_INTERVENTION_ACTIONS == 20
    assert payload["continuation_replan_actions"] == FORK20_REPLAN_ACTIONS == 10
    assert payload["required_complete_boundaries"] == FORK20_REQUIRED_BOUNDARIES == 4
    assert payload["candidate_count"] == FORK_PILOT_CANDIDATES == 9
    assert Counter(item["strategy"] for item in payload["trees"]) == Counter({
        strategy: FORK_PILOT_TREES_PER_STRATEGY
        for strategy in FORK_PILOT_STRATEGIES})
    assert all(item["chunk_idx"] + FORK20_REQUIRED_BOUNDARIES
               <= item["chunk_count"] for item in payload["trees"])
    reference_keys = {
        (item["strategy"], item["source_rollout_id"], item["chunk_idx"])
        for item in reference["payload"]["trees"]}
    paired = [item for item in payload["trees"]
              if item["paired_with_10action_tree"]]
    assert paired
    assert all((item["strategy"], item["source_rollout_id"], item["chunk_idx"])
               in reference_keys for item in paired)
    shard_counts = Counter((item["strategy"], item["shard_index"])
                           for item in payload["trees"])
    assert set(shard_counts.values()) == {
        FORK_PILOT_TREES_PER_STRATEGY // FORK_PILOT_SHARDS}


def test_fork20_notebooks_are_clean_and_encode_action_cadences():
    paths = [ROOT / "notebooks" / "80_qplanning_fork20_pilot_preflight.ipynb"]
    paths += [
        ROOT / "notebooks" / "workers"
        / f"81_collect_qplanning_fork20_pilot_worker_{index}.ipynb"
        for index in range(4)]
    paths += [ROOT / "notebooks" / "82_analyze_qplanning_fork20_pilot.ipynb"]
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                ast.parse("".join(cell["source"]), filename=f"{path.name}:{cell_index}")
        assert "20" in source
    preflight = paths[0].read_text(encoding="utf-8")
    assert "run_fork20_restoration_preflight" in preflight
    for index, path in enumerate(paths[1:5]):
        source = path.read_text(encoding="utf-8")
        assert f"SHARD_INDEX = {index}" in source
        assert "TREE_LIMIT_PER_STRATEGY = None" in source
        assert "continuation_replan_actions" in source
    analysis = paths[-1].read_text(encoding="utf-8")
    assert "paired_with_10action_tree" in analysis
    assert "mixed_20action_pct" in analysis
