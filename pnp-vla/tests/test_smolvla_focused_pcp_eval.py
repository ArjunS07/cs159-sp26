import json
from pathlib import Path

import pytest

from pnp import smolvla_tree_collection as tree
from pnp.smolvla_focused_pcp_eval import focused_manifest, EXPECTED_CHECKPOINT_SHA256
from pnp.smolvla_pcp_pilot import KINDS, complete_candidate_contract
from pnp.smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS
from pnp.smolvla_tree_bellman_finetune import _split_groups


def data():
    groups, items = [], []
    for n in range(100):
        key = dict(suite="libero_spatial", task_idx=n, episode_idx=10 + n % 20, chunk_idx=2)
        groups.append({**key, "candidate_group_id": f"root{n}", "source_experiment": FRESH8_EXPERIMENT,
            "candidates": [{"candidate_kind": kind, "success": (n % 2 == 0 if i == 0 else n % 2 != 0)}
                           for i, kind in enumerate(FRESH8_KINDS)]})
        items.append({**key, "source_rollout_id": f"source{n}", "source_success": n % 2 == 0})
    return {"groups": groups, "snapshot_digest": "frozen"}, items


def test_original_validation_only_prior_sources_excluded_and_no_duplicates():
    snapshot, items = data()
    _, validation = _split_groups(snapshot["groups"])
    exclusion = [items[int(validation[0][4:])]["source_rollout_id"]]
    result = focused_manifest(snapshot, items, exclusion)
    selected = result["items"]
    assert len(selected) == len(validation) - 1
    assert {i["original_group_id"] for i in selected} <= set(validation)
    assert not {i["source_rollout_id"] for i in selected} & set(exclusion)
    assert len({i["source_rollout_id"] for i in selected}) == len(selected)
    assert all(i["original_split"] == "validation" for i in selected)
    assert result["audit"]["target_shortfall"] == 48 - len(selected)
    balances = result["audit"]["selected_historical_stock_outcomes"]
    assert abs(balances["failure"] - balances["success"]) <= 1


def test_manifest_stable_across_input_order_and_shards_disjoint_complete():
    snapshot, items = data()
    first = focused_manifest(snapshot, items, [], target_roots=12)
    second = focused_manifest({**snapshot, "groups": list(reversed(snapshot["groups"]))}, list(reversed(items)), [], target_roots=12)
    assert first == second
    payload = {k: v for k, v in first.items() if k != "manifest_hash"}
    assert tree._digest(payload) == first["manifest_hash"]
    shards = [{i["source_rollout_id"] for i in first["items"] if i["ordinal"] % 3 == s} for s in range(3)]
    assert not shards[0] & shards[1] and not shards[0] & shards[2] and not shards[1] & shards[2]
    assert len(set.union(*shards)) == len(first["items"])
    changed = focused_manifest(snapshot, items, [], target_roots=12, checkpoint_sha="a" * 64)
    assert changed["manifest_hash"] != first["manifest_hash"]


def test_nonmixed_and_new_cohort_do_not_leak_into_selection():
    snapshot, items = data()
    _, validation = _split_groups(snapshot["groups"])
    excluded_group = next(g for g in snapshot["groups"] if g["candidate_group_id"] == validation[0])
    for candidate in excluded_group["candidates"]:
        candidate["success"] = excluded_group["candidates"][0]["success"]
    snapshot["groups"].append({**snapshot["groups"][0], "candidate_group_id": "new-root", "source_experiment": "new_cohort"})
    result = focused_manifest(snapshot, items, [])
    assert excluded_group["candidate_group_id"] not in {i["original_group_id"] for i in result["items"]}
    assert "new-root" not in {i["original_group_id"] for i in result["items"]}


def test_resume_requires_all_nine_artifacts_and_exact_critic():
    candidates = [{"candidate_kind": k, "metadata_json": {"training_data_path": "artifact", "critic_sha256": EXPECTED_CHECKPOINT_SHA256}} for k in KINDS]
    assert complete_candidate_contract(candidates, expected_critic_sha=EXPECTED_CHECKPOINT_SHA256)
    assert not complete_candidate_contract(candidates[:-1], expected_critic_sha=EXPECTED_CHECKPOINT_SHA256)
    candidates[0]["metadata_json"]["critic_sha256"] = "other"
    assert not complete_candidate_contract(candidates, expected_critic_sha=EXPECTED_CHECKPOINT_SHA256)
    candidates[0]["metadata_json"]["critic_sha256"] = EXPECTED_CHECKPOINT_SHA256
    candidates[0]["metadata_json"].pop("training_data_path")
    assert not complete_candidate_contract(candidates, expected_critic_sha=EXPECTED_CHECKPOINT_SHA256)


def test_notebook_defaults_to_cpu_dry_run_with_required_digest():
    path = Path(__file__).parents[1] / "notebooks/workers/122_smolvla_focused_flow_pcp_eval.ipynb"
    nb = json.loads(path.read_text())
    source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "RUN_INTERVENTION = False" in source and EXPECTED_CHECKPOINT_SHA256 in source
    assert "SHARD_COUNT = 3" in source and "TARGET_ROOTS = 48" in source
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "notebook122", "exec")


def test_checkpoint_sha_cannot_be_disabled_before_any_store_access():
    from pnp.smolvla_focused_pcp_eval import run_focused_flow_pcp_eval
    for bad in (None, "", "x" * 64):
        with pytest.raises(ValueError, match="SHA256 is required"):
            run_focused_flow_pcp_eval(checkpoint_path="unused.pt", expected_checkpoint_sha=bad)
