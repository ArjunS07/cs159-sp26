import json
from pathlib import Path

import pytest
from pnp.smolvla_train_action_contrast import (
    select_manifest, shard_items, EXPECTED_SNAPSHOT_DIGEST, QUOTAS)
from pnp.smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS
from pnp.smolvla_tree_bellman_finetune import _split_groups

def fixture():
    groups, items = [], []
    for n in range(800):
        key = dict(suite=("libero_spatial", "libero_object", "libero_goal", "libero_10")[n % 4],
                   task_idx=n, episode_idx=10+n % 20, chunk_idx=2)
        status = n % 3
        labels = ([False]*9 if status == 0 else [True]*9 if status == 1 else [False]+[True]*8)
        groups.append({**key, "candidate_group_id": f"root{n}",
            "source_experiment": FRESH8_EXPERIMENT,
            "source_training_data_path": f"source{n}.npz",
            "candidates": [{"candidate_kind": k, "success": y} for k,y in zip(FRESH8_KINDS,labels)]})
        items.append({**key, "source_rollout_id": f"source{n}",
                      "source_success": labels[0], "source_episode_seed": n})
    snapshot={"groups": groups + [{**g,"source_experiment": "new", "candidate_group_id": "new"+g["candidate_group_id"]} for g in groups],
              "snapshot_digest": EXPECTED_SNAPSHOT_DIGEST}
    source={"payload": {"items": items}, "manifest_hash": "source-immutable"}
    return snapshot, source, groups

def test_only_training_exact_quotas_stable_disjoint_shards():
    snapshot, source, cohort=fixture()
    result=select_manifest(snapshot,source)
    train,val=_split_groups(cohort)
    assert len(result["items"]) == 12
    assert {i["original_group_id"] for i in result["items"]} <= set(train)
    assert not {i["original_group_id"] for i in result["items"]} & set(val)
    assert result["audit"]["selected_by_stratum"] == QUOTAS
    reversed_snapshot={**snapshot,"groups":list(reversed(snapshot["groups"]))}
    reversed_source={**source,"payload":{"items":list(reversed(source["payload"]["items"]))}}
    assert result == select_manifest(reversed_snapshot,reversed_source)
    shards=[{i["source_rollout_id"] for i in shard_items(result,s,3)} for s in range(3)]
    assert all(len(s)==4 for s in shards)
    assert len(set.union(*shards))==12
    assert not shards[0] & shards[1]
    assert len(shard_items(result,0,3,1)) == 1

def test_invalid_shard_and_mismatched_source_outcome_refused():
    for index,count,limit in ((3,3,None),(0,0,None),(0,3,0)):
        with pytest.raises(ValueError):
            shard_items({"items":[]},index,count,limit)
    snapshot,source,cohort=fixture()
    train,_=_split_groups(cohort)
    key=next(g for g in cohort if g["candidate_group_id"] in train)
    item=next(i for i in source["payload"]["items"] if i["task_idx"]==key["task_idx"])
    item["source_success"]=not item["source_success"]
    with pytest.raises(ValueError,match="stored-source outcome"):
        select_manifest(snapshot,source)

def test_notebook_is_thin_explicit_opt_in_and_compiles():
    nb=json.loads((Path(__file__).parents[1]/"notebooks/workers/123_smolvla_training_action_contrast.ipynb").read_text())
    sources="\n".join("".join(c["source"]) for c in nb["cells"])
    assert "RUN_COLLECTION = False" in sources
    assert "SHARD_COUNT = 3" in sources
    assert "run_training_action_contrast" in sources
    for c in nb["cells"]:
        if c["cell_type"]=="code":
            compile("".join(c["source"]),"notebook123","exec")

