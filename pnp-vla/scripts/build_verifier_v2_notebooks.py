"""Build the V2 verifier collection, architecture sweep, and confirmation notebooks."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def notebook(cells, name):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": name},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


BOOTSTRAP = r'''import os, subprocess, sys
try:
    from google.colab import userdata
    for key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "HF_TOKEN", "WANDB_API_KEY"):
        value = userdata.get(key)
        if value: os.environ[key] = value
    repo_dir = "/content/cs159-sp26"
    gh_pat = userdata.get("GH_PAT")
    repo_url = f"https://{gh_pat}@github.com/ArjunS07/cs159-sp26.git"
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        subprocess.run(["git", "clone", "--branch", "main", repo_url, repo_dir], check=True)
    else:
        subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only", "origin", "main"], check=True)
except ImportError:
    repo_dir = os.path.abspath("..") if os.path.basename(os.getcwd()) == "pnp-vla" else os.getcwd()
package_dir = os.path.join(repo_dir, "pnp-vla")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                package_dir + "[sim,analysis]"], check=True)
if package_dir not in sys.path: sys.path.insert(0, package_dir)
import pnp
print("Loaded pnp from:", pnp.__file__)'''


PAGES = r'''def pages(table, columns, configure=lambda q: q, order_by=()):
    rows=[]; start=0
    while True:
        query = configure(store.client.table(table).select(columns))
        for column in order_by: query = query.order(column)
        batch = query.range(start, start+999).execute().data or []
        rows += batch
        if len(batch) < 1000: return rows
        start += 1000'''


COLLECTION = notebook([
    md("""# 08 — Trajectory-seeded LIBERO-PRO verifier collection

Collect 240 development and 160 blinded confirmatory groups with 12 clean
action candidates each. Apply `supabase/migrations/003_verifier_v2.sql` first.
Use the generated three-worker notebooks for the planned 48 GPU-hour run."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Load the simulator, policy, and canonical PRO identities"),
    code(r'''import hashlib, json
from collections import defaultdict
from tqdm.auto import tqdm
from pnp import libero_env, libero_pro, models
from pnp.experiments import _prepare_libero_pro_episodes
from pnp.store import SupabaseStore
from pnp.verifier import *

libero_pro.patch_torch_load()
policy, preprocess, postprocess = models.load_pi05()
device = models.default_device(); store = SupabaseStore()
episodes = _prepare_libero_pro_episodes()
for ep in episodes: ep["benchmark"] = "libero_pro"
episode_lookup = {(ep["suite"], ep["task_idx"], ep.get("ep_idx", ep.get("episode_idx"))): ep
                  for ep in episodes}
print({"PRO identities": len(episode_lookup), "device": str(device)})'''),
    md("## 3. Freeze and upload the outcome-blind seeded manifest"),
    code(PAGES + r'''

DEVELOPMENT_EXPERIMENT = "verifier-v2-pro-development"
TEST_EXPERIMENT = "verifier-v2-pro-confirmatory"
CANDIDATE_COUNT = 12
PREFIX_LENGTH = 10
SHARD_COUNT = 1   # Generated workers set this to 3.
SHARD_INDEX = 0
assert 0 <= SHARD_INDEX < SHARD_COUNT

rollouts = pages(
    "rollouts", "rollout_id,benchmark,suite,task_idx,episode_idx,success",
    lambda q: q.eq("experiment", "libero-pro-canonical-core-k3-v1").eq(
        "method", "pnp_uncertainty_only").eq("status", "completed"),
    ("rollout_id",))
rollout_ids = sorted(row["rollout_id"] for row in rollouts)
euler=[]
for start in range(0, len(rollout_ids), 100):
    ids=rollout_ids[start:start+100]
    euler += pages(
        "pnp_euler_steps", "rollout_id,chunk_idx,euler_step,u_mean",
        lambda q, ids=ids: q.in_("rollout_id", ids),
        ("rollout_id", "chunk_idx", "euler_step"))
by_rollout={row["rollout_id"]: row for row in rollouts}
uncertainty=defaultdict(list)
for row in euler:
    uncertainty[(row["rollout_id"], int(row["chunk_idx"]))].append(float(row["u_mean"]))
source=[]
for (rollout_id, chunk_idx), values in uncertainty.items():
    if rollout_id in by_rollout:
        source.append({**by_rollout[rollout_id], "chunk_idx": chunk_idx,
                       "u_mean": sum(values)/len(values), "uncertainty_stratum": "high"})
manifests = build_seeded_pro_manifest(
    source, development_target=240, test_target=160, seed=20260728)
hashes={name: collection_manifest_hash(rows) for name, rows in manifests.items()}
document={"version": 3, "candidate_count": CANDIDATE_COUNT,
          "prefix_length": PREFIX_LENGTH, "hashes": hashes, "manifests": manifests}
manifest_path=f"verifier_manifests/verifier-v2-pro-{hashes['development']}-{hashes['confirmatory_test']}.json"
store._upload(manifest_path, json.dumps(document, sort_keys=True,
                                       separators=(",", ":")).encode())
full_manifest=[]
for split, experiment in (("development", DEVELOPMENT_EXPERIMENT),
                          ("confirmatory_test", TEST_EXPERIMENT)):
    full_manifest += [{**row, "experiment": experiment} for row in manifests[split]]
manifest=full_manifest[SHARD_INDEX::SHARD_COUNT]
assert len(full_manifest) == 400
assert len({(r["suite"],r["task_idx"],r["episode_idx"],r["trajectory_seed"])
            for r in full_manifest}) == 400
print({"development": 240, "confirmatory_test": 160, "worker": len(manifest),
       "hashes": hashes, "manifest_path": manifest_path})'''),
    md("## 4. Resume-safe sharded collection"),
    code(r'''experiments=(DEVELOPMENT_EXPERIMENT, TEST_EXPERIMENT)
existing_rows=pages(
    "verifier_candidate_groups", "candidate_group_id,experiment",
    lambda q: q.in_("experiment", experiments), ("candidate_group_id",))
existing_ids={row["candidate_group_id"] for row in existing_rows}
candidate_rows=[]
for start in range(0, len(existing_ids), 100):
    ids=sorted(existing_ids)[start:start+100]
    candidate_rows += pages(
        "verifier_candidates", "candidate_id,candidate_group_id",
        lambda q, ids=ids: q.in_("candidate_group_id", ids), ("candidate_id",))
counts=defaultdict(int)
for row in candidate_rows: counts[row["candidate_group_id"]] += 1
complete={gid for gid in existing_ids if counts[gid] == CANDIDATE_COUNT}
print({"complete_existing_groups": len(complete),
       "partial_groups_to_repair": len(existing_ids-complete)})

store.start_run("verifier_pair_collection", "libero_pro", "verifier-v2-pro",
                config={"candidate_count": CANDIDATE_COUNT, "groups": 400,
                        "manifest_hashes": hashes, "shard_count": SHARD_COUNT,
                        "shard_index": SHARD_INDEX})
new_outcomes=skipped=0
for item in tqdm(manifest, desc="V2 PRO groups"):
    expected=candidate_group_id(
        "libero_pro", item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=item["experiment"],
        trajectory_seed=item["trajectory_seed"])
    if expected in complete: continue
    ep=episode_lookup[(item["suite"], item["task_idx"], item["episode_idx"])]
    env=libero_env.make_env(ep["bddl_path"])
    try:
        try:
            result=collect_replay_candidate_group(
                env, ep, policy, preprocess, postprocess, device,
                chunk_idx=item["chunk_idx"], uncertainty_stratum="high",
                prefix_length=PREFIX_LENGTH, candidate_count=CANDIDATE_COUNT,
                experiment=item["experiment"], trajectory_seed=item["trajectory_seed"],
                collection_split=item["collection_split"],
                manifest_hash=hashes[item["collection_split"]], model_revision="pi05")
        except Exception as error:
            print("group skipped:", type(error).__name__, error); result=None
        if result is None: skipped += 1; continue
        group, candidates=result
        group["metadata_json"].update({
            "collection_manifest_path": manifest_path,
            "source_rollout_id": item["rollout_id"],
            "source_success": bool(item["success"]),
            "source_u_mean": float(item["u_mean"])})
        store.register_candidate_group(group, candidates)
        complete.add(group["candidate_group_id"]); new_outcomes += len(candidates)
    finally: env.close()
store.finish_run(n_rollouts=new_outcomes)
print({"new_outcomes": new_outcomes, "skipped": skipped,
       "complete_groups_seen": len(complete)})'''),
    md("## 5. Integrity report (confirmatory labels remain sealed)"),
    code(r'''for split, experiment, target in (
    ("development", DEVELOPMENT_EXPERIMENT, 240),
    ("confirmatory_test", TEST_EXPERIMENT, 160)):
    groups=pages("verifier_candidate_groups", "candidate_group_id",
                 lambda q, e=experiment: q.eq("experiment", e), ("candidate_group_id",))
    ids={row["candidate_group_id"] for row in groups}; candidates=[]
    for start in range(0, len(ids), 100):
        batch=sorted(ids)[start:start+100]
        candidates += pages("verifier_candidates", "candidate_group_id,success",
                            lambda q, ids=batch: q.in_("candidate_group_id", ids))
    counts=defaultdict(int)
    for row in candidates: counts[row["candidate_group_id"]] += 1
    report={"target": target, "groups": len(ids),
            "complete_groups": sum(counts[gid] == CANDIDATE_COUNT for gid in ids),
            "partial_groups": sum(0 < counts[gid] < CANDIDATE_COUNT for gid in ids)}
    if split == "development":
        outcomes=defaultdict(set)
        for row in candidates: outcomes[row["candidate_group_id"]].add(bool(row["success"]))
        report["discordant_groups"] = sum(len(values)==2 for values in outcomes.values())
    print(split, report)'''),
], "08_collect_verifier_v2_pro.ipynb")


TRAINING = notebook([
    md("""# 09 — State-conditioned verifier V2 architecture sweep

This notebook fixes best-epoch restoration, compares multiplicative, FiLM,
cross-attention, action-only, and deranged-action controls, and registers one
checkpoint bundle. It never loads confirmatory candidate outcomes."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Configuration and data"),
    code(r'''from dataclasses import replace
from pathlib import Path
import copy, json, pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from pnp.store import SupabaseStore
from pnp.verifier import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT=Path(package_dir)/"analysis_outputs"/"verifier_v2"; OUTPUT.mkdir(parents=True, exist_ok=True)
store=SupabaseStore()
HISTORICAL=("libero-hybrid-schedules-k3-v1", "libero-pro-canonical-core-k3-v1")
PRIMARY=("verifier-clean-pairs-v3", "verifier-clean-pairs-v4-dev",
         "verifier-clean-pairs-v4-test", "verifier-online-selection-v1",
         "verifier-v2-pro-development")
CONFIRMATORY="verifier-v2-pro-confirmatory"
SEEDS=(42,43,44); N_FOLDS=4; PREFIX_LENGTH=10

historical=load_clean_chunk_examples(
    store, HISTORICAL, progress=tqdm, cache_dir=OUTPUT/"historical_cache")
development=load_candidate_examples(store, PRIMARY, cache_dir=OUTPUT/"candidate_cache")
audit=validate_candidate_groups(development)
assert audit["discordant_groups"] >= 100, audit
# Seal IDs only. Do not query verifier_candidates for this experiment here.
sealed=(store.client.table("verifier_candidate_groups").select("candidate_group_id")
        .eq("experiment", CONFIRMATORY).execute().data or [])
assert len(sealed) >= 120, len(sealed)
folds=candidate_cv_splits(development, [e.rollout_id for e in development], N_FOLDS, 42)
print({"historical_chunks":len(historical), "development":audit,
       "sealed_confirmatory_groups":len(sealed), "device":str(DEVICE)})'''),
    md("## 3. Fixed sweep runner"),
    code(r'''def historical_fold(candidate_validation, fold_index):
    protected=candidate_episode_identities(candidate_validation)
    clean=exclude_episode_identities(historical, protected)
    split=known_task_split(clean, seed=420+fold_index)
    val=select_examples(clean, split["val"]); val_ids=set(split["val"])
    return [e for e in clean if e.rollout_id not in val_ids], val

VALUE_CACHE={}
def pretrained_state(fold_index,seed,dropout):
    key=(fold_index,seed,dropout)
    if key not in VALUE_CACHE:
        val=select_examples(development,folds[fold_index]["val"])
        value_train,value_val=historical_fold(val,fold_index)
        cfg=AdvantageTrainConfig(seed=seed,prefix_length=PREFIX_LENGTH,
            value_epochs=60,patience=7,weight_decay=1e-2)
        base=CompactAdvantageVerifier(action_width=64,dropout=dropout)
        base,meta=pretrain_value(base,value_train,value_val,DEVICE,config=cfg)
        VALUE_CACHE[key]=(copy.deepcopy(base.state_dict()),meta)
    return VALUE_CACHE[key]

def fit_one(spec, fold_index, seed, max_epochs, patience):
    fold=folds[fold_index]
    train=select_examples(development, fold["train"])
    val=select_examples(development, fold["val"])
    config=AdvantageTrainConfig(
        seed=seed,prefix_length=PREFIX_LENGTH,value_epochs=60,rank_epochs=max_epochs,
        patience=patience,rank_lr=spec["rank_lr"],weight_decay=1e-2,
        context_lr_multiplier=.1,zero_context=spec.get("action_only",False))
    model=CompactAdvantageVerifier(action_width=64,dropout=spec["dropout"],
                                   conditioning=spec["architecture"])
    state,value_meta=pretrained_state(fold_index,seed,spec["dropout"])
    model.load_state_dict(state)
    if spec.get("shuffle"):
        train=shuffle_candidate_actions_within_group(train, seed)
    model, rank_meta=train_advantage(model,train,val,DEVICE,config=config)
    metrics,records=evaluate_candidate_ranker(
        model,val,DEVICE,config=config,n_bootstrap=1000,return_records=True)
    return model,config,{**value_meta,**rank_meta},metrics,records

ARCHITECTURES=("multiplicative","film","cross_attention")
STAGE1=[{"name":f"{a}-d{d}-lr{lr}","architecture":a,"dropout":d,"rank_lr":lr}
        for a in ARCHITECTURES for d in (.2,.4) for lr in (1e-4,3e-4)]
ACTION={"name":"action-only","architecture":"action_only","dropout":.2,
        "rank_lr":3e-4,"action_only":True}
SHUFFLED={"name":"shuffled-actions","architecture":"film","dropout":.2,
          "rank_lr":3e-4,"shuffle":True}
print({"stage1_configs":len(STAGE1),"folds":N_FOLDS})'''),
    md("## 4. Stage 1: all architectures, one seed"),
    code(r'''stage1_rows=[]; stage1_records={}
for spec in STAGE1+[ACTION,SHUFFLED]:
    all_records=[]
    for fold_index in range(N_FOLDS):
        _,_,meta,metrics,records=fit_one(spec,fold_index,42,30,5)
        stage1_rows.append({"name":spec["name"],"fold":fold_index,
                            "ranking":metrics["group_macro_ranking_accuracy"],
                            "uplift":metrics["top1_uplift_default"],
                            "best_epoch":meta["best_rank_epoch"]})
        all_records += records
    stage1_records[spec["name"]]=all_records
stage1=pd.DataFrame(stage1_rows)
summary=stage1.groupby("name").agg(ranking=("ranking","mean"),
                                    uplift=("uplift","mean"),best_epoch=("best_epoch","median"))
action_ranking=summary.loc[ACTION["name"],"ranking"]
shuffled_ranking=summary.loc[SHUFFLED["name"],"ranking"]
eligible=summary.loc[[s["name"] for s in STAGE1]].copy()
eligible["control_gap"]=eligible.ranking-max(action_ranking,shuffled_ranking)
shortlist=list(eligible.sort_values(["control_gap","ranking","uplift"],ascending=False).head(2).index)
assert len(shortlist)==2
stage1.to_csv(OUTPUT/"stage1_results.csv",index=False)
print(summary.sort_values("ranking",ascending=False)); print("shortlist",shortlist)'''),
    md("## 5. Stage 2: shortlisted configurations × three seeds"),
    code(r'''spec_by_name={spec["name"]:spec for spec in STAGE1}
stage2_rows=[]; pooled={name:[] for name in shortlist}
for name in shortlist:
    spec=spec_by_name[name]
    for seed in SEEDS:
        for fold_index in range(N_FOLDS):
            _,_,meta,metrics,records=fit_one(spec,fold_index,seed,50,7)
            stage2_rows.append({"name":name,"seed":seed,"fold":fold_index,
                                "ranking":metrics["group_macro_ranking_accuracy"],
                                "uplift":metrics["top1_uplift_default"],
                                "best_epoch":meta["best_rank_epoch"]})
            pooled[name] += records
stage2=pd.DataFrame(stage2_rows)
stage2_summary=stage2.groupby("name").agg(
    ranking=("ranking","mean"),ranking_std=("ranking","std"),
    uplift=("uplift","mean"),best_epoch=("best_epoch","median"))
stage2_summary["control_gap"]=stage2_summary.ranking-max(action_ranking,shuffled_ranking)
selected=stage2_summary.sort_values(
    ["control_gap","ranking","uplift"],ascending=False).index[0]
stage2.to_csv(OUTPUT/"stage2_results.csv",index=False)
print(stage2_summary); print("selected",selected)'''),
    md("## 6. Final refits and single checkpoint bundle"),
    code(r'''selected_spec=spec_by_name[selected]
final_epochs=max(1,int(stage2.loc[stage2.name==selected,"best_epoch"].median())+1)
split=known_task_split(historical,seed=159)
value_val=select_examples(historical,split["val"]); value_val_ids=set(split["val"])
value_train=[e for e in historical if e.rollout_id not in value_val_ids]

def final_fit(spec, train_examples):
    cfg=AdvantageTrainConfig(seed=159,prefix_length=10,value_epochs=60,
        rank_epochs=final_epochs,patience=100,rank_lr=spec["rank_lr"],weight_decay=1e-2,
        context_lr_multiplier=.1,zero_context=spec.get("action_only",False))
    model=CompactAdvantageVerifier(action_width=64,dropout=spec["dropout"],
                                   conditioning=spec["architecture"])
    model,_=pretrain_value(model,value_train,value_val,DEVICE,config=cfg)
    model,_=train_advantage(model,train_examples,[],DEVICE,config=cfg)
    return model,cfg

selected_model,selected_cfg=final_fit(selected_spec,development)
action_model,action_cfg=final_fit(ACTION,development)
shuffled_model,shuffled_cfg=final_fit(
    SHUFFLED,shuffle_candidate_actions_within_group(development,159))
bundle={"model":selected_model.state_dict(),"controls":{
    "action_only":{"model":action_model.state_dict(),"spec":ACTION},
    "shuffled_actions":{"model":shuffled_model.state_dict(),"spec":SHUFFLED}},
    "metadata":{"selected_spec":selected_spec,"final_rank_epochs":final_epochs,
                "stage2_summary":stage2_summary.reset_index().to_dict("records")}}
import io
buffer=io.BytesIO(); torch.save(bundle,buffer)
verifier_id=new_verifier_id()
store.start_run("verifier_train","libero+libero_pro","state-conditioned-verifier-v2",
                config=bundle["metadata"])
store.register_verifier(verifier_id,buffer.getvalue(),{
    "model_class":"CompactAdvantageVerifier","obs_dim":2048,"action_dim":7,
    "horizon":50,"prefix_length":10,"architecture":selected_spec["architecture"],
    "action_width":64,"dropout":selected_spec["dropout"]},
    {"development_stage2":stage2_summary.reset_index().to_dict("records")},
    {"folds":folds,"sealed_confirmatory_group_ids":sorted(
        row["candidate_group_id"] for row in sealed)},dataset_hash=dataset_hash(development))
store.finish_run(n_rollouts=len(development))
(OUTPUT/"registered_verifier.json").write_text(json.dumps({
    "verifier_id":verifier_id,"selected":selected_spec,"final_epochs":final_epochs},indent=2))
print({"registered_verifier":verifier_id,"selected":selected_spec,
       "final_rank_epochs":final_epochs})'''),
], "09_train_state_conditioned_verifier_v2.ipynb")


CONFIRMATION = notebook([
    md("""# 10 — One-shot LIBERO-PRO verifier V2 confirmation

Run only after notebook 09 registers the final bundle and all 160 confirmatory
groups are complete. This opens the sealed outcomes once and applies the
predeclared registration gates."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Load the frozen bundle and sealed candidates"),
    code(r'''import json
from pathlib import Path
import torch
from pnp.store import SupabaseStore
from pnp.verifier import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT=Path(package_dir)/"analysis_outputs"/"verifier_v2"
registration=json.loads((OUTPUT/"registered_verifier.json").read_text())
store=SupabaseStore(); checkpoint,row=store.load_verifier(registration["verifier_id"])
examples=load_candidate_examples(
    store,"verifier-v2-pro-confirmatory",cache_dir=OUTPUT/"confirmatory_cache")
audit=validate_candidate_groups(examples,expected_candidates=12)
assert audit["groups"] >= 150, audit

def load_model(spec,state):
    model=CompactAdvantageVerifier(obs_dim=int(row["obs_dim"]),action_dim=int(row["action_dim"]),
        action_width=64,dropout=spec["dropout"],conditioning=spec["architecture"])
    model.load_state_dict(state); return model.to(DEVICE).eval()

selected_spec=checkpoint["metadata"]["selected_spec"]
selected=load_model(selected_spec,checkpoint["model"])
action=load_model(checkpoint["controls"]["action_only"]["spec"],
                  checkpoint["controls"]["action_only"]["model"])
shuffled=load_model(checkpoint["controls"]["shuffled_actions"]["spec"],
                    checkpoint["controls"]["shuffled_actions"]["model"])
print({"verifier":registration["verifier_id"],"integrity":audit})'''),
    md("## 3. Predeclared paired-bootstrap gate"),
    code(r'''config=AdvantageTrainConfig(seed=20260728,prefix_length=10)
metrics,selected_records=evaluate_candidate_ranker(
    selected,examples,DEVICE,config=config,return_records=True)
_,action_records=evaluate_candidate_ranker(
    action,examples,DEVICE,config=config,return_records=True)
_,shuffled_records=evaluate_candidate_ranker(
    shuffled,examples,DEVICE,config=config,return_records=True)
action_comparison=paired_candidate_comparison(selected_records,action_records,seed=20260728)
shuffled_comparison=paired_candidate_comparison(
    selected_records,shuffled_records,seed=20260729)
gate=verifier_registration_eligibility(metrics,action_comparison,shuffled_comparison)
report={"verifier_id":registration["verifier_id"],"metrics":metrics,
        "vs_action_only":action_comparison,"vs_shuffled_actions":shuffled_comparison,
        "registration_gate":gate}
(OUTPUT/"confirmatory_report.json").write_text(json.dumps(report,indent=2,sort_keys=True))
(store.client.table("verifier_models").update({"metrics_json":report})
 .eq("verifier_id",registration["verifier_id"]).execute())
print(json.dumps(report,indent=2))
print("REGISTERED" if gate["eligible"] else "EXPLORATORY ONLY — gate not met")'''),
], "10_confirm_verifier_v2.ipynb")


def write_notebook(path, document):
    path.write_text(json.dumps(document, indent=1) + "\n")


def main():
    notebooks = ROOT / "notebooks"
    write_notebook(notebooks / "08_collect_verifier_v2_pro.ipynb", COLLECTION)
    write_notebook(notebooks / "09_train_state_conditioned_verifier_v2.ipynb", TRAINING)
    write_notebook(notebooks / "10_confirm_verifier_v2.ipynb", CONFIRMATION)


if __name__ == "__main__":
    main()
