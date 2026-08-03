"""Build the V2 verifier collection and architecture-sweep notebooks."""
from __future__ import annotations

from nb_common import BOOTSTRAP, ROOT, code, md, notebook, write_notebook


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
    code(r'''DEVELOPMENT_EXPERIMENT = "verifier-v2-pro-development"
TEST_EXPERIMENT = "verifier-v2-pro-confirmatory"
CANDIDATE_COUNT = 12
PREFIX_LENGTH = 10
SHARD_COUNT = 1   # Generated workers set this to 3.
SHARD_INDEX = 0
assert 0 <= SHARD_INDEX < SHARD_COUNT

rollouts = store.fetch_all(
    "rollouts", "rollout_id,benchmark,suite,task_idx,episode_idx,success",
    configure=lambda q: q.eq("experiment", "libero-pro-canonical-core-k3-v1").eq(
        "method", "pnp_uncertainty_only").eq("status", "completed"),
    order_by=("rollout_id",))
rollout_ids = sorted(row["rollout_id"] for row in rollouts)
euler=[]
for start in range(0, len(rollout_ids), 100):
    ids=rollout_ids[start:start+100]
    euler += store.fetch_all(
        "pnp_euler_steps", "rollout_id,chunk_idx,euler_step,u_mean",
        configure=lambda q, ids=ids: q.in_("rollout_id", ids),
        order_by=("rollout_id", "chunk_idx", "euler_step"))
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
existing_rows=store.fetch_all(
    "verifier_candidate_groups", "candidate_group_id,experiment",
    configure=lambda q: q.in_("experiment", experiments), order_by=("candidate_group_id",))
existing_ids={row["candidate_group_id"] for row in existing_rows}
candidate_rows=[]
for start in range(0, len(existing_ids), 100):
    ids=sorted(existing_ids)[start:start+100]
    candidate_rows += store.fetch_all(
        "verifier_candidates", "candidate_id,candidate_group_id",
        configure=lambda q, ids=ids: q.in_("candidate_group_id", ids), order_by=("candidate_id",))
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
    groups=store.fetch_all("verifier_candidate_groups", "candidate_group_id",
                 configure=lambda q, e=experiment: q.eq("experiment", e), order_by=("candidate_group_id",))
    ids={row["candidate_group_id"] for row in groups}; candidates=[]
    for start in range(0, len(ids), 100):
        batch=sorted(ids)[start:start+100]
        candidates += store.fetch_all("verifier_candidates", "candidate_group_id,success",
                            configure=lambda q, ids=batch: q.in_("candidate_group_id", ids))
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
import wandb
from tqdm.auto import tqdm
from pnp import notebook as nb
from pnp.verifier import *

ctx=nb.setup("verifier_v2"); store,DEVICE,OUTPUT=ctx.store,ctx.device,ctx.output
HISTORICAL=("libero-hybrid-schedules-k3-v1", "libero-pro-canonical-core-k3-v1")
PRIMARY=("verifier-clean-pairs-v3", "verifier-clean-pairs-v4-dev",
         "verifier-clean-pairs-v4-test", "verifier-online-selection-v1",
         "verifier-v2-pro-development")
SEEDS=(42,43,44); N_FOLDS=4; PREFIX_LENGTH=10

historical=load_clean_chunk_examples(
    store, HISTORICAL, progress=tqdm, cache_dir=OUTPUT/"historical_cache")
existing_development=load_candidate_examples(
    store, PRIMARY[:-1], cache_dir=OUTPUT/"candidate_cache_existing")
new_development=load_candidate_examples(
    store, PRIMARY[-1], cache_dir=OUTPUT/"candidate_cache_v2")
new_audit=validate_candidate_groups(new_development,expected_candidates=12)
assert new_audit["groups"] >= 220, new_audit
development=existing_development+new_development
audit=validate_candidate_groups(development)
assert audit["discordant_groups"] >= 100, audit
# Seal IDs only. nb.sealed_identities never queries verifier_candidates.
sealed_identities, sealed = nb.sealed_identities(store)
historical=nb.drop_sealed(historical, sealed_identities)
folds=candidate_cv_splits(development, [e.rollout_id for e in development], N_FOLDS, 42)
print({"historical_chunks":len(historical), "development":audit,
       "new_PRO_development":new_audit,
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
    code(r'''stage1_rows=[]; stage1_records={}; history_rows=[]
for spec in STAGE1+[ACTION,SHUFFLED]:
    all_records=[]
    for fold_index in range(N_FOLDS):
        _,_,meta,metrics,records=fit_one(spec,fold_index,42,30,5)
        stage1_rows.append({"name":spec["name"],"fold":fold_index,
                            "ranking":metrics["group_macro_ranking_accuracy"],
                            "uplift":metrics["top1_uplift_default"],
                            "margin":metrics["mean_score_margin"],
                            "best_epoch":meta["best_rank_epoch"]})
        history_rows += [{**point,"stage":1,"name":spec["name"],
                          "fold":fold_index,"seed":42}
                         for point in meta["rank_history"]]
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
stage2_specs=[spec_by_name[name] for name in shortlist]+[ACTION,SHUFFLED]
stage2_rows=[]; pooled={spec["name"]:[] for spec in stage2_specs}
for spec in stage2_specs:
    name=spec["name"]
    for seed in SEEDS:
        for fold_index in range(N_FOLDS):
            _,_,meta,metrics,records=fit_one(spec,fold_index,seed,50,7)
            stage2_rows.append({"name":name,"seed":seed,"fold":fold_index,
                                "ranking":metrics["group_macro_ranking_accuracy"],
                                "uplift":metrics["top1_uplift_default"],
                                "margin":metrics["mean_score_margin"],
                                "best_epoch":meta["best_rank_epoch"]})
            history_rows += [{**point,"stage":2,"name":name,
                              "fold":fold_index,"seed":seed}
                             for point in meta["rank_history"]]
            pooled[name] += records
stage2=pd.DataFrame(stage2_rows)
stage2_summary=stage2.groupby("name").agg(
    ranking=("ranking","mean"),ranking_std=("ranking","std"),
    uplift=("uplift","mean"),best_epoch=("best_epoch","median"))
stage2_action=stage2_summary.loc[ACTION["name"],"ranking"]
stage2_shuffled=stage2_summary.loc[SHUFFLED["name"],"ranking"]
stage2_summary["control_gap"]=stage2_summary.ranking-max(stage2_action,stage2_shuffled)
selected=stage2_summary.loc[shortlist].sort_values(
    ["control_gap","ranking","uplift"],ascending=False).index[0]
stage2.to_csv(OUTPUT/"stage2_results.csv",index=False)
pd.DataFrame(history_rows).to_csv(OUTPUT/"training_curves.csv",index=False)
if os.getenv("WANDB_API_KEY"):
    run=wandb.init(project="pnp-state-conditioned-verifier-v2",name="architecture-sweep")
    for name,row in stage2_summary.iterrows():
        run.log({f"summary/{name}/ranking":row.ranking,
                 f"summary/{name}/uplift":row.uplift,
                 f"summary/{name}/control_gap":row.control_gap})
    artifact=wandb.Artifact("verifier-v2-sweep","evaluation")
    for filename in ("stage1_results.csv","stage2_results.csv","training_curves.csv"):
        artifact.add_file(str(OUTPUT/filename))
    run.log_artifact(artifact); run.finish()
print(stage2_summary); print("selected",selected)'''),
    md("## 6. Final refits and single checkpoint bundle"),
    code(r'''selected_spec=spec_by_name[selected]
selected_records=aggregate_candidate_records(pooled[selected])
action_records=aggregate_candidate_records(pooled[ACTION["name"]])
shuffled_records=aggregate_candidate_records(pooled[SHUFFLED["name"]])
development_metrics=summarize_candidate_records(selected_records)
action_comparison=paired_candidate_comparison(
    selected_records,action_records,n_bootstrap=2000)
shuffled_comparison=paired_candidate_comparison(
    selected_records,shuffled_records,seed=43,n_bootstrap=2000)
development_gate=verifier_registration_eligibility(
    development_metrics,action_comparison,shuffled_comparison)
development_report={"metrics":development_metrics,"vs_action_only":action_comparison,
                    "vs_shuffled_actions":shuffled_comparison,"gate":development_gate}
epoch_by_name={name:max(1,int(rows.best_epoch.median())+1)
               for name,rows in stage2.groupby("name")}
final_epochs=epoch_by_name[selected]
split=known_task_split(historical,seed=159)
value_val=select_examples(historical,split["val"]); value_val_ids=set(split["val"])
value_train=[e for e in historical if e.rollout_id not in value_val_ids]

def final_fit(spec, train_examples, rank_epochs):
    cfg=AdvantageTrainConfig(seed=159,prefix_length=10,value_epochs=60,
        rank_epochs=rank_epochs,patience=100,rank_lr=spec["rank_lr"],weight_decay=1e-2,
        context_lr_multiplier=.1,zero_context=spec.get("action_only",False))
    model=CompactAdvantageVerifier(action_width=64,dropout=spec["dropout"],
                                   conditioning=spec["architecture"])
    model,_=pretrain_value(model,value_train,value_val,DEVICE,config=cfg)
    model,_=train_advantage(model,train_examples,[],DEVICE,config=cfg)
    return model,cfg

selected_model,selected_cfg=final_fit(selected_spec,development,epoch_by_name[selected])
action_model,action_cfg=final_fit(ACTION,development,epoch_by_name[ACTION["name"]])
shuffled_model,shuffled_cfg=final_fit(
    SHUFFLED,shuffle_candidate_actions_within_group(development,159),
    epoch_by_name[SHUFFLED["name"]])
bundle={"model":selected_model.state_dict(),"controls":{
    "action_only":{"model":action_model.state_dict(),"spec":ACTION},
    "shuffled_actions":{"model":shuffled_model.state_dict(),"spec":SHUFFLED}},
    "metadata":{"selected_spec":selected_spec,"final_rank_epochs":final_epochs,
                "parameter_count":sum(p.numel() for p in selected_model.parameters()),
                "development_report":development_report,
                "development_records":selected_records,
                "action_control_records":action_records,
                "shuffled_control_records":shuffled_records,
                "stage2_summary":stage2_summary.reset_index().to_dict("records")}}
import io
buffer=io.BytesIO(); torch.save(bundle,buffer)
verifier_id=new_verifier_id()
store.start_run("verifier_train","libero+libero_pro","state-conditioned-verifier-v2",
                config={"selected_spec":selected_spec,
                        "final_rank_epochs":final_epochs,
                        "development_gate":development_gate})
store.register_verifier(verifier_id,buffer.getvalue(),{
    "model_class":"CompactAdvantageVerifier","obs_dim":2048,"action_dim":7,
    "horizon":50,"prefix_length":10,"architecture":selected_spec["architecture"],
    "action_width":64,"dropout":selected_spec["dropout"]},
    {"development_stage2":stage2_summary.reset_index().to_dict("records"),
     "development_report":development_report},
    {"folds":folds,"sealed_confirmatory_group_ids":sorted(
        row["candidate_group_id"] for row in sealed)},dataset_hash=dataset_hash(development))
store.finish_run(n_rollouts=len(development))
(OUTPUT/"registered_verifier.json").write_text(json.dumps({
    "verifier_id":verifier_id,"selected":selected_spec,"final_epochs":final_epochs},indent=2))
print({"registered_verifier":verifier_id,"selected":selected_spec,
       "final_rank_epochs":final_epochs})'''),
], "09_train_state_conditioned_verifier_v2.ipynb")


# Notebook 10 is the hybrid-critic trainer and notebook 11 the one-shot
# confirmation/arbitration -- both generated by build_hybrid_critic_notebooks.py.
# The superseded standalone "10 — confirm verifier V2" notebook is retired.


def main():
    notebooks = ROOT / "notebooks"
    write_notebook(notebooks / "08_collect_verifier_v2_pro.ipynb", COLLECTION)
    write_notebook(notebooks / "09_train_state_conditioned_verifier_v2.ipynb", TRAINING)


if __name__ == "__main__":
    main()
