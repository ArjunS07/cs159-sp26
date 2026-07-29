"""Build the hybrid-critic training and development-only arbitration notebooks."""
from __future__ import annotations

import json
from pathlib import Path

from build_verifier_v2_notebooks import BOOTSTRAP, PAGES, code, md, notebook


ROOT = Path(__file__).resolve().parents[1]


HYBRID = notebook([
    md("""# 10 — Hybrid long/short chunk critic

Train an IQL-style twin 50-step critic on historical transitions, distill it into a
twin 10-step candidate critic, and add same-state Monte-Carlo and ranking losses.
The sealed confirmatory outcomes are never queried in this notebook."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Load transition and causal development data"),
    code(r'''from dataclasses import replace
from pathlib import Path
import copy, hashlib, io, json
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from pnp.store import SupabaseStore
from pnp.verifier import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT=Path(package_dir)/"analysis_outputs"/"hybrid_critic"; OUTPUT.mkdir(parents=True,exist_ok=True)
store=SupabaseStore()
HISTORICAL=("libero-hybrid-schedules-k3-v1","libero-pro-canonical-core-k3-v1")
DEVELOPMENT=("verifier-clean-pairs-v3","verifier-clean-pairs-v4-dev",
             "verifier-clean-pairs-v4-test","verifier-online-selection-v1",
             "verifier-v2-pro-development")
CONFIRMATORY="verifier-v2-pro-confirmatory"
SEEDS=(42,43,44); N_FOLDS=4

transitions=load_chunk_transitions(
    store,HISTORICAL,progress=tqdm,cache_dir=OUTPUT/"transition_cache")
development=load_candidate_examples(store,DEVELOPMENT,cache_dir=OUTPUT/"candidate_cache")
audit=validate_candidate_groups(development)
sealed=(store.client.table("verifier_candidate_groups").select(
    "candidate_group_id,benchmark,suite,task_idx,episode_idx")
    .eq("experiment",CONFIRMATORY).execute().data or [])
assert len(sealed)>=120,len(sealed)
sealed_identities={(r["benchmark"],r["suite"],int(r["task_idx"]),int(r["episode_idx"]))
                   for r in sealed}
transitions=exclude_episode_identities(transitions,sealed_identities)
folds=candidate_cv_splits(development,[e.rollout_id for e in development],N_FOLDS,42)

valid=np.concatenate([e.actions[e.action_mask] for e in transitions])
ACTION_MEAN=valid.mean(0).astype(np.float32)
ACTION_STD=valid.std(0).clip(min=1e-6).astype(np.float32)
print({"transitions":len(transitions),"development":audit,
       "sealed_groups":len(sealed),"device":str(DEVICE)})'''),
    md("## 3. Fixed transition split and training runner"),
    code(r'''def stable_fraction(value,modulus=5):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8],16)%modulus

def historical_split(candidate_validation):
    protected=candidate_episode_identities(candidate_validation)|sealed_identities
    clean=exclude_episode_identities(transitions,protected)
    validation={e.rollout_id for e in clean if stable_fraction(e.rollout_id)==0}
    return ([e for e in clean if e.rollout_id not in validation],
            [e for e in clean if e.rollout_id in validation])

LONG_CACHE={}
def fit_one(spec,fold_index,seed,*,shuffle=False,zero_context=False):
    fold=folds[fold_index]
    candidate_train=select_examples(development,fold["train"])
    candidate_val=select_examples(development,fold["val"])
    if shuffle:
        candidate_train=shuffle_candidate_actions_within_group(candidate_train,seed)
    historical_train,historical_val=historical_split(candidate_val)
    config=HybridCriticTrainConfig(
        seed=seed,expectile=spec["expectile"],rank_weight=spec["rank_weight"],
        long_updates=20_000,eval_interval=250,long_patience=8,
        short_epochs=50,short_patience=7,zero_context=zero_context)
    model=HybridChunkCritic(width=spec["width"],dropout=.15)
    model.set_action_statistics(ACTION_MEAN,ACTION_STD)
    long_key=(spec["width"],spec["expectile"],fold_index,seed)
    if long_key not in LONG_CACHE:
        model,long_meta=train_long_critic(
            model,historical_train,historical_val,DEVICE,config=config)
        LONG_CACHE[long_key]=(copy.deepcopy(model.state_dict()),long_meta)
    else:
        state,long_meta=LONG_CACHE[long_key]
        model.load_state_dict(state)
    model,short_meta=train_short_critic(
        model,historical_train,candidate_train,candidate_val,DEVICE,config=config)
    metrics,records=evaluate_candidate_ranker(
        model,candidate_val,DEVICE,
        config=AdvantageTrainConfig(seed=seed,prefix_length=10,zero_context=zero_context),
        n_bootstrap=1000,return_records=True)
    return model,config,long_meta,short_meta,metrics,records

SPECS=[{"name":f"w{width}-e{expectile}-r{rank}","width":width,
        "expectile":expectile,"rank_weight":rank}
       for width in (256,512) for expectile in (.7,.9) for rank in (1.,3.)]
print({"configs":len(SPECS),"folds":N_FOLDS,"seeds":SEEDS})'''),
    md("## 4. Stage 1: all configurations, one seed"),
    code(r'''stage1=[]
for spec in SPECS:
    for fold_index in range(N_FOLDS):
        _,_,long_meta,short_meta,metrics,_=fit_one(spec,fold_index,42)
        stage1.append({"name":spec["name"],"fold":fold_index,
                       "ranking":metrics["group_macro_ranking_accuracy"],
                       "uplift_default":metrics["top1_uplift_default"]})
stage1=pd.DataFrame(stage1)
shortlist=list(stage1.groupby("name").agg(
    ranking=("ranking","mean"),uplift=("uplift_default","mean")).sort_values(
        ["ranking","uplift"],ascending=False).head(2).index)
stage1.to_csv(OUTPUT/"stage1_results.csv",index=False)
print("shortlist",shortlist)'''),
    md("## 5. Stage 2: shortlisted configurations × three seeds"),
    code(r'''rows=[]; pooled={name:[] for name in shortlist}
for name in shortlist:
    spec=next(spec for spec in SPECS if spec["name"]==name)
    for seed in SEEDS:
        for fold_index in range(N_FOLDS):
            _,_,long_meta,short_meta,metrics,records=fit_one(spec,fold_index,seed)
            rows.append({"name":name,"seed":seed,"fold":fold_index,
                         "ranking":metrics["group_macro_ranking_accuracy"],
                         "uplift_default":metrics["top1_uplift_default"],
                         "uplift_random":metrics["top1_uplift_random"],
                         "long_update":long_meta["best_update"],
                         "short_epoch":short_meta["best_epoch"]})
            pooled[name]+=records
results=pd.DataFrame(rows)
summary=results.groupby("name").agg(
    ranking=("ranking","mean"),ranking_std=("ranking","std"),
    uplift_default=("uplift_default","mean"),uplift_random=("uplift_random","mean"),
    long_update=("long_update","median"),short_epoch=("short_epoch","median"))
selected=summary.sort_values(
    ["ranking","uplift_default","ranking_std"],ascending=[False,False,True]).index[0]
results.to_csv(OUTPUT/"sweep_results.csv",index=False)
print(summary.sort_values("ranking",ascending=False)); print("selected",selected)'''),
    md("## 6. Paired controls for the selected architecture"),
    code(r'''selected_spec=next(spec for spec in SPECS if spec["name"]==selected)
control_records={"action_only":[],"shuffled_actions":[]}
for seed in SEEDS:
    for fold_index in range(N_FOLDS):
        *_,records=fit_one(selected_spec,fold_index,seed,zero_context=True)
        control_records["action_only"]+=records
        *_,records=fit_one(selected_spec,fold_index,seed,shuffle=True)
        control_records["shuffled_actions"]+=records
selected_records=aggregate_candidate_records(pooled[selected])
action_records=aggregate_candidate_records(control_records["action_only"])
shuffled_records=aggregate_candidate_records(control_records["shuffled_actions"])
selected_metrics=summarize_candidate_records(selected_records,n_bootstrap=2000)
action_comparison=paired_candidate_comparison(
    selected_records,action_records,n_bootstrap=2000)
shuffled_comparison=paired_candidate_comparison(
    selected_records,shuffled_records,seed=43,n_bootstrap=2000)
gate=verifier_registration_eligibility(
    selected_metrics,action_comparison,shuffled_comparison)
development_report={"metrics":selected_metrics,"vs_action_only":action_comparison,
                    "vs_shuffled_actions":shuffled_comparison,"gate":gate}
print(json.dumps(development_report,indent=2))'''),
    md("## 7. Refit and register the development-selected bundle"),
    code(r'''# The final refit uses only development data; one development fold remains an
# early-stopping monitor. Confirmation remains sealed until notebook 11.
monitor=select_examples(development,folds[0]["val"])
historical_train,historical_val=historical_split(monitor)
final_config=HybridCriticTrainConfig(
    seed=159,expectile=selected_spec["expectile"],
    rank_weight=selected_spec["rank_weight"],
    long_updates=max(250,int(summary.loc[selected,"long_update"])),
    short_epochs=max(1,int(summary.loc[selected,"short_epoch"])+1),
    long_patience=1000,short_patience=1000)
final_model=HybridChunkCritic(width=selected_spec["width"],dropout=.15)
final_model.set_action_statistics(ACTION_MEAN,ACTION_STD)
final_model,_=train_long_critic(
    final_model,historical_train,historical_val,DEVICE,config=final_config)
final_model,_=train_short_critic(
    final_model,historical_train,development,monitor,DEVICE,config=final_config)
metadata={"selected_spec":selected_spec,"development_report":development_report,
          "development_records":selected_records,
          "action_control_records":action_records,
          "shuffled_control_records":shuffled_records,
          "parameter_count":sum(p.numel() for p in final_model.parameters()),
          "sweep_summary":summary.reset_index().to_dict("records")}
payload=hybrid_checkpoint_bytes(final_model,config=final_config,metadata=metadata)
verifier_id=new_verifier_id()
store.start_run("verifier_train","libero+libero_pro","hybrid-chunk-critic-v1",
                config={"selected_spec":selected_spec,"gate":gate})
store.register_verifier(verifier_id,payload,{
    "model_class":"HybridChunkCritic","obs_dim":2048,"action_dim":7,
    "horizon":50,"prefix_length":10,"width":selected_spec["width"],"dropout":.15},
    development_report,{"folds":folds,"sealed_confirmatory_group_ids":sorted(
        row["candidate_group_id"] for row in sealed)},
    dataset_hash=transition_dataset_hash(transitions))
store.finish_run(n_rollouts=len(development))
print({"registered":verifier_id,"selected":selected_spec,"development_gate":gate})'''),
], "10_train_hybrid_chunk_critic.ipynb")


ARBITRATION = notebook([
    md("""# 11 — Development arbitration and one-shot confirmation

Choose between the notebook-09 baseline and notebook-10 hybrid using stored
development results only. If neither clears its causal controls, stop without
opening confirmatory outcomes. Otherwise evaluate exactly one winner."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Development-only arbitration"),
    code(r'''import json, torch
from pathlib import Path
from pnp.store import SupabaseStore
from pnp.verifier import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT=Path(package_dir)/"analysis_outputs"/"verifier_confirmation"; OUTPUT.mkdir(parents=True,exist_ok=True)
store=SupabaseStore()

def latest(experiment):
    rows=(store.client.table("verifier_models").select("verifier_id,created_at,metrics_json")
          .eq("experiment",experiment).order("created_at",desc=True).limit(1).execute().data or [])
    return rows[0] if rows else None

candidates=[]
for experiment in ("state-conditioned-verifier-v2","hybrid-chunk-critic-v1"):
    row=latest(experiment)
    if not row: continue
    checkpoint,_=store.load_verifier(row["verifier_id"])
    metadata=checkpoint.get("metadata",{})
    report=(metadata.get("development_report") or
            (row.get("metrics_json") or {}).get("development_report") or
            row.get("metrics_json") or {})
    metrics=report.get("metrics") or report.get("development_metrics") or {}
    gate=report.get("gate") or report.get("registration_gate") or {}
    candidates.append({"experiment":experiment,"row":row,"checkpoint":checkpoint,
                       "metadata":metadata,"metrics":metrics,
                       "eligible":bool(gate.get("eligible",False))})

eligible=[c for c in candidates if c["eligible"]]
assert eligible, "No development model cleared its causal-control gate; confirmation stays sealed."
best_ranking=max(c["metrics"].get("group_macro_ranking_accuracy",-1) for c in eligible)
statistically_tied=[c for c in eligible
                    if best_ranking-c["metrics"].get("group_macro_ranking_accuracy",-1)<=.005]
winner=min(statistically_tied,key=lambda c:c["metadata"].get("parameter_count",10**20))
print({"winner":winner["experiment"],"verifier_id":winner["row"]["verifier_id"],
       "development_ranking":winner["metrics"].get("group_macro_ranking_accuracy")})'''),
    md("## 3. Open the sealed outcomes once and evaluate the winner"),
    code(r'''examples=load_candidate_examples(
    store,"verifier-v2-pro-confirmatory",cache_dir=OUTPUT/"confirmatory_cache")
audit=validate_candidate_groups(examples,expected_candidates=12)
assert audit["groups"]>=150,audit
checkpoint=winner["checkpoint"]
if winner["experiment"]=="hybrid-chunk-critic-v1":
    model=HybridChunkCritic(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
else:
    spec=checkpoint["metadata"]["selected_spec"]
    model=CompactAdvantageVerifier(action_width=64,dropout=spec["dropout"],
                                   conditioning=spec["architecture"])
    model.load_state_dict(checkpoint["model"])
model=model.to(DEVICE).eval()
metrics=evaluate_candidate_ranker(
    model,examples,DEVICE,config=AdvantageTrainConfig(seed=20260729,prefix_length=10))
report={"winner":winner["experiment"],"verifier_id":winner["row"]["verifier_id"],
        "integrity":audit,"confirmatory_metrics":metrics}
(OUTPUT/"confirmatory_report.json").write_text(json.dumps(report,indent=2,sort_keys=True))
(store.client.table("verifier_models").update({"metrics_json":report})
 .eq("verifier_id",winner["row"]["verifier_id"]).execute())
print(json.dumps(report,indent=2))'''),
], "11_arbitrate_and_confirm_verifier.ipynb")


def main():
    notebooks = ROOT / "notebooks"
    (notebooks / "10_train_hybrid_chunk_critic.ipynb").write_text(
        json.dumps(HYBRID, indent=1) + "\n")
    (notebooks / "11_arbitrate_and_confirm_verifier.ipynb").write_text(
        json.dumps(ARBITRATION, indent=1) + "\n")


if __name__ == "__main__":
    main()
