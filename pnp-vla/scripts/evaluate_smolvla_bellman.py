"""Recorded-policy Bellman evaluation on existing train and held-out artifacts.

Never trains, changes the split, collects a rollout, or dispatches paid workers.
Validation trajectories are materialized in a separate evaluation-only cache.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fcntl
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_smolvla_anchored_local import DIGEST, load_data
from scripts.train_smolvla_combined_local import _load_local_credentials
from scripts.run_smolvla_local_continuation import scalar_from_architecture
from pnp.smolvla_scalar_returns import INPUTS, FIELDS, restore_transitions, LateFusionScalarCritic
from pnp.smolvla_two_stage import trajectory_spec, PRETRAIN_KINDS
from pnp.pcp_critic.resumable_snapshot import load_training_fields_with_retry
from pnp.qplanning_critic.model import pool_prefix_tokens
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_anchored_critic import AnchoredCritic
from pnp.smolvla_tree_bellman_finetune import _atomic_json, _atomic_npz
from pnp.config import resolve_max_steps
from pnp.store import SupabaseStore

BASE = Path.home()/"pnp-vla-runs"
OUTPUT = BASE/"diagnostics/critic_evaluation_audit_v1"
FORMAT = "smolvla_heldout_bellman_eval_v1"
EXTRA_FIELDS = ("prefix/prefix_embeddings","prefix/prefix_pad_masks","boundary/raw_robot_state","boundary/policy_proprio")


def prepare_heldout(roots,snapshot,index):
    destination=BASE/"cache"/FORMAT/DIGEST
    destination.mkdir(parents=True,exist_ok=True)
    ids=set(index["validation_group_ids"])
    if ids & set(index["train_group_ids"]):
        raise ValueError("Overlapping evaluation splits")
    groups={g["candidate_group_id"]:g for g in snapshot["groups"] if g["candidate_group_id"] in ids}
    positions={g["candidate_group_id"]:i for i,g in enumerate(roots.entries)}
    specs=[trajectory_spec(groups[gid],kind) for gid in sorted(ids) for kind in PRETRAIN_KINDS]
    if len(specs)!=960 or len({s["path"] for s in specs})!=960:
        raise ValueError("Expected exactly 320 held-out roots and three distinct artifacts per root")
    _load_local_credentials()
    store=SupabaseStore()
    def build(spec):
        key=hashlib.sha256((spec["group_id"]+"|"+spec["kind"]).encode()).hexdigest()[:24]
        path=destination/f"trajectory_{key}.npz"
        if not path.exists():
            raw=load_training_fields_with_retry(store.fork_for_thread(),spec["path"],(*FIELDS,*EXTRA_FIELDS))
            prefix=np.asarray(raw["prefix/prefix_embeddings"],np.float16)
            pad=np.asarray(raw["prefix/prefix_pad_masks"],bool)
            if prefix.ndim==4 and prefix.shape[1]==1: prefix=prefix[:,0]
            if pad.ndim==3 and pad.shape[1]==1: pad=pad[:,0]
            n=len(raw["bellman/action"]); start=spec["start"]
            if len(prefix)!=n+1 or len(pad)!=n+1:
                raise ValueError("Incomplete prefix boundary sequence")
            if n>start:
                pooled,valid=pool_prefix_tokens(torch.from_numpy(prefix[start:n]),torch.from_numpy(pad[start:n]),128)
                pooled,valid=pooled.numpy().astype(np.float16),valid.numpy()
            else:
                pooled=np.empty((0,128,prefix.shape[-1]),np.float16); valid=np.empty((0,128),bool)
            maximum=resolve_max_steps(spec["suite"])
            steps=np.asarray(raw["boundary/step"])[start:n]+spec["step_offset"]
            old={"prefix":pooled,"pad":valid,
                 "robot":np.concatenate([np.asarray(raw["boundary/raw_robot_state"],np.float32)[start:n],
                                          ((maximum-steps)/maximum).astype(np.float32)[:,None]],axis=1),
                 "proprio":np.asarray(raw["boundary/policy_proprio"],np.float32)[start:n],
                 "action":np.empty((n-start,10,7),np.float32)}
            root=roots[positions[spec["group_id"]]] if start else None
            arrays=restore_transitions(old,raw,spec,root)
            _atomic_npz(path,{"format":np.asarray(FORMAT),"group_id":np.asarray(spec["group_id"]),
                              "kind":np.asarray(spec["kind"]),**arrays})
        with np.load(path,allow_pickle=False) as arrays:
            if str(arrays["format"])!=FORMAT or str(arrays["group_id"])!=spec["group_id"] or str(arrays["kind"])!=spec["kind"]:
                raise ValueError("Evaluation cache provenance differs")
            windows=len(arrays["success"])
            if not windows or not np.all(arrays["success"]==spec["success"]):
                raise ValueError("Evaluation cache outcome differs")
        return {**spec,"file":path.name,"windows":windows}
    entries=[]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(build,spec) for spec in specs]
        for future in as_completed(futures):
            entries.append(future.result())
            if len(entries)%30==0:
                print({"heldout_trajectories_prepared":len(entries),"planned":960},flush=True)
    result={"format":FORMAT,"snapshot_digest":DIGEST,"train_group_ids":index["train_group_ids"],
            "validation_group_ids":sorted(ids),"entries":sorted(entries,key=lambda s:(s["group_id"],s["kind"])),
            "cache_dir":str(destination),"usage":"Evaluation only; never add to ReturnSampler training cache."}
    _atomic_json(destination/"cache_index.json",result)
    return result


def model_from_checkpoint(path,device):
    saved=torch.load(path,map_location="cpu",weights_only=False)
    if saved["snapshot_digest"]!=DIGEST: raise ValueError("Checkpoint split differs")
    arch=dict(saved["architecture"])
    family=arch.get("model_family","")
    if family.startswith("anchored"):
        model=AnchoredCritic(scalar_from_architecture(arch["base_architecture"]),
                            "temporal_cnn" if "cnn" in family else "transformer",width=arch["width"],delta_scale=arch["delta_scale"])
    elif family=="late_fusion_scalar":
        arch.pop("model_family"); dims={k:arch.pop(k) for k in ("prefix_dim","robot_dim","proprio_dim")}
        model=LateFusionScalarCritic(**dims,config=QPlanningModelConfig(**arch))
    else:
        model=scalar_from_architecture(arch)
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False),saved


@torch.no_grad()
def scores(model,arrays,root_reference,branch,device):
    values=[]
    for start in range(0,len(arrays["action"]),32):
        batch={k:torch.from_numpy(arrays[k][start:start+32]).to(device) for k in INPUTS}
        if isinstance(model,AnchoredCritic):
            reference=batch["action"].clone()
            if branch and start==0:
                reference[0]=torch.as_tensor(root_reference,device=device)
            features,base=model.reference_features(batch["prefix"],batch["pad"],batch["robot"],batch["proprio"],reference,batch["action_valid"])
            z=model.logits_from_features(features,base,batch["action"],batch["action_valid"],reference)
        else:
            z=model(*(batch[k] for k in INPUTS))
        values.extend(z.squeeze(-1).sigmoid().cpu().numpy().tolist())
    return np.asarray(values,float)


def transition_errors(q,reward,discount,nstep):
    n=len(q); targets=[]; terminals=[]
    for i in range(n):
        r,d=0.,1.
        for k in range(i,min(i+nstep,n)):
            r+=d*float(reward[k]); d*=float(discount[k])
            if not d: break
        if d and i+nstep>=n: raise ValueError("Nonterminal trajectory end")
        targets.append(r+d*q[min(i+nstep,n-1)]); terminals.append(d==0)
    residual=q-np.asarray(targets)
    return residual,np.asarray(terminals)


def summarize_episodes(rows):
    ids=sorted({r["group_id"] for r in rows})
    result={"episodes":len(ids),"trajectories":len(rows),"transitions":sum(r["transitions"] for r in rows)}
    for key in ("residual_mean","residual_mae","residual_mse","mc_mse","terminal_mse","nonterminal_mse"):
        values=np.asarray([np.mean([r[key] for r in rows if r["group_id"]==gid and r[key] is not None])
                           for gid in ids if any(r["group_id"]==gid and r[key] is not None for r in rows)])
        if not len(values):result[key]=None;continue
        se=values.std(ddof=1)/np.sqrt(len(values))
        result[key]={"mean":float(values.mean()),"standard_error":float(se),"episodes":len(values)}
        if key.endswith("mse"):
            root=np.sqrt(values.mean())
            result[key.replace("mse","rmse")]={"value":float(root),"standard_error_delta_method":float(se/(2*root)) if root else None}
    return result


def main():
    torch.set_num_threads(2)
    anchored=BASE/"checkpoints/smolvla_anchored_cnn_v1"/DIGEST
    snapshot=json.loads((anchored/"snapshot.json").read_text())
    roots=load_data(BASE,anchored)
    train_path=BASE/"cache/smolvla_scalar_returns_v1_preaction"/DIGEST/"cache_index.json"
    train=json.loads(train_path.read_text());train["cache_dir"]=str(train_path.parent)
    heldout=prepare_heldout(roots["validation"],snapshot,train)
    lock=(BASE/"diagnostics/local_continuation_v1.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models={"pcp_frozen_late_mc":BASE/"checkpoints/smolvla-overnight-20260930-mc_roots_late"/DIGEST/"mc/latest.pt",
            "cnn_mc_continuation":BASE/"checkpoints/smolvla_local_continuation_v1"/DIGEST/"mc_anchored_cnn/latest.pt",
            "td_n5_continuation":BASE/"checkpoints/smolvla_local_continuation_v1"/DIGEST/"td_n5/latest.pt"}
    output={"operator":"Recorded-policy SARSA, gamma=1, n=1 and n=5; frozen online self-bootstrap",
            "aggregation":"Equal trajectories within episode, then equal episodes; standard errors across episodes",
            "validation_status":"held-out trajectories of repeatedly inspected validation episodes, exploratory",
            "models":{}}
    for name,path in models.items():
        model,saved=model_from_checkpoint(path,device)
        entry={"checkpoint":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"splits":{}}
        for split,index in (("train",train),("validation",heldout)):
            positions={g["candidate_group_id"]:i for i,g in enumerate(roots[split].entries)}
            all_rows={1:[],5:[]}
            for position,spec in enumerate(index["entries"]):
                with np.load(Path(index["cache_dir"])/spec["file"],allow_pickle=False) as archive:
                    arrays={k:archive[k] for k in (*INPUTS,"reward","discount","success")}
                reference=roots[split][positions[spec["group_id"]]]["actions"][0]
                q=scores(model,arrays,reference,spec["kind"]!="stored_source",device)
                for horizon in (1,5):
                    residual,terminal=transition_errors(q,arrays["reward"],arrays["discount"],horizon)
                    all_rows[horizon].append({"group_id":spec["group_id"],"kind":spec["kind"],"transitions":len(q),
                                              "residual_mean":float(residual.mean()),"residual_mae":float(np.abs(residual).mean()),
                                              "residual_mse":float(np.square(residual).mean()),
                                              "mc_mse":float(np.square(q-arrays["success"]).mean()),
                                              "terminal_mse":float(np.square(residual[terminal]).mean()) if terminal.any() else None,
                                              "nonterminal_mse":float(np.square(residual[~terminal]).mean()) if (~terminal).any() else None})
                if (position+1)%120==0:print({"model":name,"split":split,"scored_trajectories":position+1,"planned":len(index["entries"])},flush=True)
            entry["splits"][split]={str(h):summarize_episodes(rows) for h,rows in all_rows.items()}
            _atomic_json(OUTPUT/f"bellman_{name}_{split}_per_trajectory.json",{str(h):rows for h,rows in all_rows.items()})
        output["models"][name]=entry
        _atomic_json(OUTPUT/"bellman_report.json",output)
    lines=["# Training and held-out Bellman error","",output["operator"],"",output["aggregation"]+". "+output["validation_status"],"",
           "| Model | Split | Episodes / trajectories / transitions | One-step RMSE ± SE | Five-step RMSE ± SE | MC return RMSE ± SE |","|---|---|---:|---:|---:|---:|"]
    for name,entry in output["models"].items():
        for split,metrics in entry["splits"].items():
            one,five=metrics["1"],metrics["5"]
            def show(m,key):
                r=m[key];return f"{r['value']:.4f} ± {r['standard_error_delta_method']:.4f}"
            lines.append(f"| {name} | {split} | {one['episodes']} / {one['trajectories']} / {one['transitions']} | {show(one,'residual_rmse')} | {show(five,'residual_rmse')} | {show(one,'mc_rmse')} |")
    lines += ["","Sampled-policy transition residuals include future-action/transition variability. Low nonterminal error alone does not validate outcome prediction. This is self-bootstrap consistency, not TD training loss against the EMA target. CNN reference is the recorded stock action at ordinary continuation states and the original stock root action at fresh branch roots. Terminal and nonterminal summaries are in bellman_report.json."]
    (OUTPUT/"bellman_results.md").write_text('\n'.join(lines)+'\n')
    print({"complete":str(OUTPUT/"bellman_results.md")},flush=True)


if __name__=="__main__":main()
