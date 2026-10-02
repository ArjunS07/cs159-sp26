"""Frozen critic action derivatives at every available recorded full action proposal."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_smolvla_bellman import BASE, OUTPUT, DIGEST, INPUTS, AnchoredCritic, model_from_checkpoint, load_data
from pnp.smolvla_tree_bellman_finetune import _atomic_json, _atomic_npz


def clustered(values, identities):
    """Chunk-weighted mean with root-clustered ratio-estimator standard error."""
    ids, inverse = np.unique(identities, return_inverse=True)
    values = np.asarray(values, float)
    counts = np.bincount(inverse).astype(float)
    sums = np.zeros((len(ids), *values.shape[1:]))
    np.add.at(sums, inverse, values)
    mean = values.mean(0)
    shape = (len(ids),) + (1,) * (values.ndim - 1)
    influence = (sums - counts.reshape(shape) * mean) / counts.mean()
    se = influence.std(0, ddof=1) / np.sqrt(len(ids)) if len(ids) > 1 else np.full_like(mean, np.nan)
    return {"mean":mean.tolist(), "standard_error":se.tolist(), "episode_clusters":len(ids)}


def summarize(files):
    packed = []
    for path in files:
        with np.load(path, allow_pickle=False) as a:
            packed.append({k:a[k] for k in a.files})
    result = {}
    for outcome in (True, False):
        selected = [a for a in packed if bool(a["success"][0]) == outcome]
        if not selected:
            continue
        identities = np.concatenate([np.repeat(str(a["group_id"]), len(a["q"])) for a in selected])
        q = np.concatenate([a["q"] for a in selected])
        logit_g = np.concatenate([a["raw_logit_gradient"] for a in selected])
        std = packed[0]["action_std"]
        valid = np.concatenate([a["valid"] for a in selected])
        entry = {"chunks":len(q), "trajectories":len(selected), "episode_clusters":len(set(identities)),
                 "valid_action_scalars":int(valid.sum()*7), "q":clustered(q, identities)}
        for coordinate, factor in (("raw_action",1.), ("standardized_action",std)):
            for output, multiplier in (("logit",np.ones_like(q)), ("probability",q*(1-q))):
                g = logit_g * factor * multiplier[:,None,None]
                entry[coordinate+"_"+output] = {
                    "L2_norm":clustered(np.linalg.norm(g,axis=(1,2)), identities),
                    "signed_mean_10x7":clustered(g, identities),
                    "absolute_mean_10x7":clustered(np.abs(g), identities),
                    "L2_median_p95":np.quantile(np.linalg.norm(g,axis=(1,2)),[.5,.95]).tolist(),
                    "signed_mean_by_action_dimension":clustered(g.mean(1), identities)}
        result["success" if outcome else "failure"] = entry
    return result


def render(report):
    lines = ["# Action gradients across recorded trajectories", "", report["scope"], "", report["aggregation"], "",
             "Gradients are evaluated at each recorded full action proposal and labeled by its trajectory's eventual outcome. CNN references are held fixed; original stock root action at fresh branch roots, recorded action elsewhere. Known-invalid action slots have zero gradients. No future outcome mask is applied.", "",
             "| Model | Split | Outcome | Chunks / trajectories / episode clusters | Probability gradient L2, standardized ± SE | Logit gradient L2, standardized ± SE | Probability gradient L2, native policy units ± SE |",
             "|---|---|---|---:|---:|---:|---:|"]
    for name, entry in report["models"].items():
        for split, groups in entry["splits"].items():
            for outcome, group in groups.items():
                def show(key):
                    a=group[key]["L2_norm"]
                    return f"{a['mean']:.6f} ± {a['standard_error']:.6f}"
                lines.append(f"| {name} | {split} | {outcome} | {group['chunks']} / {group['trajectories']} / {group['episode_clusters']} | {show('standardized_action_probability')} | {show('standardized_action_logit')} | {show('raw_action_probability')} |")
    lines += ["", "Full signed and absolute 10×7 mean gradient arrays, their clustered standard errors, action-dimension means, and median/p95 norms are in trajectory_gradient_report.json. Individual native-policy-unit logit gradient arrays and predicted probabilities are preserved in trajectory_gradients/. Signed coordinates can cancel across states; magnitude and direction usefulness are separate measurements. Chunk counts are not independent episode counts. These are recorded-policy trajectories, not a repeated-PCP rollout evaluation."]
    (OUTPUT/"trajectory_gradient_results.md").write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    job_lock = (OUTPUT/"trajectory_gradients.lock").open("a+")
    fcntl.flock(job_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock = (BASE/"diagnostics/local_continuation_v1.lock").open("a+")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    roots=load_data(BASE,BASE/"checkpoints/smolvla_anchored_cnn_v1"/DIGEST)
    indexes={}
    for split,folder in (("train","smolvla_scalar_returns_v1_preaction"),("validation","smolvla_heldout_bellman_eval_v1")):
        index_path=BASE/"cache"/folder/DIGEST/"cache_index.json"
        indexes[split]=(json.loads(index_path.read_text()),index_path.parent)
    models={"pcp_frozen_late_mc":BASE/"checkpoints/smolvla-overnight-20260930-mc_roots_late"/DIGEST/"mc/latest.pt",
            "cnn_mc_continuation":BASE/"checkpoints/smolvla_local_continuation_v1"/DIGEST/"mc_anchored_cnn/latest.pt",
            "td_n5_continuation":BASE/"checkpoints/smolvla_local_continuation_v1"/DIGEST/"td_n5/latest.pt"}
    report={"scope":"All chunks in the existing three-trajectory-per-root caches (stored source, fresh seed 1, fresh seed 5): training 1,280 roots and held-out 320 roots. Other candidate trajectories are not in these caches.",
            "aggregation":"Native action units are policy outputs before critic standardization, not physical robot units. Full proposals are evaluated, including unexecuted terminal tails; only known-budget invalid slots are masked. Chunk-weighted means; standard errors clustered by originating episode/root, including correlated trajectories. Conditional on fixed checkpoints. Repeatedly examined held-out data remain exploratory.","models":{}}
    report_path=OUTPUT/"trajectory_gradient_report.json"
    if report_path.exists():report=json.loads(report_path.read_text())
    for name,path in models.items():
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        model,_=model_from_checkpoint(path,device)
        entry={"checkpoint":str(path),"sha256":sha,"splits":{}}
        for split,(index,folder) in indexes.items():
            positions={g["candidate_group_id"]:i for i,g in enumerate(roots[split].entries)}
            destination=OUTPUT/"trajectory_gradients"/name/split
            destination.mkdir(parents=True,exist_ok=True)
            files=[]
            for number,spec in enumerate(index["entries"]):
                key=hashlib.sha256((spec["group_id"]+'|'+spec["kind"]).encode()).hexdigest()[:24]
                output=destination/(key+'.npz');files.append(output)
                if output.exists():
                    with np.load(output,allow_pickle=False) as cached:
                        if str(cached["sha256"])!=sha:raise ValueError("Cached checkpoint differs")
                    continue
                with np.load(folder/spec["file"],allow_pickle=False) as archive:
                    arrays={k:archive[k] for k in (*INPUTS,"success")}
                if not np.all(arrays["success"]==arrays["success"][0]):raise ValueError("Trajectory outcome labels vary")
                q_values=[];gradient_values=[]
                reference_root=roots[split][positions[spec["group_id"]]]["actions"][0]
                for start in range(0,len(arrays["action"]),32):
                    batch={k:torch.from_numpy(arrays[k][start:start+32]).to(device) for k in INPUTS}
                    action=batch["action"].detach().requires_grad_(True)
                    batch["action"]=action
                    if isinstance(model,AnchoredCritic):
                        reference=action.detach().clone()
                        if spec["kind"]!="stored_source" and start==0:
                            reference[0]=torch.as_tensor(reference_root,device=device)
                        with torch.no_grad():
                            features,base=model.reference_features(batch["prefix"],batch["pad"],batch["robot"],batch["proprio"],reference,batch["action_valid"])
                        z=model.logits_from_features(features,base,action,batch["action_valid"],reference)
                    else:z=model(*(batch[k] for k in INPUTS))
                    g,=torch.autograd.grad(z.sum(),action)
                    g=g*batch["action_valid"][...,None]
                    if not torch.isfinite(g).all() or not torch.isfinite(z).all():raise ValueError("Nonfinite action derivative")
                    q_values.append(z.detach().sigmoid().reshape(-1).cpu().numpy())
                    gradient_values.append(g.detach().cpu().numpy())
                _atomic_npz(output,{"sha256":np.asarray(sha),"group_id":np.asarray(spec["group_id"]),"kind":np.asarray(spec["kind"]),
                                    "q":np.concatenate(q_values),"raw_logit_gradient":np.concatenate(gradient_values),
                                    "action_std":model.action_std.detach().cpu().numpy(),"valid":arrays["action_valid"],"success":arrays["success"]})
                if (number+1)%120==0:print(json.dumps({"model":name,"split":split,"trajectories":number+1,"planned":len(index["entries"])}),flush=True)
            entry["splits"][split]=summarize(files)
            report["models"][name]=entry
            _atomic_json(report_path,report);render(report)
            print(json.dumps({"completed_model_split":name+'/'+split}),flush=True)
        del model
        if device.type=="mps":torch.mps.empty_cache()
    print(json.dumps({"complete":str(OUTPUT/"trajectory_gradient_results.md")}),flush=True)

if __name__=="__main__":main()
