"""Episode-level standard errors and gradient magnitudes for the three main critics."""
import json
from pathlib import Path
import numpy as np

from scripts.validate_smolvla_q_predictions import BASE, REPORTS, auc
from scripts.audit_smolvla_critics import records_from


def mean_se(values):
    values = np.asarray(values,float)
    return {"mean":float(values.mean()),"standard_error":float(values.std(ddof=1)/np.sqrt(len(values))),"episodes":len(values)}


def auc_episode_bootstrap(y,q,replicates=1000):
    n=len(y)
    order=np.argsort(q.ravel(),kind="stable")
    sorted_q=q.ravel()[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_q))+1]
    positive=y.ravel()[order].astype(float)
    identities=np.repeat(np.arange(n),y.shape[1])[order]
    rng=np.random.default_rng(73026)
    values=[]
    for _ in range(replicates):
        counts=np.bincount(rng.integers(n,size=n),minlength=n)
        weights=counts[identities]
        pos=np.add.reduceat(weights*positive,starts)
        neg=np.add.reduceat(weights*(1-positive),starts)
        if pos.sum() and neg.sum():
            values.append(float((pos*(neg.cumsum()-.5*neg)).sum()/(pos.sum()*neg.sum())))
    return {"mean":auc(y,q),"standard_error":float(np.std(values,ddof=1)),
            "episodes":n,"bootstrap_replicates":replicates,"seed":73026}


def main():
    result = {"standard_error_unit":"episode/root; nine candidates clustered within each root",
              "interpretation":"Conditional on these trained checkpoints; not training-seed uncertainty. Reused validation is exploratory.","models":{}}
    for name,path in REPORTS.items():
        report = json.loads(path.read_text())
        entry = {}
        for split,data in report["splits"].items():
            rows = records_from(data)
            y = np.asarray([r["labels"] for r in rows],bool)
            z = np.asarray([r["logits"] for r in rows],float)
            q = np.exp(-np.logaddexp(0,-z))
            norms = np.asarray([r.get("gradient_norm",r.get("normalized_gradient_norm")) for r in rows],float)
            q_norms = norms*q[:,0]*(1-q[:,0])
            gp,action_pairs = [],[]
            for r in rows:
                all_y=np.asarray(r["labels"],bool)
                all_scores=np.asarray(r["logits"],float)
                if all_y.any() and (~all_y).any():
                    delta=all_scores[all_y,None]-all_scores[None,~all_y]
                    action_pairs.append(((delta>0)+.5*(delta==0)).mean())
                yi = np.asarray(r["labels"],bool)[1:]
                projection = np.asarray(r["gradient_projection"],float)[1:]
                if yi.any() and (~yi).any():
                    delta = projection[yi,None]-projection[None,~yi]
                    gp.append(((delta>0)+.5*(delta==0)).mean())
            entry[split] = {"episodes":len(rows),"candidate_outcomes":int(y.size),
                            "brier":mean_se(np.square(q-y).mean(1)),
                            "mc_log_loss":mean_se((np.logaddexp(0,z)-y*z).mean(1)),
                            "classification_error":mean_se(((q>=.5)!=y).mean(1)),
                            "auroc":auc_episode_bootstrap(y,q),
                            "action_pair_accuracy":mean_se(action_pairs),
                            "selected_success_rate":mean_se(y[np.arange(len(y)),z.argmax(1)]),
                            "fresh_gradient_pair_accuracy":mean_se(gp),
                            "logit_gradient_standardized_L2":mean_se(norms),
                            "probability_gradient_standardized_L2":mean_se(q_norms),
                            "logit_gradient_median_p95":np.quantile(norms,[.5,.95]).tolist(),
                            "probability_gradient_median_p95":np.quantile(q_norms,[.5,.95]).tolist(),
                            "gradient_units":"L2 norm of d(logit or probability)/d standardized action coordinates, masked to known-valid actions. CNN stock reference held fixed."}
        result["models"][name] = entry
    destination = BASE/"diagnostics/critic_evaluation_audit_v1"
    (destination/"standard_errors_and_gradients.json").write_text(json.dumps(result,indent=2))
    lines = ["# Episode-level standard errors and gradient magnitudes","",result["standard_error_unit"]+". "+result["interpretation"],"",
             "| Model | Split | Episodes / outcomes | Brier ± SE | MC log loss ± SE | Classification error ± SE | AUROC ± SE | Gradient ordering ± SE (mixed episodes) |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for name,entry in result["models"].items():
        for split,m in entry.items():
            def show(key):
                a=m[key]; return f"{a['mean']:.4f} ± {a['standard_error']:.4f}"
            lines.append(f"| {name} | {split} | {m['episodes']} / {m['candidate_outcomes']} | {show('brier')} | {show('mc_log_loss')} | {show('classification_error')} | {show('auroc')} | {show('fresh_gradient_pair_accuracy')} ({m['fresh_gradient_pair_accuracy']['episodes']}) |")
    lines += ["","## Gradient magnitude","","| Model | Split | Logit gradient L2 mean ± SE | Probability gradient L2 mean ± SE | Probability gradient median / p95 |","|---|---|---:|---:|---:|"]
    for name,entry in result["models"].items():
        for split,m in entry.items():
            a,b=m['logit_gradient_standardized_L2'],m['probability_gradient_standardized_L2']
            lo,hi=m['probability_gradient_median_p95']
            lines.append(f"| {name} | {split} | {a['mean']:.6f} ± {a['standard_error']:.6f} | {b['mean']:.6f} ± {b['standard_error']:.6f} | {lo:.6f} / {hi:.6f} |")
    lines += ["","Magnitudes use standardized action coordinates (training action standard deviations), not raw action units or the norm of a normalized PCP correction. Probability gradients include the sigmoid factor. Large gradients need not point toward better actions. Every magnitude summary uses all episodes in its split, not only mixed episodes."]
    path=destination/"standard_errors_and_gradients.md"
    path.write_text('\n'.join(lines)+'\n')
    print(path)


if __name__ == "__main__":
    main()
