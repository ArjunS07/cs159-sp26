"""Constant-Q controls for interpreting frozen-critic Bellman residuals."""
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_smolvla_bellman import BASE, DIGEST, FORMAT, OUTPUT, transition_errors, summarize_episodes


def main():
    training=BASE/"cache/smolvla_scalar_returns_v1_preaction"/DIGEST/"cache_index.json"
    validation=BASE/"cache"/FORMAT/DIGEST/"cache_index.json"
    train=json.loads(training.read_text())
    prevalence=float(np.mean([e["success"] for e in train["entries"]]))
    output={"train_only_equal_trajectory_success_prevalence":prevalence,"baselines":{}}
    for name,value in (("constant_zero",0.),("constant_one",1.),("train_prevalence",prevalence)):
        result={}
        for split,path in (("train",training),("validation",validation)):
            index=json.loads(path.read_text())
            rows={1:[],5:[]}
            for entry in index["entries"]:
                with np.load(path.parent/entry["file"],allow_pickle=False) as archive:
                    reward,discount,success=(archive[key] for key in ("reward","discount","success"))
                q=np.full(len(reward),value)
                for horizon in (1,5):
                    residual,terminal=transition_errors(q,reward,discount,horizon)
                    rows[horizon].append({"group_id":entry["group_id"],"transitions":len(q),
                                          "residual_mean":float(residual.mean()),"residual_mae":float(np.abs(residual).mean()),
                                          "residual_mse":float(np.square(residual).mean()),"mc_mse":float(np.square(q-success).mean()),
                                          "terminal_mse":float(np.square(residual[terminal]).mean()) if terminal.any() else None,
                                          "nonterminal_mse":float(np.square(residual[~terminal]).mean()) if (~terminal).any() else None})
            result[split]={str(h):summarize_episodes(r) for h,r in rows.items()}
        output["baselines"][name]=result
    (OUTPUT/"bellman_baselines.json").write_text(json.dumps(output,indent=2))
    lines=["# Constant-Q Bellman baselines","",f"Training-only success prevalence: {prevalence:.6f}. Same trajectories, episode/trajectory weighting and terminal contract as the learned critics.","",
           "| Baseline | Split | One-step Bellman RMSE ± SE | Five-step Bellman RMSE ± SE | MC return RMSE ± SE |","|---|---|---:|---:|---:|"]
    for name,splits in output["baselines"].items():
        for split,metrics in splits.items():
            def show(h,key):
                v=metrics[h][key]
                return f"{v['value']:.4f} ± {v['standard_error_delta_method']:.4f}"
            lines.append(f"| {name} | {split} | {show('1','residual_rmse')} | {show('5','residual_rmse')} | {show('1','mc_rmse')} |")
    lines += ["","Every constant-Q baseline has exactly zero residual away from reward or terminal boundaries. A low Bellman residual therefore cannot, by itself, demonstrate accurate outcomes or useful gradients. Constant-Q gradients are zero."]
    (OUTPUT/"bellman_baselines.md").write_text('\n'.join(lines)+'\n')
    print(OUTPUT/"bellman_baselines.md")


if __name__=="__main__":main()
