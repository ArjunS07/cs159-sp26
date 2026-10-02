"""Render interpretable offline critic comparisons and log the audit artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path.home()/"pnp-vla-runs/diagnostics/critic_evaluation_audit_v1"
    report = json.loads((directory/"report.json").read_text())
    names = ["pcp_frozen_late_mc","td_n5_continuation","cnn_mc_continuation"]
    labels = ["Frozen MC (PCP)","Continued TD (5-step)","Continued CNN MC"]
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for name,label in zip(names,labels):
        m = report["models"][name]["splits"]["validation"]
        bins = m["calibration_deciles"]
        axes[0].plot([b["mean_q"] for b in bins],[b["success_fraction"] for b in bins],"o-",label=label)
    axes[0].plot([0,1],[0,1],"k--",alpha=.4)
    axes[0].set(xlabel="Predicted success probability",ylabel="Observed success fraction",title="Calibration (fixed deciles)",xlim=(0,1),ylim=(0,1))
    axes[0].legend(fontsize=8)
    metrics = [report["models"][n]["splits"]["validation"] for n in names]
    x = np.arange(len(names))
    axes[1].bar(x,[m["selected_successes_logit_argmax"] for m in metrics])
    for value,label in [(209,"Stock"),(198.11,"Uniform random"),(226,"Oracle")]:
        axes[1].axhline(value,linestyle="--",label=f"{label}: {value:g}")
    axes[1].set(xticks=x,xticklabels=["Frozen MC","TD","CNN MC"],ylabel="Successful episodes / 320",title="Selection on the same recorded states",ylim=(180,235))
    axes[1].legend(fontsize=8)
    g = [m["gradient"] for m in metrics]
    means = np.array([a["fresh_gradient_pair_accuracy"] for a in g])
    intervals = np.array([a["fresh_gradient_pair_episode_bootstrap_95"] for a in g])
    axes[2].errorbar(x,means,yerr=np.stack([means-intervals[:,0],intervals[:,1]-means]),fmt="o",capsize=4)
    axes[2].axhline(.5,color="black",linestyle="--",label="Chance ordering")
    axes[2].set(xticks=x,xticklabels=["Frozen MC","TD","CNN MC"],ylabel="Success/failure displacement ordering",title="Gradient alignment: 81 mixed roots",ylim=(.35,.7))
    axes[2].legend(fontsize=8)
    fig.suptitle("Exploratory reused validation; paired episode bootstrap 95% intervals",fontsize=11)
    path = directory/"core_comparison.png"
    fig.savefig(path,dpi=180)
    plt.close(fig)
    width_names = [n for n in report["models"] if "seed" in n and n.endswith("fixed_final")]
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for name in width_names:
        width = int(name.split("_")[0].replace("width",""))
        m = report["models"][name]["splits"]["validation"]
        axes[0].scatter(width,m["brier"],color="tab:blue")
        axes[1].scatter(width,m["gradient"]["fresh_gradient_pair_accuracy"],color="tab:orange")
    for ax in axes:
        ax.set(xticks=[32,64,128],xlabel="CNN width")
    axes[0].set(ylabel="Validation Brier (lower is better)",title="Each dot is one initialization seed")
    axes[1].axhline(.5,linestyle="--",color="black")
    axes[1].set(ylabel="Fresh gradient pair accuracy",title="Fixed final update: 8,000")
    fig.savefig(directory/"cnn_seed_variability.png",dpi=180)
    plt.close(fig)
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    experiment = mlflow.set_experiment("smolvla_q_function_validation_v1")
    with mlflow.start_run(run_name="common_30_endpoint_offline_audit") as run:
        mlflow.log_params({"model_endpoints":len(report["models"]),"train_episodes":1280,
                           "validation_episodes":320,"validation_status":"exploratory_reused",
                           "new_policy_rollouts":False,"heldout_bellman_available":False})
        for name,entry in report["models"].items():
            m = entry["splits"]["validation"]
            mlflow.log_metrics({name+"/"+k:m[k] for k in ("brier","auroc","mc_log_loss","classification_accuracy_at_fixed_half","selected_successes_logit_argmax")})
        for file in directory.glob("*"):
            if file.is_file():
                mlflow.log_artifact(str(file))
        mlflow.log_artifact(str(Path(__file__).resolve().parents[1]/"docs/smolvla_critic_evaluation_protocol.md"))
        print({"plot":str(path),"mlflow_run":f"http://127.0.0.1:5001/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"})


if __name__ == "__main__":
    main()
