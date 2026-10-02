"""Large cached MC-outcome validation; no training, simulator or paid workers.

The unit of independence is the episode/root, not its nine correlated branches.
Repeatedly inspected validation is exploratory, not an untouched test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path.home() / "pnp-vla-runs"
DIGEST = "671b5b211099997fc83d1277"
REPORTS = {
    "pcp_frozen_late_mc": BASE / "diagnostics/overnight_20260930/mc_roots_late_diagnostics.json",
    "cnn_mc_continuation": BASE / f"checkpoints/smolvla_local_continuation_v1/{DIGEST}/mc_anchored_cnn/final_report.json",
    "td_n5_continuation": BASE / f"checkpoints/smolvla_local_continuation_v1/{DIGEST}/td_n5/final_report.json",
}


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool).ravel(), np.asarray(scores).ravel()
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.empty(len(scores), float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(labels, scores):
    labels, scores = np.asarray(labels, bool).ravel(), np.asarray(scores).ravel()
    if not labels.any():
        return None
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)) + 1, len(scores)]
    tp = labels.cumsum()[ends - 1]
    recall = tp / labels.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * tp / ends))


def prediction_metrics(y, logits):
    q = np.exp(-np.logaddexp(0, -logits))
    brier = np.square(q-y)
    loss = np.logaddexp(0, logits) - y*logits
    pred = q >= .5
    tp, tn = int((pred & y).sum()), int((~pred & ~y).sum())
    positives, negatives = int(y.sum()), int((~y).sum())
    calibration = []
    for lo in np.arange(0, 1, .1):
        mask = (q >= lo) & (q < lo+.1 if lo < .9 else q <= 1)
        if mask.any():
            calibration.append({"lower": round(float(lo), 1), "count": int(mask.sum()),
                "mean_q": float(q[mask].mean()), "success_fraction": float(y[mask].mean())})
    return {"brier": float(brier.mean()), "mc_log_loss": float(loss.mean()),
        "auroc": auc(y, q), "average_precision": average_precision(y, q),
        "classification_accuracy_at_fixed_half": float((pred==y).mean()),
        "success_recall_at_half": tp/positives if positives else None,
        "failure_recall_at_half": tn/negatives if negatives else None,
        "balanced_accuracy_at_half": (tp/positives+tn/negatives)/2 if positives and negatives else None,
        "calibration_deciles": calibration}, q, brier.mean(1)


def summarize(records):
    if not records or len({r["group_id"] for r in records}) != len(records):
        raise ValueError("empty or duplicated root records")
    y = np.asarray([r["labels"] for r in records], bool)
    logits = np.asarray([r["logits"] for r in records], float)
    if y.shape != logits.shape or y.ndim != 2 or y.shape[1] != 9 or not np.isfinite(logits).all():
        raise ValueError("require nine finite scores/outcomes per root")
    main, q, brier_per_root = prediction_metrics(y, logits)
    stock_logits = np.repeat(logits[:, :1], 9, axis=1)
    stock_score, _, stock_brier = prediction_metrics(y, stock_logits)
    # This is an action-information ablation, not a separately trained V model.
    random = y.mean(1)
    chosen = y[np.arange(len(y)), np.argmax(q, axis=1)]
    stock = y[:, 0]
    mixed = y.any(1) & ~y.all(1)
    pairs = []
    for yi, qi in zip(y[mixed], q[mixed]):
        delta = qi[yi, None] - qi[None, ~yi]
        pairs.append(float(((delta>0) + .5*(delta==0)).mean()))
    rng = np.random.default_rng(73026)
    idx = rng.integers(len(y), size=(4000, len(y)))
    def ci(values):
        return np.quantile(np.asarray(values)[idx].mean(1), [.025, .975]).tolist()
    main.update(roots=len(y), candidate_outcomes=int(y.size), mixed_roots=int(mixed.sum()),
        candidate_success_prevalence=float(y.mean()), majority_class_accuracy=max(float(y.mean()),1-float(y.mean())),
        stock_successes=int(stock.sum()), selected_successes=int(chosen.sum()),
        random_selection_expected_successes=float(random.sum()), random_selection_expected_rate=float(random.mean()),
        rescues=int((chosen & ~stock).sum()), spoils=int((~chosen & stock).sum()),
        within_root_pair_accuracy=float(np.mean(pairs)) if pairs else None,
        within_root_q_std_mean=float(q.std(1).mean()),
        brier_root_bootstrap_95=ci(brier_per_root),
        brier_difference_vs_repeated_stock=float((brier_per_root-stock_brier).mean()),
        brier_difference_root_bootstrap_95=ci(brier_per_root-stock_brier),
        selected_minus_random_rate=float((chosen-random).mean()),
        selected_minus_random_root_bootstrap_95=ci(chosen-random),
        repeated_stock_score_baseline=stock_score,
        bootstrap_unit="source episode/root; 4000 replicates, seed73026")
    return main


def main():
    destination = BASE / "diagnostics/q_function_validation_v1"
    destination.mkdir(parents=True, exist_ok=True)
    output = {"interpretation": "Cached final predictions; original repeated validation, exploratory. Same episodes shared across models.",
        "target": "eventual binary success after this candidate and frozen continuation, gamma=1",
        "models": {}}
    root_sets = {}
    for name, path in REPORTS.items():
        report = json.loads(path.read_text())
        result = {"source_report": str(path), "splits": {}}
        for split, data in report["splits"].items():
            records = data.get("root_records") or data["gradients"]["root_records"]
            result["splits"][split] = summarize(records)
            root_sets.setdefault(split, set(r["group_id"] for r in records))
            if root_sets[split] != {r["group_id"] for r in records}:
                raise ValueError("models do not share exact root sets")
        output["models"][name] = result
    if root_sets["train"] & root_sets["validation"]:
        raise ValueError("train/validation root overlap")
    output["unique_roots"] = {k:len(v) for k,v in root_sets.items()}
    (destination / "report.json").write_text(json.dumps(output, indent=2))
    lines = ["# Q-function prediction validation", "", output["interpretation"], "",
        "1,280 training episodes and 320 validation episodes, nine correlated candidates each: 11,520 training and 2,880 validation labels. These observations repeat across models, not additional independent data.", "",
        "| Model | Split | Accuracy at .5 | Majority baseline | AUROC | AP | Brier | MC log loss | Pair accuracy | Stock / selected / expected random successes |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for name, result in output["models"].items():
        for split, m in result["splits"].items():
            lines.append(f"| {name} | {split} | {m['classification_accuracy_at_fixed_half']:.3f} | {m['majority_class_accuracy']:.3f} | {m['auroc']:.3f} | {m['average_precision']:.3f} | {m['brier']:.4f} | {m['mc_log_loss']:.4f} | {m['within_root_pair_accuracy']:.3f} | {m['stock_successes']} / {m['selected_successes']} / {m['random_selection_expected_successes']:.2f} |")
    lines += ["", "Random selection samples one of nine candidates uniformly. Its success probability is the fraction successful at each root. Pairwise chance is .5; top-choice success is not 1/9 unless exactly one candidate succeeds.", "", "Repeated-stock-score baseline, calibration deciles and paired root bootstrap intervals are in report.json. Strong global AUROC can reflect state difficulty; within-root ordering and the repeated-score ablation measure action information. No thresholds or gates were fitted."]
    (destination / "results.md").write_text("\n".join(lines)+"\n")
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    experiment = mlflow.set_experiment("smolvla_q_function_validation_v1")
    with mlflow.start_run(run_name="fixed_final_cached_mc_outcomes") as run:
        mlflow.log_params({"unique_train_episodes":1280,"unique_validation_episodes":320,"candidates_per_root":9,
                           "new_inference":False,"snapshot_digest":DIGEST,"validation_status":"exploratory_reused"})
        for name, result in output["models"].items():
            for split, metrics in result["splits"].items():
                mlflow.log_metrics({f"{name}/{split}/{k}":v for k,v in metrics.items() if isinstance(v,(int,float))})
        mlflow.log_artifact(str(destination / "report.json"));mlflow.log_artifact(str(destination / "results.md"))
        print({"report":str(destination/"results.md"),"mlflow":f"http://127.0.0.1:5001/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"})


if __name__ == "__main__":
    main()
