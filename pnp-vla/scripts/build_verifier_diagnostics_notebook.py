"""Generate the verifier learning-diagnostics notebook."""
from __future__ import annotations

import json

from build_verifier_notebooks import BOOTSTRAP, ROOT, code, md, notebook


CELLS = [
    md("""# 07 — Verifier training diagnostics

Audit optimization, fold/seed stability, and controls from the completed W&B
runs, then estimate a 25/50/75/100% training-size curve using development data
only. Neither prospective test set is loaded."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    code(r'''from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import copy, hashlib, json, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

from pnp.store import SupabaseStore
from pnp.verifier import *

VERIFIER_ID = "3add05c827424c4a"
WANDB_PATH = "arjun-sharma07-california-institute-of-technology-caltech/pnp-clean-verifier"
CONFIGS = (
    "rank_only", "rank_plus_bce", "rank_plus_initial_pairs",
    "action_only_control", "shuffled_action_control")
SEEDS = (42, 43, 44)
N_FOLDS = 4
FRACTIONS = (.25, .50, .75, 1.0)
OUTPUT = Path(package_dir) / "analysis_outputs" / "verifier_diagnostics"
OUTPUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store = SupabaseStore()
checkpoint, verifier_row = store.load_verifier(VERIFIER_ID)
metadata = checkpoint["metadata"]
SELECTED_CONFIG = metadata["selected_config"]
print({"device": str(DEVICE), "selected_config": SELECTED_CONFIG,
       "checkpoint_train_config": metadata["train"]})'''),
    md("## 2. Pull and audit the latest 60 cross-validation runs"),
    code(r'''api = wandb.Api()
runs = list(api.runs(WANDB_PATH))
expected = {f"{c}-f{f}-s{s}" for c in CONFIGS
            for f in range(N_FOLDS) for s in SEEDS}
latest = {}
for run in runs:
    if run.name in expected:
        previous = latest.get(run.name)
        if previous is None or run.created_at > previous.created_at:
            latest[run.name] = run
assert expected == set(latest), f"missing runs: {sorted(expected-set(latest))[:10]}"

summary_rows, history_rows = [], []
for name, run in tqdm(sorted(latest.items()), desc="W&B runs"):
    config_name, fold_text, seed_text = name.rsplit("-", 2)
    fold, seed = int(fold_text[1:]), int(seed_text[1:])
    summary_rows.append({
        "config": config_name, "fold": fold, "seed": seed,
        "ranking": run.summary.get("oof/group_macro_ranking_accuracy"),
        "top1": run.summary.get("oof/top1_success"),
        "default": run.summary.get("oof/default_success"),
        "random": run.summary.get("oof/random_success"),
        "epochs": (run.summary.get("rank/epoch") or 0) + 1,
    })
    history = run.history(keys=[
        "rank/epoch", "rank/train_loss",
        "rank/val_group_macro_ranking_accuracy",
        "rank/val_top1_uplift_default", "rank/val_top1_uplift_random"],
        pandas=True)
    for row in history.to_dict("records"):
        if pd.notna(row.get("rank/epoch")):
            history_rows.append({
                "config": config_name, "fold": fold, "seed": seed,
                "epoch": int(row["rank/epoch"]),
                "train_loss": row.get("rank/train_loss"),
                "val_ranking": row.get("rank/val_group_macro_ranking_accuracy"),
                "val_uplift_default": row.get("rank/val_top1_uplift_default"),
                "val_uplift_random": row.get("rank/val_top1_uplift_random"),
            })
summaries, histories = pd.DataFrame(summary_rows), pd.DataFrame(history_rows)
summaries.to_csv(OUTPUT / "wandb_run_summaries.csv", index=False)
histories.to_csv(OUTPUT / "wandb_epoch_histories.csv", index=False)
display(summaries.groupby("config").agg(
    ranking_mean=("ranking", "mean"), ranking_std=("ranking", "std"),
    top1=("top1", "mean"), default=("default", "mean"),
    random=("random", "mean"), epochs=("epochs", "mean")))'''),
    md("## 3. Optimization, stability, and leakage controls"),
    code(r'''fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for config_name, frame in histories.groupby("config"):
    curve = frame.groupby("epoch").agg(
        train_loss=("train_loss", "mean"), val_ranking=("val_ranking", "mean"))
    axes[0].plot(curve.index, curve.train_loss, label=config_name)
    axes[1].plot(curve.index, curve.val_ranking, label=config_name)
axes[0].set(title="Ranking loss", xlabel="epoch", ylabel="train loss")
axes[1].axhline(.5, color="black", linestyle="--", linewidth=1)
axes[1].set(title="Held-out same-state ranking", xlabel="epoch", ylabel="accuracy")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

selected = summaries[summaries.config == SELECTED_CONFIG]
pivot = summaries.pivot_table(index=["fold", "seed"], columns="config", values="ranking")
control_gaps = {
    "vs_action_only": float((pivot[SELECTED_CONFIG] - pivot.action_only_control).mean()),
    "vs_shuffled": float((pivot[SELECTED_CONFIG] - pivot.shuffled_action_control).mean()),
    "shuffled_distance_from_chance": float(abs(pivot.shuffled_action_control.mean() - .5)),
}
epoch_rows=[]
for (fold, seed), frame in histories[
        histories.config == SELECTED_CONFIG].groupby(["fold", "seed"]):
    frame = frame.dropna(subset=["val_ranking"]).sort_values("epoch")
    best, final = frame.loc[frame.val_ranking.idxmax()], frame.iloc[-1]
    epoch_rows.append({
        "fold": fold, "seed": seed, "best_epoch": int(best.epoch),
        "best_val": float(best.val_ranking), "final_val": float(final.val_ranking),
        "late_drop": float(best.val_ranking-final.val_ranking)})
epoch_diagnostics = pd.DataFrame(epoch_rows)
print("selected stability", {
    "runs_above_chance": int((selected.ranking > .5).sum()), "runs": len(selected),
    "min": float(selected.ranking.min()), "max": float(selected.ranking.max()),
    "std": float(selected.ranking.std())})
print("paired control gaps", control_gaps)
display(epoch_diagnostics)
epoch_diagnostics.to_csv(OUTPUT / "selected_epoch_diagnostics.csv", index=False)'''),
    md("## 4. Load development data without test outcomes"),
    code(r'''HISTORICAL = ("libero-hybrid-schedules-k3-v1",
              "libero-pro-canonical-core-k3-v1")
V3, V4_DEV, V4_TEST = (
    "verifier-clean-pairs-v3", "verifier-clean-pairs-v4-dev",
    "verifier-clean-pairs-v4-test")
AUX = ("verifier-clean-pairs-v1", "verifier-clean-pairs-v2")
cache = OUTPUT / "cache"
historical_path = cache / "historical.pkl"
if historical_path.exists():
    historical = pickle.loads(historical_path.read_bytes())
else:
    historical = load_clean_chunk_examples(
        store, HISTORICAL, progress=tqdm, cache_dir=cache / "historical_rollouts")
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    historical_path.write_bytes(pickle.dumps(historical))
development = (
    load_candidate_examples(store, V3, cache_dir=cache / "v3") +
    load_candidate_examples(store, V4_DEV, cache_dir=cache / "v4_dev"))
auxiliary = load_candidate_examples(store, AUX, cache_dir=cache / "aux")

test_rows=[]; start=0
while True:
    batch = (store.client.table("verifier_candidate_groups")
             .select("benchmark,suite,task_idx,episode_idx")
             .eq("experiment", V4_TEST).order("candidate_group_id")
             .range(start, start+999).execute().data or [])
    test_rows += batch
    if len(batch) < 1000: break
    start += 1000
test_identities = {(r["benchmark"], r["suite"], int(r["task_idx"]), int(r["episode_idx"]))
                   for r in test_rows}
folds = candidate_cv_splits(
    development, [e.rollout_id for e in development], N_FOLDS, seed=42)
print({"historical_chunks": len(historical),
       "development_groups": len({e.candidate_group_id for e in development}),
       "auxiliary_groups": len({e.candidate_group_id for e in auxiliary}),
       "sealed_test_identities_only": len(test_identities)})'''),
    md("## 5. Controlled training-size curve"),
    code(r'''def fraction_subset(examples, fraction, seed=42):
    groups=defaultdict(list)
    for example in examples: groups[example.candidate_group_id].append(example)
    buckets=defaultdict(list)
    for gid, members in groups.items():
        first=members[0]
        buckets[(first.benchmark, len({m.success for m in members}) > 1)].append(gid)
    selected=[]
    for key, gids in sorted(buckets.items()):
        gids=sorted(gids, key=lambda gid: hashlib.sha256(f"{seed}|{gid}".encode()).hexdigest())
        count=len(gids) if fraction == 1 else max(1, round(len(gids)*fraction))
        selected += gids[:count]
    wanted=set(selected)
    return [e for e in examples if e.candidate_group_id in wanted]

cfg0 = AdvantageTrainConfig(**{
    key: value for key, value in metadata["train"].items()
    if key in AdvantageTrainConfig.__dataclass_fields__})
include_aux = SELECTED_CONFIG == "rank_plus_initial_pairs"
size_rows=[]
for fold_index, fold in enumerate(folds):
    fold_train, fold_val = select_examples(development, fold["train"]), select_examples(development, fold["val"])
    protected = candidate_episode_identities(fold_val) | test_identities
    fold_historical = exclude_episode_identities(historical, protected)
    split = known_task_split(fold_historical, seed=42+fold_index)
    value_val = select_examples(fold_historical, split["val"])
    value_val_ids=set(split["val"])
    value_train=[e for e in fold_historical if e.rollout_id not in value_val_ids]
    clean_aux=exclude_episode_identities(auxiliary, protected)
    for seed in SEEDS:
        cfg=replace(cfg0, seed=seed)
        base=CompactAdvantageVerifier()
        base, _ = pretrain_value(base, value_train, value_val, DEVICE, config=cfg)
        for fraction in FRACTIONS:
            candidates=fraction_subset(fold_train, fraction)
            if include_aux: candidates += fraction_subset(clean_aux, fraction)
            model=copy.deepcopy(base)
            mean, std=prefix_action_statistics(candidates, cfg.prefix_length)
            model.set_action_statistics(mean, std)
            model, rank_meta=train_advantage(model, candidates, fold_val, DEVICE, config=cfg)
            metrics=evaluate_candidate_ranker(model, fold_val, DEVICE, config=cfg)
            groups={e.candidate_group_id for e in candidates}
            grouped=defaultdict(set)
            for e in candidates: grouped[e.candidate_group_id].add(e.success)
            size_rows.append({
                "fraction": fraction, "fold": fold_index, "seed": seed,
                "train_groups": len(groups),
                "train_discordant": sum(len(labels)>1 for labels in grouped.values()),
                **rank_meta, **{k:v for k,v in metrics.items() if not isinstance(v, dict)}})
            print(size_rows[-1])
size_curve=pd.DataFrame(size_rows)
size_curve.to_csv(OUTPUT / "training_size_curve.csv", index=False)
display(size_curve.groupby("fraction").agg(
    groups=("train_groups","mean"), discordant=("train_discordant","mean"),
    ranking=("group_macro_ranking_accuracy","mean"),
    ranking_std=("group_macro_ranking_accuracy","std"),
    uplift_default=("top1_uplift_default","mean"),
    uplift_random=("top1_uplift_random","mean")))'''),
    md("## 6. Decision summary"),
    code(r'''curve=size_curve.groupby("fraction").agg(
    ranking=("group_macro_ranking_accuracy","mean"),
    ranking_std=("group_macro_ranking_accuracy","std"),
    uplift_random=("top1_uplift_random","mean")).reset_index()
last_gain=float(curve.iloc[-1].ranking-curve.iloc[-2].ranking)
diagnosis={
    "selected_config": SELECTED_CONFIG,
    "last_quarter_ranking_gain": last_gain,
    "data_limited_signal": last_gain > .01,
    "control_gap_vs_action_only": control_gaps["vs_action_only"],
    "control_gap_vs_shuffled": control_gaps["vs_shuffled"],
    "median_late_validation_drop": float(epoch_diagnostics.late_drop.median()),
    "recommendation": (
        "collect more discordant same-state groups" if last_gain > .01 else
        "test a modest context-action architecture change before more collection")}
(OUTPUT / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2))
display(curve); print(json.dumps(diagnosis, indent=2))'''),
]


def main():
    path = ROOT / "notebooks" / "07_verifier_training_diagnostics.ipynb"
    path.write_text(json.dumps(notebook(CELLS), indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
