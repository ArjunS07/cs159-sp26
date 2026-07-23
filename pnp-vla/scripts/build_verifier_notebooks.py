"""Generate the verifier experiment and paired-collection notebooks."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def notebook(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


BOOTSTRAP = r'''import os, subprocess, sys
try:
    from google.colab import userdata
    for key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "HF_TOKEN", "WANDB_API_KEY"):
        value = userdata.get(key)
        if value:
            os.environ[key] = value
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", package_dir + "[analysis]"], check=True)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)
import pnp
print("Loaded pnp from:", pnp.__file__)'''


LEGACY_EXPERIMENT_CELLS = [
    md("# 03 — Clean t=1 verifier experiments\n\nEach section is an explicit experiment. Run setup and data cells once; run model cells independently."),
    md("## 1. Environment setup"), code(BOOTSTRAP),
    md("## 2. Configuration and reproducibility"),
    code(r'''from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb

from pnp.store import SupabaseStore
from pnp.verifier import *

EXPERIMENTS = ("libero-hybrid-schedules-k3-v1", "libero-pro-canonical-core-k3-v1")
OUTPUT = Path(package_dir) / "analysis_outputs" / "verifier"
OUTPUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)
PREFIXES = (5, 10, 25, 50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_WANDB = bool(os.getenv("WANDB_API_KEY"))
store = SupabaseStore()
print({"device": str(DEVICE), "output": str(OUTPUT), "wandb": USE_WANDB})'''),
    md("## 3. Download and reconstruct clean environment-space chunks"),
    code(r'''CACHE = OUTPUT / "clean_examples.pkl"
if CACHE.exists():
    with CACHE.open("rb") as handle:
        examples = pickle.load(handle)
else:
    examples = load_clean_chunk_examples(store, EXPERIMENTS, progress=tqdm)
    with CACHE.open("wb") as handle:
        pickle.dump(examples, handle)
print(f"{len(examples):,} chunks from {len({e.rollout_id for e in examples}):,} rollouts")'''),
    md("## 4. Dataset audit: outcomes, masks, benchmarks, and task balance"),
    code(r'''audit = pd.DataFrame([{
    "rollout_id": e.rollout_id, "benchmark": e.benchmark, "suite": e.suite,
    "task_idx": e.task_idx, "chunk_idx": e.chunk_idx, "success": e.success,
    "valid_actions": int(e.action_mask.sum()),
} for e in examples])
display(audit.groupby("benchmark").agg(
    chunks=("chunk_idx", "size"), rollouts=("rollout_id", "nunique"),
    success_rate=("success", "mean"), partial_chunks=("valid_actions", lambda x: int((x < 50).sum()))))
display(audit.groupby(["benchmark", "suite", "task_idx"])["success"].mean().describe())'''),
    md("## 5. Primary mixed-outcome cohort and deterministic split manifests"),
    code(r'''hard = hard_task_keys(examples)
hard_examples = [e for e in examples if e.task_key in hard]
known_split = known_task_split(hard_examples, seed=42)
heldout_split = heldout_task_split(hard_examples, fold=0, seed=42)
manifests = {"known": known_split, "heldout_fold0": heldout_split}
(OUTPUT / "splits.json").write_text(json.dumps(manifests, indent=2, sort_keys=True))
print(f"hard tasks={len(hard)} chunks={len(hard_examples)}")
print({kind: {k: len(v) for k, v in split.items()} for kind, split in manifests.items()})'''),
    md("## 6. Reusable logged experiment runner"),
    code(r'''def run_experiment(name, model_factory, split, cfg, dataset=hard_examples,
                   paired_dataset=None, paired_split=None):
    parts = {key: select_examples(dataset, ids) for key, ids in split.items()}
    paired_split = paired_split or split
    paired_parts = ({key: select_examples(paired_dataset, ids) for key, ids in paired_split.items()}
                    if paired_dataset else {key: None for key in split})
    run = wandb.init(project="pnp-clean-verifier", name=name, config=cfg.__dict__,
                     mode="online" if USE_WANDB else "disabled", reinit=True)
    model = model_factory()
    action_mean, action_std = action_statistics(parts["train"])
    model.set_action_statistics(action_mean, action_std)
    model, metadata = train_verifier(
        model, parts["train"], parts["val"], DEVICE, config=cfg, wandb_run=run,
        paired_train_examples=paired_parts["train"], paired_val_examples=paired_parts["val"])
    scaler = calibrate_temperature(model, parts["cal"], DEVICE, config=cfg)
    metrics = evaluate_verifier(model, parts["test"], DEVICE, config=cfg, scaler=scaler,
                                paired_examples=paired_parts["test"])
    run.log({f"test/{k}": v for k, v in metrics.items()}); run.finish()
    result = {"name": name, "model": model, "scaler": scaler, "config": cfg,
              "metadata": metadata, "metrics": metrics, "split": split}
    print(name, json.dumps(metrics, indent=2))
    return result

results = {}'''),
    md("## 7. Observation-only shortcut baseline"),
    code(r'''# Empirical task-prior baseline (diagnostic only; task identity is never fed to the model).
train_records = {e.rollout_id: e for e in select_examples(hard_examples, known_split["train"])}
test_records = {e.rollout_id: e for e in select_examples(hard_examples, known_split["test"])}
task_rate = pd.DataFrame([{"task": e.task_key, "success": e.success} for e in train_records.values()]).groupby("task").success.mean()
prior_labels = [e.success for e in test_records.values()]
prior_probs = [float(task_rate.get(e.task_key, np.mean(prior_labels))) for e in test_records.values()]
print("task-prior", classification_metrics(prior_labels, prior_probs,
      [str(e.task_key) for e in test_records.values()]))

for seed in SEEDS:
    cfg = VerifierTrainConfig(seed=seed, score_head="state", prefix_length=10)
    results[f"state_seed{seed}"] = run_experiment(
        f"state-seed{seed}", lambda: CleanChunkVerifier(), known_split, cfg)'''),
    md("## 8. Historical flattened-MLP baseline"),
    code(r'''for seed in SEEDS:
    cfg = VerifierTrainConfig(seed=seed, prefix_length=50)
    results[f"flat_seed{seed}"] = run_experiment(
        f"flat-seed{seed}", lambda: FlattenedVerifier(), known_split, cfg)'''),
    md("## 9. Temporal ConvNet and 5/10/25/50 prefix sweep"),
    code(r'''for prefix in PREFIXES:
    for seed in SEEDS:
        cfg = VerifierTrainConfig(seed=seed, prefix_length=prefix)
        key = f"tcn_p{prefix}_seed{seed}"
        results[key] = run_experiment(key, lambda: CleanChunkVerifier(), known_split, cfg)'''),
    md("## 10. Action-only diagnostic"),
    code(r'''for seed in SEEDS:
    cfg = VerifierTrainConfig(seed=seed, prefix_length=10, zero_observation=True)
    results[f"action_seed{seed}"] = run_experiment(
        f"action-seed{seed}", lambda: CleanChunkVerifier(), known_split, cfg)'''),
    md("## 11. Within-task shuffled-action shortcut test"),
    code(r'''shuffled = shuffle_actions_within_task(hard_examples, seed=42)
for seed in SEEDS:
    cfg = VerifierTrainConfig(seed=seed, prefix_length=10)
    results[f"shuffled_seed{seed}"] = run_experiment(
        f"shuffled-seed{seed}", lambda: CleanChunkVerifier(), known_split, cfg,
        dataset=shuffled)'''),
    md("## 11b. Zero-action shortcut test"),
    code(r'''zeroed = zero_actions(hard_examples)
for seed in SEEDS:
    cfg = VerifierTrainConfig(seed=seed, prefix_length=10)
    results[f"zero_action_seed{seed}"] = run_experiment(
        f"zero-action-seed{seed}", lambda: CleanChunkVerifier(), known_split, cfg,
        dataset=zeroed)'''),
    md("## 12. Cohort ablations: all tasks, LIBERO, PRO, and transfer"),
    code(r'''cohort_results = {}
for label, cohort in {
    "all": examples,
    "libero_hard": [e for e in hard_examples if e.benchmark == "libero"],
    "pro_hard": [e for e in hard_examples if e.benchmark == "libero_pro"],
}.items():
    split = known_task_split(cohort, seed=42)
    cfg = VerifierTrainConfig(seed=42, prefix_length=10)
    cohort_results[label] = run_experiment(
        f"cohort-{label}", lambda: CleanChunkVerifier(), split, cfg, dataset=cohort)
results.update({f"cohort_{k}": v for k, v in cohort_results.items()})'''),
    md("## 12b. Cross-benchmark transfer"),
    code(r'''libero_hard = [e for e in hard_examples if e.benchmark == "libero"]
pro_hard = [e for e in hard_examples if e.benchmark == "libero_pro"]
for source, target, trained in (
    ("libero", pro_hard, cohort_results["libero_hard"]),
    ("pro", libero_hard, cohort_results["pro_hard"]),
):
    transfer = evaluate_verifier(trained["model"], target, DEVICE,
                                 config=trained["config"], scaler=trained["scaler"])
    print(source, "transfer", json.dumps(transfer, indent=2))'''),
    md("## 13. Held-out-task evaluation for the selected prefix"),
    code(r'''heldout_results = {}
SELECTED_PREFIX = 10
for fold in range(5):
    split = heldout_task_split(hard_examples, fold=fold, seed=42)
    for seed in SEEDS:
        cfg = VerifierTrainConfig(seed=seed, prefix_length=SELECTED_PREFIX)
        key = f"heldout_f{fold}_seed{seed}"
        heldout_results[key] = run_experiment(
            key, lambda: CleanChunkVerifier(), split, cfg)
results.update(heldout_results)'''),
    md("## 14. Composite eligibility and result table"),
    code(r'''result_table = pd.DataFrame([
    {"name": key, **value["metrics"]} for key, value in results.items()
]).sort_values(["task_macro_pr_auc", "brier"], ascending=[False, True])
display(result_table)
result_table.to_csv(OUTPUT / "experiment_results.csv", index=False)
print("Eligibility requires joint > state and shuffled controls in at least 2/3 seeds;")
print("after paired data, rank eligible configurations equally by PR-AUC, pair accuracy, and -Brier.")'''),
    md("## 15. Register one selected calibrated checkpoint"),
    code(r'''SELECTED_KEY = None  # Set explicitly after reviewing the table; e.g. "tcn_p10_seed42"
if SELECTED_KEY:
    selected = results[SELECTED_KEY]
    verifier_id = new_verifier_id()
    config = {"model_class": type(selected["model"]).__name__, "obs_dim": 2048,
              "action_dim": 7, "horizon": 50,
              "prefix_length": selected["config"].prefix_length,
              "train": selected["config"].__dict__}
    store.start_run("verifier_train", "libero+libero_pro", "clean-verifier-v1")
    store.register_verifier(
        verifier_id, verifier_checkpoint_bytes(selected["model"], selected["scaler"], config),
        config, selected["metrics"], selected["split"], dataset_hash=dataset_hash(hard_examples))
    store.finish_run(n_rollouts=0)
    print("registered", verifier_id)
else:
    print("Set SELECTED_KEY only after reviewing shortcut controls and held-out results.")'''),
    md("## 16. Paired-data retraining (run after notebook 04)"),
    code(r'''paired_examples = load_candidate_examples(
    store, ("verifier-clean-pairs-v1", "verifier-clean-pairs-v2",
            "verifier-clean-pairs-v3"))
if paired_examples:
    paired_split = candidate_group_split(paired_examples, seed=42)
    cfg = VerifierTrainConfig(seed=42, prefix_length=10)
    results["tcn_with_pairs"] = run_experiment(
        "tcn-with-pairs", lambda: CleanChunkVerifier(), known_split, cfg,
        paired_dataset=paired_examples, paired_split=paired_split)
    print("paired examples:", len(paired_examples))
else:
    print("No paired candidate artifacts yet; run notebook 04 first.")'''),
]


EXPERIMENT_CELLS = [
    md("""# 03 — Same-state action-advantage verifier

This notebook supersedes the historical classification sweep. It pretrains
`V(s)` on ordinary LIBERO/LIBERO-PRO rollouts, freezes that pathway, and learns
`A(s,a)` from deterministic same-state candidate groups. Model selection uses
grouped out-of-fold ranking; the locked 20% test set is opened once at the end."""),
    md("## 1. Environment setup"), code(BOOTSTRAP),
    md("## 2. Configuration"),
    code(r'''from dataclasses import asdict, replace
from pathlib import Path
import copy, json, pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb

from pnp.store import SupabaseStore
from pnp.verifier import *

HISTORICAL_EXPERIMENTS = (
    "libero-hybrid-schedules-k3-v1",
    "libero-pro-canonical-core-k3-v1",
)
PRIMARY_CANDIDATES = "verifier-clean-pairs-v3"
AUXILIARY_CANDIDATES = ("verifier-clean-pairs-v1", "verifier-clean-pairs-v2")
SEEDS = (42, 43, 44)
N_FOLDS = 4
PREFIX_LENGTH = 10
OUTPUT = Path(package_dir) / "analysis_outputs" / "verifier"
OUTPUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_WANDB = bool(os.getenv("WANDB_API_KEY"))
REGISTER_IF_ELIGIBLE = True
store = SupabaseStore()
print({"device": str(DEVICE), "output": str(OUTPUT), "wandb": USE_WANDB})'''),
    md("## 3. Load and audit historical and candidate data"),
    code(r'''CACHE = OUTPUT / "historical_clean_examples_v2.pkl"
if CACHE.exists():
    with CACHE.open("rb") as handle:
        historical = pickle.load(handle)
else:
    historical = load_clean_chunk_examples(
        store, HISTORICAL_EXPERIMENTS, progress=tqdm)
    with CACHE.open("wb") as handle:
        pickle.dump(historical, handle)

primary = load_candidate_examples(store, PRIMARY_CANDIDATES)
auxiliary = load_candidate_examples(store, AUXILIARY_CANDIDATES)

def candidate_audit(examples):
    frame = pd.DataFrame([{
        "group": e.candidate_group_id, "benchmark": e.benchmark,
        "stratum": e.uncertainty_stratum, "success": e.success,
    } for e in examples])
    group_outcomes = frame.groupby("group").success.agg(["size", "nunique"])
    return {
        "groups": int(frame.group.nunique()), "outcomes": len(frame),
        "successes": int(frame.success.sum()), "failures": int((1-frame.success).sum()),
        "discordant_groups": int((group_outcomes.nunique == 2).sum()),
        "pairwise_comparisons": int(frame.groupby("group").success.apply(
            lambda y: int(y.sum()) * int((1-y).sum())).sum()),
    }

print("historical", {
    "chunks": len(historical),
    "rollouts": len({e.rollout_id for e in historical}),
    "hash": dataset_hash(historical),
})
print("primary", candidate_audit(primary))
print("auxiliary", candidate_audit(auxiliary))
legacy = OUTPUT / "experiment_results.csv"
if legacy.exists() and not (OUTPUT / "legacy_experiment_results.csv").exists():
    (OUTPUT / "legacy_experiment_results.csv").write_bytes(legacy.read_bytes())'''),
    md("## 4. Lock the test set and write immutable manifests"),
    code(r'''locked = locked_candidate_split(primary, test_fraction=.20, seed=42)
development = select_examples(primary, locked["development"])
locked_test = select_examples(primary, locked["test"])
folds = candidate_cv_splits(primary, locked["development"], n_folds=N_FOLDS, seed=42)

split_manifest = {
    "primary_experiment": PRIMARY_CANDIDATES,
    "primary_dataset_hash": dataset_hash(primary),
    "historical_dataset_hash": dataset_hash(historical),
    "development_candidate_ids": sorted(locked["development"]),
    "locked_test_candidate_ids": sorted(locked["test"]),
    "folds": folds,
}
(OUTPUT / "advantage_splits.json").write_text(
    json.dumps(split_manifest, indent=2, sort_keys=True))
print({
    "development": candidate_audit(development),
    "locked_test": candidate_audit(locked_test),
    "manifest": str(OUTPUT / "advantage_splits.json"),
})'''),
    md("## 5. Grouped 4-fold CV × 3 seeds"),
    code(r'''CONFIGS = {
    "rank_only_v3": {"candidate_bce_weight": 0.0},
    "rank_plus_bce_v3": {"candidate_bce_weight": 0.1},
    "rank_plus_initial_pairs": {"candidate_bce_weight": 0.0, "include_aux": True},
    "action_only_control": {"candidate_bce_weight": 0.0, "zero_context": True},
    "shuffled_action_control": {"candidate_bce_weight": 0.0, "shuffle": True},
}
ELIGIBLE_CONFIGS = {
    "rank_only_v3", "rank_plus_bce_v3", "rank_plus_initial_pairs"}
cv_rows, oof_records, epoch_rows = [], [], []

for fold_index, fold in enumerate(folds):
    fold_train = select_examples(primary, fold["train"])
    fold_val = select_examples(primary, fold["val"])
    protected = fold_val + locked_test
    fold_historical = exclude_candidate_identities(historical, protected)
    historical_split = known_task_split(fold_historical, seed=42 + fold_index)
    value_train = select_examples(fold_historical, historical_split["train"])
    value_val = select_examples(fold_historical, historical_split["val"])
    clean_aux = exclude_candidate_identities(auxiliary, protected)

    for seed in SEEDS:
        base_cfg = AdvantageTrainConfig(seed=seed, prefix_length=PREFIX_LENGTH)
        base = CompactAdvantageVerifier()
        base, value_meta = pretrain_value(
            base, value_train, value_val, DEVICE, config=base_cfg)

        for name, knobs in CONFIGS.items():
            train_candidates = list(fold_train)
            if knobs.get("include_aux"):
                train_candidates += clean_aux
            if knobs.get("shuffle"):
                train_candidates = shuffle_candidate_actions_within_group(
                    train_candidates, seed=seed)
            cfg = replace(
                base_cfg,
                candidate_bce_weight=knobs["candidate_bce_weight"],
                zero_context=knobs.get("zero_context", False),
            )
            model = copy.deepcopy(base)
            mean, std = prefix_action_statistics(
                train_candidates, PREFIX_LENGTH)
            model.set_action_statistics(mean, std)
            run = wandb.init(
                project="pnp-clean-verifier",
                name=f"{name}-f{fold_index}-s{seed}",
                config=asdict(cfg),
                mode="online" if USE_WANDB else "disabled", reinit=True)
            model, rank_meta = train_advantage(
                model, train_candidates, fold_val, DEVICE,
                config=cfg, wandb_run=run)
            metrics, records = evaluate_candidate_ranker(
                model, fold_val, DEVICE, config=cfg, return_records=True)
            run.log({f"oof/{k}": v for k, v in metrics.items()
                     if not isinstance(v, dict)})
            run.finish()
            cv_rows.append({
                "config": name, "fold": fold_index, "seed": seed,
                **{k: v for k, v in metrics.items()
                   if not isinstance(v, dict)},
            })
            epoch_rows.append({
                "config": name, "fold": fold_index, "seed": seed,
                **value_meta, **rank_meta,
            })
            for record in records:
                oof_records.append({
                    "config": name, "fold": fold_index, "seed": seed, **record})

cv_table = pd.DataFrame(cv_rows)
oof_table = pd.DataFrame(oof_records)
cv_table.to_csv(OUTPUT / "advantage_cv_folds.csv", index=False)
oof_table.to_csv(OUTPUT / "advantage_oof_records.csv", index=False)
display(cv_table.groupby("config").agg(
    ranking=("group_macro_ranking_accuracy", "mean"),
    top1=("top1_success", "mean"),
    default=("default_success", "mean"),
    random=("random_success", "mean"),
    discordant=("n_discordant_groups", "sum"),
))'''),
    md("## 6. Select using pooled OOF groups (controls are diagnostic only)"),
    code(r'''oof_summaries = []
for (name, seed), rows in oof_table.groupby(["config", "seed"]):
    metrics = summarize_candidate_records(
        rows.to_dict("records"), seed=int(seed))
    oof_summaries.append({
        "config": name, "seed": int(seed),
        **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
    })
oof_summary = pd.DataFrame(oof_summaries)
selection = (oof_summary[oof_summary.config.isin(ELIGIBLE_CONFIGS)]
             .groupby("config")
             .agg(ranking=("group_macro_ranking_accuracy", "mean"),
                  uplift=("top1_uplift_random", "mean"))
             .sort_values(["ranking", "uplift"], ascending=False))
SELECTED_CONFIG = selection.index[0]
display(oof_summary)
display(selection)
print("selected:", SELECTED_CONFIG)'''),
    md("## 7. Refit on all development groups and open the locked test once"),
    code(r'''selected_knobs = CONFIGS[SELECTED_CONFIG]
final_historical = exclude_candidate_identities(historical, locked_test)
final_hist_split = known_task_split(final_historical, seed=42)
final_value_train = select_examples(final_historical, final_hist_split["train"])
final_value_val = select_examples(final_historical, final_hist_split["val"])
final_candidates = list(development)
if selected_knobs.get("include_aux"):
    final_candidates += exclude_candidate_identities(auxiliary, locked_test)

chosen_epochs = pd.DataFrame(epoch_rows)
chosen_epochs = chosen_epochs[chosen_epochs.config == SELECTED_CONFIG]
value_epochs = max(1, int(round(chosen_epochs.value_epochs_ran.median())))
rank_epochs = max(1, int(round(chosen_epochs.rank_epochs_ran.median())))
final_cfg = AdvantageTrainConfig(
    seed=42, prefix_length=PREFIX_LENGTH,
    value_epochs=value_epochs, rank_epochs=rank_epochs,
    patience=max(value_epochs, rank_epochs) + 1,
    candidate_bce_weight=selected_knobs["candidate_bce_weight"],
)
final_model = CompactAdvantageVerifier()
final_model, final_value_meta = pretrain_value(
    final_model, final_value_train, final_value_val, DEVICE, config=final_cfg)
mean, std = prefix_action_statistics(final_candidates, PREFIX_LENGTH)
final_model.set_action_statistics(mean, std)
final_model, final_rank_meta = train_advantage(
    # Epoch counts are fixed from CV; an empty validation set prevents selecting
    # an epoch on the same groups used for fitting.
    final_model, final_candidates, [], DEVICE, config=final_cfg)
locked_metrics, locked_records = evaluate_candidate_ranker(
    final_model, locked_test, DEVICE, config=final_cfg, return_records=True)
(OUTPUT / "advantage_locked_test.json").write_text(
    json.dumps(locked_metrics, indent=2, sort_keys=True))
pd.DataFrame(locked_records).to_csv(
    OUTPUT / "advantage_locked_test_records.csv", index=False)
print(json.dumps(locked_metrics, indent=2))'''),
    md("## 8. Eligibility gate and checkpoint registration"),
    code(r'''selected_oof_records = oof_table[
    oof_table.config == SELECTED_CONFIG].copy()
# Average repeated seed predictions at the independent group level.
pooled = (selected_oof_records.groupby(
    ["group_id", "benchmark", "uncertainty_stratum"], as_index=False)
    .agg(pair_accuracy=("pair_accuracy", "mean"),
         margin=("margin", "mean"), comparisons=("comparisons", "max"),
         top1=("top1", "mean"), default=("default", "mean"),
         random=("random", "mean"), oracle=("oracle", "mean")))
pooled_oof = summarize_candidate_records(pooled.to_dict("records"), seed=42)
eligible = (
    pooled_oof["ranking_accuracy_ci95"][0] > .5
    and locked_metrics["group_macro_ranking_accuracy"] > .5
    and locked_metrics["top1_success"] >= locked_metrics["default_success"]
    and locked_metrics["top1_success"] > locked_metrics["random_success"]
)
print({"eligible": eligible, "pooled_oof": pooled_oof})

if eligible and REGISTER_IF_ELIGIBLE:
    verifier_id = new_verifier_id()
    checkpoint_config = {
        "model_class": "CompactAdvantageVerifier",
        "score_type": "raw_advantage",
        "obs_dim": 2048, "action_dim": 7, "horizon": 50,
        "prefix_length": PREFIX_LENGTH,
        "selected_config": SELECTED_CONFIG,
        "train": asdict(final_cfg),
    }
    store.start_run(
        "verifier_train", "libero+libero_pro", "action-advantage-verifier-v1")
    store.register_verifier(
        verifier_id,
        verifier_checkpoint_bytes(final_model, None, checkpoint_config),
        checkpoint_config, locked_metrics, split_manifest,
        dataset_hash=dataset_hash(primary))
    store.finish_run(n_rollouts=0)
    print("registered", verifier_id)
elif eligible:
    print("Eligible; set REGISTER_IF_ELIGIBLE=True to register.")
else:
    print("Not registered: the predeclared ranking/uplift gate was not met.")'''),
]


COLLECTION_CELLS = [
    md("# 04 — Deterministic-replay verifier collection\n\nCollect up to 300 unique mid-rollout states × 4 candidates without simulator snapshots. Each branch reaches its state by replaying an identical fixed action prefix. Resumable and shardable; v1/v2 data is preserved."),
    md("## 1. Environment setup"), code(BOOTSTRAP.replace('"[analysis]"', '"[sim,analysis]"')),
    md("## 2. Load policy, store, and benchmark episode manifests"),
    code(r'''import json
from tqdm.auto import tqdm
from pnp import libero_env, libero_pro, models
from pnp.experiments import _prepare_libero_pro_episodes
from pnp.store import SupabaseStore
from pnp.verifier import *

benchmark_dict = libero_env.init_libero_benchmark()
libero_pro.patch_torch_load()
policy, preprocess, postprocess = models.load_pi05()
device = models.default_device()
store = SupabaseStore()

suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
tasks = [(suite, task) for suite in suites for task in range(benchmark_dict[suite]().n_tasks)]
standard = libero_env.build_final_episodes(benchmark_dict, tasks=tasks)
for ep in standard: ep["benchmark"] = "libero"
pro = _prepare_libero_pro_episodes()
for ep in pro: ep["benchmark"] = "libero_pro"
episode_lookup = {(e["benchmark"], e["suite"], e["task_idx"], e.get("ep_idx", e.get("episode_idx"))): e
                  for e in standard + pro}
print(len(standard), len(pro), len(episode_lookup))'''),
    md("## 3. Configure workers and build the fixed uncertainty manifest"),
    code(r'''COLLECTION_EXPERIMENT = "verifier-clean-pairs-v3"
TARGETS = {"libero": 120, "libero_pro": 180}
CANDIDATE_COUNT = 4
PREFIX_LENGTH = 10
SHARD_COUNT = 1   # Use 3 for three parallel Colab sessions.
SHARD_INDEX = 0   # Set to 0, 1, or 2 in each session.
assert 0 <= SHARD_INDEX < SHARD_COUNT

def pages(table, columns, configure):
    rows=[]; start=0
    while True:
        q = configure(store.client.table(table).select(columns)).range(start, start+999)
        batch = q.execute().data or []; rows += batch
        if len(batch) < 1000: return rows
        start += 1000

experiments = ("libero-hybrid-schedules-k3-v1", "libero-pro-canonical-core-k3-v1")
rollouts=[]
for experiment in experiments:
    rollouts += pages("rollouts", "rollout_id,benchmark,suite,task_idx,episode_idx,success", lambda q, e=experiment:
        q.eq("experiment", e).eq("method", "pnp_uncertainty_only").eq("status", "completed"))
ids = {r["rollout_id"] for r in rollouts}
euler = []
id_list = sorted(ids)
for start in range(0, len(id_list), 100):
    batch_ids = id_list[start:start+100]
    euler += pages("pnp_euler_steps", "rollout_id,chunk_idx,u_mean",
                   lambda q, batch_ids=batch_ids: q.in_("rollout_id", batch_ids))
manifest = build_stratified_manifest(rollouts, euler, TARGETS)
TOTAL_GROUPS = len(manifest)
if TOTAL_GROUPS < sum(TARGETS.values()):
    print(f"eligible unique-state capacity: {TOTAL_GROUPS}/{sum(TARGETS.values())}; "
          "collecting all available groups")
manifest = manifest[SHARD_INDEX::SHARD_COUNT]
print({k: sum(r["benchmark"] == k for r in manifest) for k in ("libero", "libero_pro")})
print({k: sum(r["uncertainty_stratum"] == k for r in manifest) for k in ("low", "mid", "high")})'''),
    md("## 4. Dry-run identity and schema checks"),
    code(r'''missing = [row for row in manifest if (row["benchmark"], row["suite"], row["task_idx"], row["episode_idx"]) not in episode_lookup]
assert not missing, missing[:3]
store.client.table("verifier_candidate_groups").select("candidate_group_id").limit(1).execute()
print("manifest identities and verifier tables are ready")'''),
    md("## 5. Collect deterministic-replay four-candidate groups"),
    code(r'''existing_group_rows = pages(
    "verifier_candidate_groups", "candidate_group_id",
    lambda q: q.eq("experiment", COLLECTION_EXPERIMENT))
existing_group_ids = {row["candidate_group_id"] for row in existing_group_rows}
all_candidate_ids = pages(
    "verifier_candidates", "candidate_group_id",
    lambda q: q)
candidate_counts = {}
for row in all_candidate_ids:
    gid = row["candidate_group_id"]
    if gid in existing_group_ids:
        candidate_counts[gid] = candidate_counts.get(gid, 0) + 1
# A crash can leave a group row with fewer than four candidates. Recollect it.
existing = {gid for gid in existing_group_ids
            if candidate_counts.get(gid, 0) == CANDIDATE_COUNT}
print({"complete_existing_groups": len(existing),
       "partial_groups_to_repair": len(existing_group_ids - existing)})
store.start_run("verifier_pair_collection", "libero+libero_pro", COLLECTION_EXPERIMENT,
                config={"groups": TOTAL_GROUPS,
                        "outcomes": TOTAL_GROUPS * CANDIDATE_COUNT,
                        "candidate_count": CANDIDATE_COUNT, "prefix_length": PREFIX_LENGTH,
                        "shard_count": SHARD_COUNT, "shard_index": SHARD_INDEX})
completed = 0
for item in tqdm(manifest, desc="candidate groups"):
    ep = episode_lookup[(item["benchmark"], item["suite"], item["task_idx"], item["episode_idx"])]
    expected_id = candidate_group_id(item["benchmark"], item["suite"], item["task_idx"],
                                     item["episode_idx"], item["chunk_idx"],
                                     namespace=COLLECTION_EXPERIMENT)
    if expected_id in existing:
        continue
    env = libero_env.make_env(ep["bddl_path"])
    try:
        try:
            pair = collect_replay_candidate_group(
                env, ep, policy, preprocess, postprocess, device,
                chunk_idx=item["chunk_idx"], uncertainty_stratum=item["uncertainty_stratum"],
                prefix_length=PREFIX_LENGTH,
                candidate_count=CANDIDATE_COUNT, experiment=COLLECTION_EXPERIMENT)
        except Exception as error:
            print("replay group skipped:", type(error).__name__, error)
            pair = None
        if pair is None:
            continue
        group, candidates = pair
        store.register_candidate_group(group, candidates)
        existing.add(group["candidate_group_id"]); completed += len(candidates)
    finally:
        env.close()
store.finish_run(n_rollouts=completed)
print("new outcomes:", completed, "total groups:", len(existing))'''),
    md("## 6. Integrity and outcome-balance report"),
    code(r'''groups = pages(
    "verifier_candidate_groups", "*",
    lambda q: q.eq("experiment", COLLECTION_EXPERIMENT))
candidates = pages(
    "verifier_candidates", "candidate_id,candidate_group_id,candidate_kind,success",
    lambda q: q)
group_ids = {g["candidate_group_id"] for g in groups}
candidates = [c for c in candidates if c["candidate_group_id"] in group_ids]
by_group = {}
for candidate in candidates:
    by_group.setdefault(candidate["candidate_group_id"], []).append(candidate)
complete = {gid for gid in group_ids if len(by_group.get(gid, [])) == CANDIDATE_COUNT}
print({"groups": len(groups), "outcomes": len(candidates),
       "complete_groups": len(complete),
       "partial_groups": len(group_ids - complete),
       "successes": sum(c["success"] for c in candidates),
       "failures": sum(not c["success"] for c in candidates),
       "discordant_groups": sum(len({c["success"] for c in by_group.get(gid, [])}) == 2
                                for gid in complete),
       "replay_groups": sum(g["pairing_mode"] == "deterministic_replay" for g in groups),
       "validated_groups": sum(bool((g.get("metadata_json") or {}).get("replay_validated"))
                               for g in groups)})'''),
]


def main():
    outputs = {
        ROOT / "notebooks" / "03_verifier_experiments.ipynb": notebook(EXPERIMENT_CELLS),
        ROOT / "notebooks" / "04_collect_verifier_pairs.ipynb": notebook(COLLECTION_CELLS),
    }
    for path, value in outputs.items():
        path.write_text(json.dumps(value, indent=1) + "\n")
        print(path)
    # Keep upload-ready worker copies synchronized with the canonical notebook.
    from generate_verifier_workers import generate_workers
    generate_workers(
        ROOT / "notebooks" / "04_collect_verifier_pairs.ipynb",
        ROOT / "notebooks" / "workers",
        shard_count=3,
    )


if __name__ == "__main__":
    main()
