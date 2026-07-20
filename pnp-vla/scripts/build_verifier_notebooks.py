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


EXPERIMENT_CELLS = [
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
    code(r'''for seed in SEEDS:
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
    code(r'''paired_examples = load_candidate_examples(store)
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


COLLECTION_CELLS = [
    md("# 04 — Exact clean-chunk paired collection\n\nCollects 125 default-vs-fresh candidate groups (250 outcomes). Apply the latest schema first."),
    md("## 1. Environment setup"), code(BOOTSTRAP.replace('"[analysis]"', '"[sim,analysis]"')),
    md("## 2. Load policy, store, and benchmark episode manifests"),
    code(r'''import json
from tqdm.auto import tqdm
from pnp import libero_env, models
from pnp.experiments import _prepare_libero_pro_episodes
from pnp.store import SupabaseStore
from pnp.verifier import *

benchmark_dict = libero_env.init_libero_benchmark()
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
    md("## 3. Build the fixed low/mid/high uncertainty manifest"),
    code(r'''def pages(table, columns, configure):
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
manifest = build_stratified_manifest(rollouts, euler, {"libero": 50, "libero_pro": 75})
assert len(manifest) == 125, len(manifest)
print({k: sum(r["benchmark"] == k for r in manifest) for k in ("libero", "libero_pro")})
print({k: sum(r["uncertainty_stratum"] == k for r in manifest) for k in ("low", "mid", "high")})'''),
    md("## 4. Dry-run identity and schema checks"),
    code(r'''missing = [row for row in manifest if (row["benchmark"], row["suite"], row["task_idx"], row["episode_idx"]) not in episode_lookup]
assert not missing, missing[:3]
store.client.table("verifier_candidate_groups").select("candidate_group_id").limit(1).execute()
print("manifest identities and verifier tables are ready")'''),
    md("## 5. Collect 125 resumable pairs (250 outcomes)"),
    code(r'''existing = {r["candidate_group_id"] for r in
            (store.client.table("verifier_candidate_groups").select("candidate_group_id")
             .eq("experiment", "verifier-clean-pairs-v1").execute().data or [])}
store.start_run("verifier_pair_collection", "libero+libero_pro", "verifier-clean-pairs-v1",
                config={"groups": 125, "outcomes": 250, "prefix_length": 10})
completed = 0
for item in tqdm(manifest, desc="candidate groups"):
    ep = episode_lookup[(item["benchmark"], item["suite"], item["task_idx"], item["episode_idx"])]
    expected_id = candidate_group_id(item["benchmark"], item["suite"], item["task_idx"],
                                     item["episode_idx"], item["chunk_idx"])
    if expected_id in existing:
        continue
    env = libero_env.make_env(ep["bddl_path"])
    try:
        try:
            pair = collect_candidate_pair(
                env, ep, policy, preprocess, postprocess, device,
                chunk_idx=item["chunk_idx"], uncertainty_stratum=item["uncertainty_stratum"],
                prefix_length=10, validate_snapshot=True)
        except RuntimeError as error:
            print("snapshot fallback:", error)
            pair = collect_initial_pair_fallback(
                env, ep, policy, preprocess, postprocess, device,
                uncertainty_stratum=item["uncertainty_stratum"], prefix_length=10,
                source_chunk_idx=item["chunk_idx"])
        if pair is None:
            pair = collect_initial_pair_fallback(
                env, ep, policy, preprocess, postprocess, device,
                uncertainty_stratum=item["uncertainty_stratum"], prefix_length=10,
                source_chunk_idx=item["chunk_idx"])
        group, candidates = pair
        store.register_candidate_group(group, candidates)
        existing.add(group["candidate_group_id"]); completed += 2
    finally:
        env.close()
store.finish_run(n_rollouts=completed)
print("new outcomes:", completed, "total groups:", len(existing))'''),
    md("## 6. Integrity and outcome-balance report"),
    code(r'''groups = store.client.table("verifier_candidate_groups").select("*").eq(
    "experiment", "verifier-clean-pairs-v1").execute().data or []
candidates = store.client.table("verifier_candidates").select(
    "candidate_id,candidate_group_id,candidate_kind,success").execute().data or []
group_ids = {g["candidate_group_id"] for g in groups}
candidates = [c for c in candidates if c["candidate_group_id"] in group_ids]
print({"groups": len(groups), "outcomes": len(candidates),
       "successes": sum(c["success"] for c in candidates),
       "failures": sum(not c["success"] for c in candidates),
       "discordant_groups": sum(len({c["success"] for c in candidates if c["candidate_group_id"] == gid}) == 2
                                for gid in group_ids),
       "snapshot_groups": sum(g["pairing_mode"] == "snapshot" for g in groups),
       "fallback_groups": sum(g["pairing_mode"] != "snapshot" for g in groups)})'''),
]


def main():
    outputs = {
        ROOT / "notebooks" / "03_verifier_experiments.ipynb": notebook(EXPERIMENT_CELLS),
        ROOT / "notebooks" / "04_collect_verifier_pairs.ipynb": notebook(COLLECTION_CELLS),
    }
    for path, value in outputs.items():
        path.write_text(json.dumps(value, indent=1) + "\n")
        print(path)


if __name__ == "__main__":
    main()
