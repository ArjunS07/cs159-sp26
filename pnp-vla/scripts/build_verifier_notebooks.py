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
    md("""# 03 — Same-state action-advantage verifier

This notebook supersedes the historical classification sweep. It pretrains
`V(s)` on ordinary LIBERO/LIBERO-PRO rollouts, freezes that pathway, and learns
`A(s,a)` from deterministic same-state candidate groups. Model selection uses
episode-grouped out-of-fold ranking; the prospective v4 test is opened once."""),
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
V3_CANDIDATES = "verifier-clean-pairs-v3"
V4_DEVELOPMENT = "verifier-clean-pairs-v4-dev"
V4_TEST = "verifier-clean-pairs-v4-test"
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
        store, HISTORICAL_EXPERIMENTS, progress=tqdm,
        cache_dir=OUTPUT / "historical_rollout_cache_v2")
    with CACHE.open("wb") as handle:
        pickle.dump(historical, handle)

candidate_cache = OUTPUT / "candidate_group_cache_v2"
v3 = load_candidate_examples(
    store, V3_CANDIDATES, cache_dir=candidate_cache / "v3")
v4_development = load_candidate_examples(
    store, V4_DEVELOPMENT, cache_dir=candidate_cache / "v4_development")
auxiliary = load_candidate_examples(
    store, AUXILIARY_CANDIDATES, cache_dir=candidate_cache / "auxiliary")

def collection_groups(experiment):
    rows=[]; start=0
    while True:
        batch = (store.client.table("verifier_candidate_groups")
                 .select("candidate_group_id,benchmark,suite,task_idx,episode_idx,metadata_json")
                 .eq("experiment", experiment)
                 .range(start, start+999).execute().data or [])
        rows += batch
        if len(batch) < 1000: break
        start += 1000
    return rows

v4_development_groups = collection_groups(V4_DEVELOPMENT)
prospective_test_groups = collection_groups(V4_TEST)
v4_development_hashes = {
    (row.get("metadata_json") or {}).get("collection_manifest_hash")
    for row in v4_development_groups}
v4_test_hashes = {
    (row.get("metadata_json") or {}).get("collection_manifest_hash")
    for row in prospective_test_groups}
assert len(v4_development_hashes) == 1 and None not in v4_development_hashes
assert len(v4_test_hashes) == 1 and None not in v4_test_hashes
prospective_test_identities = {
    (row["benchmark"], row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
    for row in prospective_test_groups}
assert len(prospective_test_groups) >= 120, len(prospective_test_groups)
print("v3 integrity", validate_candidate_groups(v3, expected_candidates=4))
v4_development_integrity = validate_candidate_groups(
    v4_development, expected_candidates=8)
assert v4_development_integrity["groups"] >= 360, v4_development_integrity
print("v4 development integrity", v4_development_integrity)
print("prospective test sealed", {
    "groups": len(prospective_test_groups),
    "manifest_hash": next(iter(v4_test_hashes)),
})
print("auxiliary integrity", validate_candidate_groups(auxiliary))
development = v3 + v4_development

def candidate_audit(examples):
    frame = pd.DataFrame([{
        "group": e.candidate_group_id, "benchmark": e.benchmark,
        "stratum": e.uncertainty_stratum, "success": e.success,
    } for e in examples])
    group_outcomes = frame.groupby("group").success.agg(["size", "nunique"])
    return {
        "groups": int(frame.group.nunique()), "outcomes": len(frame),
        "successes": int(frame.success.sum()), "failures": int((1-frame.success).sum()),
        "discordant_groups": int((group_outcomes["nunique"] == 2).sum()),
        "pairwise_comparisons": int(frame.groupby("group").success.apply(
            lambda y: int(y.sum()) * int((1-y).sum())).sum()),
    }

print("historical", {
    "chunks": len(historical),
    "rollouts": len({e.rollout_id for e in historical}),
    "hash": dataset_hash(historical),
})
print("development", candidate_audit(development))
print("auxiliary", candidate_audit(auxiliary))
legacy = OUTPUT / "experiment_results.csv"
if legacy.exists() and not (OUTPUT / "legacy_experiment_results.csv").exists():
    (OUTPUT / "legacy_experiment_results.csv").write_bytes(legacy.read_bytes())'''),
    md("## 4. Verify prospective cohorts and write immutable split manifests"),
    code(r'''development_ids = [example.rollout_id for example in development]
folds = candidate_cv_splits(
    development, development_ids, n_folds=N_FOLDS, seed=42)

split_manifest = {
    "development_experiments": [V3_CANDIDATES, V4_DEVELOPMENT],
    "prospective_test_experiment": V4_TEST,
    "v4_collection_manifest_hashes": {
        "development": next(iter(v4_development_hashes)),
        "test": next(iter(v4_test_hashes)),
    },
    "development_dataset_hash": dataset_hash(development),
    "historical_dataset_hash": dataset_hash(historical),
    "development_candidate_ids": sorted(development_ids),
    "prospective_test_group_ids": sorted(
        row["candidate_group_id"] for row in prospective_test_groups),
    "prospective_test_episode_identities": sorted(
        list(identity) for identity in prospective_test_identities),
    "folds": folds,
}
(OUTPUT / "advantage_splits.json").write_text(
    json.dumps(split_manifest, indent=2, sort_keys=True))
print({
    "development": candidate_audit(development),
    "prospective_test_groups_sealed": len(prospective_test_groups),
    "manifest": str(OUTPUT / "advantage_splits.json"),
})'''),
    md("## 5. Grouped 4-fold CV × 3 seeds"),
    code(r'''CONFIGS = {
    "rank_only": {"candidate_bce_weight": 0.0},
    "rank_plus_bce": {"candidate_bce_weight": 0.1},
    "rank_plus_initial_pairs": {"candidate_bce_weight": 0.0, "include_aux": True},
    "action_only_control": {"candidate_bce_weight": 0.0, "zero_context": True},
    "shuffled_action_control": {"candidate_bce_weight": 0.0, "shuffle": True},
}
ELIGIBLE_CONFIGS = {
    "rank_only", "rank_plus_bce", "rank_plus_initial_pairs"}
cv_rows, oof_records, epoch_rows = [], [], []

for fold_index, fold in enumerate(folds):
    fold_train = select_examples(development, fold["train"])
    fold_val = select_examples(development, fold["val"])
    protected_identities = (
        candidate_episode_identities(fold_val) | prospective_test_identities)
    fold_historical = exclude_episode_identities(
        historical, protected_identities)
    historical_split = known_task_split(fold_historical, seed=42 + fold_index)
    value_val = select_examples(fold_historical, historical_split["val"])
    value_val_ids = set(historical_split["val"])
    value_train = [
        example for example in fold_historical
        if example.rollout_id not in value_val_ids]
    clean_aux = exclude_episode_identities(auxiliary, protected_identities)

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
final_historical = exclude_episode_identities(
    historical, prospective_test_identities)
final_hist_split = known_task_split(final_historical, seed=42)
final_value_val = select_examples(final_historical, final_hist_split["val"])
final_value_val_ids = set(final_hist_split["val"])
final_value_train = [
    example for example in final_historical
    if example.rollout_id not in final_value_val_ids]
final_candidates = list(development)
if selected_knobs.get("include_aux"):
    final_candidates += exclude_episode_identities(
        auxiliary, prospective_test_identities)

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

# This is the first cell that downloads or examines prospective-test outcomes.
locked_test = load_candidate_examples(
    store, V4_TEST, cache_dir=candidate_cache / "v4_test")
locked_integrity = validate_candidate_groups(locked_test, expected_candidates=8)
assert locked_integrity["groups"] >= 120, locked_integrity
assert candidate_episode_identities(locked_test) == prospective_test_identities
split_manifest["prospective_test_dataset_hash"] = dataset_hash(locked_test)
split_manifest["locked_test_candidate_ids"] = sorted(
    example.rollout_id for example in locked_test)
(OUTPUT / "advantage_splits.json").write_text(
    json.dumps(split_manifest, indent=2, sort_keys=True))
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
        dataset_hash=dataset_hash(development + locked_test))
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


TARGETED_COLLECTION_CELLS = [
    md("""# 05 — Targeted v4 verifier collection

Collect fixed best-of-8 groups at high-uncertainty states. The development
cohort oversamples failed source rollouts; the separately named prospective
test cohort is assigned before candidate outcomes and is never rebalanced."""),
    md("## 1. Environment setup"),
    code(BOOTSTRAP.replace('"[analysis]"', '"[sim,analysis]"')),
    md("## 2. Load policy, store, and episode manifests"),
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
tasks = [(suite, task) for suite in suites
         for task in range(benchmark_dict[suite]().n_tasks)]
standard = libero_env.build_final_episodes(benchmark_dict, tasks=tasks)
for ep in standard: ep["benchmark"] = "libero"
pro = _prepare_libero_pro_episodes()
for ep in pro: ep["benchmark"] = "libero_pro"
episode_lookup = {
    (e["benchmark"], e["suite"], e["task_idx"],
     e.get("ep_idx", e.get("episode_idx"))): e
    for e in standard + pro
}
print(len(standard), len(pro), len(episode_lookup))'''),
    md("## 3. Build and persist outcome-blind v4 manifests"),
    code(r'''DEVELOPMENT_EXPERIMENT = "verifier-clean-pairs-v4-dev"
TEST_EXPERIMENT = "verifier-clean-pairs-v4-test"
DEVELOPMENT_TARGETS = {"libero": 180, "libero_pro": 270}
TEST_TARGETS = {"libero": 60, "libero_pro": 90}
CANDIDATE_COUNT = 8
PREFIX_LENGTH = 10
SHARD_COUNT = 1   # Use the generated workers for three-way collection.
SHARD_INDEX = 0
assert 0 <= SHARD_INDEX < SHARD_COUNT

def pages(table, columns, configure, order_by):
    rows=[]; start=0
    while True:
        q = configure(store.client.table(table).select(columns))
        for column in order_by:
            q = q.order(column)
        batch = q.range(start, start+999).execute().data or []; rows += batch
        if len(batch) < 1000: return rows
        start += 1000

source_experiments = (
    "libero-hybrid-schedules-k3-v1",
    "libero-pro-canonical-core-k3-v1",
)
rollouts = []
for experiment in source_experiments:
    rollouts += pages(
        "rollouts",
        "rollout_id,benchmark,suite,task_idx,episode_idx,success",
        lambda q, experiment=experiment: q.eq(
            "experiment", experiment).eq(
            "method", "pnp_uncertainty_only").eq("status", "completed"),
        ("rollout_id",))

rollout_ids = sorted({row["rollout_id"] for row in rollouts})
euler = []
for start in range(0, len(rollout_ids), 100):
    batch_ids = rollout_ids[start:start+100]
    euler += pages(
        "pnp_euler_steps", "rollout_id,chunk_idx,euler_step,u_mean",
        lambda q, ids=batch_ids: q.in_("rollout_id", ids),
        ("rollout_id", "chunk_idx", "euler_step"))

v3_groups = pages(
    "verifier_candidate_groups",
    "candidate_group_id,benchmark,suite,task_idx,episode_idx",
    lambda q: q.eq("experiment", "verifier-clean-pairs-v3"),
    ("candidate_group_id",))
excluded = {
    (row["benchmark"], row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
    for row in v3_groups
}
manifests = build_targeted_manifests(
    rollouts, euler, excluded,
    development_targets=DEVELOPMENT_TARGETS,
    test_targets=TEST_TARGETS,
    development_failure_fraction=.70,
    seed=42,
)
manifest_hashes = {
    cohort: collection_manifest_hash(rows) for cohort, rows in manifests.items()}
manifest_document = {
    "version": 2,
    "candidate_count": CANDIDATE_COUNT,
    "prefix_length": PREFIX_LENGTH,
    "development_experiment": DEVELOPMENT_EXPERIMENT,
    "test_experiment": TEST_EXPERIMENT,
    "development_targets": DEVELOPMENT_TARGETS,
    "test_targets": TEST_TARGETS,
    "excluded_v3_episode_count": len(excluded),
    "hashes": manifest_hashes,
    "manifests": manifests,
}
manifest_bytes = json.dumps(
    manifest_document, sort_keys=True, separators=(",", ":")).encode()
manifest_path = (
    f"verifier_manifests/v4-targeted-"
    f"{manifest_hashes['development']}-{manifest_hashes['test']}.json")
store._upload(manifest_path, manifest_bytes)

full_manifest = [
    {**row, "cohort": cohort,
     "experiment": (DEVELOPMENT_EXPERIMENT if cohort == "development"
                    else TEST_EXPERIMENT)}
    for cohort in ("test", "development")
    for row in manifests[cohort]
]
manifest = full_manifest[SHARD_INDEX::SHARD_COUNT]
print({
    "hashes": manifest_hashes,
    "manifest_path": manifest_path,
    "development": len(manifests["development"]),
    "test": len(manifests["test"]),
    "worker_groups": len(manifest),
    "development_source_failures": sum(
        not row["success"] for row in manifests["development"]),
    "test_source_failures": sum(not row["success"] for row in manifests["test"]),
})'''),
    md("## 4. Dry-run identity, disjointness, and schema checks"),
    code(r'''development_identities = {
    (r["benchmark"], r["suite"], r["task_idx"], r["episode_idx"])
    for r in manifests["development"]}
test_identities = {
    (r["benchmark"], r["suite"], r["task_idx"], r["episode_idx"])
    for r in manifests["test"]}
assert development_identities.isdisjoint(test_identities)
assert (development_identities | test_identities).isdisjoint(excluded)
missing = [
    row for row in full_manifest
    if (row["benchmark"], row["suite"], row["task_idx"], row["episode_idx"])
       not in episode_lookup]
assert not missing, missing[:3]
store.client.table("verifier_candidate_groups").select(
    "candidate_group_id").limit(1).execute()
print("v4 manifests, identities, and verifier tables are ready")'''),
    md("## 5. Collect fixed-eight deterministic-replay groups"),
    code(r'''experiment_names = (DEVELOPMENT_EXPERIMENT, TEST_EXPERIMENT)
existing_group_rows = pages(
    "verifier_candidate_groups",
    "candidate_group_id,experiment,metadata_json",
    lambda q: q.in_("experiment", experiment_names),
    ("candidate_group_id",))
canonical_by_id = {}
for item in full_manifest:
    gid = candidate_group_id(
        item["benchmark"], item["suite"], item["task_idx"],
        item["episode_idx"], item["chunk_idx"], namespace=item["experiment"])
    canonical_by_id[gid] = item
canonical_ids = set(canonical_by_id)

# Older workers could build overlapping shards because their paginated source
# queries had no stable ordering. Preserve those artifacts under an explicitly
# excluded experiment name instead of deleting them.
noncanonical_rows = [
    row for row in existing_group_rows
    if row["candidate_group_id"] not in canonical_ids]
for row in noncanonical_rows:
    metadata = dict(row.get("metadata_json") or {})
    metadata["quarantined_from_experiment"] = row["experiment"]
    metadata["quarantine_reason"] = "noncanonical_v4_manifest"
    (store.client.table("verifier_candidate_groups")
     .update({
         "experiment": row["experiment"] + "-orphan",
         "metadata_json": metadata,
     }).eq("candidate_group_id", row["candidate_group_id"]).execute())

canonical_rows = [
    row for row in existing_group_rows
    if row["candidate_group_id"] in canonical_ids]
for row in canonical_rows:
    item = canonical_by_id[row["candidate_group_id"]]
    metadata = dict(row.get("metadata_json") or {})
    metadata.update({
        "cohort": item["cohort"],
        "collection_manifest_hash": manifest_hashes[item["cohort"]],
        "collection_manifest_path": manifest_path,
        "source_rollout_id": item["rollout_id"],
        "source_success": bool(item["success"]),
        "source_u_mean": float(item["u_mean"]),
    })
    (store.client.table("verifier_candidate_groups")
     .update({"metadata_json": metadata})
     .eq("candidate_group_id", row["candidate_group_id"]).execute())

existing_group_ids = {row["candidate_group_id"] for row in canonical_rows}
candidate_rows = []
existing_id_list = sorted(existing_group_ids)
for start in range(0, len(existing_id_list), 100):
    ids = existing_id_list[start:start+100]
    candidate_rows += pages(
        "verifier_candidates", "candidate_id,candidate_group_id",
        lambda q, ids=ids: q.in_("candidate_group_id", ids),
        ("candidate_id",))
candidate_counts = {}
for row in candidate_rows:
    gid = row["candidate_group_id"]
    candidate_counts[gid] = candidate_counts.get(gid, 0) + 1
existing = {
    gid for gid in existing_group_ids
    if candidate_counts.get(gid, 0) == CANDIDATE_COUNT}
print({
    "quarantined_noncanonical_groups": len(noncanonical_rows),
    "complete_existing_groups": len(existing),
    "partial_groups_to_repair": len(existing_group_ids - existing),
})

store.start_run(
    "verifier_pair_collection", "libero+libero_pro", "verifier-clean-pairs-v4",
    config={
        "groups": len(full_manifest),
        "outcomes": len(full_manifest) * CANDIDATE_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "prefix_length": PREFIX_LENGTH,
        "manifest_hashes": manifest_hashes,
        "manifest_path": manifest_path,
        "shard_count": SHARD_COUNT,
        "shard_index": SHARD_INDEX,
    })
completed_outcomes = skipped = 0
for item in tqdm(manifest, desc="v4 candidate groups"):
    expected_id = candidate_group_id(
        item["benchmark"], item["suite"], item["task_idx"],
        item["episode_idx"], item["chunk_idx"], namespace=item["experiment"])
    if expected_id in existing:
        continue
    ep = episode_lookup[(
        item["benchmark"], item["suite"], item["task_idx"], item["episode_idx"])]
    env = libero_env.make_env(ep["bddl_path"])
    try:
        try:
            collected = collect_replay_candidate_group(
                env, ep, policy, preprocess, postprocess, device,
                chunk_idx=item["chunk_idx"], uncertainty_stratum="high",
                prefix_length=PREFIX_LENGTH, candidate_count=CANDIDATE_COUNT,
                experiment=item["experiment"])
        except Exception as error:
            print("v4 group skipped:", type(error).__name__, error)
            collected = None
        if collected is None:
            skipped += 1
            continue
        group, candidates = collected
        group["metadata_json"].update({
            "cohort": item["cohort"],
            "collection_manifest_hash": manifest_hashes[item["cohort"]],
            "collection_manifest_path": manifest_path,
            "source_rollout_id": item["rollout_id"],
            "source_success": bool(item["success"]),
            "source_u_mean": float(item["u_mean"]),
        })
        store.register_candidate_group(group, candidates)
        existing.add(group["candidate_group_id"])
        completed_outcomes += len(candidates)
    finally:
        env.close()
store.finish_run(n_rollouts=completed_outcomes)
print({
    "new_outcomes": completed_outcomes,
    "skipped_unreachable": skipped,
    "complete_groups_seen": len(existing),
})'''),
    md("## 6. Cohort integrity and balance report"),
    code(r'''for cohort, experiment in (
    ("development", DEVELOPMENT_EXPERIMENT),
    ("test", TEST_EXPERIMENT),
):
    groups = pages(
        "verifier_candidate_groups", "*",
        lambda q, experiment=experiment: q.eq("experiment", experiment),
        ("candidate_group_id",))
    group_ids = {group["candidate_group_id"] for group in groups}
    candidates = []
    group_id_list = sorted(group_ids)
    for start in range(0, len(group_id_list), 100):
        ids = group_id_list[start:start+100]
        candidates += pages(
            "verifier_candidates",
            "candidate_id,candidate_group_id,candidate_kind,success",
            lambda q, ids=ids: q.in_("candidate_group_id", ids),
            ("candidate_id",))
    by_group = {}
    for candidate in candidates:
        by_group.setdefault(candidate["candidate_group_id"], []).append(candidate)
    complete = {
        gid for gid in group_ids
        if len(by_group.get(gid, [])) == CANDIDATE_COUNT}
    expected_hash = manifest_hashes[cohort]
    hash_matches = sum(
        (group.get("metadata_json") or {}).get("collection_manifest_hash")
        == expected_hash for group in groups)
    print(cohort, {
        "manifest_target": len(manifests[cohort]),
        "groups": len(groups),
        "complete_groups": len(complete),
        "partial_groups": len(group_ids - complete),
        "manifest_hash_matches": hash_matches,
        "successes": sum(candidate["success"] for candidate in candidates),
        "failures": sum(not candidate["success"] for candidate in candidates),
        "discordant_groups": sum(
            len({candidate["success"] for candidate in by_group.get(gid, [])}) == 2
            for gid in complete),
        "pairwise_comparisons": sum(
            sum(candidate["success"] for candidate in by_group.get(gid, []))
            * sum(not candidate["success"] for candidate in by_group.get(gid, []))
            for gid in complete),
        "source_failures": sum(
            not bool((group.get("metadata_json") or {}).get("source_success"))
            for group in groups),
    })'''),
]


ONLINE_EVALUATION_CELLS = [
    md("""# 06 — Fresh online verifier selection evaluation

Freeze verifier `3add05c827424c4a` and evaluate best-of-8 selection on episode
identities unused by every V1–V4 candidate dataset. All eight outcomes are
rolled out for evaluation; the verifier score never uses those outcomes."""),
    md("## 1. Setup and load the frozen policy/verifier"),
    code(BOOTSTRAP.replace('"[analysis]"', '"[sim,analysis]"')),
    code(r'''import json
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from pnp import libero_env, libero_pro, models
from pnp.experiments import _prepare_libero_pro_episodes
from pnp.store import SupabaseStore
from pnp.verifier import *

VERIFIER_ID = "3add05c827424c4a"
EXPERIMENT = "verifier-online-selection-v1"
TARGETS = {"libero": 50, "libero_pro": 75}
CANDIDATE_COUNT = 8
PREFIX_LENGTH = 10

benchmark_dict = libero_env.init_libero_benchmark()
libero_pro.patch_torch_load()
policy, preprocess, postprocess = models.load_pi05()
device = models.default_device()
store = SupabaseStore()
checkpoint, verifier_row = store.load_verifier(VERIFIER_ID)
verifier = CompactAdvantageVerifier(
    obs_dim=int(verifier_row["obs_dim"]),
    action_dim=int(verifier_row["action_dim"]))
verifier.load_state_dict(checkpoint["model"])
verifier.to(device).eval()

suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
tasks = [(suite, task) for suite in suites
         for task in range(benchmark_dict[suite]().n_tasks)]
standard = libero_env.build_final_episodes(benchmark_dict, tasks=tasks)
for ep in standard: ep["benchmark"] = "libero"
pro = _prepare_libero_pro_episodes()
for ep in pro: ep["benchmark"] = "libero_pro"
episode_lookup = {
    (e["benchmark"], e["suite"], e["task_idx"],
     e.get("ep_idx", e.get("episode_idx"))): e
    for e in standard + pro}
print({"verifier": VERIFIER_ID, "episodes": len(episode_lookup)})'''),
    md("## 2. Freeze a fresh, outcome-blind manifest"),
    code(r'''def pages(table, columns, configure, order_by):
    rows=[]; start=0
    while True:
        query = configure(store.client.table(table).select(columns))
        for column in order_by:
            query = query.order(column)
        batch = query.range(start, start+999).execute().data or []
        rows += batch
        if len(batch) < 1000: return rows
        start += 1000

source_experiments = (
    "libero-hybrid-schedules-k3-v1",
    "libero-pro-canonical-core-k3-v1",
)
rollouts=[]
for experiment in source_experiments:
    rollouts += pages(
        "rollouts", "rollout_id,benchmark,suite,task_idx,episode_idx,success",
        lambda q, experiment=experiment: q.eq("experiment", experiment).eq(
            "method", "pnp_uncertainty_only").eq("status", "completed"),
        ("rollout_id",))
rollout_ids = sorted({row["rollout_id"] for row in rollouts})
euler=[]
for start in range(0, len(rollout_ids), 100):
    ids = rollout_ids[start:start+100]
    euler += pages(
        "pnp_euler_steps", "rollout_id,chunk_idx,euler_step,u_mean",
        lambda q, ids=ids: q.in_("rollout_id", ids),
        ("rollout_id", "chunk_idx", "euler_step"))

prior_groups = pages(
    "verifier_candidate_groups",
    "candidate_group_id,benchmark,suite,task_idx,episode_idx",
    lambda q: q.like("experiment", "verifier-clean-pairs-v%"),
    ("candidate_group_id",))
excluded = {
    (row["benchmark"], row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
    for row in prior_groups}
manifests = build_targeted_manifests(
    rollouts, euler, excluded,
    development_targets={"libero": 0, "libero_pro": 0},
    test_targets=TARGETS, seed=159, allow_shortfall=True)
manifest = [{**row, "experiment": EXPERIMENT}
            for row in manifests["test"]]
manifest_hash = collection_manifest_hash(manifest)
manifest_path = f"verifier_manifests/online-eval-{manifest_hash}.json"
store._upload(manifest_path, json.dumps({
    "version": 1, "verifier_id": VERIFIER_ID, "experiment": EXPERIMENT,
    "targets": TARGETS, "excluded_episode_count": len(excluded),
    "manifest_hash": manifest_hash, "manifest": manifest,
}, sort_keys=True, separators=(",", ":")).encode())
assert len({(r["benchmark"], r["suite"], r["task_idx"], r["episode_idx"])
            for r in manifest}) == len(manifest)
assert all((r["benchmark"], r["suite"], r["task_idx"], r["episode_idx"])
           not in excluded for r in manifest)
print({
    "manifest_hash": manifest_hash, "groups": len(manifest),
    "by_benchmark": {b: sum(r["benchmark"] == b for r in manifest) for b in TARGETS},
    "path": manifest_path,
})'''),
    md("## 3. Score eight candidates and collect all counterfactual outcomes"),
    code(r'''def verifier_scores(group, candidates):
    obs = np.asarray(candidates[0]["blobs"]["observation"]["obs_enc"], np.float32)
    actions = np.zeros((len(candidates), 50, 7), np.float32)
    masks = np.zeros((len(candidates), 50), bool)
    for index, candidate in enumerate(candidates):
        chunk = np.asarray(candidate["blobs"]["env_chunk"]["actions"], np.float32)[:, :7]
        n = min(50, len(chunk)); actions[index, :n] = chunk[:n]; masks[index, :n] = True
    with torch.no_grad():
        context = verifier.encode_context(
            torch.from_numpy(obs).reshape(1, -1).to(device),
            torch.tensor([group["metadata_json"]["chunk_position"]], device=device))
        scores = verifier.rank_candidates(
            context, torch.from_numpy(actions).unsqueeze(0).to(device),
            torch.from_numpy(masks).unsqueeze(0).to(device),
            PREFIX_LENGTH).squeeze(0).cpu().numpy()
    return scores

existing_groups = pages(
    "verifier_candidate_groups", "candidate_group_id",
    lambda q: q.eq("experiment", EXPERIMENT), ("candidate_group_id",))
existing_ids = {row["candidate_group_id"] for row in existing_groups}
candidate_rows=[]
for start in range(0, len(existing_ids), 100):
    ids = sorted(existing_ids)[start:start+100]
    candidate_rows += pages(
        "verifier_candidates", "candidate_id,candidate_group_id",
        lambda q, ids=ids: q.in_("candidate_group_id", ids), ("candidate_id",))
counts={}
for row in candidate_rows:
    counts[row["candidate_group_id"]] = counts.get(row["candidate_group_id"], 0) + 1
complete = {gid for gid in existing_ids if counts.get(gid) == CANDIDATE_COUNT}

store.start_run("verifier_online_evaluation", "libero+libero_pro", EXPERIMENT,
                config={"verifier_id": VERIFIER_ID, "groups": len(manifest),
                        "candidate_count": CANDIDATE_COUNT,
                        "prefix_length": PREFIX_LENGTH,
                        "manifest_hash": manifest_hash})
new_outcomes = skipped = 0
for item in tqdm(manifest, desc="fresh verifier groups"):
    gid = candidate_group_id(
        item["benchmark"], item["suite"], item["task_idx"], item["episode_idx"],
        item["chunk_idx"], namespace=EXPERIMENT)
    if gid in complete: continue
    ep = episode_lookup[(item["benchmark"], item["suite"],
                         item["task_idx"], item["episode_idx"])]
    env = libero_env.make_env(ep["bddl_path"])
    try:
        try:
            collected = collect_replay_candidate_group(
                env, ep, policy, preprocess, postprocess, device,
                chunk_idx=item["chunk_idx"], uncertainty_stratum="high",
                prefix_length=PREFIX_LENGTH, candidate_count=CANDIDATE_COUNT,
                experiment=EXPERIMENT)
        except Exception as error:
            print("evaluation group skipped:", type(error).__name__, error)
            collected = None
        if collected is None:
            skipped += 1; continue
        group, candidates = collected
        scores = verifier_scores(group, candidates)
        selected = int(np.argmax(scores))
        for index, (candidate, score) in enumerate(zip(candidates, scores)):
            candidate["metadata_json"].update({
                "verifier_id": VERIFIER_ID, "verifier_score": float(score),
                "verifier_selected": index == selected,
            })
        group["metadata_json"].update({
            "verifier_id": VERIFIER_ID,
            "selected_candidate_id": candidates[selected]["candidate_id"],
            "manifest_hash": manifest_hash, "manifest_path": manifest_path,
            "source_rollout_id": item["rollout_id"],
            "source_u_mean": float(item["u_mean"]),
        })
        store.register_candidate_group(group, candidates)
        complete.add(group["candidate_group_id"])
        new_outcomes += len(candidates)
    finally:
        env.close()
store.finish_run(n_rollouts=new_outcomes)
print({"new_outcomes": new_outcomes, "skipped": skipped,
       "complete_groups_seen": len(complete)})'''),
    md("## 4. Fresh selection report"),
    code(r'''groups = pages(
    "verifier_candidate_groups", "candidate_group_id,benchmark,metadata_json",
    lambda q: q.eq("experiment", EXPERIMENT), ("candidate_group_id",))
group_ids = {row["candidate_group_id"] for row in groups}
candidates=[]
for start in range(0, len(group_ids), 100):
    ids = sorted(group_ids)[start:start+100]
    candidates += pages(
        "verifier_candidates",
        "candidate_id,candidate_group_id,candidate_kind,success,metadata_json",
        lambda q, ids=ids: q.in_("candidate_group_id", ids), ("candidate_id",))
by_group={}
for candidate in candidates:
    by_group.setdefault(candidate["candidate_group_id"], []).append(candidate)
benchmark_by_group = {row["candidate_group_id"]: row["benchmark"] for row in groups}
records=[]
for gid, members in sorted(by_group.items()):
    if len(members) != CANDIDATE_COUNT: continue
    outcomes = np.asarray([bool(row["success"]) for row in members])
    scores = np.asarray([
        float((row.get("metadata_json") or {})["verifier_score"]) for row in members])
    positive, negative = scores[outcomes], scores[~outcomes]
    pair_accuracy = margin = float("nan"); comparisons = 0
    if len(positive) and len(negative):
        differences = positive[:, None] - negative[None, :]
        pair_accuracy = float(((differences > 0) + .5*(differences == 0)).mean())
        margin = float(differences.mean()); comparisons = int(differences.size)
    chosen = int(np.argmax(scores))
    default = next((i for i, row in enumerate(members)
                    if row["candidate_kind"] == "default"), 0)
    records.append({
        "group_id": gid, "benchmark": benchmark_by_group[gid],
        "uncertainty_stratum": "high", "pair_accuracy": pair_accuracy,
        "margin": margin, "comparisons": comparisons,
        "top1": float(outcomes[chosen]), "default": float(outcomes[default]),
        "random": float(outcomes.mean()), "oracle": float(outcomes.max()),
    })
report = summarize_candidate_records(records, seed=159)
output = Path(package_dir) / "analysis_outputs" / "verifier_online_selection.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report, indent=2))'''),
]


def main():
    outputs = {
        ROOT / "notebooks" / "03_verifier_experiments.ipynb": notebook(EXPERIMENT_CELLS),
        ROOT / "notebooks" / "04_collect_verifier_pairs.ipynb": notebook(COLLECTION_CELLS),
        ROOT / "notebooks" / "05_collect_targeted_verifier_groups.ipynb": notebook(
            TARGETED_COLLECTION_CELLS),
        ROOT / "notebooks" / "06_evaluate_online_verifier_selection.ipynb": notebook(
            ONLINE_EVALUATION_CELLS),
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
        worker_prefix="04_verifier_pairs_worker",
    )
    generate_workers(
        ROOT / "notebooks" / "05_collect_targeted_verifier_groups.ipynb",
        ROOT / "notebooks" / "workers",
        shard_count=3,
        worker_prefix="05_targeted_verifier_worker",
    )


if __name__ == "__main__":
    main()
