"""Build the commit-horizon re-collection notebook (nb 14) + sharded workers.

The decisive test for the commitment-horizon study: re-run candidate collection over the SAME
states as the existing verifier data, but committing more of each candidate chunk. Existing data
uses ``prefix_length=10`` (the standard LeRobot pi0.5-on-LIBERO horizon); here we collect
``prefix_length in {25, 50}``. Because ``prefix_length`` only controls how many actions
``_run_continuation`` commits before handing back to the base policy -- the replay state and the
candidate chunks are seeded identically -- this is a clean PAIRED comparison: same candidates,
different commit depth. If decidable mass and the oracle ceiling rise with the horizon, test-time
selection is horizon-gated (alive in the high-latency regime), not cooked (see nb 15 / Gate B).

Model/collection meat lives in ``pnp.verifier.collection``; this notebook only sources the state
identities from the existing candidate groups, rebuilds the episode lookup, and replays.
"""
from __future__ import annotations

from nb_common import BOOTSTRAP, ROOT, code, md, notebook, write_notebook


COMMIT_HORIZON_COLLECTION = notebook([
    md("""# 14 — Commit-horizon re-collection (H ∈ {25, 50})

Re-collect candidate groups over the **same states** as the existing verifier data, committing
more of each candidate chunk. Existing data is `prefix_length=10`; this notebook adds `25` and
`50`. `prefix_length` only changes how many candidate actions `_run_continuation` commits before
the base policy resumes, so the replay state and the candidate chunks are **identical** across
horizons — a clean paired comparison. Apply the verifier migrations first. Use the generated
three-worker notebooks for the full run; nb 15 reads the result and prints Gate B."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Load the policy and rebuild the LIBERO + LIBERO-PRO episode lookup"),
    code(r'''from collections import defaultdict
from tqdm.auto import tqdm
from pnp import libero_env, libero_pro, models
from pnp.experiments import _prepare_libero_pro_episodes
from pnp.store import SupabaseStore
from pnp.verifier import *

benchmark_dict = libero_env.init_libero_benchmark()
libero_pro.patch_torch_load()
policy, preprocess, postprocess = models.load_pi05()
device = models.default_device(); store = SupabaseStore()

suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
tasks = [(suite, task) for suite in suites for task in range(benchmark_dict[suite]().n_tasks)]
standard = libero_env.build_final_episodes(benchmark_dict, tasks=tasks)
for ep in standard: ep["benchmark"] = "libero"
pro = _prepare_libero_pro_episodes()
for ep in pro: ep["benchmark"] = "libero_pro"
episode_lookup = {(e["benchmark"], e["suite"], e["task_idx"], e.get("ep_idx", e.get("episode_idx"))): e
                  for e in standard + pro}
print({"libero": len(standard), "libero_pro": len(pro),
       "identities": len(episode_lookup), "device": str(device)})'''),
    md("""## 3. Source the state identities from the existing candidate groups

Same DEVELOPMENT experiments as nb 12/13. Each source group carries its `trajectory_seed`, so
replaying with that seed reproduces the exact same canonical state and candidate chunks. We keep
the original per-group `candidate_count` so the candidate sets match one-for-one."""),
    code(r'''COMMIT_HORIZONS = (25, 50)          # existing data already covers prefix_length=10
SOURCE_EXPERIMENTS = ("verifier-clean-pairs-v3", "verifier-clean-pairs-v4-dev",
                      "verifier-clean-pairs-v4-test", "verifier-online-selection-v1",
                      "verifier-v2-pro-development")
EXPERIMENT_FMT = "verifier-commit%d-v1"
SHARD_COUNT = 1   # Generated workers set this to 3.
SHARD_INDEX = 0
assert 0 <= SHARD_INDEX < SHARD_COUNT

source_groups = store.fetch_all(
    "verifier_candidate_groups",
    "candidate_group_id,benchmark,suite,task_idx,episode_idx,chunk_idx,"
    "trajectory_seed,uncertainty_stratum",
    configure=lambda q: q.in_("experiment", list(SOURCE_EXPERIMENTS)),
    order_by=("candidate_group_id",))

# Original candidate_count per source group (reproduce the same candidate set).
source_ids = sorted(g["candidate_group_id"] for g in source_groups)
counts = defaultdict(int)
for start in range(0, len(source_ids), 100):
    batch = source_ids[start:start + 100]
    for row in store.fetch_all("verifier_candidates", "candidate_group_id",
                               configure=lambda q, b=batch: q.in_("candidate_group_id", b)):
        counts[row["candidate_group_id"]] += 1

# Dedupe to unique physical states; drop any we cannot replay from the episode lookup.
identities, seen, missing = [], set(), 0
for g in source_groups:
    key = (g["benchmark"], g["suite"], g["task_idx"], g["episode_idx"],
           g["chunk_idx"], g["trajectory_seed"])
    if key in seen:
        continue
    seen.add(key)
    if (g["benchmark"], g["suite"], g["task_idx"], g["episode_idx"]) not in episode_lookup:
        missing += 1; continue
    identities.append({
        "benchmark": g["benchmark"], "suite": g["suite"], "task_idx": g["task_idx"],
        "episode_idx": g["episode_idx"], "chunk_idx": g["chunk_idx"],
        "trajectory_seed": g["trajectory_seed"],
        "uncertainty_stratum": g["uncertainty_stratum"] or "high",
        "candidate_count": counts[g["candidate_group_id"]] or 4})
identities.sort(key=lambda r: (r["benchmark"], r["suite"], r["task_idx"], r["episode_idx"],
                               r["chunk_idx"], str(r["trajectory_seed"])))
manifest = identities[SHARD_INDEX::SHARD_COUNT]
print({"source_groups": len(source_groups), "unique_states": len(identities),
       "unreplayable_skipped": missing, "worker_states": len(manifest),
       "horizons": COMMIT_HORIZONS})'''),
    md("""## 4. Resume-safe sharded re-collection (states × horizons)

Each state is replayed once per horizon into its own `verifier-commit{H}-v1` experiment. The env
is reset internally on every call, so one `make_env` serves both horizons. A group is skipped only
when it already has its full candidate count."""),
    code(r'''target_experiments = [EXPERIMENT_FMT % h for h in COMMIT_HORIZONS]
existing = store.fetch_all("verifier_candidate_groups", "candidate_group_id,experiment",
                           configure=lambda q: q.in_("experiment", target_experiments),
                           order_by=("candidate_group_id",))
existing_ids = sorted({row["candidate_group_id"] for row in existing})
existing_counts = defaultdict(int)
for start in range(0, len(existing_ids), 100):
    batch = existing_ids[start:start + 100]
    for row in store.fetch_all("verifier_candidates", "candidate_group_id",
                               configure=lambda q, b=batch: q.in_("candidate_group_id", b)):
        existing_counts[row["candidate_group_id"]] += 1

store.start_run("verifier_commit_horizon_collection", "libero+libero_pro",
                "verifier-commit-horizon",
                config={"horizons": list(COMMIT_HORIZONS), "unique_states": len(identities),
                        "shard_count": SHARD_COUNT, "shard_index": SHARD_INDEX,
                        "source_experiments": list(SOURCE_EXPERIMENTS)})
new_outcomes = skipped = 0
for item in tqdm(manifest, desc="commit-horizon states"):
    ep = episode_lookup[(item["benchmark"], item["suite"], item["task_idx"], item["episode_idx"])]
    ccount = int(item["candidate_count"])
    todo = []
    for h in COMMIT_HORIZONS:
        exp = EXPERIMENT_FMT % h
        gid = candidate_group_id(item["benchmark"], item["suite"], item["task_idx"],
                                 item["episode_idx"], item["chunk_idx"], namespace=exp,
                                 trajectory_seed=item["trajectory_seed"])
        if existing_counts.get(gid, 0) != ccount:
            todo.append((h, exp))
    if not todo:
        continue
    env = libero_env.make_env(ep["bddl_path"])
    try:
        for h, exp in todo:
            try:
                result = collect_replay_candidate_group(
                    env, ep, policy, preprocess, postprocess, device,
                    chunk_idx=item["chunk_idx"], uncertainty_stratum=item["uncertainty_stratum"],
                    prefix_length=h, candidate_count=ccount, experiment=exp,
                    trajectory_seed=item["trajectory_seed"],
                    collection_split="development", manifest_hash="", model_revision="pi05")
            except Exception as error:
                print("state skipped:", h, type(error).__name__, error); result = None
            if result is None:
                skipped += 1; continue
            group, candidates = result
            group["metadata_json"].update({"commit_horizon": h,
                                           "source_experiments": list(SOURCE_EXPERIMENTS)})
            store.register_candidate_group(group, candidates)
            new_outcomes += len(candidates)
    finally:
        env.close()
store.finish_run(n_rollouts=new_outcomes)
print({"new_outcomes": new_outcomes, "skipped": skipped})'''),
    md("## 5. Integrity + first decidable-mass read per horizon"),
    code(r'''for h in COMMIT_HORIZONS:
    exp = EXPERIMENT_FMT % h
    groups = store.fetch_all("verifier_candidate_groups", "candidate_group_id",
                             configure=lambda q, e=exp: q.eq("experiment", e),
                             order_by=("candidate_group_id",))
    ids = sorted({r["candidate_group_id"] for r in groups})
    outcomes, per_group = defaultdict(set), defaultdict(int)
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        for row in store.fetch_all("verifier_candidates", "candidate_group_id,success",
                                   configure=lambda q, b=batch: q.in_("candidate_group_id", b)):
            outcomes[row["candidate_group_id"]].add(bool(row["success"]))
            per_group[row["candidate_group_id"]] += 1
    discordant = sum(len(v) == 2 for v in outcomes.values())
    print({"horizon": h, "groups": len(ids), "discordant_groups": discordant,
           "discordant_fraction": round(discordant / len(ids), 3) if ids else None,
           "total_candidates": sum(per_group.values())})'''),
], "14_collect_commit_horizon.ipynb")


def main():
    notebooks = ROOT / "notebooks"
    path = notebooks / "14_collect_commit_horizon.ipynb"
    write_notebook(path, COMMIT_HORIZON_COLLECTION)
    # Keep upload-ready three-way worker copies synchronized with the canonical notebook.
    from generate_verifier_workers import generate_workers
    generate_workers(path, notebooks / "workers", shard_count=3,
                     worker_prefix="14_collect_commit_horizon_worker")


if __name__ == "__main__":
    main()
