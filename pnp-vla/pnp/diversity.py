"""Two-model episode-bootstrap training and LIBERO-PRO diversity-signal analysis.

The training bootstrap is task-stratified: every member receives the same number of draws from
every task, but demonstrations are sampled with replacement independently per member. Repeated
episodes are implemented by repeating their frame ranges in the training sampler, so no video or
parquet data needs to be copied.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DIVERSITY_MANIFEST_VERSION = 1
DIVERSITY_DATASET_REPO = "HuggingFaceVLA/libero"
DIVERSITY_SOURCE_MODEL = "lerobot/pi05_base"
DIVERSITY_EXPECTED_TASKS = 40
DIVERSITY_EXPECTED_EPISODES = 1693
DIVERSITY_EXPERIMENT_PREFIX = "pi05-diversity-signal-v1"
DIVERSITY_V2_EXPERIMENT_PREFIX = "pi05-diversity-signal-v2"
DIVERSITY_PAIR_KEYS = ["suite", "task_idx", "episode_idx", "init_state_hash"]
DIVERSITY_PREFIX_CHUNKS = (1, 2, 4, 8)
DIVERSITY_FIXED_REFINEMENT_THRESHOLD = 0.03


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _task_key(row: dict) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, (list, tuple)) and tasks:
        return " || ".join(sorted(map(str, tasks)))
    for name in ("task", "task_name", "task_index"):
        if row.get(name) is not None:
            return str(row[name])
    raise ValueError(f"episode {row.get('episode_index')} has no task identity")


def build_bootstrap_manifest(episode_rows: Iterable[dict], *, n_members: int = 2,
                             seed: int = 159,
                             dataset_repo_id: str = DIVERSITY_DATASET_REPO,
                             dataset_revision: str | None = None,
                             source_model: str = DIVERSITY_SOURCE_MODEL,
                             source_model_revision: str | None = None) -> dict:
    """Build independent within-task episode bootstraps for ``n_members`` models."""
    rows = []
    for raw in episode_rows:
        row = dict(raw)
        episode_index = int(row["episode_index"])
        rows.append({"episode_index": episode_index, "task_key": _task_key(row)})
    if not rows:
        raise ValueError("cannot bootstrap an empty episode table")
    if len({row["episode_index"] for row in rows}) != len(rows):
        raise ValueError("episode_index must be unique in source metadata")

    by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_task[row["task_key"]].append(row["episode_index"])
    by_task = {task: sorted(indices) for task, indices in sorted(by_task.items())}
    members = []
    for member_index in range(n_members):
        member_seed = seed + 10_007 * member_index
        rng = np.random.default_rng(member_seed)
        draws, task_draws = [], {}
        for task, indices in by_task.items():
            sampled = rng.choice(indices, size=len(indices), replace=True).astype(int).tolist()
            task_draws[task] = sampled
            draws.extend(sampled)
        counts = Counter(draws)
        all_episodes = sorted(row["episode_index"] for row in rows)
        members.append({
            "member_index": member_index,
            "seed": member_seed,
            "draws": draws,
            "task_draws": task_draws,
            "multiplicities": {str(index): count for index, count in sorted(counts.items())},
            "unique_episode_indices": sorted(counts),
            "out_of_bag_episode_indices": sorted(set(all_episodes) - set(counts)),
        })
    manifest = {
        "manifest_version": DIVERSITY_MANIFEST_VERSION,
        "design": "independent_task_stratified_episode_bootstrap_with_replacement",
        "seed": seed,
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": dataset_revision,
        "source_model": source_model,
        "source_model_revision": source_model_revision,
        "n_source_episodes": len(rows),
        "n_tasks": len(by_task),
        "source_episodes": rows,
        "task_episode_indices": by_task,
        "members": members,
    }
    manifest["source_metadata_hash"] = _canonical_hash(rows)
    manifest["manifest_hash"] = _canonical_hash(manifest)
    validate_bootstrap_manifest(manifest)
    return manifest


def validate_bootstrap_manifest(manifest: dict) -> None:
    if manifest.get("manifest_version") != DIVERSITY_MANIFEST_VERSION:
        raise ValueError("unsupported diversity manifest version")
    task_episodes = {
        str(task): set(map(int, episodes))
        for task, episodes in manifest["task_episode_indices"].items()
    }
    source = set().union(*task_episodes.values())
    if len(source) != int(manifest["n_source_episodes"]):
        raise ValueError("manifest source episode count is inconsistent")
    if len(manifest["members"]) != 2:
        raise ValueError("the signal experiment requires exactly two members")
    for expected_index, member in enumerate(manifest["members"]):
        if int(member["member_index"]) != expected_index:
            raise ValueError("members must be indexed 0,1")
        flattened = []
        for task, source_indices in task_episodes.items():
            draws = list(map(int, member["task_draws"].get(task, [])))
            if len(draws) != len(source_indices):
                raise ValueError(f"member {expected_index} changed draw count for task {task}")
            if not set(draws).issubset(source_indices):
                raise ValueError(f"member {expected_index} crossed task boundaries")
            flattened.extend(draws)
        if Counter(flattened) != Counter(map(int, member["draws"])):
            raise ValueError(f"member {expected_index} draw list is inconsistent")
        recorded = {int(index): int(count)
                    for index, count in member["multiplicities"].items()}
        if Counter(flattened) != Counter(recorded):
            raise ValueError(f"member {expected_index} multiplicities are inconsistent")
        if set(map(int, member["out_of_bag_episode_indices"])) != source - set(flattened):
            raise ValueError(f"member {expected_index} out-of-bag list is inconsistent")

    recorded_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    if not recorded_hash or recorded_hash != _canonical_hash(unhashed):
        raise ValueError("manifest hash is missing or inconsistent; do not train from this file")


def save_bootstrap_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_bootstrap_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_bootstrap_manifest(manifest)
    return manifest


def bootstrap_manifest_summary(manifest: dict) -> pd.DataFrame:
    rows = []
    for member in manifest["members"]:
        unique = set(map(int, member["unique_episode_indices"]))
        rows.append({
            "member_index": int(member["member_index"]), "seed": int(member["seed"]),
            "draws": len(member["draws"]), "unique_episodes": len(unique),
            "out_of_bag_episodes": len(member["out_of_bag_episode_indices"]),
            "unique_fraction": len(unique) / manifest["n_source_episodes"],
        })
    first = set(manifest["members"][0]["unique_episode_indices"])
    second = set(manifest["members"][1]["unique_episode_indices"])
    overlap = len(first & second) / max(1, len(first | second))
    out = pd.DataFrame(rows)
    out["member_unique_jaccard"] = overlap
    out["manifest_hash"] = manifest["manifest_hash"]
    return out


def build_bootstrap_manifest_from_lerobot(*, dataset_repo_id: str = DIVERSITY_DATASET_REPO,
                                          dataset_revision: str | None = None,
                                          source_model: str = DIVERSITY_SOURCE_MODEL,
                                          source_model_revision: str | None = None,
                                          seed: int = 159) -> dict:
    """Download only LeRobot metadata and construct the two-member manifest."""
    from huggingface_hub import HfApi
    from lerobot.datasets import LeRobotDatasetMetadata

    api = HfApi()
    dataset_revision = dataset_revision or api.dataset_info(dataset_repo_id).sha
    source_model_revision = source_model_revision or api.model_info(source_model).sha
    metadata = LeRobotDatasetMetadata(dataset_repo_id, revision=dataset_revision)
    rows = [dict(metadata.episodes[index]) for index in range(len(metadata.episodes))]
    manifest = build_bootstrap_manifest(
        rows, seed=seed, dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision, source_model=source_model,
        source_model_revision=source_model_revision)
    if manifest["n_tasks"] != DIVERSITY_EXPECTED_TASKS:
        raise ValueError(
            f"expected {DIVERSITY_EXPECTED_TASKS} LIBERO tasks, found {manifest['n_tasks']}")
    if manifest["n_source_episodes"] != DIVERSITY_EXPECTED_EPISODES:
        raise ValueError(
            f"expected {DIVERSITY_EXPECTED_EPISODES} LIBERO demonstrations, found "
            f"{manifest['n_source_episodes']}")
    return manifest


def bootstrap_sampler_class(manifest: dict, member_index: int):
    """Return an EpisodeAwareSampler replacement implementing manifest multiplicities."""
    validate_bootstrap_manifest(manifest)
    member = manifest["members"][int(member_index)]
    multiplicities = {int(index): int(count)
                      for index, count in member["multiplicities"].items()}

    class ManifestBootstrapEpisodeAwareSampler:
        def __init__(self, dataset_from_indices, dataset_to_indices,
                     episode_indices_to_use=None, drop_n_first_frames=0,
                     drop_n_last_frames=0, shuffle=False):
            if episode_indices_to_use is not None:
                raise ValueError(
                    "bootstrap training must load the full dataset; do not set dataset.episodes")
            if drop_n_first_frames < 0 or drop_n_last_frames < 0:
                raise ValueError("dropped frame counts must be non-negative")
            indices = []
            for episode_index, count in sorted(multiplicities.items()):
                start = int(dataset_from_indices[episode_index]) + int(drop_n_first_frames)
                end = int(dataset_to_indices[episode_index]) - int(drop_n_last_frames)
                if end <= start:
                    continue
                episode_frames = list(range(start, end))
                for _ in range(count):
                    indices.extend(episode_frames)
            if not indices:
                raise ValueError("bootstrap sampler has no valid frames")
            self.indices = indices
            self.shuffle = bool(shuffle)

        def __iter__(self):
            import torch
            if self.shuffle:
                for offset in torch.randperm(len(self.indices)):
                    yield self.indices[int(offset)]
            else:
                yield from self.indices

        def __len__(self):
            return len(self.indices)

    ManifestBootstrapEpisodeAwareSampler.__name__ = (
        f"ManifestBootstrapEpisodeAwareSampler{member_index}")
    return ManifestBootstrapEpisodeAwareSampler


def install_bootstrap_sampler(manifest: dict, member_index: int):
    """Patch the exact symbol used by the pinned LeRobot trainer."""
    import lerobot.scripts.lerobot_train as trainer

    sampler = bootstrap_sampler_class(manifest, member_index)
    trainer.EpisodeAwareSampler = sampler
    return trainer


def require_full_finetune_gpu(*, minimum_gib: float = 70.0) -> dict:
    """Fail before model download when a requested full fine-tune cannot fit."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("pi0.5 fine-tuning requires a CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 2 ** 30
    if total_gib < minimum_gib:
        raise RuntimeError(
            f"Full pi0.5 fine-tuning requires an ~80GB accelerator; found "
            f"{props.name} with {total_gib:.1f} GiB. Use an A100/H100 80GB, or explicitly "
            "switch the notebook to expert-only training rather than silently changing the design.")
    return {"gpu": props.name, "memory_gib": total_gib}


def diversity_experiment(member_index: int,
                         experiment_prefix: str = DIVERSITY_EXPERIMENT_PREFIX) -> str:
    member_index = int(member_index)
    if member_index not in (0, 1):
        raise ValueError("diversity member_index must be 0 or 1")
    experiment_prefix = str(experiment_prefix).strip()
    if not experiment_prefix:
        raise ValueError("diversity experiment_prefix cannot be empty")
    return f"{experiment_prefix}-m{member_index}"


def run_diversity_signal_worker(*, member_index: int, model_repo_id: str,
                                shard_count: int = 1, shard_index: int = 0,
                                episodes_per_task: int = 2,
                                manifest_hash: str = "",
                                experiment_prefix: str = DIVERSITY_EXPERIMENT_PREFIX):
    """Collect one model's matched 13-suite PRO rollouts with K=5 uncertainty telemetry."""
    from . import models
    from .config import Method, RolloutConfig
    from .experiments import (_prepare_libero_pro_expanded_episodes, _run_collection,
                              identity_shard)
    from .store import SupabaseStore, gather_provenance
    from huggingface_hub import HfApi

    if not manifest_hash:
        raise ValueError("manifest_hash is required; load the shared Drive manifest")
    episodes = identity_shard(
        _prepare_libero_pro_expanded_episodes(episodes_per_task), shard_count, shard_index)
    config = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, save_trajectory=True,
        save_generated_chunks=True, skip_unused_renders=True, render_lead=2)
    model_revision = HfApi().model_info(model_repo_id).sha
    policy, preprocess, postprocess = models.load_pi05(
        repo_id=model_repo_id, revision=model_revision)
    device, store = models.default_device(), SupabaseStore()
    experiment = diversity_experiment(member_index, experiment_prefix)
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=device, experiment=experiment, episodes=episodes,
        methods=[(Method.UNCERTAINTY, config)],
        cohort=f"{experiment_prefix}_m{member_index}",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_diversity_signal",
        run_metadata={"member_index": int(member_index), "model_repo_id": model_repo_id,
                      "experiment_prefix": experiment_prefix,
                      "bootstrap_manifest_hash": manifest_hash,
                      "model_revision": model_revision, "pnp_k": 5,
                      "pnp_steps": [3, 4], "episodes_per_task": episodes_per_task},
        provenance=gather_provenance(
            model_repo_id=model_repo_id, model_revision=model_revision),
    )


def diversity_model_source(store, *, member_index: int,
                            experiment_prefix: str = DIVERSITY_V2_EXPERIMENT_PREFIX
                            ) -> tuple[str, str]:
    """Return the single immutable (repo, revision) recorded by a member's baseline runs."""
    experiment = diversity_experiment(member_index, experiment_prefix)
    runs = store.fetch_all(
        "experiment_runs", "model_repo_id,model_revision",
        configure=lambda query: query.eq("experiment", experiment),
        order_by=("run_id",))
    sources = {
        (str(run.get("model_repo_id") or ""), str(run.get("model_revision") or ""))
        for run in runs if run.get("model_repo_id") and run.get("model_revision")
    }
    if len(sources) != 1:
        raise ValueError(
            f"{experiment} must contain exactly one recorded model source; found {sorted(sources)}")
    return next(iter(sources))


def diversity_baseline_cohort(store, *, expected_per_suite: int = 10,
                              experiment_prefix: str = DIVERSITY_V2_EXPERIMENT_PREFIX
                              ) -> pd.DataFrame:
    """Load and validate the exact matched baseline identities to be refined."""
    from .config import Method

    frames = []
    for member_index in (0, 1):
        experiment = diversity_experiment(member_index, experiment_prefix)
        rows = store.fetch_all(
            "rollouts",
            "rollout_id,suite,task_idx,episode_idx,init_state_hash,status,success,method,"
            "pnp_k,pnp_step_indices",
            configure=lambda query, exp=experiment: query.eq(
                "experiment", exp).eq("method", Method.UNCERTAINTY),
            order_by=("rollout_id",))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"no completed diversity baseline found for {experiment}")
        frame["member_index"] = member_index
        if frame.status.ne("completed").any():
            raise ValueError(f"{experiment} contains non-completed baseline rows")
        if set(frame.pnp_k.astype(int)) != {5}:
            raise ValueError(f"{experiment} baseline is not uniformly K=5")
        schedules = {tuple(value or []) for value in frame.pnp_step_indices}
        if schedules != {(3, 4)}:
            raise ValueError(f"{experiment} baseline schedules are {sorted(schedules)}, not (3,4)")
        if frame.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"{experiment} contains duplicate baseline identities")
        counts = frame.groupby("suite").size()
        if len(counts) != 13 or set(counts.astype(int)) != {int(expected_per_suite)}:
            raise ValueError(
                f"{experiment} must contain {expected_per_suite} baselines in each of 13 suites; "
                f"found {counts.to_dict()}")
        frames.append(frame)
    first = set(map(tuple, frames[0][DIVERSITY_PAIR_KEYS].to_numpy()))
    second = set(map(tuple, frames[1][DIVERSITY_PAIR_KEYS].to_numpy()))
    if first != second:
        raise ValueError(
            f"member baseline cohorts differ: m0-only={len(first - second)}, "
            f"m1-only={len(second - first)}")
    return pd.concat(frames, ignore_index=True)


def run_diversity_refinement_worker(*, member_index: int,
                                    shard_count: int = 4, shard_index: int = 0,
                                    episodes_per_task: int = 10,
                                    manifest_hash: str = "",
                                    experiment_prefix: str = DIVERSITY_V2_EXPERIMENT_PREFIX):
    """Collect a 10-episode/task K=5 (3,4) baseline/refinement cohort.

    Both arms are requested deliberately. Their IDs are behavior-derived, so completed
    uncertainty-only/refinement rows are skipped and only missing rows execute. The model repo and
    immutable revision come from the existing baseline experiment, preventing a later Hub upload
    from silently changing the paired policy.
    """
    from . import models
    from .config import Method, RolloutConfig
    from .experiments import (_prepare_libero_pro_expanded_episodes, _run_collection,
                              identity_shard)
    from .store import SupabaseStore, gather_provenance

    if not manifest_hash:
        raise ValueError("manifest_hash is required; load the shared Drive manifest")
    store = SupabaseStore()
    model_repo_id, model_revision = diversity_model_source(
        store, member_index=member_index, experiment_prefix=experiment_prefix)
    manifest = _prepare_libero_pro_expanded_episodes(episodes_per_task)
    expected_identities = 13 * 10 * int(episodes_per_task)
    if len(manifest) != expected_identities:
        raise ValueError(
            f"expected {expected_identities} identities for 13 suites x 10 tasks x "
            f"{episodes_per_task} episodes, found {len(manifest)}")
    episodes = identity_shard(manifest, shard_count, shard_index)
    shard_identities = {
        (ep["suite"], ep["task_idx"], ep["ep_idx"], ep["init_state_hash"])
        for ep in episodes}
    experiment = diversity_experiment(member_index, experiment_prefix)
    baseline_rows = store.fetch_all(
        "rollouts", "suite,task_idx,episode_idx,init_state_hash,status,success,method,pnp_k,"
        "pnp_step_indices",
        configure=lambda query: query.eq(
            "experiment", experiment).eq("method", Method.UNCERTAINTY),
        order_by=("rollout_id",))
    baseline_member = pd.DataFrame(baseline_rows)
    if not baseline_member.empty:
        if baseline_member.status.ne("completed").any():
            raise ValueError(f"{experiment} contains non-completed baseline rows")
        if set(baseline_member.pnp_k.astype(int)) != {5} or {
                tuple(value or []) for value in baseline_member.pnp_step_indices} != {(3, 4)}:
            raise ValueError(f"{experiment} contains a baseline other than K=5 steps (3,4)")
        shard_baseline = baseline_member[
            baseline_member[DIVERSITY_PAIR_KEYS].apply(tuple, axis=1).isin(shard_identities)]
    else:
        shard_baseline = baseline_member
    initial_tally = {} if shard_baseline.empty else {
        (suite, Method.UNCERTAINTY): [len(group), int(group.success.astype(bool).sum())]
        for suite, group in shard_baseline.groupby("suite", sort=True)}
    common = dict(pnp_steps=(3, 4), pnp_k=5, save_trajectory=True,
                  skip_unused_renders=True, render_lead=2)
    methods = [
        (Method.UNCERTAINTY, RolloutConfig(**common, save_generated_chunks=True)),
        (Method.REFINEMENT, RolloutConfig(**common, refine=True)),
    ]
    print({"member": int(member_index), "model_repo_id": model_repo_id,
           "model_revision": model_revision, "experiment_prefix": experiment_prefix,
           "target_cohort_size": len(manifest), "episodes_in_shard": len(episodes),
           "requested_arms": [name for name, _ in methods]})
    policy, preprocess, postprocess = models.load_pi05(
        repo_id=model_repo_id, revision=model_revision)
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(),
        experiment=experiment,
        episodes=episodes, methods=methods,
        cohort=f"{experiment_prefix}_selective_refinement_m{member_index}",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_diversity_selective_refinement",
        run_metadata={"member_index": int(member_index), "model_repo_id": model_repo_id,
                      "experiment_prefix": experiment_prefix,
                      "bootstrap_manifest_hash": manifest_hash,
                      "model_revision": model_revision, "pnp_k": 5,
                      "pnp_steps": [3, 4], "episodes_per_task": episodes_per_task,
                      "requested_methods": [name for name, _ in methods]},
        provenance=gather_provenance(
            model_repo_id=model_repo_id, model_revision=model_revision),
        report_every=50, initial_tally=initial_tally,
    )


def _fetch_for_rollouts(store, table: str, rollout_ids: list[str], columns: str = "*") -> list[dict]:
    rows = []
    for start in range(0, len(rollout_ids), 100):
        batch = rollout_ids[start:start + 100]
        rows.extend(store.fetch_all(
            table, columns, configure=lambda query, ids=batch: query.in_("rollout_id", ids),
            order_by=("rollout_id",)))
    return rows


def fetch_diversity_signal(store, *, experiment_prefix: str = DIVERSITY_EXPERIMENT_PREFIX,
                           methods: Iterable[str] | None = None
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch matched diversity rows; default to baseline so notebook 19 remains stable."""
    from .config import Method

    methods = tuple(methods) if methods is not None else (Method.UNCERTAINTY,)
    if not methods:
        raise ValueError("at least one diversity method must be requested")
    rollout_frames, step_frames = [], []
    for member_index in (0, 1):
        experiment = diversity_experiment(member_index, experiment_prefix)
        rows = store.fetch_all(
            "rollouts", "*", configure=lambda query, exp=experiment: query.eq("experiment", exp),
            order_by=("rollout_id",))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"no rollouts found for {experiment}")
        frame = frame[frame.method.isin(methods)].copy()
        if frame.empty:
            raise ValueError(f"no {list(methods)} rollouts found for {experiment}")
        frame["member_index"] = member_index
        runs = store.fetch_all(
            "experiment_runs", "run_id,model_revision,model_repo_id",
            configure=lambda query, exp=experiment: query.eq("experiment", exp),
            order_by=("run_id",))
        run_revisions = {
            str(run["run_id"]): str(run.get("model_revision") or "") for run in runs}
        frame["model_revision"] = frame.run_id.astype(str).map(run_revisions)
        revisions = frame.model_revision.replace("", np.nan).dropna().unique()
        if len(revisions) != 1:
            raise ValueError(
                f"{experiment} must contain exactly one model revision, found {list(revisions)}")
        rollout_frames.append(frame)
        ids = frame.rollout_id.astype(str).tolist()
        steps = pd.DataFrame(_fetch_for_rollouts(
            store, "pnp_euler_steps", ids,
            "rollout_id,chunk_idx,euler_step,u_mean"))
        steps["member_index"] = member_index
        step_frames.append(steps)
    return pd.concat(rollout_frames, ignore_index=True), pd.concat(step_frames, ignore_index=True)


def fetch_diversity_selective_refinement(
        store, *, experiment_prefix: str = DIVERSITY_V2_EXPERIMENT_PREFIX
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the two matched baseline/refinement arms for both v2 members."""
    from .config import Method

    return fetch_diversity_signal(
        store, experiment_prefix=experiment_prefix,
        methods=(Method.UNCERTAINTY, Method.REFINEMENT))


def _rank_auc(labels, scores) -> float:
    labels = np.asarray(labels, bool)
    scores = pd.Series(np.asarray(scores, float))
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep].reset_index(drop=True)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return math.nan
    rank_sum = float(scores.rank(method="average").to_numpy()[labels].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def _paired_signal(rollouts: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    first_u = (steps[steps.chunk_idx == 0].groupby(["member_index", "rollout_id"])
               .u_mean.mean().rename("u_first_chunk").reset_index())
    frame = rollouts.merge(first_u, on=["member_index", "rollout_id"], how="left")
    incomplete = frame.status.ne("completed")
    if incomplete.any():
        bad = frame.loc[incomplete, ["member_index", "rollout_id", "status"]]
        raise ValueError(f"diversity analysis requires completed rows:\n{bad.to_string(index=False)}")
    if frame.u_first_chunk.isna().any():
        bad = frame.loc[frame.u_first_chunk.isna(), ["member_index", "rollout_id"]]
        raise ValueError(f"missing first-chunk uncertainty:\n{bad.to_string(index=False)}")
    members = []
    for index in (0, 1):
        member = frame[frame.member_index == index].copy()
        if member.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"member {index} has duplicate episode identities")
        keep = DIVERSITY_PAIR_KEYS + [
            "rollout_id", "success", "status", "u_mean_episode", "u_first_chunk",
            "generated_chunks_path", "n_steps"]
        member = member[keep].rename(columns={
            column: f"{column}_m{index}" for column in keep if column not in DIVERSITY_PAIR_KEYS})
        members.append(member)
    paired = members[0].merge(members[1], on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    if len(paired) != len(members[0]) or len(paired) != len(members[1]):
        raise ValueError("the two model experiments do not contain identical episode identities")
    paired["success_m0"] = paired.success_m0.astype(bool)
    paired["success_m1"] = paired.success_m1.astype(bool)
    paired["discordant"] = paired.success_m0 != paired.success_m1
    paired["transition"] = np.select(
        [~paired.success_m0 & paired.success_m1,
         paired.success_m0 & ~paired.success_m1,
         paired.success_m0 & paired.success_m1],
        ["m0_fail_m1_success", "m0_success_m1_fail", "both_success"],
        default="both_fail")
    return paired


def _signal_summary(group: pd.DataFrame) -> dict:
    s0, s1 = group.success_m0.to_numpy(bool), group.success_m1.to_numpy(bool)
    first_choose_m0 = group.u_first_chunk_m0.to_numpy(float) <= group.u_first_chunk_m1.to_numpy(float)
    episode_choose_m0 = group.u_mean_episode_m0.to_numpy(float) <= group.u_mean_episode_m1.to_numpy(float)
    first_selected = np.where(first_choose_m0, s0, s1)
    episode_selected = np.where(episode_choose_m0, s0, s1)
    discordant = s0 != s1
    first_accuracy = float(first_selected[discordant].mean()) if discordant.any() else math.nan
    episode_accuracy = float(episode_selected[discordant].mean()) if discordant.any() else math.nan
    lower_first_score = group.u_first_chunk_m1 - group.u_first_chunk_m0
    lower_episode_score = group.u_mean_episode_m1 - group.u_mean_episode_m0
    m0_wins = s0[discordant]
    return {
        "n_pairs": len(group), "n_discordant": int(discordant.sum()),
        "discordant_fraction": float(discordant.mean()),
        "model0_sr": float(s0.mean()), "model1_sr": float(s1.mean()),
        "mean_member_sr": float((s0.mean() + s1.mean()) / 2),
        "best_member_sr": float(max(s0.mean(), s1.mean())),
        "oracle_either_success_sr": float((s0 | s1).mean()),
        "lower_first_chunk_u_sr": float(first_selected.mean()),
        "lower_episode_u_sr_posthoc": float(episode_selected.mean()),
        "lower_first_chunk_u_accuracy_discordant": first_accuracy,
        "lower_episode_u_accuracy_discordant_posthoc": episode_accuracy,
        "lower_first_chunk_u_win_auc": _rank_auc(
            m0_wins, lower_first_score.to_numpy(float)[discordant]),
        "lower_episode_u_win_auc_posthoc": _rank_auc(
            m0_wins, lower_episode_score.to_numpy(float)[discordant]),
        "first_chunk_u_ties": int(np.isclose(
            group.u_first_chunk_m0, group.u_first_chunk_m1).sum()),
        "m0_fail_m1_success": int((~s0 & s1).sum()),
        "m0_success_m1_fail": int((s0 & ~s1).sum()),
        "both_success": int((s0 & s1).sum()), "both_fail": int((~s0 & ~s1).sum()),
    }


def analyze_diversity_signal(rollouts: pd.DataFrame, steps: pd.DataFrame
                             ) -> dict[str, pd.DataFrame]:
    paired = _paired_signal(rollouts, steps)
    overall = pd.DataFrame([{"scope": "overall", **_signal_summary(paired)}])
    by_suite = pd.DataFrame([
        {"suite": suite, **_signal_summary(group)}
        for suite, group in paired.groupby("suite", sort=True)])
    return {"diversity_paired_episodes": paired,
            "diversity_signal_overall": overall,
            "diversity_signal_by_suite": by_suite}


def _paired_delta_ci(baseline, policy, *, n_boot: int = 5000,
                     seed: int = 159) -> tuple[float, float]:
    baseline = np.asarray(baseline, bool)
    policy = np.asarray(policy, bool)
    if len(baseline) != len(policy) or not len(baseline):
        return math.nan, math.nan
    delta = policy.astype(float) - baseline.astype(float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    return tuple(map(float, np.quantile(sampled, [.025, .975])))


def _selective_score_specs() -> list[dict]:
    specs = [{"score_name": "first_chunk", "column": "u_prefix_1",
              "signal_kind": "prefix", "chunk_count": 1,
              "interpretation": "deployable_initial_observation"}]
    specs.extend(
        {"score_name": f"prefix_{count}_chunks", "column": f"u_prefix_{count}",
         "signal_kind": "prefix", "chunk_count": count,
         "interpretation": "posthoc_separate_trajectories"}
        for count in DIVERSITY_PREFIX_CHUNKS if count != 1)
    specs.append({"score_name": "full_episode", "column": "u_full_episode",
                  "signal_kind": "full_episode", "chunk_count": math.nan,
                  "interpretation": "posthoc_separate_trajectories"})
    specs.extend(
        {"score_name": f"chunk_{index}", "column": f"u_chunk_{index}",
         "signal_kind": "individual_chunk", "chunk_count": index + 1,
         "interpretation": ("deployable_initial_observation" if index == 0
                            else "posthoc_separate_trajectories")}
        for index in range(8))
    return specs


def build_selective_refinement_pairs(rollouts: pd.DataFrame, steps: pd.DataFrame
                                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair both models and arms, then materialize lower-U policies for several horizons."""
    from .config import Method

    required_methods = {Method.UNCERTAINTY, Method.REFINEMENT}
    if set(rollouts.method.unique()) != required_methods:
        raise ValueError(
            f"selective refinement requires exactly {sorted(required_methods)}; "
            f"found {sorted(rollouts.method.unique())}")
    incomplete = rollouts.status.ne("completed")
    if incomplete.any():
        bad = rollouts.loc[incomplete, ["member_index", "method", "rollout_id", "status"]]
        raise ValueError(f"selective refinement requires completed rows:\n{bad.to_string(index=False)}")

    observed = rollouts[rollouts.method == Method.UNCERTAINTY].copy()
    refined = rollouts[rollouts.method == Method.REFINEMENT].copy()
    observed_ids = set(observed.rollout_id.astype(str))
    observed_steps = steps[steps.rollout_id.astype(str).isin(observed_ids)].copy()
    chunk_scores = (observed_steps.groupby(["member_index", "rollout_id", "chunk_idx"])
                    .u_mean.mean().unstack("chunk_idx"))
    score_rows = chunk_scores.index.to_frame(index=False)
    for index in range(8):
        score_rows[f"u_chunk_{index}"] = (
            chunk_scores[index].to_numpy() if index in chunk_scores else math.nan)
    for count in DIVERSITY_PREFIX_CHUNKS:
        columns = [index for index in range(count) if index in chunk_scores]
        if len(columns) != count:
            score_rows[f"u_prefix_{count}"] = math.nan
            continue
        values = chunk_scores[columns]
        score_rows[f"u_prefix_{count}"] = values.mean(axis=1).where(
            values.notna().all(axis=1)).to_numpy()
    score_rows = score_rows.merge(
        observed[["member_index", "rollout_id", "u_mean_episode"]],
        on=["member_index", "rollout_id"], validate="one_to_one")
    score_rows = score_rows.rename(columns={"u_mean_episode": "u_full_episode"})

    members = []
    score_columns = [spec["column"] for spec in _selective_score_specs()]
    score_columns = list(dict.fromkeys(score_columns))
    for member_index in (0, 1):
        member_observed = observed[observed.member_index == member_index].copy()
        member_refined = refined[refined.member_index == member_index].copy()
        for label, frame in (("observed", member_observed), ("refined", member_refined)):
            if frame.duplicated(DIVERSITY_PAIR_KEYS).any():
                raise ValueError(f"member {member_index} has duplicate {label} identities")
        member = member_observed[DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]].merge(
            member_refined[DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]],
            on=DIVERSITY_PAIR_KEYS, suffixes=("_observed", "_refined"),
            validate="one_to_one")
        if len(member) != len(member_observed) or len(member) != len(member_refined):
            raise ValueError(f"member {member_index} baseline/refinement identities do not match")
        member = member.merge(
            score_rows[score_rows.member_index == member_index][
                ["rollout_id"] + score_columns],
            left_on="rollout_id_observed", right_on="rollout_id", validate="one_to_one")
        member = member.drop(columns="rollout_id").rename(columns={
            column: f"{column}_m{member_index}"
            for column in member.columns if column not in DIVERSITY_PAIR_KEYS})
        members.append(member)
    paired = members[0].merge(members[1], on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    if len(paired) != len(members[0]) or len(paired) != len(members[1]):
        raise ValueError("the two members do not contain identical selective-refinement identities")

    policy_frames = []
    for order, spec in enumerate(_selective_score_specs()):
        column = spec["column"]
        valid = paired[f"{column}_m0"].notna() & paired[f"{column}_m1"].notna()
        frame = paired.loc[valid, DIVERSITY_PAIR_KEYS].copy()
        u0 = paired.loc[valid, f"{column}_m0"].to_numpy(float)
        u1 = paired.loc[valid, f"{column}_m1"].to_numpy(float)
        choose_m0 = u0 <= u1
        baseline0 = paired.loc[valid, "success_observed_m0"].to_numpy(bool)
        baseline1 = paired.loc[valid, "success_observed_m1"].to_numpy(bool)
        refined0 = paired.loc[valid, "success_refined_m0"].to_numpy(bool)
        refined1 = paired.loc[valid, "success_refined_m1"].to_numpy(bool)
        frame["score_name"] = spec["score_name"]
        frame["score_order"] = order
        frame["signal_kind"] = spec["signal_kind"]
        frame["chunk_count"] = spec["chunk_count"]
        frame["interpretation"] = spec["interpretation"]
        frame["u_m0"] = u0
        frame["u_m1"] = u1
        frame["selected_member"] = np.where(choose_m0, 0, 1)
        frame["selected_u"] = np.minimum(u0, u1)
        frame["baseline_success"] = np.where(choose_m0, baseline0, baseline1)
        frame["refined_success"] = np.where(choose_m0, refined0, refined1)
        frame["model0_baseline_success"] = baseline0
        frame["model1_baseline_success"] = baseline1
        policy_frames.append(frame)
    return paired, pd.concat(policy_frames, ignore_index=True)


def build_member_refinement_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    """Long-form per-member baseline/refinement outcomes for each U horizon."""
    frames = []
    for member_index in (0, 1):
        for order, spec in enumerate(_selective_score_specs()):
            score_column = f"{spec['column']}_m{member_index}"
            valid = paired[score_column].notna()
            frame = paired.loc[valid, DIVERSITY_PAIR_KEYS].copy()
            frame["member_index"] = member_index
            frame["score_name"] = spec["score_name"]
            frame["score_order"] = order
            frame["signal_kind"] = spec["signal_kind"]
            frame["interpretation"] = spec["interpretation"]
            frame["u"] = paired.loc[valid, score_column].to_numpy(float)
            frame["baseline_success"] = paired.loc[
                valid, f"success_observed_m{member_index}"].to_numpy(bool)
            frame["refined_success"] = paired.loc[
                valid, f"success_refined_m{member_index}"].to_numpy(bool)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_checkpoint_refinement_pairs(
        rollouts: pd.DataFrame, steps: pd.DataFrame, *,
        checkpoint_name: str = "source_checkpoint") -> pd.DataFrame:
    """Long-form uncertainty/refinement pairs for one checkpoint and matched identities."""
    from .config import Method

    required_methods = {Method.UNCERTAINTY, Method.REFINEMENT}
    if set(rollouts.method.unique()) != required_methods:
        raise ValueError(
            f"checkpoint refinement requires exactly {sorted(required_methods)}; "
            f"found {sorted(rollouts.method.unique())}")
    if rollouts.status.ne("completed").any():
        raise ValueError("checkpoint refinement requires completed rollout rows")
    observed = rollouts[rollouts.method == Method.UNCERTAINTY].copy()
    refined = rollouts[rollouts.method == Method.REFINEMENT].copy()
    for label, frame in (("observed", observed), ("refined", refined)):
        if frame.duplicated(DIVERSITY_PAIR_KEYS).any():
            raise ValueError(f"checkpoint contains duplicate {label} identities")
    paired = observed[DIVERSITY_PAIR_KEYS + [
        "rollout_id", "success", "u_mean_episode"]].merge(
            refined[DIVERSITY_PAIR_KEYS + ["rollout_id", "success"]],
            on=DIVERSITY_PAIR_KEYS, suffixes=("_observed", "_refined"),
            validate="one_to_one")
    if paired.empty:
        raise ValueError("checkpoint has no matched baseline/refinement identities")

    observed_ids = set(paired.rollout_id_observed.astype(str))
    if steps.empty or "rollout_id" not in steps:
        raise ValueError("checkpoint has no saved PnP Euler-step uncertainty rows")
    observed_steps = steps[steps.rollout_id.astype(str).isin(observed_ids)].copy()
    if observed_steps.empty:
        raise ValueError("checkpoint has no Euler-step rows for its matched baseline rollouts")
    chunk_scores = (observed_steps.groupby(["rollout_id", "chunk_idx"])
                    .u_mean.mean().unstack("chunk_idx"))
    score_rows = chunk_scores.index.to_frame(index=False)
    for index in range(8):
        score_rows[f"u_chunk_{index}"] = (
            chunk_scores[index].to_numpy() if index in chunk_scores else math.nan)
    for count in DIVERSITY_PREFIX_CHUNKS:
        columns = [index for index in range(count) if index in chunk_scores]
        if len(columns) != count:
            score_rows[f"u_prefix_{count}"] = math.nan
            continue
        values = chunk_scores[columns]
        score_rows[f"u_prefix_{count}"] = values.mean(axis=1).where(
            values.notna().all(axis=1)).to_numpy()
    score_rows = score_rows.merge(
        observed[["rollout_id", "u_mean_episode"]], on="rollout_id",
        validate="one_to_one").rename(columns={"u_mean_episode": "u_full_episode"})
    paired = paired.merge(
        score_rows, left_on="rollout_id_observed", right_on="rollout_id",
        validate="one_to_one")

    frames = []
    for order, spec in enumerate(_selective_score_specs()):
        valid = paired[spec["column"]].notna()
        frame = paired.loc[valid, DIVERSITY_PAIR_KEYS].copy()
        frame["member_index"] = checkpoint_name
        frame["score_name"] = spec["score_name"]
        frame["score_order"] = order
        frame["signal_kind"] = spec["signal_kind"]
        frame["interpretation"] = spec["interpretation"]
        frame["u"] = paired.loc[valid, spec["column"]].to_numpy(float)
        frame["baseline_success"] = paired.loc[valid, "success_observed"].to_numpy(bool)
        frame["refined_success"] = paired.loc[valid, "success_refined"].to_numpy(bool)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _member_refinement_summary(group: pd.DataFrame) -> dict:
    baseline = group.baseline_success.to_numpy(bool)
    refined = group.refined_success.to_numpy(bool)
    lo, hi = _paired_delta_ci(baseline, refined)
    return {
        "n_pairs": len(group), "baseline_sr": float(baseline.mean()),
        "refinement_sr": float(refined.mean()),
        "delta_pp": float(100 * (refined.mean() - baseline.mean())),
        "delta_ci_low_pp": float(100 * lo), "delta_ci_high_pp": float(100 * hi),
        "F_to_S": int((~baseline & refined).sum()),
        "S_to_F": int((baseline & ~refined).sum()),
        "failure_auc": _rank_auc(~baseline, group.u.to_numpy(float)),
        "mean_u": float(group.u.mean()),
    }


def _analyze_refinement_pairs(member_pairs: pd.DataFrame, *, grid_size: int,
                              min_window: int) -> dict[str, pd.DataFrame]:
    overall = pd.DataFrame([
        {"member_index": member, "score_name": score_name,
         "score_order": int(group.score_order.iloc[0]),
         "signal_kind": group.signal_kind.iloc[0],
         "interpretation": group.interpretation.iloc[0],
         **_member_refinement_summary(group)}
        for (member, score_name), group in member_pairs.groupby(
            ["member_index", "score_name"], sort=False)
    ]).sort_values(["member_index", "score_order"])
    by_suite = pd.DataFrame([
        {"member_index": member, "score_name": score_name, "suite": suite,
         "score_order": int(group.score_order.iloc[0]),
         **_member_refinement_summary(group)}
        for (member, score_name, suite), group in member_pairs.groupby(
            ["member_index", "score_name", "suite"], sort=False)
    ]).sort_values(["member_index", "score_order", "suite"])

    bin_rows = []
    for (member, score_name), group in member_pairs.groupby(
            ["member_index", "score_name"], sort=False):
        group = group.copy()
        n_bins = min(10, len(group))
        group["uncertainty_bin"] = pd.qcut(
            group.u.rank(method="first"), n_bins, labels=False) + 1
        for uncertainty_bin, selected in group.groupby("uncertainty_bin", sort=True):
            bin_rows.append({
                "member_index": member, "score_name": score_name,
                "score_order": int(group.score_order.iloc[0]),
                "uncertainty_bin": int(uncertainty_bin),
                **_member_refinement_summary(selected),
            })
    bins = pd.DataFrame(bin_rows)

    sweep_rows = []
    for (member, score_name), group in member_pairs.groupby(
            ["member_index", "score_name"], sort=False):
        window_group = group.rename(columns={"u": "selected_u"})
        sweep_rows.extend({
            "member_index": member, "score_name": score_name,
            "score_order": int(group.score_order.iloc[0]),
            "analysis_type": "exploratory_in_sample", **row}
            for row in _window_rows(
                window_group, grid_size=grid_size, min_window=min_window))
    sweep = pd.DataFrame(sweep_rows)
    eligible = sweep[sweep.eligible & sweep.delta_pp.notna()].sort_values(
        ["member_index", "score_order", "delta_pp", "n_refined", "lower", "upper"],
        ascending=[True, True, False, False, True, True])
    top = eligible.groupby(["member_index", "score_name"], sort=False).head(10).copy()
    top["rank"] = top.groupby(["member_index", "score_name"], sort=False).cumcount() + 1
    return {
        "member_refinement_pairs": member_pairs,
        "member_refinement_overall": overall,
        "member_refinement_by_suite": by_suite,
        "member_refinement_by_uncertainty_bin": bins,
        "member_refinement_window_sweep": sweep,
        "member_refinement_top_windows": top,
    }


def analyze_member_refinement(paired: pd.DataFrame, *, grid_size: int = 25,
                              min_window: int = 10) -> dict[str, pd.DataFrame]:
    return _analyze_refinement_pairs(
        build_member_refinement_pairs(paired), grid_size=grid_size,
        min_window=min_window)


def analyze_checkpoint_refinement(
        rollouts: pd.DataFrame, steps: pd.DataFrame, *,
        checkpoint_name: str = "source_checkpoint", grid_size: int = 25,
        min_window: int = 10) -> dict[str, pd.DataFrame]:
    """Apply the per-model AUC/window analysis to one matched checkpoint."""
    pairs = build_checkpoint_refinement_pairs(
        rollouts, steps, checkpoint_name=checkpoint_name)
    return _analyze_refinement_pairs(
        pairs, grid_size=grid_size, min_window=min_window)


def _selective_policy_summary(group: pd.DataFrame, threshold: float) -> dict:
    baseline = group.baseline_success.to_numpy(bool)
    refined = group.refined_success.to_numpy(bool)
    model0 = group.model0_baseline_success.to_numpy(bool)
    model1 = group.model1_baseline_success.to_numpy(bool)
    discordant = model0 != model1
    selected = group.selected_u.to_numpy(float) >= threshold
    policy = np.where(selected, refined, baseline)
    lo, hi = _paired_delta_ci(baseline, policy)
    selector_accuracy = float(baseline[discordant].mean()) if discordant.any() else math.nan
    selector_score = group.u_m1.to_numpy(float) - group.u_m0.to_numpy(float)
    return {
        "n_pairs": len(group),
        "model0_sr": float(model0.mean()),
        "model1_sr": float(model1.mean()),
        "best_fixed_member_sr": float(max(model0.mean(), model1.mean())),
        "lower_u_baseline_sr": float(baseline.mean()),
        "lower_u_delta_vs_best_fixed_pp": float(
            100 * (baseline.mean() - max(model0.mean(), model1.mean()))),
        "n_model_discordant": int(discordant.sum()),
        "lower_u_selector_accuracy_discordant": selector_accuracy,
        "lower_u_selector_win_auc": _rank_auc(
            model0[discordant], selector_score[discordant]),
        "lower_u_refine_all_sr": float(refined.mean()),
        "fixed_threshold": float(threshold),
        "n_refined_fixed": int(selected.sum()),
        "coverage_refined_fixed": float(selected.mean()),
        "fixed_threshold_sr": float(policy.mean()),
        "fixed_delta_vs_lower_u_pp": float(100 * (policy.mean() - baseline.mean())),
        "fixed_delta_ci_low_pp": float(100 * lo),
        "fixed_delta_ci_high_pp": float(100 * hi),
        "fixed_F_to_S": int((selected & ~baseline & refined).sum()),
        "fixed_S_to_F": int((selected & baseline & ~refined).sum()),
        "refine_all_delta_vs_lower_u_pp": float(100 * (refined.mean() - baseline.mean())),
        "refine_all_F_to_S": int((~baseline & refined).sum()),
        "refine_all_S_to_F": int((baseline & ~refined).sum()),
    }


def _window_rows(group: pd.DataFrame, *, grid_size: int, min_window: int) -> list[dict]:
    score = group.selected_u.to_numpy(float)
    baseline = group.baseline_success.to_numpy(bool)
    refined = group.refined_success.to_numpy(bool)
    rows = []
    for lower_index, lower in enumerate(np.linspace(0., .06, grid_size)):
        for upper_index, upper in enumerate(np.linspace(.01, .08, grid_size)):
            if upper <= lower:
                continue
            selected = (score >= lower) & (score <= upper)
            policy = np.where(selected, refined, baseline)
            eligible = int(selected.sum()) >= min_window
            rows.append({
                "lower": float(lower), "upper": float(upper),
                "lower_grid_index": lower_index, "upper_grid_index": upper_index,
                "n_refined": int(selected.sum()), "coverage_refined": float(selected.mean()),
                "eligible": eligible,
                "selective_sr": float(policy.mean()) if eligible else math.nan,
                "delta_pp": float(100 * (policy.mean() - baseline.mean())) if eligible else math.nan,
                "selected_F_to_S": int((selected & ~baseline & refined).sum()),
                "selected_S_to_F": int((selected & baseline & ~refined).sum()),
            })
    return rows


def selective_refinement_window_sweep(policy_pairs: pd.DataFrame, *, grid_size: int = 25,
                                      min_window: int = 10) -> pd.DataFrame:
    rows = []
    for score_name, group in policy_pairs.groupby("score_name", sort=False):
        metadata = group.iloc[0]
        rows.extend({"score_name": score_name, "score_order": int(metadata.score_order),
                     "interpretation": metadata.interpretation,
                     "analysis_type": "exploratory_in_sample", **row}
                    for row in _window_rows(group, grid_size=grid_size,
                                            min_window=min_window))
    return pd.DataFrame(rows)


def selective_refinement_loso(policy_pairs: pd.DataFrame, *, grid_size: int = 25,
                              min_window: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose a window on 12 suites and apply it once to the held-out suite."""
    folds, predictions = [], []
    for score_name, group in policy_pairs.groupby("score_name", sort=False):
        for suite in sorted(group.suite.unique()):
            train, test = group[group.suite != suite], group[group.suite == suite].copy()
            sweep = pd.DataFrame(_window_rows(
                train, grid_size=grid_size, min_window=min_window))
            eligible = sweep[sweep.eligible & sweep.delta_pp.notna()].sort_values(
                ["delta_pp", "n_refined", "lower", "upper"],
                ascending=[False, False, True, True])
            if eligible.empty:
                raise ValueError(f"no eligible LOSO window for {score_name} excluding {suite}")
            best = eligible.iloc[0]
            selected = test.selected_u.between(best.lower, best.upper).to_numpy(bool)
            baseline = test.baseline_success.to_numpy(bool)
            refined = test.refined_success.to_numpy(bool)
            policy = np.where(selected, refined, baseline)
            folds.append({
                "score_name": score_name, "held_out_suite": suite,
                "lower": float(best.lower), "upper": float(best.upper),
                "train_delta_pp": float(best.delta_pp),
                "train_n_refined": int(best.n_refined), "test_n": len(test),
                "test_n_refined": int(selected.sum()),
                "test_baseline_sr": float(baseline.mean()),
                "test_selective_sr": float(policy.mean()),
                "test_delta_pp": float(100 * (policy.mean() - baseline.mean())),
            })
            prediction = test[DIVERSITY_PAIR_KEYS + ["score_name", "score_order",
                                                       "interpretation"]].copy()
            prediction["selected"] = selected
            prediction["baseline_success"] = baseline
            prediction["refined_success"] = refined
            prediction["policy_success"] = policy
            predictions.append(prediction)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summaries = []
    for score_name, group in prediction_frame.groupby("score_name", sort=False):
        baseline = group.baseline_success.to_numpy(bool)
        policy = group.policy_success.to_numpy(bool)
        selected = group.selected.to_numpy(bool)
        refined = group.refined_success.to_numpy(bool)
        lo, hi = _paired_delta_ci(baseline, policy)
        summaries.append({
            "score_name": score_name, "score_order": int(group.score_order.iloc[0]),
            "interpretation": group.interpretation.iloc[0], "n_pairs": len(group),
            "n_refined": int(selected.sum()), "coverage_refined": float(selected.mean()),
            "baseline_sr": float(baseline.mean()), "loso_selective_sr": float(policy.mean()),
            "delta_pp": float(100 * (policy.mean() - baseline.mean())),
            "delta_ci_low_pp": float(100 * lo), "delta_ci_high_pp": float(100 * hi),
            "F_to_S": int((selected & ~baseline & refined).sum()),
            "S_to_F": int((selected & baseline & ~refined).sum()),
        })
    return pd.DataFrame(folds), pd.DataFrame(summaries).sort_values("score_order")


def analyze_diversity_selective_refinement(
        rollouts: pd.DataFrame, steps: pd.DataFrame,
        *, fixed_threshold: float = DIVERSITY_FIXED_REFINEMENT_THRESHOLD,
        grid_size: int = 25, min_window: int = 10) -> dict[str, pd.DataFrame]:
    paired, policy_pairs = build_selective_refinement_pairs(rollouts, steps)
    overall = pd.DataFrame([
        {"score_name": score_name, "score_order": int(group.score_order.iloc[0]),
         "signal_kind": group.signal_kind.iloc[0],
         "interpretation": group.interpretation.iloc[0],
         **_selective_policy_summary(group, fixed_threshold)}
        for score_name, group in policy_pairs.groupby("score_name", sort=False)
    ]).sort_values("score_order")
    by_suite = pd.DataFrame([
        {"score_name": score_name, "suite": suite,
         "score_order": int(group.score_order.iloc[0]),
         "interpretation": group.interpretation.iloc[0],
         **_selective_policy_summary(group, fixed_threshold)}
        for (score_name, suite), group in policy_pairs.groupby(
            ["score_name", "suite"], sort=False)
    ]).sort_values(["score_order", "suite"])
    sweep = selective_refinement_window_sweep(
        policy_pairs, grid_size=grid_size, min_window=min_window)
    eligible = sweep[sweep.eligible & sweep.delta_pp.notna()].sort_values(
        ["score_order", "delta_pp", "n_refined", "lower", "upper"],
        ascending=[True, False, False, True, True])
    top_windows = eligible.groupby("score_name", sort=False).head(10).copy()
    top_windows["rank"] = top_windows.groupby("score_name", sort=False).cumcount() + 1
    folds, loso = selective_refinement_loso(
        policy_pairs, grid_size=grid_size, min_window=min_window)
    return {
        **analyze_member_refinement(
            paired, grid_size=grid_size, min_window=min_window),
        "selective_refinement_paired_episodes": paired,
        "selective_refinement_policy_pairs": policy_pairs,
        "selective_refinement_overall": overall,
        "selective_refinement_fixed_by_suite": by_suite,
        "selective_refinement_window_sweep": sweep,
        "selective_refinement_top_windows": top_windows,
        "selective_refinement_loso_folds": folds,
        "selective_refinement_loso_summary": loso,
    }


def add_first_chunk_action_disagreement(store, paired: pd.DataFrame,
                                        *, action_steps: int = 10,
                                        action_dim: int = 7,
                                        progress=None) -> pd.DataFrame:
    """Download exact generated chunks and compare the first matched model decision."""
    output = paired.copy()
    rows = output.to_dict("records")
    iterator = progress(rows, desc="first-chunk disagreement") if progress else rows
    l2_mean, l2_flat, cosine = [], [], []
    bucket = store.client.storage.from_(store.bucket)
    for row in iterator:
        chunks = []
        for member_index in (0, 1):
            path = row.get(f"generated_chunks_path_m{member_index}")
            if path is None or pd.isna(path) or not str(path):
                chunks.append(None); continue
            payload = bucket.download(path)
            with np.load(io.BytesIO(payload)) as archive:
                chunk = np.asarray(archive["chunks"])[0, :action_steps, :action_dim]
            chunks.append(chunk)
        if any(chunk is None for chunk in chunks):
            l2_mean.append(math.nan); l2_flat.append(math.nan); cosine.append(math.nan)
            continue
        delta = chunks[0] - chunks[1]
        l2_mean.append(float(np.linalg.norm(delta, axis=1).mean()))
        l2_flat.append(float(np.linalg.norm(delta)))
        a, b = chunks[0].reshape(-1), chunks[1].reshape(-1)
        cosine.append(float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))))
    output["first_chunk_action_l2_mean"] = l2_mean
    output["first_chunk_action_l2_flat"] = l2_flat
    output["first_chunk_action_cosine"] = cosine
    return output


def action_disagreement_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for discordant, group in paired.groupby("discordant", sort=True):
        rows.append({
            "outcome_group": "discordant" if discordant else "same outcome", "n": len(group),
            "mean_l2": float(group.first_chunk_action_l2_mean.mean()),
            "median_l2": float(group.first_chunk_action_l2_mean.median()),
            "mean_cosine": float(group.first_chunk_action_cosine.mean()),
        })
    auc = _rank_auc(paired.discordant, paired.first_chunk_action_l2_mean)
    out = pd.DataFrame(rows)
    out["action_disagreement_auc_for_discordant_outcome"] = auc
    return out


def diversity_signal_figures(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    written = []
    by_suite = tables["diversity_signal_by_suite"].sort_values("suite")
    labels = by_suite.suite.str.replace("libero_", "", regex=False)
    x, width = np.arange(len(by_suite)), .2
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for offset, column, label, color in [
        (-1.5, "model0_sr", "model 0", "#4C78A8"),
        (-.5, "model1_sr", "model 1", "#F58518"),
        (.5, "lower_first_chunk_u_sr", "lower first-chunk U", "#54A24B"),
        (1.5, "oracle_either_success_sr", "oracle either succeeds", "#B279A2")]:
        ax.bar(x + offset * width, 100 * by_suite[column], width, label=label, color=color)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set(ylabel="Success rate (%)", ylim=(0, 100),
           title="Two-model diversity signal on matched LIBERO-PRO episodes")
    ax.legend(fontsize=8); fig.tight_layout()
    path = output / "diversity_success_by_suite.png"; fig.savefig(path, dpi=180); plt.close(fig)
    written.append(path)

    paired = tables["diversity_paired_episodes"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = paired.transition.map({
        "both_fail": "#9D9D9D", "both_success": "#4C78A8",
        "m0_fail_m1_success": "#F58518", "m0_success_m1_fail": "#54A24B"})
    ax.scatter(paired.u_first_chunk_m0, paired.u_first_chunk_m1,
               c=colors, alpha=.65, s=22)
    limits = [float(min(paired.u_first_chunk_m0.min(), paired.u_first_chunk_m1.min())),
              float(max(paired.u_first_chunk_m0.max(), paired.u_first_chunk_m1.max()))]
    ax.plot(limits, limits, "k--", lw=1)
    ax.set(xlabel="Model 0 first-chunk uncertainty",
           ylabel="Model 1 first-chunk uncertainty",
           title="Matched uncertainty and outcome transitions")
    fig.tight_layout(); path = output / "diversity_first_chunk_uncertainty.png"
    fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    if "first_chunk_action_l2_mean" in paired:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        groups = [paired.loc[~paired.discordant, "first_chunk_action_l2_mean"].dropna(),
                  paired.loc[paired.discordant, "first_chunk_action_l2_mean"].dropna()]
        ax.boxplot(groups, labels=["same outcome", "discordant outcome"], showfliers=False)
        ax.set(ylabel="First 10 actions: mean model-to-model L2",
               title="Does action diversity create complementary outcomes?")
        fig.tight_layout(); path = output / "diversity_action_disagreement.png"
        fig.savefig(path, dpi=180); plt.close(fig); written.append(path)
    return written


def diversity_selective_refinement_figures(
        tables: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    written = []
    member_overall = tables["member_refinement_overall"]
    member_suite = tables["member_refinement_by_suite"]
    member_bins = tables["member_refinement_by_uncertainty_bin"]
    member_sweep = tables["member_refinement_window_sweep"]
    horizon_names = ["first_chunk", "prefix_2_chunks", "prefix_4_chunks",
                     "prefix_8_chunks", "full_episode"]
    for member_index in (0, 1):
        suite = member_suite[(member_suite.member_index == member_index) &
                             (member_suite.score_name == "full_episode")].sort_values("suite")
        labels = suite.suite.str.replace("libero_", "", regex=False)
        x, width = np.arange(len(suite)), .38
        fig, ax = plt.subplots(figsize=(15, 5.5))
        ax.bar(x - width / 2, 100 * suite.baseline_sr, width,
               label="uncertainty-only baseline", color="#4C78A8")
        ax.bar(x + width / 2, 100 * suite.refinement_sr, width,
               label="PnP refinement", color="#F58518")
        for index, delta in enumerate(suite.delta_pp):
            ax.text(index + width / 2, 100 * suite.refinement_sr.iloc[index] + 1,
                    f"{delta:+.1f}", ha="center", fontsize=7)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set(ylabel="Success rate (%)", ylim=(0, 105),
               title=f"Model {member_index}: matched refinement effect by suite")
        ax.legend(fontsize=8); fig.tight_layout()
        path = output / f"member_{member_index}_refinement_delta_by_suite.png"
        fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

        horizons = member_overall[(member_overall.member_index == member_index) &
                                  member_overall.score_name.isin(horizon_names)].copy()
        horizons["score_name"] = pd.Categorical(
            horizons.score_name, categories=horizon_names, ordered=True)
        horizons = horizons.sort_values("score_name")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].axhline(0, color="black", lw=1)
        axes[0].plot(np.arange(len(horizons)), horizons.delta_pp, marker="o", color="#F58518")
        axes[0].fill_between(np.arange(len(horizons)), horizons.delta_ci_low_pp,
                             horizons.delta_ci_high_pp, alpha=.18, color="#F58518")
        axes[0].set(ylabel="Refinement delta SR (pp)", title="Matched SR change")
        axes[1].axhline(.5, color="black", ls="--", lw=1, label="chance")
        axes[1].plot(np.arange(len(horizons)), horizons.failure_auc,
                     marker="o", color="#4C78A8")
        axes[1].set(ylim=(0, 1), ylabel="ROC AUC for baseline failure",
                    title="Does uncertainty detect failure?")
        tick_labels = [str(value).replace("_", " ") for value in horizons.score_name]
        for axis in axes:
            axis.set_xticks(np.arange(len(horizons)), tick_labels, rotation=25, ha="right")
        fig.suptitle(f"Model {member_index}: uncertainty horizon diagnostics")
        fig.tight_layout()
        path = output / f"member_{member_index}_delta_and_failure_auc.png"
        fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

        bins = member_bins[(member_bins.member_index == member_index) &
                           member_bins.score_name.isin(["first_chunk", "full_episode"])]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.axhline(0, color="black", lw=1)
        for score_name, group in bins.groupby("score_name", sort=False):
            ax.plot(group.uncertainty_bin, group.delta_pp, marker="o",
                    label=score_name.replace("_", " "))
        ax.set(xlabel="Within-model uncertainty decile (low to high)",
               ylabel="Refinement delta SR (pp)",
               title=f"Model {member_index}: where does refinement help?")
        ax.legend(fontsize=8); fig.tight_layout()
        path = output / f"member_{member_index}_refinement_by_uncertainty.png"
        fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

        for score_name in ("first_chunk", "full_episode"):
            group = member_sweep[(member_sweep.member_index == member_index) &
                                 (member_sweep.score_name == score_name)]
            delta = group.pivot(index="lower", columns="upper", values="delta_pp").sort_index()
            count = group.pivot(index="lower", columns="upper", values="n_refined").reindex_like(delta)
            extent = [float(delta.columns.min()), float(delta.columns.max()),
                      float(delta.index.min()), float(delta.index.max())]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            image = axes[0].imshow(delta.to_numpy(), origin="lower", aspect="auto",
                                   cmap="RdYlGn", extent=extent)
            sample = axes[1].imshow(count.to_numpy(), origin="lower", aspect="auto",
                                    cmap="Blues", extent=extent)
            fig.colorbar(image, ax=axes[0], label="In-sample delta SR (pp)")
            fig.colorbar(sample, ax=axes[1], label="Episodes refined")
            for axis in axes:
                axis.set(xlabel="Upper uncertainty", ylabel="Lower uncertainty")
            axes[0].set_title("Exploratory selective-refinement effect")
            axes[1].set_title("Window sample size")
            fig.suptitle(
                f"Model {member_index}: {score_name.replace('_', ' ')} uncertainty window")
            fig.tight_layout()
            path = output / f"member_{member_index}_window_{score_name}.png"
            fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    overall = tables["selective_refinement_overall"]
    loso = tables["selective_refinement_loso_summary"]
    horizon = overall[overall.score_name.isin(horizon_names)].merge(
        loso[["score_name", "loso_selective_sr"]], on="score_name", validate="one_to_one")
    horizon["score_name"] = pd.Categorical(
        horizon.score_name, categories=horizon_names, ordered=True)
    horizon = horizon.sort_values("score_name")
    x, width = np.arange(len(horizon)), .2
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, column, label, color in [
        (-1.5, "lower_u_baseline_sr", "lower-U baseline", "#4C78A8"),
        (-.5, "lower_u_refine_all_sr", "refine selected model always", "#F58518"),
        (.5, "fixed_threshold_sr", "refine when U >= 0.03", "#54A24B"),
        (1.5, "loso_selective_sr", "LOSO-selected window", "#B279A2")]:
        ax.bar(x + offset * width, 100 * horizon[column], width, label=label, color=color)
    ax.set_xticks(x, [str(value).replace("_", " ") for value in horizon.score_name])
    ax.set(ylabel="Success rate (%)", title="Model selection plus selective refinement by U horizon")
    ax.legend(fontsize=8); fig.tight_layout()
    path = output / "selective_refinement_by_horizon.png"
    fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].axhline(0, color="black", lw=1)
    axes[0].plot(np.arange(len(horizon)), horizon.lower_u_delta_vs_best_fixed_pp,
                 marker="o", color="#54A24B")
    axes[0].set(ylabel="Lower-U selector delta vs best fixed model (pp)",
                title="Aggregation SR gain")
    axes[1].axhline(.5, color="black", ls="--", lw=1)
    axes[1].plot(np.arange(len(horizon)), horizon.lower_u_selector_win_auc,
                 marker="o", color="#4C78A8")
    axes[1].set(ylim=(0, 1), ylabel="Win-selection ROC AUC",
                title="Can relative uncertainty select the winner?")
    tick_labels = [str(value).replace("_", " ") for value in horizon.score_name]
    for axis in axes:
        axis.set_xticks(np.arange(len(horizon)), tick_labels, rotation=25, ha="right")
    fig.suptitle("Two-model lower-uncertainty aggregation by horizon")
    fig.tight_layout()
    path = output / "aggregate_selector_delta_and_auc.png"
    fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    chunks = overall[overall.signal_kind == "individual_chunk"].sort_values("score_order")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", lw=1)
    ax.plot(np.arange(len(chunks)), chunks.fixed_delta_vs_lower_u_pp,
            marker="o", label="fixed U >= 0.03")
    ax.fill_between(np.arange(len(chunks)), chunks.fixed_delta_ci_low_pp,
                    chunks.fixed_delta_ci_high_pp, alpha=.18)
    ax.set_xticks(np.arange(len(chunks)), chunks.score_name.str.replace("chunk_", ""))
    ax.set(xlabel="Chunk index (separate completed trajectories after chunk 0)",
           ylabel="SR change vs lower-U baseline (pp)",
           title="Where does uncertainty-gated refinement appear useful?")
    fig.tight_layout(); path = output / "selective_refinement_by_chunk.png"
    fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    by_suite = tables["selective_refinement_fixed_by_suite"]
    first = by_suite[by_suite.score_name == "first_chunk"].sort_values("suite")
    labels = first.suite.str.replace("libero_", "", regex=False)
    x, width = np.arange(len(first)), .38
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(x - width / 2, 100 * first.lower_u_baseline_sr, width,
           label="lower-U baseline", color="#4C78A8")
    ax.bar(x + width / 2, 100 * first.fixed_threshold_sr, width,
           label="first-chunk gate U >= 0.03", color="#54A24B")
    for index, delta in enumerate(first.fixed_delta_vs_lower_u_pp):
        ax.text(index + width / 2, 100 * first.fixed_threshold_sr.iloc[index] + 1,
                f"{delta:+.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set(ylabel="Success rate (%)", ylim=(0, 105),
           title="Deployable initial-observation selective refinement by suite")
    ax.legend(fontsize=8); fig.tight_layout()
    path = output / "selective_refinement_first_chunk_by_suite.png"
    fig.savefig(path, dpi=180); plt.close(fig); written.append(path)

    sweep = tables["selective_refinement_window_sweep"]
    for score_name in ("first_chunk", "full_episode"):
        group = sweep[sweep.score_name == score_name]
        delta = group.pivot(index="lower", columns="upper", values="delta_pp").sort_index()
        count = group.pivot(index="lower", columns="upper", values="n_refined").reindex_like(delta)
        extent = [float(delta.columns.min()), float(delta.columns.max()),
                  float(delta.index.min()), float(delta.index.max())]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        image = axes[0].imshow(delta.to_numpy(), origin="lower", aspect="auto",
                               cmap="RdYlGn", extent=extent)
        sample = axes[1].imshow(count.to_numpy(), origin="lower", aspect="auto",
                                cmap="Blues", extent=extent)
        fig.colorbar(image, ax=axes[0], label="In-sample policy delta SR (pp)")
        fig.colorbar(sample, ax=axes[1], label="Episodes refined")
        for axis in axes:
            axis.set(xlabel="Upper uncertainty", ylabel="Lower uncertainty")
        axes[0].set_title("Exploratory SR change")
        axes[1].set_title("Window sample size")
        fig.suptitle(f"Selective-refinement window sweep: {score_name.replace('_', ' ')}")
        fig.tight_layout()
        path = output / f"selective_refinement_window_{score_name}.png"
        fig.savefig(path, dpi=180); plt.close(fig); written.append(path)
    if "source_checkpoint_by_suite" in tables:
        source = tables["source_checkpoint_by_suite"].sort_values("suite")
        labels = source.suite.str.replace("libero_", "", regex=False)
        x, width = np.arange(len(source)), .24
        fig, ax = plt.subplots(figsize=(15, 5.5))
        ax.bar(x - width, 100 * source.source_baseline_sr, width,
               label="shared source checkpoint", color="#9D9D9D")
        ax.bar(x, 100 * source.model0_baseline_sr, width,
               label="model 0", color="#4C78A8")
        ax.bar(x + width, 100 * source.model1_baseline_sr, width,
               label="model 1", color="#F58518")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set(ylabel="Unrefined success rate (%)", ylim=(0, 105),
               title="Bootstrapped members versus their shared source checkpoint")
        ax.legend(fontsize=8); fig.tight_layout()
        path = output / "members_vs_source_checkpoint_by_suite.png"
        fig.savefig(path, dpi=180); plt.close(fig); written.append(path)
    if "source_checkpoint_refinement_window_sweep" in tables:
        source_sweep = tables["source_checkpoint_refinement_window_sweep"]
        for score_name in ("first_chunk", "full_episode"):
            group = source_sweep[source_sweep.score_name == score_name]
            delta = group.pivot(
                index="lower", columns="upper", values="delta_pp").sort_index()
            count = group.pivot(
                index="lower", columns="upper", values="n_refined").reindex_like(delta)
            extent = [float(delta.columns.min()), float(delta.columns.max()),
                      float(delta.index.min()), float(delta.index.max())]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            image = axes[0].imshow(delta.to_numpy(), origin="lower", aspect="auto",
                                   cmap="RdYlGn", extent=extent)
            sample = axes[1].imshow(count.to_numpy(), origin="lower", aspect="auto",
                                    cmap="Blues", extent=extent)
            fig.colorbar(image, ax=axes[0], label="In-sample delta SR (pp)")
            fig.colorbar(sample, ax=axes[1], label="Episodes refined")
            for axis in axes:
                axis.set(xlabel="Upper uncertainty", ylabel="Lower uncertainty")
            axes[0].set_title("Exploratory selective-refinement effect")
            axes[1].set_title("Window sample size")
            fig.suptitle(
                "Shared source checkpoint: " + score_name.replace("_", " ") +
                " uncertainty window")
            fig.tight_layout()
            path = output / f"source_checkpoint_window_{score_name}.png"
            fig.savefig(path, dpi=180); plt.close(fig); written.append(path)
    return written
