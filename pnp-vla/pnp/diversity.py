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
DIVERSITY_PAIR_KEYS = ["suite", "task_idx", "episode_idx", "init_state_hash"]


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


def diversity_experiment(member_index: int) -> str:
    return f"{DIVERSITY_EXPERIMENT_PREFIX}-m{int(member_index)}"


def run_diversity_signal_worker(*, member_index: int, model_repo_id: str,
                                shard_count: int = 1, shard_index: int = 0,
                                episodes_per_task: int = 2,
                                manifest_hash: str = ""):
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
    experiment = diversity_experiment(member_index)
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=device, experiment=experiment, episodes=episodes,
        methods=[(Method.UNCERTAINTY, config)], cohort=f"diversity_m{member_index}",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="pi05_diversity_signal",
        run_metadata={"member_index": int(member_index), "model_repo_id": model_repo_id,
                      "bootstrap_manifest_hash": manifest_hash,
                      "model_revision": model_revision, "pnp_k": 5,
                      "pnp_steps": [3, 4], "episodes_per_task": episodes_per_task},
        provenance=gather_provenance(
            model_repo_id=model_repo_id, model_revision=model_revision),
    )


def _fetch_for_rollouts(store, table: str, rollout_ids: list[str], columns: str = "*") -> list[dict]:
    rows = []
    for start in range(0, len(rollout_ids), 100):
        batch = rollout_ids[start:start + 100]
        rows.extend(store.fetch_all(
            table, columns, configure=lambda query, ids=batch: query.in_("rollout_id", ids),
            order_by=("rollout_id",)))
    return rows


def fetch_diversity_signal(store) -> tuple[pd.DataFrame, pd.DataFrame]:
    rollout_frames, step_frames = [], []
    for member_index in (0, 1):
        experiment = diversity_experiment(member_index)
        rows = store.fetch_all(
            "rollouts", "*", configure=lambda query, exp=experiment: query.eq("experiment", exp),
            order_by=("rollout_id",))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"no rollouts found for {experiment}")
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
