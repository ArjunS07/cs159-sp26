"""Package-owned Colab worker for PCP-search training-data collection."""
from __future__ import annotations

import contextlib
import io
from typing import Iterable, Literal

from ..config import Method, RolloutConfig
from ..store import SupabaseStore, gather_provenance
from .data import validate_training_artifact
from .manifest import ManifestItem, RolloutManifest
from .registry import ManifestRegistry


EXPERIMENT = "pcp-search-rollouts-v1"


def build_collection_config(manifest: RolloutManifest) -> RolloutConfig:
    expected = manifest.collection_config
    if expected.get("n_action_steps") != 10 or expected.get("refine") is not False:
        raise ValueError("manifest is not a vanilla 10-action PCP-search collection")
    config = RolloutConfig(
        pnp_steps=tuple(expected.get("pnp_steps", (3, 4))),
        pnp_k=int(expected.get("pnp_k", 5)),
        num_inference_steps=int(expected.get("num_inference_steps", 10)),
        n_action_steps=10,
        save_uncertainty=True,
        save_pcp_features=True,
        save_ahats=True,
        save_time_uncertainty=True,
        save_trajectory=True,
        save_generated_chunks=True,
        save_training_data=True,
        # Terminal boundaries arrive unexpectedly on success, so cameras stay live. This is
        # slower but guarantees the terminal raw/model observations are fresh.
        skip_unused_renders=False,
        video="off",
    )
    if config.refine or config.correction_lambda is not None or config.num_samples is not None:
        raise AssertionError("PCP-search collection must execute exact stock policy actions")
    return config


def manifest_shard(items: Iterable[ManifestItem], shard_count: int,
                   shard_index: int) -> list[ManifestItem]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    return [item for item in sorted(items, key=lambda value: value.ordinal)
            if item.ordinal % shard_count == shard_index]


BatchStrategy = Literal["mixed_task", "same_task"]


def rollout_batches(items: Iterable[ManifestItem], batch_size: int, *,
                    strategy: BatchStrategy = "mixed_task") -> list[list[ManifestItem]]:
    """Return deterministic batches for independent-env VLA rollout collection.

    ``run_episode_batch`` owns a separate MuJoCo environment per lane and only
    stacks PI05 tensor inputs, so BDDLs need not match.  The older same-task
    scheduler was safe but left most of a nominal batch of 16 empty: the PRO
    manifest has only 6--10 states per task.  ``mixed_task`` fills each VLA
    call across tasks while preserving every item's rollout seed and identity.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ordered = sorted(items, key=lambda value: value.ordinal)
    if strategy == "mixed_task":
        return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
    if strategy != "same_task":
        raise ValueError(f"unknown PCP-search batch strategy {strategy!r}")
    grouped: dict[tuple[str, int], list[ManifestItem]] = {}
    for item in ordered:
        grouped.setdefault(item.task_key, []).append(item)
    return [task_items[start:start + batch_size]
            for task_key in sorted(grouped)
            for task_items in (grouped[task_key],)
            for start in range(0, len(task_items), batch_size)]


def resolve_manifest_episodes(manifest: RolloutManifest) -> dict[int, dict]:
    """Resolve index-only membership against the benchmark pinned by the manifest."""
    benchmark = manifest.collection_config.get("benchmark", "libero")
    if benchmark == "libero_pro":
        return _resolve_libero_pro_manifest_episodes(manifest)
    if benchmark != "libero":
        raise ValueError(f"unsupported PCP-search benchmark {benchmark!r}")
    from .. import libero_env

    tasks = sorted({item.task_key for item in manifest.items})
    state_indices = sorted({item.init_state_index for item in manifest.items})
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        benchmark_dict = libero_env.init_libero_benchmark()
        episodes = libero_env.build_final_episodes(
            benchmark_dict, episode_idxs=state_indices, tasks=tasks)
    by_identity = {
        (episode["suite"], episode["task_idx"], episode["ep_idx"]): episode
        for episode in episodes
    }
    resolved = {}
    for item in manifest.items:
        key = (item.suite, item.task_idx, item.init_state_index)
        if key not in by_identity:
            tail = captured.getvalue()[-2000:]
            raise ValueError(f"manifest state {key} is unavailable in pinned LIBERO\n{tail}")
        episode = dict(by_identity[key])
        episode.update(
            behavior_seed_index=item.behavior_seed_index,
            manifest_id=manifest.manifest_id,
            manifest_ordinal=item.ordinal,
            pcp_partition_id=manifest.collection_config.get("partition_id", "standard_libero"),
            pcp_data_split=manifest.collection_config.get("data_split", "train"),
            pcp_train_eligible=bool(manifest.collection_config.get("train_eligible", True)),
        )
        resolved[item.ordinal] = episode
    return resolved


def _resolve_libero_pro_manifest_episodes(manifest: RolloutManifest) -> dict[int, dict]:
    """Install the pinned PRO assets once, then resolve manifest indices exactly.

    LIBERO-PRO modifies the installed benchmark package, so this setup intentionally lives in
    the package worker rather than a notebook cell.  The notebook remains a thin launcher.
    """
    from .. import libero_pro

    suites = sorted({item.suite for item in manifest.items})
    state_indices = sorted({item.init_state_index for item in manifest.items})
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        pro_dir = libero_pro.clone_libero_pro()
        libero_pro.install_assets(suites=suites, pro_dir=pro_dir)
        libero_pro.apply_env_patches(pro_dir=pro_dir)
        libero_pro.patch_torch_load()
        benchmark_dict = libero_pro.reload_benchmark()
        episodes = libero_pro.build_libero_pro_episodes(
            benchmark_dict, suites=suites, episode_idxs=state_indices)
    by_identity = {
        (episode["suite"], episode["task_idx"], episode["ep_idx"]): episode
        for episode in episodes
    }
    resolved = {}
    for item in manifest.items:
        key = (item.suite, item.task_idx, item.init_state_index)
        if key not in by_identity:
            tail = captured.getvalue()[-2000:]
            raise ValueError(f"manifest PRO state {key} is unavailable in pinned LIBERO-PRO\n{tail}")
        episode = dict(by_identity[key])
        episode.update(
            behavior_seed_index=item.behavior_seed_index,
            manifest_id=manifest.manifest_id,
            manifest_ordinal=item.ordinal,
            pcp_partition_id=manifest.collection_config["partition_id"],
            pcp_data_split=manifest.collection_config["data_split"],
            pcp_train_eligible=bool(manifest.collection_config["train_eligible"]),
        )
        resolved[item.ordinal] = episode
    print(captured.getvalue()[-4000:])
    return resolved


def run_pcp_search_worker(*, manifest_id: str, shard_count: int = 4,
                          shard_index: int = 0,
                          rollout_batch_size: int = 24,
                          batch_strategy: BatchStrategy = "mixed_task",
                          experiment: str = EXPERIMENT) -> None:
    """Collect one resumable shard of a frozen PCP-search manifest.

    This is the only call a GPU notebook needs. All querying, asset resolution, model pinning,
    transition capture, validation, upload, and resume state live in the package.
    """
    from tqdm.auto import tqdm
    from .. import models
    from ..libero_env import make_env
    from ..rollout import run_episode_batch

    # Validate before any model/asset load, so a notebook typo costs no GPU time.
    rollout_batches([], rollout_batch_size, strategy=batch_strategy)

    store = SupabaseStore()
    registry = ManifestRegistry(store)
    manifest = registry.load(manifest_id)
    if manifest.manifest_id != manifest_id:
        raise ValueError("loaded manifest ID does not match requested content")
    if manifest.collection_config.get("benchmark") == "libero_pro":
        from .pro import validate_pro_manifest
        validate_pro_manifest(manifest)
    items = manifest_shard(manifest.items, shard_count, shard_index)
    episodes = resolve_manifest_episodes(manifest)
    config = build_collection_config(manifest)
    completed_ordinals = registry.completed_ordinals(manifest_id)
    # Recover cleanly from the narrow crash window between the rollout upsert and membership
    # status upsert. Never spend another GPU rollout when the validated artifact already exists.
    ready_rows = store.fetch_all(
        "rollouts", "rollout_id,training_validation_json",
        configure=lambda query: query.eq("experiment", experiment).eq("training_ready", True),
        order_by=("rollout_id",))
    ready_by_id = {row["rollout_id"]: row for row in ready_rows}
    for item in items:
        if item.ordinal in completed_ordinals:
            continue
        rollout_id = store.rollout_id(
            experiment, episodes[item.ordinal], Method.PCP_SEARCH_COLLECT, config)
        if rollout_id in ready_by_id:
            registry.record_result(
                manifest_id, item.ordinal, rollout_id, status="training_ready",
                reason="recovered_existing_validated_rollout",
                validation=ready_by_id[rollout_id].get("training_validation_json") or {})
            completed_ordinals.add(item.ordinal)
    pending = [item for item in items if item.ordinal not in completed_ordinals]
    print(f"{manifest_id} worker {shard_index}/{shard_count}: "
          f"{len(pending)}/{len(items)} pending")
    if not pending:
        return

    policy, preprocess, postprocess = models.load_pi05(
        repo_id=manifest.policy_repo_id, revision=manifest.policy_revision)
    device = models.default_device()
    provenance = gather_provenance(
        model_repo_id=manifest.policy_repo_id, model_revision=manifest.policy_revision)
    benchmark = manifest.collection_config.get("benchmark", "libero")
    store.start_run(
        driver="pcp_search_collection", benchmark=benchmark, experiment=experiment,
        provenance=provenance,
        config={
            "manifest_id": manifest_id,
            "manifest_schema_version": manifest.schema_version,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "rollout_batch_size": rollout_batch_size,
            "batch_strategy": batch_strategy,
            "n_action_steps": 10,
            "generated_chunk_size": manifest.collection_config["generated_chunk_size"],
            "pnp_steps": list(config.pnp_steps),
            "pnp_k": config.pnp_k,
            "save_training_data": True,
            "partition_id": manifest.collection_config.get("partition_id", "standard_libero"),
            "data_split": manifest.collection_config.get("data_split", "train"),
        })
    logged = 0
    try:
        with tqdm(total=len(pending), desc=f"pcp-search[{shard_index}]", unit="rollout",
                  dynamic_ncols=True) as progress:
            for item_batch in rollout_batches(pending, rollout_batch_size, strategy=batch_strategy):
                episode_batch = [episodes[item.ordinal] for item in item_batch]
                envs = [make_env(episode["bddl_path"]) for episode in episode_batch]
                try:
                    results = run_episode_batch(
                        envs, episode_batch, policy, preprocess, postprocess, device, config)
                    for item, episode, result in zip(item_batch, episode_batch, results):
                        rollout_id = store.rollout_id(
                            experiment, episode, Method.PCP_SEARCH_COLLECT, config)
                        store.log_result(
                            rollout_id, episode, Method.PCP_SEARCH_COLLECT, config, result)
                        if result.get("status") == "completed" and result.get("training_data"):
                            validation = validate_training_artifact(result["training_data"])
                            registry.record_result(
                                manifest_id, item.ordinal, rollout_id,
                                status="training_ready", validation=validation)
                        else:
                            registry.record_result(
                                manifest_id, item.ordinal, rollout_id, status="errored",
                                reason=result.get("error_msg") or "missing training artifact")
                        logged += 1
                        progress.update()
                        progress.set_postfix_str(
                            f"{item.suite.removeprefix('libero_')}/{item.task_idx} "
                            f"success={int(bool(result.get('success')))}", refresh=False)
                finally:
                    for env in envs:
                        env.close()
    except BaseException:
        store.finish_run(status="failed", n_rollouts=logged)
        raise
    store.finish_run(status="completed", n_rollouts=logged)
    print(f"{manifest_id} worker {shard_index}/{shard_count}: logged {logged} new rollouts")
