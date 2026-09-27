"""Disjoint SmolVLA P&P source rollouts for a second fresh8 tree cohort."""
from __future__ import annotations

import contextlib
import io

from .config import SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS, LIBERO_RENDER_LEAD, LIBERO_SKIP_RENDERS,
    LIBERO_SUITES, _run_collection, identity_shard,
)
from .five_step_diversity_experiment import identity_manifest_hash, identity_manifest_payload
from .smolvla_tree_source_experiment import build_smolvla_tree_source_method
from .store import SupabaseStore, gather_provenance


SOURCE_EXPERIMENT = "smolvla-libero-tree-source-idx30-49-v1"
EPISODE_INDICES = tuple(range(30, 50))
SOURCE_IDENTITIES = 800
SOURCE_SHARDS = 2


def prepare_scaling_source_episodes():
    """Fail before running if any of the 40 tasks lacks indices 30--49."""
    from . import libero_env

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        benchmark_dict = libero_env.init_libero_benchmark()
        tasks = [(suite, task_idx) for suite in LIBERO_SUITES
                 for task_idx in range(benchmark_dict[suite]().n_tasks)]
        episodes = libero_env.build_final_episodes(
            benchmark_dict, episode_idxs=EPISODE_INDICES, tasks=tasks)
    keys = {(ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"]), ep["init_state_hash"])
            for ep in episodes}
    expected = {(suite, task_idx, index) for suite, task_idx in tasks
                for index in EPISODE_INDICES}
    found = {(suite, task_idx, index) for suite, task_idx, index, _ in keys}
    if (len(tasks) != 40 or len(episodes) != SOURCE_IDENTITIES
            or len(keys) != len(episodes) or found != expected):
        raise ValueError(
            f"new source cohort is incomplete: {len(episodes)}/{SOURCE_IDENTITIES} "
            f"identities, missing={sorted(expected - found)[:5]}; "
            f"LIBERO output={captured.getvalue()[-500:]}")
    return episodes


def run_scaling_source_worker(*, shard_index: int,
                              episode_limit: int | None = 1,
                              rollout_batch_size: int = 8,
                              store=None):
    """Resume-safe source collection; use indices 30--49 only."""
    from . import models

    if shard_index not in range(SOURCE_SHARDS):
        raise ValueError("source workers are fixed to shards 0 and 1")
    if episode_limit is not None and episode_limit < 1:
        raise ValueError("episode_limit must be positive or None")
    all_episodes = prepare_scaling_source_episodes()
    episodes = identity_shard(all_episodes, SOURCE_SHARDS, shard_index)
    if len(episodes) != SOURCE_IDENTITIES // SOURCE_SHARDS:
        raise ValueError("source shard has wrong size")
    if episode_limit is not None:
        episodes = episodes[:episode_limit]
    manifest_hash = identity_manifest_hash(all_episodes)
    method, config = build_smolvla_tree_source_method()
    if (config.n_action_steps != LIBERO_ACTION_STEPS
            or config.render_lead != LIBERO_RENDER_LEAD
            or config.skip_unused_renders != LIBERO_SKIP_RENDERS):
        raise RuntimeError("scaling source must match the original source policy")
    print({"experiment": SOURCE_EXPERIMENT,
           "shard": f"{shard_index}/{SOURCE_SHARDS}",
           "episodes_this_call": len(episodes),
           "full_cohort": len(all_episodes),
           "indices": [30, 49],
           "source_policy": method,
           "manifest_hash": manifest_hash,
           "episode_limit": episode_limit}, flush=True)
    policy, preprocess, postprocess = models.load_smolvla()
    store = store or SupabaseStore()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=SOURCE_EXPERIMENT,
        episodes=episodes, methods=[(method, config)],
        cohort="smolvla_tree_source_idx30_49_v1",
        shard_count=SOURCE_SHARDS, shard_index=shard_index,
        benchmark="libero", driver="smolvla_scaling_source_collection",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(all_episodes),
            "episode_indices": list(EPISODE_INDICES),
            "frozen_identity_manifest_hash": manifest_hash,
            "frozen_identity_manifest": identity_manifest_payload(all_episodes),
            "integration_steps": 10,
            "n_action_steps": LIBERO_ACTION_STEPS,
            "generated_chunk_size": 50,
            "pnp_steps": list(config.pnp_steps),
            "pnp_k_by_step": list(config.pnp_k_by_step),
            "future_priority_fraction": .65,
            "future_uniform_fraction": .35,
            "source_artifact_schema": "pcp-search-v1 rich boundary artifact",
            "collection_revision": "v2-terminal-camera-refresh",
            "video": "off",
        },
        report_every=0, report_every_identities=10,
        historical_sr=False, progress_include_overall=True,
        progress_count_label="identities", rollout_batch_size=rollout_batch_size,
        provenance=provenance, resume_completed_only=True,
    )
    return {"experiment": SOURCE_EXPERIMENT, "shard": shard_index,
            "requested_this_call": len(episodes), "full_shard": SOURCE_IDENTITIES // SOURCE_SHARDS}
