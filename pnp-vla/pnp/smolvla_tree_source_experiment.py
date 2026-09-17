"""Rich SmolVLA source collection for fixed-manifest tree harvesting."""
from __future__ import annotations

import contextlib
import io

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    LIBERO_SUITES,
    _run_collection,
    identity_shard,
)
from .five_step_diversity_experiment import (
    identity_manifest_hash,
    identity_manifest_payload,
)
from .smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_TREE_SOURCE_EXPERIMENT = "smolvla-libero-tree-source-idx10-29-v1"
SMOLVLA_TREE_SOURCE_EPISODE_INDICES = tuple(range(10, 30))
SMOLVLA_TREE_SOURCE_SHARDS = 2
SMOLVLA_TREE_SOURCE_IDENTITIES = 800
SMOLVLA_TREE_PRIORITY_FRACTION = 0.65
SMOLVLA_TREE_UNIFORM_FRACTION = 0.35


def build_smolvla_tree_source_method():
    """Deployment P&P plus every source artifact needed by later tree/training phases."""
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=True,
        n_action_steps=LIBERO_ACTION_STEPS,
        save_time_uncertainty=True,
        save_trajectory=True,
        save_generated_chunks=True,
        save_training_data=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    return Method.SMOLVLA_PNP_S123_K311, config


def prepare_smolvla_tree_source_episodes():
    """Build the exact, eval-disjoint standard-LIBERO source identity manifest."""
    from . import libero_env

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        benchmark_dict = libero_env.init_libero_benchmark()
        tasks = [
            (suite, task_idx)
            for suite in LIBERO_SUITES
            for task_idx in range(benchmark_dict[suite]().n_tasks)
        ]
        episodes = libero_env.build_final_episodes(
            benchmark_dict,
            episode_idxs=SMOLVLA_TREE_SOURCE_EPISODE_INDICES,
            tasks=tasks,
        )
    identities = {
        (ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"]), ep["init_state_hash"])
        for ep in episodes
    }
    if len(episodes) != SMOLVLA_TREE_SOURCE_IDENTITIES or len(identities) != len(episodes):
        tail = captured.getvalue()[-2000:]
        raise AssertionError(
            f"expected {SMOLVLA_TREE_SOURCE_IDENTITIES} unique indices-10:29 identities, "
            f"found {len(episodes)} rows/{len(identities)} identities\n{tail}")
    observed_indices = {int(ep["ep_idx"]) for ep in episodes}
    if observed_indices != set(SMOLVLA_TREE_SOURCE_EPISODE_INDICES):
        raise AssertionError(f"source manifest has unexpected indices: {sorted(observed_indices)}")
    return episodes


def run_smolvla_tree_source_worker(
        *, shard_count: int = SMOLVLA_TREE_SOURCE_SHARDS, shard_index: int = 0,
        rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_TREE_SOURCE_EXPERIMENT):
    """Collect one of two fixed shards; completed rows are resume-safe."""
    from . import models

    if shard_count != SMOLVLA_TREE_SOURCE_SHARDS:
        raise ValueError(f"source collection is frozen to {SMOLVLA_TREE_SOURCE_SHARDS} shards")
    if shard_index not in range(shard_count):
        raise ValueError(f"invalid source shard {shard_index}/{shard_count}")
    all_episodes = prepare_smolvla_tree_source_episodes()
    episodes = identity_shard(all_episodes, shard_count, shard_index)
    if len(episodes) != SMOLVLA_TREE_SOURCE_IDENTITIES // shard_count:
        raise AssertionError(f"unexpected shard size {len(episodes)}")
    manifest_hash = identity_manifest_hash(all_episodes)
    method, config = build_smolvla_tree_source_method()

    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "source_indices": [10, 29],
        "eval_indices_reserved": [0, 9],
        "full_manifest_identities": len(all_episodes),
        "identities_in_this_shard": len(episodes),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "manifest_hash": manifest_hash,
        "policy": "always-on P&P steps=(1,2,3), K=(3,1,1)",
        "integration_steps": 10,
        "n_action_steps": LIBERO_ACTION_STEPS,
        "generated_chunk_size": 50,
        "uncertainty": "step-specific and (3,1,1)-weighted U10/U20/U50",
        "future_root_manifest_mix": "65% uncertainty-prioritized / 35% uniform",
        "rich_source_artifact": (
            "actions, rewards, exact simulator states, boundary images/proprio, "
            "generated chunks, seeds, and SmolVLA prefix tensors"),
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: per-suite and overall running SR every 10 completed identities.")

    policy, preprocess, postprocess = models.load_smolvla()
    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    store = SupabaseStore()
    _run_collection(
        store=store,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=models.default_device(),
        experiment=experiment,
        episodes=episodes,
        methods=[(method, config)],
        cohort="smolvla_tree_source_idx10_29",
        shard_count=shard_count,
        shard_index=shard_index,
        benchmark="libero",
        driver="smolvla_tree_source_collection",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(all_episodes),
            "episode_indices": list(SMOLVLA_TREE_SOURCE_EPISODE_INDICES),
            "frozen_identity_manifest_hash": manifest_hash,
            "frozen_identity_manifest": identity_manifest_payload(all_episodes),
            "integration_steps": 10,
            "n_action_steps": LIBERO_ACTION_STEPS,
            "generated_chunk_size": 50,
            "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "uncertainty_horizons": [10, 20, 50],
            "future_priority_fraction": SMOLVLA_TREE_PRIORITY_FRACTION,
            "future_uniform_fraction": SMOLVLA_TREE_UNIFORM_FRACTION,
            "source_artifact_schema": "pcp-search-v1 rich boundary artifact",
            "video": "off",
        },
        report_every=0,
        report_every_identities=10,
        historical_sr=False,
        progress_include_overall=True,
        progress_count_label="identities",
        rollout_batch_size=rollout_batch_size,
        provenance=provenance,
        resume_completed_only=True,
    )
