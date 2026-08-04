"""Stable, package-owned rollout drivers for thin Colab worker launchers."""
from __future__ import annotations

import contextlib
import io
import time

from .config import Method, RolloutConfig
from .store import SupabaseStore


SCHEDULES = (
    (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (1, 3, 5, 7, 9), (3, 6, 9), (2, 5, 8),
)
K = 3
BASE_INFERENCE_STEPS = 10
OBSERVED_STEPS = tuple(sorted({step for schedule in SCHEDULES for step in schedule}))
FULL_ABLATION_TASKS = {
    ("libero_spatial", 5), ("libero_spatial", 8),
    ("libero_goal", 0), ("libero_goal", 1), ("libero_goal", 2),
    ("libero_goal", 3), ("libero_goal", 5), ("libero_goal", 6),
}
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPERIMENT = "libero-hybrid-schedules-k3-v1"
PRO_EXPERIMENT = "libero-pro-canonical-core-k3-v1"


def build_full_methods(schedules=SCHEDULES, k=K):
    """The 12-config refine-last matrix used on the historical hard-task cohort."""
    extra_steps = sorted({BASE_INFERENCE_STEPS + k * len(schedule) for schedule in schedules})
    methods = [
        (Method.UNCERTAINTY, RolloutConfig(
            pnp_steps=OBSERVED_STEPS, pnp_k=k, save_pcp_features=True)),
        *((Method.EXTRA_STEPS, RolloutConfig(num_inference_steps=steps))
          for steps in extra_steps),
    ]
    for schedule in schedules:
        probe = dict(pnp_steps=schedule, pnp_k=k)
        methods.append((Method.REFINEMENT, RolloutConfig(**probe, refine=True)))
    if len(methods) != 12:
        raise AssertionError(f"expected 12 full methods, got {len(methods)}")
    return methods


def build_broad_methods(full_methods=None):
    """Observed + 16-step control + established refine-last (4,5)."""
    full_methods = full_methods or build_full_methods()
    methods = [
        item for item in full_methods
        if (item[0] == Method.UNCERTAINTY
            or (item[0] == Method.EXTRA_STEPS and item[1].num_inference_steps == 16)
            or (item[0] == Method.REFINEMENT and item[1].pnp_steps == (4, 5)
                and not item[1].refine_average))
    ]
    if len(methods) != 3:
        raise AssertionError(f"expected 3 broad methods, got {len(methods)}")
    return methods


def build_pro_methods(k=K):
    """Canonical PRO core: observed/PCP + matched compute + established refinement."""
    methods = [
        (Method.UNCERTAINTY, RolloutConfig(
            pnp_steps=OBSERVED_STEPS, pnp_k=k, save_pcp_features=True)),
        (Method.EXTRA_STEPS, RolloutConfig(num_inference_steps=16)),
        (Method.REFINEMENT, RolloutConfig(
            pnp_steps=(4, 5), pnp_k=k, refine=True)),
    ]
    if len(methods) != 3:
        raise AssertionError(f"expected 3 PRO methods, got {len(methods)}")
    return methods


def identity_shard(episodes, shard_count: int, shard_index: int):
    """Return a stable, disjoint identity shard independent of input ordering."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    ordered = sorted(episodes, key=lambda ep: (
        ep["suite"], ep["task_idx"], ep.get("ep_idx", ep.get("episode_idx", 0)),
        ep.get("init_state_hash", ""),
    ))
    return ordered[shard_index::shard_count]


def _prepare_libero_episodes():
    from . import libero_env

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        benchmark_dict = libero_env.init_libero_benchmark()
        tasks = [
            (suite, task_idx)
            for suite in LIBERO_SUITES
            for task_idx in range(benchmark_dict[suite]().n_tasks)
        ]
        episodes = libero_env.build_final_episodes(benchmark_dict, tasks=tasks)
    if len(episodes) != 400:
        raise AssertionError(f"expected 400 LIBERO identities, got {len(episodes)}")
    return episodes


def _prepare_libero_pro_episodes():
    """Install the official canonical assets and build exactly 600 PRO identities."""
    from . import libero_pro

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        pro_dir = libero_pro.clone_libero_pro()
        libero_pro.install_assets(
            suites=libero_pro.CANONICAL_PRO_SUITES, pro_dir=pro_dir)
        libero_pro.apply_env_patches(pro_dir=pro_dir)
        libero_pro.patch_torch_load()
        benchmark_dict = libero_pro.reload_benchmark()
        episodes = libero_pro.build_libero_pro_episodes(
            benchmark_dict, suites=libero_pro.CANONICAL_PRO_SUITES,
            episode_idxs=range(10))
    identities = {
        (ep["suite"], ep["task_idx"], ep["ep_idx"], ep["init_state_hash"])
        for ep in episodes
    }
    if len(episodes) != 600 or len(identities) != 600:
        tail = captured.getvalue()[-2000:]
        raise AssertionError(
            f"expected 600 unique canonical PRO identities, got "
            f"{len(episodes)} rows/{len(identities)} identities\n{tail}")
    if not all(ep["canonical_member"] for ep in episodes):
        raise AssertionError("canonical PRO manifest contains a non-canonical identity")
    print("LIBERO-PRO canonical manifest: 600 identities x 3 configs = 1800 rollouts")
    return episodes


def _run_collection(*, store, policy, preprocess, postprocess, device, experiment, episodes,
                    methods, cohort, shard_count, shard_index,
                    benchmark="libero", driver="hybrid_schedules", run_metadata=None,
                    rollout_batch_size: int = 2):
    from tqdm.auto import tqdm
    from .libero_env import make_env
    from .rollout import run_episode_batch
    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be at least 1")

    expected = len(episodes) * len(methods)
    done = store.existing_keys(experiment)
    pending = sum(
        store.rollout_id(experiment, ep, name, cfg) not in done
        for ep in episodes for name, cfg in methods
    )
    print(f"{cohort}: {len(episodes)} identities x {len(methods)} configs; "
          f"{pending}/{expected} pending")
    refinement_schedules = [
        config.pnp_steps for name, config in methods if name == Method.REFINEMENT
    ]
    run_config = {"cohort": cohort, "schedules": refinement_schedules, "pnp_k": K,
                  "n_configs": len(methods), "shard_count": shard_count,
                  "shard_index": shard_index, "rollout_batch_size": rollout_batch_size}
    run_config.update(run_metadata or {})
    store.start_run(
        driver=driver, benchmark=benchmark, experiment=experiment,
        config=run_config,
    )
    completed = 0
    occupied_lanes = batch_slots = 0
    inference_ms_total = 0.0
    collection_t0 = time.time()
    try:
        with tqdm(total=pending, desc=f"{cohort}[{shard_index}]", unit="rollout",
                  dynamic_ncols=True) as progress:
            task_keys = sorted({(ep["suite"], ep["task_idx"]) for ep in episodes})
            for task_key in task_keys:
                task_episodes = [ep for ep in episodes
                                 if (ep["suite"], ep["task_idx"]) == task_key]
                envs = []
                try:
                    for _ in range(min(rollout_batch_size, len(task_episodes))):
                        envs.append(make_env(task_episodes[0]["bddl_path"]))
                    # A batch shares all sampler behavior; grouping by method/config also keeps
                    # resumption deterministic when only part of a task is already complete.
                    for name, cfg in methods:
                        todo = list(store.iter_todo(
                            experiment, task_episodes, [(name, cfg)], done=done))
                        for start in range(0, len(todo), rollout_batch_size):
                            group = todo[start:start + rollout_batch_size]
                            batch_t0 = time.perf_counter()
                            results = run_episode_batch(
                                envs[:len(group)], [item[0] for item in group], policy,
                                preprocess, postprocess, device, cfg)
                            inference_ms_total += (time.perf_counter() - batch_t0) * 1000.0
                            occupied_lanes += len(group); batch_slots += rollout_batch_size
                            for (ep, method, config, rid), result in zip(group, results):
                                store.log_result(rid, ep, method, config, result)
                                completed += 1; progress.update()
                                elapsed_h = max((time.time() - collection_t0) / 3600, 1e-9)
                                util = occupied_lanes / max(batch_slots, 1)
                                progress.set_postfix(success=int(result["success"]), method=method,
                                    batch_util=f"{util:.0%}", rollouts_h=f"{completed / elapsed_h:.1f}",
                                    inference_ms=f"{result['inference_ms_total']:.0f}", refresh=False)
                finally:
                    for env in envs:
                        env.close()
    except BaseException:
        store.finish_run(status="failed", n_rollouts=completed)
        raise
    store.finish_run(n_rollouts=completed)
    peak_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available(): peak_gb = torch.cuda.max_memory_allocated() / 2**30
    except ImportError:
        pass
    elapsed_h = max((time.time() - collection_t0) / 3600, 1e-9)
    print(f"{cohort}: logged {completed} new rollouts; batch utilization "
          f"{occupied_lanes / max(batch_slots, 1):.1%}; {completed / elapsed_h:.1f} rollouts/hour; "
          f"inference {inference_ms_total / 1000:.1f}s; peak CUDA {peak_gb:.2f} GiB")


def run_libero_hybrid_worker(*, shard_count: int, shard_index: int,
                             experiment: str = EXPERIMENT,
                             rollout_batch_size: int = 2):
    """Load pi0.5 once and execute one disjoint shard of the hybrid LIBERO plan."""
    from . import models

    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be at least 1")
    episodes = identity_shard(_prepare_libero_episodes(), shard_count, shard_index)
    ablation = [
        ep for ep in episodes if (ep["suite"], ep["task_idx"]) in FULL_ABLATION_TASKS
    ]
    broad = [
        ep for ep in episodes if (ep["suite"], ep["task_idx"]) not in FULL_ABLATION_TASKS
    ]
    full_methods = build_full_methods()
    broad_methods = build_broad_methods(full_methods)

    print(f"worker {shard_index}/{shard_count}: {len(ablation)} full-ablation + "
          f"{len(broad)} broad identities")
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    store = SupabaseStore()
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess, device=device,
        experiment=experiment, episodes=ablation, methods=full_methods,
        cohort="full_ablation", shard_count=shard_count, shard_index=shard_index,
        rollout_batch_size=rollout_batch_size,
    )
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess, device=device,
        experiment=experiment, episodes=broad, methods=broad_methods,
        cohort="broad_validation", shard_count=shard_count, shard_index=shard_index,
        rollout_batch_size=rollout_batch_size,
    )


def run_libero_pro_worker(*, shard_count: int, shard_index: int,
                          experiment: str = PRO_EXPERIMENT,
                          rollout_batch_size: int = 2):
    """Run one of six disjoint shards of the 600-identity canonical PRO core plan."""
    from . import libero_pro, models

    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be at least 1")
    episodes = identity_shard(_prepare_libero_pro_episodes(), shard_count, shard_index)
    methods = build_pro_methods()
    expected = len(episodes) * len(methods)
    print(f"PRO worker {shard_index}/{shard_count}: {len(episodes)} identities x "
          f"{len(methods)} configs = {expected} rollouts")
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    store = SupabaseStore()
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess, device=device,
        experiment=experiment, episodes=episodes, methods=methods,
        cohort="canonical", shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="canonical_pro_core",
        run_metadata={"libero_pro_revision": libero_pro.LIBERO_PRO_REVISION},
        rollout_batch_size=rollout_batch_size,
    )
