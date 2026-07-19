"""Stable, package-owned rollout drivers for thin Colab worker launchers."""
from __future__ import annotations

import contextlib
import io

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


def _run_collection(*, store, policy, preprocess, postprocess, device, experiment, episodes,
                    methods, cohort, shard_count, shard_index):
    from tqdm.auto import tqdm
    from .rollout import iter_task_envs, run_episode

    expected = len(episodes) * len(methods)
    done = store.existing_keys(experiment)
    pending = sum(
        store.rollout_id(experiment, ep, name, cfg) not in done
        for ep in episodes for name, cfg in methods
    )
    print(f"{cohort}: {len(episodes)} identities x {len(methods)} configs; "
          f"{pending}/{expected} pending")
    store.start_run(
        driver="hybrid_schedules", benchmark="libero", experiment=experiment,
        config={"cohort": cohort, "schedules": SCHEDULES, "pnp_k": K,
                "n_configs": len(methods), "shard_count": shard_count,
                "shard_index": shard_index},
    )
    completed = 0
    try:
        with tqdm(total=pending, desc=f"{cohort}[{shard_index}]", unit="rollout",
                  dynamic_ncols=True) as progress:
            for env, task_episodes in iter_task_envs(episodes):
                for ep, name, cfg, rid in store.iter_todo(
                        experiment, task_episodes, methods, done=done):
                    result = run_episode(
                        env, ep, policy, preprocess, postprocess, device, cfg)
                    store.log_result(rid, ep, name, cfg, result)
                    completed += 1
                    progress.update()
                    progress.set_postfix(
                        success=int(result["success"]), method=name, refresh=False)
    except BaseException:
        store.finish_run(status="failed", n_rollouts=completed)
        raise
    store.finish_run(n_rollouts=completed)
    print(f"{cohort}: logged {completed} new rollouts")


def run_libero_hybrid_worker(*, shard_count: int, shard_index: int,
                             experiment: str = EXPERIMENT):
    """Load pi0.5 once and execute one disjoint shard of the hybrid LIBERO plan."""
    from . import models

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
    )
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess, device=device,
        experiment=experiment, episodes=broad, methods=broad_methods,
        cohort="broad_validation", shard_count=shard_count, shard_index=shard_index,
    )
