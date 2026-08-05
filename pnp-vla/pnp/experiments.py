"""Stable, package-owned rollout drivers for thin Colab worker launchers."""
from __future__ import annotations

import collections
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
PRO_EXPERIMENT = "libero-pro-canonical-core-k3-v1"

# ── Expanded 16-suite PRO run ────────────────────────────────────────────────
# A single schedule at a larger K, over every perturbation family, to test whether the
# per-iteration disagreement DECAY across the K perturbations predicts correctability. The
# decay signal lives in pnp_action_vectors.u_iter / u_iter_vec, recorded unconditionally by
# run_probe -- no a_hats blobs needed (112 bytes/step instead of 7 KB).
PRO_EXPANDED_EXPERIMENT = "pro-16suite-k5-steps34-v1"
PRO_EXPANDED_K = 5
PRO_EXPANDED_STEPS = (3, 4)
PRO_EXPANDED_EPISODES = 20          # episodes/task; short suites contribute what they ship
# The matched-compute control is ~28% of wall time and carries no telemetry, so it is deferred.
# Flip to True and re-run the same workers to fill it in: it is a distinct config_hash, so
# iter_todo requests only the missing control rollouts and skips everything already collected.
PRO_EXPANDED_INCLUDE_CONTROL = False


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


def build_pro_expanded_methods(k=PRO_EXPANDED_K, steps=PRO_EXPANDED_STEPS,
                               include_control=PRO_EXPANDED_INCLUDE_CONTROL):
    """Expanded PRO arms: observed no-op, refine-last, and optionally matched compute.

    Every arm spends `10 + k*len(steps)` velocity-field evaluations per chunk, so the control is
    honest at this K rather than reused from the K=3 matrix. It is optional because it carries no
    telemetry: dropping it costs only the compute confound defence, and it can be back-filled
    later under the same experiment label without recollecting anything (its config_hash is
    distinct, so iter_todo asks for exactly the missing rollouts).
    """
    probe = dict(pnp_steps=tuple(steps), pnp_k=k)
    control_steps = BASE_INFERENCE_STEPS + k * len(steps)
    methods = [(Method.UNCERTAINTY, RolloutConfig(**probe, save_pcp_features=True))]
    if include_control:
        methods.append(
            (Method.EXTRA_STEPS, RolloutConfig(num_inference_steps=control_steps)))
    methods.append((Method.REFINEMENT, RolloutConfig(**probe, refine=True)))
    expected = 3 if include_control else 2
    if len(methods) != expected:
        raise AssertionError(f"expected {expected} expanded PRO methods, got {len(methods)}")
    if any(config.refine_average for _, config in methods):
        raise AssertionError("refine-average is out of scope for the expanded PRO run")
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


def _prepare_libero_pro_expanded_episodes(episodes_per_task=PRO_EXPANDED_EPISODES):
    """Install all 16 expanded-cohort PRO suites and build their episode manifest.

    Unlike the canonical path this asserts no fixed identity count: `_with_milk` suites ship 10
    init states per task while the rest ship more, so the total depends on the installed assets.
    The count is printed and returned instead of hard-coded (see ROLLOUT_PLAN.md).
    """
    from . import libero_pro

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            pro_dir = libero_pro.clone_libero_pro()
            libero_pro.install_assets(
                suites=libero_pro.EXPANDED_PRO_SUITES, pro_dir=pro_dir)
            libero_pro.apply_env_patches(pro_dir=pro_dir)
            libero_pro.patch_torch_load()
            benchmark_dict = libero_pro.reload_benchmark()
            episodes = libero_pro.build_libero_pro_episodes(
                benchmark_dict, suites=libero_pro.EXPANDED_PRO_SUITES,
                episode_idxs=range(episodes_per_task))
    finally:
        # Print even on failure: the states/task table and the alias/asset warnings are exactly
        # what diagnose a bad install, and they are worthless if an exception swallows them.
        print(captured.getvalue()[-4000:])

    identities = {
        (ep["suite"], ep["task_idx"], ep["ep_idx"], ep["init_state_hash"])
        for ep in episodes
    }
    if len(identities) != len(episodes):
        raise AssertionError(
            f"expanded PRO manifest has duplicate identities "
            f"({len(episodes)} rows / {len(identities)} identities)")
    suites = sorted({ep["suite"] for ep in episodes})
    if len(suites) != len(libero_pro.EXPANDED_PRO_SUITES):
        missing = sorted(set(libero_pro.EXPANDED_PRO_SUITES) - set(suites))
        raise AssertionError(f"expanded PRO manifest is missing suites: {missing}")
    if not all(ep["expanded_member"] for ep in episodes):
        raise AssertionError("expanded PRO manifest contains a non-expanded identity")

    per_suite = collections.Counter(ep["suite"] for ep in episodes)
    print(f'{"suite":<36}{"episodes":>9}{"max_steps":>11}')
    for suite in libero_pro.EXPANDED_PRO_SUITES:
        steps = sorted({ep["max_steps"] for ep in episodes if ep["suite"] == suite})
        print(f"{suite:<36}{per_suite[suite]:>9}{str(steps[0] if steps else '-'):>11}")
    print(f"LIBERO-PRO expanded manifest: {len(episodes)} identities x 3 configs = "
          f"{len(episodes) * 3} rollouts")
    return episodes


def _run_collection(*, store, policy, preprocess, postprocess, device, experiment, episodes,
                    methods, cohort, shard_count, shard_index,
                    benchmark="libero", driver="hybrid_schedules", run_metadata=None):
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
    refinement_schedules = [
        config.pnp_steps for name, config in methods if name == Method.REFINEMENT
    ]
    run_config = {"cohort": cohort, "schedules": refinement_schedules, "pnp_k": K,
                  "n_configs": len(methods), "shard_count": shard_count,
                  "shard_index": shard_index}
    run_config.update(run_metadata or {})
    store.start_run(
        driver=driver, benchmark=benchmark, experiment=experiment,
        config=run_config,
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


def run_libero_pro_worker(*, shard_count: int, shard_index: int,
                          experiment: str = PRO_EXPERIMENT):
    """Run one of six disjoint shards of the 600-identity canonical PRO core plan."""
    from . import libero_pro, models

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
    )


def run_libero_pro_expanded_worker(*, shard_count: int, shard_index: int,
                                   experiment: str = PRO_EXPANDED_EXPERIMENT,
                                   episodes_per_task: int = PRO_EXPANDED_EPISODES,
                                   include_control: bool = PRO_EXPANDED_INCLUDE_CONTROL):
    """Run one shard of the expanded 16-suite PRO plan at K=5, refine-last (3,4).

    Apply supabase/migrations/004_u_iter.sql FIRST. The per-iteration disagreement this run
    exists to measure goes to pnp_action_vectors.u_iter / u_iter_vec; without those columns the
    first rollout's insert fails with an unknown-column error and the worker stops. It fails
    loudly rather than dropping the signal, but it fails after the rollouts row is already
    upserted -- so that identity would then be skipped as done. Delete the experiment's rows
    before retrying.

    Resumable: rollout ids are a pure function of (experiment, identity, logical config), so a
    disconnected worker restarted with the same shard_index skips completed work. Toggling
    include_control changes only WHICH configs are requested, never the ids of the others, so the
    control can be deferred now and back-filled later without recollecting anything.
    """
    from . import libero_pro, models

    episodes = identity_shard(
        _prepare_libero_pro_expanded_episodes(episodes_per_task), shard_count, shard_index)
    methods = build_pro_expanded_methods(include_control=include_control)
    expected = len(episodes) * len(methods)
    print(f"PRO expanded worker {shard_index}/{shard_count}: {len(episodes)} identities x "
          f"{len(methods)} configs = {expected} rollouts "
          f"(matched-compute control {'ON' if include_control else 'DEFERRED'})")
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    store = SupabaseStore()
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess, device=device,
        experiment=experiment, episodes=episodes, methods=methods,
        cohort="expanded", shard_count=shard_count, shard_index=shard_index,
        benchmark="libero_pro", driver="expanded_pro_16suite",
        run_metadata={"libero_pro_revision": libero_pro.LIBERO_PRO_REVISION,
                      "pnp_k": PRO_EXPANDED_K, "pnp_steps": list(PRO_EXPANDED_STEPS),
                      "episodes_per_task": episodes_per_task,
                      "include_control": include_control},
    )
