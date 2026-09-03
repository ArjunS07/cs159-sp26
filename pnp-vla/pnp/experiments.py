"""Stable, package-owned rollout drivers for thin Colab worker launchers."""
from __future__ import annotations

import collections
import contextlib
import io
import math
import time
from collections.abc import Mapping

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
# Corrected standard-LIBERO rerun.  The v1 workers inherited the checkpoint's
# 50-action execution horizon; LeRobot's published pi0.5 LIBERO evaluation
# generates 50 actions but executes 10 before replanning.
LIBERO_ACTION_STEPS = 10
LIBERO_10STEP_EXPERIMENT = "libero-hybrid-schedules-k3-a10-v2"
# Same OffScreenRenderEnv/robosuite path as PRO. The measured renderer needs two enabled
# env steps before the observation consumed at the next policy call is fresh.
LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT = "libero-u-horizon-k5-steps34-a10-v1"
LIBERO_HORIZON_DIAGNOSTIC_K = 5
LIBERO_HORIZON_DIAGNOSTIC_STEPS = (3, 4)

LIBERO_SKIP_RENDERS = True
LIBERO_RENDER_LEAD = 2
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
# Camera rendering is 90% of a LIBERO step (measured: 250.3 ms on, 24.5 ms off) and the policy
# reads an observation only at chunk boundaries. A re-enabled camera serves a cached frame for one
# step and renders on the second (27.1 ms then 249.3 ms), so lead 2 is measured, not assumed.
PRO_EXPANDED_SKIP_RENDERS = True
PRO_EXPANDED_RENDER_LEAD = 2


def build_full_methods(schedules=SCHEDULES, k=K,
                       n_action_steps=LIBERO_ACTION_STEPS):
    """The corrected 12-config standard-LIBERO matrix.

    The execution horizon is explicit so it cannot silently inherit the
    50-action value embedded in ``lerobot/pi05_libero_finetuned``.
    """
    extra_steps = sorted({BASE_INFERENCE_STEPS + k * len(schedule) for schedule in schedules})
    horizon = dict(n_action_steps=n_action_steps)
    perf = dict(skip_unused_renders=LIBERO_SKIP_RENDERS,
                render_lead=LIBERO_RENDER_LEAD)
    methods = [
        (Method.UNCERTAINTY, RolloutConfig(
            pnp_steps=OBSERVED_STEPS, pnp_k=k, save_pcp_features=True,
            **horizon, **perf)),
        *((Method.EXTRA_STEPS, RolloutConfig(
            num_inference_steps=steps, **horizon, **perf))
          for steps in extra_steps),
    ]
    for schedule in schedules:
        probe = dict(pnp_steps=schedule, pnp_k=k)
        methods.append((Method.REFINEMENT, RolloutConfig(
            **probe, refine=True, **horizon, **perf)))
    if len(methods) != 12:
        raise AssertionError(f"expected 12 full methods, got {len(methods)}")
    return methods



def build_libero_horizon_diagnostic_methods():
    """One corrected standard-LIBERO no-op arm with rich Q-screening telemetry."""
    config = RolloutConfig(
        pnp_steps=LIBERO_HORIZON_DIAGNOSTIC_STEPS,
        pnp_k=LIBERO_HORIZON_DIAGNOSTIC_K,
        n_action_steps=LIBERO_ACTION_STEPS,
        save_pcp_features=True,
        save_time_uncertainty=True,
        save_trajectory=True,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    if config.refine or config.refine_average:
        raise AssertionError("the horizon diagnostic must remain a no-op probe")
    return [(Method.UNCERTAINTY, config)]


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
    # Performance-only, excluded from LOGICAL_FIELDS, so enabling this leaves every rollout_id
    # untouched and already-collected rollouts are still skipped on resume.
    perf = dict(skip_unused_renders=PRO_EXPANDED_SKIP_RENDERS,
                render_lead=PRO_EXPANDED_RENDER_LEAD)
    methods = [(Method.UNCERTAINTY, RolloutConfig(**probe, **perf, save_pcp_features=True))]
    if include_control:
        methods.append(
            (Method.EXTRA_STEPS, RolloutConfig(num_inference_steps=control_steps, **perf)))
    methods.append((Method.REFINEMENT, RolloutConfig(**probe, **perf, refine=True)))
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


# Baseline success rates from the prior 16-suite run, for sanity-checking magnitude while a
# collection is in flight. REFERENCE ONLY: that run used different K, probe steps and (for some
# suites) a different horizon, so small differences are expected. What matters is that a suite
# lands in the same ballpark rather than collapsing to 0.
HISTORICAL_PRO_BASELINE_SR = {
    "libero_object_temp_x0.1": 0.54, "libero_object_temp_x0.2": 0.04,
    "libero_object_temp_x0.3": 0.00, "libero_object_temp_y0.1": 0.45,
    "libero_object_temp_y0.2": 0.33, "libero_object_temp_y0.3": 0.12,
    "libero_goal_swap": 0.09, "libero_object_swap": 0.04,
    "libero_spatial_swap": 0.18, "libero_10_swap": 0.00,
    "libero_goal_task": 0.10, "libero_object_task": 0.00,
    "libero_goal_with_milk": 0.95, "libero_spatial_with_milk": 0.83,
    "libero_object_with_mug": 0.95, "libero_goal_with_yellow_book": 0.90,
}

_METHOD_LABELS = {Method.UNCERTAINTY: "observed", Method.REFINEMENT: "refine",
                  Method.FRACTIONAL_M2: "fractional m=2",
                  Method.FRACTIONAL_M4: "fractional m=4",
                  Method.SUFFIX_SENSITIVITY: "diagnostic base",
                  Method.TAPERED_REFINEMENT: "tapered refine",
                  Method.PREFIX_REFINEMENT: "prefix only",
                  Method.REDUCED_STRENGTH_REFINEMENT: "inner beta",
                  Method.U20_GRADIENT: "U20 gradient",
                  Method.LATENT_RANDOM_CONTROL: "random control",
                  Method.U20_GRADIENT_GATE_015: "gate <=.015",
                  Method.U20_GRADIENT_GATE_020: "gate <=.020",
                  Method.THRESHOLD_REFINEMENT: "U-gated refine",
                  Method.DELAYED_REFINEMENT: "delayed refine",
                   Method.EXTRA_STEPS: "control",
                   Method.CHUNK_SOURCE_SOURCE: "source x2",
                   Method.CHUNK_SOURCE_MULTI_QUERY: "source multi-query",
                   Method.CHUNK_SOURCE_M1: "source + m1",
                   Method.FIVE_STEP_SINGLE_QUERY: "5-step x1",
                   Method.FIVE_STEP_LOWEST_U20: "5-step x3 low U20",
                   Method.FIVE_STEP_LOWEST_U20_REFINE: "5-step x3 + refine",
                   Method.FIVE_STEP_SINGLE_REFINE: "5-step x1 refine",
                   Method.THREE_STEP_SINGLE_REFINE: "3-step x1 refine",
                   Method.THREE_STEP_SINGLE_QUERY: "3-step x1"}


def format_progress_table(tally, method_names, historical_sr=None, *, include_overall=False,
                          count_label=None) -> str:
    """Running success rate per (suite, method), optionally against a historical baseline.

    `tally` maps (suite, method) -> [n_rollouts, n_successes]. ``historical_sr`` may be one
    suite-to-rate mapping (the legacy ``historical`` column), a label-to-suite-to-rate mapping
    for multiple named references, or ``False`` to omit references.
    """
    show_historical = historical_sr is not False
    historical_sr = (HISTORICAL_PRO_BASELINE_SR
                     if historical_sr is None else historical_sr)
    if not show_historical:
        references = []
    elif (isinstance(historical_sr, Mapping) and historical_sr
          and all(isinstance(value, Mapping) for value in historical_sr.values())):
        references = list(historical_sr.items())
    else:
        references = [("historical", historical_sr)]
    labels = [(name, _METHOD_LABELS.get(name, name)) for name in method_names]
    lines = [count_label or "n = rollouts done in THIS shard, summed over the suite's tasks "
             "(not episodes per task)",
             f"{'suite':<32}" + "".join(f"{label:>16}" for _, label in labels)
             + "".join(f"{label:>18}" for label, _ in references)]
    for suite in sorted({key[0] for key in tally}):
        cells = ""
        for name, _ in labels:
            n, wins = tally.get((suite, name), (0, 0))
            cells += f"{'-':>16}" if not n else f"{f'{wins / n:.0%} ({wins}/{n})':>16}"
        for _, rates in references:
            reference = rates.get(suite)
            cells += f"{'-':>18}" if reference is None else f"{reference:>17.0%} "
        lines.append(f"{suite:<32}{cells}")
    if include_overall:
        cells = ""
        for name, _ in labels:
            counts = [tally.get((suite, name), (0, 0))
                      for suite in sorted({key[0] for key in tally})]
            n, wins = sum(item[0] for item in counts), sum(item[1] for item in counts)
            cells += f"{'-':>16}" if not n else f"{f'{wins / n:.0%} ({wins}/{n})':>16}"
        for _, rates in references:
            values = [float(value) for value in rates.values() if value is not None]
            reference = sum(values) / len(values) if values else None
            cells += f"{'-':>18}" if reference is None else f"{reference:>17.0%} "
        lines.append(f"{'OVERALL':<32}{cells}")
    return "\n".join(lines)


def format_matched_progress_table(identity_outcomes, method_names, references) -> str:
    """Count every new/historical arm on exactly the same completed identities.

    Outcomes map a four-field identity to {method: success}; each reference maps that
    same identity to success. Partial identities are excluded, not counted as failures.
    Historical OVERALL is identity-weighted, including on partial-suite previews.
    """
    complete = {key: outcomes for key, outcomes in identity_outcomes.items()
                if set(method_names).issubset(outcomes)}
    tally = collections.defaultdict(lambda: [0, 0])
    for key, outcomes in complete.items():
        for label, reference in references.items():
            if key not in reference:
                raise ValueError(f"{label}: missing completed identity {key}")
        for name in method_names:
            tally[(key[0], name)][0] += 1
            tally[(key[0], name)][1] += int(outcomes[name])
        for label, reference in references.items():
            tally[(key[0], label)][0] += 1
            tally[(key[0], label)][1] += int(reference[key])
    return format_progress_table(
        tally, [*method_names, *references], historical_sr=False, include_overall=True,
        count_label=(f"Exact matched table: {len(complete)} identities completed in THIS shard; "
                     "every column uses these identities. Cells = SR (successes/identities); "
                     "hist = reused rollouts."))


def format_probe_diagnostic_table(tally, method_names) -> str:
    """Running means for newly completed rows; persisted per-step values live in artifacts."""
    metrics = (
        ("u_first10", "U first10"), ("u_first20", "U first20"),
        ("u_full", "U full"),
        ("contraction_first10", "C first10"),
        ("contraction_first20", "C first20"),
        ("contraction_full", "C full"),
        ("suffix_to_prefix_l2", "tail->first10 L2"),
        ("suffix_gripper_flip", "gripper flip"))
    lines = ["probe means for NEW rollouts in this invocation",
             f"{'arm':<18}" + "".join(f"{label:>18}" for _, label in metrics)]
    for method in method_names:
        cells = []
        for metric, _ in metrics:
            total, count = tally.get((method, metric), (0.0, 0))
            cells.append("-" if not count else f"{total / count:.5f}")
        lines.append(f"{_METHOD_LABELS.get(method, method):<18}"
                     + "".join(f"{cell:>18}" for cell in cells))
    return "\n".join(lines)


def expanded_pro_suites():
    """The expanded cohort minus the suites measured at 0% success (see ZERO_SR_PRO_SUITES)."""
    from . import libero_pro
    excluded = set(libero_pro.ZERO_SR_PRO_SUITES)
    return [suite for suite in libero_pro.EXPANDED_PRO_SUITES if suite not in excluded]


def _prepare_libero_pro_expanded_episodes(episodes_per_task=PRO_EXPANDED_EPISODES,
                                          suites=None, episode_idxs=None):
    """Install the expanded-cohort PRO suites and build their episode manifest.

    Unlike the canonical path this asserts no fixed identity count: `_with_milk` suites ship 10
    init states per task while the rest ship more, so the total depends on the installed assets.
    The count is printed and returned instead of hard-coded (see ROLLOUT_PLAN.md).
    """
    from . import libero_pro

    suites = list(suites) if suites is not None else expanded_pro_suites()
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            pro_dir = libero_pro.clone_libero_pro()
            libero_pro.install_assets(suites=suites, pro_dir=pro_dir)
            libero_pro.apply_env_patches(pro_dir=pro_dir)
            libero_pro.patch_torch_load()
            benchmark_dict = libero_pro.reload_benchmark()
            selected_episode_idxs = (range(episodes_per_task) if episode_idxs is None
                                     else tuple(episode_idxs))
            if not selected_episode_idxs:
                raise ValueError("episode_idxs must contain at least one index")
            if any(isinstance(index, bool) or int(index) != index or index < 0
                   for index in selected_episode_idxs):
                raise ValueError("episode_idxs must contain non-negative integers")
            if len(set(map(int, selected_episode_idxs))) != len(selected_episode_idxs):
                raise ValueError("episode_idxs must be unique")
            episodes = libero_pro.build_libero_pro_episodes(
                benchmark_dict, suites=suites,
                episode_idxs=tuple(map(int, selected_episode_idxs)))
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
    collected = sorted({ep["suite"] for ep in episodes})
    if len(collected) != len(suites):
        missing = sorted(set(suites) - set(collected))
        raise AssertionError(f"expanded PRO manifest is missing suites: {missing}")
    excluded = sorted(set(collected) & set(libero_pro.ZERO_SR_PRO_SUITES))
    if excluded:
        raise AssertionError(f"0%-SR suites must not be collected: {excluded}")
    if not all(ep["expanded_member"] for ep in episodes):
        raise AssertionError("expanded PRO manifest contains a non-expanded identity")

    per_suite = collections.Counter(ep["suite"] for ep in episodes)
    print(f'{"suite":<36}{"episodes":>9}{"max_steps":>11}')
    for suite in suites:
        steps = sorted({ep["max_steps"] for ep in episodes if ep["suite"] == suite})
        print(f"{suite:<36}{per_suite[suite]:>9}{str(steps[0] if steps else '-'):>11}")
    print(f"excluded (0% SR, no F->S transitions to learn from): "
          f"{', '.join(libero_pro.ZERO_SR_PRO_SUITES)}")
    print(f"LIBERO-PRO expanded manifest: {len(episodes)} identities "
          f"across {len(suites)} suites")
    return episodes


def _run_collection(*, store, policy, preprocess, postprocess, device, experiment, episodes,
                    methods, cohort, shard_count, shard_index,
                    benchmark="libero", driver="hybrid_schedules", run_metadata=None,
                    report_every=50, provenance=None, initial_tally=None,
                    candidate_bundles_by_method=None, historical_sr=None,
                    rollout_batch_size: int = 2, report_every_identities=None,
                    initial_identity_methods=None, progress_include_overall=False,
                    progress_count_label=None, resume_completed_only=False,
                    matched_reference_outcomes=None, initial_identity_outcomes=None):
    from tqdm.auto import tqdm
    from .libero_env import make_env
    from .rollout import run_episode_batch
    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be at least 1")

    expected = len(episodes) * len(methods)
    done = store.existing_keys(
        experiment, **({"status": "completed"} if resume_completed_only else {}))
    pending = sum(
        store.rollout_id(experiment, ep, name, cfg) not in done
        for ep in episodes for name, cfg in methods
    )
    print(f"{cohort}: {len(episodes)} identities x {len(methods)} configs; "
          f"{pending}/{expected} pending")
    refinement_schedules = [config.pnp_steps for _, config in methods if config.refine]
    run_config = {"cohort": cohort, "schedules": refinement_schedules, "pnp_k": K,
                  "n_configs": len(methods), "shard_count": shard_count,
                  "shard_index": shard_index, "rollout_batch_size": rollout_batch_size}
    run_config.update(run_metadata or {})
    store.start_run(
        driver=driver, benchmark=benchmark, experiment=experiment,
        config=run_config, provenance=provenance,
    )
    completed = 0
    method_names = [name for name, _ in methods]
    tally = collections.defaultdict(lambda: [0, 0])   # (suite, method) -> [n, successes]
    probe_tally = collections.defaultdict(lambda: [0.0, 0])
    for key, counts in (initial_tally or {}).items():
        tally[key] = [int(counts[0]), int(counts[1])]
    identity_methods = collections.defaultdict(set)
    for value in initial_identity_methods or ():
        identity_methods[tuple(value[:4])].add(value[4])
    identity_outcomes = {tuple(key): dict(value)
                         for key, value in (initial_identity_outcomes or {}).items()}
    for key, outcomes in identity_outcomes.items():
        identity_methods[key].update(outcomes)

    def progress_table():
        if matched_reference_outcomes is not None:
            return format_matched_progress_table(
                identity_outcomes, method_names, matched_reference_outcomes)
        return format_progress_table(
            tally, method_names, historical_sr=historical_sr,
            include_overall=progress_include_overall, count_label=progress_count_label)

    completed_identities = sum(
        set(method_names).issubset(names) for names in identity_methods.values())
    next_identity_report = None
    if report_every_identities:
        if (isinstance(report_every_identities, bool)
                or int(report_every_identities) != report_every_identities
                or report_every_identities < 1):
            raise ValueError("report_every_identities must be a positive integer or None")
        report_every_identities = int(report_every_identities)
        next_identity_report = (
            completed_identities // report_every_identities + 1) * report_every_identities
    occupied_lanes = batch_slots = 0
    inference_ms_total = 0.0
    collection_t0 = time.time()
    try:
        with tqdm(total=pending, desc=f"{cohort}[{shard_index}]", unit="rollout",
                  dynamic_ncols=True) as progress:
            def _record(ep, name, cfg, rid, result):
                nonlocal completed, completed_identities, next_identity_report
                store.log_result(rid, ep, name, cfg, result)
                if result.get("status") != "completed":
                    tqdm.write(
                        f"ERROR {ep['suite']} task={ep['task_idx']} "
                        f"episode={ep.get('ep_idx', ep.get('episode_idx'))} "
                        f"method={name}: {result.get('error_msg')}")
                completed += 1
                progress.update()
                if result.get("status") != "completed":
                    progress.set_postfix_str(
                        f"{ep['suite'].removeprefix('libero_')} "
                        f"{_METHOD_LABELS.get(name, name)} ERROR (retry on resume)",
                        refresh=False)
                    return
                counts = tally[(ep["suite"], name)]
                counts[0] += 1
                counts[1] += int(result["success"])
                identity = (ep["suite"], int(ep["task_idx"]),
                            int(ep.get("ep_idx", ep.get("episode_idx", 0))),
                            ep.get("init_state_hash", ""))
                was_complete = set(method_names).issubset(identity_methods[identity])
                identity_methods[identity].add(name)
                identity_outcomes.setdefault(identity, {})[name] = bool(result["success"])
                is_complete = set(method_names).issubset(identity_methods[identity])
                if is_complete and not was_complete:
                    completed_identities += 1
                for metric, value in (result.get("probe_diagnostics") or {}).items():
                    if value is not None and math.isfinite(value):
                        probe_tally[(name, metric)][0] += float(value)
                        probe_tally[(name, metric)][1] += 1
                # Running SR for this suite/arm, not just the last rollout's outcome.
                progress.set_postfix_str(
                    f"{ep['suite'].removeprefix('libero_')} "
                    f"{_METHOD_LABELS.get(name, name)} "
                    f"sr={counts[1] / counts[0]:.0%} ({counts[1]}/{counts[0]})",
                    refresh=False)
                identity_report = bool(
                    next_identity_report is not None
                    and completed_identities >= next_identity_report)
                if identity_report:
                    while completed_identities >= next_identity_report:
                        next_identity_report += report_every_identities
                    tqdm.write(
                        f"\n--- {cohort}[{shard_index}] after {completed_identities} "
                        f"identities completed in this shard; {completed} new rollouts logged ---")
                    tqdm.write(progress_table())
                    if probe_tally:
                        tqdm.write(format_probe_diagnostic_table(probe_tally, method_names))
                elif report_every and completed % report_every == 0:
                    tqdm.write(f"\n--- {cohort}[{shard_index}] after {completed} rollouts "
                               f"({completed}/{pending}) ---")
                    tqdm.write(progress_table())
                    if probe_tally:
                        tqdm.write(format_probe_diagnostic_table(probe_tally, method_names))

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
                        bundles = (candidate_bundles_by_method or {}).get(name)
                        if bundles is not None:
                            # Multi-policy selection is serial-only; one env per episode.
                            for ep, method, config, rid in todo:
                                result = run_episode(
                                    envs[0], ep, policy, preprocess, postprocess, device,
                                    config, candidate_bundles=bundles)
                                _record(ep, method, config, rid, result)
                            continue
                        for start in range(0, len(todo), rollout_batch_size):
                            group = todo[start:start + rollout_batch_size]
                            batch_t0 = time.perf_counter()
                            results = run_episode_batch(
                                envs[:len(group)], [item[0] for item in group], policy,
                                preprocess, postprocess, device, cfg)
                            inference_ms_total += (time.perf_counter() - batch_t0) * 1000.0
                            occupied_lanes += len(group); batch_slots += rollout_batch_size
                            for (ep, method, config, rid), result in zip(group, results):
                                _record(ep, method, config, rid, result)
                finally:
                    for env in envs:
                        env.close()
    except BaseException:
        store.finish_run(status="failed", n_rollouts=completed)
        raise
    store.finish_run(n_rollouts=completed)
    print(f"{cohort}: logged {completed} new rollouts; "
          f"{completed_identities} identities complete in this shard")
    if tally:
        print(progress_table())
    if probe_tally:
        print(format_probe_diagnostic_table(probe_tally, method_names))
    peak_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available(): peak_gb = torch.cuda.max_memory_allocated() / 2**30
    except ImportError:
        pass
    elapsed_h = max((time.time() - collection_t0) / 3600, 1e-9)
    print(f"{cohort}: batch utilization {occupied_lanes / max(batch_slots, 1):.1%}; "
          f"{completed / elapsed_h:.1f} rollouts/hour; inference {inference_ms_total / 1000:.1f}s; "
          f"peak CUDA {peak_gb:.2f} GiB")


def run_libero_hybrid_worker(*, shard_count: int, shard_index: int,
                             experiment: str = LIBERO_10STEP_EXPERIMENT,
                             rollout_batch_size: int = 2):
    """Run one corrected, 10-actions-per-query standard-LIBERO shard."""
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
          f"{len(broad)} broad identities; execution horizon="
          f"{LIBERO_ACTION_STEPS}, experiment={experiment}")
    policy, preprocess, postprocess = models.load_pi05()
    device = models.default_device()
    store = SupabaseStore()
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess, device=device,
        experiment=experiment, episodes=ablation, methods=full_methods,
        cohort="full_ablation", shard_count=shard_count, shard_index=shard_index,
        report_every=25, historical_sr=False,
        run_metadata={"n_action_steps": LIBERO_ACTION_STEPS,
                      "generated_chunk_size": 50,
                      "skip_unused_renders": LIBERO_SKIP_RENDERS,
                      "render_lead": LIBERO_RENDER_LEAD},
        rollout_batch_size=rollout_batch_size,
    )
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess, device=device,
        experiment=experiment, episodes=broad, methods=broad_methods,
        cohort="broad_validation", shard_count=shard_count, shard_index=shard_index,
        report_every=25, historical_sr=False,
        run_metadata={"n_action_steps": LIBERO_ACTION_STEPS,
                      "generated_chunk_size": 50,
                      "skip_unused_renders": LIBERO_SKIP_RENDERS,
                      "render_lead": LIBERO_RENDER_LEAD},
        rollout_batch_size=rollout_batch_size,
    )



def run_libero_horizon_diagnostic_worker(
        *, shard_count: int = 4, shard_index: int = 0,
        experiment: str = LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT):
    """Collect U10/U20/U50, contraction, PCP, and trajectory data on standard LIBERO."""
    import pandas as pd

    from . import models

    all_episodes = _prepare_libero_episodes()
    episodes = identity_shard(all_episodes, shard_count, shard_index)
    methods = build_libero_horizon_diagnostic_methods()
    method, config = methods[0]
    store = SupabaseStore()

    # Display the corrected historical 10-action baseline, selected by its behavior hash.
    historical_method, historical_config = next(
        item for item in build_full_methods() if item[0] == Method.UNCERTAINTY)
    historical_hash = store.config_hash(store._logical_key(
        historical_method, historical_config))
    historical_rows = pd.DataFrame(store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq(
            "experiment", LIBERO_10STEP_EXPERIMENT).eq(
            "method", Method.UNCERTAINTY).eq("config_hash", historical_hash),
        order_by=("rollout_id",)))
    if historical_rows.empty:
        raise ValueError(
            f"no corrected standard-LIBERO baseline rows found in "
            f"{LIBERO_10STEP_EXPERIMENT}")
    identity_columns = ["suite", "task_idx", "episode_idx", "init_state_hash"]
    manifest_identities = {
        (ep["suite"], ep["task_idx"], ep.get("ep_idx", ep.get("episode_idx")),
         ep["init_state_hash"])
        for ep in all_episodes}
    historical_rows = historical_rows[
        historical_rows.status.eq("completed")
        & historical_rows[identity_columns].apply(tuple, axis=1).isin(
            manifest_identities)].copy()
    if historical_rows.duplicated(identity_columns).any():
        raise ValueError("duplicate corrected standard-LIBERO baseline identities")
    if len(historical_rows) != len(all_episodes):
        raise ValueError(
            f"expected {len(all_episodes)} corrected historical baseline rows, "
            f"found {len(historical_rows)}")
    historical_sr = historical_rows.groupby("suite").success.mean().astype(float).to_dict()

    expected_hash = store.config_hash(store._logical_key(method, config))
    shard_identities = {
        (ep["suite"], ep["task_idx"], ep.get("ep_idx", ep.get("episode_idx")),
         ep["init_state_hash"])
        for ep in episodes}
    existing = pd.DataFrame(store.fetch_all(
        "rollouts",
        "suite,task_idx,episode_idx,init_state_hash,status,success,method,config_hash",
        configure=lambda query: query.eq("experiment", experiment).eq(
            "method", method).eq("config_hash", expected_hash),
        order_by=("rollout_id",)))
    if not existing.empty:
        existing = existing[
            existing.status.eq("completed")
            & existing[identity_columns].apply(tuple, axis=1).isin(
                shard_identities)]
    initial_tally = {} if existing.empty else {
        (suite, method): [len(group), int(group.success.astype(bool).sum())]
        for suite, group in existing.groupby("suite", sort=True)}

    print({
        "experiment": experiment,
        "cohort": "standard LIBERO: 4 suites x 10 tasks x 10 episodes",
        "target_identities": len(all_episodes),
        "identities_in_shard": len(episodes),
        "rollouts_in_shard": len(episodes),
        "arm": Method.UNCERTAINTY,
        "pnp_k": LIBERO_HORIZON_DIAGNOSTIC_K,
        "pnp_steps": list(LIBERO_HORIZON_DIAGNOSTIC_STEPS),
        "execution_horizon": LIBERO_ACTION_STEPS,
        "generated_chunk_size": 50,
        "diagnostic_horizons": [10, 20, 50],
        "contraction_horizons": [10, 20, 50],
        "sinks": ["PCP/Q features", "U-time + contraction", "trajectory"],
    })
    print("Periodic output: current SR, corrected historical SR, U10/U20/U50, and contraction.")

    policy, preprocess, postprocess = models.load_pi05()
    _run_collection(
        store=store, policy=policy, preprocess=preprocess, postprocess=postprocess,
        device=models.default_device(), experiment=experiment, episodes=episodes,
        methods=methods, cohort="libero_horizon_diagnostic",
        shard_count=shard_count, shard_index=shard_index,
        benchmark="libero", driver="pi05_libero_horizon_diagnostic",
        run_metadata={
            "target_identities": len(all_episodes), "pnp_k": config.pnp_k,
            "pnp_steps": list(config.pnp_steps),
            "n_action_steps": config.n_action_steps,
            "uncertainty_horizons": [10, 20, 50],
            "contraction_horizons": [10, 20, 50],
            "save_pcp_features": True, "refinement": False},
        report_every=25, initial_tally=initial_tally, historical_sr=historical_sr)


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


def run_libero_pro_expanded_worker(*, shard_count: int, shard_index: int,
                                   experiment: str = PRO_EXPANDED_EXPERIMENT,
                                   episodes_per_task: int = PRO_EXPANDED_EPISODES,
                                   include_control: bool = PRO_EXPANDED_INCLUDE_CONTROL,
                                   rollout_batch_size: int = 2):
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
        rollout_batch_size=rollout_batch_size,
    )
