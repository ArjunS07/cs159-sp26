"""Leakage-free stock-U20 calibration on standard-LIBERO indices 10--14."""
from __future__ import annotations

import contextlib
import io
import json
import re
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Method, RolloutConfig, SMOLVLA_REPO_ID
from .experiments import (
    LIBERO_ACTION_STEPS,
    LIBERO_RENDER_LEAD,
    LIBERO_SKIP_RENDERS,
    LIBERO_SUITES,
    _run_collection,
)
from .smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
)
from .smolvla_blend_ablation_experiment import (
    build_smolvla_consensus_s03_method,
)
from .store import SupabaseStore, gather_provenance


SMOLVLA_U20_GATE_CALIBRATION_EXPERIMENT = (
    "smolvla-libero-stock-u20-gate-calibration-idx10-14-s03-v1")
SMOLVLA_U20_GATE_CALIBRATION_INDICES = tuple(range(10, 15))
SMOLVLA_U20_GATE_CALIBRATION_IDENTITIES = 200
SMOLVLA_U20_GATE_QUANTILE = 0.75

_U_TIME_KEY = re.compile(r"^c(?P<chunk>\d+)_s(?P<step>\d+)_u_time$")


def build_smolvla_u20_gate_calibration_method():
    """Measure P&P uncertainty while executing the exact stock SmolVLA chunk."""
    config = RolloutConfig(
        pnp_steps=SMOLVLA_SCHEDULE_STEPS,
        pnp_k=max(SMOLVLA_SCHEDULE_K_BY_STEP),
        pnp_k_by_step=SMOLVLA_SCHEDULE_K_BY_STEP,
        refine=False,
        num_inference_steps=10,
        n_action_steps=LIBERO_ACTION_STEPS,
        save_time_uncertainty=True,
        save_trajectory=False,
        skip_unused_renders=LIBERO_SKIP_RENDERS,
        render_lead=LIBERO_RENDER_LEAD,
    )
    if config.refine:
        raise AssertionError("U20 calibration must execute the stock chunk")
    return Method.UNCERTAINTY, config


def build_smolvla_u20_gate_calibration_methods():
    """Matched stock-measurement and always-on s=0.3/K=3 blend arms."""
    stock = build_smolvla_u20_gate_calibration_method()
    blend_name, blend_config = build_smolvla_consensus_s03_method()
    # The calibration only needs the blend outcome. These persistence-only sinks do not alter
    # behavior or config identity and would needlessly duplicate notebook-96's large artifacts.
    blend = (blend_name, replace(
        blend_config, save_time_uncertainty=False, save_trajectory=False))
    if blend[0] != Method.SMOLVLA_CONSENSUS_PROJECT_S03_K3:
        raise AssertionError("expected notebook-106 s=0.3 K=3 projection as the blend arm")
    if blend[1].consensus_projection_step != 7 or blend[1].n_action_steps != 10:
        raise AssertionError("blend calibration must use s=0.3 and execute 10 actions")
    return [stock, blend]


def prepare_smolvla_u20_gate_calibration_episodes():
    """Return all 40 LIBERO tasks at held-out calibration state indices 10--14."""
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
            episode_idxs=SMOLVLA_U20_GATE_CALIBRATION_INDICES,
            tasks=tasks,
        )
    identities = {
        (ep["suite"], int(ep["task_idx"]), int(ep["ep_idx"]), ep["init_state_hash"])
        for ep in episodes
    }
    if (len(episodes) != SMOLVLA_U20_GATE_CALIBRATION_IDENTITIES
            or len(identities) != len(episodes)):
        tail = captured.getvalue()[-2000:]
        raise AssertionError(
            f"expected {SMOLVLA_U20_GATE_CALIBRATION_IDENTITIES} unique calibration "
            f"identities, found {len(episodes)} rows/{len(identities)} identities\n{tail}")
    if {int(ep["ep_idx"]) for ep in episodes} != set(
            SMOLVLA_U20_GATE_CALIBRATION_INDICES):
        raise AssertionError("calibration manifest contains unexpected state indices")
    return episodes


def run_smolvla_u20_gate_calibration_worker(
        *, rollout_batch_size: int = 8,
        experiment: str = SMOLVLA_U20_GATE_CALIBRATION_EXPERIMENT):
    """Collect stock trajectories and per-boundary pair-weighted U20."""
    from . import models

    episodes = prepare_smolvla_u20_gate_calibration_episodes()
    methods = build_smolvla_u20_gate_calibration_methods()
    print({
        "experiment": experiment,
        "model": SMOLVLA_REPO_ID,
        "calibration_indices": [10, 14],
        "evaluation_indices_reserved": [0, 9],
        "identities": len(episodes),
        "arms": [
            "stock chunk with measurement-only P&P",
            "always-on stock/refine 50/50 average -> s=0.3 K=3 projection",
        ],
        "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
        "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
        "gate_signal": "pair-weighted U20 at each decision boundary",
        "planned_gate_quantile": SMOLVLA_U20_GATE_QUANTILE,
        "integration_steps": 10,
        "n_action_steps": LIBERO_ACTION_STEPS,
        "generated_chunk_size": 50,
        "rollout_batch_size": rollout_batch_size,
        "video": "off",
    })
    print("Periodic output: both arms' per-suite and overall SR every 10 matched identities.")

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
        methods=methods,
        cohort="smolvla_stock_u20_gate_calibration_idx10_14_s03",
        shard_count=1,
        shard_index=0,
        benchmark="libero",
        driver="smolvla_stock_u20_gate_calibration",
        run_metadata={
            "model_repo_id": SMOLVLA_REPO_ID,
            "target_identities": len(episodes),
            "episode_indices": list(SMOLVLA_U20_GATE_CALIBRATION_INDICES),
            "integration_steps": 10,
            "n_action_steps": LIBERO_ACTION_STEPS,
            "generated_chunk_size": 50,
            "pnp_steps": list(SMOLVLA_SCHEDULE_STEPS),
            "pnp_k_by_step": list(SMOLVLA_SCHEDULE_K_BY_STEP),
            "gate_uncertainty_horizon": 20,
            "planned_gate_quantile": SMOLVLA_U20_GATE_QUANTILE,
            "stock_measurement_only": True,
            "blend_projection_step": 7,
            "blend_projection_s": 0.3,
            "blend_projection_k": 3,
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


def pair_weighted_u20_profile(payload: bytes) -> tuple[float, ...]:
    """Decode one artifact into one (3,1,1)-weighted U20 value per chunk."""
    by_chunk: dict[int, dict[int, float]] = defaultdict(dict)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        for key in archive.files:
            match = _U_TIME_KEY.match(key)
            if match is None:
                continue
            profile = np.asarray(archive[key], np.float32).reshape(-1)
            if len(profile) < 20:
                raise ValueError(f"{key} has fewer than 20 action positions")
            by_chunk[int(match.group("chunk"))][int(match.group("step"))] = float(
                profile[:20].mean())
    if not by_chunk:
        raise ValueError("uncertainty artifact contains no U-time profiles")
    expected_chunks = list(range(max(by_chunk) + 1))
    if sorted(by_chunk) != expected_chunks:
        raise ValueError("uncertainty chunks are not contiguous")
    weights = dict(zip(SMOLVLA_SCHEDULE_STEPS, SMOLVLA_SCHEDULE_K_BY_STEP))
    result = []
    for chunk in expected_chunks:
        if set(by_chunk[chunk]) != set(weights):
            raise ValueError(
                f"chunk {chunk} has Euler steps {sorted(by_chunk[chunk])}, "
                f"expected {sorted(weights)}")
        result.append(sum(
            weights[step] * by_chunk[chunk][step] for step in weights
        ) / sum(weights.values()))
    if not np.isfinite(result).all():
        raise ValueError("U20 profile contains non-finite values")
    return tuple(map(float, result))


def _calibration_rows(store, *, experiment: str, method, config,
                      require_complete: bool) -> list[dict]:
    config_hash = store.config_hash(store._logical_key(method, config))
    rows = store.fetch_all(
        "rollouts",
        ("rollout_id,suite,task_idx,episode_idx,init_state_hash,status,success,method,"
         "config_hash,ahats_path,n_chunks"),
        configure=lambda query: query.eq("experiment", experiment).eq(
            "method", method).eq("config_hash", config_hash).eq("status", "completed"),
        order_by=("rollout_id",),
    )
    identities = {
        (row["suite"], int(row["task_idx"]), int(row["episode_idx"]),
         row["init_state_hash"])
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError(f"{method} calibration rows contain duplicate identities")
    if require_complete and len(rows) != SMOLVLA_U20_GATE_CALIBRATION_IDENTITIES:
        raise ValueError(
            f"expected {SMOLVLA_U20_GATE_CALIBRATION_IDENTITIES} completed {method} "
            f"calibration rows, found {len(rows)}")
    if method == Method.UNCERTAINTY:
        missing = [row["rollout_id"] for row in rows if not row.get("ahats_path")]
        if missing:
            raise ValueError(f"stock rows lack uncertainty artifacts: {missing[:3]}")
    return rows


def _download_with_retry(store, path: str, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            return store._download(path)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def summarize_smolvla_u20_gate_calibration(
        *, experiment: str = SMOLVLA_U20_GATE_CALIBRATION_EXPERIMENT,
        output_path: str | Path | None = None,
        require_complete: bool = True,
        store=None) -> dict:
    """Print and optionally save pooled/per-suite fixed U20 quantile thresholds."""
    store = store or SupabaseStore()
    stock_method, blend_method = build_smolvla_u20_gate_calibration_methods()
    stock_rows = _calibration_rows(
        store, experiment=experiment, method=stock_method[0], config=stock_method[1],
        require_complete=require_complete)
    blend_rows = _calibration_rows(
        store, experiment=experiment, method=blend_method[0], config=blend_method[1],
        require_complete=require_complete)

    def identity(row):
        return (str(row["suite"]), int(row["task_idx"]), int(row["episode_idx"]),
                str(row["init_state_hash"]))

    blend_by_identity = {identity(row): row for row in blend_rows}
    stock_identities = {identity(row) for row in stock_rows}
    if stock_identities != set(blend_by_identity):
        raise ValueError("stock and blend calibration arms are not exact identity matches")
    values_by_suite: dict[str, list[float]] = defaultdict(list)
    first_by_suite: dict[str, list[float]] = defaultdict(list)
    paired = []
    for index, row in enumerate(stock_rows, 1):
        profile = pair_weighted_u20_profile(
            _download_with_retry(store, str(row["ahats_path"])))
        if len(profile) != int(row["n_chunks"]):
            raise ValueError(
                f"{row['rollout_id']} decoded {len(profile)} chunks but n_chunks="
                f"{row['n_chunks']}")
        values_by_suite[str(row["suite"])].extend(profile)
        first_by_suite[str(row["suite"])].append(profile[0])
        blend_row = blend_by_identity[identity(row)]
        paired.append({
            "suite": str(row["suite"]),
            "first_u20": float(profile[0]),
            "episode_u20": float(np.mean(profile)),
            "stock_success": bool(row["success"]),
            "blend_success": bool(blend_row["success"]),
        })
        if index % 25 == 0 or index == len(stock_rows):
            print(
                f"[u20 calibration] decoded {index}/{len(stock_rows)} matched identities",
                flush=True)

    quantiles = (0.50, 0.65, 0.75, 0.80, 0.90)

    def describe(values):
        array = np.asarray(values, dtype=float)
        return {
            "n": int(len(array)),
            "mean": float(array.mean()),
            **{f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in quantiles},
        }

    pooled = [value for values in values_by_suite.values() for value in values]
    pooled_first = [value for values in first_by_suite.values() for value in values]
    stock_success = np.asarray([row["stock_success"] for row in paired], dtype=bool)
    blend_success = np.asarray([row["blend_success"] for row in paired], dtype=bool)
    first_u20 = np.asarray([row["first_u20"] for row in paired], dtype=float)

    sweep = []
    # Quantile-based thresholds are stable under arbitrary U scaling and avoid selecting an
    # apparent optimum from every unique sample. The final indices-0:9 run remains untouched.
    for quantile in np.arange(0.50, 0.951, 0.05):
        threshold = float(np.quantile(first_u20, quantile))
        choose_blend = first_u20 >= threshold
        selected = np.where(choose_blend, blend_success, stock_success)
        sweep.append({
            "quantile": float(round(quantile, 2)),
            "threshold": threshold,
            "blend_selected": int(choose_blend.sum()),
            "blend_selected_pct": float(100 * choose_blend.mean()),
            "selected_successes": int(selected.sum()),
            "selected_sr_pct": float(100 * selected.mean()),
            "selected_minus_stock_pp": float(100 * (selected.mean() - stock_success.mean())),
        })
    best = max(
        sweep,
        key=lambda row: (row["selected_sr_pct"], -abs(row["quantile"] - 0.75)),
    )
    transitions = {
        "F_to_F": int((~stock_success & ~blend_success).sum()),
        "F_to_S": int((~stock_success & blend_success).sum()),
        "S_to_F": int((stock_success & ~blend_success).sum()),
        "S_to_S": int((stock_success & blend_success).sum()),
    }
    summary = {
        "experiment": experiment,
        "matched_identities": len(paired),
        "completed_rollouts": len(stock_rows) + len(blend_rows),
        "episode_indices": list(SMOLVLA_U20_GATE_CALIBRATION_INDICES),
        "signal": "pair-weighted U20; steps=(1,2,3), weights=(3,1,1)",
        "recommended_quantile": SMOLVLA_U20_GATE_QUANTILE,
        "paired_outcomes": {
            "stock_sr_pct": float(100 * stock_success.mean()),
            "always_blend_sr_pct": float(100 * blend_success.mean()),
            "always_blend_minus_stock_pp": float(
                100 * (blend_success.mean() - stock_success.mean())),
            **transitions,
        },
        "all_boundaries": {
            "pooled": describe(pooled),
            "by_suite": {
                suite: describe(values_by_suite[suite]) for suite in sorted(values_by_suite)
            },
        },
        "first_boundary_only": {
            "pooled": describe(pooled_first),
            "by_suite": {
                suite: describe(first_by_suite[suite]) for suite in sorted(first_by_suite)
            },
        },
        "first_boundary_episode_arm_threshold_sweep": sweep,
        "best_first_boundary_proxy": best,
    }
    summary["recommended_per_suite_q75"] = {
        suite: stats["q75"]
        for suite, stats in summary["all_boundaries"]["by_suite"].items()
    }
    print(json.dumps(summary, indent=2))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[u20 calibration] saved {path}")
    return summary
