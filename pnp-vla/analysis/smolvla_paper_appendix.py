"""Appendix analyses for the SmolVLA paper-figure notebook.

This module deliberately keeps the validation-only uncertainty diagnostics
(LIBERO state indices 10--14) separate from the frozen online-gate evaluation
(indices 0--9).
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from analysis.horizon_diagnostics import failure_auc_table, _download_with_retry
from pnp.diversity import DIVERSITY_PAIR_KEYS
from pnp.smolvla_followup_experiments import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_SCHEDULE_STEPS,
)
from pnp.smolvla_u20_gate_calibration import (
    SMOLVLA_U20_GATE_CALIBRATION_EXPERIMENT,
    build_smolvla_u20_gate_calibration_methods,
)
from pnp.smolvla_u20_gated_blend_experiment import (
    SMOLVLA_U20_GATED_S03_EXPERIMENT,
    SMOLVLA_U20_Q75_BY_SUITE,
    build_smolvla_u20_gated_s03_method,
)
from pnp.smolvla_u20_gated_refinement_experiment import (
    SMOLVLA_U20_GATED_REFINEMENT_EXPERIMENT,
    build_smolvla_u20_gated_refinement_method,
)


_UNCERTAINTY_KEY = re.compile(
    r"^c(?P<chunk>\d+)_s(?P<step>\d+)_u_time$")


def _export_figure(fig, output: Path, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", **kwargs)


def _completed_rows(store, experiment, method_config, *, select="*"):
    method, config = method_config
    config_hash = store.config_hash(store._logical_key(method, config))
    rows = pd.DataFrame(store.fetch_all(
        "rollouts",
        select,
        configure=lambda query: query.eq("experiment", experiment)
        .eq("method", method).eq("config_hash", config_hash)
        .eq("status", "completed"),
        order_by=("rollout_id",),
    ))
    if rows.duplicated(DIVERSITY_PAIR_KEYS).any():
        raise ValueError(f"duplicate identities for {experiment}/{method}/{config_hash}")
    return rows, method, config_hash


def _safe_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size != 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def build_early_random_u10_figures(
        store,
        output,
        *,
        suite_order,
        suite_labels,
        prefix_k=(1, 2, 4, 8),
        random_window_draws=250,
        n_boot=2000,
        random_seed=20260929):
    """Plot first-k and random-consecutive-k stock U10 failure AUROC.

    The source is the measurement-only stock arm on the frozen validation
    identities (state indices 10--14). First-k means the first *up to* k
    decision boundaries, so shorter episodes remain in the analysis.
    """
    output = Path(output)
    output.mkdir(exist_ok=True)
    stock_arm = build_smolvla_u20_gate_calibration_methods()[0]
    stock, method, config_hash = _completed_rows(
        store, SMOLVLA_U20_GATE_CALIBRATION_EXPERIMENT, stock_arm)
    if len(stock) != 200:
        raise ValueError(f"expected 200 validation identities, found {len(stock)}")
    if set(stock.episode_idx.astype(int)) != set(range(10, 15)):
        raise ValueError("early/random-k source is not the frozen indices 10--14 cohort")
    if stock.ahats_path.isna().any():
        raise ValueError("measurement-only stock rows are missing uncertainty artifacts")

    cache_file = output / f"appendix_stock_u10_chunks_{config_hash[:12]}.pkl"
    if cache_file.exists():
        chunks = pd.read_pickle(cache_file)
        print("Loaded cached validation stock U10 chunk profiles.")
    else:
        expected_pairs = dict(zip(
            SMOLVLA_SCHEDULE_STEPS, SMOLVLA_SCHEDULE_K_BY_STEP))
        decoded = []
        for row in tqdm(stock.to_dict("records"), desc="validation stock U10 artifacts"):
            payload = _download_with_retry(store, str(row["ahats_path"]))
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                for key in archive.files:
                    match = _UNCERTAINTY_KEY.match(key)
                    if match is None:
                        continue
                    step = int(match.group("step"))
                    if step not in expected_pairs:
                        continue
                    profile = np.asarray(archive[key], dtype=float).reshape(-1)
                    iter_key = key.removesuffix("_u_time") + "_u_iter_time"
                    pair_profile = np.asarray(archive[iter_key], dtype=float)
                    if len(profile) != 50 or pair_profile.shape != (
                            expected_pairs[step], 50):
                        raise ValueError(
                            f"{row['rollout_id']}/{key}: malformed uncertainty artifact")
                    decoded.append({
                        "rollout_id": row["rollout_id"],
                        "suite": row["suite"],
                        "chunk_idx": int(match.group("chunk")),
                        "flow_step": step,
                        "u10": float(profile[:10].mean()),
                    })
        records = pd.DataFrame(decoded)
        observed = tuple(sorted(records.flow_step.unique()))
        if observed != tuple(SMOLVLA_SCHEDULE_STEPS):
            raise ValueError(f"expected flow steps {SMOLVLA_SCHEDULE_STEPS}, found {observed}")
        counts = records.groupby(["rollout_id", "chunk_idx"]).flow_step.nunique()
        if not counts.eq(len(SMOLVLA_SCHEDULE_STEPS)).all():
            raise ValueError("at least one validation boundary is missing a flow-step profile")
        wide = records.pivot(
            index=["rollout_id", "suite", "chunk_idx"],
            columns="flow_step", values="u10")
        weights = dict(zip(SMOLVLA_SCHEDULE_STEPS, SMOLVLA_SCHEDULE_K_BY_STEP))
        chunks = wide.reset_index()
        chunks["u10_weighted"] = sum(
            weights[step] * chunks[step] for step in SMOLVLA_SCHEDULE_STEPS
        ) / sum(weights.values())
        chunks = chunks[["rollout_id", "suite", "chunk_idx", "u10_weighted"]]
        chunks.to_pickle(cache_file)

    observed_counts = chunks.groupby("rollout_id").size()
    expected_counts = stock.set_index("rollout_id").n_chunks.astype(int)
    if not observed_counts.reindex(expected_counts.index).eq(expected_counts).all():
        raise ValueError("decoded validation chunk counts disagree with rollout metadata")

    features = stock[DIVERSITY_PAIR_KEYS + ["rollout_id", "success", "n_chunks"]].copy()
    features["success"] = features.success.astype(bool)
    episode = chunks.groupby("rollout_id").u10_weighted.mean().rename("u10_episode")
    features = features.merge(episode.reset_index(), on="rollout_id", validate="one_to_one")
    for k in prefix_k:
        prefix = (chunks[chunks.chunk_idx.lt(k)].groupby("rollout_id")
                  .u10_weighted.mean().rename(f"u10_first{k}"))
        used = (chunks[chunks.chunk_idx.lt(k)].groupby("rollout_id").size()
                .rename(f"chunks_used_first{k}"))
        features = features.merge(
            prefix.to_frame().join(used).reset_index(),
            on="rollout_id", validate="one_to_one")

    score_order = [f"u10_first{k}" for k in prefix_k] + ["u10_episode"]
    first_auc = failure_auc_table(
        features, score_order, by_suite=True, n_boot=n_boot)
    first_auc.to_csv(output / "appendix_u10_first_k_failure_auc.csv", index=False)
    features.to_csv(output / "appendix_validation_stock_u10_features.csv", index=False)

    plot_suites = list(suite_order) + ["pooled"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#111111"]
    x = np.arange(len(score_order))
    fig, ax = plt.subplots(figsize=(8.1, 5.1), constrained_layout=True)
    for suite, color in zip(plot_suites, colors):
        group = first_auc[first_auc.suite.eq(suite)].set_index("score_name").reindex(score_order)
        values = group.failure_auc.to_numpy(float)
        lows = group.auc_ci_low.to_numpy(float)
        highs = group.auc_ci_high.to_numpy(float)
        label = "Pooled" if suite == "pooled" else suite_labels[suite]
        width = 2.7 if suite == "pooled" else 1.6
        ax.plot(x, values, marker="o", color=color, linewidth=width, label=label)
        ax.fill_between(x, lows, highs, color=color, alpha=.07)
    ax.axhline(.5, color="#777777", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, [f"First {k}" for k in prefix_k] + ["Full episode"])
    ax.set(xlabel="Stock chunks used to compute $U_{10}$",
           ylabel="Failure ROC-AUC", ylim=(0, 1),
           title=r"Early-chunk versus episode-level $U_{10}$")
    ax.grid(alpha=.22); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    _export_figure(fig, output, "appendix_u10_first_k_failure_auc")
    plt.show()

    profiles = {
        rollout_id: group.sort_values("chunk_idx").reset_index(drop=True)
        for rollout_id, group in chunks.groupby("rollout_id", sort=False)}
    rollout_order = features.rollout_id.tolist()
    random_rows = []
    for draw in tqdm(range(random_window_draws), desc="random consecutive U10 windows"):
        rng = np.random.default_rng(random_seed + draw)
        for k in prefix_k:
            sampled = []
            for rollout_id in rollout_order:
                profile = profiles[rollout_id]
                width = min(k, len(profile))
                start = int(rng.integers(0, len(profile) - width + 1))
                sampled.append(float(
                    profile.iloc[start:start + width].u10_weighted.mean()))
            sampled = np.asarray(sampled, dtype=float)
            for suite in plot_suites:
                mask = (np.ones(len(features), dtype=bool) if suite == "pooled" else
                        features.suite.eq(suite).to_numpy())
                failure = ~features.success.to_numpy(bool)
                random_rows.append({
                    "draw": draw,
                    "window_k": k,
                    "suite": suite,
                    "episodes": int(mask.sum()),
                    "failures": int(failure[mask].sum()),
                    "failure_auc": _safe_auc(failure[mask], sampled[mask]),
                })
    random_auc = pd.DataFrame(random_rows)
    random_summary = (random_auc.groupby(["suite", "window_k"], as_index=False)
                      .agg(episodes=("episodes", "first"),
                           failures=("failures", "first"),
                           auc_mean=("failure_auc", "mean"),
                           auc_median=("failure_auc", "median"),
                           auc_q05=("failure_auc", lambda values: values.quantile(.05)),
                           auc_q95=("failure_auc", lambda values: values.quantile(.95))))
    random_auc.to_csv(output / "appendix_u10_random_k_auc_draws.csv", index=False)
    random_summary.to_csv(output / "appendix_u10_random_k_auc_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.7, 5.1), constrained_layout=True)
    for suite, color in zip(plot_suites, colors):
        group = random_summary[random_summary.suite.eq(suite)].sort_values("window_k")
        label = "Pooled" if suite == "pooled" else suite_labels[suite]
        width = 2.7 if suite == "pooled" else 1.6
        ax.plot(group.window_k, group.auc_median, marker="o", color=color,
                linewidth=width, label=label)
        ax.fill_between(group.window_k, group.auc_q05, group.auc_q95,
                        color=color, alpha=.08)
    ax.axhline(.5, color="#777777", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(prefix_k)
    ax.set(xlabel="Random consecutive stock chunks $k$",
           ylabel="Failure ROC-AUC across redraws", ylim=(0, 1),
           title=r"Random-window $U_{10}$ failure prediction")
    ax.grid(alpha=.22); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    _export_figure(fig, output, "appendix_u10_random_k_failure_auc")
    plt.show()

    provenance = pd.DataFrame([{
        "analysis": "first-k and random-consecutive-k U10 failure AUROC",
        "experiment": SMOLVLA_U20_GATE_CALIBRATION_EXPERIMENT,
        "method": method,
        "config_hash": config_hash,
        "cohort": "standard LIBERO state indices 10-14",
        "episodes": len(features),
        "pnp_steps": str(tuple(SMOLVLA_SCHEDULE_STEPS)),
        "pairs_per_step": str(tuple(SMOLVLA_SCHEDULE_K_BY_STEP)),
        "random_window_draws": random_window_draws,
        "random_seed_base": random_seed,
    }])
    provenance.to_csv(output / "appendix_u10_temporal_provenance.csv", index=False)
    return {
        "features": features,
        "first_k_auc": first_auc,
        "random_auc_draws": random_auc,
        "random_auc_summary": random_summary,
        "provenance": provenance,
    }


def _decode_gate_telemetry(value, gate_key):
    if value is None:
        payload = {}
    elif isinstance(value, str):
        payload = json.loads(value)
    elif isinstance(value, dict):
        payload = value
    else:
        payload = {}
    gate = (payload or {}).get(gate_key) or {}
    return int(gate.get("n_fired") or 0), int(gate.get("n_considered") or 0)


def _load_suite_specific_gate(
        store, *, experiment, builder, gate_key, suite_order, expected_ids):
    parts = []
    provenance = []
    utilization = []
    for suite in suite_order:
        method, config = builder(suite)
        rows, _, config_hash = _completed_rows(
            store, experiment, (method, config),
            select=("rollout_id,suite,task_idx,episode_idx,init_state_hash,"
                    "success,status,ms_candidate_u"))
        rows = rows[rows.suite.eq(suite)].copy()
        if len(rows) != 100:
            raise ValueError(f"{experiment}/{suite}: expected 100 rows, found {len(rows)}")
        rows["success"] = rows.success.astype(bool)
        fires = considered = 0
        for value in rows.ms_candidate_u:
            row_fires, row_considered = _decode_gate_telemetry(value, gate_key)
            fires += row_fires
            considered += row_considered
        if considered == 0:
            raise ValueError(f"{experiment}/{suite}: gate telemetry is missing")
        utilization.append({
            "suite": suite,
            "episodes": len(rows),
            "boundaries_fired": fires,
            "boundaries_considered": considered,
            "gate_fire_rate_pct": 100.0 * fires / considered,
        })
        provenance.append({
            "experiment": experiment,
            "suite": suite,
            "method": method,
            "config_hash": config_hash,
            "u20_q75_threshold": SMOLVLA_U20_Q75_BY_SUITE[suite],
        })
        parts.append(rows)
    gated = pd.concat(parts, ignore_index=True)
    ids = set(map(tuple, gated[DIVERSITY_PAIR_KEYS].to_numpy()))
    if len(gated) != 400 or ids != expected_ids:
        raise ValueError(
            f"{experiment}: gated cohort mismatch; rows={len(gated)}, "
            f"missing={len(expected_ids - ids)}, extra={len(ids - expected_ids)}")
    utilization = pd.DataFrame(utilization)
    utilization = pd.concat([
        utilization,
        pd.DataFrame([{
            "suite": "OVERALL",
            "episodes": int(utilization.episodes.sum()),
            "boundaries_fired": int(utilization.boundaries_fired.sum()),
            "boundaries_considered": int(utilization.boundaries_considered.sum()),
            "gate_fire_rate_pct": (
                100.0 * utilization.boundaries_fired.sum()
                / utilization.boundaries_considered.sum()),
        }]),
    ], ignore_index=True)
    return gated, utilization, pd.DataFrame(provenance)


def _success_rows(named_arms, suite_order):
    rows = []
    for label, frame in named_arms:
        for suite in list(suite_order) + ["OVERALL"]:
            group = frame if suite == "OVERALL" else frame[frame.suite.eq(suite)]
            rows.append({
                "arm": label,
                "suite": suite,
                "episodes": len(group),
                "successes": int(group.success.sum()),
                "success_rate_pct": 100.0 * group.success.mean(),
            })
    return pd.DataFrame(rows)


def _transition_rows(stock, candidate, label, suite_order):
    paired = stock[DIVERSITY_PAIR_KEYS + ["success"]].merge(
        candidate[DIVERSITY_PAIR_KEYS + ["success"]],
        on=DIVERSITY_PAIR_KEYS, validate="one_to_one",
        suffixes=("_stock", "_candidate"))
    rows = []
    for suite in list(suite_order) + ["OVERALL"]:
        group = paired if suite == "OVERALL" else paired[paired.suite.eq(suite)]
        stock_success = group.success_stock.astype(bool)
        candidate_success = group.success_candidate.astype(bool)
        rows.append({
            "method": label,
            "suite": suite,
            "episodes": len(group),
            "failure_to_success": int((~stock_success & candidate_success).sum()),
            "success_to_failure": int((stock_success & ~candidate_success).sum()),
            "delta_pp": 100.0 * (candidate_success.mean() - stock_success.mean()),
        })
    return pd.DataFrame(rows)


def _plot_selective_comparison(
        *, output, stem, title, success, utilization, suite_order, suite_labels,
        colors):
    categories = list(suite_order) + ["OVERALL"]
    labels = [suite_labels[suite] for suite in suite_order] + ["Overall"]
    arm_order = list(success.arm.drop_duplicates())
    x = np.arange(len(categories))
    width = .24
    fig, axes = plt.subplots(
        1, 2, figsize=(14.2, 5.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.9, 1]})
    for offset, arm, color in zip((-width, 0, width), arm_order, colors):
        group = success[success.arm.eq(arm)].set_index("suite").reindex(categories)
        bars = axes[0].bar(x + offset, group.success_rate_pct, width,
                           color=color, label=arm)
        axes[0].bar_label(
            bars,
            labels=[f"{int(n)}/{int(d)}" for n, d in
                    zip(group.successes, group.episodes)],
            padding=2, fontsize=8, rotation=90)
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="Success rate (%)", ylim=(0, 105),
                title="Outcome comparison")
    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    axes[0].grid(axis="y", alpha=.22); axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)

    gate = utilization.set_index("suite").reindex(categories)
    gate_bars = axes[1].bar(
        x, gate.gate_fire_rate_pct, color="#6F6F6F", width=.66)
    axes[1].bar_label(
        gate_bars,
        labels=[f"{value:.1f}%" for value in gate.gate_fire_rate_pct],
        padding=3, fontsize=9)
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set(ylabel="Decision boundaries selected (%)", ylim=(0, 100),
                title=r"Online $U_{20}$-Q75 gate utilization")
    axes[1].grid(axis="y", alpha=.22); axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=16)
    _export_figure(fig, Path(output), stem)
    plt.show()


def build_selective_refinement_figures(
        store,
        output,
        *,
        arms,
        stock_label,
        refinement_label,
        aggregation_label,
        suite_order,
        suite_labels):
    """Plot frozen online U20-Q75 gates on the 400-episode evaluation set."""
    output = Path(output)
    output.mkdir(exist_ok=True)
    stock = arms[stock_label]
    expected_ids = set(map(tuple, stock[DIVERSITY_PAIR_KEYS].to_numpy()))
    gated_refinement, refinement_use, refinement_provenance = _load_suite_specific_gate(
        store,
        experiment=SMOLVLA_U20_GATED_REFINEMENT_EXPERIMENT,
        builder=build_smolvla_u20_gated_refinement_method,
        gate_key="refinement_chunk_gate",
        suite_order=suite_order,
        expected_ids=expected_ids,
    )
    gated_aggregation, aggregation_use, aggregation_provenance = _load_suite_specific_gate(
        store,
        experiment=SMOLVLA_U20_GATED_S03_EXPERIMENT,
        builder=build_smolvla_u20_gated_s03_method,
        gate_key="consensus_projection_gate",
        suite_order=suite_order,
        expected_ids=expected_ids,
    )

    refinement_success = _success_rows([
        ("Stock 10 Actions", stock),
        ("Always Refinement", arms[refinement_label]),
        ("Selective Refinement", gated_refinement),
    ], suite_order)
    aggregation_success = _success_rows([
        ("Stock 10 Actions", stock),
        ("Always Refined Aggregation", arms[aggregation_label]),
        ("Selective Refined Aggregation", gated_aggregation),
    ], suite_order)
    transitions = pd.concat([
        _transition_rows(stock, gated_refinement, "Selective Refinement", suite_order),
        _transition_rows(stock, gated_aggregation, "Selective Refined Aggregation", suite_order),
    ], ignore_index=True)

    refinement_success.to_csv(
        output / "appendix_selective_refinement_success.csv", index=False)
    aggregation_success.to_csv(
        output / "appendix_selective_aggregation_success.csv", index=False)
    refinement_use.assign(method="Selective Refinement").to_csv(
        output / "appendix_selective_refinement_gate_utilization.csv", index=False)
    aggregation_use.assign(method="Selective Refined Aggregation").to_csv(
        output / "appendix_selective_aggregation_gate_utilization.csv", index=False)
    transitions.to_csv(
        output / "appendix_selective_intervention_transitions.csv", index=False)
    pd.concat([
        refinement_provenance.assign(analysis="selective_refinement"),
        aggregation_provenance.assign(analysis="selective_refined_aggregation"),
    ], ignore_index=True).to_csv(
        output / "appendix_selective_intervention_provenance.csv", index=False)

    _plot_selective_comparison(
        output=output,
        stem="appendix_online_selective_refinement",
        title=r"Online selective refinement with a frozen $U_{20}$-Q75 gate",
        success=refinement_success,
        utilization=refinement_use,
        suite_order=suite_order,
        suite_labels=suite_labels,
        colors=("#3E73B8", "#F55174", "#7A7A7A"),
    )
    _plot_selective_comparison(
        output=output,
        stem="appendix_online_selective_refined_aggregation",
        title=r"Online selective aggregation with a frozen $U_{20}$-Q75 gate",
        success=aggregation_success,
        utilization=aggregation_use,
        suite_order=suite_order,
        suite_labels=suite_labels,
        colors=("#3E73B8", "#9448BD", "#7A7A7A"),
    )
    return {
        "gated_refinement": gated_refinement,
        "gated_aggregation": gated_aggregation,
        "refinement_success": refinement_success,
        "aggregation_success": aggregation_success,
        "refinement_utilization": refinement_use,
        "aggregation_utilization": aggregation_use,
        "transitions": transitions,
    }


