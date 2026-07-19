"""Deterministic machine-readable outputs, findings summary, and figures."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_table(frame: pd.DataFrame, name: str, output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{name}.csv", index=False)


def legacy_delta(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compatible historical anchors only; intentionally excludes invalid method aggregates."""
    historical = {"observed_pooled_roc_auc": .7987104, "observed_pooled_pr_auc": .5397236,
                  "observed_macro_suite_roc_auc": .8550581}
    detector = tables["detector_summary"].set_index("estimate_scope")
    current = {"observed_pooled_roc_auc": detector.loc["pooled", "roc_auc"],
               "observed_pooled_pr_auc": detector.loc["pooled", "pr_auc"],
               "observed_macro_suite_roc_auc": detector.loc["macro_suite", "roc_auc"]}
    return pd.DataFrame([{"metric": key, "historical_value": old,
                          "recomputed_value": current[key], "delta": current[key] - old,
                          "provenance": "user-supplied preliminary regression anchor"}
                         for key, old in historical.items()])


def success_figure(table: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    frame = table.sort_values("sr")
    fig, ax = plt.subplots(figsize=(8, 4))
    err = [frame.sr - frame.ci_low, frame.ci_high - frame.sr]
    labels = [f"{x} (n={n})" for x, n in zip(frame.condition_label, frame.n)]
    ax.barh(labels, 100 * frame.sr, xerr=100 * pd.DataFrame(err).to_numpy(), capsize=3)
    ax.set_xlim(0, 100); ax.set_xlabel("Success rate (%)"); ax.set_title("Balanced all-identity comparison")
    fig.tight_layout(); (output / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figures" / "success_all_identity.pdf", bbox_inches="tight")
    fig.savefig(output / "figures" / "success_all_identity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_figure(fig, name: str, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{name}.png", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def publication_figures(tables: dict[str, pd.DataFrame], output: Path) -> None:
    """Render every defensible figure supported by the rollout-only snapshot."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Full-ablation configuration SR: exact conditions, Wilson intervals, full 0-100 axis.
    frame = tables["success_full_ablation"].sort_values("sr")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([f"{x} (n={n})" for x, n in zip(frame.condition_label, frame.n)], 100 * frame.sr,
            xerr=np.vstack((100 * (frame.sr - frame.ci_low), 100 * (frame.ci_high - frame.sr))),
            capsize=3, color="#4C78A8")
    ax.set(xlim=(0, 100), xlabel="Success rate (%)",
           title="Full-ablation success by exact configuration")
    _save_figure(fig, "success_full_ablation", output)

    # Paired effect forest plot, split by cohort and with a zero reference.
    frame = tables["paired_comparisons"].sort_values(["cohort", "delta_pp"])
    labels = [f"{r.condition_label} — {r.cohort} (n={r.n})" for r in frame.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(frame.delta_pp, labels,
                xerr=np.vstack((frame.delta_pp - frame.delta_ci_low_pp,
                                frame.delta_ci_high_pp - frame.delta_pp)),
                fmt="o", capsize=3, color="#E45756")
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("Paired success-rate difference (pp)")
    ax.set_title("Paired effects versus observed/no-op baseline")
    _save_figure(fig, "paired_effects", output)

    # Discordant transitions make both recovery and degradation visible.
    frame = tables["paired_comparisons"].sort_values(["cohort", "condition_label"])
    y = np.arange(len(frame)); fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y, frame.F_to_S, label="F→S recovery", color="#54A24B")
    ax.barh(y, -frame.S_to_F, label="S→F degradation", color="#E45756")
    ax.set_yticks(y, [f"{r.condition_label} — {r.cohort} (n={r.n})" for r in frame.itertuples()])
    ax.axvline(0, color="black", lw=.8); ax.set_xlabel("Discordant pairs (degradation ← 0 → recovery)")
    ax.set_title("Paired outcome transitions"); ax.legend()
    _save_figure(fig, "paired_transitions", output)

    # Pooled and suite-specific detector ROC-AUC with bootstrap intervals.
    suite = tables["detector_by_suite"].copy()
    pooled = tables["detector_summary"].query("estimate_scope == 'pooled'").iloc[0]
    labels = ["pooled"] + suite.suite.tolist()
    auc = np.r_[pooled.roc_auc, suite.roc_auc]
    low = np.r_[pooled.roc_ci_low, suite.roc_ci_low]
    high = np.r_[pooled.roc_ci_high, suite.roc_ci_high]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(auc, labels, xerr=np.vstack((auc - low, high - auc)), fmt="o", capsize=3)
    ax.axvline(.5, color="black", ls="--", lw=1); ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC (identity-bootstrap 95% CI)"); ax.set_title("Observed-arm failure detector")
    _save_figure(fig, "detector_auc_by_suite", output)

    # Legacy ROC-curve view, corrected to use only prospective observed-arm telemetry.
    curves = tables["detector_roc_curves"]
    auc_lookup = dict(zip(suite.suite, suite.roc_auc)); auc_lookup["pooled"] = pooled.roc_auc
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, group in curves.groupby("suite"):
        width = 2 if name == "pooled" else 1.3
        ax.plot(group.fpr, group.tpr, lw=width, label=f"{name} (AUC={auc_lookup[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="random")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate",
           title="Observed-arm failure-detector ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    _save_figure(fig, "detector_roc_curves", output)

    # Seven action dimensions and declared aggregate scores.
    frame = tables["detector_per_dof"].sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(frame.score, frame.roc_auc, color="#72B7B2")
    ax.axvline(.5, color="black", ls="--", lw=1); ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC"); ax.set_title("Observed-arm detector by action dimension")
    _save_figure(fig, "detector_per_dof", output)

    # Early telemetry is prospective; full episode is the complete observed trajectory.
    frame = tables["detector_early_window"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(frame)); width = .36
    ax.bar(x - width / 2, frame.roc_auc, width, label="ROC-AUC")
    ax.bar(x + width / 2, frame.pr_auc, width, label="PR-AUC")
    ax.set_xticks(x, frame.window); ax.set_ylim(0, 1); ax.set_ylabel("Area under curve")
    ax.set_title("Early-window versus full-episode detector"); ax.legend()
    _save_figure(fig, "detector_early_window", output)

    # Legacy SR-vs-uncertainty view with suite-local deciles and denominators.
    frame = tables["detector_uncertainty_bins"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for suite_name, group in frame.groupby("suite"):
        ax.plot(group.mean_uncertainty, 100 * group.success_rate, marker="o",
                label=suite_name.replace("libero_", ""))
    ax.set(xlabel="Mean observed uncertainty (suite decile)", ylabel="Success rate (%)",
           ylim=(0, 100), title="Success rate versus observed uncertainty")
    ax.legend(fontsize=8); _save_figure(fig, "success_vs_uncertainty", output)

    frame = tables["detector_risk_coverage"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(100 * frame.coverage, 100 * frame.risk, marker="o")
    ax.set(xlabel="Coverage retained (%)", ylabel="Failure risk (%)", xlim=(0, 100),
           title="Observed-arm risk–coverage curve")
    _save_figure(fig, "detector_risk_coverage", output)

    # Euler-step discriminability profile from the single shared step-indexed rollout.
    frame = tables["detector_euler_profile"]
    suites = sorted(frame.suite.unique())
    fig, axes = plt.subplots(1, len(suites), figsize=(4 * len(suites), 4), sharey=True)
    for ax, suite_name in zip(np.atleast_1d(axes), suites):
        group = frame[frame.suite == suite_name]
        for fail, label, color in [(0, "success", "#4C78A8"), (1, "failure", "#E45756")]:
            line = group[group.fail == fail]
            ax.errorbar(line.euler_step, line["mean"], yerr=line["sem"], marker="o",
                        capsize=2, label=label, color=color)
        ax.set_title(suite_name.replace("libero_", "")); ax.set_xlabel("Euler step")
    axes = np.atleast_1d(axes); axes[0].set_ylabel("Mean uncertainty ± SEM"); axes[-1].legend(fontsize=8)
    fig.suptitle("Observed uncertainty by Euler step and outcome")
    _save_figure(fig, "uncertainty_by_euler_step", output)

    # Episode-time trajectory, normalized independently within each observed rollout.
    frame = tables["detector_time_profile"]
    suites = sorted(frame.suite.unique())
    fig, axes = plt.subplots(1, len(suites), figsize=(4 * len(suites), 4), sharey=True)
    for ax, suite_name in zip(np.atleast_1d(axes), suites):
        group = frame[frame.suite == suite_name]
        for fail, label, color in [(0, "success", "#4C78A8"), (1, "failure", "#E45756")]:
            line = group[group.fail == fail]
            ax.plot(10 * line.episode_progress_bin + 5, line["mean"], marker="o", label=label, color=color)
        ax.set_title(suite_name.replace("libero_", "")); ax.set_xlabel("Episode progress (%)")
    axes = np.atleast_1d(axes); axes[0].set_ylabel("Mean uncertainty"); axes[-1].legend(fontsize=8)
    fig.suptitle("Observed uncertainty over episode time")
    _save_figure(fig, "uncertainty_over_time", output)

    # Legacy suite phase comparison, now restricted to the three balanced 400-row conditions.
    frame = tables["success_by_suite"]
    broad = frame[frame.groupby("config_hash").config_hash.transform("size") == 4].copy()
    labels = broad.condition_label.drop_duplicates().tolist()
    suites = sorted(broad.suite.unique()); x = np.arange(len(suites)); width = .8 / len(labels)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, label in enumerate(labels):
        group = broad[broad.condition_label == label].set_index("suite").loc[suites]
        offset = (i - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, 100 * group.sr, width, label=label)
    ax.set_xticks(x, [s.replace("libero_", "") for s in suites]); ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)"); ax.set_title("Balanced conditions by suite"); ax.legend(fontsize=7)
    _save_figure(fig, "success_balanced_by_suite", output)

    # Observed per-task heatmap supersedes ad-hoc suite/task plots while showing exact values.
    task = tables["success_by_task"]
    task = task[task.condition_label.str.startswith("observed/no-op")]
    pivot = task.pivot(index="suite", columns="task_idx", values="sr").sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(100 * pivot.to_numpy(), vmin=0, vmax=100, aspect="auto", cmap="Blues")
    ax.set_yticks(np.arange(len(pivot)), [x.replace("libero_", "") for x in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns); ax.set_xlabel("Task index")
    ax.set_title("Observed/no-op success rate by suite and task")
    fig.colorbar(image, ax=ax, label="Success rate (%)")
    _save_figure(fig, "observed_success_task_heatmap", output)

    if "geometry_directional" in tables and not tables["geometry_directional"].empty:
        frame = tables["geometry_directional"].copy()
        labels = np.where(frame.success, "success", "failure")
        x = np.arange(len(frame)); width = .36
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, frame.parallel_variance_mean, width, label="parallel")
        ax.bar(x + width / 2, frame.lateral_variance_mean, width, label="lateral")
        ax.set_xticks(x, labels); ax.set_ylabel("Mean directional variance")
        ax.set_title("Observed-arm directional uncertainty geometry"); ax.legend()
        _save_figure(fig, "geometry_directional", output)


def write_report(experiment: str, snapshot: Path, validation: dict,
                 tables: dict[str, pd.DataFrame], availability: dict) -> Path:
    # A rerun is a clean deterministic render, not an accumulation of stale formats/figures.
    for directory in (snapshot / "tables", snapshot / "figures"):
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
    for obsolete in (snapshot / "availability.json", snapshot / "findings.md"):
        obsolete.unlink(missing_ok=True)
    for name, frame in tables.items():
        write_table(frame, name, snapshot)
    delta = legacy_delta(tables)
    write_table(delta, "legacy_delta", snapshot)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["availability"] = availability
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    success_figure(tables["success_all_identity"], snapshot)
    publication_figures(tables, snapshot)
    return snapshot
