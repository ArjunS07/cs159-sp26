"""Deterministic machine-readable tables, availability metadata, and figures."""
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


def _clean_rendered_outputs(output: Path) -> None:
    """Remove stale render products while preserving raw snapshot inputs."""
    for directory in (output / "tables", output / "figures"):
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
    for obsolete in (output / "availability.json", output / "findings.md"):
        obsolete.unlink(missing_ok=True)


def _write_availability(output: Path, availability: dict) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["availability"] = availability
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def denoising_figures(tables: dict[str, pd.DataFrame], output: Path,
                      *, prefix: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    profile = tables[f"{prefix}denoising_profile_by_outcome"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, label in zip(
            axes, ("u_mean", "a_std_mean", "action_motion"),
            ("Mean uncertainty", "Mean action SD", "Mean action motion")):
        for fail, outcome, color in [(0, "success", "#4C78A8"), (1, "failure", "#E45756")]:
            line = profile[profile.fail == fail]
            ax.errorbar(line.euler_step, line[metric], yerr=line[f"{metric}_sem"],
                        marker="o", capsize=2, label=outcome, color=color)
        ax.set(xlabel="Denoising step", ylabel=label)
    axes[-1].legend()
    fig.suptitle("Observed-arm denoising dynamics by outcome")
    _save_figure(fig, f"{prefix}denoising_profiles", output)

    step_auc = tables[f"{prefix}denoising_step_metrics"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric, group in step_auc.groupby("metric"):
        ax.plot(group.euler_step, group.roc_auc, marker="o",
                label=metric.replace("_", " "))
    ax.axhline(.5, color="black", ls="--", lw=1)
    ax.set(xlabel="Denoising step", ylabel="Failure ROC-AUC", ylim=(0, 1),
           title="Detector information across denoising")
    ax.legend(fontsize=8)
    _save_figure(fig, f"{prefix}denoising_step_auc", output)

    models = tables[f"{prefix}denoising_oof_models"]
    models = models[models.scope == "pooled"].copy()
    models["label"] = (models.window.str.replace("_", " ") + " — " +
                       models.model.str.replace("_", " "))
    models = models.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(models.roc_auc, models.label,
                xerr=np.vstack((models.roc_auc - models.roc_ci_low,
                                models.roc_ci_high - models.roc_auc)),
                fmt="o", capsize=3)
    ax.axvline(.5, color="black", ls="--", lw=1)
    ax.set(xlim=(0, 1), xlabel="Held-out failure ROC-AUC (95% bootstrap CI)",
           title="Out-of-fold denoising-trajectory detectors")
    _save_figure(fig, f"{prefix}denoising_oof_models", output)


def publication_figures(tables: dict[str, pd.DataFrame], output: Path) -> None:
    """Render every defensible figure supported by the rollout-only snapshot."""
    import matplotlib.pyplot as plt
    import numpy as np

    denoising_figures(tables, output)

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
    _clean_rendered_outputs(snapshot)
    for name, frame in tables.items():
        write_table(frame, name, snapshot)
    delta = legacy_delta(tables)
    write_table(delta, "legacy_delta", snapshot)
    _write_availability(snapshot, availability)
    success_figure(tables["success_all_identity"], snapshot)
    publication_figures(tables, snapshot)
    return snapshot


def pro_figures(tables: dict[str, pd.DataFrame], output: Path) -> None:
    """Focused legacy-parity figures for the canonical PRO collection."""
    import matplotlib.pyplot as plt
    import numpy as np

    overall = tables["pro_success_overall"].sort_values("sr")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([f"{x} (n={n})" for x, n in zip(overall.condition_label, overall.n)],
            100 * overall.sr,
            xerr=np.vstack((100 * (overall.sr - overall.ci_low),
                            100 * (overall.ci_high - overall.sr))), capsize=3)
    ax.set(xlim=(0, 100), xlabel="Success rate (%)", title="Canonical LIBERO-PRO success")
    _save_figure(fig, "pro_success_overall", output)

    suite = tables["pro_success_by_suite"]
    suites = sorted(suite.suite.unique()); labels = suite.condition_label.drop_duplicates().tolist()
    x = np.arange(len(suites)); width = .8 / len(labels)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(labels):
        group = suite[suite.condition_label == label].set_index("suite").loc[suites]
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, 100 * group.sr, width, label=label)
    ax.set_xticks(x, [s.replace("libero_", "") for s in suites], rotation=20, ha="right")
    ax.set_ylim(0, 100); ax.set_ylabel("Success rate (%)"); ax.legend(fontsize=7)
    ax.set_title("Canonical PRO conditions by suite")
    _save_figure(fig, "pro_success_by_suite", output)

    perturb = tables["pro_success_by_perturbation"]
    position = perturb[perturb.suite_family == "position_perturb"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, axis_name in zip(axes, ("x", "y")):
        group = position[position.perturb_axis == axis_name]
        for label, line in group.groupby("condition_label"):
            line = line.sort_values("perturb_strength")
            ax.plot(line.perturb_strength, 100 * line.sr, marker="o", label=label)
        ax.set(xlabel=f"Position perturbation strength ({axis_name})", ylim=(0, 100))
    axes[0].set_ylabel("Success rate (%)"); axes[1].legend(fontsize=7)
    fig.suptitle("PRO position-perturbation robustness")
    _save_figure(fig, "pro_position_strength", output)

    task = tables["pro_success_by_task"]
    observed_task = task[task.condition_label.str.startswith("observed/no-op")]
    refined_task = task[task.condition_label.str.startswith("refine-last")]
    observed_pivot = observed_task.pivot(index="suite", columns="task_idx", values="sr").sort_index()
    refined_pivot = refined_task.pivot(index="suite", columns="task_idx", values="sr").reindex(observed_pivot.index)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(100 * observed_pivot.to_numpy(), vmin=0, vmax=100, aspect="auto", cmap="Blues")
    im1 = axes[1].imshow(100 * (refined_pivot - observed_pivot).to_numpy(), vmin=-50, vmax=50,
                         aspect="auto", cmap="RdYlGn")
    for ax in axes:
        ax.set_yticks(np.arange(len(observed_pivot)),
                      [s.replace("libero_", "") for s in observed_pivot.index])
        ax.set_xticks(np.arange(len(observed_pivot.columns)), observed_pivot.columns)
        ax.set_xlabel("Task index")
    axes[0].set_title("Observed/no-op SR"); axes[1].set_title("Refinement lift (pp)")
    fig.colorbar(im0, ax=axes[0], label="Success rate (%)")
    fig.colorbar(im1, ax=axes[1], label="Δ success rate (pp)")
    _save_figure(fig, "pro_task_success_and_lift", output)

    paired = tables["pro_paired_overall"].sort_values("delta_pp")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.errorbar(paired.delta_pp, paired.condition_label,
                xerr=np.vstack((paired.delta_pp - paired.delta_ci_low_pp,
                                paired.delta_ci_high_pp - paired.delta_pp)), fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("Paired success-rate difference (pp)")
    ax.set_title("PRO paired effects versus observed/no-op")
    _save_figure(fig, "pro_paired_effects", output)

    transitions = tables["pro_paired_by_suite"]
    transitions = transitions[transitions.condition_label.str.startswith("refine-last")].set_index("suite").loc[suites]
    fig, ax = plt.subplots(figsize=(11, 5)); bottom = np.zeros(len(suites))
    for column, label, color in [("F_to_F", "F→F", "#E45756"), ("F_to_S", "F→S", "#54A24B"),
                                  ("S_to_F", "S→F", "#F58518"), ("S_to_S", "S→S", "#4C78A8")]:
        ax.bar(np.arange(len(suites)), transitions[column], bottom=bottom, label=label, color=color)
        bottom += transitions[column].to_numpy()
    ax.set_xticks(np.arange(len(suites)), [s.replace("libero_", "") for s in suites], rotation=20, ha="right")
    ax.set_ylabel("Episode count"); ax.set_title("Observed/no-op → refine-last (4,5) transitions")
    ax.legend(); _save_figure(fig, "pro_refinement_transitions", output)

    detector_suite = tables["pro_detector_by_suite"]
    pooled = tables["pro_detector_summary"].query("estimate_scope == 'pooled'").iloc[0]
    names = ["pooled"] + detector_suite.suite.tolist()
    auc = np.r_[pooled.roc_auc, detector_suite.roc_auc]
    low = np.r_[pooled.roc_ci_low, detector_suite.roc_ci_low]
    high = np.r_[pooled.roc_ci_high, detector_suite.roc_ci_high]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(auc, names, xerr=np.vstack((auc - low, high - auc)), fmt="o", capsize=3)
    ax.axvline(.5, color="black", ls="--"); ax.set_xlim(0, 1); ax.set_xlabel("ROC-AUC (95% CI)")
    ax.set_title("PRO observed-arm detector by suite")
    _save_figure(fig, "pro_detector_auc", output)

    curves = tables["pro_detector_roc_curves"]
    lookup = dict(zip(detector_suite.suite, detector_suite.roc_auc)); lookup["pooled"] = pooled.roc_auc
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, group in curves.groupby("suite"):
        ax.plot(group.fpr, group.tpr, lw=2 if name == "pooled" else 1.2,
                label=f"{name} ({lookup[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(xlim=(0, 1), ylim=(0, 1),
        xlabel="False positive rate", ylabel="True positive rate", title="PRO detector ROC curves")
    ax.legend(fontsize=7, loc="lower right"); _save_figure(fig, "pro_detector_roc_curves", output)

    dof = tables["pro_detector_per_dof"].sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(dof.score, dof.roc_auc)
    ax.axvline(.5, color="black", ls="--"); ax.set_xlim(0, 1); ax.set_xlabel("ROC-AUC")
    ax.set_title("PRO detector by action dimension")
    _save_figure(fig, "pro_detector_per_dof", output)

    scores = tables["pro_detector_score_distribution"].sort_values("fail")
    outcome_labels = np.where(scores.fail.astype(bool), "failure", "success")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(outcome_labels, scores["median"],
                yerr=np.vstack((scores["median"] - scores.q25, scores.q75 - scores["median"])),
                fmt="o", capsize=5)
    ax.set_ylabel("Observed episode uncertainty (median and IQR)")
    ax.set_title("PRO uncertainty by outcome")
    _save_figure(fig, "pro_uncertainty_by_outcome", output)

    for table_name, file_name, x_column, x_label in [
        ("pro_detector_time_profile", "pro_uncertainty_over_time",
         "episode_progress_bin", "Episode progress bin")]:
        frame = tables[table_name]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
        for ax, suite_name in zip(axes.ravel(), suites):
            group = frame[frame.suite == suite_name]
            for fail, label, color in [(0, "success", "#4C78A8"), (1, "failure", "#E45756")]:
                line = group[group.fail == fail]
                ax.plot(line[x_column], line["mean"], marker="o", label=label, color=color)
            ax.set_title(suite_name.replace("libero_", ""), fontsize=8); ax.set_xlabel(x_label)
        axes[0, 0].set_ylabel("Mean uncertainty"); axes[1, 0].set_ylabel("Mean uncertainty")
        axes[0, 2].legend(fontsize=7); _save_figure(fig, file_name, output)

    denoising_figures(tables, output, prefix="pro_")

    sweep = tables["pro_legacy_threshold_sweep"]
    for score_name, group in sweep.groupby("score_name"):
        delta = group.pivot(index="lower", columns="upper", values="delta_pp").sort_index().sort_index(axis=1)
        count = group.pivot(index="lower", columns="upper", values="n_refined").reindex_like(delta)
        extent = [delta.columns.min(), delta.columns.max(), delta.index.min(), delta.index.max()]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        image = axes[0].imshow(delta.to_numpy(), origin="lower", aspect="auto", cmap="RdYlGn",
                               vmin=-10, vmax=10, extent=extent)
        sample = axes[1].imshow(count.to_numpy(), origin="lower", aspect="auto", cmap="Blues", extent=extent)
        fig.colorbar(image, ax=axes[0], label="In-sample policy Δ SR (pp)")
        fig.colorbar(sample, ax=axes[1], label="Episodes refined")
        for ax in axes: ax.set(xlabel="Upper threshold", ylabel="Lower threshold")
        axes[0].set_title("Exploratory policy effect"); axes[1].set_title("Sample size")
        fig.suptitle(f"Legacy exploratory threshold sweep — {score_name}")
        _save_figure(fig, f"pro_legacy_threshold_{score_name.replace('+', '_')}", output)

    cv = tables["pro_threshold_cross_validation"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for score_name, group in cv.groupby("score_name"):
        ax.scatter([score_name] * len(group), group.delta_pp, alpha=.8)
        ax.errorbar(score_name, group.delta_pp.mean(), yerr=group.delta_pp.std(ddof=1), fmt="D", color="black")
    ax.axhline(0, color="black", lw=1); ax.set_ylabel("Held-out selective-policy Δ SR (pp)")
    ax.set_title("Cross-validated selective refinement")
    _save_figure(fig, "pro_threshold_cross_validated", output)

    transfer = tables["pro_standard_degradation"]
    if not transfer.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(transfer.delta_pp, transfer.base_suite,
                    xerr=np.vstack((transfer.delta_pp - transfer.delta_ci_low_pp,
                                    transfer.delta_ci_high_pp - transfer.delta_pp)), fmt="o", capsize=3)
        ax.axvline(0, color="black", lw=1); ax.set_xlabel("PRO minus standard observed SR (pp)")
        ax.set_title("Standard-to-PRO degradation (unpaired)")
        _save_figure(fig, "pro_standard_degradation", output)

    directional = tables.get("pro_geometry_directional", pd.DataFrame())
    if not directional.empty:
        labels = np.where(directional.success, "success", "failure")
        x = np.arange(len(directional)); width = .36
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, directional.parallel_variance_mean, width, label="parallel")
        ax.bar(x + width / 2, directional.lateral_variance_mean, width, label="lateral")
        ax.set_xticks(x, labels); ax.set_ylabel("Mean directional variance")
        ax.set_title("PRO observed-arm directional geometry"); ax.legend()
        _save_figure(fig, "pro_geometry_directional", output)


def write_pro_report(snapshot: Path, tables: dict[str, pd.DataFrame], availability: dict) -> Path:
    _clean_rendered_outputs(snapshot)
    for name, frame in tables.items():
        write_table(frame, name, snapshot)
    _write_availability(snapshot, availability)
    pro_figures(tables, snapshot)
    return snapshot
