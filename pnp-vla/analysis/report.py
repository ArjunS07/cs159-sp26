"""Deterministic machine-readable outputs, findings summary, and figures."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_table(frame: pd.DataFrame, name: str, output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{name}.csv", index=False)
    frame.to_parquet(tables / f"{name}.parquet", index=False)


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


def _pct(value) -> str:
    return f"{100 * value:.2f}%"


def markdown_summary(experiment: str, snapshot_id: str, validation: dict,
                     tables: dict[str, pd.DataFrame], availability: dict) -> str:
    all_sr = tables["success_all_identity"].sort_values("condition_label")
    paired = tables["paired_comparisons"].sort_values(["cohort", "condition_label"])
    detector = tables["detector_summary"].set_index("estimate_scope")
    lines = [f"# Analysis findings: {experiment}", "",
             f"Snapshot `{snapshot_id}` passed validation with {validation['n_rollouts']:,} rollouts, "
             f"{validation['n_identities']} identities, and {validation['n_configurations']} configurations.", "",
             "## Balanced all-identity success", ""]
    for row in all_sr.to_dict("records"):
        lines.append(f"- {row['condition_label']}: {row['successes']}/{row['n']} = {_pct(row['sr'])} "
                     f"(Wilson 95% CI {_pct(row['ci_low'])}–{_pct(row['ci_high'])}).")
    lines += ["", "## Paired effects", ""]
    for row in paired.to_dict("records"):
        lines.append(f"- {row['cohort']}, {row['condition_label']}: Δ {row['delta_pp']:+.2f} pp; "
                     f"F→S {row['F_to_S']}, S→F {row['S_to_F']}, n={row['n']}; "
                     f"paired 95% CI {row['delta_ci_low_pp']:+.2f} to {row['delta_ci_high_pp']:+.2f} pp; "
                     f"two-sided discordant-pair p={row['p_raw']:.4g}.")
    pooled = detector.loc["pooled"]
    macro = detector.loc["macro_suite"]
    lines += ["", "## Prospective observed-arm detector", "",
              f"The observed/no-op arm has n={int(pooled['n'])} and {int(pooled['failures'])} failures. "
              f"Pooled ROC-AUC is {pooled['roc_auc']:.4f} (identity-bootstrap 95% CI "
              f"{pooled['roc_ci_low']:.4f}–{pooled['roc_ci_high']:.4f}); PR-AUC is {pooled['pr_auc']:.4f}. "
              f"The macro mean of per-suite ROC-AUCs is {macro['roc_auc']:.4f}.", "",
              "Refinement uncertainty is post-treatment and is excluded from these detector estimates.", "",
              "## Availability and caveats", ""]
    for name, state in availability.items():
        lines.append(f"- {name}: `{state['status']}` — {state['reason']}")
    lines += ["", "Point estimates and p-values are reported without claims of statistical significance. "
              "Eight full-ablation schedule comparisons use Holm-adjusted p-values in the machine-readable table.", ""]
    return "\n".join(lines)


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

    # Outcome-conditioned score summaries (raw refinement scores are intentionally absent).
    frame = tables["detector_score_distribution"].sort_values("fail")
    labels = ["success" if not bool(x) else "failure" for x in frame.fail]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(labels, frame["mean"], yerr=frame["std"], fmt="o", capsize=5, label="mean ± SD")
    ax.scatter(labels, frame["median"], marker="D", label="median")
    ax.set_ylabel("Mean episode uncertainty"); ax.set_title("Observed-arm uncertainty by outcome")
    ax.legend(); _save_figure(fig, "detector_score_by_outcome", output)

    # Reliability is descriptive because uncertainty is a ranking score, not a fitted probability.
    frame = tables["detector_reliability"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frame.mean_score, frame.observed_failure_rate, marker="o")
    for row in frame.itertuples(): ax.annotate(f"n={row.n}", (row.mean_score, row.observed_failure_rate), fontsize=7)
    ax.set(xlabel="Mean uncertainty in bin", ylabel="Observed failure rate", ylim=(0, 1),
           title="Observed-arm reliability by uncertainty decile")
    _save_figure(fig, "detector_reliability", output)

    frame = tables["detector_risk_coverage"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(100 * frame.coverage, 100 * frame.risk, marker="o")
    ax.set(xlabel="Coverage retained (%)", ylabel="Failure risk (%)", xlim=(0, 100),
           title="Observed-arm risk–coverage curve")
    _save_figure(fig, "detector_risk_coverage", output)

    # Cross-validated threshold results; thresholds were selected on training folds only.
    frame = tables["detector_threshold_cross_validation"]
    metrics = ["precision", "recall", "specificity", "accuracy"]
    means, errors = frame[metrics].mean(), frame[metrics].std(ddof=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics, means, yerr=errors, capsize=4, color="#F58518")
    ax.set_ylim(0, 1); ax.set_ylabel("Held-out fold metric (mean ± SD)")
    ax.set_title("Cross-validated detector threshold performance")
    _save_figure(fig, "detector_threshold_cv", output)

    # Observed behavioral baseline by suite, with Wilson intervals and 0-100 axis.
    frame = tables["observed_success_by_suite"].sort_values("sr")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([f"{s} (n={n})" for s, n in zip(frame.suite, frame.n)], 100 * frame.sr,
            xerr=np.vstack((100 * (frame.sr - frame.ci_low), 100 * (frame.ci_high - frame.sr))),
            capsize=3, color="#4C78A8")
    ax.set(xlim=(0, 100), xlabel="Observed/no-op success rate (%)", title="Observed baseline by suite")
    _save_figure(fig, "observed_success_by_suite", output)

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
    for name, frame in tables.items():
        write_table(frame, name, snapshot)
    delta = legacy_delta(tables); write_table(delta, "legacy_delta", snapshot)
    (snapshot / "availability.json").write_text(json.dumps(availability, indent=2, sort_keys=True) + "\n")
    summary = markdown_summary(experiment, snapshot.name, validation, tables, availability)
    (snapshot / "findings.md").write_text(summary)
    success_figure(tables["success_all_identity"], snapshot)
    publication_figures(tables, snapshot)
    return snapshot
