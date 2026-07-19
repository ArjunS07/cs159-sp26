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


def write_report(experiment: str, snapshot: Path, validation: dict,
                 tables: dict[str, pd.DataFrame], availability: dict) -> Path:
    for name, frame in tables.items():
        write_table(frame, name, snapshot)
    delta = legacy_delta(tables); write_table(delta, "legacy_delta", snapshot)
    (snapshot / "availability.json").write_text(json.dumps(availability, indent=2, sort_keys=True) + "\n")
    summary = markdown_summary(experiment, snapshot.name, validation, tables, availability)
    (snapshot / "findings.md").write_text(summary)
    success_figure(tables["success_all_identity"], snapshot)
    return snapshot
