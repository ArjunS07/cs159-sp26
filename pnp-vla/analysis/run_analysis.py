"""Regenerate analysis tables + figures from Supabase (replaces 03_analysis.ipynb).

Usage:
    python -m analysis.run_analysis --experiment slice-v1 [--out out/] [--no-latex]
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd

    from pnp.store import SupabaseStore
    from . import load, metrics, style

    style.set_style(use_latex=not args.no_latex)
    os.makedirs(args.out, exist_ok=True)
    tables_dir = os.path.join(args.out, "tables")
    figs_dir = os.path.join(args.out, "figures")
    os.makedirs(tables_dir, exist_ok=True)

    store = SupabaseStore()
    df = load.rollouts(store, args.experiment)
    if df.empty:
        print(f"no rollouts for experiment={args.experiment}")
        return

    def _save_table(t: pd.DataFrame, name: str):
        p = os.path.join(tables_dir, f"{name}.csv")
        t.to_csv(p, index=False); print("wrote", p)

    # A1 — success rate by method (+ suite)
    _save_table(metrics.success_rate(df, by=("method",)), "sr_by_method")
    _save_table(metrics.success_rate(df, by=("suite", "method")), "sr_by_suite_method")

    # A5 — uncertainty detector (stratified AUC) on the no-op uncertainty pass
    det_rows = []
    for method, g in df.groupby("method"):
        if g["u_mean_episode"].notna().sum() < 3:
            continue
        gg = g.dropna(subset=["u_mean_episode"])
        auc, k = metrics.stratified_auc(gg, "u_mean_episode")
        pooled = metrics.detector_metrics(gg["u_mean_episode"], gg["fail"])
        det_rows.append({"method": method, "stratified_auc": auc, "n_suites": k,
                         "pooled_roc_auc": pooled["roc_auc"], "pooled_pr_auc": pooled["pr_auc"]})
    if det_rows:
        _save_table(pd.DataFrame(det_rows), "detector_auc")

    # A2 — phase comparison (uncertainty no-op vs refinement variants)
    phase_rows = []
    for label, ra in [("refine_last", False), ("refine_avg", True)]:
        pc = metrics.phase_comparison(df, "pnp_uncertainty_only", "pnp_refinement", ra) \
            if (df["method"] == "pnp_refinement").any() else \
            metrics.phase_comparison(df, "pnp_uncertainty_only", "pnp_refinement_avg")
        pc["variant"] = label
        phase_rows.append(pc)
    _save_table(pd.DataFrame(phase_rows), "phase_comparison")

    # A5 figure — per-suite detector AUC bar
    steps = load.euler_steps(store, args.experiment)
    if not steps.empty:
        _save_table(metrics.per_dof_auc(steps), "per_dof_auc")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sr = metrics.success_rate(df, by=("method",)).sort_values("SR_pct")
    ax.barh(sr["method"], sr["SR_pct"],
            color=[style.method_color(m) for m in sr["method"]])
    ax.set_xlabel("Success rate (\\%)"); ax.set_title("Success rate by method")
    style.savefig(fig, "sr_by_method", figs_dir)

    print(f"\ndone -> {args.out}/  (tables/ + figures/)")


if __name__ == "__main__":
    main()
