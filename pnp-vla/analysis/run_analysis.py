"""Regenerate every analysis table + figure from Supabase (replaces 03_analysis.ipynb).

Usage:
    python -m analysis.run_analysis --experiment <label> [--out figures/]

Emits: phase_comparison, transition_chart, threshold_2d_heatmap, roc_curves, per_dim_auc,
pca_isotropy, crossmodel_detector_auc (+ CSV tables). Applies analysis.style.set_style().

TODO(phase 6): implement CLI + orchestration over analysis.load / analysis.metrics.
"""


def main() -> None:
    raise NotImplementedError("phase 6")


if __name__ == "__main__":
    main()
