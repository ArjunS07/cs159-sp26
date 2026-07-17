"""Analysis metrics, ported from pnp_pro_analysis_final.ipynb (A1-A5 + geometric B).

- A1 load & summarise (per-suite/method SR)
- A2 phase comparison (uncertainty no-op vs refinement, paired; delta pp)
- A3 paired transition analysis (F->S / S->F)
- A4 threshold-window sweep (2D SR-change + N-episode heatmaps, dim-subset variants)
- A5 detector_metrics (ROC/PR-AUC, Spearman, best-F1) + stratified_auc (per-suite averaged)
- B geometry: _sarle_bc, per-DOF localization, dim-subset sweep, PC1 vs Marchenko-Pastur
- PCP 3-way SR by lambda; cross-model detector-AUC (from 03_analysis.ipynb)

Variant selection is a filter (method + refine_average) — no DB merge, no REFINE_VARIANT tag.

TODO(phase 6): implement.
"""
