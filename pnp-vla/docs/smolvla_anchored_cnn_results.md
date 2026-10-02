# Anchored action critic comparison — September 30, 2026

Both arms completed their fixed 2,000-update MC/BCE budgets on the existing 1,280/320 root split. The same original MC baseline and its 128-dimensional stock readout were frozen. Only the centered nonlinear action residual was trained. Transformer and CNN have 398,720 and 397,568 trainable parameters respectively. Average exposure was 6.25 presentations per training root, versus 250 in the earlier 32-root fitting diagnostic.

| Fixed final metric | Anchored Transformer | Anchored temporal CNN |
| --- | ---: | ---: |
| Training selected successes, stock 832/1,280 | 793/1,280 | 800/1,280 |
| Training fresh-only pair accuracy | 0.4985 | 0.5324 |
| Validation selected successes, stock 209/320 | 200/320 | 199/320 |
| Validation rescues / spoils | 4 / 13 | 7 / 17 |
| Validation fresh-only pair accuracy | 0.4977 | 0.4927 |
| Validation Brier | 0.1004 | 0.0980 |
| Validation non-stock BCE | 0.3565 | 0.3491 |
| Validation fresh gradient-projection pair accuracy | 0.4982 | 0.5030 |

Repeating the frozen stock probability gives validation BCE 0.3308 and non-stock BCE 0.3388. The stock row contributes a constant 0.02963 to total candidate BCE; its predictions cannot change in either arm. The candidate-set oracle succeeds on 226/320 roots.

Astra's paired root bootstrap (10,000 draws, seed 42930) finds a modest CNN training fresh-pair gain of 0.03387, interval [0.00348, 0.06354]. On validation its fresh-pair difference is -0.00505, interval [-0.07511, 0.06376], and selection differs by -1/320. This is no demonstrated held-out architecture benefit. Repeated validation use and a single training seed make these exploratory comparisons.

Both residuals are active. Median normalized validation gradient norms are 23.1 for the Transformer and 30.4 for the CNN, but gradient-projection ordering remains near chance. Magnitude does not establish a useful PCP direction; projections against recorded candidate outcomes are not true gradient labels. No simulator intervention was run in this experiment.

Keep stock as the preferred selector on current evidence. This experiment does not reject convolution generally, prove a data ceiling, or establish convergence. Its limited per-root exposure and frozen stock representation must be considered before further architectural conclusions. No additional training was launched.

## Logs and reports

- [Transformer MLflow run](http://127.0.0.1:5001/#/experiments/15/runs/2d78ddf7ebe54e38acd321378c1a1859)
- [CNN MLflow run](http://127.0.0.1:5001/#/experiments/15/runs/5afc548f4dd4430aa6c82be0b5bdf433)
- [Transformer full report](/Users/arjunsharma/pnp-vla-runs/checkpoints/smolvla_anchored_cnn_v1/671b5b211099997fc83d1277/transformer/final_report.json)
- [CNN full report](/Users/arjunsharma/pnp-vla-runs/checkpoints/smolvla_anchored_cnn_v1/671b5b211099997fc83d1277/temporal_cnn/final_report.json)

Each full report includes metrics and gradient diagnostics over all 1,600 roots. Both final checkpoints contain optimizer states and histories; see [run instructions](smolvla_anchored_cnn_run.md) for resumption.
