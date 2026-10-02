# Episode-level standard errors and gradient magnitudes

episode/root; nine candidates clustered within each root. Conditional on these trained checkpoints; not training-seed uncertainty. Reused validation is exploratory.

| Model | Split | Episodes / outcomes | Brier ± SE | MC log loss ± SE | Classification error ± SE | AUROC ± SE | Gradient ordering ± SE (mixed episodes) |
|---|---|---:|---:|---:|---:|---:|---:|
| pcp_frozen_late_mc | train | 1280 / 11520 | 0.0917 ± 0.0046 | 0.2980 ± 0.0134 | 0.1253 ± 0.0070 | 0.9410 ± 0.0047 | 0.5052 ± 0.0145 (361) |
| pcp_frozen_late_mc | validation | 320 / 2880 | 0.0965 ± 0.0099 | 0.3281 ± 0.0314 | 0.1243 ± 0.0144 | 0.9306 ± 0.0108 | 0.5743 ± 0.0280 (81) |
| cnn_mc_continuation | train | 1280 / 11520 | 0.0494 ± 0.0023 | 0.1665 ± 0.0067 | 0.0655 ± 0.0034 | 0.9798 ± 0.0018 | 0.6614 ± 0.0136 (361) |
| cnn_mc_continuation | validation | 320 / 2880 | 0.1353 ± 0.0123 | 0.6264 ± 0.0698 | 0.1628 ± 0.0150 | 0.8774 ± 0.0165 | 0.4778 ± 0.0304 (81) |
| td_n5_continuation | train | 1280 / 11520 | 0.0813 ± 0.0041 | 0.2778 ± 0.0130 | 0.1020 ± 0.0059 | 0.9469 ± 0.0043 | 0.5080 ± 0.0142 (361) |
| td_n5_continuation | validation | 320 / 2880 | 0.0868 ± 0.0092 | 0.3025 ± 0.0296 | 0.1132 ± 0.0134 | 0.9357 ± 0.0105 | 0.5214 ± 0.0350 (81) |

## Gradient magnitude

| Model | Split | Logit gradient L2 mean ± SE | Probability gradient L2 mean ± SE | Probability gradient median / p95 |
|---|---|---:|---:|---:|
| pcp_frozen_late_mc | train | 0.550826 ± 0.011554 | 0.041224 ± 0.001990 | 0.016093 / 0.177008 |
| pcp_frozen_late_mc | validation | 0.588584 ± 0.023543 | 0.046567 ± 0.004633 | 0.016360 / 0.192260 |
| cnn_mc_continuation | train | 627.134292 ± 8.741706 | 63.028365 ± 2.122141 | 34.526484 / 220.158276 |
| cnn_mc_continuation | validation | 608.582472 ± 17.115746 | 61.239336 ± 4.162753 | 33.069081 / 229.513311 |
| td_n5_continuation | train | 0.440059 ± 0.010240 | 0.028574 ± 0.001439 | 0.008601 / 0.128412 |
| td_n5_continuation | validation | 0.487082 ± 0.025447 | 0.036606 ± 0.004064 | 0.008843 / 0.167446 |

Magnitudes use standardized action coordinates (training action standard deviations), not raw action units or the norm of a normalized PCP correction. Probability gradients include the sigmoid factor. Large gradients need not point toward better actions. Every magnitude summary uses all episodes in its split, not only mixed episodes.
