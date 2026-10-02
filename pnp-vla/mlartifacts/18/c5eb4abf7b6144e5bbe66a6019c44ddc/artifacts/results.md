# Q-function prediction validation

Cached final predictions; original repeated validation, exploratory. Same episodes shared across models.

1,280 training episodes and 320 validation episodes, nine correlated candidates each: 11,520 training and 2,880 validation labels. These observations repeat across models, not additional independent data.

| Model | Split | Accuracy at .5 | Majority baseline | AUROC | AP | Brier | MC log loss | Pair accuracy | Stock / selected / expected random successes |
|---|---|---|---|---|---|---|---|---|---|
| pcp_frozen_late_mc | train | 0.875 | 0.605 | 0.941 | 0.955 | 0.0917 | 0.2980 | 0.503 | 832 / 785 / 774.78 |
| pcp_frozen_late_mc | validation | 0.876 | 0.619 | 0.931 | 0.949 | 0.0965 | 0.3281 | 0.562 | 209 / 205 / 198.11 |
| cnn_mc_continuation | train | 0.934 | 0.605 | 0.980 | 0.985 | 0.0494 | 0.1665 | 0.743 | 832 / 854 / 774.78 |
| cnn_mc_continuation | validation | 0.837 | 0.619 | 0.877 | 0.889 | 0.1353 | 0.6264 | 0.506 | 209 / 198 / 198.11 |
| td_n5_continuation | train | 0.898 | 0.605 | 0.947 | 0.958 | 0.0813 | 0.2778 | 0.503 | 832 / 772 / 774.78 |
| td_n5_continuation | validation | 0.887 | 0.619 | 0.936 | 0.952 | 0.0868 | 0.3025 | 0.512 | 209 / 200 / 198.11 |

Random selection samples one of nine candidates uniformly. Its success probability is the fraction successful at each root. Pairwise chance is .5; top-choice success is not 1/9 unless exactly one candidate succeeds.

Repeated-stock-score baseline, calibration deciles and paired root bootstrap intervals are in report.json. Strong global AUROC can reflect state difficulty; within-root ordering and the repeated-score ablation measure action information. No thresholds or gates were fitted.
