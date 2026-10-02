# Constant-Q Bellman baselines

Training-only success prevalence: 0.616927. Same trajectories, episode/trajectory weighting and terminal contract as the learned critics.

| Baseline | Split | One-step Bellman RMSE ± SE | Five-step Bellman RMSE ± SE | MC return RMSE ± SE |
|---|---|---:|---:|---:|
| constant_zero | train | 0.3004 ± 0.0043 | 0.6190 ± 0.0073 | 0.7854 ± 0.0080 |
| constant_zero | validation | 0.3075 ± 0.0090 | 0.6277 ± 0.0144 | 0.7991 ± 0.0158 |
| constant_one | train | 0.1685 ± 0.0040 | 0.3601 ± 0.0078 | 0.6189 ± 0.0102 |
| constant_one | validation | 0.1689 ± 0.0086 | 0.3577 ± 0.0164 | 0.6012 ± 0.0209 |
| train_prevalence | train | 0.1551 ± 0.0015 | 0.3249 ± 0.0024 | 0.4861 ± 0.0030 |
| train_prevalence | validation | 0.1573 ± 0.0032 | 0.3264 ± 0.0049 | 0.4809 ± 0.0061 |

Every constant-Q baseline has exactly zero residual away from reward or terminal boundaries. A low Bellman residual therefore cannot, by itself, demonstrate accurate outcomes or useful gradients. Constant-Q gradients are zero.
