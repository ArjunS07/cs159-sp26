# Training and held-out Bellman error

Recorded-policy SARSA, gamma=1, n=1 and n=5; frozen online self-bootstrap

Equal trajectories within episode, then equal episodes; standard errors across episodes. held-out trajectories of repeatedly inspected validation episodes, exploratory

| Model | Split | Episodes / trajectories / transitions | One-step RMSE ± SE | Five-step RMSE ± SE | MC return RMSE ± SE |
|---|---|---:|---:|---:|---:|
| pcp_frozen_late_mc | train | 1280 / 3840 / 58695 | 0.1315 ± 0.0028 | 0.1919 ± 0.0037 | 0.2522 ± 0.0048 |
| pcp_frozen_late_mc | validation | 320 / 960 / 14553 | 0.1348 ± 0.0068 | 0.2001 ± 0.0081 | 0.2696 ± 0.0111 |
| cnn_mc_continuation | train | 1280 / 3840 / 58695 | 0.1303 ± 0.0028 | 0.1877 ± 0.0034 | 0.2546 ± 0.0045 |
| cnn_mc_continuation | validation | 320 / 960 / 14553 | 0.1402 ± 0.0069 | 0.2065 ± 0.0080 | 0.2771 ± 0.0103 |
| td_n5_continuation | train | 1280 / 3840 / 58695 | 0.1001 ± 0.0025 | 0.1617 ± 0.0032 | 0.2217 ± 0.0045 |
| td_n5_continuation | validation | 320 / 960 / 14553 | 0.1165 ± 0.0052 | 0.1837 ± 0.0076 | 0.2503 ± 0.0107 |

Sampled-policy transition residuals include future-action/transition variability. Low nonterminal error alone does not validate outcome prediction. This is self-bootstrap consistency, not TD training loss against the EMA target. CNN reference is the recorded stock action at ordinary continuation states and the original stock root action at fresh branch roots. Terminal and nonterminal summaries are in bellman_report.json.
