# Stock-action gradients by successful and unsuccessful rollout

Gradients evaluated at the stock action; stratified by the eventual outcome of that stock rollout. Observational association, not correction success.

Norms are L2 probability gradients in standardized action coordinates. Each cell is mean ± episode-level SE; median and 95th percentile are in the JSON.

| Model | Split | Successful mean ± SE (n) | Unsuccessful mean ± SE (n) |
|---|---|---:|---:|
| pcp_frozen_late_mc | train | 0.039911 ± 0.002359 (832) | 0.043662 ± 0.003625 (448) |
| pcp_frozen_late_mc | validation | 0.041433 ± 0.004164 (209) | 0.056234 ± 0.010792 (111) |
| cnn_mc_continuation | train | 61.213260 ± 2.318778 (832) | 66.399274 ± 4.267794 (448) |
| cnn_mc_continuation | validation | 63.000969 ± 4.572666 (209) | 57.922387 ± 8.383015 (111) |
| td_n5_continuation | train | 0.026415 ± 0.001586 (832) | 0.032583 ± 0.002859 (448) |
| td_n5_continuation | validation | 0.031154 ± 0.003470 (209) | 0.046872 ± 0.009683 (111) |
