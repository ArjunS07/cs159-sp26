# Training and held-out critic errors

Same complete cohort: 1,280 training episodes (11,520 outcomes) and 320 held-out validation episodes (2,880 outcomes). Nine candidates per episode are correlated. Held-out validation has been repeatedly inspected; results are exploratory. Cells show training / held-out.

| Model / endpoint | Brier ↓ | MC log loss ↓ | Classification error ↓ | AUROC ↑ | Action pair accuracy ↑ | Fresh gradient pair accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| pcp_frozen_late_mc | 0.0917 / 0.0965 | 0.2980 / 0.3281 | 0.125 / 0.124 | 0.941 / 0.931 | 0.503 / 0.562 | 0.505 / 0.574 |
| cnn_mc_continuation | 0.0494 / 0.1353 | 0.1665 / 0.6264 | 0.066 / 0.163 | 0.980 / 0.877 | 0.743 / 0.506 | 0.661 / 0.478 |
| td_n5_continuation | 0.0813 / 0.0868 | 0.2778 / 0.3025 | 0.102 / 0.113 | 0.947 / 0.936 | 0.503 / 0.512 | 0.508 / 0.521 |
| overnight_mc_long | 0.0961 / 0.0914 | 0.3192 / 0.3165 | 0.128 / 0.111 | 0.931 / 0.925 | 0.504 / 0.516 | 0.500 / 0.528 |
| overnight_mc_roots_contextdrop | 0.0905 / 0.0964 | 0.2947 / 0.3286 | 0.127 / 0.126 | 0.941 / 0.930 | 0.509 / 0.566 | 0.508 / 0.572 |
| overnight_mc_roots | 0.0918 / 0.0965 | 0.2981 / 0.3282 | 0.125 / 0.124 | 0.941 / 0.931 | 0.502 / 0.564 | 0.506 / 0.573 |
| overnight_td_long | 0.1421 / 0.1250 | 0.4626 / 0.4150 | 0.194 / 0.175 | 0.903 / 0.907 | 0.516 / 0.492 | 0.514 / 0.477 |
| overnight_td_n5 | 0.1068 / 0.1022 | 0.3555 / 0.3394 | 0.139 / 0.143 | 0.922 / 0.922 | 0.519 / 0.490 | 0.516 / 0.483 |
| width128_seed123/fixed_final | 0.0442 / 0.1321 | 0.1484 / 0.6573 | 0.060 / 0.160 | 0.984 / 0.885 | 0.789 / 0.524 | 0.681 / 0.498 |
| width128_seed123/early_selected | 0.1056 / 0.0964 | 0.3456 / 0.3308 | 0.141 / 0.119 | 0.917 / 0.914 | 0.500 / 0.500 | 0.500 / 0.500 |
| width128_seed314/fixed_final | 0.0469 / 0.1358 | 0.1565 / 0.6675 | 0.065 / 0.159 | 0.983 / 0.881 | 0.770 / 0.504 | 0.671 / 0.504 |
| width128_seed314/early_selected | 0.1041 / 0.0956 | 0.3407 / 0.3279 | 0.140 / 0.118 | 0.919 / 0.915 | 0.504 / 0.514 | 0.504 / 0.499 |
| width128_seed42/fixed_final | 0.0466 / 0.1366 | 0.1550 / 0.6616 | 0.064 / 0.169 | 0.983 / 0.880 | 0.771 / 0.520 | 0.670 / 0.514 |
| width128_seed42/early_selected | 0.1003 / 0.0964 | 0.3302 / 0.3298 | 0.131 / 0.120 | 0.921 / 0.916 | 0.524 / 0.517 | 0.499 / 0.497 |
| width32_seed123/fixed_final | 0.0726 / 0.1152 | 0.2410 / 0.4336 | 0.098 / 0.145 | 0.956 / 0.896 | 0.616 / 0.544 | 0.567 / 0.523 |
| width32_seed123/early_selected | 0.1008 / 0.0964 | 0.3311 / 0.3292 | 0.134 / 0.124 | 0.920 / 0.915 | 0.509 / 0.516 | 0.499 / 0.496 |
| width32_seed314/fixed_final | 0.0713 / 0.1170 | 0.2376 / 0.4465 | 0.094 / 0.149 | 0.957 / 0.890 | 0.585 / 0.515 | 0.550 / 0.521 |
| width32_seed314/early_selected | 0.1028 / 0.0967 | 0.3372 / 0.3296 | 0.138 / 0.123 | 0.918 / 0.914 | 0.506 / 0.546 | 0.502 / 0.522 |
| width32_seed42/fixed_final | 0.0711 / 0.1162 | 0.2346 / 0.4487 | 0.096 / 0.147 | 0.959 / 0.900 | 0.614 / 0.512 | 0.572 / 0.513 |
| width32_seed42/early_selected | 0.1027 / 0.0967 | 0.3373 / 0.3294 | 0.138 / 0.123 | 0.919 / 0.915 | 0.489 / 0.521 | 0.486 / 0.531 |
| width64_seed123/fixed_final | 0.0543 / 0.1210 | 0.1803 / 0.5160 | 0.074 / 0.147 | 0.977 / 0.894 | 0.703 / 0.503 | 0.605 / 0.524 |
| width64_seed123/early_selected | 0.1056 / 0.0964 | 0.3456 / 0.3308 | 0.141 / 0.119 | 0.917 / 0.914 | 0.500 / 0.500 | 0.500 / 0.500 |
| width64_seed314/fixed_final | 0.0526 / 0.1246 | 0.1745 / 0.5114 | 0.072 / 0.157 | 0.978 / 0.891 | 0.708 / 0.511 | 0.629 / 0.531 |
| width64_seed314/early_selected | 0.1015 / 0.0961 | 0.3337 / 0.3283 | 0.134 / 0.122 | 0.919 / 0.915 | 0.514 / 0.533 | 0.505 / 0.507 |
| width64_seed42/fixed_final | 0.0536 / 0.1182 | 0.1788 / 0.5215 | 0.072 / 0.143 | 0.977 / 0.894 | 0.712 / 0.486 | 0.620 / 0.495 |
| width64_seed42/early_selected | 0.1056 / 0.0964 | 0.3456 / 0.3308 | 0.141 / 0.119 | 0.917 / 0.914 | 0.500 / 0.500 | 0.500 / 0.500 |
| anchored_initial/transformer | 0.0946 / 0.1004 | 0.3137 / 0.3465 | 0.124 / 0.125 | 0.926 / 0.912 | 0.521 / 0.504 | 0.507 / 0.498 |
| anchored_initial/temporal_cnn | 0.0921 / 0.0980 | 0.3054 / 0.3399 | 0.122 / 0.117 | 0.930 / 0.911 | 0.547 / 0.509 | 0.527 / 0.503 |
| original_scalar_mc_update2000 | 0.1057 / 0.0967 | 0.3458 / 0.3314 | 0.141 / 0.119 | 0.917 / 0.914 | 0.513 / 0.512 | 0.513 / 0.506 |
| original_scalar_td_update2000 | 0.1819 / 0.1766 | 0.5469 / 0.5433 | 0.272 / 0.246 | 0.827 / 0.824 | 0.496 / 0.516 | 0.496 / 0.530 |

Brier is squared error against recorded eventual binary success. MC log loss evaluates those same outcomes for every model, including TD-trained models; neither is Bellman error. Classification uses the fixed probability threshold 0.5. Pair accuracy is conditional on mixed outcomes at the same state; gradient ordering uses fresh candidates only. Chance ordering is 0.5.

Held-out Bellman residuals are not yet available: the existing transition cache contains training episodes only. Training Bellman residuals have not yet been evaluated in this report. Do not substitute training minibatch soft-target BCE for either metric.

The model families and training data differ, so these are descriptive model comparisons rather than a matched MC-versus-TD experiment.
