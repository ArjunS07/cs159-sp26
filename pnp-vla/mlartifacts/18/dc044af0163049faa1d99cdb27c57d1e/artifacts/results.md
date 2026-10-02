# Common critic audit

Existing reused validation: 320 episodes, nine correlated candidate outcomes each. No new policy rollouts. Training results and full calibration/gradient diagnostics are in report.json.

| Model / endpoint | Accuracy | AUROC | Brier ↓ | MC log loss ↓ | Fresh gradient pair | Selected / oracle |
|---|---:|---:|---:|---:|---:|---:|
| pcp_frozen_late_mc | 0.876 | 0.931 | 0.0965 | 0.3281 | 0.574 | 205 / 226 |
| cnn_mc_continuation | 0.837 | 0.877 | 0.1353 | 0.6264 | 0.478 | 198 / 226 |
| td_n5_continuation | 0.887 | 0.936 | 0.0868 | 0.3025 | 0.521 | 200 / 226 |
| overnight_mc_long | 0.889 | 0.925 | 0.0914 | 0.3165 | 0.528 | 201 / 226 |
| overnight_mc_roots_contextdrop | 0.874 | 0.930 | 0.0964 | 0.3286 | 0.572 | 205 / 226 |
| overnight_mc_roots | 0.876 | 0.931 | 0.0965 | 0.3282 | 0.573 | 205 / 226 |
| overnight_td_long | 0.825 | 0.907 | 0.1250 | 0.4150 | 0.477 | 202 / 226 |
| overnight_td_n5 | 0.857 | 0.922 | 0.1022 | 0.3394 | 0.483 | 199 / 226 |
| width128_seed123/fixed_final | 0.840 | 0.885 | 0.1321 | 0.6573 | 0.498 | 199 / 226 |
| width128_seed123/early_selected | 0.881 | 0.914 | 0.0964 | 0.3308 | 0.500 | 209 / 226 |
| width128_seed314/fixed_final | 0.841 | 0.881 | 0.1358 | 0.6675 | 0.504 | 201 / 226 |
| width128_seed314/early_selected | 0.882 | 0.915 | 0.0956 | 0.3279 | 0.499 | 195 / 226 |
| width128_seed42/fixed_final | 0.831 | 0.880 | 0.1366 | 0.6616 | 0.514 | 201 / 226 |
| width128_seed42/early_selected | 0.880 | 0.916 | 0.0964 | 0.3298 | 0.497 | 199 / 226 |
| width32_seed123/fixed_final | 0.855 | 0.896 | 0.1152 | 0.4336 | 0.523 | 206 / 226 |
| width32_seed123/early_selected | 0.876 | 0.915 | 0.0964 | 0.3292 | 0.496 | 203 / 226 |
| width32_seed314/fixed_final | 0.851 | 0.890 | 0.1170 | 0.4465 | 0.521 | 203 / 226 |
| width32_seed314/early_selected | 0.877 | 0.914 | 0.0967 | 0.3296 | 0.522 | 197 / 226 |
| width32_seed42/fixed_final | 0.853 | 0.900 | 0.1162 | 0.4487 | 0.513 | 200 / 226 |
| width32_seed42/early_selected | 0.877 | 0.915 | 0.0967 | 0.3294 | 0.531 | 204 / 226 |
| width64_seed123/fixed_final | 0.853 | 0.894 | 0.1210 | 0.5160 | 0.524 | 197 / 226 |
| width64_seed123/early_selected | 0.881 | 0.914 | 0.0964 | 0.3308 | 0.500 | 209 / 226 |
| width64_seed314/fixed_final | 0.843 | 0.891 | 0.1246 | 0.5114 | 0.531 | 200 / 226 |
| width64_seed314/early_selected | 0.878 | 0.915 | 0.0961 | 0.3283 | 0.507 | 197 / 226 |
| width64_seed42/fixed_final | 0.857 | 0.894 | 0.1182 | 0.5215 | 0.495 | 198 / 226 |
| width64_seed42/early_selected | 0.881 | 0.914 | 0.0964 | 0.3308 | 0.500 | 209 / 226 |
| anchored_initial/transformer | 0.875 | 0.912 | 0.1004 | 0.3465 | 0.498 | 200 / 226 |
| anchored_initial/temporal_cnn | 0.883 | 0.911 | 0.0980 | 0.3399 | 0.503 | 199 / 226 |
| original_scalar_mc_update2000 | 0.881 | 0.914 | 0.0967 | 0.3314 | 0.506 | 199 / 226 |
| original_scalar_td_update2000 | 0.754 | 0.824 | 0.1766 | 0.5433 | 0.530 | 202 / 226 |

Stock: 209/320; expected uniform random: 198.11/320; oracle: 226/320. Fresh gradient ordering uses only the 81 fresh-mixed validation roots; chance ordering is .5.

## Coverage limitations

Training and held-out Bellman errors are now available for the three main critics in bellman_results.md, with constant-Q controls in bellman_baselines.md. The separate held-out evaluation cache is excluded from training. Bellman evaluation remains pending for the other endpoints. Cached scalar derivatives have observational alignment evidence but no new numerical derivative test. CNN endpoints have an eight-root numerical probe and 10×7 gradient summaries; those do not establish useful intervention directions.

- legacy_distributional_ranking_and_conditioning_variants: Require explicit target/input adapters and leakage/provenance review before probability comparisons.
