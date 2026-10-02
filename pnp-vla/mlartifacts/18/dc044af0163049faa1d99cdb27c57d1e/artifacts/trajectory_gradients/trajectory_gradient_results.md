# Action gradients across recorded trajectories

All chunks in the existing three-trajectory-per-root caches (stored source, fresh seed 1, fresh seed 5): training 1,280 roots and held-out 320 roots. Other candidate trajectories are not in these caches.

Chunk-weighted means; standard errors clustered by originating episode/root, including correlated trajectories. Conditional on fixed checkpoints. Repeatedly examined held-out data remain exploratory.

Gradients are evaluated at each recorded full action proposal and labeled by its trajectory's eventual outcome. CNN references are held fixed; original stock root action at fresh branch roots, recorded action elsewhere. Known-invalid action slots have zero gradients. No future outcome mask is applied.

| Model | Split | Outcome | Chunks / trajectories / episode clusters | Probability gradient L2, standardized ± SE | Logit gradient L2, standardized ± SE | Probability gradient L2, native policy units ± SE |
|---|---|---|---:|---:|---:|---:|
| pcp_frozen_late_mc | train | success | 23696 / 2369 / 874 | 0.046028 ± 0.001838 | 0.513035 ± 0.008615 | 0.049951 ± 0.002033 |
| pcp_frozen_late_mc | train | failure | 34999 / 1471 / 604 | 0.042769 ± 0.001391 | 0.539351 ± 0.009017 | 0.047112 ± 0.001546 |
| pcp_frozen_late_mc | validation | success | 6265 / 613 / 222 | 0.047537 ± 0.003836 | 0.543886 ± 0.020564 | 0.051679 ± 0.004289 |
| pcp_frozen_late_mc | validation | failure | 8288 / 347 / 139 | 0.042440 ± 0.002623 | 0.562198 ± 0.017875 | 0.046689 ± 0.002884 |
| cnn_mc_continuation | train | success | 23696 / 2369 / 874 | 63.240916 ± 1.940261 | 648.197786 ± 5.481691 | 72.263834 ± 2.248232 |
| cnn_mc_continuation | train | failure | 34999 / 1471 / 604 | 50.360821 ± 1.415699 | 443.703923 ± 6.506704 | 57.990110 ± 1.639158 |
| cnn_mc_continuation | validation | success | 6265 / 613 / 222 | 63.139636 ± 3.706418 | 636.130224 ± 10.409546 | 72.080328 ± 4.301816 |
| cnn_mc_continuation | validation | failure | 8288 / 347 / 139 | 51.614609 ± 2.901885 | 454.067102 ± 13.098753 | 59.405719 ± 3.356429 |
| td_n5_continuation | train | success | 23696 / 2369 / 874 | 0.028589 ± 0.001179 | 0.401286 ± 0.006344 | 0.031683 ± 0.001327 |
| td_n5_continuation | train | failure | 34999 / 1471 / 604 | 0.026555 ± 0.000963 | 0.390419 ± 0.007562 | 0.029509 ± 0.001076 |
| td_n5_continuation | validation | success | 6265 / 613 / 222 | 0.033339 ± 0.002769 | 0.423239 ± 0.014953 | 0.036999 ± 0.003122 |
| td_n5_continuation | validation | failure | 8288 / 347 / 139 | 0.027475 ± 0.002159 | 0.411287 ± 0.017606 | 0.030412 ± 0.002373 |

Full signed and absolute 10×7 mean gradient arrays, their clustered standard errors, action-dimension means, and median/p95 norms are in trajectory_gradient_report.json. Individual raw logit gradient arrays and predicted probabilities are preserved in trajectory_gradients/. Signed coordinates can cancel across states; magnitude and direction usefulness are separate measurements. Chunk counts are not independent episode counts. These are recorded-policy trajectories, not a repeated-PCP rollout evaluation.

Action coordinates are native policy outputs, not physical robot units. Full recorded proposals include terminal tails that may not have been executed. For CNN, the residual is zero at candidate==reference while the partial derivative holding the reference fixed can remain large.
