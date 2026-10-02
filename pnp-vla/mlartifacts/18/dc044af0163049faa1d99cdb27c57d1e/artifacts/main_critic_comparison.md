# Main critic comparison: training history, errors and gradients

All models use cached features from the pretrained, frozen SmolVLA policy. The original scalar critics were trained from random weights, then later stages reused those critic checkpoints. These are different architectures, data exposures and update budgets, not a matched MC-versus-TD objective comparison.

| Name | Training history | What is frozen? |
|---|---|---|
| Frozen MC (PCP) | Original scalar MC: 2,000 updates; additional action pathway plus 4,000 MC updates mixing root-candidate and trajectory batches. | Entire final 6,000-update critic fixed for PCP evaluation. |
| Continued CNN MC | Original 2,000-update MC baseline frozen; nonlinear CNN residual trained 2,000 updates, then continued 8,000 more (10,000 residual updates). | Original baseline and stock reference readout; CNN residual trained. |
| Continued five-step TD | Separate scalar TD: 2,000 initial one-step updates, then 4,000 five-step updates and 8,000 more (14,000 total). | Entire critic fixed during evaluation; trained with a moving target network. |

One Bellman transition is one action chunk of up to ten environment steps. Five-step targets span up to five chunks (50 steps), stopping at termination.

## Episode-level standard errors and gradient magnitudes

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

## Training and held-out Bellman error

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

## Constant-Q Bellman baselines

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


## Interpretation

Continued TD has the lowest measured prediction and Bellman errors among these three endpoints, but weak same-state gradient ordering. The CNN has much larger stock-action gradients and a substantial root-candidate prediction generalization gap. Its residual deliberately scales action displacements by 100; large derivatives are not evidence of helpful correction directions. Frozen MC has modest gradient alignment on only 81 fresh-mixed validation roots. All comparisons remain exploratory. No further PCP evaluation was launched.

Bellman errors use all 58,695 training and 14,553 validation transitions, grouped into 1,280 / 320 episodes with three trajectories each. Gradient magnitudes use all 1,280 / 320 root states; ordering uses only 361 / 81 fresh-mixed roots. Episode-level standard errors condition on fixed checkpoints and do not measure seed or unseen-task uncertainty.
