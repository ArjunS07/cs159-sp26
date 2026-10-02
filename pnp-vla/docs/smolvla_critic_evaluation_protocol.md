# SmolVLA critic evaluation before further PCP

Status: offline evaluation authorized; broader PCP deferred. No training, new paid
rollouts, checkpoint promotion, fitted evaluation thresholds, or objective changes.

## Questions and evidence

| Question | Primary evidence | Interpretation / baseline |
|---|---|---|
| Does Q predict eventual success? | Episode-weighted Brier and MC log loss; AUROC/AP; accuracy, balanced accuracy and both class recalls at fixed 0.5 | Compare train-only constant success prevalence, majority class, and the model's stock score repeated across candidates. Binary rollout outcomes are noisy observations of expected success, not exact Q labels. |
| Are probabilities calibrated? | Reliability diagram with fixed probability deciles, counts, prediction/outcome means, Brier decomposition and calibration intercept/slope | Descriptive only; do not fit a calibrator on the reporting set. ECE depends on binning and is secondary. Report overconfident failures explicitly. |
| Does Q use action information usefully? | Within-root success/failure pair ordering, all nine and fresh-only; argmax success; rescues/spoils; oracle regret | Pair chance is 0.5. Uniform random success equals each root's successful-candidate fraction. Stock, uniform random and oracle are distinct baselines. Repeated-stock scores are an action-information ablation, not an independently trained V baseline. |
| Is Q temporally consistent? | Recorded-policy one-step and five-step residual mean, MAE, RMSE, terminal/nonterminal breakdown, residual quantiles | Use success reward, gamma=1, correct terminal masks, and recorded next actions (SARSA), never a max over candidates. Compare both frozen online self-bootstrap and, for TD models, saved EMA-target bootstrap. Apply the same evaluation operator to MC and TD. |
| Does low Bellman error reflect useful prediction? | Bellman results alongside MC return error, constant-zero/constant-one/train-prevalence baselines, terminal accuracy and horizon breakdown | A near-constant predictor can have tiny nonterminal error and still be wrong about outcomes. Sample residual MSE includes transition/policy noise; it is not automatically the squared expected Bellman residual. Soft-target BCE is not a Bellman RMSE. |
| Are derivatives numerically correct? | Autograd versus central finite differences along matched directions at three normalized epsilons; finite values; repeatability | Check probability and logit gradients separately, hold context/reference/masks fixed, and run in eval mode. Near-zero derivatives require absolute-error checks; relative error alone is unstable. This establishes numerical correctness, not useful directions. |
| Do gradients point toward better observed actions? | Per-root gradient dot product with each observed candidate displacement; success/failure pair ordering; rescue-positive/spoil-negative fractions | Use standardized action coordinates and the same root as the expansion point. Report all-nine and fresh-only. Existing branch displacements test first-order alignment, not the success of an actual gradient intervention. |
| Are gradients stable and local? | Gradient norm quantiles, probability saturation, local Q changes versus linear predictions, cosine/directional agreement under small action changes and across seeds | Include signed changes and curvature at predefined radii. Report norm and cosine together: agreement between tiny gradients is not useful evidence. Seed variability is available for the CNN matrix; other models are single-seed exploratory results. |
| Where do models fail? | Predefined task/suite, early/middle/late stage, horizon, stock-success/failure, candidate type and uncertainty breakdowns | Stage/uncertainty groups must be defined without candidate outcomes. Failure-conditioned summaries are descriptive and never used as the primary cohort. Include sample sizes and missing groups. |
| Do scores transfer beyond the familiar cohort? | Frozen-model predictions on newly collected, outcome-unfiltered episodes at predefined stages | Current validation has been repeatedly inspected. New states from familiar tasks are not unseen-task generalization. Check episode/state provenance against every model's training data. |
| Will the gradient help a policy? | Later paired zero/ascent/descent/matched-random rollout comparison and repeated-PCP full episodes | Requires new simulator outcomes. No offline score, Bellman error or finite-difference test can substitute for this. Defer until the offline audit is reviewed. |

## Cohorts, independence and uncertainty

1. Existing complete root cohort: 1,280 training roots and 320 validation roots,
   nine candidates each (11,520 / 2,880 outcome labels). Use every root, including
   unanimous successes and failures. Mixed-root ordering is a separately labelled
   conditional analysis (84 validation roots), not the full evaluation population.
2. Oracle ceiling is 226/320. Stock is 209/320; expected uniform random is
   198.11/320. Fixed argmax ties use candidate order, stock first. Also report
   random tie-breaking sensitivity when tied top scores occur.
3. Preserve per-episode weighting, paired episode bootstrap intervals for model
   differences, and task-block sensitivity intervals when task identities exist.
   Candidates, transitions and decision points from the same episode are correlated.
   Report 40 task identities separately from episode counts; no extra independence
   is gained by evaluating more models on the same labels.
4. The original scalar transition cache contains training-group trajectories only.
   A separate evaluation-only cache now restores all 960 saved trajectories for
   the 320 validation roots, preserving full proposed actions, terminal chunks and
   known-budget masks. Training and held-out one/five-step Bellman diagnostics are
   available for the three main critics. This cache is excluded from the training
   runner and is not an untouched test set. Never use a sampled training minibatch
   loss as validation Bellman error.
5. Transition aggregation: calculate metrics per episode first, then average;
   also provide transition-weighted summaries as secondary. Retain terminal
   transitions and distinguish true termination from truncation in the target
   contract. Current success-before-fixed-horizon target treats exhausted horizon
   as zero bootstrap. Record the policy used for continuation and next actions.
6. Prospective cohort: start with 200 independent episodes distributed over all
   40 familiar tasks, using critic-training-disjoint identities and a fresh behavior
   seed. Freeze early/middle/late observation schedules before collection. Select
   no state using candidate results. Report stages not reached because episodes
   terminate; don't silently replace them with surviving or harder states.
   Candidate-based labels and transition records must be collected once and shared
   across all compatible frozen critics. Explicit spend approval precedes collection.

## Model registry and fair comparisons

Include original scalar MC and TD baselines; all six overnight MC/TD variants;
anchored transformer/CNN baselines; the two continuation endpoints; and nine CNN
width/seed arms at both fixed-final and chronological rule-selected checkpoints.
Keep legacy distributional/ranking or differently conditioned critics in a separate
registry with their actual target and input requirements. They are not silently
discarded or assigned probability metrics without a valid probability output.

Every registry row records checkpoint path/SHA, family, objective, seed, update,
training dataset/split, continuation policy, target meaning, input contract,
selection rule, and which evaluations are supported. An unavailable result is
marked unavailable with a reason, not zero and not omitted from coverage.

Use the same root IDs AND candidate labels/order for every paired comparison.
Changing architectures, exposure or training data makes MC-versus-TD comparisons
descriptive rather than a controlled objective experiment. The CNN widths/seeds
are a matched study. Rule-selected results remain exploratory because selection
used the same validation split; retain fixed-final results and do not pick the
best model or seed using this report.

## Gradient correction coordinates

When gradients are available, retain the full 10-by-7 action gradient and valid
mask per root, plus the normalized-coordinate version. Summarize per-coordinate
and per-time-step signed gradient, absolute magnitude, energy fraction, norm
quantiles, and stability. Verify the action codec before labelling physical axes;
otherwise use coordinate 0 through 6. Gripper and translation/rotation coordinates
have different semantics and scales. Record raw and standardized units separately.

Counterfactual axis-only or time-only correction success cannot be inferred from
gradient magnitude. Such ablations require a later frozen paired simulator
protocol with zero, ascent, descent and equal-size random controls. The same applies
to choosing a step size, gating by uncertainty or deploying repeated PCP.

## Deliverables and gates

- Human-readable comparison table, reliability plots, error/horizon plots,
  selection table with oracle/random/stock, gradient alignment/stability plots,
  and explicit per-model coverage matrix.
- Machine-readable per-root predictions/outcomes and per-transition residuals,
  checkpoint/dataset hashes, fixed analysis settings, seed and bootstrap unit.
- Focused correctness tests: terminal reward/no-bootstrap behavior, n-step
  accumulation, absence of train/validation overlap, candidate-label alignment,
  numerical score ties, known gradients/finite differences, and episode weighting.
- Review outcome prediction AND action information AND gradient evidence before
  choosing any model for a prospective PCP test. There is no automatic promotion
  threshold or checkpoint promotion in this evaluation.
