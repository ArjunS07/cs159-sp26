# Q-Planning-Inspired Corrector: Offline Training Plan v2

Status: authoritative record of the implemented and completed offline Q10/Q50 training workflow,
plus the current frozen-critic evaluation protocol. This supersedes
`qplanning_corrector_training_plan.md`.

## Evaluation handoff

- Q10 and Q50 training use executed closed-loop windows. Q50 spans five 10-action replans; it
  is not the unexecuted tail of one generated chunk.
- Notebooks 66 and 67 are the earlier development launchers on PRO220. That cohort is not a
  guaranteed train-disjoint final test and should not be used for confirmatory claims.
- The current evaluation is notebook 68, split into four fixed workers over the reserved
  160-identity position-perturbation PRO manifest. Every worker runs newly collected stock VLA,
  Q10, and Q50 outcomes on the same 40 identities and prints an exact matched table every ten
  complete identities.
- Inference samples 64 candidates with 3 Euler steps, Q-softmax averages the top 16, executes
  10 actions, and replans. No PnP, uncertainty gate, video, frames, or generated-chunk blobs.
- Stock uses the ordinary 10-step decoder and also executes ten actions. Q10 and Q50 use the same
  deterministic candidate-seed schedule and identical candidates at their shared first boundary.
- Both critics remain frozen. Do not perform online updates on the reserved 160 identities.

## Objective

Use the already collected rollouts to train two critics:

- **Q50-paper-style (primary):** the PI0.5 analogue of released Q-Planning. Train on 50-step executed trajectory windows, rewards over those 50 environment steps, and a bootstrap state 50 steps later. At deployment, score a generated 50-action candidate but execute only its first 10 actions before replanning.
- **Q10-causal (control):** train on the 10 executed actions between planning boundaries, rewards over those 10 steps, and the next planning-boundary state.

The first run excludes uncertainty from critic inputs, targets, losses, and sampling. Logged U10/U20 is used only for offline analysis after training.

## Why Q50 is defined this way

The released Q-Planning implementation predicts 32-action candidates, trains its critic with 32-action/reward trajectory windows and a `gamma^32` bootstrap, but executes only 10 actions before replanning. Its online replay stores individual actions actually executed; a training window can therefore span several generated plans. At inference, the critic instead scores a single generated 32-action candidate.

Q50 reproduces that choice with PI0.5's native 50-action horizon. This creates the same approximation: the training action window is a 50-step sequence actually followed under repeated replanning, while the inference candidate is one 50-action prediction whose tail will be discarded. The approximation must be documented and tested rather than described as exact causal Q-learning.

## Fixed comparison

| Setting | Q10-causal | Q50-paper-style |
|---|---:|---:|
| Offline action window | executed `t:t+10` | executed `t:t+50` |
| Offline reward window | 10 environment steps | 50 environment steps |
| Bootstrap observation | `s_(t+10)` | `s_(t+50)` |
| Bootstrap action window | next executed 10-step window | next executed 50-step window |
| Discount | `gamma^10` | `gamma^50` |
| Inference candidate input | generated actions 0-9 | generated actions 0-49 |
| Actions executed before replanning | 10 | 10 |
| State, language, robot inputs | identical | identical |
| Architecture except max action length | identical | identical |
| Optimizer, effective batch, updates, seed | identical | identical |

This is not a one-variable ablation because both the action and Bellman horizons change. It compares a paper-faithful construction against a replanning-aligned construction.

## Bellman targets

Q10:

\[
y_t^{10}=\sum_{j=0}^{9}\gamma^j r_{t+j}
+\gamma^{10}(1-d_t^{10})\bar Q_{10}(s_{t+10},a_{t+10:t+20}).
\]

Q50:

\[
y_t^{50}=\sum_{j=0}^{49}\gamma^j r_{t+j}
+\gamma^{50}(1-d_t^{50})\bar Q_{50}(s_{t+50},a_{t+50:t+100}).
\]

Targets that cross success, termination, truncation, or the available end of an episode do not bootstrap. Reward and action padding beyond termination is masked. Use `gamma = 0.99`.

The scalar target is projected to 101 HL-Gauss bins over `[0, 1]`. The online critic is optimized by categorical cross-entropy. A non-trainable EMA copy supplies bootstrap values and is updated with `tau = 0.005`.

## Existing-data requirements

No new collection should be needed if the completed artifacts contain the fields specified by the collection plan:

- complete executed action trajectory;
- per-step rewards and terminal/success masks;
- observations and cached PI context at every 10-step planning boundary;
- physical robot state and policy proprioception;
- generated 50-action chunks at planning boundaries;
- task, suite, phase, seed, episode, and manifest identities.

Q10 examples are anchored every 10 environment steps. Q50 examples should also be anchored at those same planning boundaries, then gather the next 50 executed trajectory actions and use the observation five planning boundaries later. This avoids inventing observations between the saved boundaries.

Before model allocation, print and verify:

1. complete compatible manifests and one immutable source PI revision;
2. episode and usable-transition counts by collection phase and suite;
3. no overlap between train, validation, and untouched evaluation identities;
4. no held-out/evaluation-only LIBERO-PRO suites in training;
5. correct action normalization and action dimensionality;
6. first 10 generated actions agree with the recorded executed stock actions after conversion;
7. terminal and padding masks for both horizons;
8. the fraction of episodes long enough to provide nonterminal Q50 bootstraps.

## Shared architecture

- Frozen cached PI visual/language context tokens.
- Explicit projected tokens for robot physical state and policy proprioception.
- One token per candidate action plus learned temporal-position embeddings.
- One learned RL/readout token prepended to the action tokens.
- Transformer decoder with self-attention over RL/action tokens and cross-attention to state/context tokens.
- Final RL-token representation passed to one 101-bin HL-Gauss head.
- One online Q network and one EMA target copy; no Q1/Q2 pair.

Provisional decoder shared by both runs:

- width 768;
- 8 decoder blocks;
- 12 attention heads;
- feed-forward width 3072;
- dropout 0.1;
- learned action positions up to length 50.

The RL token is an ordinary learned parameter of shape `(1, 1, 768)`, initialized with a small random normal value and expanded across the batch. It is not text. Bellman-loss gradients teach it to summarize the state and candidate action sequence.

## Fixed training schedule

Both completed critics were trained from the same immutable `pcpcds-*` snapshot using:

- 8,000 optimizer updates;
- effective batch size 64;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- 500-step linear warmup followed by cosine decay;
- BF16;
- gradient clipping at 1.0;
- EMA `tau = 0.005`;
- identical declared seed.

Notebook 64 queues Q50 and then Q10 from one local `source_parts_v2` mirror and reconstructs
training windows in memory instead of writing horizon-specific copies. Its checked-in microbatch
defaults are conservative; the exact microbatch used by a completed run is recorded in that
checkpoint. Gradient accumulation always preserves effective batch 64.

## Periodic output

Every 100 updates:

```text
Q50 step 2500/8000 | train_ce ... | q_mean ... | grad ... | GPU ... GB | elapsed ... | ETA ...
```

Every 500 updates, evaluate on a fixed set of already recorded validation episodes excluded from optimizer updates:

```text
validation | HL-Gauss CE ... | Q MAE ... | Q(success) ... | Q(failure) ... | failure AUC ...
```

This validation performs no simulator rollout and does not alter the fixed 8,000-step schedule.
A resumable checkpoint is written every 1,000 updates and at the end, including data/split hashes,
source PI revision, horizon, architecture, optimizer, and EMA state. To bound Drive usage, each
new save replaces the preceding checkpoint for that critic; only the newest checkpoint remains.

## Uncertainty use in v1 training

Do not feed U10/U20 to either critic and do not use it to weight training examples. After both runs, use held-out logged uncertainty to report:

- Q calibration and failure discrimination by U20 range;
- whether Q10 or Q50 degrades in high-U states;
- exploratory Q-plus-U threshold/window analyses clearly labelled offline.

Later experiments may add an uncertainty token, auxiliary U/failure head, external activation gate, or candidate objective `Q - beta * U`. These must remain separate from the initial Q10/Q50 comparison.

## Q50+U20 follow-up

Notebook `70_train_qplanning_q50_u20.ipynb` implements the first uncertainty-aware follow-up
without changing the immutable data split. It trains Q50 from scratch for the same 8,000 updates
and effective batch size 64. At each planning boundary it:

- feeds measured U20 from that boundary as a log-normalized context token;
- predicts Q with the ordinary RL-token HL-Gauss head;
- predicts mean U20 over the next four planning boundaries with a second learned token;
- optimizes `Q cross-entropy + 0.25 * SmoothL1(future U20)`.

U10, U50, contraction, held-out position-perturbation episodes, and online replay are excluded
from this run. The purpose is to test whether an uncertainty-aware critic learns a useful
action-dependent future-risk signal before adding deployment selection or continual learning.

## Interpretation and next stage

The paper's largest gains do not come from critic architecture alone. Its offline Q-planner improves more modestly, while the large gains follow repeated Q-guided collection and Q-only updates. It also uses many diverse short-denoising candidates at inference. Therefore, successful offline training here establishes only that the critic is calibrated and action-sensitive enough to justify a matched rollout pilot.

Offline validation compares Q10 and Q50 using final validation loss, return MAE,
success/failure separation, failure AUC, and same-state candidate-ranking diagnostics. The next
simulator test is the predeclared three-arm PRO160 evaluation in notebook 68. Report stock, Q10,
and Q50 separately on all 160 matched identities; do not tune a threshold or planner setting on
those outcomes and then describe the same set as untouched. Continual Q-only updates remain a
later experiment, after a frozen critic demonstrates a credible action-selection signal.

## Training execution

The full Q50 and Q10 runs have been completed and saved. Notebook 64 remains the reproducible
launcher: it mirrors compressed source parts to ephemeral local storage once, streams both
horizons from that mirror, prints throughput and ETA every 100 updates, validates every 500, and
queues Q10 after Q50. Training directly from Drive or rebuilding persistent Q10/Q50 window caches
is obsolete because it is substantially slower and can exhaust Colab storage.

## Implemented deliverables

1. Immutable snapshot validation and transition construction for horizons 10 and 50.
2. Shared RL-token critic, EMA target, HL-Gauss trainer, and inference scorer in
   `pnp/qplanning_critic/`.
3. Combined full-run launcher `notebooks/64_train_qplanning_q50_then_q10.ipynb`.
4. Development evaluation launchers 66/67 for PRO220.
5. Four matched, train-disjoint evaluation workers
   `notebooks/workers/68_eval_qplanning_heldout160_worker_{0,1,2,3}.ipynb`.
6. Tests for split leakage, horizon construction, terminal masking, normalization,
   RL-token/action shapes, EMA/checkpoint behavior, inference selection, and worker contracts.
