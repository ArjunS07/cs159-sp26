# Q-Planning-Inspired Corrector: Offline Training Plan v2

Status: authoritative proposed plan. This supersedes `qplanning_corrector_training_plan.md`. No trainer or notebooks are implemented here.

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

Train each model once on the existing immutable snapshot:

- 8,000 optimizer updates;
- effective batch size 64;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- 500-step linear warmup followed by cosine decay;
- BF16;
- gradient clipping at 1.0;
- EMA `tau = 0.005`;
- identical declared seed.

If Q50 requires a smaller GPU microbatch, use gradient accumulation to preserve effective batch 64. Do not recollect data or restart at an intermediate checkpoint.

## Periodic output

Every 100 updates:

```text
Q50 step 2500/8000 | train_ce ... | q_mean ... | grad ... | GPU ... GB | elapsed ... | ETA ...
```

Every 500 updates, evaluate on a fixed set of already recorded validation episodes excluded from optimizer updates:

```text
validation | HL-Gauss CE ... | Q MAE ... | Q(success) ... | Q(failure) ... | failure AUC ...
```

This validation performs no simulator rollout and does not alter the fixed 8,000-step schedule. Save resumable checkpoints every 1,000 updates and at the end, including data/split hashes, source PI revision, horizon, architecture, optimizer, and EMA state.

## Uncertainty use in v1 training

Do not feed U10/U20 to either critic and do not use it to weight training examples. After both runs, use held-out logged uncertainty to report:

- Q calibration and failure discrimination by U20 range;
- whether Q10 or Q50 degrades in high-U states;
- exploratory Q-plus-U threshold/window analyses clearly labelled offline.

Later experiments may add an uncertainty token, auxiliary U/failure head, external activation gate, or candidate objective `Q - beta * U`. These must remain separate from the initial Q10/Q50 comparison.

## Interpretation and next stage

The paper's largest gains do not come from critic architecture alone. Its offline Q-planner improves more modestly, while the large gains follow repeated Q-guided collection and Q-only updates. It also uses many diverse short-denoising candidates at inference. Therefore, successful offline training here establishes only that the critic is calibrated and action-sensitive enough to justify a matched rollout pilot.

After training, compare Q10 and Q50 on identical held-out data using final validation loss, return MAE, success/failure separation, failure AUC, and valid same-state candidate-ranking diagnostics. Then take the stronger model into a matched 220-episode stock-versus-Q-planner pilot. Do not start continual online updates until that planner shows a credible action-selection signal.

## Expected compute

With cached PI context and only the critic decoder trained, estimate approximately:

- Q10: 3-7 hours on one A100 80 GB;
- Q50: 5-10 hours on one A100 80 GB;
- two A100 runtimes in parallel: one overnight wall-clock window.

The later implementation should print measured throughput and ETA after the first 100 updates. Copy artifacts from Drive to local runtime storage before training to avoid Drive I/O dominating the run.

## Planned implementation deliverables

When requested:

1. one manifest/snapshot validator and transition builder supporting horizons 10 and 50;
2. one shared critic implementation and trainer;
3. one Q50-paper-style training notebook;
4. one Q10-causal training notebook;
5. one compact offline comparison notebook;
6. tests for split leakage, horizon construction, terminal masking, normalization, RL-token/action shapes, EMA updates, and checkpoint resume.
