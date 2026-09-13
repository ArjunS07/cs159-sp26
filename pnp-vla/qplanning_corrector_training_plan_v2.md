# Q-Planning Corrector: Experiment Log v2

Last updated: 2026-09-12

This is the current team handoff for the Q-Planning-inspired corrector work. It records what was
implemented, what was run, what happened, and which follow-ups are in progress. It supersedes the
older `qplanning_corrector_training_plan.md`.

## Current status

| Stage | Status | Main result |
|---|---|---|
| Q10/Q50 offline critics | Completed | Trained successfully on the immutable mixed LIBERO dataset |
| Matched OOD PRO160 evaluation | Completed | Net neutral/slightly negative versus stock; no demonstrated SR benefit |
| Q50 + future-U20 auxiliary head | Trained; evaluation stopped early | All tested uncertainty coefficients were degrading SR during the partial run |
| Failure/U20-prioritized replay | Training in progress | Four fresh Q50 arms split across notebooks 72 |
| LIBERO-only Q50 base | Ready to run | Notebook 73; intended for a staged LIBERO → PRO continual experiment |
| Paper-style continual/online replay | Not yet implemented | Candidate next stage after replay-priority results |

## Data and evaluation partitions

The immutable training snapshot used by the completed Q10/Q50 critics is
`pcpcds-98d1f32dff8213841529b3c6`. It contains 1,880 train-eligible rollouts:

- 600 standard-LIBERO rollouts: 400 initial plus 200 adaptive;
- 1,280 non-position LIBERO-PRO rollouts: two 640-rollout tranches;
- zero members of the 160-rollout position-perturbation evaluation partition.

The eight fresh-PRO sentinels overlap the corresponding 640-rollout manifest and do not add eight
more unique episodes. The snapshot uses an 80/20 split grouped by benchmark, suite, task, and
physical-state identity; different behavior seeds for one physical state remain in the same split.

Every collection rollout:

- predicts a native 50-action PI0.5 chunk;
- executes only the first 10 actions and replans;
- stores the complete executed trajectory, sparse rewards, terminal masks, physical robot state,
  policy proprioception, and cached frozen PI context at every 10-action boundary;
- stores the generated 50-action boundary chunk;
- logs measurement-only K=5 P&P traces at Euler steps 3 and 4, including U10/U20/U50;
- does not execute refinement or candidate selection during data collection.

The primary OOD evaluation uses 160 position-perturbation LIBERO-PRO identities. These suites are
disjoint by perturbation category from the non-position PRO suites used for critic training. The
older PRO220 development set was not guaranteed train-disjoint and must not be used for
confirmatory claims.

## Architecture we implemented

The implementation lives in `pnp/qplanning_critic/`.

The critic is parameter-disjoint from the frozen PI0.5 actor. Unlike the paper's separate DinoV2
and T5 encoders, this implementation consumes frozen PI visual/language context tokens saved
during rollout collection. This adapts Q-Planning to the data already collected without repeatedly
running a second large observation encoder during training.

The state/action module contains:

- projected frozen PI visual/language context tokens;
- explicit projected physical robot-state and policy-proprioception tokens;
- one token per candidate action plus learned temporal-position embeddings;
- one learned RL/readout token;
- an 8-layer transformer decoder, width 768, 12 attention heads, and FFN width 3072;
- self-attention over the RL/action tokens and cross-attention to the state/context tokens;
- a 101-bin HL-Gauss categorical value head over `[0, 1]`;
- one online Q network and one non-trainable EMA target copy with `tau=0.005`;
- no Q1/Q2 double-critic pair.

The RL token is a learned parameter of shape `(1, 1, 768)`. It is not text or a manually defined
reward token. Bellman-loss gradients train it to summarize the state and candidate action sequence.

### Q10 and Q50 adaptation

Both datasets are anchored at the saved 10-action planning boundaries.

| Setting | Q10 causal control | Q50 paper-style adaptation |
|---|---:|---:|
| Offline action window | Executed `t:t+10` | Executed `t:t+50` |
| Reward window | 10 environment actions | 50 environment actions |
| Bootstrap observation | `s_(t+10)` | `s_(t+50)` |
| Bootstrap action | Next executed 10-action window | Next executed 50-action window |
| Discount | `gamma^10` | `gamma^50` |
| Inference candidate scored | Generated actions 0–9 | Generated actions 0–49 |
| Actions executed before replanning | 10 | 10 |

Q50 concatenates 50 actions actually executed under five successive 10-action replans. It does
**not** train on the unexecuted 10–49 tail of one generated chunk. At inference it scores one
generated 50-action candidate and executes only its first 10 actions. This train/inference
approximation follows the structure of released Q-Planning, but it is an approximation and must
not be described as an exact causal target for the generated candidate.

The Bellman targets are:

\[
y_t^{10}=\sum_{j=0}^{9}\gamma^j r_{t+j}
+\gamma^{10}(1-d_t^{10})\bar Q_{10}(s_{t+10},a_{t+10:t+20}),
\]

\[
y_t^{50}=\sum_{j=0}^{49}\gamma^j r_{t+j}
+\gamma^{50}(1-d_t^{50})\bar Q_{50}(s_{t+50},a_{t+50:t+100}).
\]

Targets crossing success, termination, truncation, or the available episode end do not bootstrap.
Padding after termination is masked. Scalar targets are projected onto the HL-Gauss bins and the
online critic is trained with categorical cross-entropy.

## Experiment 1: ordinary offline Q10 and Q50

Notebook 64 trained Q50 and Q10 from scratch from the same snapshot. The shared
`source_parts_v2` cache stores compressed source artifacts once; Q10/Q50 windows are reconstructed
one rollout at a time rather than duplicated on disk.

Training contract:

- 8,000 optimizer updates;
- effective batch 64;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- 500-update linear warmup followed by cosine decay;
- BF16 and gradient clipping at 1.0;
- identical declared seed;
- training print every 100 updates;
- fixed offline validation every 500 updates;
- resumable checkpoint every 1,000 updates, retaining only the latest checkpoint.

Uncertainty was excluded from the inputs, loss, targets, and replay sampler.

### Inference protocol

The frozen-critic evaluation used notebooks
`workers/68_eval_qplanning_heldout160_worker_{0,1,2,3}.ipynb`:

- stock PI0.5: ordinary 10 Euler steps, execute 10 actions;
- Q planners: sample 64 candidates using 3 Euler steps;
- score all candidates with Q10 or Q50;
- retain the 16 highest-Q candidates;
- return their Q-softmax weighted average;
- execute 10 actions and replan;
- no P&P refinement, uncertainty gate, video, frames, or generated-chunk blobs.

Stock, Q10, and Q50 were run on the same 160 identities. The Q planners shared the same
deterministic candidate seed schedule.

### OOD result

The updated notebook `69_analyze_qplanning_heldout160.ipynb` reports:

| Arm | Episodes | SR | Change versus stock | Paired 95% interval | F→S / S→F |
|---|---:|---:|---:|---:|---:|
| Stock PI0.5 | 160 | 63.125% | — | — | — |
| Q10 planner | 160 | 62.500% | −0.625 pp | [−6.25, +5.00] pp | 10 / 11 |
| Q50 planner | 160 | 61.875% | −1.250 pp | [−6.25, +3.75] pp | 8 / 10 |

Interpretation: both planners were statistically compatible with a net-zero effect and were
slightly negative in point estimate. This evaluation did not demonstrate that the offline critic
improves OOD position-perturbation success.

The critic was trained to estimate discounted return, not to classify failure directly. Failure
prediction is therefore only a diagnostic derived from low predicted value. Episode-aggregated
Q/candidate signals separated failures better than first-boundary signals, but did not provide a
reliable early deployment gate.

### P&P uncertainty diagnostics on the matched overlap

Worker-41 uncertainty artifacts overlap 140 of the 160 stock episodes; the
`libero_object_temp_x0.3` suite is absent. On those 140 episodes:

| Signal | Failure ROC-AUC |
|---|---:|
| Episode U10 | 0.761 |
| Episode U20 | 0.778 |
| Episode U50 | 0.712 |
| First-chunk U10 | 0.474 |
| First-chunk U20 | 0.500 |
| First-chunk U50 | 0.487 |

Whole-episode U20 remains a useful post-hoc failure signal, but the first chunk is approximately
chance on this OOD subset. A high whole-episode AUC cannot by itself support an online gate because
the score is only available after the trajectory unfolds. First-k-prefix signals improve with
additional chunks, but delaying intervention also reduces how much of the trajectory remains
correctable. Per-action-dimension uncertainty was not persisted by workers 41 or 68.

## Experiment 2: preliminary Q50 + uncertainty head

Notebook 70 trained a new Q50 critic from scratch for 8,000 updates with two additions:

1. measured current-boundary U20 was log-normalized and supplied as an extra context token;
2. a second learned token/head predicted mean U20 over the next four planning boundaries.

The loss was:

\[
L = L_{\mathrm{Q,HL\text{-}Gauss}}
  + 0.25\,L_{\mathrm{SmoothL1}}(\widehat{U20}_{t+1:t+4},U20_{t+1:t+4}).
\]

Q remained an ordinary Bellman-trained value estimate; the U head was an auxiliary prediction
task. U10, U50, contraction, held-out position episodes, and online replay were excluded from
training.

Evaluation workers 71 tested uncertainty coefficients `beta in {0.25, 0.5, 1.0}`. Candidate
ranking used `z(Q) - beta*z(predicted future U20)`, retained the top 16 candidates, and then used
the ordinary Q-softmax blend among those candidates.

The evaluation was stopped early because every coefficient was tracking below matched stock and
accuracy continued to degrade. Treat this as a negative preliminary screen, not a precise final
effect estimate. It suggests that an episode-risk auxiliary target does not automatically produce
candidate-level counterfactual rankings that are safe to optimize at inference.

## Experiment 3: prioritized replay, currently running

Notebooks 72 train four ordinary Q50 critics from step zero. Architecture, Bellman targets,
optimizer, seed, 8,000-update schedule, and validation split are fixed. Uncertainty changes only
which recorded windows enter training; it is not an input or auxiliary loss.

Every microbatch is:

- 50% ordinary replay preserving the baseline window marginal;
- 50% trajectory-first prioritized replay, preventing long episodes from dominating merely
  because they contain more boundaries.

The arms are:

| Notebook | Arm | Priority definition |
|---|---|---|
| Worker 0 | Failure-prioritized | Recorded failed episodes |
| Worker 0 | Episode U20 | Top-quartile mean episode U20 within suite |
| Worker 1 | Four-chunk U20 | Top-quartile consecutive four-chunk blocks within suite and block position |
| Worker 1 | Eight-chunk U20 | Top-quartile consecutive eight-chunk blocks within suite and block position |

Files:

- `notebooks/workers/72_train_q50_priority_worker_0.ipynb`
- `notebooks/workers/72_train_q50_priority_worker_1.ipynb`
- `pnp/qplanning_critic/replay_priority.py`

These runs ask whether uncertainty is more useful for allocating critic-training updates than as
a quantity directly optimized during candidate selection. No outcome is available yet.

## Experiment 4: LIBERO-only base for staged adaptation

Notebook 73 is ready to train a fresh ordinary Q50 critic on only the 600 standard-LIBERO
rollouts:

- 4,000 optimizer updates;
- uniform replay;
- the parent snapshot's grouped train/validation membership is preserved;
- only LIBERO artifacts are downloaded;
- action normalization is recomputed from LIBERO training actions only;
- all LIBERO-PRO artifacts, outcomes, and normalization statistics are excluded.

The shorter 4,000-update schedule avoids applying the mixed 1,880-rollout run's full 8,000-update
sample reuse to a dataset about one third its size. File:
`notebooks/73_train_q50_libero_only_base.ipynb`.

This checkpoint is intended to support a clean staged experiment:

1. train the base critic on standard LIBERO;
2. continue Q-only training on the 1,280 non-position PRO rollouts;
3. compare uniform PRO replay against U-prioritized PRO replay;
4. optionally retain 25–50% standard-LIBERO replay to measure/limit critic forgetting;
5. evaluate transfer to the position-perturbation category.

Because the existing 160 position episodes have now been inspected repeatedly, they remain useful
for exploratory comparison but should not be called an untouched final test for a method designed
after seeing their results. A new confirmatory split or perturbation family would be needed for a
strong final claim.

## Plausible continual/online extensions

### Offline staged continual replay

Start every arm from the same LIBERO-only checkpoint and hold the number of additional optimizer
updates fixed. Compare:

- uniform non-position-PRO replay;
- failure-prioritized non-position-PRO replay;
- episode-U20-prioritized replay;
- local four- or eight-chunk-U20-prioritized replay;
- the best U-priority rule with a fixed LIBERO retention fraction.

This is the lowest-cost continual-learning test because all trajectories already exist.

### Paper-style online replay

The Q-Planning paper's larger gains came from repeated planner deployment and Q-only updates:
collect planner-generated successes and failures, append them to replay, update only Q, and repeat.
Our offline critics have not yet undergone that loop. A faithful pilot should:

- freeze PI0.5 throughout;
- begin from one predeclared critic;
- collect matched planner rollouts on training-eligible tasks only;
- append complete executed transitions, not just successful episodes;
- run a fixed number of Q updates per collection round;
- keep a separate evaluation set out of replay and hyperparameter selection.

Uncertainty could prioritize which deployment episodes are replayed or which states receive
additional fork/counterfactual collection. This should be compared against failure prioritization,
because high uncertainty and failure are correlated but not equivalent.

### Uncertainty-guided data acquisition

A more expensive extension is to use high episode/prefix U20 to request additional trajectory
forks from selected states, producing counterfactual action/outcome evidence. This directly targets
the current critic's weakness: it is trained on one realized action sequence per state but is asked
at inference to compare many counterfactual candidates from that state.

## Important interpretation constraints

- Q is a return estimator; failure discrimination is by proxy and is not its direct supervised
  objective.
- Strong whole-episode uncertainty is post-hoc. It cannot be presented as an online gate without
  a separately executed causal policy.
- Offline threshold/window sweeps reuse observed outcomes and are hypothesis-generating, not
  rollout evidence.
- Q50 training uses executed actions spanning replans, whereas Q50 inference scores one generated
  chunk. Keep this approximation explicit.
- The 3-step candidate decoder changes the proposal distribution relative to stock's 10-step
  decoder; planner comparisons include both candidate generation and Q aggregation.
- The 160 position episodes are train-disjoint, but they are no longer method-development-blind.
- Do not train on evaluation-only position suites when claiming transfer to unseen perturbation
  categories.

## Reproduction map

| Purpose | Notebook/code |
|---|---|
| Dataset snapshot/preflight | `notebooks/56_pcp_critic_preflight.ipynb` |
| Original Q50 then Q10 training | `notebooks/64_train_qplanning_q50_then_q10.ipynb` |
| Matched stock/Q10/Q50 PRO160 rollout | `notebooks/workers/68_eval_qplanning_heldout160_worker_*.ipynb` |
| Main OOD analysis | `notebooks/69_analyze_qplanning_heldout160.ipynb` |
| Q50 + future-U20 training | `notebooks/70_train_qplanning_q50_u20.ipynb` |
| Q50 + future-U20 evaluation | `notebooks/workers/71_eval_qplanning_q50_u20_heldout160_worker_*.ipynb` |
| Replay-priority training | `notebooks/workers/72_train_q50_priority_worker_*.ipynb` |
| LIBERO-only base training | `notebooks/73_train_q50_libero_only_base.ipynb` |
| Core Q model/training/inference | `pnp/qplanning_critic/` |
| Replay-priority implementation | `pnp/qplanning_critic/replay_priority.py` |
| LIBERO-only filtered workflow | `pnp/qplanning_critic/libero_base.py` |
