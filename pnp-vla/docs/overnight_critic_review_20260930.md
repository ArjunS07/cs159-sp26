# Astra review and overnight critic experiments — September 30

## Diagnosis

At the original fixed 2,000-update endpoint, neither scalar critic learned
useful training-root action ranking: MC pair accuracy approximately 0.512 and
TD approximately 0.495 on all 1,280 training roots. Stock succeeded on 832;
MC selected 788 successful candidates and TD selected 766. Median within-root
probability standard deviations were approximately 0.000070 and 0.000121.
These differ from mean spreads, which have outlier contributions.

The existing trajectory sampler trains on source episodes plus only fresh
branches 1 and 5. Those three candidates provide 198 mixed training roots;
all nine candidates provide 368. Astra estimated a median of only 5.08
intervention examples per root in the original 64,000 training draws. Sampling
with replacement means 64,000 draws do not cover all 58,695 windows.

Under this sampler, one-step TD directly reaches a terminal transition on
11.86% of draws; five-step TD reaches one on 51.28%. Longer bootstrapping is
therefore a targeted test of sparse reward propagation.

## Fixed experiment matrix

All jobs initialize from the appropriate original MC or TD checkpoint and
run 4,000 additional updates, retaining its learned normalization and weights.
Each starts a fresh AdamW optimizer, warmup and cosine schedule, peak learning
rate 0.00005. Same frozen 1,280/320 root split, batch size 32, gamma one,
scalar sigmoid and ordinary BCE. No ranking loss or GAE.

| Job | Change |
| --- | --- |
| `mc_roots` | 18 root examples: two uniformly selected roots, all nine candidates; plus 14 trajectory windows |
| `td_n5` | Five-step recorded-policy SARSA, terminal bootstrap zero |
| `mc_long` | Longer original trajectory MC control |
| `td_long` | Longer original one-step TD control |
| `mc_roots_late` | Root MC plus a direct flattened-action/state interaction path |
| `mc_roots_contextdrop` | Root MC plus 10% prefix-token dropout; shared context mask across each root's candidates |

The late-fusion residual is initialized to zero, preserving the base model's
initial predictions and action gradients. It adds capacity and a different
action path together; any gain should be attributed to that combined change.
The architecture comparison uses the same root-focused MC sampling and fresh
optimizer as its control. No synthetic action inherits an old rollout label.

The root-focused arm changes both candidate coverage and sampling emphasis;
it does not isolate either individually. The actual root fraction is 18/32,
or 56.25%, after rounding to complete nine-candidate groups.

## Reporting and recovery

Evaluate fixed final checkpoints, never pick a favorable validation update.
Report selection versus stock, rescues/spoils, Brier, within-root pair accuracy,
training-versus-validation differences, probability spread and gradient
projections against recorded candidate outcomes. Projection agreement is a
directional diagnostic, not measured gradient ground truth or PCP rollout
performance. Existing validation roots are repeatedly inspected development
data; retain future untouched episodes for a confirmatory test.

Astra checked targets and architecture initialization numerically. Terminal
success/failure tests passed for one-, five- and ten-step backups. The zero
late-fusion initialization preserved logits and action gradients exactly.
Queue recovery was strengthened to require the fixed final checkpoint and
run missing diagnostics independently of training completion.

Run the queue with `scripts/run_smolvla_overnight.py`. It starts no new jobs
after 8am Pacific, September 30, and lets an already running job finish.
Jobs checkpoint every 50 updates and resume automatically. Local results,
audit files and logs are under `~/pnp-vla-runs/diagnostics/overnight_20260930`;
model checkpoints and MLflow experiments use the `smolvla-overnight-20260930-`
prefix. Remote data is read only; no Colab launch, Modal charge or model upload
is part of this queue.
