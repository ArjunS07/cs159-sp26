# SmolVLA Q10: architecture and training diagnosis

## Value being learned

At a decision boundary, let (x_t) contain the current observation, task, robot state,
and remaining episode budget. Let (a_{t:t+9}) be the ten actions that will actually be
executed. The selection critic should estimate

\[
Q^{\pi}_{10}(x_t,a_{t:t+9})
=\Pr(\text{episode succeeds}\mid x_t,\operatorname{do}(a_{t:t+9}),
\text{then continue with frozen SmolVLA P\&P }\pi).
\]

The root Monte Carlo label is the observed terminal success (Y\in\{0,1\}).
At a recorded continuation boundary, the policy-evaluation TD target is

\[
y_t=r_t+(1-d_t)\bar Q^{\pi}_{10}(x_{t+10},a^{\pi}_{t+10:t+19}),
\]

where (r_t) is a success event during the ten executed steps and (d_t) marks
termination. The next action must be the recorded action from the *frozen*
continuation policy. A max over next actions would change the estimand to an
improvement/control objective. With binary terminal success and \(\gamma=1\),
both MC and TD target a success probability. TD has bootstrap bias; MC has
single-continuation outcome variance. Keeping remaining time in the input is
necessary for this finite-horizon objective.

## Current implementation and open architectural question

`pnp/smolvla_success_critic.py` uses a frozen SmolVLA prefix pooled to 128 tokens,
robot state, proprioception, remaining time, and a normalized 10-by-7 action
sequence. A three-layer, width-256 transformer decoder puts one RL token before
the action tokens. The value head has 101 bins on [0, 1]. With a 512-wide input
prefix, the model has about 3.33 million trainable parameters.

The root-MC arm applies binary cross-entropy to the *expectation* of those bins;
the TD arm projects a scalar bootstrap target to a narrow Gaussian histogram.
Neither experiment needs a learned distribution over returns to answer the
selection question. A scalar logit with sigmoid output is the simpler matched
baseline for both arms. For TD, soft-label binary cross-entropy with the detached
target (y_t\in[0,1]\) preserves the same expectation fixed point. Keep the 101-bin
head as an ablation until validation shows a benefit.

The 3.33-million-parameter decoder is substantial relative to a few hundred
independent roots. Compare it with a smaller action-conditioned head before
attributing weak ranking to the data alone. Any architecture comparison must
use the same root split, action normalization, and training budget.

## Data and evaluation contract

Nine outcomes from one root share the same state and continuation random seed.
They improve within-state comparisons but do not count as nine independent
states. The 65% uncertainty-selected roots define the development distribution;
report performance separately for uncertainty-selected and uniform roots.
Keep every root and its nine branches in one split. The final test needs new
source episode indices and ideally a second continuation seed on a small fixed
sample, since each current branch outcome uses one future random stream.

For every checkpoint report stock success, oracle success, selected success,
rescues, spoils, pairwise accuracy on mixed roots, and calibration (Brier and
binary cross-entropy). Selection uses a fixed argmax over recorded candidates;
any tuned acceptance threshold requires a separate tuning set. Use paired
root-level intervals for selected-minus-stock, and examine per-suite results.

## Small diagnosis while fresh8 collection runs

Use 256 complete v3 roots and a frozen 80/20 root split. Train short, matched
root-MC and tree-TD arms from scratch with the same 3-layer model and report
their held-out root metrics. This is a pipeline and objective diagnostic, not
an efficacy claim: the validation set is only about 51 roots and the mixed-root
subset is smaller. The success trainer now accepts a v4 experiment and candidate
ordering, but it does not switch datasets automatically; a distinct immutable
v4 snapshot key and a full 800-tree completion check are required.

The frozen 256-root v3 snapshot has digest `d25aa525f602dfac19774105`:
205 training and 51 validation roots, 57 and 21 mixed roots respectively.
Validation stock succeeds on 37 roots; an oracle over the nine recorded
candidates succeeds on 41. Thus only four validation stock failures can be
rescued. The 20,203 training TD windows include only 9.13% first/fork windows
under uniform window sampling, so a TD arm may train mostly on downstream
continuation behavior rather than intervention choices.
The stock success rate is 147/205 on training roots and 37/51 on validation;
randomly choosing among the nine recorded candidates has 33.11 expected
validation successes. The candidate success rate is 65.9% in training and
64.9% in validation. Stock is therefore a strong baseline on this sample.

At 200 updates, the root-MC arm selected 32/51 successes (1 rescue, 6 spoils;
pairwise accuracy 0.568) and the TD arm selected 30/51 (1 rescue, 8 spoils;
pairwise accuracy 0.496). Both are below stock's 37/51. These are short-run
diagnostics, not reliable estimates of final model quality. A longer matched
run is needed to distinguish undertraining from objective or architecture
problems.

At 1,000 MC updates, selected success rose to 36/51 (2 rescues, 3 spoils),
still below stock's 37/51. Training loss reached about 0.08 on the last batch,
while held-out binary cross-entropy was 0.743 and pairwise accuracy 0.529.
This pattern is consistent with overfitting; the small holdout and only four
recoverable stock failures limit the strength of that conclusion.

At 1,000 TD updates, selection was 28/51 (1 rescue, 10 spoils), pairwise
accuracy 0.380, and held-out binary cross-entropy 0.658. More updates did not
repair the root-ranking problem in this short run. This does not disprove TD;
the current uniform-window sampler devotes about 9% of windows to fork roots.

There is also a stronger, already completed 480-tree v3 result in the Modal
Volume (`reports/full_fc647fad3a438bc65c066a59.json`): after 2,000 updates,
stock succeeded on 66/96 held-out roots, root MC selected 60/96 (0 rescues,
6 spoils; pairwise accuracy 0.562), and tree TD selected 56/96 (2 rescues,
12 spoils; pairwise accuracy 0.475). The oracle among recorded candidates was
74/96. Both experiments therefore show that ungated argmax selection is
currently worse than retaining the source action.

The next controlled change should be a stock-preserving selector. Define
\(\Delta_i=Q(x,a_i)-Q(x,a_0)\), where \(a_0\) is the stored-source action.
Switch only if \(\max_{i>0}\Delta_i>\tau\). Choose \(\tau\) on development
data and report its result on fresh root identities. A within-root pairwise
loss can be added to root-MC BCE to directly train the ordering of successful
and failed candidates. A root-balanced TD sampler is a separate ablation;
avoid changing both objective and sampling in one comparison.

After all 800 v4 trees complete, freeze one 160-root validation set and use
nested training sets of roughly 100, 200, 400, and 640 roots. This separates a
data-scaling trend from changes in the holdout composition. Do not select the
holdout from branch outcomes. Compare root MC, TD, and a smaller scalar-head
critic on that same split.

## Corrector implication

The present data supports selecting among SmolVLA P&P chunks. It does not by
itself validate an unconstrained gradient update to the chunk: the critic may
assign high values to actions outside its training distribution. A corrector
experiment should start with reranking fresh P&P candidates, then test a small
regularized action update or P&P projection and evaluate those newly proposed
chunks in the simulator. That separates the critic's ranking ability from the
proposal mechanism's ability to produce executable improvements.
