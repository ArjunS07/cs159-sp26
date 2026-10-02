# Briefing: Uncertainty-Gated Test-Time Search for a Flow-Matching VLA — Verifier/Selector Status

## How to use this brief

You are being asked for judgment, critique, and ideas on a stalled research component.
You have read access to the codebase, but this document is written so you should **not
need to read code to form an opinion** — treat direct code inspection as a last resort to
confirm a specific mechanical detail, not as the way to get oriented. Everything load-bearing
(architecture, data schema, collection protocol, evaluation methodology, and all current
empirical results) is transcribed below, including exact numbers.

We want: an independent read on the diagnosis, alternative hypotheses we may have missed,
and a broad set of concrete next-step ideas (cheap and expensive), with tradeoffs. We are
**not** looking for validation of a predetermined plan. Where we have drawn a direct inference
from the data (e.g., non-saturating best-of-k → there is a candidate-budget design space),
we say so and mark it as an inference, not a conclusion you must accept.

---

## 1. Project thesis

Working title: **"When Should a VLA Think Harder? Uncertainty-Gated Test-Time Search for
Flow-Matching VLA Policies."**

The base policy is a **pi0.5-style flow-matching VLA**, kept **frozen**. Core idea: at most
states the policy is fine and you should just execute its default action chunk; at a minority
of high-uncertainty states, you should spend extra test-time compute to *search* for a better
action. The system has two parts:

1. **A gate (uncertainty probe U):** cheap, tells you *when* to think harder.
2. **A selector/verifier:** given several candidate action chunks proposed at the same state,
   tells you *which* one to execute (endpoint best-of-n selection).

The intended deployment operator is **Operator A: endpoint best-of-n** — at a gated state,
sample n candidate chunks, score each, execute the best (KL-budgeted resampling). A
mid-trajectory search variant ("Operator B", SMC / search-over-paths) was explicitly
**dropped** and is out of scope. This matters because it means the verifier only ever scores
*finished* (t=1) action chunks, which is exactly what the current training data contains.

The component that has stalled is the **selector/verifier**. The gate (U) is considered
largely established (U correlates with failure; see the stratum success table below).

---

## 2. The base policy and the uncertainty probe (context, not under active change)

- **Flow-matching action expert.** Produces a velocity field integrated from t=0 (noise) to
  t=1 (data). A one-step Tweedie/endpoint prediction is available: `ẑ₁ = z_t + (1−t)·v_θ`.
- **Shared VLM prefix.** PaliGemma-style backbone; the language/vision prefix is shared across
  candidates at a state, so scoring n candidates is wall-clock sublinear in n (the expensive
  prefix is computed once).
- **Uncertainty probe U (self-consistency).** A "predict-and-perturb" probe: at a fixed noise
  level `s`, take the one-step endpoint estimate `a_hat = x − s·v(x, s)`, re-perturb it with
  fresh noise, and repeat K times; `U = mean|a_hat_{i+1} − a_hat_i|` measures how self-consistent
  the endpoint prediction is. Empirically correlates with task failure.
  **Where U is measured (verified against the stored data):** the probe runs at **every euler
  step of the denoising schedule** — 9 steps with noise level `s` sweeping 0.9 (near noise) down
  to 0.1 (near data), *not* a single point. Per-step U is not flat: in the collected data it is
  lowest near `s=0.9`, peaks mid-trajectory around `s=0.5`, and eases by `s=0.1`. The stored
  `uncertainty_stratum ∈ {low, mid, high, unknown}` is a **tercile of the trajectory MEAN** of
  those per-step values for the decision chunk. The raw per-step values persist in a separate
  `pnp_euler_steps` table (columns include `s`, `u_mean`); only the collapsed bucket reached the
  verifier examples. Consequence: U at a *specific* noise level (e.g. endpoint vs mid-trajectory)
  is a distinct, currently-unused signal that could stratify success differently than the average.
- **No cheap policy likelihood.** Flow matching does not give a cheap exact log-likelihood, so
  likelihood-based candidate scoring is not directly available. U is the available
  uncertainty signal and it *is* persisted.

---

## 3. Data collection protocol

Collected by rolling out the frozen policy in a **deterministic simulator (LIBERO)** and, at
selected states, branching into multiple candidate continuations.

- **Candidate groups.** At a decision state, a group of candidate action chunks is formed:
  one `default` candidate (the policy's index-0 sample) plus several `fresh_noise_i`
  candidates (re-sampled chunks from the same policy at the same state). **Median 8 candidates
  per group (range 4–12).**
- **Common Random Numbers (CRN), already implemented.** Each candidate's continuation is run
  under a **candidate-independent** branch seed (`branch_seed = seed ^ 0x51A7`; continuation
  noise derived via `chunk_noise_seed(branch_seed, replan)`). The simulator state is exactly
  restored before each branch. **Consequence:** within a group, differences in outcome between
  candidates are attributable to the differing action prefix, not to continuation luck. This is
  a design fact worth internalizing — a lot of "the labels are noisy" hypotheses are already
  controlled for.
- **Outcome label.** Each candidate gets a binary `success` from running its continuation to
  task completion/failure.
- **Prefix length.** Continuations execute the first `prefix_length` steps of the candidate
  chunk before replanning; the current default used throughout is `prefix_length = 10` (out of
  a 50-step chunk).

---

## 4. Data schema (what is and is not stored)

Per-candidate record (`CleanChunkExample`):

- `obs_enc` — **2048-d mean-pooled prefix embedding** of the VLM (frozen). NOTE: only the
  *mean-pooled* vector is persisted, **not the token sequence**.
- `actions` — the action chunk, shape `(50, 7)`.
- `action_mask` — validity mask over the 50 steps.
- `chunk_position` — scalar position of this chunk within the episode.
- `candidate_kind` — `"default"` or `"fresh_noise_i"`.
- `candidate_group_id` — groups candidates sampled at the same state.
- `success` — binary outcome.
- `uncertainty_stratum` — `low | mid | high | unknown`.
- `n_steps`, `return_target` — available dense-ish signals.
- `benchmark` — task/suite id.

A transition variant (`ChunkTransitionExample`) additionally carries `reward`, `discount`,
`return_target`.

**Not stored** (would require new collection to obtain): the VLM token sequence (only the mean
is kept), any privileged simulator ground truth (object poses, contacts, grasp state), and any
policy likelihood.

**Datasets.** Development pool (used for all analysis and training):
`verifier-clean-pairs-v3`, `verifier-clean-pairs-v4-dev`, `verifier-clean-pairs-v4-test`,
`verifier-online-selection-v1`, `verifier-v2-pro-development`.
**Sealed confirmatory cohort:** `verifier-v2-pro-confirmatory` — a held-out one-shot test set.
It must not be consulted until a candidate verifier has cleared development-only controls; the
codebase guards it as `CONFIRMATORY_EXPERIMENT`. Opening it early spends its (one-shot)
evidential value.

---

## 5. Model architectures tried

### 5a. `CompactAdvantageVerifier` (notebook 09) — the current focus

A frozen value pathway plus a small same-state action ranker.

- Inputs: `obs_dim=2048`, `action_dim=7`, `context_dim=128`, a small `action_width`, dropout.
- Structure: an obs encoder + position encoder produce a state context; a `state_head` gives a
  value `V(s)`; the ranking path encodes the **executed action prefix** (first `prefix_length`
  steps, default 10) via residual temporal conv blocks (mean+max pooled) and produces an
  advantage `A(s, a)`. Production score is `A(s, a)`; `V(s) + A(s, a)` is available for
  diagnostics.
- **Four conditioning modes** (how state enters the advantage head):
  - `action_only` — state is zeroed (pure action-shape scoring; used as the control).
  - `multiplicative` — advantage head sees `[action, action * state]`.
  - `film` — FiLM blocks modulate the temporal features by the state.
  - `cross_attention` — a state query attends over the action tokens.
- **Ranking loss:** Bradley–Terry over all discordant candidate pairs,
  `softplus(-(pos − neg)).mean()`.
- **Action normalization:** `set_action_statistics(mean, std)` normalizes actions, but
  notebook 09 **never calls it**, so actions are used unnormalized (identity). This is
  common-mode across all four conditioning arms, so it does not explain differences between
  arms — only absolute scale and cross-notebook comparability.
- **Registered checkpoints** (in the model store):
  - `2f1a683673ee4c93` — `CompactAdvantageVerifier`, conditioning = **film**,
    `state-conditioned-verifier-v2` (this is the model used to produce the scored results
    below).
  - `3add05c827424c4a` — `CompactAdvantageVerifier`, `action-advantage-verifier-v1` (older
    schema, superseded).

### 5b. `HybridChunkCritic` (notebook 10)

An IQL twin **long** critic (50-step) distilled into a twin **short** critic (10-step), plus
Monte-Carlo and ranking losses (expectile ≈ 0.7, distill expectile ≈ 0.8, a ranking-weight
term). Different architecture; not scorable by the notebook-09 scoring path. In prior
evaluation it scored barely above chance and is not the current focus.

---

## 6. Evaluation methodology

- **`group_macro_ranking_accuracy`** — the headline metric. For each *discordant* group
  (a group that contains **both** a success and a failure among its candidates), compute the
  fraction of (success, failure) candidate pairs where the model scored the success higher;
  then macro-average over discordant groups. **0.5 = chance.** Only discordant groups carry
  ranking signal, so the discordant count is the effective sample size.
- **Bootstrap CI** — cluster bootstrap over groups, 10k resamples, [2.5%, 97.5%].
- **Action-only control** — re-score with state zeroed (`zero_context=True`); the gap
  `ranking − action_only` ("control_gap") measures what the state conditioning adds.
- **Oracle uplift** — `mean(best candidate success in group) − mean(default candidate
  success)`; the ceiling for any selector on the current candidate pool.
- **Registration eligibility gate** — four CI-lower-bound checks that a verifier must pass to
  be considered real: ranking > 0.5, uplift > 0, beats action-only > 0, beats shuffled > 0.
- **Mode-level analysis (added in diagnostics):** cluster candidates within a group in
  executed-prefix action space (k-means + silhouette, min silhouette 0.15); then measure
  whether the model orders *modes* correctly, and what fraction of BT training pairs are
  *within* a single mode (near-identical prefixes) vs *across* modes.

---

## 7. Prior result that triggered the investigation

Both trained verifiers underperformed on the **pooled** development set:

- Notebook 09 (`CompactAdvantageVerifier`): ~0.57–0.60 pooled group-macro ranking, and
  `control_gap ≈ 0` (did not beat the action-only control).
- Notebook 10 (`HybridChunkCritic`): barely above chance.

Two things made the naive "the method is dead" verdict **under-determined**:
1. The labels are already CRN-clean (Section 3), so "continuation luck" is not the explanation.
2. The pooled evaluation averages over the wrong distribution (all states, not U-gated states),
   with the wrong operator (argmax over all candidates, not gated best-of-n over modes), at the
   wrong granularity (chunk pairs, not mode ordering).

This motivated a zero-new-sim diagnostic pass on already-collected data, stratified to the
deployment-relevant regime.

---

## 8. Current empirical findings (diagnostics on existing data, model = `2f1a…`, film)

All numbers below are on the **development pool**, scored with the registered film verifier.
The action-only control is computed for free by zeroing the state on the same model.

### 8a. Stratified ranking by uncertainty stratum

| stratum | groups | discordant | ranking | CI (95%) | action_only | control_gap | oracle_uplift |
|---|---|---|---|---|---|---|---|
| high | 1001 | 186 | 0.524 | [0.485, 0.566] | 0.499 | 0.026 | 0.069 |
| low  | 96   | 9   | 0.778 | [0.593, 0.926] | 0.463 | 0.315 | 0.031 |
| mid  | 86   | 16  | 0.448 | [0.271, 0.625] | 0.443 | 0.005 | 0.035 |

(The `low` and `mid` rows have 9 and 16 discordant groups respectively — very low power; their
extreme rankings and wide CIs are not reliable.)

### 8b. Stratified ranking by within-group prefix-mode spread (tercile)

| spread tercile | groups | discordant | ranking | CI (95%) | action_only | control_gap | oracle_uplift |
|---|---|---|---|---|---|---|---|
| high | 394 | 63  | 0.561 | [0.484, 0.637] | 0.539 | 0.022 | 0.079 |
| low  | 395 | 36  | 0.481 | [0.391, 0.569] | 0.467 | 0.014 | 0.030 |
| mid  | 394 | 112 | 0.527 | [0.470, 0.581] | 0.475 | 0.052 | 0.081 |

### 8c. Mode-level analysis

- **Mode-ranking accuracy:** 0.576 (n = 191 groups that have ≥2 distinct modes with differing
  success). Modestly above chance.
- **Within-mode fraction of BT pairs:** 0.478 (n = 211). Roughly **half** of the pairwise
  training gradient compares candidates *within the same action mode* (near-identical
  prefixes).
- **Spearman(model score, cluster-success soft label):** 0.068. The score has almost no
  monotone relationship with how successful an action mode actually is.

### 8d. Oracle ceiling (single-pool, current candidate counts)

- Overall oracle uplift: **0.0634**.
- Multimodal groups (≥2 modes): 0.0645 (1101 groups).
- Unimodal groups: 0.0488 (82 groups).
- High-spread tercile: 0.0787 (394 groups).

### 8e. Training-free baselines and per-stratum success

- Selector success: **majority-mode = 0.6418**, **default = 0.6500**, oracle headroom = 0.0634.
  (Picking the most common mode is *worse* than doing nothing.)
- Group success by uncertainty stratum: **high = 0.6295** (1001), low = 0.6641 (96),
  **mid = 0.7587** (86). High-uncertainty states have the lowest baseline success.

### 8f. Deployment-stratum gate readout (high prefix-mode spread)

`ranking = 0.561, CI [0.484, 0.637], control_gap = 0.022, oracle_uplift = 0.079,
discordant_n = 63`. Against a pre-registered gate (ALIVE required ranking ≥ 0.62 and CI-lower
> 0.50 and control_gap > 0 and oracle ≥ 0.05; REPRESENTATION required a flat CI but oracle
≥ 0.10; NO-SUPPORT required oracle < 0.03), this fell **between** branches → the readout was
"AMBIGUOUS, revisit thresholds."

### 8g. Best-of-k oracle rescue curve (exact, sampling-without-replacement over actual outcomes)

For each group with a budget of k, the true probability a random k-subset contains a success,
averaged over groups. Candidates/group: median 8, min 4, max 12.

High-uncertainty stratum:

| k | oracle success | groups with ≥k candidates |
|---|---|---|
| 1 | 0.632 | 1001 |
| 2 | 0.667 | 1001 |
| 4 | 0.689 | 1001 |
| 8 | 0.716 | 931 |

All groups: k=1 → 0.644, k=2 → 0.679, k=4 → 0.700, k=8 → 0.716.

---

## 9. Direct inferences we have drawn (offered as inferences, open to challenge)

1. **Labels are prefix-attributable, not luck.** CRN is implemented and the sim is
   deterministic, so within-group outcome differences reflect the differing action prefix.
   Hypotheses premised on "the outcome labels are just noise" are largely already controlled.
2. **The selector is the failing component, not (obviously) the data.** Across every stratum
   with adequate power, ranking sits at ~0.52 with CI straddling 0.5; control_gap ≈ 0.02
   (state conditioning adds ~nothing over action-only); Spearman ≈ 0.07 (score barely relates
   to mode quality). The model behaves close to a random selector on the relevant slice.
3. **The best-of-k oracle is monotone and non-saturating through k=8** (still +2.7 points at
   the last step). Inference: **marginal returns to a larger candidate budget n remain
   positive** in this regime — there is a candidate-budget / n design space, and the single-pool
   oracle uplift (~0.08) is a *lower bound* on what best-of-n could offer at larger n.
4. **~48% of BT training pairs are within-mode.** Inference: a large share of the training
   gradient is spent trying to separate near-identical prefixes, which may dilute the
   cross-mode signal that actually matters for selection. (This is a plausible mechanism, not
   an established cause.)
5. **The target regime has recoverable value.** High-uncertainty states have the lowest
   baseline success (0.63) *and* real, rising oracle headroom (0.63 → 0.72 at k=8). The states
   the project wants to intervene on are the states with the most to gain.
6. **Naive selection heuristics do not work.** Majority-mode selection (0.642) is below default
   (0.650); there is no free lunch from simple cluster-mass voting.
7. **Only the high-uncertainty and mode-spread terciles have adequate discordant-group counts.**
   The `low`/`mid` uncertainty rows (9, 16 discordant) are underpowered and should not drive
   conclusions.

---

## 10. Cost tiers and constraints (for grounding any proposal)

- **Cheap (existing data, no new sim):** re-score, re-cluster, best-of-k analysis, and
  **retrain the verifier on the collected pool** with different losses/targets/architectures.
  Anything computable from `obs_enc`, `actions`, `success`, `uncertainty_stratum`, `n_steps`,
  `return_target` is in reach now.
- **Moderate (new sim collection required):** persisting the VLM **token sequence** (only the
  2048-d mean is stored today); **prefix-length-aligned relabeling** (re-running continuations
  at a different execution horizon, e.g. matching a ~25-step deployment replan instead of 10);
  **more candidates per group** (to push further along the best-of-k curve); **graded /
  counterfactual negatives**; **active collection** at decision-critical states.
- **Expensive / currently blocked:** any **privileged simulator ground truth** (object poses,
  contacts, grasp state) — not stored, needs new collection to obtain; **policy likelihood** —
  not cheaply available from flow matching.
- **Hard constraint:** the sealed confirmatory cohort (`verifier-v2-pro-confirmatory`) is
  one-shot; it may only be opened once, for a single winner that has already cleared
  development-only controls.

---

## 11. Open questions we would value an outside view on

1. **Scoring target / horizon.** Scoring currently uses the first 10 steps of a 50-step chunk,
   while continuations also execute 10 steps and the intended deployment replan horizon may be
   larger (~25). Is prefix-10 scoring under- or over-informative for selection, and is horizon
   misalignment a plausible driver of the flat ranking?
2. **Is the flat selector a training/objective problem, a representation problem, or a support
   problem?** We have partial evidence for each: within-mode gradient dilution (objective),
   only the 2048-d mean embedding stored (representation), and a modest single-pool oracle
   (support) — though best-of-k suggests support grows with n.
3. **What is the right unit of prediction** — individual chunk, action mode, or something else?
   Mode-ranking accuracy (0.576) is higher than chunk-pair ranking; is mode-level the right
   target, and how should mode/cluster success labels be defined and validated?
4. **How much of the ~8-point oracle headroom at high-U is realistically capturable** by a
   learnable selector, and what selector class would you expect to capture it?
5. **Candidate budget n.** Given non-saturating best-of-k, is the higher-leverage move to
   improve the *selector* or to increase *n* (or to co-design them under a compute budget)?
6. **Alternative uses of the frozen policy's structure.** The endpoint (Tweedie) prediction,
   the shared prefix, and U are all available. Are there scoring signals we are leaving on the
   table (e.g., agreement/consistency among candidates, U computed per-candidate rather than
   per-group, endpoint-geometry features) that don't require new labels?

---

## 12. Idea backlog collected so far (unranked; feasibility noted)

- **Cross-mode-only BT training + cluster-success soft labels** — cheap; targets the ~48%
  within-mode dilution and shifts the objective to mode ordering.
- **Prefix-length-aligned relabeling** — moderate; re-collect continuations at the deployment
  replan horizon so training/eval horizon matches deployment.
- **Graded / counterfactual negatives** — moderate/new sim; measure the discriminability
  resolution limit and give the ranker harder, better-separated pairs.
- **Larger candidate budget n per group** — moderate/new sim; ride the non-saturating best-of-k
  curve; characterize the best-of-n vs compute tradeoff.
- **Privileged teacher probe** — expensive; train a teacher on sim ground truth vs a student on
  `obs_enc` to decisively separate "inputs are insufficient" from "labels/objective are wrong."
  Blocked on persisting ground truth.
- **Token-sequence representation + cross-attention** — moderate/new sim; only the mean
  embedding is stored today; re-collect tokens and let the ranker attend over them.
- **Active collection at decision-critical (high-U, multimodal) states** — new sim; increase the
  fraction of informative discordant groups.
- **Free dense pretraining on `n_steps` / `return_target`** — cheap; auxiliary signal already
  stored.
- **Per-candidate uncertainty / consistency features** — cheap-to-moderate; use U or
  cross-candidate agreement as a scoring input rather than only as a gate.
- **Dropped (out of scope):** mid-trajectory / Operator-B scoring (mid-solve Tweedie views,
  mutation-consistency regularizers); grasp/contact label supervision (labels don't exist).

---

## 13. What a useful response looks like

- An independent read on Section 9's inferences — which you find solid, which you'd challenge,
  and what alternative explanations we may have overlooked.
- A prioritized set of next experiments across the cost tiers in Section 10, each with the
  hypothesis it tests and the observation that would confirm/refute it.
- Any reframing of the contribution if the selector cannot be made to work (e.g., leaning on
  the gate + honest negative), and the conditions under which you'd recommend that pivot.
- Ideas that exploit the frozen policy's available structure (endpoint prediction, shared
  prefix, U) without requiring new labels, if you see any.
