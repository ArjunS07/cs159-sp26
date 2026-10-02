# PI 2 independent research audit
Date: 2026-09-08  
Repository inspected: cs159-sp26, commit 6169b50  
Decision: substantially revise the research framing; retain the execution-aligned critic work.

**Verdict.** PI 1 correctly identifies the weak selector as a bottleneck and separates failure prediction from benefit prediction. However, its main novelty claim is substantially covered by omitted prior work, its always-search prerequisite is too strong, and its proposed branch experiment does not yet distinguish hindsight rescue from predictable improvement. I would fund a bounded investigation of **whether intervention outcomes can teach a reliable search gate under distribution shift**, with bare low-step sampling as an independent efficiency baseline. I would not fund the full menu of pruning, guidance, MCMC, and representation learning at once.

I read the complete [project handoff](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PROJECT_STATUS_HANDOFF.md) and [original PI 1 proposal](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08_PI1_ORIGINAL.md), inspected relevant runtime, data, training, splitting, and diagnostic code, and checked primary papers. This is a conceptual and source-code audit. It includes no new training, simulator outcomes, latency measurements, database checks, or access to the sealed cohort. Recorded experimental results remain attributed to the handoff.

**The largest correction: the proposed contribution already has direct competitors.**

PI 1's literature table covers many critic and sampling papers but misses the closest adaptive-compute work:

| Primary source | Relevant overlap | Consequence |
|---|---|---|
| [ELASTIC, June 30, 2026, v1](https://arxiv.org/html/2606.31132v1), §§4, 5, appendix 7.3–7.5 | Learns sequential and parallel compute allocation around frozen generative control policies, including pi0.5; uses a value-based task-quality reward. | Adaptive steps plus candidate counts and a success–latency frontier are already established goals. |
| [VLA-ATTC, May 2026, v1](https://arxiv.org/html/2605.01194v1), §§4.1–4.3, 5.3 | Gates extra candidates using disagreement between two proposals and selects with a relative action critic. | Uncertainty-triggered search with shared context is directly occupied. |
| [Dynamic Test-Time Compute Scaling in Control, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/49eadcc4a329fc6b74b9f8a82b78cbc3-Paper-Conference.pdf) | Uses observed task difficulty to adapt integration budgets and solver choices. | A difficulty-conditioned denoising schedule is insufficient novelty. |
| [RoboMonkey, 2025](https://arxiv.org/abs/2506.17811) | Studies VLA sampling, verification, and robustness outside the training distribution. | Robustness through sampling and verification is also established. |

ELASTIC's appendix generates counterfactual compute allocations whose terminal quality is assigned by its verifier. The narrower distinction proposed here is to supervise a simple search gate with **actual paired environment continuations**, and test where critic-predicted benefit differs from realized benefit. ELASTIC also evaluates real outcomes; we must not imply otherwise. Our question concerns the gate's training target and its reliability under shift.

VLA-ATTC's uncertainty analysis uses human difficulty judgments, and its preference-data argument partly relies on solver-step differences. This motivates testing measured intervention benefit directly. It does not prove that its gate is ineffective.

These are prospective distinctions, not established novelty. Paired effects, contextual decision rules, and selective intervention are standard ideas. A defensible result needs evidence that this particular supervision changes decisions and improves their value, or reveals a reproducible limitation of existing surrogates. Replacing pi0.5's critic input or moving to LIBERO-PRO by itself is too small a claim.

**What PI 1 gets right and should survive revision.**

The strongest parts are the ten-action intervention contract; the warning about Q50 trained on feedback-dependent executed sequences; separation of failure probability from computation benefit; skepticism about P&P posterior interpretations; and insistence on matched closed-loop evaluation. Keeping an actual default candidate available is sensible, although an inaccurate critic can still reject it incorrectly.

The representation concern is source-supported. [sampler.py:179](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/sampler.py:179) captures the output of embed_prefix before language-model prefill, and stores that original tensor afterward. It is not the contextualized representation discussed in [RL Token](https://arxiv.org/html/2604.23073v1). But these are already learned image/language features: “pre-prefill” does not mean raw pixels or wholly uncontextualized vision features. Contextualized features are a reasonable controlled comparison, not a promised fix.

The existing discounted Q10 target is a useful baseline with a different objective from finite-budget success. The current collection builder explicitly requires vanilla ten-action execution and disallows search/refinement; see [collection.py:18](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/pcp_search/collection.py:18). PI 1's mixed-policy warning is primarily a future replay-buffer concern, not evidence that the present immutable corpus mixes improved planners. Manifest-specific sampler settings still require checking.

**Always-search success is sufficient encouragement, not a necessary prerequisite.**

Suppose search improves success probability by 0.20 on 20% of states and reduces it by 0.05 on the remainder. Its average effect is zero. A gate that identifies the first subset improves expected outcome by 0.04 while searching only 20% of states, before charging the gate cost. Requiring a positive always-search average would reject this useful policy.

The correct prerequisite is an improvement that is **predictable from permitted information and survives independent evaluation**. This can be a global selector gain or a positive conditional gain for one prespecified simple gate. A retrospectively chosen subgroup with favorable outcomes is not enough.

The converse also matters: a selector can improve everywhere by the same amount, leaving little room for a learned gate to outperform random allocation at the same cost. Establish heterogeneity that is predictable, not merely heterogeneity in observed binary outcomes.

**The historical oracle is a hindsight bound, not measured recoverability.**

The implementation [diagnostics.py:98](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/diagnostics.py:98) takes the maximum observed binary outcome within each candidate group. For fixed state h and candidate set C, distinguish:

H = E_seed[max_j Y_j] - E_seed[Y_default]  
O = max_j E_seed[Y_j] - E_seed[Y_default].

H lets the choice depend on future continuation randomness. O picks using candidate success probabilities before that randomness is realized. H is at least O. A selector that knows the state and candidates but not future continuation noise cannot generally attain H. Two candidates can have equal expected value yet succeed on different continuation seeds, producing positive hindsight rescue.

A deterministic simulator does not remove this distinction: a stochastic frozen continuation policy supplies randomness. A fixed recorded continuation seed makes historical labels reproducible, but does not establish performance over fresh seeds. If every source of future randomness is intentionally fixed and available to the deployment rule, that is a narrower deterministic benchmark claim.

The reported 6.34-point uplift therefore motivates a repeatability audit. It is not a budget justification on its own. Report seed-specific hindsight headroom, repeated-seed candidate means, and outcomes from a frozen selector on independent continuations. Maximizing noisy sample means remains optimistic; selecting on one seed subset and evaluating on another measures an attainable selection procedure more honestly. With few seeds, state the remaining uncertainty.

Repeated continuations of one candidate set do not measure variability across fresh candidate pools. Add new proposal seeds on a small fixed subset, and use fresh proposal seeds in final closed-loop evaluation.

**The recorded low-step compute premise needs an explicit correction.**

The pilot's “three-step single” is behaviorally a vanilla three-step policy but computationally an instrumented one. Its config enables one five-iteration probe; see [coarse_refinement_experiment.py:20](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/coarse_refinement_experiment.py:20). For non-invasive strategies, [sampler.py:154](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/sampler.py:154) first obtains the original sampler's answer, then runs a measurement trajectory. Each probe iteration calls the velocity field; see [pnp.py:173](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/pnp.py:173).

Under these scalar-worker configurations, with no extra diagnostic branches, source-level expected counts per decision are:

| Recorded arm | Original sampler | Instrumented trajectory | P&P evaluations | Total |
|---|---:|---:|---:|---:|
| Historical stock10 uncertainty arm | 10 | 10 | 2 × 5 | 30 |
| Three-step single query | 3 | 3 | 1 × 5 | 11 |
| Three-step single refinement | 0 | 3 | 1 × 5 | 8 |
| Five-step single refinement | 0 | 5 | 2 × 5 | 15 |

These are code-derived evaluation counts, not newly measured runtime. Check logged counters before using them in an empirical table. The historical stock config is in [uncertainty_gradient_experiment.py:15](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/uncertainty_gradient_experiment.py:15).

A refined arm can appear cheaper than its unrefined comparator because only the latter duplicates sampling for measurement. Bare deployment counts are 10 and 3 for ordinary stock10 and ordinary three-step sampling. Three-step refinement still needs its five probe evaluations. Separate profiling must preserve that distinction.

The sampler deliberately returns the original action for measure-only runs and isolates probe randomness. This supports using their success results as evidence about low-step behavior; it does not make their recorded latency a bare-policy measurement. Confirm action equivalence in the intended deployment path.

Profile the actual deployed gate and sampler on the same GPU with common observations, warm-up, synchronized timing, fixed precision, and controlled batching. Include backbone prefill, critic, transfers, and all probes required for decisions. Disable diagnostic-only duplicate passes and disk/network logging. Report median and tail decision latency, total inference time per episode, and GPU/memory settings. Paused simulation does not establish performance when the physical world advances during inference.

**The learning experiment currently entangles several changes.**

Q10 versus a contextualized-feature Monte Carlo critic changes both representation and objective. A win would not identify which change helped. Use a small matched experiment:

| Representation | Success supervision |
|---|---|
| Current pre-prefill features | Same finite-budget Monte Carlo labels |
| Contextualized frozen features | Same labels, split, head, optimization budget |

Keep the existing Q10 TD model as a separate system baseline. If resources permit a second controlled comparison, hold representation fixed while comparing TD and Monte Carlo supervision. Avoid attributing the joint system difference to either factor alone.

Likewise, distinguish what a control tests. An action-only scorer can be useful even if the visual model does not beat it. Its success would refute a claim that visual conditioning is necessary, not the utility of search. A state-only scorer gives all candidates the same score; specify its tie rule. Use a default-preserving tie rule for the primary baseline and report random selection separately. Permute actions within state groups during control training to destroy candidate-label correspondence without corrupting the state distribution.

Do not make calibrated success probabilities a prerequisite for ranking. Calibration matters for probability claims and risk/benefit thresholds. Held-out realized utility can validate a selector without interpreting its raw score as a probability.

**Two data contracts need correction before spending heavily on training.**

First, terminal-dependent masks are a concrete train/deploy information mismatch, although exploitation has not been measured. [data.py:180](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/qplanning_critic/data.py:180) zeros and masks unexecuted positions after termination; [model.py:72](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/qplanning_critic/model.py:72) exposes validity through decoder attention. Simply setting every mask to true leaves terminal-dependent zeros. For Q10, recover the ten actions proposed before the branch from saved generated chunks, verify their executed portion matches the environment action convention, and supply proposal validity determined at decision time. Only known remaining-budget limits may shorten that input. Keep actual execution masks for targets and losses. Abnormal incomplete logging is not automatically task failure. For Q50, this patch does not repair the separate feedback/action semantics problem.

Second, finite-budget success depends on time remaining. The window stores start_step, but the current critic signature consumes no explicit remaining budget. Add budget information to the proposed success model and its controls. Do not silently change the immutable baseline or old checkpoint format.

The general instruction to keep identities together is correct, but not every helper guarantees it. [locked_candidate_split:500](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/data.py:500) partitions candidate groups, whereas candidate_cv_splits groups episodes. This is an API hazard, not evidence that the recorded V2 results leaked; current V2 generator calls the episode-safe CV path. Audit actual train/gate/audit manifests across data families, including repeated initial states and branch descendants. Bootstrapping states independently is only justified when those states are independent units; otherwise resample their source identities.

**A gate needs an observable information contract and honest labels.**

At the first gate, the default chunk may already exist. Its critic score, action summary, proprioception, task instruction, and remaining time are available. Spread across extra candidates is not available until those candidates have been generated. P&P also requires paid evaluations. A sensible design has a cheap first decision and an optional paid probe, with both costs counted.

For a fixed intervention A, train on paired outcomes D_A = Y_A - Y_default, not on the selector's predicted Q difference. Both signs matter: rescues are D=+1 and spoils are D=-1. Pairing with common random numbers can reduce variance, but does not make labels noiseless or create extra independent states.

Use disjoint identities for selector fitting, gate fitting/calibration, and a locked development audit, or nested identity-level cross-fitting. The selector used to create a gate-training choice must not have been fitted using that state's outcome labels. The audit must not select the gate threshold it evaluates. If the selector, candidate budget, solver, fallback, or continuation changes, the old gate target no longer describes the same intervention.

A compact gate can estimate conditional mean D, or separately estimate rescue and spoil frequencies and subtract them. Neither is a new causal estimator. The scientific test is whether its actual intervention policy outperforms simpler allocation rules.

On a fixed state panel with fixed continuation, the incremental value is E[g(X)D]. Under constant intervention cost and search frequency kappa, a state-independent random gate has value kappa E[D]. That is a useful diagnostic comparison. Closed-loop gates alter future states and remaining computation; the fixed-panel identity is not an episode policy-value formula. Evaluate the full policy with its actual budget.

**A narrower executable program.**

The revised [vision](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08.md) supplies the active sequence. Its key changes are:

1. Resolve deployment timing and learning-target contracts before a full critic sweep.
2. Use existing eligible data and a small repeated-continuation pilot to decide whether action choice has repeatable value.
3. Freeze one selector and one compute intervention. Allow either global or independently validated conditional gain to justify further work.
4. Compare direct outcome-supervised gating with uncertainty, default value, random allocation, and a critic-surrogate gate under the same candidate mechanism.
5. Require fresh end-to-end confirmation, with an explicitly separate shift test for any shift-generalization claim.

Use a first tranche of 48 independent source identities, one boundary each, four candidates, and two shared continuation seeds: 384 branch continuations. Choose states by a recorded rule before seeing their branch outcomes. A predeclared subset can receive extra seeds. This is a feasibility and variance pilot, not a powered efficacy study. At roughly historical headroom, only a handful of rescues may appear. Decide the next sample size from discordance, identity/task clustering, useful effect size, and GPU-hours; do not promise a useful gate from this first tranche.

A representative state panel must coexist with any uncertainty-balanced or failure-enriched diagnostic panel. If sampling probabilities are known, weight estimates to the declared target distribution. “Recoverable mistake” is an outcome-defined label and should not silently determine representative inclusion. If the target is a random boundary rather than a random episode, state that and account for episode-length weighting.

**Scope, publication standard, and stopping judgment.**

I would keep P&P as one candidate feature and defer progressive pruning, Q gradients, pCN, adaptive execution horizons, and online RL. The pCN invariant-distribution argument is mathematically reasonable for a fixed bounded scorer and deterministic sampler, but it does not solve the project's demonstrated problem: recognizing useful actions. Its mathematical neatness is not a reason to spend the next experiment on it.

The minimum credible positive result is a direct-outcome gate that beats cheaper, well-tuned gates at matched measured cost on fresh identities, with rescue/spoil accounting and a separate shift test. A within-operator ablation establishes the target's effect. Any stronger claim of outperforming adaptive compute methods requires a faithful or clearly labeled resource-matched ELASTIC comparison, including training interaction costs.

A credible diagnostic result is that substantial hindsight rescue collapses under independent continuations, or that recognizable action value fails under shift despite useful proposal headroom. That result needs sufficient precision and replication across several task/shift settings. One failed critic or a nonsignificant 48-state pilot is not a general limitation of frozen VLAs.

Keep three outcomes distinct: useful benefit established; practically useful benefit ruled out to a declared margin; insufficient evidence within the available budget. A confidence interval crossing zero does not, by itself, distinguish the latter two.

**PI 2 decision.** Keep the scientific ambition, but change what earns the next unit of effort. The asset is the ability to measure counterfactual robot outcomes with controlled provenance. Use it to test whether computation benefit is predictable and transferable. Another search mechanism should wait until that evidence identifies a reason to build one.
