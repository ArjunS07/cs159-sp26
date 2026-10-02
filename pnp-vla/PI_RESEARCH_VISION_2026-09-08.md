# Research vision after PI 2 review: learn when search changes robot outcomes
Date: 2026-09-08  
Status: revised research proposal, not an experimental result.

**Recommendation.** Study whether a simple gate trained on paired simulator outcomes can allocate search more reliably than uncertainty or critic-score surrogates, especially under distribution shift. Keep the VLA frozen, execute ten actions before replanning, and begin with a fixed small candidate selector. Treat cheap ordinary sampling as an independent efficiency baseline. Make additional search mechanisms contingent on evidence.

This revision incorporates the [independent PI 2 audit](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI2_RESEARCH_AUDIT_2026-09-08.md). The [original PI 1 proposal](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08_PI1_ORIGINAL.md) is preserved. Recorded results come from the [handoff](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PROJECT_STATUS_HANDOFF.md); no new rollout, training, runtime measurement, or sealed-cohort analysis was performed for this revision.

**What the project can currently claim.**

The infrastructure is stronger than the evidence for an improvement method. The handoff records an immutable 1,880-rollout critic corpus, replayable candidate interventions, fixed policy provenance, and paired evaluation identities. These support a controlled study.

The conditioned verifier did not beat its action-only control in Stage 2. That weakens the particular conditioning claim, not every possible selector. No completed Q10/Q50 training result was documented at handoff.

On the repeatedly used 220 identities, stock10 succeeded 117 times, three-step sampling 118 times, and three-step refinement 120 times. Three-step refinement's stock-relative interval was [-2.27, 5.45] percentage points. These are exploratory results compatible with benefit, negligible effect, and some harm. The three-step result motivates a proper efficiency experiment; it does not establish noninferiority.

Historical oracle uplift of about 6.34 percentage points is the gain from choosing the best observed candidate outcome. It is a hindsight statistic on an uncertainty-enriched development pool. It is not episode-level headroom or an estimate of attainable fresh-seed improvement. Repeat continuation seeds before using it to justify a large search program.

**The prospective contribution and its closest competition.**

Adaptive inference compute is already occupied. [ELASTIC](https://arxiv.org/html/2606.31132v1) learns joint denoising and candidate allocation for frozen policies, including pi0.5, using value-based task-quality feedback. [VLA-ATTC](https://arxiv.org/html/2605.01194v1) uses proposal disagreement to gate candidate search and a relative critic to select. [Dynamic Test-Time Compute Scaling in Control](https://papers.neurips.cc/paper_files/paper/2025/file/49eadcc4a329fc6b74b9f8a82b78cbc3-Paper-Conference.pdf) adapts solver configurations to difficulty. [V-GPS](https://arxiv.org/abs/2410.13816) establishes frozen-policy value reranking, and [RoboMonkey](https://arxiv.org/abs/2506.17811) studies VLA sampling and verification under distribution shift.

The proposed distinction is therefore narrow: **train the allocation decision from realized paired rescue and harm, and determine whether it transfers better than surrogate allocation signals.** ELASTIC's appendix uses verifier-scored counterfactual compute allocations; that is a particularly relevant comparison to direct environment-outcome labels. We do not claim that these earlier systems lack end-to-end evaluation.

The project should answer three questions:

1. Do alternative action prefixes have repeatable differences in success probability?
2. Can a deployable selector recognize those differences, globally or on a predictable subset?
3. Can a cheap observable gate identify that subset under fresh initializations and a declared shift?

A positive answer to only the first question is an oracle study. A positive answer to the first two is a selector result. The third is the adaptive-compute result. None is guaranteed to be a standalone publication.

**Fix the intervention and objective.**

Let h denote available history, instruction, proprioception, and remaining environment-step budget. Let u be the next ten proposed actions. Define

q_ref(h,u) = P(success before the episode budget expires | execute u, then follow pi_ref).

For the initial branch study, pi_ref is one pinned vanilla stock10 sampler with ten-action execution, fixed preprocessing, and a recorded continuation-seed scheme. Stop a prefix if the environment terminates. This objective differs from the existing per-step-discounted TD return. Preserve the existing Q10 model as a named baseline rather than silently changing its semantics.

An unused generated suffix is not part of the ten-action intervention. Make the proposed deployment critic depend only on the executed prefix. Q50 trained on fifty actions assembled across five feedback-driven replans is not a direct value estimator for one generated fifty-action plan whose first ten actions are executed. Long-to-short value learning remains possible with an explicitly derived target; [Decoupled Q-Chunking](https://arxiv.org/abs/2512.10926) supplies relevant precedent.

For a fixed candidate set, retain the default and select an actual completed candidate. Use deterministic argmax with a default-preserving tie rule initially. Do not add weighted action averaging, a temperature sweep, and fallback thresholds simultaneously. Inclusion of the default is an option, not a guarantee against harm.

**Repair the input contract before interpreting learning curves.**

The current sampler stores pre-prefill embeddings rather than contextualized language-model outputs; see [sampler.py:179](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/sampler.py:179). Compare those representations using the same compact action head, labels, split, and training budget. Start with masked pooled contextualized features; freeze the backbone. If saved embeddings, attention masks, positions, and the pinned model suffice, reconstruct features on a small fixed subset before processing the whole corpus. Validate equivalence to an online extraction. Do not undertake reconstruction-pretrained compression merely to resemble [RL Token](https://arxiv.org/html/2604.23073v1).

The current dataset exposes action truncation after observed termination; see [data.py:180](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/qplanning_critic/data.py:180). A new proposal does not come with knowledge of when it will succeed. Recover its complete pre-decision ten-action proposal where available, check normalization and agreement with executed actions, and separate target validity from scorer input validity. Replacing masks alone leaves terminal-dependent zero padding. Add known time remaining to the success model. Missing/corrupt records need explicit handling rather than failure labels.

Keep this representation comparison under common Monte Carlo success supervision. Compare the existing TD Q10 system separately; otherwise differences in representation and objective become inseparable. Prefer a compact model before a broad architecture sweep. Use action-only, state-only, random-selection, and within-state shuffled-action controls. A successful action-only selector remains scientifically useful even if conditioning adds nothing.

**The first experiment should establish repeatable action value.**

Reuse compatible existing development branches first, checking their execution and continuation contracts. Build a new pilot of 48 independent source identities with one boundary per identity, chosen by a recorded sampling rule before observing branch outcomes. At each boundary, collect the default plus three fresh policy candidates, and evaluate each under two shared continuation seeds: 384 branch continuations.

This is a proposed feasibility cap, not a powered efficacy sample. Use a fixed subset for additional continuation seeds and a fresh candidate pool if resources allow. Measure branch wall time, discordance, task variation, and label repeatability to plan the next tranche. Do not train a large gate or declare failure from a handful of rescue events.

Maintain a representative panel for policy-value estimates. A separate balanced diagnostic panel may oversample uncertainty levels or difficult phases, with explicit inclusion probabilities and weights where population estimates are desired. “Recoverable” is an outcome, not a pre-collection sampling category.

Report four quantities separately:

- Default and random-candidate success.
- Hindsight oracle success using observed labels.
- Repeated-seed differences between candidate success probabilities, with uncertainty.
- A frozen selector's fresh-seed gain, decomposed into rescue and spoil.

Evaluate nested candidate counts on the same eligible states. Increasing candidate count changes both proposal opportunity and exposure to critic errors. A critic trained on ten-step samples must also be evaluated on three-step proposals before using it to select them.

**Do not require always-search to win before testing selectivity.**

Always-search can have zero mean gain while helping one predictable subset and harming another. It is enough to establish either a global gain or an independently validated positive conditional gain for a simple prespecified rule. Keep always-search as a comparator.

Avoid unrestricted subgroup mining. Fit the selector on training identities, the gate and threshold on separate development identities, and evaluate both on locked audit identities. Nested identity-level cross-fitting is an alternative when data are scarce. The selector generating a gate-training choice must not use that state's outcome labels.

For fixed search intervention A, define paired labels

D_A = Y_A - Y_default,

and learn the conditional mean of D_A from information available at the allocation decision. Separate rescue and spoil predictors are an interpretable alternative. Train against realized outcomes, not Q_A - Q_default. A learned score need not be calibrated to rank candidates, but probability claims and risk thresholds require separate calibration.

A cheap first gate can use the already generated default action, its critic score, proprioception, instruction, remaining budget, and inexpensive summaries. Additional-candidate spread and P&P disagreement require paid computation. Treat a paid probe as an optional second decision whose cost is charged even if search is rejected. No feature may use future success, terminal masks, or outcomes of unexecuted branches.

At a common panel distribution, E[g(X)D_A] is the gate's incremental branch value. Under equal search cost and frequency kappa, a random gate gives kappa E[D_A]. This diagnostic isolates allocation quality. Repeated deployment changes the state distribution and must be tested separately.

**Keep the initial operator small and make compute accounting real.**

The first selection study should use one fixed solver and four completed candidates, including the default. In parallel, profile bare single-sample three-step and ten-step inference. If the cheap sampler preserves useful proposals, test cheap best-of-four with the same validated scorer. Freeze that operator before fitting its gate; changing solver or candidate count changes the gate's target.

The current “three-step single” measurement path runs an original three-step sampler, a separate three-step telemetry trajectory, and five probe calls. Source-level counts are therefore 11 evaluations for recorded three-step single, 8 for three-step refinement, and 30 for the historical instrumented stock10 arm. These counts are not bare deployment costs or new measured results; the audit traces their code. A bare three-step baseline uses three sampler evaluations, and a bare stock10 baseline uses ten.

Benchmark the deployed path, counting all required probes, critic passes, context encoding, and transfers. Compare on the same hardware and observation panel with synchronized timing and common batching assumptions. Keep diagnostic logging outside deployment latency. Report mean/median and tail decision latency, total episode inference cost, and executed steps. Training interaction and feature-extraction costs belong in a separate resource table.

The primary claim concerns measured inference efficiency in paused simulation. Real-time control under observation delay would require a separate experiment.

**The minimal comparison set.**

For the selected candidate mechanism, compare:

| Policy | What it tests |
|---|---|
| Bare stock10 and bare three-step single | Whether search beats ordinary sampling or only an inefficient implementation |
| Always-search | Total value and cost of the fixed selector |
| State-independent random gate | Whether allocation adds value at the same expected cost |
| Default-value gate | Whether a cheap predicted difficulty signal suffices |
| Uncertainty gate | Whether P&P or proposal disagreement earns its measurement cost |
| Critic-surrogate benefit gate | Whether the gate target can be the selector's predicted advantage |
| Direct outcome-supervised gate | Whether paired environment labels improve allocation |

Use the same candidate selector and available information for the two benefit-target gates as far as possible. Match training-state coverage; report the extra environment labels required by direct supervision. Give inexpensive heuristics the same threshold-selection protocol. A gate using two proposals must charge both, even when it exits early.

Equal search frequency alone does not ensure equal end-to-end cost: probe overhead, batching, and changed episode lengths matter. Choose thresholds on development data, report realized cost on fresh episodes, and compare attainable frontier points with uncertainty. Do not retrospectively retune on the test set.

The same-operator controls establish mechanism. Claims against published adaptive-compute systems additionally require a faithful or clearly labeled resource-matched ELASTIC comparison. An uncertainty gate with our critic is an ablation inspired by VLA-ATTC, not a reproduction of its relative critic.

**Decision sequence and scope.**

| Stage | Deliverable | Decision |
|---|---|---|
| 0 | Input/target contract audit, deployment timing, representative panel manifest | Is the learning/economics premise coherent? |
| 1 | Small repeated-continuation panel and existing-data analysis | Is there repeatable candidate value worth learning? |
| 2 | Fixed Q10 baseline and matched compact feature comparison | Can any deployable selector improve globally or on a validated subset? |
| 3 | Frozen-operator gate-target comparison on locked development identities | Does actual-outcome supervision beat simpler signals after cost? |
| 4 | Fresh closed-loop comparison and separate shift evaluation | Does the observed benefit survive repeated use and shift? |
| 5 | One frozen confirmatory contrast with a declared effect margin | Is the selected result supported at useful precision? |

Stop spending on selector complexity if adequately precise evidence rules out a practically useful gain. If data are too sparse to distinguish gain from noise, record an inconclusive result and its resource requirement. Stop P&P development if it adds no net value over cheaper features. If only cheap sampling succeeds, pursue an explicitly scoped noninferiority/efficiency result. If repeatable action value exists but current models cannot recognize it, investigate representation or supervision; do not label all frozen-policy search futile.

Defer progressive seed pruning, gradient guidance, noise-space MCMC, adaptive action horizons, and full online RL. [Progressive Seed Pruning](https://arxiv.org/abs/2607.21591), [QGF](https://arxiv.org/html/2606.11087v1), and [QPILOTS](https://arxiv.org/html/2606.14801v1) supply relevant later baselines. P&P perturbations are not automatically posterior draws, critic gradients need intervention validation, and an exact MCMC target based on an inaccurate critic does not solve action selection. These are useful constraints on follow-up work, not additional tasks in the current program.

**What would make the result credible.**

Define the target population and a useful success gain or noninferiority margin before sizing confirmation. Preserve whole source identities across selector, gate, audit, and confirmation partitions; cluster multiple states and seeds by their source identity. Use task-level summaries for claims across tasks. Thousands of candidate pairs are not thousands of independent episodes.

The repeatedly analyzed 220 identities are development data. Position perturbations already inspected during tuning are also exposed, even when excluded from critic training. Fresh initializations on those tasks can test within-task generalization; a stronger unseen-shift claim needs a separately withheld shift family. Keep the sealed verifier cohort under its existing protocol and use an appropriate fresh cohort for the new closed-loop method.

Freeze candidate generation, scorer, gate features, thresholds, budget, random-seed scheme, and primary contrast before confirmation. Do not sum one-step branch effects to predict whole-episode success. Report paired episode success, rescue/spoil, runtime, and concentration of gains by task/shift.

The decisive evidence is a direct-outcome gate that captures more realized improvement at the same measured inference cost than well-tuned cheaper rules, and retains that advantage under the stated generalization test. A careful negative result can instead identify whether the bottleneck is hindsight-only headroom, selector error, uninformative gate features, or distribution shift. The project should be judged by which of those possibilities it resolves.
