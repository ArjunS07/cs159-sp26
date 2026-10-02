# PI3 final research plan: learn the value of search for frozen VLAs

Date: 2026-09-08  
Decision: fund a bounded investigation of outcome-supervised compute allocation, with an independent low-step efficiency track.  
Status: research recommendation. No new training, simulator experiments, timing measurements, database audit, or sealed-cohort evaluation was performed for this document.

**My recommendation is to study when a frozen VLA has better actions available, whether a selector can recognize them, and whether the robot can predict that benefit before paying for search.** Use paired simulator interventions to answer those questions in that order. The prospective method is deliberately small: a frozen pi0.5, a critic for the next ten executed actions, four completed policy candidates, and a cheap gate trained on realized improvement and harm.

The scientific ambition is larger than that implementation. We want to establish the conditions under which inference compute buys reliable control improvement. In particular, can outcome supervision teach an allocator to account for errors in its own selector, including under distribution shift? That is a meaningful question in the current literature. A different P&P schedule, another critic architecture, or a small gain on the familiar 220 identities would not resolve it.

I agree with PI2's narrowing of the program and the revised PI1 plan. My additions are to make selector error the central mechanism, separate hindsight from repeatable value more aggressively, test whether extra outcome labels are best spent on the gate or the critic, and move a small closed-loop check earlier. I would give this program an initial two-week feasibility window, conditional on GPU availability, followed by continuation only when the experiments justify it.

I read the complete [revised PI1 recommendation](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08.md), [original PI1 recommendation](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08_PI1_ORIGINAL.md), [PI2 audit](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI2_RESEARCH_AUDIT_2026-09-08.md), and [project handoff](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PROJECT_STATUS_HANDOFF.md). I also inspected the relevant sampler, collection, critic-input, target, and runtime configuration code and checked primary literature. The revised PI1 document incorporates PI2; their agreement is a synthesis, not three independent pieces of empirical evidence.

**The existing evidence supports investment in a diagnostic experiment, not a claim that the method works.**

The handoff records an unusually useful asset: replayable branches, pinned policy provenance, and 1,880 critic-training rollouts containing 1,054 successes and 826 failures. It also records approximately 1,532 distinct episode/initial-state groups. The effective learning sample is therefore much smaller than the number of action windows.

The conditioned verifier's Stage 2 ranking accuracy was 0.5959, versus 0.5965 for action-only. That is evidence against the demonstrated benefit of that conditioning implementation. An action-only selector might still improve decisions, and its usefulness should be evaluated on its own terms.

The September stock10, three-step single, and three-step refinement arms succeeded on 117, 118, and 120 of the same 220 identities. The refinement-versus-stock interval was [-2.27, 5.45] percentage points. These are exploratory results with substantial uncertainty and repeated method selection. The most attractive inference is that ordinary three-step generation deserves an efficiency experiment. Neither equivalence nor a speedup has been established.

The historical 6.34-point oracle uplift is a maximum over observed branch outcomes in an uncertainty-enriched pool. It may contain both real action-quality differences and selection on future randomness. No completed Q10/Q50 training result was documented at handoff. These distinctions determine the next experiment; they are not reasons to dismiss the project.

**The literature supports the problem, but substantially raises the novelty bar.**

The broader VLA trend is to combine strong pretrained behavior with experience, task context, and mechanisms for choosing or refining actions. pi0.5 derives its generalization from heterogeneous co-training; RECAP incorporates autonomous experience and corrections into policy training; pi0.7 emphasizes rich context conditioning and steering. My inference is that a study on a pinned pi0.5 checkpoint should explain a transferable control principle, while describing its empirical scope precisely. It should not equate one LIBERO-finetuned checkpoint with the frontier of generalist robotics. [pi0.5](https://arxiv.org/abs/2504.16054v1), [RECAP / pi*0.6](https://arxiv.org/abs/2511.14759v2), [pi0.7](https://arxiv.org/abs/2604.15483v2).

| Prior work checked | What it establishes or directly overlaps | Consequence for this plan |
|---|---|---|
| [V-GPS](https://arxiv.org/abs/2410.13816) and [RoboMonkey](https://arxiv.org/abs/2506.17811) | Value-guided frozen-policy selection; VLA sampling and verification, including generalization settings. | Best-of-N and improved robustness from verification are established baselines. |
| [DA-SIP, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/49eadcc4a329fc6b74b9f8a82b78cbc3-Abstract-Conference.html) | Difficulty-dependent integration budgets and solver choices. | Adaptive denoising alone is insufficient novelty. |
| [VLA-ATTC, May 2026, v2](https://arxiv.org/html/2605.01194v2) | Two-proposal disagreement triggers search; a relative critic selects candidates. Its difficulty validation includes human judgments. | Compare uncertainty with measured search benefit. An uncertainty gate using our critic is an ablation, not a reproduction of RAC. |
| [ELASTIC, June 2026, v1](https://arxiv.org/html/2606.31132v1) | Joint sequential/parallel allocation for frozen generative policies, including pi0.5. Counterfactual compute allocations receive verifier-derived quality labels. | The closest allocation competitor. Direct environment-outcome targets are a narrow distinction requiring evidence, not a first claim to adaptive compute. |
| [FASTER: Value-Guided Sampling for Fast RL, April 2026](https://arxiv.org/html/2604.19730v1) | Learns denoising-space values to filter candidates, with a practical initial-noise selector and VLA experiments. | Early seed selection is already directly occupied. This is a more relevant later comparator than relying only on image-generation pruning analogies. |
| [RL Token, April 2026, v2](https://arxiv.org/html/2604.23073v2) | Compresses final-layer VLA representations for small online actor-critic heads; allows execution chunks shorter than generated horizons. | Motivates a contextualized-feature baseline. The local readout token is a different construction. |
| [Decoupled Q-Chunking](https://arxiv.org/abs/2512.10926v2) | Separates long value propagation from shorter policy chunks through a derived partial-chunk critic. | Longer backups can be useful; they require explicit semantics. |
| [QGF](https://arxiv.org/abs/2606.11087v1) and [QPILOTS](https://arxiv.org/abs/2606.14801v1), June 2026 | Test-time value-gradient steering of frozen flow policies; QPILOTS includes clean-estimate and learned posterior-sample variants. | Any later guidance project needs these close comparisons. |
| [Q-VGM, August 24 revision](https://arxiv.org/abs/2606.08015v3) and [Q-Planning, August 2026](https://arxiv.org/html/2608.21204v1) | Q-VGM uses IQL/online TD and critic-derived velocity targets to update the policy. Q-Planning keeps the policy frozen and improves a Q-based planner through deployment data. | Keep the adaptation regimes distinct. The repository's Q-planning encoder is an adaptation of the paper's independent DinoV2/T5 design. |

The June and earlier papers define the mid-2026 context; the August papers are relevant updates available by this plan's date. Paper results above are the authors' reports, not replications here.

The proposed contribution is: **a controlled account of when search's predicted advantage disagrees with its realized advantage, and evidence that a compact allocator trained on paired outcomes can exploit that distinction at useful cost.** Distribution-shift transfer is a hypothesis to test. Direct outcome supervision does not confer automatic robustness.

**The right theoretical object is the value of a specified intervention.**

Let h contain the available history, instruction, proprioception, and remaining episode budget. Let u be the proposed ten-action prefix. Fix a reference policy pi_b, initially vanilla ten-step denoising with ten-action execution. Define

q_b(h,u) = P(success before the remaining budget expires | do(execute u), then follow pi_b).

Stop execution if the environment terminates. Remaining time is part of the state. An undiscounted terminal success objective is intentional: the existing gamma=0.99-per-environment-step TD objective also prefers earlier success, and values success 100 steps away at about 0.366 of immediate success. Preserve that objective as a separately named TD comparison.

For a candidate set C containing default u_0, let j* maximize true q_b and let j_hat be the deployable selector's choice. Define

O(h,C) = max_j q_b(h,u_j) - q_b(h,u_0),  
R(h,C) = max_j q_b(h,u_j) - q_b(h,u_j_hat),  
Delta(h,C) = q_b(h,u_j_hat) - q_b(h,u_0) = O(h,C) - R(h,C).

This elementary decomposition is the organizing principle:

- O measures the opportunity supplied by the proposal distribution and candidate budget.
- R measures the selector's regret on those candidates.
- Delta is the actual intervention benefit before compute cost.

Replay fixes a full simulator state. A repeated-seed oracle at that state may distinguish hidden conditions unavailable to the deployment history h; treat it as a diagnostic ceiling, and evaluate every learned choice using only permitted observations. More candidates can increase opportunity and simultaneously expose larger scoring errors. Better discrimination between easy and hard states need not reduce selector regret. If the default is already best, O is zero even in a difficult state.

There is a particularly revealing diagnostic for argmax selection with the default included:

predicted_Delta = max_j q_hat(h,u_j) - q_hat(h,u_0) >= 0.

The selector always believes its chosen replacement is at least as good. Actual Delta can be negative. An allocator trained only on this predicted margin can learn a compute threshold, but its targets never directly express harmful selection. Outcome labels can express that harm. Whether the harm is predictable from cheap features is the empirical question. This observation applies to this specific surrogate, not every possible allocation algorithm.

For a fixed state distribution, define X as information available before buying extra candidates, D = Y_search - Y_default, m(X)=E[D|X], and incremental search cost c(X). With a fixed per-unit compute price lambda and constant gate overhead omitted from the choice, the optimal binary decision is

g*(X) = 1{m(X) > lambda c(X)}.

For deterministic binary g, its regret relative to this oracle is

E[ |m(X)-lambda c(X)| 1{g(X) != g*(X)} ].

This explains why useful decision discrimination matters more than globally excellent failure AUC. It also explains why always-search need not win: m can be positive on a predictable subset and negative elsewhere. At constant cost and search fraction kappa, random allocation gains kappa E[D]; a useful gate must exploit predictable heterogeneity beyond that comparator.

These are standard decision-theoretic identities, not proposed theoretical novelty. A hard episode-level compute budget introduces remaining compute into the decision state; the binary local rule is then an approximation. Start with a priced-compute frontier, and describe any deterministic cap and fallback explicitly.

There is also a precise connection to control. With time included in the state, the finite-horizon performance-difference identity evaluates reference-policy advantages along the states visited by the *new* policy. Pointwise nonnegative expected reference advantage would imply policy improvement. Positive average advantage on old baseline states does not give that guarantee. Repeated deployment can move the robot into poorly covered states, so branch gains must be checked in closed loop.

**P&P is a possible measurement, and its mathematical interpretation must stay modest.**

Disagreement can reflect several successful approaches, a consequential grasp ambiguity, or uniformly unsuccessful behavior. Under a local smooth approximation, action covariance Sigma affects value variation through grad(q)^T Sigma grad(q); a scalar action-spread measure discards that directional information. Even value variation does not by itself establish that a particular selector benefits from more candidates.

The inspected [P&P probe](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/pnp.py:138) forms sequentially dependent predict–perturb updates. Its contraction measures self-consistency of that procedure. It is not automatically epistemic uncertainty, a posterior probability, or evidence of increasing task value.

For a specified stochastic generative process, reward-tilted steering involves an intermediate soft value such as tau log E[exp(q_b(h,U)/tau) | X_t=x]. A critic evaluated at an average clean estimate generally differs from this quantity. SVDD and Feynman–Kac steering supply the relevant foundations. With a fixed deterministic ODE solver, the endpoint from a full intermediate state is fixed; adding P&P noise changes the process. These facts motivate empirical tests, not a posterior-sampling claim for the current implementation. [SVDD](https://arxiv.org/abs/2408.08252v5), [Feynman–Kac steering](https://arxiv.org/abs/2501.06848v5).

Keep U10 as the primary P&P feature because ten actions are executed. Retain U20/U50 as historical or diagnostic comparisons. I expect cheap state/action features to absorb much of P&P's useful signal; P&P should remain only if its incremental decision value exceeds its measured cost.

**Experiment 0 repairs the information contract and establishes the economics.**

This is the first work package, with a target of two to three engineering days before expensive collection. The duration is a planning estimate.

| Contract | Required action | Acceptance check |
|---|---|---|
| Representation | Compare current pre-prefill features with frozen contextualized prefix outputs. | On a small fixed subset, reconstruct from saved inputs and compare with online extraction at declared numerical tolerance. |
| Proposal input | Score complete prefixes known before execution, including actions after an eventual early success. | Recover saved generated chunks, verify executed-prefix agreement and normalization; future termination cannot affect scorer inputs. |
| Target | Construct finite-budget success labels with known remaining time. | Separate true failure/time-limit outcomes from incomplete or corrupt logging. |
| Continuation | Pin checkpoint, sampler, execution horizon, environment budget, and RNG scheme. | Replayed identical branches reproduce their outcome under the same seed; continuation RNG is independent of candidate-generation RNG. |
| Data separation | Build source-identity/branch-descendant groups across all data families. | No underlying initialization group crosses selector, gate, audit, or confirmation boundaries. |
| Runtime | Benchmark bare three- and ten-step sampling, then actual candidate scoring. | Count all required context encoding, denoising, critic passes, probes, transfers, and gate inference. |

The [sampler](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/sampler.py:179) saves embed_prefix output before the language-model prefill. Those are learned features, but they are not the contextualized final-layer features used in RLT. Begin with masked pooling, keeping image and language summaries identifiable where practical, and the same compact action head in both arms. If pooling fails, that does not rule out useful information in the tokens.

The [window builder](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/qplanning_critic/data.py:180) truncates and zero-pads actions at observed termination; the [critic](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/qplanning_critic/model.py:72) exposes validity in attention. Making masks all-valid alone leaves the terminal-dependent zeros. Separate proposal-input validity from execution/target validity. Do not spend a full training budget on a known train/deploy mismatch merely to complete an old notebook. Preserve legacy artifacts and run a versioned, repaired TD baseline.

Q50 assembles actions produced across five feedback-dependent replans. It should not score a single generated 50-action plan as if that were the same intervention. Keep the new scorer suffix-invariant. If long backups later become necessary, derive a reference-policy multistep target or a DQC-style auxiliary objective.

The present collection code enforces vanilla ten-action execution and excludes refinement/search; mixed-planner replay is a future concern, not a demonstrated defect in this corpus. Check manifest-specific solver settings before pooling. [Collection contract](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/pcp_search/collection.py:18).

For runtime, use one pinned GPU/software setup, warmup, synchronized measurements, shared-context candidate batching, and the same observation panel. Disable diagnostic-only duplicate passes and disk/network logging. Measure mean, median, and p95 decision latency, plus total inference time and environment steps per episode.

Code inspection confirms the distinction emphasized by PI2: the recorded three-step measurement arm has 3 original + 3 telemetry + 5 probe evaluations; refinement has 3 + 5; the instrumented stock arm has 10 + 10 + 10. Bare deployment counts are 3 and 10. These are source-derived counts, not measured speedups. [Sampler paths](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/sampler.py:154), [coarse configurations](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/coarse_refinement_experiment.py:20).

The deployment claim initially concerns inference efficiency in paused simulation. Physical reaction latency requires an observation-delay experiment and suitable control execution; it cannot be inferred from that benchmark.

**Experiment 1 asks whether the action opportunity survives future randomness.**

Use eligible historical branches for debugging and training where their contracts match. For the new pilot, sample 48 independent source identities and one baseline decision boundary per identity. Define the target population before sampling: for example, a macro-weighted distribution over declared tasks/suites, then a random initialization and a uniformly sampled nonterminal boundary from its baseline trajectory. This is an episode-uniform boundary distribution, not the deployment-frequency distribution over all time steps.

At each boundary, collect the default plus three independent stock10 candidates. Execute each prefix and continue with the pinned stock10 reference under two shared continuation seeds. This is 48 x 4 x 2 = **384 branch continuations**, plus any source rollouts and replay overhead.

Make the repeatability extension concrete rather than leaving it as an unspecified option:

| Tranche | Design | Additional branch continuations |
|---|---|---:|
| Initial breadth | 48 identities x 4 candidates x 2 continuation seeds | 384 |
| Repeatability | 12 identities selected before outcomes; same 4 candidates; 6 additional seeds | 288 |
| Proposal-pool check | Same 12 identities; an independent second pool of 4 candidates; 2 fresh continuation seeds | 96 |
| Total if both extensions run | All of the above | 768 |

The first 384 runs establish functionality, cost, and variance. Commit the extensions after replay and runtime checks pass. Neither tranche is a powered efficacy test or enough data for a broad gate. If the measured budget cannot support repeatability, mark the inference correspondingly limited.

At fixed h and C, distinguish

H = E_seed[max_j Y_j] - E_seed[Y_0],  
O = max_j E_seed[Y_j] - E_seed[Y_0].

H >= O. If equal-quality candidates succeed under different continuations, H can be positive while O is zero. Maximizing a few noisy sample means also overestimates O. Select an empirical best candidate on one seed subset and evaluate it on an independent subset; this measures the attainable performance of that empirical selection procedure, not the exact oracle. Report candidate means and uncertainty separately.

The outputs are default/random success, seed-specific hindsight uplift, cross-seed selection gain, a fixed learned selector's gain, and paired rescue/spoil counts. Use nested K=1,2,4 subsets on the same states. Additional candidate pools test whether results depend on a lucky proposal set.

Shared random numbers are an experimental coupling. They can reduce variance, but may not. Rescue and spoil probabilities depend on that coupling; their difference is the marginal success gain. Keep the coupling fixed and use marginal success difference as the primary quantity. Do not describe individual paired rescues as invariant facts about a state.

A separately labeled diagnostic panel can balance uncertainty or task phase. It cannot replace representative evaluation. Use inclusion weights when claiming target-population averages, and never define representative inclusion by future “recoverability.”

**Experiment 2 tests a compact selector before expanding the search algorithm.**

Reuse the 1,880-rollout corpus for initialization, with identity-safe splits and known continuation semantics. Treat its many correlated windows as repeated information from a limited number of source identities. Fit on observational rollout returns, then use compatible training-only branch data to learn and assess within-state action discrimination.

Start with the following small comparison:

| Model | Supervision | Purpose |
|---|---|---|
| Repaired Q10 TD baseline | Existing discounted TD objective, with deployment-valid inputs | Establish what the implemented approach can do. |
| Compact critic on current features | Finite-budget Monte Carlo success | Baseline for the representation comparison. |
| Same compact critic on contextualized features | Identical success labels, head, split, and optimization budget | Isolate feature location. |
| Action-only and state-only controls | Same success objective and known budget | Separate action priors from state difficulty. |
| Within-state shuffled-action control | Destroy action/outcome correspondence only within training groups | Detect shortcuts in candidate discrimination. |

Use a compact temporal action encoder and small fusion head first. Fix the recipe, use three training seeds, and inspect learning curves as independent training identities increase. Use binary cross-entropy for success prediction; no categorical-head entropy interpretation is needed. Keep the large existing decoder as a system baseline rather than the default architecture search.

Hold the initial candidate operator to four completed samples, deterministic argmax, and a default-preserving tie rule. State-only therefore chooses the default; report random selection separately. Retaining the default permits a conservative choice, but does not guarantee one.

The primary metric is held-out selected-outcome gain over default, with gain over random as a control. Report within-state ranking, calibration, and rescue/spoil as diagnostics. Global failure AUC and Bellman error cannot certify a useful selector.

If the compact critic is promising but action distinctions remain weak, allow **one bounded repair**: either an action-conditioned token-attention readout or an additional within-state ranking loss, chosen from the observed failure mode. Keep the other factors fixed. A privileged simulator-state critic can help localize an observability/representation bottleneck; its failure is not proof that the task is unlearnable. No privileged state may enter deployment.

Proceed when there is independently validated global gain or gain on a simple rule chosen before its audit. Always-search does not need to improve on average. If evidence is imprecise, expand the number of independent identities according to the measured variance; avoid an unbounded architecture sweep.

Run a small closed-loop development check as soon as one selector is credible, before investing heavily in gate training. Compare default, always-search, and one prespecified selective rule on fresh development episodes. If improvement disappears, inspect the newly visited states and remaining-time distribution. Permit one targeted data-coverage repair, then re-audit on new identities. Preserve the reference-policy label definition throughout.

**Experiment 3 determines whether cheap proposals improve the compute frontier.**

The minimal solver/candidate matrix is (3 steps, 1 candidate), (10,1), (3,4), and (10,4). First evaluate their completed proposals on a shared development panel, then advance promising arms to paired episodes. A three-step sample is a sample from a different numerical sampler; lower solver error is not synonymous with higher task success.

Use the same scorer to expose proposal-distribution effects, while checking its ranking on each solver. If additional three-step training examples are needed, state that as a separate matched-data adaptation. Equal critic architecture does not imply equal critic accuracy across proposal distributions.

One useful outcome is that four cheap completed samples preserve the alternatives of expensive search. Another is that a bare three-step policy is already the best practical choice. Both merit reporting. Extra probes should be compared with spending the same measured budget on ordinary proposals.

Choose the base sampler and one search operator on development data before fitting the final gate. If the base changes to three-step sampling, rebuild the relevant continuation labels for that reference policy; stock10-continuation labels do not silently become three-step-continuation values.

Do not start progressive pruning here. Completed candidates make the first causal comparison interpretable. If denoising later dominates cost and completed search clearly works, compare against FASTER before inventing another denoising-space selector.

**Experiment 4 is the central test: do outcome labels teach better allocation?**

Freeze candidate count, sampler, selector checkpoint, tie rule, and continuation policy. Partition source identities into selector training, gate fitting, gate threshold selection, and locked audit sets. Identity-level nested cross-fitting is an alternative when data are scarce, but the final frozen selector still needs an independent audit. A selector cannot use a gate-training state's outcome labels to choose the candidate whose effect becomes that state's target.

For each gate-training state, record the default outcome and the frozen selector's outcome under paired continuations. The label is D in {-1,0,1}, or its repeated-seed mean. When the chosen action equals the default, its effect is exactly zero; reuse that branch rather than fabricating independent noise. Once the selector is frozen, only the selected/default continuations are required for most gate-training states. Retain a smaller all-candidate panel for diagnosing headroom and regret.

Use an inexpensive regularized predictor of E[D|X]. Start with a low-dimensional state/action summary model; allow one small nonlinear model if the simpler model underfits. Keep all zero-effect examples. Rescues and spoils can also be modeled separately, but their difference, not rescue probability alone, drives allocation.

Permitted first-stage features include remaining time, proprioception, instruction/context features, default-prefix summaries, and the default critic score. Log feature-extraction cost. Extra-candidate spread and P&P require paid computation. Reuse a second proposal in the four-candidate search when possible, and charge for it even when the gate exits.

| Allocation rule | Training target or signal | Main question |
|---|---|---|
| Never / always search | Fixed endpoints | What does selection buy at full frequency? |
| Random gate | State-independent search at matched expected frequency | Is there useful allocation information? |
| Default-value gate | Low predicted default success | Does predicted difficulty suffice? |
| Task/phase rule | Instruction/task and normalized remaining time | Is apparent sophistication just scheduling? |
| Disagreement gate | Executed-prefix proposal disagreement or U10 | Does uncertainty earn its measurement cost? |
| Critic-surrogate gate | Predicted selected-minus-default value | Is the selector's belief enough? |
| Outcome gate | Paired realized selected-minus-default success | Does learning actual benefit change decisions usefully? |

For the key target comparison, give the surrogate and outcome gates the same input X, architecture, state coverage, and threshold-selection protocol. Train the surrogate gate to predict the post-search critic margin from X; do not supply that future margin as a free pre-search feature. This isolates the supervision target.

Then run an **equal-interaction-budget comparison**. Allow the surrogate system to spend the same additional simulator-label budget on improving its critic, using a separate training partition, before freezing and evaluating the resulting system. This answers the stronger question: are outcome labels especially valuable for allocation, or did the initial comparison simply give one system more useful feedback? Report learning curves against independent labeled identities and simulator transitions, not just training epochs.

Use a small predeclared set of development cost targets rather than a large threshold sweep. Match measured full-policy cost, since search frequency does not capture paid probes, batching, and changed episode length. Thresholds are frozen before audit. Report achieved costs and uncertainty; do not manufacture exact cost matches by retuning on test outcomes.

A promising gate should improve over well-tuned cheap rules on locked audit identities. To claim superiority over published adaptive-compute methods, add a faithful or explicitly resource-matched ELASTIC comparison. That system-level comparison is distinct from the within-operator target ablation; matching one component alone is not a reproduction. Training interactions and feature-processing cost belong in the comparison.

**Experiment 5 separates within-task generalization from transfer under shift.**

Reserve final shifts before developing the new method. Inventory every cohort used for training, plotting, threshold choice, and previous discussion. The familiar 220 identities are development data. Position-perturbation training exclusion does not restore their confirmatory status after repeated analysis.

Use three declared evaluation strata:

| Stratum | Permitted interpretation |
|---|---|
| Fresh initializations/proposal seeds on known tasks and known perturbation families | Within-task generalization. |
| A wholly withheld shift family or parameter region with no tuning feedback | Transfer to that specified unseen shift. |
| A controlled recovery perturbation, if feasible and specified before testing | Recovery after that intervention; not general real-world robustness. |

For example, task-aware selection might transfer across moderate geometry changes but fail under instruction/object substitutions that corrupt both proposals and visual value estimates. These are hypotheses. Use rendering changes to probe perception and physical changes to probe dynamics only when the benchmark intervention actually isolates those factors. Verify feasibility and task success definitions.

At each shift, use a separate diagnostic branch panel to ask whether opportunity declined, selector regret increased, or gate ranking failed. The main test still uses fresh paired full episodes with deployment-valid observations. A gate might correctly stop wasting compute when no candidate helps; it cannot restore policy support that is absent.

All policies receive the same initial-state distribution, environment-step limit, and declared seed scheme. Report macro-averaged task performance, pooled results, per-task/shift effects, and their uncertainty. Cluster repeated states/seeds by source identity. Different perturbation versions of the same underlying task also warrant grouped analyses for broad generalization claims.

Make zero-shot gate transfer the main shift test. A recalibrated gate may be an informative secondary result, with its target-shift labels and adaptation cost reported separately. Do not select the shifted-data threshold and call the result zero-shot transfer.

A second policy family or physical-platform replication would strengthen a broad VLA claim after the main mechanism works. A second pi0.5 checkpoint is useful sensitivity evidence, but a weaker generality test. Do not replace the pinned policy midstream merely because a newer model exists.

**The decision tree must distinguish an unpromising mechanism from an underpowered experiment.**

| Finding | Decision |
|---|---|
| Replay, proposal validity, or feature reconstruction fails | Repair that contract before interpreting critic performance. |
| Hindsight headroom shrinks under independent continuation seeds | Quantify the shrinkage; reduce reliance on historical oracle uplift. Expand only if a useful residual effect remains plausible. |
| Adequate repeated-seed evidence rules out useful opportunity at the tested candidate budget | Stop that search branch. Continue the cheap-sampling track if warranted. |
| Repeatable opportunity exists, but selectors fail | Inspect representation, within-state supervision, and label coverage. Allow the bounded repair; do not add search complexity. |
| Always-search is neutral, but an independently audited rule selects beneficial states | Continue selective allocation. |
| Search works, but all reasonable gates match random allocation | Report a selector result. Drop the learned-allocation claim. |
| Cheap default-value or task/phase gating matches the outcome gate | Use the simpler rule; the proposed supervision advantage is unsupported. |
| Outcome gate beats cheap rules, but loses its edge with equal label budgets | Reframe as a data-allocation finding; do not claim a superior target independent of data cost. |
| Branch gain vanishes in closed loop | Diagnose changed occupancy and compounding harm; allow one targeted collection repair. |
| Gain survives in-domain but disappears under the withheld shift | Bound the generalization claim and explain which component failed. |
| Bare three-step sampling is noninferior and materially cheaper | Pursue an explicitly scoped efficiency result. |
| Confidence intervals still span worthwhile benefit and no benefit at the resource cap | Record inconclusive evidence. Do not call it a negative theorem about VLAs. |

Use confidence bounds at fixed planned audit points. Repeated informal peeking cannot supply valid stopping evidence. For a practically useful-effect threshold delta, an upper confidence bound below delta supports stopping that scoped claim; an interval crossing zero alone does not.

**Set meaningful effect and resource targets before confirmation.**

My planning defaults are a three-percentage-point success improvement worth detecting, and, for the separate efficiency track, a two-point noninferiority margin with at least 25% less measured inference cost. These are research choices, not observed effects or universal deployment tolerances. Freeze them before new confirmatory outcomes. A superiority interval excluding zero establishes a positive effect; claiming at least three points would require the lower bound to exceed three points.

After Experiment 0, choose one primary compute budget from measured feasible settings. A useful initial target is at most 1.25 times stock10's mean total episode inference cost, with p95 decision latency reported as a secondary constraint. The central confirmatory contrast is the outcome gate against the strongest eligible cheap allocation baseline selected in development at the same budget. Retain stock10 as an anchor so that beating a poor gate cannot masquerade as useful policy improvement.

For paired binary outcomes, with discordance d and difference delta, Var(D)=d-delta^2. The familiar independent-pair approximation gives

n approximately (1.96+0.84)^2 d / delta^2.

At d=0.10 and delta=0.03, this is about 870 paired episodes for 80% power at two-sided 5%. The actual design must account for task/source clustering, pilot uncertainty in discordance, and the chosen comparison. This is an illustration, not an order to run 870 episodes. Noninferiority has its own one-sided sizing calculation and should not inherit the superiority sample size.

Count physical/simulator work honestly. The 768-continuation pilot excludes source rollout collection, replay, and candidate generation. Gate training can usually use two branches per selected/default comparison. Confirmation with K policy arms on n identities needs K*n full episodes per seed replication. Measure seconds per continuation, episode length, feature-extraction time, and peak memory before converting those counts into GPU-hours.

Target a feasibility review after roughly two weeks of active implementation/experimentation. Continue to selector/gate studies only if repeatability and learning curves justify their projected cost. Reserve a substantial portion of the remaining rollout budget for independent audit and confirmation before collecting a large gate dataset. A method that consumes every available initialization during development leaves no credible final result.

A concrete next-work queue is:

1. Version the proposal-valid Q10 inputs, finite-budget labels, remaining-time feature, and split manifest.
2. Produce a small online-versus-reconstructed feature check and a bare 3/10-step timing table.
3. Generate the 48-identity pilot manifest and preselect its 12 repeatability identities.
4. Train the compact representation comparison and repaired TD baseline using eligible existing data.
5. Finish the repeated-seed panel; publish candidate and selector effects with uncertainty.
6. Run the early closed-loop development check and the four-arm solver/candidate comparison.
7. Freeze one operator, then collect only the additional paired labels required for gate fitting and its independent audit.
8. Select one primary contrast and budget, freeze all configurations, and run fresh confirmation plus the declared shift test.

**My research bets, in descending order, are deliberately uneven.**

I have the most confidence that honest bare-sampler profiling will expose useful efficiency opportunities and that fixing the critic's information contract will make the learning experiment more interpretable. I have moderate confidence in modest action-selection improvements on task-relevant ambiguous decisions. I have lower confidence that a learned gate will substantially outperform a tuned default-value or task/phase rule once all costs and label budgets are matched.

I expect semantic shifts that make every candidate wrong to resist this approach. I also expect visually aliased contact situations to limit a critic based on aggressively pooled representations. Those failures would point toward improved observation/history or a better policy prior, rather than more sampling from the same law.

I would defer P&P refinement variants, Q50 as a deployment scorer, weighted action averaging, gradient guidance, noise-space MCMC, adaptive execution horizons, and full online RL. Each can be useful in the right regime; the current evidence does not identify that regime. If a validated selector eventually leaves a clear denoising-cost bottleneck, examine FASTER. If valuable local action changes appear learnable, compare QGF/QPILOTS and validate gradient directions against simulator outcomes before building new guidance machinery.

The paper I would aim to write has three decisive results: candidate headroom that remains meaningful under independent continuations; a measured gap between critic-predicted and actual search benefit; and a fresh success–compute frontier showing whether direct outcome supervision improves allocation under the declared shifts. An equal-label-budget curve is essential to interpreting the third result.

A strong diagnostic paper could instead show a reproducible collapse of apparent oracle benefit, or a transfer failure localized to proposal support, selector regret, or allocation. It would need adequate precision and evidence across several settings. One failed model or a nonsignificant pilot is insufficient.

The longer-term vision is a robot that allocates computation according to the improvement it can actually realize, including the limitations of its own evaluator. The first funded deliverable is the proposal-valid, repeated-continuation selector experiment.

