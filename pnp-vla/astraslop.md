# PnP-VLA research masterplan: reliable policy improvement from a frozen VLA

**PI decision · 10 September 2026**  
Research and results cutoff: 8 September 2026. Repository baseline: commit 6169b50.  
This is the authoritative execution plan. It supersedes the earlier research recommendations; those documents remain the record of supporting analysis.

**The objective is to improve a frozen VLA’s closed-loop success under a practical interaction and inference budget.**

Train a small evaluator to distinguish the consequences of plausible actions, use it to select among completed policy proposals, and establish whether that intervention improves full episodes. The core scientific hypothesis is that a strong pretrained policy already proposes useful alternatives at some decisions, and that a compact evaluator can recognize them from deployment-valid observations.

The project’s assets are the frozen policy, existing rollout data, and infrastructure for replaying decisions. The research opportunity is to turn those assets into reliable local policy improvement. Adaptive compute is a conditional extension: it becomes worthwhile only when useful action selection exists and its benefit can be predicted cheaply.

The initial system is deliberately specific: pinned π0.5; fifty generated actions; ten executed actions before replanning; four completed candidates including the ordinary default; a small contextual critic; and terminal task success as the learning objective.

The priorities are:

1. Correct the intervention, input, and representation contracts.
2. Collect a bounded fork dataset and train the compact critic.
3. Evaluate selected actions on independent continuations and full episodes.
4. Measure the ordinary three-step and ten-step sampling frontier.
5. Add selective computation only when the first four results justify it.

The month’s deliverable is a defensible decision about this mechanism, supported by reproducible experiments. A new loss, P&P operator, or learned gate is not required for that deliverable.

**Research judgment and scope.**

My highest-confidence investments are correcting the known critic bottlenecks and measuring the bare sampler honestly. I have moderate confidence in modest improvements from contextual action selection. I have lower confidence that paired-loss training adds materially beyond a properly trained pointwise critic, and lower confidence still that a learned pre-search gate beats simple allocation rules after costs and data budgets are matched.

Moderate geometry changes are the more promising transfer regime: the policy may propose both successful and unsuccessful variants. Semantic shifts are harder because the policy and evaluator may share the same instruction-grounding error. Sampling more from that shared error need not help.

Keep the VLA frozen for the main experiment. Train only the small evaluator and, conditionally, an allocator. Defer early pruning, gradient guidance, noise-space MCMC, action averaging, adaptive execution horizons, and full online RL. Reconsider one of these only when a measured bottleneck makes it the next necessary experiment.

**What the evidence establishes.**

The handoff records 1,880 rollouts, including 1,054 successes and 826 failures, with approximately 1,532 underlying episode/initialization groups. Overlapping action windows do not create independent episodes.

| Existing observation | Interpretation for this plan |
|---|---|
| Conditioned Stage 2 ranking accuracy 0.5959; separately trained action-only accuracy 0.5965 | Conditioning did not improve the tested system. This is a negative result for that implementation. |
| Historical observed-outcome oracle uplift of 6.34 percentage points in an uncertainty-enriched pool | There may be useful alternatives, but the maximum over noisy outcomes overstates repeatable opportunity. |
| Nearly half of discordant pairs lie within the same recorded action mode | Small contact-relevant differences can matter; a story based only on distinct behavioral modes is inadequate. |
| Stock10, bare three-step, and three-step refinement succeed on 117, 118, and 120 of 220 identities | Low-step sampling deserves evaluation. These results establish neither improvement nor equivalence. |
| Refinement-minus-stock interval spans −2.27 to +5.45 points | The familiar cohort is too imprecise and too repeatedly analyzed to establish the desired effect. |
| No completed Q10/Q50 training result documented at handoff | Existing training code or notebooks are not evidence of a successful critic. |

These are historical observations. Preparation of this masterplan launched no training or simulator experiments. Earlier analysis included source inspection, an untrained CPU architecture probe, and arithmetic checks.

Three concrete code findings determine the first implementation work:

| Finding | Why it matters | Required correction |
|---|---|---|
| Ranking freezes an observation encoder previously trained for state value, compressing 2048 dimensions to 128. [Model](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/model.py:218), [training](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/train.py:376). | General difficulty prediction can discard information needed to choose between actions. | Compare frozen versus jointly trained small observation encoders; keep the VLA frozen. |
| The cross-attention variant has no action-time encoding and bypasses the temporal convolution. [Model](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/model.py:198). | Its score is invariant to permutations within the valid action prefix. | Preserve action order through temporal processing or explicit positions. FiLM already processes order. |
| The replay collector advances source trajectories using full generated chunks; its optional execution setting affects continuation, and the V2 recipe omits that setting. [Collector](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/collection.py:363), [recipe](/Users/arjunsharma/development/cs159-sp26/pnp-vla/scripts/build_verifier_v2_notebooks.py:113). | That code path does not implement the intended ten-action source and continuation policy. | Use recorded ten-action source trajectories and explicit ten-action continuations. Audit historical provenance before assigning rows these semantics. |

The action-order finding was confirmed structurally: reversing ten actions changed untrained cross-attention scores by at most 2.98e-8, while FiLM scores changed. This establishes an architectural property, not a measured performance improvement.

**The intervention and data contract.**

| Component | Fixed initial definition |
|---|---|
| Policy | `lerobot/pi05_libero_finetuned` |
| Checkpoint revision | `8e174154ef5f6c60a8da12ae99c303d8963138c1` |
| Reference policy π_b | Ten denoising steps, execute ten actions, then replan |
| Proposal pool | Four independent completed samples under the same sampler; one is the ordinary default |
| Scored action | The proposed ten-action prefix, available before execution |
| Outcome | Terminal task success within the remaining environment-step budget |
| Continuation | π_b after the candidate prefix, unless a separately versioned experiment explicitly changes it |
| Evaluator inputs | Permitted observations/history, instruction, proprioception, remaining steps, and candidate prefix |
| Excluded inputs | Future success, future termination masks/padding, and privileged simulator object state |
| Grouping unit | Underlying task/initialization; repeated seeds and boundaries stay grouped |

Stop execution at termination or the episode budget. Record missing or corrupt runs separately from genuine failures.

The current saved mean prefix comes from before language-model prefill. It is a learned representation, but it is not the final contextualized token array. Reconstruct contextual features from original observations or exact replay; they cannot be recovered by inverting the stored mean.

Recover original pre-decision proposals where terminal-dependent masks or zeros have entered critic inputs. Changing the mask alone does not restore overwritten actions. Keep remaining time explicit. Fifty actually executed actions spanning five replans are not interchangeable with one proposed fifty-action plan.

Every record must identify the source trajectory, task/init group, checkpoint, source/intervention/continuation horizons, solver, elapsed and remaining steps, candidate seed, continuation seed, normalization version, and collection split. Keep candidate-generation randomness separate from future-continuation randomness.

**The theoretical model determines what each experiment can claim.**

Let x contain the full simulator state, policy-relevant history, and remaining time; h denotes the information available to the deployed evaluator. Define

q_b(x,u) = P(success within the remaining budget | execute prefix u, then follow π_b).

For candidate pool C containing default u_0, and a deployable selector choosing û:

O(x,C) = max_j q_b(x,u_j) − q_b(x,u_0)

R(x,C) = max_j q_b(x,u_j) − q_b(x,û)

Δ(x,C) = q_b(x,û) − q_b(x,u_0) = O(x,C) − R(x,C).

O is opportunity in the proposal pool; R is selector regret; Δ is actual intervention benefit. More candidates can increase opportunity while exposing more critic errors. With uniform value error at most ε on a fixed pool, argmax regret is at most 2ε. This motivates improving discrimination on the proposals actually considered.

A simulator-state oracle can use distinctions unavailable in h. Its gap from a learned selector therefore combines information limitations and estimation error.

State-value sufficiency does not imply action-ranking sufficiency. Two states can have action values (0.9,0.1) and (0.1,0.9), while both have value 0.5 under uniform action sampling. A representation can predict their state values perfectly and discard the distinction needed to select well.

For repeated control, the finite-horizon performance-difference identity is

J(π_sel) − J(π_b) = E_π_sel[Σ_t A_b(x_t,u_t)],

where A_b=q_b−V_b and t indexes executed decision chunks. Positive average advantage on base-policy states is insufficient; the new policy must choose well at the states it actually visits.

Continuation quality also matters. A prefix that prepares a different grasp may have low q_b if the base continuation abandons it. A negative result for one changed prefix followed by π_b rules out that intervention, not all coordinated policy improvement.

**E0 — establish correctness and cost before larger collection.**

Use twenty development states to verify replayed observations, simulator state, wrapper/controller state, remaining steps, and executed action sequences. Check source advancement and continuation separately. The collector may replay actions and then correct physical state; equality of one flattened simulator vector is insufficient.

Verify that diagnostic capture leaves the ordinary policy output unchanged. Compare online and reconstructed features. Audit proposal validity, normalization, terminal handling, and metadata against the contract above.

Run the cheap legacy FiLM ablation first: observation encoder frozen versus jointly trained, with data, loss, optimizer budget, and action encoder fixed. Historical data with full-chunk continuation can support this within-semantics ablation, but must not be silently pooled with stock10-continuation labels.

Profile 3/10 denoising steps × 1/4 completed candidates on common observations. Fix hardware, precision, warm-up, synchronization, and batching. Separate context encoding, prefill, denoising, critic cost, transfers, and gate/probe cost. Exclude diagnostic-only duplicate passes and storage/network logging from deployment latency.

Start reconstruction on a small matched sample. Measure its cost before processing the roughly 150-GiB embedding corpus.

**E0 exit:** a versioned intervention contract, replay/feature audit, eligible-data manifest, frozen/joint encoder result, and timing table. Resolve contract failures before interpreting learning performance.

**E1 — collect a bounded diagnostic dataset.**

Begin with 64 distinct underlying task/initialization identities, targeting eight task/suite conditions with eight initializations each across two already-development regimes. Confirm availability before freezing the manifest. Select one live decision boundary per identity with a declared task/phase rule. Avoid selection by candidate outcomes or exclusively high uncertainty.

This is a balanced diagnostic sample. It is not an unbiased sample of every deployment decision.

| Allocation | Branch continuations |
|---|---:|
| 64 identities × 4 fixed candidates × 2 independent continuation replications | 512 |
| Conditional reserve for additional identities, repeated continuations, or refreshed pools | Up to 480 |
| Initial diagnostic cap | 992 |

Source rollouts, replay, candidate generation, and failed collection attempts are additional costs. Record simulator transitions and wall time.

Preselect sixteen identities for deeper diagnostics before seeing candidate outcomes. Repeat them if continuation noise blocks a decision. When refreshing the other candidates, retain the same default action and reuse its outcomes under the same continuation seeds. This separates future randomness from randomness in the sampled alternatives.

Allocate the reserve using measured variance, state coverage, and collection cost. More distinct states may help learning more than many repetitions at a few states. Record the adaptive development rule; confirmation remains independent.

**The primary measure is held-out selected-minus-default success.** Fit pilot models with identity-level out-of-fold predictions. The small panel can expose large problems and guide collection; it cannot establish a three-point full-policy improvement.

Observed hindsight headroom remains secondary. For descriptive candidate-selection curves, choose candidates on one set of continuation seeds and evaluate on another. Call this the performance of that finite-data selection procedure, not a noise-free oracle.

Repeated-seed effects can be summarized by d_jk^(r)=Y_j^(r)−Y_k^(r). Independent replications satisfy

E[d_jk^(1)d_jk^(2) | x,C] = (q_b(x,u_j)−q_b(x,u_k))².

Average within states, then across identity groups; retain negative finite-sample estimates. This statistic diagnoses persistent action differences but can be weak. For a toy 0.6-versus-0.5 pair with independent future outcomes, its mean is 0.01 and its standard error across 64 pairs is about 0.0625. A null estimate must not veto the project.

Common random numbers may improve precision, but rescues, spoils, and discordance depend on the coupling. Marginal success differences do not. State the coupling and prioritize the marginal difference.

**E1 exit:** audited branches, direct intervention estimates, out-of-fold learning diagnostics, and a justified reserve allocation. Inconclusive evidence can justify more data when meaningful benefit remains plausible and the proposed collection addresses a specific limitation.


**E2 — train the smallest credible contextual action critic.**

Start with a scalar success predictor f_φ(h,u) in [0,1], trained by pointwise binary cross-entropy on eligible Monte Carlo outcomes. Use a compact temporal action encoder and a small observation-token readout, initially width 128–256 and one or two readout blocks. Preserve action time and camera/spatial identity where available. Train the small readout jointly with the scoring head; freeze VLA parameters.

Include instruction, proprioception, and remaining time. Use existing rollouts only when their proposals, continuation policy, and sampling provenance support the specified target. Outcome-enriched sampling requires appropriate correction before claiming deployment probability calibration.

The comparison order is:

| Comparison | Question |
|---|---|
| Separately trained action-only versus repaired pooled-context critic | Does the available context improve action selection? |
| Pre-prefill token arrays versus final contextualized arrays, with the same small head and loss | Does contextualization improve the useful representation? |
| Pointwise versus pointwise-plus-paired loss on the same labels | Does emphasizing within-state errors help? Secondary ablation. |
| Monte Carlo versus reference-policy TD with matched features and objective | Does bootstrapping improve learning under the available data? |

A zero-context ablation is a diagnostic, not a separately trained action-only baseline. Shuffled context can reveal reliance on context but may create out-of-distribution inputs. Evaluate within task/phase as well as across the full panel.

Run pointwise models first. If paired training is warranted, use

L_pair = average_j<k,r [(f_φ(h,u_j)−f_φ(h,u_k))−(Y_j^(r)−Y_k^(r))]²,

with state-normalized loss terms and one declared coefficient, initially 1. Keep concordant outcomes. For residuals e_j=f_φ(h,u_j)−Y_j,

Σ_(j<k)(e_j−e_k)² = N Σ_j(e_j−mean(e))².

The loss reweights centered errors; it adds no information beyond the same labels. Its benefit is an empirical representation/optimization question. If both feature and loss treatments run, use the corresponding 2×2 on a common reconstructible data subset.

For a fair MC–TD comparison, use the same finite-budget success objective: no time discount, correct termination, explicit remaining horizon, and a reference-policy continuation target. The historical γ=0.99 objective is a separate system comparator; it discounts success 100 environment steps away to approximately 0.366.

Use three training seeds per fixed recipe. Fit normalization on training data and split by source identity before extracting windows. Group related perturbations for transfer analyses. Keep model selection separate from audit outcomes. Rank models primarily by held-out intervention utility, then examine ranking accuracy, calibration, and training-seed variation.

A null result on 64 states is not decisive about representation learnability. If plausible opportunity and learning/coverage diagnostics justify expansion, the default next tranche is **256 new identities × 4 candidates × 2 seeds = 2,048 continuations**: 192 identities for fitting and 64 for development validation. These are conditional allocations, not automatic commitments.

Freeze one selector checkpoint, proposal sampler, tie rule, and candidate count before the independent audit. Initial audit allocation: **128 new identities × selected/default × 2 seeds = 512 continuations**. Choose the action without audit outcomes. Reuse the default branch when the selected prefix is identical; its effect is exactly zero.

**E2 exit:** an independently evaluated selector and a scoped conclusion about context, data, and loss. If useful opportunity exists but the bounded critic family cannot exploit it, diagnose coverage or observability only when that diagnosis changes the decision. Do not expand architectures indiscriminately.

**E3 — test a single intervention and repeated control.**

Once one selector is credible, use 48 fresh development episode identities and four policies, totaling 192 episodes:

| Policy | Purpose |
|---|---|
| Stock10, execute10 | Reference |
| Bare three-step, execute10 | Independent efficiency hypothesis |
| Stock10 with selection at one predeclared decision opportunity | Single-intervention effect followed by π_b |
| Stock10 with selection at every decision | Repeated control and changed state occupancy |

Pair initializations and use a declared noise-coupling scheme. Fix the single-intervention opportunity without future outcomes. Episodes that finish before that opportunity remain in the analysis with no intervention.

If single interventions help and repeated selection fails, use one bounded diagnostic/repair cycle. A prespecified mixture invoking the existing selector at 25% of decisions can test intervention frequency without training a gate. It does not guarantee safety or improvement.

If continuation dependence is the unresolved issue, cross default/chosen prefixes with base/frozen-selected continuations. The interaction

[q_sel(u_1)−q_sel(u_0)] − [q_b(u_1)−q_b(u_0)]

measures whether the prefix’s relative value changes with the future policy. Freeze both continuations before labeling, and reuse existing base-continuation branches.

Choose the repair from the evidence: add labels at newly visited states while retaining q_b, or explicitly refresh policy evaluation for a new continuation. Keep new target semantics separate. Freeze again and evaluate on fresh identities. Permit one targeted cycle within a declared budget.

A neutral always-search average does not rule out selectivity. A simple subset rule chosen in development and validated independently may still identify useful interventions.

**E3 exit:** evidence that the selector improves a full policy, or a localized failure with an explicit stop/repair decision.

**E4 — determine which completed sampler earns its cost.**

Evaluate the 3/10 denoising-step × 1/4-candidate matrix. Couple the solver comparison through the same initial noise, repeat over independent noise draws, and use the same reference continuation for branch outcomes. Inspect the ten executed actions. Diversity in discarded tails is useful only if it predicts or changes those actions.

Use a common critic to expose proposal-distribution effects, while checking its accuracy separately on each sampler. If sampler-specific critic adaptation is needed, report it as a matched-data treatment. Equal architecture does not imply equal critic accuracy.

Measure mean, median, and p95 decision latency; total episode inference time; environment steps; and peak memory. Shared context and batching matter. Amdahl’s law gives a useful check: if context accounts for fraction p of ten-step latency, a linear denoising-cost approximation gives

C_3/C_10 ≈ p + 0.3(1−p).

At p=0.8, the saving is only 14%. Count actual deployment work rather than velocity evaluations alone.

Report both episode cost and cost on a common observation panel. Faster failures can reduce episode compute. If simulation pauses during inference, latency improvements do not establish faster physical reaction; that claim requires a clocked evaluation or hardware.

Promote bare three-step sampling only after the separate efficiency evaluation supports it. If π_b changes, future labels must name the new continuation; old labels retain their original meaning.

The diffusion rationale is precise. For a linear flow-matching path a_t=(1−t)a_0+tε and exact conditional velocity,

a_t−t v*(a_t,t,h)=E[a_0|a_t,h].

This is a conditional mean, not a sampled completed action. Generally Q(E[a]) differs from E[Q(a)], particularly around nonlinear contact transitions. More accurate integration need not improve task return. [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264).

The implemented [P&P probe](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/pnp.py:139) produces dependent clean-estimate/re-noising iterations. Its spread has no automatic posterior or epistemic interpretation. Test it only if it predicts independent intervention benefit beyond cheaper features after its cost is charged. Completed-candidate ranking success does not establish useful critic gradients on intermediate or off-distribution actions.

**E4 exit:** a measured success–compute frontier and one fixed base/search operator for any later allocator.

**E5 — conditional allocation study.**

Open this stage only when selection is useful globally or on an independently validated subset, the potential compute saving is meaningful, and cheap features plausibly predict benefit.

Let I_0 contain information available after generating the default action but before buying extra proposals. The gate needs

m(I_0)=E[Δ(x,C)|I_0],

averaging over the unseen candidate pool and hidden state. Fixed-pool repeatability does not establish predictability before that pool is drawn. A post-search guard can diagnose harmful selection but has already paid for search.

At fixed search fraction κ and constant incremental cost, the gain over random allocation is Cov(g(I_0),m(I_0)). Large variation in realized outcomes is insufficient; the variation must be predictable from available information.

With gate overhead C_g and measured incremental search cost ΔC, saving compute relative to always-search requires

C_g < E[(1−g)ΔC].

This is a cost condition, not a utility guarantee. Charge paid uncertainty probes even when the gate declines search.

Freeze the selector first. Collect paired selected/default outcomes and fit a small regularized predictor of D=Y_selected−Y_default, including zeros and harms. Separate fitting, threshold selection, and audit identities. A selector must not use a state’s target outcomes to choose the action whose effect becomes its gate label.

Compare never/always, random, default-value, task/phase, disagreement, critic-surrogate, and outcome-supervised allocation. Match achieved cost. For the target ablation, give surrogate and outcome gates the same pre-search features and architecture; predict the later critic margin rather than supplying it as a free input.

Then compare at equal training-interaction budget: allow the alternative to spend the extra labels on its critic. If a simple rule or better-trained critic matches the gate, use it. Retest the complete gated policy in closed loop.

**E5 exit:** a demonstrated allocation advantage at matched cost and interactions, or a decision to retain the simpler selector/allocator.


**Generalization and confirmation.**

Inventory every cohort previously used for training, plotting, threshold choice, or research discussion. The familiar 220 identities are development data. A perturbation excluded from training is not a pristine test after repeated analysis.

Before new development, reserve one unused shift family or parameter region and the independent episode identities needed for confirmation. If sufficient new identities are unavailable, narrow the claim. Extra policy seeds are replications, not new initializations; some suites provide only ten initial states.

Use three declared scopes:

| Evaluation scope | Permitted conclusion |
|---|---|
| Fresh held-out identities within known tasks/regimes | Within-task generalization |
| Entirely withheld shift family or parameter region | Transfer to that specified shift |
| Second policy family or physical platform, after the main mechanism works | Broader VLA or deployment generality |

Use a separate diagnostic branch panel to localize shifted failures: missing good proposals, critic/observation failure, or allocation failure. Keep it distinct from the untouched episode panel that supplies the final performance claim.

Compare geometry and semantic shifts when feasible. A semantic intervention must change and verify the success predicate along with the instruction. Do not manufacture labels by editing text alone. Simulator object state may support an explicitly privileged diagnostic, but never enters the deployed critic.

Relative ranking could survive a shift that applies a common positive scale and offset to all candidate values, even as absolute calibration changes. An instruction change can reverse the ordering instead. This motivates measuring ranking and calibration separately; it does not assert that real geometry shifts obey that affine model.

Freeze the final checkpoint, sampler, execution horizon, tie rule, gate threshold if any, primary contrast, cost budget, and cohort before confirmation. Report training-seed sensitivity from development and identify the model-selection rule. Do not select a favorable test seed or retune on test outcomes.

The default primary claim is improved episode success versus π_b at a fixed practical compute budget. The strongest matched selector control supplies the mechanism comparison. If the final contribution is adaptive allocation, its primary matched-cost comparison is the strongest eligible cheap allocator chosen in development; retain π_b as the anchor.

Planning defaults are:

- A three-percentage-point episode-success improvement worth detecting.
- A first practical compute target to investigate of at most 1.25× stock10 mean episode inference cost, revised only from development feasibility evidence before confirmation.
- For the independent efficiency result, a two-point noninferiority margin and at least 25% less measured inference cost.

These are proposed research thresholds. A positive superiority interval establishes improvement; claiming at least three points requires its lower bound to exceed three points. Noninferiority needs its own one-sided design.

For paired binary outcomes with discordance d and target difference δ, a simple independent-pair sizing approximation is

n ≈ (1.96+0.84)² d / δ².

At d=0.10 and δ=0.03, n≈871 paired identities for 80% power at two-sided 5%. Account for task/init clustering, pilot uncertainty, and the actual comparison before committing a run count. The 48-identity development batch is not confirmation.

Report task-macro and pooled success, paired differences, task/shift breakdowns, interval estimates, and the complete cost measurements. Cluster repeated seeds/boundaries by initialization and show task-grouped uncertainty where relevant. Predeclare primary comparisons and use a suitable multiplicity procedure for multiple confirmatory claims. This follows the broader need for uncertainty-aware RL evaluation. [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264).

**The literature sets the novelty bar.**

The literature below was reviewed through 8 September. June and earlier work supplies the mid-2026 context; August entries are later updates. Their reported results have not been replicated here.

| Closest literature | What it requires of this project |
|---|---|
| [π0.5](https://arxiv.org/abs/2504.16054v1), [RECAP / π*0.6](https://arxiv.org/abs/2511.14759v2), [π0.7](https://arxiv.org/abs/2604.15483v2) | Situate the study among heterogeneous pretraining, experience-driven improvement, and richer context conditioning. One pinned LIBERO-finetuned model does not represent the whole generalist frontier. |
| [V-GPS](https://arxiv.org/abs/2410.13816), [RoboMonkey](https://arxiv.org/abs/2506.17811) | Treat frozen-policy candidate generation and value-based verification as established baselines. |
| [DA-SIP](https://proceedings.neurips.cc/paper_files/paper/2025/hash/49eadcc4a329fc6b74b9f8a82b78cbc3-Abstract-Conference.html), [VLA-ATTC](https://arxiv.org/html/2605.01194v2), [ELASTIC](https://arxiv.org/html/2606.31132v1) | Adaptive denoising, disagreement-triggered search, and joint sequential/parallel allocation are already occupied. A learned gate alone is insufficient novelty. |
| [FASTER: Value-Guided Sampling for Fast RL](https://arxiv.org/html/2604.19730v1) | Any later early-candidate or noise-space selection method needs this close comparator. |
| [RL Token](https://arxiv.org/html/2604.23073v2), [Decoupled Q-Chunking](https://arxiv.org/abs/2512.10926v2) | Compact contextual representations and separate backup/execution horizons have precedents; validate the local implementation’s semantics. |
| [QGF](https://arxiv.org/abs/2606.11087v1), [QPILOTS](https://arxiv.org/abs/2606.14801v1) | Test-time critic guidance of frozen flow policies already exists. A future guidance project needs direct comparisons and independent gradient validation. |
| [Q-VGM, August revision](https://arxiv.org/abs/2606.08015v3), [Q-Planning](https://arxiv.org/html/2608.21204v1), [Decoupling Policy Extraction](https://arxiv.org/abs/2608.20909) | Distinguish policy training from frozen-policy planning; separately learned proposals and critic reranking are themselves a contemporary direction. |
| [LIBERO-CF](https://arxiv.org/abs/2602.17659), [CounterAlign](https://arxiv.org/abs/2608.21740) | Test semantic grounding and shared shortcuts. Instruction-relabeling supervision differs from simulator action interventions, but “counterfactual supervision” is not a first claim. |
| [Flow Q-Learning](https://arxiv.org/abs/2502.02538v2) | If an improvement operator eventually works, amortizing it is a legitimate alternative to increasingly complex inference. It is not an additional workstream now. |

ELASTIC’s allocation labels use verifier-derived quality, while its critics receive outcome-related supervision. A direct-outcome gate would need to show the value of evaluating the deployed selector’s actual intervention, not claim that existing allocation methods ignore outcomes. There is also an [unreviewed implementation precedent for paired intervention gating](https://amohan.dev/blog/2026/repairing-frozen-visuomotor-policy-cfr-flow-matching/); it limits broad first claims without establishing this project’s hypothesis.

The prospective contribution is a controlled account of **when outcome-supervised action selection improves a frozen VLA, which information and data make that possible, and how much improvement survives under feedback, shift, and realistic compute constraints**.

At present this is a falsifiable research thesis, not an established novel algorithm. Correcting code defects is necessary engineering. A method contribution requires an advantage over repaired, properly matched baselines.

If the claim is that forked supervision is a better use of interactions, compare it with spending a matched simulator-transition budget on ordinary policy rollouts. If the claim is allocation superiority, include a faithful or explicitly scoped ELASTIC/ATTC comparison. A component ablation is not a reproduction of an entire published system.

**Decision tree: what happens after each finding.**

| Finding | Next decision |
|---|---|
| Replay, proposal validity, or target semantics fail | Repair E0; pause larger label collection. |
| Repeatability statistic is null, but utility estimates remain imprecise | Do not stop on that statistic. Fit/audit the compact model and allocate reserve to the uncertainty that matters. |
| Adequate evidence excludes worthwhile candidate opportunity | Stop that search operator at the tested budget, state distribution, and continuation. Retain the cheap-sampler track. |
| Opportunity is plausible, but all compact critics fail after the bounded tranche | Diagnose coverage or observability only if it changes the decision; stop allocation work. |
| The simple corrected pointwise critic works | Advance it. A paired-loss gain is unnecessary. |
| Single interventions help but repeated control fails | Use one bounded mixture/continuation diagnostic, then one evidence-directed repair. |
| Always-search is neutral but an independently audited simple subset helps | Selective allocation remains viable. |
| Selection works but cheap features do not predict benefit, or gate cost removes savings | Keep selection; omit the learned gate. |
| A simple allocator or equal-budget critic training matches the outcome gate | Use the simpler or more data-efficient system. |
| Bare three-step sampling meets the efficiency criterion | Promote it as a practical baseline; require later search to improve its frontier. |
| Improvement survives in-domain but fails on the withheld shift | Bound the generalization claim and localize the failure. |
| The resource cap is reached with intervals spanning meaningful benefit and no benefit | Report an inconclusive result and its precision. |

Use scheduled audit points or a valid sequential procedure. For a claim requiring effect δ, an upper confidence bound below δ can support stopping that scoped claim. An interval merely crossing zero cannot.

**Resource ledger and execution schedule.**

| Work package | Proposed allocation | Release condition |
|---|---|---|
| E0 contracts and profiling | Twenty-state audit; small feature reconstruction and legacy ablation | Start immediately as the first implementation work |
| E1 core | 512 branch continuations | Contract verified |
| E1 reserve | Up to 480 additional continuations | Specific uncertainty or coverage need identified |
| E2 learning expansion | 2,048 continuations over 256 new identities | Pilot evidence, learning diagnostics, and measured cost justify expansion |
| E2 frozen-selector audit | Initially 512 continuations over 128 new identities | One operator frozen; identities independent of fitting/selection |
| E3 early closed-loop check | 192 episodes over 48 new identities | Credible selector ready |
| One diagnostic/repair cycle | Budget fixed from measured cost at that review | A specific failure distinguishes the appropriate repair |
| E5 gate study | Budget fixed only after E3/E4 | Useful selection, predictable benefit, and positive cost opportunity |
| Final confirmation | Powered from the selected contrast and clustering | Cohorts, configurations, thresholds, and budget locked |

If all named branch tranches are used, they total at most **3,552 continuations before any additional repair, gate training, or final confirmation**, plus the 192-episode development check. Source collection, replay, feature extraction, and candidate generation are additional. Identical selected/default branches may reduce actual runs.

Do not convert these counts into GPU-hours before measuring throughput. Do not satisfy “new identity” counts by changing only policy seeds. Reserve confirmation resources before releasing the larger development allocations.

| Timing from implementation start | Required deliverable |
|---|---|
| First several days | Contract, eligibility/split manifest, replay/feature report, frozen/joint encoder check, timing table |
| Weeks 1–2 | Core branch dataset, out-of-fold intervention diagnostics, reserve decision |
| Weeks 2–3 | Compact critic comparisons, learning curves, conditional expansion, frozen selector |
| Weeks 3–4 | Independent branch audit, early closed-loop comparison, completed-sampling frontier |
| End-of-month review | Continue/repair/stop decision and one locked confirmation plan; gate work only if earned |

This is a dependency schedule, conditional on available compute and measured throughput. The month need not include a fully powered final study.

Every completed experiment should produce an immutable configuration/cohort manifest, raw outcomes including failures and exclusions, reproducible analysis, uncertainty estimates, and a resource ledger. The principal figures are: independent intervention utility with repeatability diagnostics; representation/data learning curves; and the closed-loop success–compute frontier including the withheld shift. Add an allocation frontier only if E5 earns it.

**Source record and authority.**

This masterplan consolidates the [project handoff](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PROJECT_STATUS_HANDOFF.md), [original PI1 vision](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08_PI1_ORIGINAL.md), [PI2 audit](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI2_RESEARCH_AUDIT_2026-09-08.md), [revised PI1 vision](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08.md), [PI3 research plan](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI3_FINAL_RESEARCH_PLAN_2026-09-08.md), and [four-perspective review](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI3_FOUR_PERSPECTIVE_REVIEW_2026-09-08.md). Earlier documents supply provenance; the priorities, conditional budgets, and decision rules in this document govern the proposed work.

The research vision is a frozen generalist policy that improves through a small evaluator trained on consequences. The evaluator must recognize the action distinction that matters in the current situation. The first funded proof is that it can make better decisions under real feedback.
