# PI3 research decision: learn which action differences matter

Date: 2026-09-08 · Revised recommendation, superseding [version 1](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI3_FINAL_RESEARCH_PLAN_2026-09-08_V1.md)

**Fund a focused study of intervention-trained action values for a frozen VLA. Make adaptive compute a conditional second result.**

The research question I would own is: **can a small evaluator learn which differences between plausible VLA actions actually change eventual success, and use that knowledge to improve closed-loop control?** The project has a useful advantage: a strong proposal policy and infrastructure for replaying decisions. Use those assets to learn from controlled alternatives at the same state.

My first recommendation put too much weight on an outcome-trained compute gate. That requires three things to work: useful alternatives must exist, the selector must find them, and the benefit must be predictable before generating those alternatives. The current evidence establishes none of these convincingly. They deserve separate experiments and separate decisions. A successful action-value result should survive even if a learned gate adds nothing.

The positive scientific hypothesis is that **supervision on action contrasts, applied to a representation that preserves task and temporal information, is more useful for local policy improvement than learning a compressed representation of general task difficulty.** This is a finite-data hypothesis, not a claim that ordinary value learning is theoretically inconsistent. It leads to a specific representation/loss experiment, an intervention dataset, and a closed-loop test.

My priorities are:

1. Correct the action-value experiment and run a bounded repeatability study.
2. Test a small, jointly trained contextual action critic against strong matched controls.
3. Establish whether its single-decision improvements survive repeated use.
4. Measure the bare sampler's success–latency frontier in parallel with this work.
5. Fund a learned compute gate only if there is both predictable benefit and meaningful compute to save.

I would allocate the next month around the first three decisions. I would not make a new P&P operator, early pruning, or gradient steering the month’s deliverable. Their value depends on evidence we do not yet have.

This recommendation follows complete reads of the [original PI1 vision](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08_PI1_ORIGINAL.md), [PI2 audit](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI2_RESEARCH_AUDIT_2026-09-08.md), [revised PI1 vision](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI_RESEARCH_VISION_2026-09-08.md), and [handoff](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PROJECT_STATUS_HANDOFF.md), plus further source inspection and primary-literature research. Revised PI1 incorporates PI2; their agreement is not independent experimental replication.

The only new execution for this revision was a small, untrained CPU architecture probe and arithmetic checks. There was no new robot rollout, model training, GPU timing, database audit, or sealed-cohort evaluation. Empirical claims below are attributed accordingly.

**The deeper audit changes how I interpret the negative verifier result.**

The recorded conditioned Stage 2 verifier ranked at 0.5959, versus 0.5965 for a separately trained action-only control. That remains a negative result for the tested system. But three concrete implementation choices make it a weak test of whether observations can support action selection:

| Finding in the inspected source | Consequence | Required response |
|---|---|---|
| The ranking phase freezes the observation encoder after state-value training. The 2048-dimensional observation vector is compressed to 128 dimensions before ranking learns. See [model.py:218](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/model.py:218) and [train.py:376](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/train.py:376). | A representation adequate for predicting general success can discard information needed to distinguish actions. The later ranking module cannot recover it. | First test joint training of the small observation encoder and ranking head. Keep the VLA frozen. |
| The cross-attention variant attends to linearly embedded action tokens without temporal positions, bypassing the temporal convolution. See [model.py:198](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/model.py:198). | Within the valid prefix, its score is invariant to action order. That is an inappropriate invariance for a controller. | Add explicit action-time information or use the existing temporal encoder. Do not attribute the FiLM result to this defect; FiLM already processes order. |
| Replay collection advances the source trajectory using complete generated chunks. Its optional ten-action setting affects subsequent continuation, not source rollout advancement. The V2 worker recipe omits that setting entirely. See [collection.py:363](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/verifier/collection.py:363) and [worker generator:113](/Users/arjunsharma/development/cs159-sp26/pnp-vla/scripts/build_verifier_v2_notebooks.py:113). | This code path samples states from full-chunk execution and, by default, labels a ten-action intervention followed by full-chunk continuation. It does not implement the intended stock10/execute10 decision distribution. | Use recorded ten-action source trajectories and explicit ten-action continuations for the new experiment. Audit provenance before assigning historical rows these semantics. |

The second finding has a direct structural check. In evaluation mode, reversing the first ten actions of random inputs changed the untrained cross-attention scores by at most 2.98e-8; the FiLM scores changed by 0.349. The CPU probe used torch seed 27, one 2048-dimensional observation, four fifty-step seven-dimensional action proposals, an all-valid mask, and prefix length ten. This confirms order invariance in that implementation. It is not a trained-performance result.

The first finding has a simple counterexample. Consider two observable states with action values (0.9, 0.1) and (0.1, 0.9). Under a uniform proposal policy, both state values are 0.5. A value-only representation can collapse these states perfectly for its training objective, yet lose the information needed to select the good action. Preserving the distinction permits mean success 0.9; collapsing it leaves 0.5. This possibility warrants a cheap ablation before concluding that expensive new visual features are necessary.

Other contracts still matter. The saved mean prefix comes from before language-model prefill, rather than the final contextualized tokens; it is nevertheless a learned image/language representation, not raw pixels. Some Q-planning inputs expose future termination through masks and zero padding. Q50 windows can contain actions actually executed across five replans, which are different objects from one proposed fifty-action open-loop plan. Remaining environment steps must be an input to a finite-budget success critic. These are modeling distinctions with direct effects on the target.

My interpretation is therefore sharper than “try a better critic”: **the existing negative result combines a restricted information pathway, an incorrect temporal invariance in one variant, and potentially different control semantics. Repair these particular mechanisms, then permit the result to be negative.**

**What the existing outcomes justify.**

The handoff's immutable snapshot contains 1,880 rollouts: 1,054 successes and 826 failures, with roughly 1,532 underlying episode/initialization groups. Tens of thousands of overlapping windows do not create tens of thousands of independent learning examples.

The historical oracle uplift is 6.34 percentage points over observed outcomes in an uncertainty-enriched pool. High-uncertainty groups dominate; they include 186 discordant groups out of 1,001. Almost half of discordant candidate pairs belong to the same recorded action mode. That does not make them unimportant: small within-mode differences can decide a grasp or contact. It does mean that a story about discovering distinct behavioral modes is insufficient.

The September comparison has 117/220 stock10 successes, 118/220 for bare three-step sampling, and 120/220 for three-step refinement. The refinement-minus-stock interval spans -2.27 to +5.45 points. The 220 identities include 110 tasks with two initializations each, and have been repeatedly analyzed. These results motivate profiling and a new evaluation; they establish neither improvement nor equivalence. No completed Q10/Q50 training result was documented at handoff.

**Define the intervention before learning its value.**

Keep the pinned policy initially: lerobot/pi05_libero_finetuned, revision 8e174154ef5f6c60a8da12ae99c303d8963138c1. Generate fifty actions but execute only the first ten. Use ten denoising steps for both source rollouts and the reference continuation policy, π_b, until a separate experiment justifies a change.

Let x include the full simulator state, policy-relevant history, and remaining time. The deployed evaluator sees only h: permitted observations/history, instruction, proprioception, and remaining steps. For a fixed candidate prefix u, define

q_b(x,u) = P(terminal task success within the remaining budget | execute u, then follow π_b).

Execution stops at termination. Prefixes are proposals available before the outcome; future terminal masks cannot be features. There is no newly invented shaping reward. Existing discounted TD values remain a separately named comparison: γ=0.99 per environment step discounts success 100 steps away to approximately 0.366.

Given four completed candidates C = {u_0,…,u_3}, including the ordinary default u_0, and a selector choosing û:

O(x,C) = max_j q_b(x,u_j) − q_b(x,u_0)

R(x,C) = max_j q_b(x,u_j) − q_b(x,û)

Δ(x,C) = q_b(x,û) − q_b(x,u_0) = O(x,C) − R(x,C).

Opportunity O belongs to the proposal distribution and budget. Regret R belongs to the selector and its available information. Actual benefit Δ can be negative. A simulator-state oracle may exploit distinctions unavailable in h, so its gap from a learned critic is not entirely reducible estimation error.

If a critic has uniform absolute error at most ε on this candidate set, argmax regret is at most 2ε. That standard bound explains the practical priority: improve discrimination on the proposals actually considered. Increasing candidate count before controlling ranking error can expose more exploitable errors.

The present architecture already adds a state value and an advantage score; the state-only term cancels when ranking a fixed pool. Renaming the score “advantage” contributes nothing. The experiment below changes the representation’s supervision and the intervention data that train it.

**The first new statistic should test repeatable action effects.**

Taking the best observed binary outcome exaggerates opportunity. In a toy pool of four actions all having success probability 0.5, independent future randomness gives expected hindsight uplift 1 − 0.5^4 − 0.5 = 0.4375, despite zero true opportunity. This toy is not an estimate for the historical common-seed collector. It demonstrates why observed discordance alone is inadequate.

For a fixed state and fixed pair of candidates, collect two independent continuation-seed replications. Within each replication, share random numbers across candidates when valid. Define

d_jk^(r) = Y_j^(r) − Y_k^(r).

Independence between replications gives

E[d_jk^(1) d_jk^(2) | x,C] = (q_b(x,u_j) − q_b(x,u_k))².

Thus the average cross-seed product estimates persistent squared action differences without maximizing noisy success estimates. With m replications, use the unbiased statistic

S_jk = [(Σ_r d_jk^(r))² − Σ_r (d_jk^(r))²] / [m(m−1)].

Average within a source state, then across source identities; use clustered uncertainty. A finite-sample estimate can be negative. Do not clip it before inference. Common random numbers can help or hurt precision; the identity requires independence between replications, not independence between candidates within a replication.

This is a diagnostic I propose for this project, based on a standard cross-product identity. It does not establish useful mean uplift, observable predictability, or a new estimation theorem. Alongside it, report independently evaluated selected-minus-default returns. For descriptive “best candidate” curves, choose candidates using one set of seeds and evaluate on other seeds; label this as the performance of that finite-data selection procedure, not an unbiased true oracle.


**Experiment A: establish the contract and the cheapest credible baseline.**

The first deliverable is a small reproducible audit, not another large collection job.

- On twenty development states, verify replayed observations, simulator state, wrapper/controller state, remaining steps, and the first executed action sequence. Test source execution and continuation separately. The current collector uses action replay and may correct physical state afterward; checking only a flattened simulator vector is insufficient.
- Verify that diagnostics leave the original sampler unchanged, and compare captured features with features reconstructed from the same observation. Version the source horizon, intervention horizon, continuation horizon, solver settings, checkpoint, candidate seeds, and continuation seeds.
- Audit proposed actions, masks, normalization, and time inputs. Replace invalid inputs with the original pre-decision proposals; simply changing the terminal mask does not restore overwritten actions.
- On eligible existing development data, rerun the compact FiLM critic with its observation encoder frozen versus jointly trained. Keep data, optimizer budget, action encoder, and loss fixed. This directly tests the compression hypothesis. Repair cross-attention’s action-order issue separately.
- Profile ordinary generation at 3 and 10 denoising steps, then batched generation with four candidates. Exclude duplicate diagnostic passes from the bare-policy measurement.

Historical data with full-chunk continuation can support a controlled legacy-model ablation under those semantics. It must not be pooled silently with new stock10-continuation labels. If provenance is insufficient, classify those rows as ineligible for the new target.

Do not reconstruct the entire roughly 150-GiB embedding corpus first. Reconstruct a small matched sample and measure the cost. Historical mean embeddings cannot be inverted into contextual token arrays; obtaining those arrays requires original observations or exact replay.

**Experiment B: fund at most 992 branch continuations to find out whether the problem has signal.**

Use 64 distinct underlying task/initialization identities, targeting eight task/suite conditions with eight initializations each, spread across two already-development regimes. Confirm availability before fixing that allocation. Choose one live decision boundary per identity using a predeclared task/phase sampling rule. Do not select states using candidate outcomes or exclusively high P&P uncertainty.

This is a balanced diagnostic panel, not an unbiased estimate of every decision encountered during deployment. Keep final task/shift cohorts out of it.

| Component | Allocation | New branch continuations |
|---|---:|---:|
| Main panel | 64 identities × 4 fixed candidates × 2 continuation seeds | 512 |
| More precise repetition | 16 identities × the same 4 candidates × 6 additional seeds | 384 |
| Fresh candidate pool | Those 16 identities × 3 new alternatives × the original 2 seeds | 96 |
| Total | Default action and its outcomes reused in the fresh-pool comparison | 992 |

Choose the 16 repeatability identities before viewing candidate outcomes. Preserve their default action while refreshing the other three candidates. This distinguishes variation in future continuation from variation in which alternatives happen to be sampled.

The first 512 branches estimate the cross-seed action-effect statistic. The repeated subset measures how fragile candidate rankings are. The refreshed pools investigate whether benefit is a property of information available before search or of a fortunate pool drawn afterward.

Collect terminal success, not a short-distance proxy. Source rollout, replay, candidate generation, and failed collection attempts are additional costs. Record simulator transitions and wall time, then decide whether the later allocations fit the actual budget.

Existing eligible stock10/execute10 rollouts can provide pointwise success supervision. New branches provide direct within-state contrasts. Use identity-level out-of-fold predictions for pilot diagnostics; report the small sample size honestly. This panel can detect large problems and guide the next allocation. It cannot establish a three-point policy improvement or a robust neural learning result by itself.

**Experiment C: test the representation and supervision hypothesis, with a bounded model family.**

Use a compact temporal action encoder and a small attention readout from observation tokens. Action tokens receive time positions. Preserve camera/spatial identity where available. Include instruction, proprioception, and remaining time. Train the readout jointly with the scoring head; freeze all VLA parameters. Start around width 128–256 with one or two readout blocks. There is no evidence yet for an eight-layer, width-768 critic.

Use a scalar success predictor f_φ(h,u) in [0,1]. Compare ordinary pointwise binary cross-entropy with the same loss plus a paired difference term:

L_pair = average_j<k,r [(f_φ(h,u_j) − f_φ(h,u_k)) − (Y_j^(r) − Y_k^(r))]².

Normalize both loss terms by source state and use one predeclared coefficient, initially 1. The pointwise term anchors success probabilities; the paired term emphasizes within-state action differences. Concordant observed outcomes remain valid noisy zero differences. Their inclusion does not assert that the actions have identical true values.

At population level, both proper pointwise prediction and the difference target are compatible with the correct conditional values. There is no theorem that the paired loss must win. Its proposed benefit is finite-data representation learning: reduce reliance on between-state difficulty when learning the distinctions used for selection.

The decisive comparison is a 2×2, using the same head and training budget:

| Frozen VLA features | Pointwise loss | Pointwise + paired loss |
|---|---|---|
| Pre-prefill token arrays | Feature/loss reference | Does paired supervision help without final contextualization? |
| Final contextualized token arrays | Does contextualization alone help? | Does it interact usefully with paired supervision? |

Use a separately trained action-only model, the repaired legacy pooled-feature model, and a repaired Q10 TD model as controls. Keep the TD target’s objective explicit. If a representation cannot be extracted equivalently for all rows, run the factorial on the common reconstructible subset.

Train three seeds per fixed recipe. Split by underlying source identity; group related perturbations when testing task transfer. No source trajectory may supply training windows and evaluation candidate groups. Fit normalization only on training data. Report the four predeclared comparisons rather than selecting an unexplained “best checkpoint.”

Measure independent-seed selected-minus-default success first, then ranking, calibration, and context sensitivity. A zero-context ablation of a trained model is not a separately trained action-only baseline. Shuffled context is a diagnostic for dependence, not necessarily a valid in-distribution control. Check whether improvements occur within tasks and phases; otherwise an apparent contextual advantage may just identify easy tasks.

Bradley–Terry training on discordant pairs is not inherently wrong. For any valid paired coupling, P(win) − P(loss) = q_j − q_k. The new loss is an empirical alternative, not a correction of a universal ranking inconsistency.

If Experiment B shows useful repeatability but the pilot is too small for learning, the next default tranche is **256 new identities × 4 candidates × 2 seeds = 2,048 branches**. Reserve 64 identities for development validation; use the other 192 for fitting. Plot learning curves by independent identities and environment transitions. Extend source-state diversity before spending many more seeds on already well-characterized states.

After freezing one selector, obtain an independent, two-arm audit: initially 128 new identities × selected/default × 2 seeds = 512 branches. The chosen action is fixed without seeing audit outcomes. If it equals the default, reuse that outcome and record zero effect exactly. Size any expansion using measured precision; these starting counts are not powered guarantees.

**Experiment D: test one intervention, then repeated control.**

Run a small development check once a selector is credible. A concrete first batch is 48 fresh episode identities and four policies, totaling 192 episodes:

| Policy | Purpose |
|---|---|
| Stock10, execute10 | Reference |
| Bare three-step, execute10 | Cheap-sampler efficiency hypothesis |
| Stock10 with the four-candidate selector at one predeclared decision opportunity | Tests a single intervention followed by the reference policy |
| Stock10 with that selector at every decision | Tests repeated use and changed state occupancy |

For the single-intervention policy, fix the opportunity rule without future outcomes. Episodes that finish before it occurs remain in the comparison with no intervention. Do not drop them. Use shared initializations and a declared noise-coupling scheme across arms.

Why this distinction matters: the fork critic estimates the return from one changed prefix followed by π_b. Repeated selection changes the distribution of future states. For the augmented Markov state and finite undiscounted episode, the performance-difference identity is

J(π_sel) − J(π_b) = E_π_sel[Σ_t A_b(x_t,u_t)],

where A_b(x,u)=q_b(x,u)−V_b(x), and t indexes executed decision chunks. Exact pointwise improvement supports policy improvement, but positive average advantage on a stock-policy panel is insufficient. The full-episode effect depends on advantage at the states the new policy actually visits.

If the single intervention helps and repeated selection fails, inspect newly visited states and the accumulation of spoils. Permit one targeted round of fork labels at states visited by the selected policy, with the original continuation target retained. Freeze the repaired model and evaluate on new identities. This is a bounded distribution-coverage repair, not permission for indefinite online RL.

If the four-candidate selector works, compare completed three-step and ten-step candidate pools using the same critic, checking critic accuracy on each proposal distribution. Advance the promising 3/10-step × 1/4-candidate configurations to closed-loop development as budget permits. If the reference continuation is changed to three-step sampling, new value labels must name that policy; old labels do not change meaning.

**Experiment E: a compute gate must pass both an information test and an economic test.**

There is an information gap the first plan underemphasized. Let I_0 contain everything available after the default action is generated but before extra candidates are bought. Let C denote the still-random candidate pool. The quantity a pre-search gate needs is

m(I_0) = E[Δ(x,C) | I_0].

Repeated seeds at a fixed pool estimate its effect, not automatically this expectation over unseen pools and hidden state. A gate may fail because useful alternatives appear unpredictably, even when a post-search selector is good. The fresh-pool component of Experiment B directly probes this issue.

At constant search cost and expected search fraction κ, a gate’s advantage over random allocation is

E[g(I_0)m(I_0)] − κ E[m(I_0)] = Cov(g(I_0),m(I_0)).

Large variability in realized search outcomes does not imply large predictable variability in m. A post-search guard that sees all candidates is a useful diagnostic comparator, but it has already paid for search and cannot be credited with avoiding that cost.

Profile context encoding, prefill, denoising, critic evaluation, data movement, and gate features separately. In an illustrative model where context accounts for fraction p of ten-step latency, reducing denoising from ten to three steps gives cost ratio

C_3/C_10 ≈ p + 0.3(1−p).

If p=0.8, the ratio is 0.86: only 14% savings. Also, four batched proposals sharing context may be inexpensive enough that an elaborate gate has little room to help. Measure this before designing the allocator.

With gate overhead C_g and incremental search cost ΔC, merely saving compute relative to always-search requires

C_g < E[(1−g)ΔC].

That is a necessary cost condition, not a utility guarantee. Paid uncertainty probes belong in C_g, including on decisions where the gate declines search.

Only after these checks would I collect a larger gate dataset. Freeze the selector and search operator first. Collect paired selected/default outcomes; use D = Y_selected − Y_default, including zeros and harms. Start with a regularized small predictor of E[D|I_0]. Use independent fitting, threshold-selection, and audit identities, or properly nested identity-level cross-fitting during development.

Compare random, default-value, task/phase, disagreement, critic-surrogate, and outcome-supervised allocation at matched achieved cost. For the clean target ablation, give surrogate and outcome gates identical pre-search features and architecture. Train the surrogate to predict the later critic margin; do not provide that margin as a free input.

For the stronger system comparison, let the alternative spend the same extra simulator interactions on improving its critic. A gate that wins only because it received more useful labels does not establish superior data allocation. If a simple task/phase rule matches it, deploy the simple rule.


**The diffusion argument should be precise about what uncertainty means.**

For the linear flow-matching path a_t=(1−t)a_0+tε, an exact conditional velocity field satisfies v*(a_t,t,h)=E[ε−a_0|a_t,h]. Consequently,

a_t − t v*(a_t,t,h) = E[a_0|a_t,h].

This identity requires the specified path and exact field. It says that the one-step clean estimate is a conditional mean; it is not a sampled complete action trajectory. The framing follows the standard flow-matching construction. [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264).

For a nonlinear task-value function, Q(E[a_0|a_t,h]) generally differs from E[Q(a_0)|a_t,h]. Around contact transitions, there is no useful universal smoothness or convexity assumption that equates them. Averaging two good but incompatible actions may produce a poor action.

The implemented [P&P probe](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/pnp.py:138) alternates a clean estimate with re-noising at fixed noise level. Its successive samples are dependent. This operation does not by itself sample the posterior conditioned on the original noisy point. Spread depends on noise level, the vector field’s local sensitivity, and proposal geometry. It does not automatically estimate epistemic uncertainty or the value of spending more compute.

More ODE steps better approximate a particular learned flow under suitable numerical assumptions. They do not guarantee higher task success: the flow was trained to model behavior, and discretization error need not align with return. This is why completed three-step candidates deserve a direct experiment, while a clean-action critic applied to arbitrary intermediate estimates needs separate validation.

P&P earns a role only if its executed-prefix statistic predicts independently measured benefit beyond cheaper default-value and action-spread features, after probe cost is included. Fix a small number of probe settings on development data. If that comparison fails, retire P&P from the method.

If guidance becomes justified later, validate its proposed direction through simulator interventions before optimizing against the critic more aggressively. Compare directly with [QGF](https://arxiv.org/abs/2606.11087v1), [QPILOTS](https://arxiv.org/abs/2606.14801v1), and [Flow Map Q-Guidance](https://arxiv.org/abs/2605.12416v1). Completed-candidate ranking success does not establish gradient accuracy off the proposal distribution.

**The mid-2026 literature makes this a demanding empirical thesis.**

The following works constrain what would count as a contribution. June and earlier entries provide the requested mid-2026 context; later entries are explicitly updates available by this September recommendation. These are primary-source reports, not replications here.

| Relevant work | Implication for this project |
|---|---|
| [π0.5](https://arxiv.org/abs/2504.16054v1), [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759v2), [π0.7](https://arxiv.org/abs/2604.15483v2) | Heterogeneous pretraining, experience-driven policy improvement, and richer context conditioning are central to modern VLAs. A pinned π0.5 study must demonstrate a control principle; it cannot stand in for the entire frontier. Freezing the backbone is an experimental constraint, not an argument that adaptation is undesirable. |
| [V-GPS](https://arxiv.org/abs/2410.13816), [RoboMonkey](https://arxiv.org/abs/2506.17811) | Value-guided frozen-policy selection and VLA sampling/verification already exist. Four-candidate reranking is a baseline architecture. |
| [DA-SIP](https://proceedings.neurips.cc/paper_files/paper/2025/hash/49eadcc4a329fc6b74b9f8a82b78cbc3-Abstract-Conference.html), [VLA-ATTC v2](https://arxiv.org/html/2605.01194v2) | Difficulty-adaptive denoising and disagreement-triggered VLA search already exist. ATTC also uses a relative critic. An uncertainty gate around our critic is a controlled ablation, not a faithful reproduction of ATTC. |
| [ELASTIC, June](https://arxiv.org/html/2606.31132v1) | Joint sequential/parallel allocation is directly occupied, including π0.5. Its counterfactual compute labels use verifier scores, but its critics themselves receive outcome-related supervision. The distinction here is direct evaluation of the deployed selector’s intervention, not a claim that ELASTIC ignores outcomes. |
| [FASTER: Value-Guided Sampling for Fast RL, April](https://arxiv.org/html/2604.19730v1) | Denoising-space value learning and early candidate filtering already have VLA evidence. Any later pruning project needs this comparator. This is distinct from the similarly named real-time VLA paper. |
| [RL Token v2](https://arxiv.org/html/2604.23073v2), [Decoupled Q-Chunking](https://arxiv.org/abs/2512.10926v2) | Compact contextual representations and separation of value-backup/action-execution horizons have precedents. They motivate controls; they do not validate our local representation or license conflating generated and executed chunks. |
| [Q-VGM v3, August 24](https://arxiv.org/abs/2606.08015v3), [Q-Planning, August](https://arxiv.org/html/2608.21204v1) | Distinguish policy updates from frozen-policy planning. Q-VGM’s revised method uses critic-derived velocity targets to train the policy; Q-Planning improves Q-based planning while retaining a frozen policy. The local Q-planning implementation is an adaptation, not a paper reproduction. |
| [Decoupling Policy Extraction, August](https://arxiv.org/abs/2608.20909) | Separately trained behavior proposals and inference-time critic reranking are themselves an explicit contemporary research direction. The decomposition alone is not novel. |
| [When Vision Overrides Language / LIBERO-CF](https://arxiv.org/abs/2602.17659), [CounterAlign, August](https://arxiv.org/abs/2608.21740) | Semantic shortcuts and instruction-inconsistent actions deserve direct tests. CounterAlign constructs counterfactual instruction negatives rather than the simulator action interventions proposed here. “Counterfactual supervision” alone is not a novelty claim. |

There is also an unreviewed July implementation report that explicitly proposes paired baseline/repair outcomes for a gate. It ultimately uses an eight-step distance proxy because its long-horizon state branching was unreliable. Its limited evaluation does not validate our hypothesis, but the conceptual overlap rules out a broad first claim for intervention-trained gating. [Repairing a Frozen Visuomotor Policy With Counterfactual Regret Labels & Flow Matching](https://amohan.dev/blog/2026/repairing-frozen-visuomotor-policy-cfr-flow-matching/).

The defensible ambition is therefore specific: **identify repeatable, observation-accessible action effects; show that learning them improves a frozen VLA under feedback; and measure how much of that benefit can be retained at lower inference cost.** The loss and cross-product identity are tools for this investigation. Do not market either as a new foundational algorithm.

A publishable method result would need a convincing advantage over a repaired pointwise critic and credible V-GPS/RoboMonkey-style selection controls. An adaptive-compute superiority claim additionally needs a faithful or clearly scoped ELASTIC/ATTC comparison. Match training interactions, available observations, proposal budgets, and actual latency; disclose deviations. A second policy family would be necessary for a broad claim across VLAs.

**Generalization should test the mechanism, not merely add more perturbation names.**

Keep three failure explanations separate:

- **Proposal support:** all available actions are wrong, so search has little opportunity.
- **Observation/critic failure:** good actions exist, but the evaluator cannot identify them.
- **Allocation failure:** selection can help, but its benefit is not predicted at useful cost.

Reserve one genuinely unused shift family or parameter region before new development. Inventory previous training, plotting, threshold choice, and discussion. The familiar 220 identities are development data. Withholding position perturbations from training does not make them pristine after repeated tuning.

Use a diagnostic branch panel under the withheld shift to localize failure, separate from the fresh episode panel used for the final performance claim. Try to distinguish moderate geometry changes from instruction/object substitutions. For a semantic intervention, change and verify the task’s success predicate along with the instruction; merely editing the text can produce invalid labels.

The transfer hypothesis is also specific. If a shift changes candidate values by a common state-dependent offset, or by a common positive scale and offset, their ordering survives even when absolute success calibration changes. Instruction changes can instead reverse the correct ordering. This elementary distinction motivates comparing relative discrimination and absolute calibration under separate shifts; it does not assert that real geometry shifts obey the affine model.

My prediction is that moderate geometry changes are the more favorable regime: the policy may already propose both successful and unsuccessful variants. Semantic failures are harder because policy and critic can share the same visual shortcut. More samples from that shared error may add no useful alternative. Final contextualized features could help, but also inherit the same shortcut; their superiority under shift must be measured.

Use deployment-valid inputs only. Simulator object state can diagnose observability or define an explicitly privileged oracle, but it cannot enter the deployed critic. The logged robot proprioception is not privileged object state.

Report task-macro and pooled success, per-shift effects, rescues, spoils, and their intervals. Cluster repeated seeds and boundaries by underlying initialization. Where perturbations share the same base task, also show task-grouped uncertainty. Some suites have only ten initial states: additional policy seeds are replications, not new initializations.

**The decision tree is a set of funding decisions.**

| Evidence at the scheduled review | Decision |
|---|---|
| Replay or target semantics are unreliable | Stop label collection and repair the contract. Do not interpret the critic result. |
| Cross-seed action differences are negligible, and independent candidate evaluation excludes useful default-relative opportunity | Stop search at this proposal budget and distribution. Continue the bare-sampler track if promising. |
| Persistent action differences exist, but no candidate beats the default usefully | Do not confuse variation with opportunity. Investigate proposal support only within a bounded extra budget. |
| Useful alternatives exist; unfreezing the encoder or preserving action order resolves the deficit | Continue with the simple corrected model. Attribute the gain to the tested change. |
| Opportunity exists but the controlled critic family cannot exploit it after the bounded data tranche | Diagnose observability and coverage; stop the allocation project. One unresolved critic is not a universal impossibility result. |
| Contextual paired training beats matched controls and improves fresh single interventions | Advance to repeated-control and shift tests. |
| Single interventions improve but repeated control fails | Allow one targeted occupancy-coverage repair; re-evaluate on new identities. |
| Selection works but cheap pre-search features cannot rank its benefit, or gate overhead erases savings | Keep the selection result. Stop learned gating. |
| Always-search is neutral but an independently audited simple subset rule improves outcomes | Selective allocation remains viable. A positive always-search average is not required. |
| Outcome gating matches task/phase, default-value, or equal-label-budget critic improvement | Use the simpler or more data-efficient system. Drop the stronger gate claim. |
| Bare three-step sampling is noninferior and substantially cheaper | Promote it as the practical baseline; pursue only search that improves its frontier. |
| At the resource cap, intervals still include both useful improvement and no benefit | Record an inconclusive study and its precision. Do not manufacture a positive or negative conclusion. |

For a scoped claim that requires an effect of at least δ, an upper confidence bound below δ can justify stopping that claim. An interval merely crossing zero cannot. Use fixed audit points or a valid sequential procedure, rather than repeated informal peeking.

**Set the confirmation target after profiling, before looking at confirmation outcomes.**

The main proposed success result is improvement from the frozen-backbone contextual selector over stock10 and the strongest matched critic control. Choose the primary contrast in development. If a gate becomes the main method, its strongest eligible cheap allocation baseline becomes the primary matched-cost contrast, while stock10 remains an anchor.

A reasonable planning target is a three-percentage-point episode-success improvement at a predeclared practical compute budget. The first budget to investigate is at most 1.25 times stock10’s mean inference cost, but profiling may show it infeasible. Set the feasible budget before confirmation. For the separate efficiency claim, my planning defaults are a two-point noninferiority margin and at least 25% less inference cost. These are proposed research thresholds, not measurements or universal deployment requirements.

A positive superiority interval establishes improvement; claiming an improvement of at least three points requires the lower bound to exceed three points. Noninferiority needs its own one-sided design.

For paired binary episode outcomes with discordance d and target difference δ, an independent-pair approximation is

n ≈ (1.96+0.84)² d / δ².

At d=0.10 and δ=0.03, n≈871 paired identities for 80% power at two-sided 5%. Task clustering and repeated seeds change the design. The 48-identity development batch is not confirmation. If the available independent cohorts cannot support the desired claim, narrow the claim or report the uncertainty.

Measure mean and p95 decision latency, total episode inference time, environment steps, peak memory, and success jointly. A policy that fails faster may consume less episode compute. Therefore also report cost on a common decision panel, where work is compared independently of outcome-driven episode length. Match achieved cost without retuning thresholds on test outcomes.

**A concrete month of work.**

| Work window | Required artifact | Condition for releasing more work |
|---|---|---|
| First several days | Versioned intervention contract; twenty-state replay check; frozen/joint encoder ablation; 3/10-step timing table | Inputs and labels correspond to the intended controller |
| Remainder of week 1 into week 2 | The 992-branch panel, including preselected repeated seeds and refreshed pools | Useful opportunity remains plausible and its precision justifies further learning |
| Weeks 2–3 | Fixed representation/loss comparisons and identity-level learning curves; optionally the 2,048-branch training tranche | Observable action selection beats appropriate controls on independent outcomes |
| Weeks 3–4 | Frozen-selector branch audit and 192-episode development comparison; proposal/latency frontier | Repeated control is credible; compute savings and benefit heterogeneity justify any gate |
| After that review | One locked confirmation plan and, only if justified, a gate study | Reserve independent evaluation cohorts and enough budget to resolve the chosen effect |

These are sequencing estimates, not promises of GPU-hours. Profile collection throughput before committing the larger tranches. Source replay and long continuations can dominate cost. Protect the confirmation allocation; do not spend every available identity developing the model.

My expected outcomes are uneven. I am most confident that the present verifier experiment can be made substantially more informative, and that bare-sampler profiling is worth doing. I have moderate confidence that a small critic with preserved context and temporal structure can improve some decisions. I have lower confidence that paired loss adds materially beyond a properly trained pointwise critic, and lower confidence still that a learned pre-search gate beats simple allocation rules after costs and data budgets are matched.

The research vision is a frozen generalist policy whose behavior can improve through a small evaluator trained on consequences. The evaluator should learn *which action distinction matters in this situation*, rather than merely recognizing a situation that often fails. If that succeeds, adaptive computation becomes a disciplined way to deploy the improvement. If it fails, the experiments should reveal whether the missing resource was a better action proposal, more informative observations, or better outcome supervision.

The first figure I want is persistent action-effect signal alongside hindsight headroom and independent default-relative opportunity. The second is a controlled representation/loss learning curve with held-out intervention utility. The third is single-intervention and repeated-control success at measured compute cost, including the withheld shift. A gate frontier is a fourth figure earned by those results—not the premise on which the project depends.
