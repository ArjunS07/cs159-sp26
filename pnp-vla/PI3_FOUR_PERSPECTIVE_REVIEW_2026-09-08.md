# PI3: four independent perspectives on the research decision

2026-09-08 · Decision-focused review of the [research plan](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PI3_FINAL_RESEARCH_PLAN_2026-09-08.md)

**Keep the project focused on reliable policy improvement from a frozen proposal policy. Reduce the importance assigned to a new loss, a new uncertainty statistic, or a new compute gate.** The four perspectives below ask different questions and do not receive equal weight on every decision.

| Perspective | Its central question | The change it requires |
|---|---|---|
| Deep RL | Does improving this decision improve the policy that will actually run afterward? | Separate state-coverage failure from continuation-policy dependence; use a conservative deployment diagnostic. |
| Diffusion | Does extra generation change the distribution of useful executed prefixes? | Compare completed proposals with coupled seeds and charge their full cost; do not identify denoising uncertainty with decision value. |
| Empirical methodology | Will the proposed experiment resolve a decision at its available sample size? | Demote the squared-effect statistic and make the second collection tranche conditional. |
| Statistical learning | What information or inductive bias does the proposed learning objective add? | Recognize the paired-loss identity; prioritize data and representation before loss design. |

These are analytical judgments. I checked the loss identity and illustrative variance calculations numerically. No new VLA training or simulator evaluation was run.

**1. Deep RL: this is approximate policy iteration with an unusually strong proposal distribution.**

The frozen VLA supplies plausible actions. A critic estimates the consequences of those actions under a particular future policy. A selector defines a new policy. That is the useful RL interpretation; it immediately exposes two distinct sources of failure.

First, repeated selection visits states different from those in the training panel. Better state coverage can repair that problem while retaining the same reference-policy value target.

Second, the value of an action can depend on what the robot does later. A prefix that positions the gripper for an alternative grasp may be useful only if later actions complete that grasp. If the reference continuation abandons it, the prefix’s q_b value can be low even though a coordinated policy would succeed.

This does not make q_b the wrong target. It is the correct target for “one changed prefix, then return to the base policy.” It limits the conclusion available from a negative result. Failure to find useful single-prefix interventions rules out that operator under that continuation; it does not rule out coordinated policy improvement.

A simple bottleneck model illustrates the issue. Suppose the prefix improves the probability of reaching a useful intermediate state by ε, and the reference policy succeeds afterward with probability ρ. The terminal-success difference is ρε. If ε=0.2 and ρ=0.1, the observed value difference is only 0.02. The long-horizon label faithfully measures the requested intervention while providing weak supervision about the intermediate distinction.

**Necessary improvement:** if single interventions and repeated control disagree, use a small diagnostic that varies both the prefix and the continuation:

| | Base continuation | Frozen selected-policy continuation |
|---|---|---|
| Default prefix | q_b(u_0) | q_sel(u_0) |
| Chosen prefix | q_b(u_1) | q_sel(u_1) |

The interaction

[q_sel(u_1)−q_sel(u_0)] − [q_b(u_1)−q_b(u_0)]

tests whether the value of switching prefixes changes with the continuation. Freeze both policies before obtaining these labels. Reuse the base-continuation branches when possible. This is warranted only when the observed discrepancy would change the next training decision.

If the problem is changed state coverage, add examples from the selected policy’s states. If it is continuation dependence, one explicit policy-evaluation refresh may be more appropriate. Keep a single bounded repair budget; do not silently mix labels from different continuations.

I would also use a simple mixture when unrestricted repeated selection is harmful: at each decision, invoke the existing selector with probability α, otherwise use the base policy. Compare α=0, one prespecified intermediate value such as 0.25, and α=1. This introduces no learned gate. It tests how intervention frequency changes compounding error and compute.

Mixture updates have a classical role in conservative policy iteration. They reduce the size of a policy change, but do not provide a practical safety guarantee for an inaccurate critic. [Kakade and Langford, Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf).

One further correction is necessary for the baseline: compare Monte Carlo and TD under the same finite-budget success objective before attributing a difference to their learning methods. Use a reference-policy TD target with the remaining horizon represented and no time discount for that comparison. The historical γ=0.99 target can remain a separately named system baseline. Mixing objectives confounds the experiment.

**RL verdict:** fund a valid small critic and one bounded policy-improvement step. Treat rollout continuation, execution horizon, and state coverage as parts of the scientific problem.

**2. Diffusion: useful action diversity is a property of the executed-prefix distribution.**

For fixed observation h, let F_K(z;h) be the complete action trajectory generated using K solver steps. The deployed proposal is

u = P_H F_K(z;h),

where P_H keeps the H actions actually executed before replanning. The relevant proposal law is the distribution of u. Diversity in the discarded tail is useful only insofar as it predicts or changes that prefix.

This matters because a fifty-action generator can look highly multimodal while proposing nearly identical first ten actions. Conversely, a small prefix difference can decide contact success. Euclidean spread over the full trajectory is neither a lower bound on useful alternatives nor a reliable ranking criterion.

The numerical approximation and the control objective are different. More solver steps aim to approximate the learned flow more faithfully. They need not improve task success if the learned flow’s distribution includes failures. The flow-matching framework establishes the generative construction, not monotonic task-return improvement with solver count. [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264).

**Necessary improvement:** for the planned 3-versus-10-step comparison, reuse the same initial noise for the paired numerical comparison, while also drawing multiple independent noise seeds. Inspect differences in executed prefixes and evaluate their outcomes under the same reference continuation. This distinguishes changes induced by solver resolution from changes induced by candidate sampling. It costs little beyond organizing the experiment correctly.

Completed candidate selection is the clean first mechanism. Applying an outcome critic to early clean estimates changes its input distribution; P&P adds a dependent re-noising process whose spread has no automatic interpretation as epistemic uncertainty. A new pruning or steering algorithm would introduce another learning problem before the current evaluator has earned trust.

There is also a useful longer-term distinction: a flow policy can serve as a behavior prior without requiring expensive iterative improvement at every deployment decision. FQL, for example, separates an expressive flow behavior model from an RL-trained one-step policy. Its results are not a replication on this project’s VLA, but they show why successful critic learning need not lead inevitably to a complicated inference sampler. [Flow Q-Learning](https://arxiv.org/abs/2502.02538v2).

That does not justify adding actor distillation now. It means that, after a reliable improvement operator exists, amortization is a legitimate alternative to increasingly elaborate test-time compute allocation.

**Diffusion verdict:** retain the frozen flow as a strong proposal mechanism. Measure useful prefix diversity and the completed-sampling frontier. Require evidence before working inside the denoising trajectory.

**3. Empirical methodology: the pilot must have enough signal to change a decision.**

The previous plan correctly recognized that taking the maximum observed binary outcome can manufacture apparent opportunity. Its proposed cross-seed statistic is unbiased for squared action differences. Unbiasedness, however, does not make it a powerful diagnostic at this scale.

Take a toy pair with success probabilities 0.6 and 0.5. Under independent future outcomes within each replication, D=Y_1−Y_0 has mean 0.1 and second moment 0.5. For independent replications,

E[D_1D_2]=0.01,

Var(D_1D_2)=0.5²−0.1⁴=0.2499.

A mean across 64 such independent pairs has standard error approximately 0.0625—much larger than the signal 0.01.

Even an ideal monotone common-random-number coupling does not eliminate the problem. In that toy, D is one with probability 0.1 and zero otherwise. Its two-replication product is one with probability 0.01. Across 64 pairs, the probability of seeing no positive product is 0.99^64≈0.526, despite a real ten-point action-value difference.

These calculations concern a single-pair toy, not a power calculation for the entire four-candidate panel. They nevertheless show why a null squared-effect estimate must not veto the project.

**Necessary improvement:** use independently evaluated selected-minus-default outcomes as the primary utility measure. Keep repeated-seed statistics for explaining apparent opportunity and estimating noise. Do not require a significant squared-effect result before fitting or auditing the small critic.

Keep the first 64 identities × 4 candidates × 2 seeds = 512 continuations. Treat the remaining 480 of the existing 992-continuation cap as a conditional allocation, rather than automatically spending 384 on six extra seeds at sixteen states. Use initial variance and replay-cost measurements to choose between more identities, more repeats, and refreshed pools. Record that adaptive development rule; keep final confirmation independent.

Another subtle point: mean paired improvement is invariant to a valid coupling with fixed marginals, but rescues, spoils, and observed discordance are not. Two actions with q values 0.51 and 0.49 can have discordance anywhere from 0.02 to 1 under different valid couplings. Common seeds therefore define a useful variance-reduction and debugging convention, not a unique intrinsic count of episodes “saved” or “destroyed.” Report the coupling and prioritize the marginal success difference.

Finally, if simulation pauses while the policy computes, lower inference latency does not itself improve feedback timing inside that benchmark. It is still a valid efficiency result. A claim about reacting faster on a physical robot requires a clocked evaluation or hardware evidence. Do not infer that control benefit from an ordinary stepped simulator.

The central methodological lesson is consistent with the RL evaluation literature: aggregate point estimates from a few runs can support different conclusions once uncertainty and task variation are considered. [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264).

**Empirical verdict:** run fewer, more interpretable comparisons; release further collection according to what the first batch can resolve. Preserve a credible independent endpoint.

**4. Statistical learning: the intervention data matter more than naming the loss.**

For a state with N candidates, let e_j=f(h,u_j)−Y_j. The proposed all-pairs squared loss satisfies

Σ_(j<k) (e_j−e_k)² = N Σ_j (e_j−mean(e))².

It is exactly a penalty on within-state centered prediction errors. It does not create extra information beyond the same N labels, and the N(N−1)/2 pairs are not independent examples.

This identity does not make the loss useless. It explains its inductive bias: it emphasizes errors that change action ordering, while ignoring a common offset that does not affect within-state selection. With a shared, capacity-limited representation, that reweighting could help. The benefit is a finite-data optimization/representation question, not a new form of supervision by itself.

The proposal should distinguish three claims:

1. Better preserved context enables useful action discrimination.
2. Forked outcome collection is a good use of simulator interactions.
3. Reweighting the same labels toward within-state differences improves learning.

The first two are the main opportunities. The third is a small ablation.

**Necessary improvement:** run the corrected pointwise critic first. Keep paired-loss training as a matched secondary treatment, using the same labels and budget. If it helps, explain the result as improved emphasis on within-state errors. A Brier-loss version can be checked directly against the equivalent centered-residual formulation; implementing both as separate algorithms would be redundant.

If the paper’s eventual claim concerns the value of forked supervision, compare its acquisition strategy against spending a matched simulator-transition budget on ordinary policy rollouts. Ordinary on-policy outcomes are not inherently causally invalid: with sufficient observed state and appropriate coverage, they can learn action values too. The possible advantage of forks is controlled coverage of alternatives and useful covariance, balanced against expensive replay and reduced state diversity. That is an empirical data-efficiency question.

The earlier frozen-encoder and action-order findings remain actionable. They identify specific bottlenecks worth repairing. They do not prove the frozen VLA contains all information needed for the critic. A learned action-value predictor can still fail because observations alias different physical situations, because the representation omits their distinction, or because too few states were labeled.

Only if those alternatives obstruct a funding decision would I add a privileged-state diagnostic or short-history feature comparison. An arbitrary architecture expansion would not tell us which problem occurred.

**ML verdict:** repair the known information bottlenecks, train a compact proper predictor, and test its decisions. Let the evidence justify additional representation or loss complexity.

**The reconciled recommendation.**

The scientific objective is now more precise: **under a limited interaction and inference budget, can outcome-supervised action selection produce a reliable policy improvement from a frozen VLA proposal distribution?**

The next execution sequence remains short:

1. Verify the ten-action intervention/continuation contract and fix the known encoder/order issues.
2. Collect the 512-branch core panel and profile the 3/10-step completed samplers.
3. Fit and independently audit the small pointwise contextual critic, with an action-only control. Add the paired loss as one secondary comparison.
4. Test one intervention and repeated selection. If they disagree, use one bounded mixture/continuation diagnostic to choose the appropriate repair.
5. Spend the remaining diagnostic budget on the uncertainty that blocks the decision. Fund a learned gate only after useful selection and a meaningful cost-saving opportunity exist.

This keeps the strongest parts of the existing plan and removes two sources of unnecessary commitment: treating a centered-error loss as the main idea, and treating a noisy squared-effect estimate as a mandatory gateway.

My best bet is that data validity, preserved task information, and restrained policy changes will matter more than a sophisticated denoising procedure. If that bet is wrong, this sequence should reveal it with a limited amount of new work.
