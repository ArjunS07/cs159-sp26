# PI research vision: selective improvement of a frozen VLA
Date: 2026-09-08

**Recommendation.** Organize the project around a falsifiable question: can a frozen VLA spend extra inference compute only where a validated action selector can improve the next executed decision? Establish a critic over the ten executed actions, validate it through paired interventions, and then study compute allocation. Keep P&P as a candidate measurement and proposal mechanism whose usefulness must be earned empirically.

This is a research judgment, not a report of new experimental results. Repository results below come from [PROJECT_STATUS_HANDOFF.md](/Users/arjunsharma/development/cs159-sp26/pnp-vla/PROJECT_STATUS_HANDOFF.md). I inspected the sampler, P&P probe, Q-planning window construction, critic, and target calculation, and read relevant primary literature. I did not rerun rollouts, train models, query current Supabase state, or open the sealed cohort.

**What the evidence supports.**

The project has substantial experimental infrastructure: frozen model provenance, paired initializations, replayable simulator branches, separated randomness, an immutable 1,880-rollout dataset, and explicit execution horizons. These make it possible to investigate mechanisms rather than rely on aggregate benchmark changes.

The present empirical case for an improvement method is weak. The selected conditioned verifier did not exceed its action-only control in Stage 2. This does not establish that visual conditioning is useless; it establishes that the particular representation, supervision, and training procedure did not demonstrate its value. No completed Q10/Q50 training results were documented at handoff.

On the 220 matched September identities, stock succeeded in 117 episodes, three-step sampling in 118, and three-step sampling with refinement in 120. The last method's stock-relative interval was [-2.27, 5.45] percentage points. These data are compatible with useful benefit, negligible effect, or modest harm. Selecting the largest estimate across many variants exaggerates its evidential strength.

The three-step result is nevertheless interesting as an efficiency hypothesis. Similar observed success with fewer velocity evaluations motivates a direct latency and noninferiority study. It is not yet evidence of equivalent performance or a threefold end-to-end speedup.

The historical branch oracle headroom, approximately 6.34 percentage points on its development state pool, is worth pursuing. It is neither a ceiling on episode-level improvement nor an estimate of the gain a deployable selector will achieve. That pool was strongly enriched for high-uncertainty states; the high-uncertainty verifier ranking was only about 0.524. Recompute candidate-budget curves on the same eligible state cohort at every budget, since changing coverage can confound the curve.

**The literature changes the novelty bar.**

| Work | Consequence for this project |
|---|---|
| [V-GPS, CoRL 2024](https://arxiv.org/abs/2410.13816) | Reranking actions from a frozen generalist policy using offline values is established. This should be a baseline. |
| [RL Token, April 2026](https://arxiv.org/html/2604.23073v1) | A compact representation of contextualized VLA features can support lightweight RL. Its RL execution chunk can be shorter than the VLA's generated horizon. |
| [Decoupled Q-Chunking, ICLR 2026](https://openreview.net/pdf?id=aqGNdZQL9l) | Long value propagation and short policy execution can be separated, but the semantics of closed-loop data and open-loop chunks require care. |
| [QGF, June 2026](https://arxiv.org/html/2606.11087v1) | Clean-estimate Q-gradient guidance is already a strong, simple test-time baseline. Its experiments also illustrate that increasing predicted Q can exploit critic error. |
| [QPILOTS, June 2026](https://arxiv.org/html/2606.14801v1) | Both endpoint-estimate guidance and posterior-sample guidance already have close precedents, including frozen pi0.5 experiments. |
| [Flow Map Q-Guidance / QGBS, May 2026](https://arxiv.org/html/2605.12416v1) | Re-noising, critic-dependent trust regions, and Q-guided beam search are already explored. Their combination alone is not a strong novelty claim. |
| [Q-VGM v3, August 24, 2026](https://arxiv.org/html/2606.08015v3) | The current method uses offline IQL, online TD, and critic-derived targets to train the action expert. It is relevant to critic design, but differs from this project's frozen-policy scope. Pin the version; June's Cal-QL recipe is not the current specification. |
| [Q-Planning, August 2026](https://arxiv.org/html/2608.21204v1) | Frozen-policy proposals, Q-weighted aggregation, and critic-only online improvement are very close prior work. The local implementation is an adaptation, not a faithful reproduction of its independent DinoV2/T5 encoding architecture. |
| [SVDD](https://arxiv.org/html/2408.08252) and [Feynman–Kac steering](https://arxiv.org/abs/2501.06848) | Intermediate search is about expected terminal utility under a specified generative process. Scoring a clean estimate is an approximation to that object. |
| [Progressive Seed Pruning, July 2026](https://arxiv.org/abs/2607.21591) | Early elimination of seeds is a particularly relevant alternative to adding stochastic branches to a deterministic flow. Its image-generation evidence motivates a robotics experiment, not a transfer guarantee. |

The strongest prospective contribution is **empirically validated allocation of inference compute under distribution shift**, with an explanation of which decisions benefit and why. Neither uncertainty thresholding alone nor attaching Q-guidance to pi0.5 is sufficient.

**Three modeling contracts should precede additional search machinery.**

First, define the intervention. At a decision boundary, write u for the next ten actions and h for the available observation history, instruction, proprioception, and remaining episode budget. The appropriate reference-policy value is:

q10(h,u) = expected return after executing u, then following a specified reference policy.

The generated positions 10–49 are discarded at replanning. For identical executed prefixes and identical continuation rules, changing only the unused suffix does not change the intervention. A deployment scorer should therefore be suffix-invariant by construction.

The local Q10 windows align action, reward, next state, and discount over the executed ten-step interval. Q50 windows concatenate fifty actual actions across five replans, while the contemplated inference input is one generated fifty-action plan. This changes both the action distribution and the relationship to future feedback. Even in a deterministic simulator, it does not make Q50 the value of executing ten actions and replanning. With partial observations or stochastic dynamics, additional conditioning issues arise. These are reasons to make Q10 primary, not a claim that every long-horizon critic is invalid. The [DQC analysis](https://openreview.net/pdf?id=aqGNdZQL9l) is directly relevant.

If Q10 later suffers from slow value propagation, investigate an explicitly derived long-to-short auxiliary target or multistep reference-policy evaluation. Do not interpret a raw Q50/Q10 comparison as isolating only the desirability of longer horizons.

Second, identify the representation. In [sampler.py](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/sampler.py:179), the code obtains prefix_embs from embed_prefix, passes those embeddings through the language model, and stores the original prefix_embs. They are not the contextualized final-layer token representations described in [RLT](https://arxiv.org/html/2604.23073v1).

The existing embeddings may be useful. But asking a modest critic dataset to learn the missing multimodal integration is a different problem from reading out already integrated VLA representations. Before a larger architecture sweep, compare the existing features against contextualized prefix features from the frozen backbone on one fixed subset with identical labels, splits, action heads, and training budgets. Where saved input embeddings and masks suffice, regenerate features without new environment interaction. Freeze the backbone. Use pooled contextualized features as the simplest new baseline; add reconstruction-trained compression only if justified. Reconstruction accuracy alone does not establish action-value sufficiency.

Third, name the continuation policy and objective. The local [target calculation](/Users/arjunsharma/development/cs159-sp26/pnp-vla/pnp/qplanning_critic/train.py:77) uses the next dataset action. This is a SARSA-style behavior-continuation backup. It can support reference-policy evaluation on suitable data; it does not automatically become current-planner evaluation when a replay buffer mixes policies.

The claim that a behavior-cloned policy imitates demonstrations does not establish equality between its action law and the demonstration law. Consequently, I would not adopt Q-Planning's blanket unbiasedness explanation for dataset-next-action targets as a theorem about this project. After policy changes, either keep evaluating a clearly specified fixed continuation, or explicitly target the new planner and collect the corresponding coverage.

Also distinguish success probability from discounted return. With per-environment-step gamma=0.99, success one hundred steps away receives roughly 0.366 of immediate success reward. For a success-rate objective, start with finite-budget Monte Carlo success prediction, conditioning on time remaining, alongside the existing discounted TD baseline. A finite-horizon undiscounted formulation is possible; it needs explicit time and terminal semantics rather than a silent gamma change.

Audit terminal-dependent action masks before interpreting offline metrics. The current dataset shortens windows at observed termination, and the critic sees action validity. A mask reflecting future success is unavailable when scoring a fresh proposal. This is a potential shortcut, not a measured failure. Separate target-validity masks from information allowed into the scorer; use pre-decision proposed prefixes where available, and assess calibration on full-validity windows separately.

**The first research bet: an intervention-validated Q10 selector.**

My highest-confidence bet is a better specified and better supervised selector, initially using ordinary completed policy candidates.

Use existing rollout data for broad value learning. Complete one fixed Q10 training run as a baseline, but judge it through held-out within-state choices. Global failure AUC, successful-versus-failed trajectory separation, and Bellman error are insufficient: a state-only predictor can perform well on these while providing no ranking signal.

Construct a modest, balanced development panel of replayable states. Include routine motion, contact transitions, recoverable mistakes, and apparently unrecoverable states. Include low, middle, and high uncertainty states; do not let the original gate decide the entire training distribution.

At each state, produce the default plus seven fresh candidates. Execute each ten-action prefix and then follow the same frozen continuation policy with common random numbers. Multiple continuation seeds on a smaller subset establish how much a single branch outcome varies with the continuation. Report the eight candidates as one correlated state group.

Train and compare a small number of models: the existing Q10 baseline; a simple contextualized-feature action critic; and action-only/state-only controls. A privileged-state critic on development data is a diagnostic ceiling if the required simulator observations are available, never part of the deployable claim. If the privileged critic succeeds and visual critics fail, representation is implicated. If neither works despite useful branch headroom, investigate data size, labels, and optimization.

Use Monte Carlo or proper binary prediction losses for probability estimation. Paired ranking supervision can be a controlled addition, but do not let a ranking loss masquerade as calibrated probability estimation. If failure-enriched collection changes outcome frequencies, account for the sampling scheme before claiming deployment calibration.

Primary diagnostic: expected selected outcome minus default outcome on untouched state groups, with rescue and spoil probabilities reported separately. Secondary diagnostics: gain versus random selection, same-state ordering, action-only control gap, and gain as candidate count grows.

The first deployment operator should select an actual completed candidate and retain the default as an option. Compare argmax-Q and tempered categorical selection. A Q-weighted average of conflicting grasp or motion modes can create an action no candidate proposed; it deserves a separate baseline, not an assumption of policy support.

An ensemble can provide useful disagreement features. Its standard deviation is not automatically a calibrated error bound. Likewise, the local HL-Gauss categorical head smooths scalar regression targets; its entropy is not automatically epistemic uncertainty or a learned distribution of actual returns.

**The central scientific bet: predict benefit from computation.**

A failure detector estimates P(failure | h). The gate needs something different:

G(h) = E[return with the specified search intervention - return with the default | information available before search] - lambda × extra compute.

These quantities can disagree. A confident policy can be wrong but recoverable. A highly uncertain state can have no successful candidates. The same uncertainty can describe harmless trajectory diversity or a consequential gripper decision.

Start with an always-search selector that works. Then collect matched default/search outcomes and fit a deliberately simple benefit predictor. Candidate gate features are P&P U10, reference value, execution-relevant action spread, and inexpensive critic disagreement. Compare it against a random gate with the same search frequency, an uncertainty-only gate, and a low-value gate. Account for the cost of every gate feature.

Distinguish two decisions. Before generating extra candidates, decide whether to buy search. After candidates exist, decide whether their evidence warrants replacing the default. A post-search Q margin cannot retroactively justify the compute already spent.

The outcome is a success-versus-latency frontier. The intended claim is that the gate captures more realized improvement per unit compute than simple schedules. Failure AUC is not the acceptance criterion.

P&P should be included only if it adds predictive value beyond cheaper features. The project retains its scientific question even if P&P is not selected.

**The second research bet: spend the budget across seeds and solver accuracy.**

The three-step pilot suggests that numerical refinement may have diminishing value on this benchmark. Test whether additional compute is better spent on another initial seed than on finishing every seed with ten integration steps.

The simplest comparison is single three-step, single ten-step, and several completed three-step candidates ranked by the same critic. Profile vision encoding, language-model prefill, action denoising, P&P queries, critic scoring, and data transfers separately. Shared encoding and batching matter; summed neural evaluations alone are not latency.

Only after cheap candidates are useful should the project try progressive elimination. For a fixed ten-step trajectory grid, eight seeds evaluated through step two, four survivors through step five, and two survivors through step ten cost 8×2 + 4×3 + 2×5 = 38 candidate-step evaluations, versus 40 for four fully completed seeds. This is illustrative accounting, not a recommended production schedule or a latency guarantee.

Before adopting it, complete every seed in a development pilot and measure how often early pruning discards the eventual best candidate and the eventually successful candidate. Preserve actual trajectory continuation: restarting the same noise with a different solver grid defines a different path.

Do not assume a clean-action critic remains calibrated on early endpoint estimates. Use completed short-solver candidates first; if needed, learn a separate denoising-stage score that predicts the value of the exact completed candidate on a fixed grid. For a deterministic ODE, that future endpoint is determined by the current state and integration rule. P&P perturbations instead assess robustness or alternate proposals; they are not automatically conditional draws from that original trajectory.

The [PSP paper](https://arxiv.org/abs/2607.21591) supplies the relevant compute-allocation precedent. A robotics contribution would require measured survival of good actions and better closed-loop performance at matched latency.

**What P&P can and cannot justify mathematically.**

For a specified stochastic denoising process, reward-tilted sampling uses an intermediate soft value of the form V_t(x) = tau log E[exp(q10(h,U)/tau) | X_t=x]. This differs from scoring an average clean estimate. [SVDD](https://arxiv.org/html/2408.08252) and [FK steering](https://arxiv.org/abs/2501.06848) provide the relevant framework.

Two consequences matter here. Averaging plausible actions before scoring need not preserve their value. And arbitrary re-noising plus deterministic denoising does not inherit a posterior-sampling guarantee. The implemented P&P iterations are also sequentially dependent; their spread is not an iid posterior uncertainty estimate.

[QPILOTS](https://arxiv.org/html/2606.14801v1) explicitly investigates point-estimate versus posterior-sample guidance. A proposal to average critic gradients over P&P samples must therefore establish what those samples mean and beat that close baseline; it cannot rely on the resemblance to a posterior expectation.

Use QGF as the first gradient baseline if the selector passes validation. Test local gradient quality with simulator outcomes: compare small positive-gradient perturbations, negative-gradient perturbations, and equal-displacement random perturbations, followed by the same chosen reconstruction and continuation rule. Increased predicted Q alone is not evidence that the direction improves control. Gate gradient work on this test.

**One elegant optional reference algorithm.**

If local search is pursued, there is a clean gradient-free control in initial-noise space. Let F_h(z) be a fixed deterministic VLA sampler and target a noise distribution proportional to Normal(z;0,I) × exp(beta q10(h,F_h(z)[:10])).

Propose z' = sqrt(1-rho²) z + rho epsilon, epsilon ~ Normal(0,I), and accept with min(1, exp(beta [q(z')-q(z)])). The proposal is reversible with respect to the Gaussian reference, so the acceptance rule has the stated invariant target. It avoids policy likelihoods, action-space averaging, and differentiating through the VLA. This follows the [pCN construction](https://arxiv.org/abs/1202.0709); reward-aware noise-space MCMC also has diffusion precedents such as [Psi-Sampler](https://arxiv.org/abs/2506.01320).

This is a reference experiment, not the main proposal. A few iterations need not mix, every proposal costs a complete sampler pass, and an exact invariant law for an inaccurate critic does not guarantee robot success. Returning the best visited action or appending a default-fallback rule changes the sampling claim. Its practical question is whether correlated seed proposals beat independent candidates at the same compute.

**A bounded experimental program.**

| Order | Experiment | Decision it resolves |
|---|---|---|
| 1 | Audit feature location, mask information, objective, and continuation semantics; profile three-step versus ten-step inference. | Whether the present learning target and efficiency premise are sound. |
| 2 | Fixed Q10 training plus contextualized-feature comparison; evaluate on a balanced paired-branch panel. | Whether useful same-state action ranking is learnable with available resources. |
| 3 | One frozen selector in closed-loop development rollouts; compare stock, cheap single sample, and cheap best-of-N. | Whether one-step branch improvements survive repeated deployment. |
| 4 | Random, uncertainty, low-value, and learned-benefit gates at matched total cost. | Whether selective computation provides a contribution beyond always-search. |
| 5 | Progressive elimination; one gradient or local-noise-search baseline if earlier diagnostics support it. | Whether more complex search improves the frontier. |
| 6 | Freeze method, thresholds, budget, and primary contrast; run fresh confirmation. | Whether the selected result generalizes without development selection bias. |

Use whole-identity separation and include all related rollouts/branches in the same partition. State-level bootstraps are suitable for branch diagnostics; episode comparisons must use episode pairing. For generalization across tasks, supplement identity intervals with task-level summaries or hierarchical uncertainty. Do not treat windows or candidate pairs as independent sample counts.

The 220 identities already used repeatedly are development data for future method selection. Position perturbations may be excluded from critic training yet still be exposed through repeated tuning and analysis. Training exclusion alone does not preserve confirmatory status. Preserve the sealed verifier cohort for its stated protocol; a new adaptive closed-loop method also needs an appropriate fresh end-to-end cohort.

Predefine a practically worthwhile gain or acceptable noninferiority margin before selecting sample size. Under an illustrative independent-pair approximation with 10% discordant outcomes, detecting a three-percentage-point improvement with 80% power at two-sided 5% significance takes on the order of 900 paired episodes; task clustering can increase this. The existing pilot is valuable for estimating discordance, not for promising a precise future sample size.

**Stop conditions and research judgment.**

Stop increasing search complexity if the selector cannot improve held-out within-state decisions over the default and meaningful controls. Stop spending on P&P if it adds no benefit-prediction value after its cost is counted. Stop early pruning if it loses good candidates faster than its compute savings compensate. Stop gradient guidance if the simulator does not validate its local improvement direction. Do not launch a full online RL program simply because a recent paper reports large gains at a different data scale.

My ranking is: first, execution-aligned critic learning with contextualized features and counterfactual validation; second, benefit-based compute allocation; third, coarse candidate sampling and selective completion; fourth, bounded gradient or local-noise refinement. Adaptive action horizons and full SMC remain follow-up projects.

A defensible prospective paper would establish that additional inference compute helps when there are valuable alternatives, a selector that can recognize them, and a gate that can predict the benefit. The decisive figure would show stock, cheap sampling, always-search, and selectively allocated search on the same success–latency axes, with uncertainty and rescue/spoil accounting. Every component should earn its place through that experiment.
