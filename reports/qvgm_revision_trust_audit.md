# Q-VGM revision and methodological trust audit

Date of audit: 2026-08-25  
Paper: arXiv:2606.08015  
Versions compared: [v1 (2026-06-06)](https://arxiv.org/abs/2606.08015v1), [v2 (2026-07-19)](https://arxiv.org/abs/2606.08015v2), [v3 (2026-08-24)](https://arxiv.org/abs/2606.08015v3)

## Executive conclusion

Your concern is justified. This is not a normal polishing pass. Between v1 and v3, Q-VGM kept its central policy-extraction idea but replaced the critic methodology twice, replaced nearly the entire empirical program, changed the RL-token training rule twice, and removed implementation details that would be necessary for faithful reproduction. The official TeX diff from v1 to v3 is 569 insertions and 550 deletions in a roughly 750-line paper.

The key distinction is:

- The **core Q-VGM idea is fairly stable**: query a critic near a frozen/reference policy's clean-action estimate, improve that action by iterative Q-gradient ascent with keep-best selection, convert the displacement to a late-denoising velocity target, and train locally without BPTT.
- The **Q-function recipe is not stable**: v1 Cal-QL chunk critic -> v2 stepwise IQL critic -> v3 scalar chunk IQL offline plus standard TD online.
- The **RL-token concept is stable, but its training rule and evidence are not**: frozen in v1, jointly optimized with the critic in v2, frozen again in v3. The component ablations that supported it in v1/v2 are absent from v3.
- The **paper is not currently sufficient to reproduce the reported method**. There is no official code link in the paper or arXiv source, and v3 removes the appendix that gave the critic widths, token dimension, autoencoder shape, and expectile value used in earlier versions. It also omits most optimization and data-mixing hyperparameters.

My trust verdict is therefore:

> Treat Q-VGM as a promising research hypothesis, not as a trustworthy training recipe. Do not keep using the June critic merely because it appeared in the first public paper. If you continue, pin your own specification, migrate intentionally to the v3 critic semantics, and independently validate the RL token and action-gradient pathway.

I would not infer misconduct or fabricated results from these changes alone. The more defensible diagnosis is a rapidly moving preprint with poor version discipline, weak reproducibility, and empirical claims that should not be carried across revisions.

## Revision timeline at a glance

| Area | v1: June 6 | v2: July 19 | v3: August 24 | Assessment |
|---|---|---|---|---|
| Scope | Fully offline flow-VLA fine-tuning | Fully offline off-policy RL | Full offline-to-online framework | Major expansion |
| Second author | Jiayu Sun | Yitian Liu | Yitian Liu | Authorship changed with the methodological rewrite |
| Critic objective | Ensemble Cal-QL | Two-head stepwise IQL | Scalar chunk IQL offline; TD online | Method replaced twice |
| Critic output | Scalar chunk Q | H position-specific Q values, summed for guidance | Scalar chunk Q | v2 design abandoned |
| Offline backup | Dataset next chunk | In-support stepwise value heads | In-support scalar value head | v1 no longer authoritative |
| Online backup | None | None | Current-policy next chunk with clipped target Q | New in v3 |
| RL token | Pretrained and frozen | Pretrained, then jointly trained with critic + reconstruction loss | Pretrained and frozen | v2 reversal; v3 aligns with original RLT |
| Actor critic aggregation | Ensemble mean | Minimum of two heads for guidance | Target minimum; online actor explicitly uses mean; offline actor aggregation unspecified | Still ambiguous |
| Guidance schedule | Smooth gate over all steps | Hard last-M mask, M=5 | Hard last-M mask, M=5 | Substantive v1 -> v2 change |
| Gradient safeguards | Gradient clipping + action projection + keep-best | Gradient clipping + keep-best | Keep-best only in equations; clipping/projection no longer specified | Safety details removed |
| Simulation | LIBERO 4 suites + RoboTwin 2.0 | LIBERO 4 suites | LIBERO 3 suites + online comparison | Experimental program replaced |
| Real robot | Two 7-DoF single-arm tasks | Three quantified bimanual tasks + qualitative plug insertion | Three bimanual tasks with a new baseline table | Tasks/platform/results replaced |
| Component ablations | RL token, action injection, ensemble, and policy safeguards | Same categories, new values | Only critic-objective ablation | Evidence for current architecture removed |
| Implementation appendix | Present | Present, substantially rewritten | Removed | Reproducibility regressed |

## What changed in the Q-function

### v1: Cal-QL with a dataset-action SARSA-style backup

The June paper trains an ensemble Cal-QL critic on chunk transitions. It uses the dataset's next action chunk in the Bellman target and freezes the critic before policy extraction. The ensemble mean is used for action ranking and Q-gradients.

The written v1 target is effectively:

```text
R_t = sum_{j=0}^{H-1} r_{t+j}
y   = R_t + gamma * Q_target(s_{t+H}, A_{t+H}^{dataset})
```

This is a problematic specification for a chunk-level transition:

- The in-chunk return is not discounted.
- The bootstrap is multiplied by `gamma`, not `gamma^H`.
- The displayed target has no terminal mask.
- The prose calls the method Cal-QL, but the particular dataset-next-action backup is SARSA-like; the conservative regularizer is then layered on top.
- Important Cal-QL details are absent: how candidate actions for the log-sum-exp are drawn, conservative-loss weight, return calibration implementation, discount, target update, batch construction, and ensemble size.

These are not small omissions when the actor consumes `dQ/dA`. A critic can have reasonable scalar predictions while producing unusable or adversarial local gradients.

### v2: stepwise IQL and joint representation learning

July discards the Cal-QL method and introduces H Q-values per chunk, one for each action position. A separate H-output value head supplies an IQL backup. Interior positions bootstrap to the next value position in the same chunk; the final position bootstraps to position zero of the next chunk. The minimum of two Q-heads is used for target construction and for Q-VGM guidance.

This version also jointly optimizes the RL-token encoder and critic while keeping a reconstruction loss active. It is a different representation-learning algorithm, not a clarification of v1.

The v2 appendix is more careful about terminal and partial-chunk masking and gives an IQL expectile of 0.8. It is internally more detailed than v1, but this entire stepwise design is abandoned in v3.

### v3: scalar chunk IQL offline, standard TD online

August returns to a scalar chunk Q but uses the mathematically conventional chunk return and discount:

```text
r_chunk = sum_{i=0}^{H-1} gamma^i r_{t+i}

offline: y = r_chunk + gamma^H (1-d) V(s')
online:  y = r_chunk + gamma^H (1-d) min_i Q_target_i(s', A')
         A' is sampled from the current policy without gradients
```

This is the critic definition I would regard as current. It fixes the most obvious chunk-backup defects in v1 and avoids v1's conservative penalty during offline learning. The current ablation on LIBERO-Spatial reports IQL 88.8, CQL 87.8, SARSA 87.0, and Cal-QL 84.6. The authors' new explanation is that conservative penalties flatten Q near the data and weaken `dQ/dA`.

That explanation is plausible and directly undercuts the June recommendation. If you are still using Cal-QL because v1 called it a core component, you are implementing a method the current paper now reports as its worst critic-objective variant.

However, v3 still leaves critical choices unspecified:

- IQL expectile (0.8 in v2, not stated in v3)
- number of Q heads
- exact Q/V network depth and width
- optimizer, learning rate, batch size, update-to-data ratio, target EMA, and training duration
- discount and chunk length in each experiment
- replay sampling and demonstration/rollout mixture
- action normalization details
- reward labeling and terminal conventions
- offline actor's ensemble aggregation
- Q-gradient ascent step size, number of steps, and whether gradients or actions are clipped
- exploration temperature and action-noise magnitude
- online cycle size, critic epochs per cycle, actor learning rate, and reference-policy EMA

### Ensemble aggregation is not version-stable

This matters because averaging and taking a minimum can yield different gradient directions:

- v1: ensemble **mean** for ranking and gradients.
- v2: elementwise **minimum of two Q-heads** for both value construction and Q-VGM.
- v3: clipped minimum for TD targets, but the online actor explicitly defines the guidance critic as the **mean** of the live Q-heads. The offline algorithm merely says “Q” and does not define the aggregation.

Do not silently inherit one of these. Make it a named configuration and ablate mean versus minimum versus disagreement-gated guidance.

## What changed in the RL token

### Stable part

All three versions use the same high-level idea: compress the frozen VLA prefix into one latent token, concatenate it with projected proprioception, and use that as critic state. Per-layer action injection is meant to stop the much larger state vector from overwhelming the low-dimensional action chunk.

The original Physical Intelligence RLT method trains an encoder-decoder bottleneck to autoregressively reconstruct frozen VLA prefix embeddings, then freezes both the VLA and RL-token module before RL. Its RL token is the encoder output at a learned special-token position. The v3 freeze rule therefore matches the cited primary RLT method more closely than v2 does.

### Unstable part

- v1: a 2-layer, 2048-dimensional, 8-head encoder-decoder is pretrained by MSE reconstruction, reaches greater than 0.95 held-out cosine similarity, and is frozen for critic and policy training.
- v2: the same pretraining initializes the encoder, but the encoder is then jointly optimized with IQL while reconstruction remains as a regularizer.
- v3: the encoder is pretrained and frozen again, but the token dimension, transformer depth/heads, loss, pretraining data, reconstruction quality, and critic-state dimension disappear from the paper.

The v2 joint-training experiment is not presented as an ablation against freezing, and v3 gives no explanation for reverting it. Therefore, the paper provides no direct evidence for which choice is better in Q-VGM.

There is also a reproducibility gap between “following RLT” and the Q-VGM implementation. The cited RLT paper explains the learned special token and autoregressive decoder construction; Q-VGM's text only says “autoencoder” and does not fully specify token selection, decoder masking, prefix-token subset, language handling, or whether reconstruction is autoregressive. If your RL-token implementation was built only from Q-VGM v1, compare it against the primary RLT construction rather than assuming these are interchangeable.

### The old architecture ablations cannot validate the current critic

The v1 component ablation reports:

- replacing RLT with a ResNet: 92.5 -> 82.5
- removing per-layer action injection: 92.5 -> 87.5
- replacing the ensemble with one head: 92.5 -> 89.5

The v2 rerun reports smaller but similar drops:

- RL token -> ResNet: 92.5 -> 87.4
- no per-layer injection: 92.5 -> 88.2
- one head: 92.5 -> 90.1

v3 deletes these ablations and instead tests critic objectives only. Because v1 used Cal-QL and v2 used stepwise IQL, neither set establishes that the same architecture components are necessary under v3's scalar chunk-IQL critic. The direction of the evidence is encouraging, but the numbers must not be cited as validation of the current system.

## Q-VGM policy-extraction changes

The core mechanism survives, but several implementation-sensitive details moved.

### Time convention

v1 uses noise at `t=1` and clean action at `t=0`; v2/v3 reverse this to noise at `tau=0` and clean action at `tau=1`. The apparent sign changes in the look-forward action and residual velocity are largely a notational conversion. Mixing equations across versions would still create a real sign bug.

### Where alignment is applied

- v1 uses a smooth gate `s(t)=(1-t)^p` over the trajectory.
- v2/v3 use a hard mask on the last `M=5` denoising steps.

That is an algorithm change. v1's all-step ablation does not directly validate the precise hard-mask choice adopted later.

### Gradient and action safeguards

- v1 projects improved actions to the valid action range and clips the Q-gradient magnitude.
- v2 retains gradient clipping but removes the written action projection.
- v3 removes both clipping and projection from the method equations, leaving only keep-best selection.

Keep-best only protects according to the same learned critic that generated the step. It does not protect against a critic that is wrong off-support. For robotics, I would keep explicit normalized-action projection, gradient-norm clipping, and ensemble-disagreement monitoring unless an ablation shows they are harmful.

### Reference policy

The offline algorithm consistently uses a fixed base/reference policy for look-forward estimates. In v3's online algorithm, the reference is initialized from the offline actor and then updated by EMA. Elsewhere v3 still calls the base velocity “frozen.” This is resolvable if “frozen” means no gradient during an actor update, but the paper should say so; a permanently frozen reference and a moving EMA reference are not the same online algorithm.

## The experimental paper was effectively replaced

### Simulation

v1 reports four LIBERO suites, RoboTwin 2.0, and an offline-only method. Its LIBERO collection budget is 300 rollout episodes per suite. Initial average success is 75.0 and Q-VGM is 92.5.

v2 keeps four LIBERO suites, removes RoboTwin, and says it reuses 500 SFT evaluation episodes per suite plus benchmark demonstrations. Initial average success changes to 79.0 while Q-VGM remains 92.5. It claims roughly 400x fewer episodes than PPO based on 500 rollout episodes for Spatial.

v3 drops LIBERO-Long, changes all three remaining suite results, uses 150 rollout episodes **per task** plus demonstrations, and adds offline-to-online training. Since each included suite has ten tasks, this is 1,500 rollout episodes per suite: 5x the v1 per-suite budget and 3x the v2 per-suite budget. Initial average success is now 86.9, offline Q-VGM is 93.0, and offline-to-online Q-VGM is 99.7.

The headline sample-efficiency comparison also changes from about 400x fewer episodes in v2 to about 6x fewer in v3. These claims answer different questions under different protocols; they are not successive refinements of one experiment.

The 99.7 versus PPO's 99.5 final success should be read as a match, not meaningful superiority. The paper reports no training-seed variance or confidence interval, and the 0.2-point gap is tiny relative to unreported training variation.

### Real robot

v1 uses two single-arm tabletop tasks and reports 40.0 -> 67.5 average success.

v2 switches to a bimanual platform. It says “four tasks,” quantitatively lists only three (20/20, 15/20, 19/20), and shows plug insertion qualitatively without a success rate.

v3 removes plug insertion, reports three tasks with different outcomes (19/20, 20/20, 20/20), adds SFT and offline baseline results, and headlines 66.7 -> 98.3 average success.

The v3 real-robot table is more informative than v2, but 20 in-distribution trials per task, no uncertainty, and an unexplained change in outcomes are not enough to treat 98.3% as a stable estimate.

## Editorial and reproducibility warning signs

These do not prove scientific misconduct, but they lower confidence in version control and review readiness:

1. The arXiv v3 metadata title says “Q-Value-Gradient Matching,” while the v3 PDF title says “Q-Guided Value-Gradient Matching.” v2 has the same mismatch between title and expansion in the abstract.
2. The current arXiv comments still say 13 pages, 3 figures, and 4 tables. The actual v3 PDF has 12 pages, 2 numbered figures, and 3 tables. The metadata appears inherited from v1.
3. v2 claims four real-robot tasks but tabulates three and gives only a qualitative plug-insertion example.
4. The second author changes between v1 and v2 at the same time as the paper's methodology and experiments are extensively rewritten, without a revision note.
5. The v3 architecture/training appendix is removed even though the method becomes broader.
6. No optimizer-level reproducibility details, random seeds, uncertainty estimates, raw curves, or official code repository are supplied.
7. The earlier “approximately 3.9M parameters per Q-head” does not obviously match the stated two-hidden-layer dimensions. With a 2,304-dimensional state, a 35-dimensional action, and hidden widths 1,024 then 512, a straightforward per-layer-action-injection MLP is roughly 2.94M parameters, not 3.9M. An unmentioned extra layer could explain it, which is exactly why code or a precise layer table is needed.

## Recommendation for your implementation

### Stop treating “Q-VGM critic” as one configuration

Name the variants explicitly in configs and experiment logs, for example:

```text
qvgm_v1_calql_chunk_frozen_rlt_meanq
qvgm_v2_iql_stepwise_joint_rlt_minq
qvgm_v3_iql_chunk_frozen_rlt_<aggregation>
qvgm_v3_online_td_chunk_frozen_rlt_meanq
```

This prevents accidental hybrids, especially mixing v1's backup with v3's actor or v2's stepwise head with v3's scalar equations.

### If you want to follow the current paper

Use this as the minimum current specification:

1. Freeze the VLA prefix backbone.
2. Pretrain the RL-token encoder-decoder on prefix reconstruction, then freeze it. Follow the primary RLT special-token/autoregressive reconstruction design unless Q-VGM code becomes available.
3. Feed `LayerNorm([z_rl ; projected proprioception])` to the critic. Treat the 2,048/2,304 dimensions and 1,024 -> 512 MLP as legacy hints, not v3-specified facts.
4. Predict a scalar Q for the normalized action chunk, not v2's H stepwise outputs.
5. Re-inject the flattened action chunk at every hidden layer.
6. Use two Q-heads unless your own ablation supports another choice; define guidance aggregation separately from TD-target aggregation.
7. Offline, train an action-free scalar V by expectile regression and use `r_chunk + gamma^H (1-d) V(s')`.
8. Online, sample `A'` from the current policy without gradients and use the minimum target Q in the TD backup.
9. Query gradients at the reference-policy look-forward action, keep the best ascent iterate, and train only on the last five denoising steps.
10. Explicitly choose action projection, Q-gradient clipping, reference EMA, and ensemble-disagreement behavior. v3 does not settle them.

### Validation gates before trusting the critic gradient

Scalar Bellman loss is not enough. At minimum, log and test:

- finite-difference agreement with autograd `dQ/dA`
- local improvement rate: fraction of projected gradient steps that improve held-out Monte Carlo return, not merely predicted Q
- keep-best fallback rate and chosen ascent-step histogram
- gradient norm by denoising time and distance from dataset actions
- twin-head value and gradient disagreement
- action sensitivity versus a state-only or shuffled-action critic
- calibration and ranking accuracy on held-out successful, failed, and perturbed chunks
- frozen versus jointly trained RL token
- input-only versus per-layer action injection
- mean-Q versus min-Q guidance
- IQL versus Cal-QL on your own buffer
- at least three training seeds and confidence intervals on rollout success

Add a hard stop if Q increases while held-out return, success classification, or ensemble agreement deteriorates. That is the failure mode Q-VGM is most exposed to.

## Bottom line

- **Can you trust the June critic recipe?** No. The authors themselves replaced it, and v3's ablation now disfavors it.
- **Can you trust the current critic recipe as complete?** No. Its high-level backup is more coherent, but the implementation is under-specified.
- **Can you keep the RL-token idea?** Yes, provisionally. The freeze-after-reconstruction design is consistent with the original RLT source, but Q-VGM has not shown that its current results depend on the exact architecture you are using.
- **Can you keep per-layer action injection?** It is a sensible and version-stable hypothesis, but validate it locally; the supporting ablation was removed from the current experimental setup.
- **Should you abandon Q-VGM entirely?** Not necessarily. Preserve the stable policy-extraction idea, migrate away from the v1 critic, and make your own ablations the source of truth.

## Decision addendum: a stable Q baseline for this repository

### Closest modern VLA/LIBERO analogue

The closest methodological match is **Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning** (Giridhar et al., 2026). It combines a frozen flow-matching VLA, a separately parameterized Q-function over action chunks, LIBERO evaluation, and inference-time candidate selection. It also releases runnable code and configs. This makes it a much better implementation reference than Q-VGM, but not an independently established result: its arXiv v1 was posted on 21 August 2026, only days before this audit.

Use the papers as a stack rather than trusting any one paper:

| Role | Primary reference | What to borrow |
|---|---|---|
| Modern VLA/LIBERO integration | [Q-Planning](https://arxiv.org/abs/2608.21204) and its [released code](https://github.com/varungiridhar/qplanning-code) | Frozen BC policy, chunk-Q, candidate scoring, top-K Q-weighted averaging, held-out deployment seeds |
| Bellman semantics | [Q-Chunking](https://arxiv.org/abs/2507.07969) | Treat the actually executed K-step sequence as the super-action; use the discounted K-step reward and `gamma**K` bootstrap |
| Gradient-guidance extension | [QGF](https://arxiv.org/abs/2606.11087) and its [code](https://github.com/zhouzypaul/qgf) | Query Q on a predicted clean action and add a bounded test-time gradient only after the critic is validated as a ranker |
| Offline-RL control | [IQL](https://arxiv.org/abs/2110.06169) | Keep as an ablation if the buffer is broad and off-policy; do not make it part of the first end-to-end baseline |

### Recommended baseline: `pcp_qrank_v0`

The first baseline should answer only this question: **given several chunks sampled by the frozen VLA, can the critic rank which executed prefix is most likely to succeed?** It should not yet train the VLA, use Q gradients, or claim that its learned latent tokens reproduce Physical Intelligence's RL-token method.

1. **Freeze the VLA.** Continue consuming its detached prefix embeddings and proprioception. The critic can train its own small state pooler, but no critic update may reach the VLA.
2. **Make the Bellman action equal the execution interval.** This repository replans every 10 actions but currently conditions Q on the full 50-action proposal. For a literal semi-MDP chunk-Q, use the first `K=10` actions as `A_t`, the rewards from those same ten environment steps, and a `gamma**K` bootstrap. The unexecuted 40-action suffix is correlated policy metadata, not part of the action actually taken.
3. **Use the next recorded executed-prefix action for the target.** Start with the stable SARSA-style target used by Q-Planning: `y = R_K + gamma**K * Q_target(s_next, A_next)` and zero the bootstrap at termination. Do not sample, maximize, or differentiate through the VLA inside the training target.
4. **Replace scalar Huber/MSE regression with an HL-Gauss categorical return head.** For sparse binary success reward, use 101 bins over approximately `[0, 1]`, convert the Bellman scalar target to a Gaussian-smoothed categorical label, and optimize cross-entropy. Read scalar Q as the expected bin value. Q-Planning identifies this and chunking as its two stability devices.
5. **Keep EMA targets.** `gamma=0.99`, AdamW around `3e-4`, and target EMA rate `0.005` are reasonable initial values already shared by this repository and Q-Planning. Use a lower learning rate for any large pretrained encoder; the current small pooler does not require that split.
6. **Keep two heads only as a declared safety extension.** Train two independently initialized categorical heads and score with the minimum expected value. Also log both heads and their disagreement. Run a single-head replication as an ablation because Q-Planning itself does not establish that clipped double-Q is necessary here.
7. **No Cal-QL penalty in v0.** Candidate selection already restricts inference to VLA samples. The current broad uniform-action conservative penalty adds action-scale and OOD assumptions that are not needed to test whether chunk-Q ranking works. Keep current Cal-QL and IQL modes as named comparison arms.
8. **Use Q-weighted candidate selection.** Reuse the repository's candidate-scoring interface. Start with `N=16` for iteration speed, then reproduce Q-Planning's LIBERO setting (`N=64`, top `K_elite=16`, temperature 1, three denoising steps). Average the selected full proposals, but execute only the first ten actions before replanning. Compare weighted averaging with argmax, uniform candidate averaging, and the untouched VLA using identical candidates and seeds.

### What is already correct, and what should change

The data path already computes the discounted within-chunk reward, terminal masking, and `gamma**width` continuation correctly. The candidate-scoring adapter and its minimum-of-two-heads output are also useful starting interfaces.

The main recipe changes are:

- change the critic's action input from the 50-step generated proposal to the 10-step executed prefix;
- change the scalar Q head and Smooth-L1 TD loss to a categorical HL-Gauss head and cross-entropy;
- make the recorded-next-action TD objective the default baseline, without the Cal-QL candidate penalty;
- describe the trainable cross-attention tokens as **critic latent/query tokens**, not pretrained RL tokens. They have no reconstruction pretraining or freeze stage and therefore are not the RLT method claimed in Q-VGM;
- keep direct Q-gradient action correction disabled during baseline validation.

### Acceptance gates before inference-time experiments

Use the saved LIBERO simulator states to construct a counterfactual ranking test: at held-out decision boundaries, restore the same state, execute multiple frozen-VLA candidates, and compare predicted rank with realized success/return. This is more informative than Bellman validation loss.

Do not promote the critic unless, over at least three critic seeds:

- candidate pairwise ranking/AUC is consistently above chance on held-out tasks or initial states;
- Q-weighted selection beats the untouched VLA and uniform averaging under paired rollout seeds;
- Q spread does not collapse and twin disagreement does not grow on selected candidates;
- the selected candidates remain within the frozen VLA sample support;
- the improvement survives an untouched final test split that was not used for checkpoint selection.

Only after these gates pass should `corrected_action` be evaluated. For that phase, borrow QGF's clean-action look-forward principle, add explicit action bounds and a trust region around the VLA candidate, measure finite-difference gradient agreement, and require real counterfactual return improvement rather than predicted-Q improvement.

### Decision

Use **Q-Planning as the integration template, Q-Chunking as the Bellman contract, and QGF as the later gradient-guidance template**. Do not use Q-VGM as the authority for critic architecture or RL-token training. In this repository, the defensible starting point is a small, parameter-disjoint, categorical 10-step chunk ranker attached to the frozen VLA; Q-VGM-style gradients are a second-stage hypothesis.

## Source record

- Local v1 PDF SHA-256: `3d30002add8dc921caf8250876c0cc51d38ed838f6dd17b2339b30178fb277a9`
- Downloaded v2 PDF SHA-256: `587f8f878daaafd39c68fcac5d6e440886093e6cbee6d653ea2cc102d2cf755d`
- Downloaded v3 PDF SHA-256: `1746c8d5810dccbb1c3a8450ce8dd6bf1b4e965a56622cb2157321bdd36026cf`
- Official primary RL-token source: [Physical Intelligence, “Precise Manipulation with Efficient Online RL”](https://www.pi.website/research/rlt)
