# PnP-VLA project status and factual handoff

**Status date:** 2026-09-08 (America/New_York)  
**Repository:** `cs159-sp26`, package subdirectory `pnp-vla/`  
**Checked-out branch:** `main`  
**Checked-out commit:** `6169b50` (`Stream Q-planning windows from source cache`)  
**Remote state at preparation:** local `main` and `origin/main` both pointed to `6169b50`  
**Purpose:** factual project record and repository handoff for a future agent  

This document consolidates:

- repository code and Markdown documentation at commit `6169b50`;
- recent Git history through 2026-09-08;
- prior project conversations available in Codex;
- read-only Supabase metadata checks performed on 2026-09-08;
- matched rollout summaries computed with the repository's analysis code;
- the focused local test run performed after pulling `origin/main`.

This file records claims, configurations, results, and open implementation states. It does not
select a preferred research direction or prescribe an experiment.

---

## 1. Reading conventions and source hierarchy

### 1.1 Meaning of status terms

| Term | Meaning in this document |
|---|---|
| Implemented | Code exists in the checked-out repository. |
| Notebook exists | A committed `.ipynb` wrapper exists. This does not imply that it was run. |
| Completed remotely | Supabase contains the expected completed result rows. |
| Registered | A row exists in the relevant Supabase model or snapshot registry. |
| Output-cleared | The committed notebook contains zero saved output objects. |
| Historical | Describes an earlier implementation, experiment family, or conversation state. |
| Superseded | A later repository path or documented contract replaces it for new work. |
| Sealed | The confirmatory cohort is intentionally excluded from development access. |

### 1.2 Evidence precedence

When sources disagree, the following ordering identifies the current mechanical state:

1. Current source code at `6169b50`.
2. Current Supabase metadata and immutable registry entries.
3. Current generated notebooks and their imported package entry points.
4. Dated status documents.
5. Older plans and prior conversation summaries.

The older documents remain useful for experiment rationale and recorded results. They are not
automatically current implementation specifications.

### 1.3 Important documentation inconsistencies

- `qplanning_corrector_training_plan_v2.md` says that no trainer or notebooks are implemented.
  At `6169b50`, both the trainer and notebooks 64/65 exist. The plan's algorithmic contract is
  still reflected in the implementation, but that implementation-status sentence is stale.
- `README.md` describes the earlier verifier workflow and names an older confirmation notebook.
  `notebooks/README.md` contains the more current notebook inventory.
- The committed notebooks have cleared outputs. Result existence must be checked in Supabase or
  in dated status documents rather than inferred from notebook output cells.
- `VERIFIER_STATUS_DOSSIER.md`, `ORACLE_BRIEF.md`, `ANALYSIS_REFACTOR_PLAN.md`, and
  `reports/qvgm_revision_trust_audit.md` were untracked local files at handoff time. Their contents
  were used as factual source material, but they were not part of commit `6169b50`.

---

## 2. Project-level objective and experimental setting

### 2.1 System under study

The package runs a frozen pi0.5 flow-matching vision-language-action policy on LIBERO and
LIBERO-PRO. It adds test-time sampling, uncertainty measurement, candidate selection, and
action-correction mechanisms while logging rollout metadata and artifacts to Supabase.

The working project title recorded in `ORACLE_BRIEF.md` is:

> When Should a VLA Think Harder? Uncertainty-Gated Test-Time Search for Flow-Matching VLA
> Policies.

The project separates two functions:

1. **Gate:** an uncertainty signal indicates when additional test-time computation is invoked.
2. **Selector/corrector:** a mechanism chooses or constructs an action chunk at an invoked state.

The base VLA is frozen in the experiments documented here. The policy repository identifier in
the code is `lerobot/pi05_libero_finetuned`. The immutable PCP critic snapshot is pinned to
revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`.

### 2.2 Inference environment

- Simulation benchmarks: LIBERO and LIBERO-PRO.
- Real action dimensionality: 7.
  - indices 0–2: Cartesian position;
  - indices 3–5: axis-angle rotation;
  - index 6: gripper.
- Model-native action chunk length: 50.
- Current collection/replanning convention: execute 10 actions, then replan.
- Earlier data included open-loop execution of all 50 actions. New PCP-search collection was
  introduced specifically with `n_action_steps=10`.
- GPU inference and model training run in Colab notebooks.
- Notebooks are intended to be thin wrappers that clone/pull the repository, install
  `pnp-vla`, and call package code.
- Local analysis reads Supabase and does not require the simulator for metadata-only reports.

### 2.3 Episode limits in current configuration

`pnp/config.py` defines:

| Base suite | Maximum environment steps |
|---|---:|
| `libero_spatial` | 220 |
| `libero_object` | 280 |
| `libero_goal` | 300 |
| `libero_10` | 520 |
| `libero_90` | 400 |

Perturbed LIBERO-PRO suite names are resolved by longest matching base-suite prefix. The generic
fallback is 280 steps.

### 2.4 Current deployment operator recorded in the verifier documentation

The verifier documentation defines **Operator A** as endpoint best-of-N:

1. sample multiple completed action chunks at one decision state;
2. score each completed chunk;
3. execute the selected chunk's prefix;
4. replan after the configured execution horizon.

An earlier mid-denoising-trajectory search operator, called **Operator B**, was explicitly marked
out of scope in `ORACLE_BRIEF.md`. Some later refinement experiments do modify the denoising
process, but the verifier training target remains completed action chunks.

---

## 3. Core concepts and terminology

### 3.1 Flow-matching action generation

The pi0.5 action expert integrates a velocity field from noise toward a clean action. In the
current repository convention, the sampler begins near `s=1.0` (noise) and proceeds toward
`s=0.0` (clean). Older paper notation may use the reverse convention.

A one-step clean/endpoint estimate is formed from the current latent and velocity field. P&P
uses this estimate as the center for re-perturbation and repeated prediction.

### 3.2 Predict-and-Perturb uncertainty

The P&P probe repeatedly:

1. predicts a clean action estimate from a denoising state;
2. re-perturbs the estimate with fresh noise;
3. predicts again;
4. measures disagreement among the resulting clean-action estimates.

The logged uncertainty is a self-consistency measurement of the frozen policy. It is not a
policy likelihood and is not a learned reward.

The standard collected telemetry includes uncertainty at selected Euler steps and may include:

- `U10`: disagreement computed over the first 10 action positions;
- `U20`: disagreement over the first 20 positions;
- `U50`: disagreement over the full 50-action chunk;
- per-action or per-step profiles when the corresponding storage flags are enabled.

`ORACLE_BRIEF.md` records an earlier verifier dataset in which uncertainty was measured at every
Euler step, and the stored `uncertainty_stratum` was a tercile of the mean across the denoising
trajectory. Raw per-step values are stored in `pnp_euler_steps`.

### 3.3 P&P refinement

P&P refinement feeds a P&P-derived clean estimate back into the active denoising trajectory.
Current configuration supports multiple variants:

- last-estimate versus average-estimate refinement;
- fractional refinement horizons;
- threshold-gated refinement;
- delayed refinement by action-chunk index;
- tail-tapered refinement;
- executed-prefix-only refinement;
- reduced inner-update strength;
- direct gradient descent on P&P uncertainty;
- equal-RMS random latent-update control;
- action-displacement gating for uncertainty-gradient updates.

These behaviors are represented as fields on `RolloutConfig`; `method` is a taxonomy label rather
than a separate implementation class.

### 3.4 Candidate selection

Candidate-selection experiments generate multiple policy samples at the same state. Depending on
the experiment, selection uses uncertainty, a learned verifier, a learned critic, or a defined
combination. The September five-step experiment used three ordinary candidates and selected the
candidate with lowest U20 before an optional refinement rerun.

### 3.5 PCP/Q correction

The repository contains two uses of “PCP”:

- historical `pnp/pcp.py`: an earlier success-prediction corrector that applies action gradients;
- `pnp/pcp_critic/`: a newer twin clean-action critic over frozen VLA prefix embeddings,
  physical state, and a 50-action chunk.

The conceptual PCP sequence discussed in prior conversations was:

1. predict a base-policy action;
2. move a search center in a critic gradient direction;
3. sample policy-supported reconstructions around that center;
4. score reconstructed candidates;
5. optionally measure stability of the selected candidate;
6. retain the original action when acceptance conditions are not met.

This conceptual sequence is not equivalent to the current Q-planning candidate-ranking baseline.

### 3.6 “RL token” usages in the repository

The term refers to more than one construction:

1. **Stored pi0.5 prefix embeddings:** frozen multimodal prefix-token embeddings captured from
   `embed_prefix(...)`. These are inputs to a learned critic representation; they are not
   themselves a newly trained RL token.
2. **`pnp/pcp_critic.RLTokenEncoder`:** four learned query tokens cross-attend to frozen prefix
   embeddings and receive projected physical state. This encoder is trained with the PCP critic
   in the current code.
3. **Q-planning `rl_token`:** one learned readout token is prepended to the action-token sequence
   in a transformer decoder. It attends through the decoder to frozen prefix context and
   physical-state memory.
4. **RLT paper construction discussed in prior conversations:** pretrain an encoder-decoder to
   reconstruct frozen VLA prefix representations, then freeze the encoder before critic
   training. That separate reconstruction-pretraining pipeline is not implemented by
   `pnp/qplanning_critic/`.

---

## 4. Methodological history recorded in the project

### 4.1 Initial P&P work

The repository contains early standard-LIBERO and LIBERO-PRO P&P experiments covering:

- uncertainty-only rollouts;
- additional denoising-step controls;
- multiple refinement schedules;
- fractional and tapered refinement;
- candidate resampling and selection;
- action-horizon and suffix-sensitivity diagnostics;
- multiple model/checkpoint diversity experiments.

Many of these experiments predate the explicit 10-action execution contract. Their rows remain
distinguishable through experiment labels, configuration hashes, provenance, and, for newer rows,
the explicit `n_action_steps` field.

### 4.2 Verifier/candidate-ranker program

The verifier program collected groups of action candidates from a common simulator state and
trained models to rank successful candidates above failed candidates.

Important collection facts:

- one candidate group corresponds to one decision state;
- candidate groups include one default sample and multiple fresh-noise samples;
- candidate continuations restore the same simulator state;
- continuation randomness is candidate-independent through a shared branch seed;
- the simulator is deterministic under the saved state and seed;
- outcome is task success after executing the candidate prefix and continuing the episode;
- the standard verifier prefix length is 10 actions;
- splits are episode-safe rather than candidate-row-random;
- the confirmatory cohort is stored separately.

### 4.3 Q-VGM revision audit

The local audit compared arXiv:2606.08015 versions 1, 2, and 3. The recorded method changes are:

| Component | v1 | v2 | v3 |
|---|---|---|---|
| Offline critic objective | Cal-QL | stepwise IQL | scalar chunk IQL |
| Critic output | scalar chunk Q | one Q per action position | scalar chunk Q |
| Offline backup | dataset-next-action/SARSA-like | stepwise value heads | scalar value head |
| Online backup | absent | absent | TD with current-policy next action and clipped target Q |
| RL-token training | pretrained/frozen | jointly trained with reconstruction regularization | pretrained/frozen |
| Guidance aggregation | ensemble mean | minimum of two heads | target minimum; online guidance mean; offline aggregation unspecified |
| Guidance schedule | smooth all-step gate | last five denoising steps | last five denoising steps |

The audit also records that the v3 paper reports IQL 88.8, CQL 87.8, SARSA 87.0, and Cal-QL
84.6 on its LIBERO-Spatial critic-objective comparison. These are paper-reported values, not
results from this repository.

### 4.4 Q-planning baseline introduction

The project subsequently defined a Q-planning-inspired critic comparison with two horizons:

- Q50 paper-style;
- Q10 causal/replanning-aligned.

The repository implements both using the same architecture and training schedule apart from the
action/Bellman horizon.

### 4.5 September coarse-search experiments

Two matched 220-identity experiments were added and run:

1. five-step candidate-diversity experiment;
2. single-query five-step/three-step refinement experiment.

Their source and analysis code are present in the repository, and their completed rows were
verified in Supabase on 2026-09-08.

---

## 5. Previous verifier models, training, and results

### 5.1 CompactAdvantageVerifier

Implementation: `pnp/verifier/model.py` and `pnp/verifier/train.py`.

Inputs and dimensions recorded in the status dossier:

- mean-pooled frozen observation embedding: 2048 dimensions;
- action dimension: 7;
- action chunk: 50 positions;
- ranker consumes the first 10 positions;
- context dimension: 128;
- default action width: 32;
- default dropout: 0.10.

The model has:

- a state-value path `V(s)`;
- an action/state advantage path `A(s,a)`;
- production ranking score `A(s,a)`;
- diagnostic joint score `V(s)+A(s,a)`.

Conditioning modes:

| Mode | Mechanism |
|---|---|
| `action_only` | State vector is zeroed. |
| `multiplicative` | The head receives action features and their elementwise product with state features. |
| `film` | State produces scale and shift parameters within temporal residual blocks. |
| `cross_attention` | A state query attends over action tokens. |

Training uses pairwise Bradley–Terry/logistic loss over every positive-negative candidate pair in
a discordant group. The value pathway is pretrained and frozen during ranking. The recorded runs
used zero auxiliary candidate BCE weight.

### 5.2 Compact verifier Stage 1 results

Single seed 42, four folds, development data:

| Configuration | Ranking accuracy | Top-1 uplift | Mean best epoch |
|---|---:|---:|---:|
| `film-d0.4-lr0.0001` | 0.618178 | 0.010108 | 4.0 |
| `multiplicative-d0.2-lr0.0003` | 0.605871 | 0.011170 | 3.0 |
| `film-d0.4-lr0.0003` | 0.603631 | 0.009472 | 9.0 |
| `film-d0.2-lr0.0001` | 0.599022 | 0.007723 | 2.5 |
| `multiplicative-d0.2-lr0.0001` | 0.593473 | 0.003697 | 8.5 |
| `film-d0.2-lr0.0003` | 0.590805 | 0.012142 | 1.5 |
| `cross_attention-d0.4-lr0.0003` | 0.588036 | 0.015361 | 3.5 |
| `action-only` | 0.586659 | 0.010143 | 6.5 |
| `cross_attention-d0.4-lr0.0001` | 0.584545 | 0.012822 | 4.0 |
| `cross_attention-d0.2-lr0.0001` | 0.583324 | 0.007020 | 3.5 |
| `multiplicative-d0.4-lr0.0003` | 0.571767 | 0.005999 | 1.0 |
| `multiplicative-d0.4-lr0.0001` | 0.571484 | 0.000900 | 6.5 |
| `cross_attention-d0.2-lr0.0003` | 0.571469 | 0.001924 | 0.5 |
| `shuffled-actions` | 0.554412 | 0.002684 | 4.0 |

The Stage 1 shortlist was `film-d0.4-lr0.0001` and
`multiplicative-d0.2-lr0.0003`.

### 5.3 Compact verifier Stage 2 results

Three seeds by four folds:

| Configuration | Ranking | Ranking SD | Uplift | Best epoch | Control gap |
|---|---:|---:|---:|---:|---:|
| `action-only` | 0.596533 | 0.055936 | 0.008618 | 5.0 | 0.000000 |
| `film-d0.4-lr0.0001` | 0.595900 | 0.031455 | 0.008499 | 5.0 | -0.000633 |
| `multiplicative-d0.2-lr0.0003` | 0.577512 | 0.023324 | 0.007636 | 3.0 | -0.019021 |
| `shuffled-actions` | 0.563976 | 0.043446 | 0.004102 | 5.0 | -0.032557 |

The selected development configuration was `film-d0.4-lr0.0001`. “Selected” here denotes the
better shortlisted conditioned configuration; it does not denote registration-gate passage.

### 5.4 HybridChunkCritic

Implementation: `pnp/verifier/critic.py` and `pnp/verifier/critic_train.py`.

Recorded architecture:

- twin long critics over 50 actions;
- twin short critics over 10 actions;
- one state-value network;
- width 256 by default;
- state/action fusion includes `[state, position, action, state*action]`;
- three residual MLP blocks per critic member;
- candidate ranking uses the pessimistic minimum of the short twin critics.

Recorded training sequence:

1. train long twin critics and state value with IQL on historical transitions;
2. freeze long critics and value network;
3. train short critics with long-to-short distillation, Monte Carlo regression, and pairwise
   ranking.

Stage 2 recorded results:

| Configuration | Ranking | Ranking SD | Default uplift | Random uplift | Long update | Short epoch |
|---|---:|---:|---:|---:|---:|---:|
| `w512-e0.9-r1.0` | 0.567023 | 0.023724 | 0.008004 | 0.014434 | 2125.0 | 5.5 |
| `w256-e0.9-r1.0` | 0.566007 | 0.022289 | 0.007142 | 0.013571 | 2250.0 | 3.5 |

The status dossier records that the hybrid model's action-only and shuffled-action controls had
not yet been run at the time those numbers were written.

### 5.5 Registration gate

`verifier_registration_eligibility` requires all four bootstrap lower-bound conditions:

1. ranking-accuracy 95% CI lower bound greater than 0.5;
2. top-1 uplift over default 95% CI lower bound greater than 0;
3. paired ranking gap over action-only control 95% CI lower bound greater than 0;
4. paired ranking gap over shuffled-actions control 95% CI lower bound greater than 0.

The compact verifier's Stage 2 selected model had a control gap of `-0.000633` relative to the
stronger control. The status dossier therefore records failure of condition 3. No row or document
reviewed for this handoff records that a verifier passed the full gate.

### 5.6 Sealed confirmatory cohort

- Experiment/cohort name: `verifier-v2-pro-confirmatory`.
- Recorded asserted size: at least 120 groups; the original collection plan used 160 groups.
- Identities are excluded from development training.
- Notebook 11 is designed to open the cohort once for one development-eligible winner.
- No reviewed source records that the sealed cohort has been opened.

---

## 6. Existing-data verifier diagnostics

The following numbers are transcribed from `ORACLE_BRIEF.md`. They describe the development pool
scored by registered film verifier checkpoint `2f1a683673ee4c93`.

### 6.1 Ranking by uncertainty stratum

| Stratum | Groups | Discordant groups | Ranking | 95% CI | Action-only | Control gap | Oracle uplift |
|---|---:|---:|---:|---|---:|---:|---:|
| High | 1001 | 186 | 0.524 | [0.485, 0.566] | 0.499 | 0.026 | 0.069 |
| Low | 96 | 9 | 0.778 | [0.593, 0.926] | 0.463 | 0.315 | 0.031 |
| Mid | 86 | 16 | 0.448 | [0.271, 0.625] | 0.443 | 0.005 | 0.035 |

### 6.2 Ranking by within-group executed-prefix spread

| Spread tercile | Groups | Discordant groups | Ranking | 95% CI | Action-only | Control gap | Oracle uplift |
|---|---:|---:|---:|---|---:|---:|---:|
| High | 394 | 63 | 0.561 | [0.484, 0.637] | 0.539 | 0.022 | 0.079 |
| Low | 395 | 36 | 0.481 | [0.391, 0.569] | 0.467 | 0.014 | 0.030 |
| Mid | 394 | 112 | 0.527 | [0.470, 0.581] | 0.475 | 0.052 | 0.081 |

### 6.3 Mode-level and oracle statistics

- Mode-ranking accuracy: 0.576 over 191 groups with at least two modes and different outcomes.
- Fraction of Bradley–Terry pairs assigned to the same action mode: 0.478 over 211 groups.
- Spearman correlation between model score and cluster-success soft label: 0.068.
- Overall oracle uplift: 0.0634.
- Multimodal-group oracle uplift: 0.0645 over 1,101 groups.
- Unimodal-group oracle uplift: 0.0488 over 82 groups.
- High-spread-tercile oracle uplift: 0.0787 over 394 groups.
- Majority-mode selector success: 0.6418.
- Default-candidate success: 0.6500.
- Group success by uncertainty stratum:
  - high: 0.6295 over 1,001 groups;
  - low: 0.6641 over 96 groups;
  - mid: 0.7587 over 86 groups.

### 6.4 Best-of-K oracle rescue curve

High-uncertainty stratum:

| Candidate budget K | Oracle success | Groups with at least K candidates |
|---:|---:|---:|
| 1 | 0.632 | 1001 |
| 2 | 0.667 | 1001 |
| 4 | 0.689 | 1001 |
| 8 | 0.716 | 931 |

All groups: K=1 yielded 0.644, K=2 yielded 0.679, K=4 yielded 0.700, and K=8 yielded
0.716.

---

## 7. PCP-search data collection and immutable critic snapshot

### 7.1 Initial standard-LIBERO PCP-search collection

The first completed PCP-search cohort contained 400 standard-LIBERO rollouts:

- 400/400 completed;
- 400/400 marked `training_ready`;
- 388 successes and 12 failures in the initial 400-row verification;
- 6,833 H=10 Bellman transitions;
- experiment: `pcp-search-rollouts-v1`;
- manifest: `pcps-c810651498933ba955c51560`;
- partition backfilled as `standard_libero`;
- data split `train`;
- `pcp_train_eligible=true`.

Later standard-LIBERO collection increased the standard partition to 600 rollouts. The immutable
snapshot contains 465 successes and 135 failures for those 600 rows.

### 7.2 PRO collection partitioning

The final train-eligible snapshot includes two 640-rollout PRO training partitions:

| Partition | Rollouts | Successes | Failures |
|---|---:|---:|---:|
| `pcp-search-pro-suite-partition-v1` | 640 | 312 | 328 |
| `pcp-search-pro-fresh-state-train-v1` | 640 | 277 | 363 |

Position-perturbation suites were used as an entire held-out category in the revised PRO strategy.
The six position-heldout sentinel rollouts were tagged `pcp_train_eligible=false` and are not in
the 1,880-row training snapshot.

Sentinel manifest IDs recorded in prior project history:

- train sentinel: `pcps-811f2b0d64141795538e3ac2` (8 rows);
- position-heldout sentinel: `pcps-889be71659ca92a959a4ff2b` (6 rows).

### 7.3 Immutable critic snapshot

Supabase table: `pcp_critic_dataset_snapshots`.

Registered snapshot:

| Field | Value |
|---|---|
| Snapshot ID | `pcpcds-98d1f32dff8213841529b3c6` |
| Name | `qcorrect-existing-data-v1` |
| Created | 2026-09-08 01:55:08 UTC |
| Policy repo | `lerobot/pi05_libero_finetuned` |
| Policy revision | `8e174154ef5f6c60a8da12ae99c303d8963138c1` |
| Artifact schema | 1 |
| Total rollouts | 1,880 |
| Training rollouts | 1,509 |
| Validation rollouts | 371 |
| Successes | 1,054 |
| Failures | 826 |
| Standard LIBERO | 600: 465 success, 135 failure |
| LIBERO-PRO | 1,280: 589 success, 691 failure |

The train/validation split is by rollout/episode identity as represented in the immutable snapshot.
The snapshot contains the action normalization statistics calculated from train-eligible data.

Action means stored in snapshot provenance:

```text
[-0.14768936, -0.08603272, 0.10705525, -0.03094882,
  0.06250120,  0.04529032, -0.19174916]
```

Action standard deviations:

```text
[0.78828621, 0.78123492, 0.91235483, 0.79009593,
 0.76538551, 0.77465618, 0.96778482]
```

### 7.4 Prefix and Bellman corpus scale recorded in prior audits

The previous corpus audit recorded approximately:

- 38,629 ten-action Bellman transitions;
- 40,509 state-boundary prefixes, including next states;
- 381,595 executed environment steps;
- 1,532 distinct episode/initial-state groups;
- 120 tasks;
- 12 suites.

These values refer to the 1,880-rollout train-eligible corpus and its decision boundaries.

### 7.5 Training artifact contents

Current Q-planning loading requires these arrays:

```text
actions_normalized
rewards
terminated
truncated
step_success
boundary/step
boundary/raw_robot_state
boundary/policy_proprio
prefix/prefix_embeddings
prefix/prefix_pad_masks
bellman/action
bellman/executed_normalized
bellman/validity_mask
```

The broader lossless PCP-search artifact may also contain:

- processed images;
- raw camera images;
- token IDs and attention data;
- generated 50-action chunks;
- full step-level physical state;
- simulator state;
- P&P clean estimates and uncertainty telemetry;
- exact metadata needed to reproduce preprocessing.

### 7.6 Prefix meaning

At each planning boundary, `PI05.embed_prefix(...)` generates a token sequence representing the
preprocessed images and language context. The captured float16 prefix embeddings and validity mask
are frozen model outputs. The critic can either consume them directly or compress them with a
learned representation module.

The stored prefix is not the final four-token `RLTokenEncoder` output and is not the single learned
Q-planning decoder readout token.

---

## 8. Storage and egress audit facts

The prior read-only Supabase Storage audit enumerated 119,101 objects. Object metadata summed to
approximately 160.30 GiB, while the Supabase dashboard at the time displayed 24.7 GB used. The
audit noted that dashboard accounting and live object-metadata totals differed.

| Storage family | Objects | Metadata-summed size |
|---|---:|---:|
| PCP training data | 17,097 | 156.55 GiB |
| PnP chunks | 6,132 | 1.92 GiB |
| A-hats | 7,131 | 0.77 GiB |
| Generated chunks | 18,234 | 0.53 GiB |
| Trajectories | 33,304 | 0.40 GiB |
| Verifier candidates | 37,194 | 0.13 GiB |
| Verifier checkpoints | 4 | 0.012 GiB |
| Total | 119,101 | 160.30 GiB |

The audit attributed approximately 150 GiB of uncompressed payload to frozen pi0.5 prefix
embeddings. It also recorded:

- processed image tensors: approximately 68 GiB raw footprint;
- raw agent/wrist images: approximately 29 GiB raw footprint;
- 2-D attention masks: approximately 35 GiB raw footprint;
- token IDs, position IDs, and other masks: below 0.5 GiB;
- simulator/full step state: below 0.5 GiB;
- Bellman actions/rewards/termination: below 0.2 GiB.

These raw field estimates overlap within compressed multipart artifacts and therefore are not
additive to the 160.30 GiB object total.

Creation-date audit for PCP training objects:

| UTC date | Objects | Size | Cumulative size |
|---|---:|---:|---:|
| 2026-08-23 | 3,145 | 27.964 GiB | 27.964 GiB |
| 2026-08-24 | 5,750 | 53.396 GiB | 81.360 GiB |
| 2026-08-25 | 1,398 | 12.663 GiB | 94.023 GiB |
| 2026-08-26 | 929 | 8.523 GiB | 102.546 GiB |
| 2026-08-27 | 3,609 | 33.376 GiB | 135.922 GiB |
| 2026-08-28 | 2,266 | 20.624 GiB | 156.547 GiB |

The audit recorded that data created before 2026-08-23 totaled approximately 2.15 GiB. It also
recorded approximately 1.60 GiB of auxiliary A-hat, P&P-chunk, trajectory, and generated-chunk
artifacts created alongside the PCP collection.

Notebook 56 originally scanned/downloaded complete training artifacts during preflight. Commits
`320aafa`, `eb7fe4d`, `63a313d`, and `6169b50` added resumable scanning, faster caching, isolated
parallel Supabase clients, and streaming Q-planning windows.

---

## 9. Current critic implementations

### 9.1 PCP twin critic (`pnp/pcp_critic`)

#### Architecture

- action horizon: 50;
- action dimension: 7;
- model width: 256;
- learned RL query tokens: 4;
- residual FiLM blocks per Q head: 4;
- Q heads: 2;
- dropout: 0.10;
- one separate state-value head;
- frozen pi0.5 prefix embeddings are inputs, not model parameters;
- physical robot state and policy proprioception are projected into the RL-token queries;
- each Q head pools normalized action tokens with mean and max;
- action context is reinjected through FiLM in every residual block;
- minimum of twin Q values is available for scoring and targets.

#### Training configuration defaults

| Field | Default |
|---|---:|
| Objective | `calql` |
| Seed | 42 |
| Gamma | 0.99 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Updates | 10,000 |
| Evaluation interval | 250 |
| Early-stop patience | 10 |
| Target EMA rate | 0.005 |
| Gradient clipping | 1.0 |
| IQL expectile | 0.7 |
| Conservative weight | 1.0 |
| Local synthetic candidates | 4 |
| Broad synthetic candidates | 4 |
| Local action standard deviation | 0.10 |
| Monte Carlo calibration weight | 1.0 |

The Cal-QL branch generates behavior-local and broad normalized action candidates. The IQL branch
uses the state-value head and expectile loss.

#### Registry status

Supabase table `pcp_critic_models` contained zero rows in the 2026-09-08 read-only check. Thus no
PCP critic checkpoint was registered at that time.

Notebook lineage:

- 56: immutable dataset preflight/snapshot;
- 57: Cal-QL or IQL PCP critic training;
- 58: offline evaluation of a registered checkpoint;
- 59: guarded adapter smoke test.

### 9.2 Q-planning critic (`pnp/qplanning_critic`)

#### Q10 and Q50 data definitions

Q10:

```text
action window: executed actions t:t+10
reward window: 10 environment steps
bootstrap state: next saved planning boundary at t+10
bootstrap action: next executed 10-action window
discount: gamma^10 when bootstrapping is valid
inference candidate: actions 0:10
```

Q50:

```text
action window: 50 actions actually executed under repeated 10-step replanning
reward window: 50 environment steps
bootstrap state: saved planning boundary at t+50
bootstrap action: next executed 50-action window
discount: gamma^50 when bootstrapping is valid
inference candidate: one generated 50-action chunk
```

Both are anchored only at saved planning boundaries. Q50 never substitutes an unexecuted generated
tail into the offline training trajectory.

Termination, truncation, success, and the available end of an episode disable bootstrapping across
the boundary. Actions after a terminal point are padded and masked.

#### Architecture

- prefix embeddings projected to width 768;
- robot state projected to width 768;
- policy proprioception projected to width 768;
- one projected token per action position;
- learned action-position embeddings;
- one learned RL/readout token;
- 8 transformer decoder layers;
- 12 attention heads;
- feed-forward width 3072;
- dropout 0.10;
- 101 categorical value bins over `[0,1]`;
- HL-Gauss target sigma 0.01;
- one online Q network and one EMA target copy;
- no twin Q pair in this implementation.

The decoder target sequence is `[RL token; action tokens]`. Its memory is `[prefix tokens; robot
token; proprio token]`. The 101-bin logits from the decoded RL token define the scalar expected Q.

#### Training schedule

| Field | Full-run value |
|---|---:|
| Seed | 42 |
| Gamma | 0.99 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Effective batch size | 64 |
| Default microbatch | 16 |
| Optimizer updates | 8,000 |
| Linear warmup | 500 updates |
| Post-warmup schedule | cosine decay |
| Print interval | 100 |
| Validation interval | 500 |
| Checkpoint interval | 1,000 |
| EMA rate | 0.005 |
| Gradient clipping | 1.0 |
| Numeric mode | BF16 when supported |

Validation reports:

- HL-Gauss cross-entropy;
- Bellman-target MAE;
- Monte Carlo return MAE;
- mean predicted Q for successful transitions;
- mean predicted Q for failed transitions;
- failure AUC;
- validation transition count.

Checkpoints record:

- format identifier `qplanning_critic_v1`;
- optimizer update;
- immutable snapshot ID;
- cache digest;
- source policy identifier/revision;
- architecture and training config;
- online and EMA target model states;
- optimizer state;
- validation history;
- Torch, NumPy, and Python RNG states.

#### Cache behavior

Current data code supports:

- a materialized Q-planning cache;
- a streaming cache backed by downloaded source artifacts;
- bounded parallel downloads;
- separate Supabase clients per download worker;
- rollout-grouped batch sampling to preserve disk locality;
- resume-safe cache indices;
- a first-10 generated-versus-executed consistency diagnostic.

#### Registry/run status

- Immutable input snapshot exists: `pcpcds-98d1f32dff8213841529b3c6`.
- Notebooks 64 and 65 exist and are output-cleared.
- `pcp_critic_models` contained no registered model rows.
- No Q10 or Q50 training metrics were present in the repository or registry check.

---

## 10. September 2026 matched rollout experiments

### 10.1 Cohort and provenance

Both experiments use the same frozen 220-identity PRO cohort:

- 11 suites;
- 20 identities per suite;
- identical model repository and revision enforced by analysis code;
- exact method/config hashes enforced;
- whole identity matching across every arm;
- saved chunk size required to equal 50;
- episode seed, maximum steps, and chunk size required to match across arms.

The historical comparison arm uses the exact 10-step stock rows from the earlier direct-U20
gradient experiment rather than recollecting a new stock baseline.

### 10.2 Five-step diversity experiment

Experiment label: `pi05-five-step-diversity-pro220-v1`.

Remote status on 2026-09-08:

- 660 completed rows;
- zero non-completed rows in the queried experiment;
- two worker runs;
- 220 rows for each new method;
- 348 successes across the three newly collected arms.

Arms:

| Analysis arm | Method string | Behavior |
|---|---|---|
| Baseline | historical exact stock method | 10-step execution/replanning |
| Single | `five_step_single_query` | One five-step policy query |
| Select | `five_step_x3_lowest_u20` | Three ordinary candidates; select lowest U20 |
| Refine | `five_step_x3_lowest_u20_then_refine` | Select lowest-U20 candidate, then rerun/refine that initial noise |

Overall matched results:

| Arm | Successes / 220 | Success rate | Wilson 95% CI |
|---|---:|---:|---|
| Historical 10-step stock | 117 | 53.18% | [46.59%, 59.66%] |
| Five-step single query | 112 | 50.91% | [44.34%, 57.44%] |
| Five-step ×3 lowest U20 | 117 | 53.18% | [46.59%, 59.66%] |
| Five-step ×3 then refine | 119 | 54.09% | [47.49%, 60.55%] |

Paired comparisons, using a 2,000-resample bootstrap during handoff preparation:

| Comparison | Delta | Bootstrap CI | F→S | S→F | Paired p |
|---|---:|---|---:|---:|---:|
| Five-step single minus stock | -2.27 pp | [-5.45, 1.36] | 5 | 10 | 0.3018 |
| Five-step ×3 select minus stock | 0.00 pp | [-4.10, 4.09] | 11 | 11 | 1.0000 |
| Five-step ×3 refine minus stock | +0.91 pp | [-2.73, 4.55] | 10 | 8 | 0.8145 |
| Five-step ×3 select minus five-step single | +2.27 pp | [-2.27, 6.82] | 14 | 9 | 0.4049 |
| Five-step refine minus five-step select | +0.91 pp | [-2.73, 4.55] | 10 | 8 | 0.8145 |

### 10.3 Coarse single-query refinement experiment

Experiment label: `pi05-coarse-single-refinement-pro220-v1`.

Remote status on 2026-09-08:

- 660 completed rows;
- zero non-completed rows in the queried experiment;
- two worker runs;
- 220 rows for each method;
- 357 successes across the three newly collected arms.

New arms:

| Analysis arm | Method string | Behavior |
|---|---|---|
| Five refine | `five_step_single_refine` | Five-step single query with refinement at configured coarse step |
| Three single | `three_step_single_query` | Three-step single query without refinement |
| Three refine | `three_step_single_refine` | Three-step single query with one refinement |

Seven-arm matched results, including historical arms:

| Arm | Successes / 220 | Success rate | Wilson 95% CI |
|---|---:|---:|---|
| Historical stock 10 | 117 | 53.18% | [46.59%, 59.66%] |
| Historical five-step single | 112 | 50.91% | [44.34%, 57.44%] |
| Historical five-step ×3 select | 117 | 53.18% | [46.59%, 59.66%] |
| Historical five-step ×3 refine | 119 | 54.09% | [47.49%, 60.55%] |
| Five-step single + refine | 119 | 54.09% | [47.49%, 60.55%] |
| Three-step single | 118 | 53.64% | [47.04%, 60.11%] |
| Three-step single + refine | 120 | 54.55% | [47.94%, 60.99%] |

Paired comparisons, using a 2,000-resample bootstrap during handoff preparation:

| Comparison | Delta | Bootstrap CI | F→S | S→F | Paired p |
|---|---:|---|---:|---:|---:|
| Five-step single + refine minus stock | +0.91 pp | [-2.28, 4.09] | 8 | 6 | 0.7905 |
| Three-step single minus stock | +0.45 pp | [-2.27, 3.64] | 6 | 5 | 1.0000 |
| Three-step single + refine minus stock | +1.36 pp | [-2.27, 5.45] | 10 | 7 | 0.6291 |
| Five-step refine minus five-step single | +3.18 pp | [-0.45, 7.27] | 13 | 6 | 0.1671 |
| Three-step refine minus three-step single | +0.91 pp | [-2.27, 4.09] | 8 | 6 | 0.7905 |
| Three-step refine minus five-step refine | +0.45 pp | [-3.64, 4.55] | 11 | 10 | 1.0000 |
| Five-step single refine minus historical five-step ×3 refine | 0.00 pp | [-3.18, 3.18] | 7 | 7 | 1.0000 |

### 10.4 Per-suite success rates for the seven-arm cohort

Each entry is `successes/20`.

| Suite | Stock10 | Five single | Five select | Five select+refine | Five single+refine | Three single | Three refine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `libero_goal_swap` | 7 | 4 | 5 | 6 | 6 | 5 | 6 |
| `libero_goal_task` | 2 | 2 | 3 | 2 | 2 | 2 | 2 |
| `libero_goal_with_yellow_book` | 18 | 18 | 20 | 20 | 19 | 20 | 19 |
| `libero_object_swap` | 3 | 3 | 4 | 4 | 4 | 3 | 3 |
| `libero_object_temp_x0.1` | 17 | 16 | 15 | 17 | 18 | 18 | 19 |
| `libero_object_temp_x0.2` | 7 | 8 | 9 | 9 | 8 | 7 | 8 |
| `libero_object_temp_y0.1` | 20 | 19 | 20 | 19 | 20 | 19 | 19 |
| `libero_object_temp_y0.2` | 11 | 11 | 10 | 11 | 10 | 11 | 13 |
| `libero_object_temp_y0.3` | 3 | 4 | 4 | 4 | 4 | 4 | 4 |
| `libero_object_with_mug` | 20 | 19 | 20 | 20 | 20 | 20 | 20 |
| `libero_spatial_swap` | 9 | 8 | 7 | 7 | 8 | 9 | 7 |

### 10.5 Available telemetry not included in the success tables

The five-step analysis code can derive without simulator execution:

- per-boundary candidate U10/U20/U50;
- selected candidate index frequencies;
- selected U20 and within-boundary U20 spread;
- pairwise cosine similarity/distance;
- action L2 distance;
- gripper disagreement;
- first-boundary-only diversity summaries;
- refinement-path U20 changes;
- inference milliseconds per boundary;
- velocity-field evaluations per boundary.

The coarse analysis can additionally download the small uncertainty-profile artifacts for its
three new arms and produce:

- U10/U20/U50 tables;
- P&P contraction summaries;
- failure-detection summaries;
- per-arm compute summaries.

The read-only success analysis used for this handoff did not download those artifacts.

---

## 11. Current status matrix

| Component | Code | Notebook | Remote data/model status as of 2026-09-08 |
|---|---|---|---|
| Standard/PRO P&P rollout system | Implemented | Many worker notebooks | Multiple completed experiment families |
| Verifier V2 collection | Implemented | 08 + workers | Development and sealed cohorts documented |
| Compact verifier training | Implemented | 09 | Development results and registered film checkpoint documented |
| Hybrid long/short critic | Implemented | 10 | Development results documented; controls incomplete in status dossier |
| Verifier arbitration | Implemented | 11 | No recorded full gate pass; sealed cohort remains documented as sealed |
| PCP-search manifest collection | Implemented | 53–55 + workers | 1,880 train-eligible rows in immutable snapshot |
| PCP critic snapshot | Implemented | 56 | Registered snapshot `pcpcds-98d1...` |
| PCP twin critic training | Implemented | 57 | No registered model row |
| PCP critic offline evaluation | Implemented | 58 | Requires a registered checkpoint |
| PCP search adapter smoke | Implemented | 59 | Guarded offline-only adapter path exists |
| Five-step diversity pilot | Implemented | 60 workers + 61 | 660/660 rows completed; matched results in Section 10 |
| Coarse single-refinement pilot | Implemented | 62 workers + 63 | 660/660 rows completed; matched results in Section 10 |
| Q50 paper-style critic | Implemented | 64 | Snapshot available; no recorded completed training metrics |
| Q10 causal critic | Implemented | 65 | Snapshot available; no recorded completed training metrics |
| Full analysis refactor | Plan plus partial/current modules | Multiple analysis notebooks | `ANALYSIS_REFACTOR_PLAN.md` lists unfinished staged replacement work |

---

## 12. Repository map for future agents

### 12.1 Top-level package files

| Path | Role |
|---|---|
| `README.md` | Package overview, Colab bootstrap, local analysis setup, earlier verifier workflow. |
| `pyproject.toml` | Package metadata and dependency groups: base, `sim`, `analysis`, `train`. |
| `libero_collection_plan.md` | Earlier collection planning document. |
| `qplanning_corrector_training_plan_v2.md` | Detailed Q10/Q50 design contract; implementation-status sentence is stale. |
| `VERIFIER_STATUS_DOSSIER.md` | Factual verifier architecture/data/results summary; untracked at handoff. |
| `ORACLE_BRIEF.md` | Verifier diagnostic briefing and existing-data numbers; untracked at handoff. |
| `ANALYSIS_REFACTOR_PLAN.md` | Planned analysis-package replacement and validation requirements; untracked at handoff. |
| `../reports/qvgm_revision_trust_audit.md` | Version-by-version Q-VGM audit; untracked at handoff. |

### 12.2 Core runtime modules in `pnp/`

| File | Role |
|---|---|
| `config.py` | Central constants, `Method` taxonomy, and `RolloutConfig`. Start here for experiment semantics. |
| `env_setup.py` | Colab dependency/runtime validation and environment setup. |
| `libero_env.py` | LIBERO environment creation, observation conversion, and simulator interaction. |
| `libero_pro.py` | LIBERO-PRO asset/task/suite handling. |
| `models.py` | pi0.5 loading and preprocessing utilities. |
| `sampler.py` | Hooked flow sampler; prefix capture, P&P probes, candidate generation, refinement, and telemetry. |
| `pnp.py` | P&P uncertainty/refinement operations used by the sampler. |
| `pcp.py` | Historical Q-gradient corrector path. Do not confuse with `pnp/pcp_critic/`. |
| `rollout.py` | Episode loop, 10-action execution, artifact accumulation, result construction. |
| `experiments.py` | Shared experiment drivers and method/config construction. |
| `store.py` | Supabase Postgres/Storage API, provenance, artifact serialization, multipart upload/download. |
| `tap.py` | Prefix/model hook utilities, including prefix capture. |
| `notebook.py` | Notebook-facing helpers and progress display. |
| `diversity.py` | Multi-model/candidate diversity utilities. |

### 12.3 Verifier package

| File | Role |
|---|---|
| `verifier/data.py` | Candidate-group and transition data structures, splitting, loading. |
| `verifier/collection.py` | Deterministic-replay candidate collection. |
| `verifier/model.py` | `CompactAdvantageVerifier` and conditioning variants. |
| `verifier/train.py` | Value pretraining, ranker training, controls, evaluation. |
| `verifier/critic.py` | `HybridChunkCritic`: long/short twins and value network. |
| `verifier/critic_train.py` | IQL long-critic and distilled short-critic training. |
| `verifier/diagnostics.py` | Existing-data stratification, mode-level, oracle, and control diagnostics. |

### 12.4 PCP-search collection package

| File | Role |
|---|---|
| `pcp_search/manifest.py` | Immutable manifest structures and hashing. |
| `pcp_search/registry.py` | Supabase manifest publication/loading and immutable checks. |
| `pcp_search/control.py` | Control-plane functions for sentinel/full manifest publication. |
| `pcp_search/collection.py` | Worker-side execution of manifest-assigned rollouts. |
| `pcp_search/data.py` | Training-artifact assembly and validation. |
| `pcp_search/pro.py` | PRO suite/category allocation and held-out-category logic. |
| `pcp_search/survey.py` | Read-only profiling of historical collection data. |
| `pcp_search/task_selection.py` | Standard-LIBERO task/difficulty allocation. |
| `pcp_search/monitor.py` | Manifest result/progress inspection. |
| `pcp_search/README.md` | PCP-search-specific workflow documentation. |

### 12.5 PCP critic package

| File | Role |
|---|---|
| `pcp_critic/config.py` | Twin critic, training objective, and adapter configs. |
| `pcp_critic/model.py` | Four-query RL-token encoder, twin action-conditioned Q heads, value head. |
| `pcp_critic/objectives.py` | Bellman, IQL expectile, and calibrated conservative losses. |
| `pcp_critic/data.py` | Eligible-row filtering, snapshot construction, H=10 transition extraction, compact cache. |
| `pcp_critic/resumable_snapshot.py` | Resume-safe artifact scanning and local progress state. |
| `pcp_critic/train.py` | PCP critic training/evaluation/checkpoint production. |
| `pcp_critic/workflow.py` | Notebook-facing preflight/training orchestration. |
| `pcp_critic/registry.py` | Immutable snapshot and model registry; Storage checkpoint payloads. |
| `pcp_critic/deploy.py` | Offline-only guarded scoring/correction adapter. |

### 12.6 Q-planning package

| File | Role |
|---|---|
| `qplanning_critic/config.py` | Q10/Q50 model and fixed training configuration. |
| `qplanning_critic/data.py` | Executed-trajectory window construction; materialized and streaming caches. |
| `qplanning_critic/model.py` | Transformer-decoder critic with one learned readout token and HL-Gauss head. |
| `qplanning_critic/train.py` | Fixed 8,000-update trainer, validation, EMA, checkpoint resume. |
| `qplanning_critic/workflow.py` | Snapshot/cache preflight and smoke/full notebook entry point. |

### 12.7 Experiment-specific runtime modules

| File | Role |
|---|---|
| `five_step_diversity_experiment.py` | Frozen 220-identity three-arm five-step worker definition and run driver. |
| `coarse_refinement_experiment.py` | Frozen 220-identity three-arm coarse five/three-step driver. |
| `uncertainty_gradient_experiment.py` | Direct U20-gradient and control rollout definitions. |
| `uncertainty_gradient_gate_experiment.py` | Action-displacement-gated U-gradient experiment definitions. |
| `uncertainty_critic.py` | Earlier uncertainty critic architecture. |
| `uncertainty_critic_train.py` | Earlier uncertainty critic training. |
| `uncertainty_critic_v2.py` | Step-aligned residual U20 critic variant. |

### 12.8 Analysis package

| File | Role |
|---|---|
| `analysis/load.py` | Legacy/current Supabase data loading helpers. |
| `analysis/snapshot.py` | Local versioned Supabase snapshot creation. |
| `analysis/validate.py` | Coverage, pairing, uniqueness, and artifact validation. |
| `analysis/conditions.py` | Canonical condition/cohort naming. |
| `analysis/statistics.py` | Wilson intervals, paired bootstrap, discordant-pair tests, AUC helpers. |
| `analysis/metrics.py` | Shared metric computations. |
| `analysis/standard_libero.py` | Standard-LIBERO result analyses. |
| `analysis/pro.py` | LIBERO-PRO analyses. |
| `analysis/pro_expanded.py` | Expanded PRO cohort analyses. |
| `analysis/pcp.py` | Historical PCP feature/checkpoint/evaluation analysis. |
| `analysis/geometry.py` | Multimodality, PCA/isotropy, and directional geometry. |
| `analysis/denoising.py` | Denoising-step telemetry analysis. |
| `analysis/horizon_diagnostics.py` | U10/U20/U50 and action-horizon comparisons. |
| `analysis/suffix_sensitivity.py` | Unused-suffix influence and tapered behavior. |
| `analysis/uncertainty_gradient.py` | Direct uncertainty-gradient pilot analysis. |
| `analysis/uncertainty_gradient_gate.py` | Action-gated uncertainty-gradient analysis. |
| `analysis/five_step_diversity.py` | Four-arm matched September analysis and candidate geometry. |
| `analysis/coarse_refinement.py` | Seven-arm matched September analysis and probe telemetry. |
| `analysis/report.py` | Shared tables/figures/report generation. |
| `analysis/style.py` | Plot style. |
| `analysis/run_analysis.py` | Analysis CLI orchestration. |

### 12.9 Supabase schema and migrations

| Path | Role |
|---|---|
| `supabase/schema.sql` | Canonical schema and storage setup. |
| `migrations/002_verifier.sql` | Verifier model/group/candidate tables. |
| `migrations/003_verifier_v2.sql` | V2 verifier split/trajectory columns and indices. |
| `migrations/004_u_iter.sql` | Iteration-level uncertainty additions. |
| `migrations/005_pcp_search.sql` | PCP-search eligibility columns, manifests, results, and training-ready view. |
| `migrations/006_pcp_critic.sql` | Immutable critic snapshot and model registries. |

Primary tables:

| Table | Role |
|---|---|
| `experiments` | Experiment definitions. |
| `experiment_runs` | Run-level config and model provenance. |
| `rollouts` | One row per rollout/config identity, status, outcome, and artifact references. |
| `pnp_euler_steps` | Per-denoising-step uncertainty summaries. |
| `pnp_action_vectors` | Action-vector telemetry. |
| `baseline_uncertainty` | Baseline uncertainty records. |
| `q_correctors` | Historical Q-corrector registry. |
| `verifier_models` | Verifier checkpoint registry. |
| `verifier_candidate_groups` | Candidate-group metadata and split. |
| `verifier_candidates` | Candidate rows and outcomes. |
| `encoding_cache` | Stored encoding references. |
| `pcp_search_manifests` | Immutable PCP collection manifests. |
| `pcp_search_manifest_results` | Per-manifest rollout result linkage/status. |
| `pcp_critic_dataset_snapshots` | Immutable critic dataset snapshots. |
| `pcp_critic_models` | PCP critic checkpoint registry. |

Storage bucket constant: `artifacts`.

### 12.10 Notebook lineage

All top-level notebooks listed below were output-cleared at handoff time.

| Number/range | Role |
|---|---|
| 00 | Verify LIBERO-PRO assets, identity, and smoke rollout. |
| 01 | General standard/PRO experiment driver. |
| 08 | Verifier V2 PRO collection controller. |
| 09 | State-conditioned verifier architecture sweep. |
| 10 | Hybrid long/short chunk critic. |
| 11 | Development arbitration and sealed confirmation. |
| 12 | Existing-data verifier diagnostics. |
| 13 | Model-free wave-0 screening. |
| 14–15 | Commit-horizon collection and comparison. |
| 16 | Expanded 13-suite PRO analysis. |
| 17–23 | Two-model diversity training, analysis, selective refinement, and chunk selection. |
| 24–30 | Source-policy threshold, multi-query, prefix-gate, and delayed-refinement experiments. |
| 31–44 | Action-horizon, fractional/tapered/prefix refinement, suffix sensitivity, and horizon diagnostics. |
| 45–51 | Uncertainty-critic, direct-U20-gradient, and action-gate experiments. |
| 52 workers | Standard-LIBERO PCP-search collection. |
| 53–55 | PCP-search survey, manifest control, PRO workers, and monitor. |
| 56–59 | PCP critic snapshot, training, offline evaluation, and adapter smoke. |
| 60–61 | Five-step diversity workers and analysis. |
| 62–63 | Coarse single-refinement workers and analysis. |
| 64 | Q50 paper-style training. |
| 65 | Q10 causal training. |

Current September worker files:

```text
notebooks/workers/60_five_step_diversity_pro220_worker_0.ipynb
notebooks/workers/60_five_step_diversity_pro220_worker_1.ipynb
notebooks/workers/62_coarse_single_refinement_pro220_worker_0.ipynb
notebooks/workers/62_coarse_single_refinement_pro220_worker_1.ipynb
```

The repository status at handoff showed
`notebooks/workers/53_pcp_search_pro_worker_3.ipynb` as deleted locally. Worker files 0, 1, and 2
were still present. The deletion was not part of the pull and was not modified during preparation
of this document.

### 12.11 Notebook-generation and utility scripts

Scripts named `build_*_notebook.py`, `build_*_notebooks.py`, or `generate_*_workers.py` are the
source generators for many committed notebooks. When changing shared notebook structure, inspect
the generator before editing generated `.ipynb` JSON directly.

Notable scripts:

| Script | Role |
|---|---|
| `scripts/nb_common.py` | Shared notebook-construction helpers. |
| `scripts/colab_bootstrap.py` | Standard Colab bootstrap cell construction. |
| `scripts/generate_pcp_search_workers.py` | PCP-search worker generation. |
| `scripts/build_five_step_diversity_notebooks.py` | Notebook 60 worker generation. |
| `scripts/build_five_step_analysis_notebook.py` | Notebook 61 generation. |
| `scripts/build_coarse_refinement_notebooks.py` | Notebook 62 worker generation. |
| `scripts/build_coarse_refinement_analysis_notebook.py` | Notebook 63 generation. |
| `scripts/audit_supabase_storage.py` | Read-only storage audit utility; untracked at handoff. |
| `scripts/backfill_legacy.py` | Trust-gated import of older SQLite data. |

### 12.12 Tests

The test suite covers:

- environment setup and EGL behavior;
- rollout construction, rendering skips, matrices, and resume behavior;
- storage serialization and resume behavior;
- P&P fractional, tapered, prefix-strength, threshold, suffix, and horizon variants;
- verifier collection, data, model, and diagnostics;
- uncertainty critics and uncertainty-gradient experiments;
- PCP-search collection, data, PRO allocation, and task selection;
- PCP critic data/model/objective behavior;
- resumable snapshot scanning;
- Q-planning windows, caching, model, and training/checkpoint behavior;
- five-step and coarse experiment/analysis paths;
- analysis snapshots, validation, PRO, denoising, and horizon diagnostics.

Focused verification performed on 2026-09-08:

```text
99 passed in 8.76s
```

The focused command covered:

```text
tests/test_qplanning_critic.py
tests/test_qplanning_training.py
tests/test_resumable_snapshot.py
tests/test_five_step_diversity_experiment.py
tests/test_five_step_diversity_analysis.py
tests/test_coarse_refinement_experiment.py
tests/test_coarse_refinement_analysis.py
tests/test_rollout.py
tests/test_store.py
```

No full-suite result was generated during this handoff.

---

## 13. Current Git and local-worktree facts

The repository was fast-forwarded from `4405757` to `6169b50`. The pull added 41 files or file
changes with approximately 6,235 insertions and 77 deletions.

Pulled commits:

| Commit | Local date | Subject |
|---|---|---|
| `841bac7` | 2026-09-03 | Add five-step diversity pilot workers and candidate telemetry |
| `a76267d` | 2026-09-03 | Add five-step diversity analysis notebook and pairwise cosine plots |
| `31eb9da` | 2026-09-03 | Add single-query five/three-step refinement workers |
| `8a257f1` | 2026-09-04 | Add coarse refinement PRO220 analysis notebook |
| `e55cedd` | 2026-09-07 | Add Q10 and Q50 critic training tests |
| `320aafa` | 2026-09-07 | Make PCP snapshot scans resumable |
| `eb7fe4d` | 2026-09-07 | Speed up Q-planning artifact caching |
| `63a313d` | 2026-09-08 | Isolate parallel Supabase download clients |
| `6169b50` | 2026-09-08 | Stream Q-planning windows from source cache |

Local pre-existing changes after the pull and before adding this handoff file:

```text
 M .DS_Store
 M pnp-vla/.gitignore
 D pnp-vla/notebooks/workers/53_pcp_search_pro_worker_3.ipynb
?? .gitignore
?? pnp-vla/.evn
?? pnp-vla/ANALYSIS_REFACTOR_PLAN.md
?? pnp-vla/ORACLE_BRIEF.md
?? pnp-vla/VERIFIER_STATUS_DOSSIER.md
?? pnp-vla/scripts/audit_supabase_storage.py
?? reports/
```

Additional facts:

- `pnp-vla/.evn` was zero bytes.
- The local `pnp-vla/.gitignore` change adds `oracle-discussion/` as a local-only ignored path.
- None of the pre-existing changes above were reverted or staged while preparing this document.

---

## 14. Data-integrity and experimental-control contracts

### 14.1 Identity and pairing

- Match rollouts by suite, task index, episode index, and initialization-state hash.
- Keep every configuration's coverage explicit.
- Do not aggregate different config hashes under the same `method` label.
- Keep whole episodes/identities in one split.
- Require exact model repository and revision agreement before pooling.
- Require exact frozen-manifest hash agreement for the September 220-identity analyses.

### 14.2 Action execution

- New experiments explicitly set the executed prefix horizon.
- Current PCP/Q-planning work uses 10 executed actions before replanning.
- The model may produce 50 actions even when only the first 10 are executed.
- Q10 uses the executed 10-action causal interval.
- Q50 uses 50 actions from the actual closed-loop trajectory formed by repeated replanning.

### 14.3 Held-out data

- Position perturbations are excluded from the PRO critic-training partitions in the revised
  collection strategy.
- `pcp_train_eligible=false` is the hard database filter for critic snapshot construction.
- The verifier confirmatory cohort is separate from development.
- Analysis code rejects identities outside the frozen manifest.

### 14.4 Provenance

`experiment_runs` records:

- experiment and run IDs;
- configuration JSON;
- package Git SHA and dirty state;
- policy repository and revision;
- package/library versions where available;
- host/runtime metadata.

Artifact and model registries additionally record hashes, schema versions, and immutable IDs.

### 14.5 Statistical units

- Verifier ranking is macro-averaged by candidate group.
- Candidate-pair correlations within a group are not treated as independent rollout counts.
- September success comparisons are paired by the same 220 identities.
- Wilson intervals describe individual arm success rates.
- Paired bootstrap intervals and discordant-pair tests describe matched arm differences.
- The handoff's September paired intervals used 2,000 bootstrap resamples; analysis defaults use
  5,000 where not overridden.

---

## 15. Recorded unresolved states and missing results

This section lists absences or incomplete states without assigning priority.

1. `pcp_critic_models` had no registered model rows on 2026-09-08.
2. No completed Q10 or Q50 training metrics were found in Supabase registry metadata or committed
   notebook outputs.
3. Notebook 57 has a PCP Cal-QL/IQL training path, but no registered checkpoint was present.
4. Notebook 58 requires a registered checkpoint for offline evaluation.
5. The verifier Stage 2 compact conditioned model did not exceed the recorded action-only control.
6. The hybrid critic status record does not include completed action-only and shuffled-action
   controls.
7. No record reviewed for this handoff states that the sealed verifier confirmatory cohort was
   opened.
8. The Q-planning implementation uses a learned decoder readout token; a separate RLT
   reconstruction-pretraining/freeze pipeline is not implemented there.
9. The lossless prefix corpus occupies the majority of Supabase Storage metadata size.
10. The September rollout experiments have complete success data, but their detailed candidate
    geometry and uncertainty-profile artifact analyses were not recomputed during this handoff.
11. `ANALYSIS_REFACTOR_PLAN.md` specifies a staged replacement analysis system; the repository
    contains many of the named modules, but this handoff did not audit every acceptance criterion
    in that plan.
12. Committed notebook outputs are cleared, so notebook execution history is not recoverable from
    Git alone.
13. The local worktree contains untracked documentation and a deleted PRO worker notebook.

---

## 16. Fast orientation procedure for a future agent

The following is a factual navigation sequence, not an experiment recommendation.

### 16.1 To understand current rollout semantics

Read in order:

```text
pnp/config.py
pnp/sampler.py
pnp/pnp.py
pnp/rollout.py
pnp/experiments.py
```

Relevant search terms:

```text
n_action_steps
pnp_steps
pnp_k
refine
candidate_seed_scheme
save_training_data
save_time_uncertainty
```

### 16.2 To inspect the completed September experiments

Read:

```text
pnp/five_step_diversity_experiment.py
analysis/five_step_diversity.py
pnp/coarse_refinement_experiment.py
analysis/coarse_refinement.py
notebooks/61_analyze_five_step_diversity_pro220.ipynb
notebooks/63_analyze_coarse_single_refinement_pro220.ipynb
```

Experiment labels:

```text
pi05-five-step-diversity-pro220-v1
pi05-coarse-single-refinement-pro220-v1
```

### 16.3 To inspect critic data

Read:

```text
pnp/pcp_search/data.py
pnp/pcp_critic/data.py
pnp/pcp_critic/resumable_snapshot.py
pnp/qplanning_critic/data.py
```

Immutable snapshot:

```text
pcpcds-98d1f32dff8213841529b3c6
```

### 16.4 To distinguish critic families

```text
pnp/verifier/model.py            # compact pairwise ranker
pnp/verifier/critic.py           # hybrid long/short critic
pnp/pcp_critic/model.py          # twin Q + four-query RL-token encoder
pnp/qplanning_critic/model.py    # single categorical Q + decoder readout token
pnp/pcp.py                       # historical gradient corrector
```

### 16.5 To inspect database state without downloading large artifacts

Use `SupabaseStore.fetch_all(...)` on metadata columns from:

```text
rollouts
experiment_runs
pcp_search_manifests
pcp_search_manifest_results
pcp_critic_dataset_snapshots
pcp_critic_models
verifier_models
verifier_candidate_groups
verifier_candidates
```

Avoid calling artifact download methods when only counts/status/configuration are required.
`training_data_path`, `ahats_path`, and related columns reference Storage blobs.

### 16.6 To run local focused verification

The project-local virtual environment points to a working Python 3.13 installation. The repository
root also contains `.python-version` requesting Python 3.12, which was not installed through
`pyenv` during handoff preparation. Running `.venv/bin/python -m pytest` bypassed that mismatch.

Example focused invocation:

```bash
cd pnp-vla
.venv/bin/python -m pytest -q \
  tests/test_qplanning_critic.py \
  tests/test_qplanning_training.py \
  tests/test_resumable_snapshot.py \
  tests/test_five_step_diversity_experiment.py \
  tests/test_five_step_diversity_analysis.py \
  tests/test_coarse_refinement_experiment.py \
  tests/test_coarse_refinement_analysis.py \
  tests/test_rollout.py \
  tests/test_store.py \
  -p no:cacheprovider
```

### 16.7 Colab dependency contract

`pyproject.toml` currently pins the simulation stack to:

- Transformers 5.5.4;
- LeRobot commit `01dcb4c29222bc9f2388cebf87f0e79965a9508b`;
- LIBERO 0.1.1;
- MuJoCo `>=3.1.6,<3.10`.

The setup code preserves Colab's native Torch/TorchVision/CUDA stack and stores Hugging Face
model files under local `/content/hf_home` by default.

---

## 17. Source documents

Repository/local documents used in this handoff:

- [`README.md`](README.md)
- [`notebooks/README.md`](notebooks/README.md)
- [`qplanning_corrector_training_plan_v2.md`](qplanning_corrector_training_plan_v2.md)
- [`VERIFIER_STATUS_DOSSIER.md`](VERIFIER_STATUS_DOSSIER.md)
- [`ORACLE_BRIEF.md`](ORACLE_BRIEF.md)
- [`ANALYSIS_REFACTOR_PLAN.md`](ANALYSIS_REFACTOR_PLAN.md)
- [`../reports/qvgm_revision_trust_audit.md`](../reports/qvgm_revision_trust_audit.md)
- [`pnp/pcp_search/README.md`](pnp/pcp_search/README.md)
- [`supabase/schema.sql`](supabase/schema.sql)

Code paths used to verify current contracts:

- [`pnp/config.py`](pnp/config.py)
- [`pnp/store.py`](pnp/store.py)
- [`pnp/pcp_critic/`](pnp/pcp_critic/)
- [`pnp/qplanning_critic/`](pnp/qplanning_critic/)
- [`pnp/five_step_diversity_experiment.py`](pnp/five_step_diversity_experiment.py)
- [`pnp/coarse_refinement_experiment.py`](pnp/coarse_refinement_experiment.py)
- [`analysis/five_step_diversity.py`](analysis/five_step_diversity.py)
- [`analysis/coarse_refinement.py`](analysis/coarse_refinement.py)

External papers discussed in the project history:

- Q-VGM / Q-Guided Value-Gradient Matching: arXiv:2606.08015.
- RL Token: arXiv:2604.23073 and Physical Intelligence's `rlt.pdf`.
- Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning:
  arXiv:2608.21204.

---

## 18. One-page factual status summary

- The repository is synchronized to `origin/main` at `6169b50`.
- The frozen base policy produces 50-action chunks; current collection executes 10 actions before
  replanning.
- P&P uncertainty and many refinement/search variants are implemented.
- The earlier compact verifier and hybrid critic have development results but no documented full
  registration-gate pass.
- The sealed verifier confirmatory cohort has no documented opening.
- PCP-search collection produced an immutable 1,880-rollout critic snapshot with 1,054 successes
  and 826 failures across standard LIBERO and LIBERO-PRO.
- The snapshot contains 1,509 training and 371 validation rollouts and is pinned to pi0.5 revision
  `8e174154...`.
- Supabase contained the immutable dataset snapshot and zero registered PCP critic models on
  2026-09-08.
- The Q10/Q50 trainer, streaming cache, resume logic, and notebooks are implemented.
- No completed Q10/Q50 training metrics were found.
- Both September 220-identity pilots completed all 660 new rollout rows.
- Seven-arm matched success ranged from 50.91% to 54.55%; stock was 53.18%.
- The largest stock-relative point estimate was +1.36 percentage points for three-step refine,
  with a 2,000-bootstrap interval of [-2.27, 5.45] percentage points.
- The focused current-code verification passed 99 tests.
- The worktree contains pre-existing modified, deleted, and untracked files listed in Section 13.

