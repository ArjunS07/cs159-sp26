# PnP-VLA Verifier / Corrector — Current Status Dossier

Factual description only. No claims about *why* results are what they are. Everything below is drawn from the repo (`pnp-vla/pnp/verifier/`, notebooks 03–11) and the two most recent training runs (notebooks 09 and 10).

---

## 0. The task the verifier is trying to solve

pnp-vla runs the **pi0.5 flow-matching VLA policy** on LIBERO and LIBERO-PRO. At each action-chunk decision point, the policy's Euler denoising loop is tapped to produce, at one fixed decision state, **multiple candidate action chunks**: one `default` (the policy's own unperturbed first draw) plus additional `fresh_noise_i` draws. The **verifier/critic** scores each candidate at that same state; the top-scoring candidate replaces the default for execution.

- Everything is conditioned on a **frozen** observation embedding `obs_enc` — the mean-pooled prefix embedding from the policy's `embed_prefix` pass (PaliGemma-based pi0-style VLA, action expert separate). Dim = **2048**. Nothing backprops into it.
- Action chunk: horizon **50**, action_dim **7**. Candidates share one canonical sim state and one `obs_enc`.
- A separate mechanism, the **PCP "Q-corrector"** (`pnp/pcp.py`), gradient-nudges a single action toward higher predicted success. It is NOT what notebooks 09/10 evaluate — those are the **candidate-ranking** verifier/critic.

The registration decision is gated on the ranking model beating two controls (see §5). Notebook **11** arbitrates between the 09 baseline and the 10 hybrid critic on development data; only if one clears the causal controls does it open the sealed confirmatory cohort once.

---

## 1. The two models under test

### 1a. Notebook 09 — `CompactAdvantageVerifier` (state-conditioned advantage head)
`pnp/verifier/model.py`. Defaults: `obs_dim=2048, action_dim=7, context_dim=128, action_width=32, dropout=0.10`.

Two pathways:
- **Value pathway** `V(s)`: `obs_encoder` (LN→Lin2048→256→GELU→Drop→Lin→128→LN) + `position_encoder` (scalar chunk_position → 32) → `state_head` → scalar. Pretrained on rollout success, then **frozen** during ranking.
- **Advantage pathway** `A(s,a)`: encodes the first `prefix_length=10` steps of the action chunk. State summary = `context_rank([obs,position]) → 32`.

Four conditioning modes (how state fuses into the action score):
| mode | fusion |
|---|---|
| **action_only** | state vector zeroed; only raw action features reach the head (architecturally = multiplicative with state=0) |
| **multiplicative** (default) | 2 dilated `ResidualTemporalBlock`s → mean+max pool → `advantage_head([action, action*state])`. Single late elementwise product. |
| **film** | state produces per-feature (γ,β) scale+shift **inside** each of 2 `FiLMResidualTemporalBlock`s → `film_head`. Fusion throughout the encoder. |
| **cross_attention** | state → query token; 4-head `MultiheadAttention` over the 10 (pre-conv) action tokens with padding mask → `cross_head`. |

Production ranking score = `A(s,a)`. `score_candidates = V(s) + A(s,a)`.

**Training** (`AdvantageTrainConfig`, `train_advantage`):
- Value pathway frozen every epoch. Two AdamW param groups: action path at `rank_lr`, `context_rank` at `rank_lr * context_lr_multiplier` (default **0.1**).
- Loss = **pairwise ranking** over all positive×negative candidate pairs within a group: `softplus(-(A_pos − A_neg)).mean()` (smooth logistic). Batch = one candidate group at a time.
- Optional `candidate_bce_weight` aux (BCE on `V+A` joint logits); **0.0** in these runs.
- Early stop on val `group_macro_ranking_accuracy`, patience per config.

### 1b. Notebook 10 — `HybridChunkCritic` (IQL twin critic + distillation)
`pnp/verifier/critic.py`, `critic_train.py`. Defaults: `obs_dim=2048, action_dim=7, long_horizon=50, short_horizon=10, width=256, dropout=0.10`.

- `long_critics`: **twin** (double-Q) `ChunkCriticMember`s over the full **50-step** chunk.
- `short_critics`: **twin** members over the first **10 steps** — the deployable candidate-selection critic.
- `value_network`: single `StateValueNetwork` V(s) (obs+position only).
- Each `ChunkCriticMember`: state / position / action encoders → `fusion(Linear(width*4→width))` over `[state, position, action, state*action]` (explicit state×action interaction) → 3 `ResidualMLPBlock`s → scalar head.
- Ranking score per candidate = `short_values(first 10 steps).amin(0)` (pessimistic min over twins).

**Training** (`HybridCriticTrainConfig`; key defaults: `gamma=.999, expectile=.7, distill_expectile=.8, rank_weight=1.0, mc_weight=1.0, distill_weight=1.0, lr=3e-4, batch=256, long_updates=20000, short_epochs=50`):
- **`train_long_critic`** — IQL on historical **transitions**: twin-Q Smooth-L1 to TD target `r + γ·V_target(s')` (EMA target net, rate .005); V via **expectile regression** toward `min-Q` (`expectile=.7`). Trains `long_critics` + `value_network` only.
- **`train_short_critic`** — freezes long critics + V; trains `short_critics` on three losses per step:
  1. **Distillation**: expectile (`.8`) regression of short-twin (10-step) toward frozen long-critic `min-Q` (50-step) on historical transitions.
  2. **Monte-Carlo**: Smooth-L1 of short value to `return_target` (else `float(success)`) on candidate groups.
  3. **Ranking**: same `softplus(-(pos−neg))` pairwise loss, weighted by `rank_weight`.
- Early stop on `group_macro_ranking_accuracy` (reuses 09's `evaluate_candidate_ranker`).

Both models therefore share the **same evaluation code, same metrics, same controls, same registration gate.**

---

## 2. Metrics (identical across 09 and 10)

Per candidate group, using advantage/critic scores:
- **`group_macro_ranking_accuracy`** = mean over **discordant** groups of pairwise accuracy: fraction of (success × failure) candidate pairs where the model scores success above failure (ties = 0.5). **0.5 = chance.**
- **`top1_uplift_default`** = mean over all groups of `success(top-1 pick) − success(default candidate)`.
- **`top1_uplift_random`** = mean of `success(top-1) − mean success in group`.
- **`mean_score_margin`** = mean (pos − neg) score gap on discordant groups.
- **Discordant group** = a group whose candidates disagree on outcome (≥1 success AND ≥1 failure). Only these carry ranking signal. All-same-outcome groups are dropped from ranking accuracy.
- **CIs**: group-level (cluster) bootstrap — resample whole groups' scalar stats, 10k resamples, [2.5%, 97.5%]. `paired_candidate_comparison` bootstraps conditioned−control differences **matched by group_id**.

---

## 3. Data

- **Candidate groups** (development): 5 datasets — `verifier-clean-pairs-v3`, `-v4-dev`, `-v4-test`, `verifier-online-selection-v1`, `verifier-v2-pro-development`. Newest PRO cohort = **12 candidates/group**; older cohorts default 4.
- **Historical transitions** (for value pretrain / long critic): `libero-hybrid-schedules-k3-v1`, `libero-pro-canonical-core-k3-v1`.
- **Sealed confirmatory**: `verifier-v2-pro-confirmatory` — IDs only; its episode identities are **excluded** from all training. Never opened by 09/10.
- **Asserted sizes** (nb 09): new PRO groups **≥ 220**; development **discordant_groups ≥ 100**; sealed **≥ 120**. `N_FOLDS=4`, `PREFIX_LENGTH=10`, `SEEDS=(42,43,44)`.
- **Collection method**: deterministic-replay branching — replay a fixed 10-step action prefix to a mid-rollout state, then draw `default` + fresh-noise candidates; outcome = whether `env.check_success()` fires after each candidate runs to completion. States chosen by policy uncertainty stratum (low/mid/high).
- **Splits**: episode-safe throughout (`known_task_split`, `candidate_cv_splits`); whole episodes assigned to one fold; historical data re-purged per fold of that fold's candidate episodes; sealed identities removed before anything.
- No verifier result numbers are stored in the repo (notebook outputs cleared; results live in Supabase/W&B). One earlier verifier checkpoint id exists (`3add05c827424c4a`) with no recorded metrics.

---

## 4. RESULTS

### Notebook 09 — Stage 1 (all 14 configs = 12 architectures + 2 controls, single seed 42, mean over 4 folds)
```
                                  ranking    uplift  best_epoch
film-d0.4-lr0.0001             0.618178  0.010108         4.0
multiplicative-d0.2-lr0.0003   0.605871  0.011170         3.0
film-d0.4-lr0.0003             0.603631  0.009472         9.0
film-d0.2-lr0.0001             0.599022  0.007723         2.5
multiplicative-d0.2-lr0.0001   0.593473  0.003697         8.5
film-d0.2-lr0.0003             0.590805  0.012142         1.5
cross_attention-d0.4-lr0.0003  0.588036  0.015361         3.5
action-only                    0.586659  0.010143         6.5
cross_attention-d0.4-lr0.0001  0.584545  0.012822         4.0
cross_attention-d0.2-lr0.0001  0.583324  0.007020         3.5
multiplicative-d0.4-lr0.0003   0.571767  0.005999         1.0
multiplicative-d0.4-lr0.0001   0.571484  0.000900         6.5
cross_attention-d0.2-lr0.0003  0.571469  0.001924         0.5
shuffled-actions               0.554412  0.002684         4.0
shortlist ['film-d0.4-lr0.0001', 'multiplicative-d0.2-lr0.0003']
```

### Notebook 09 — Stage 2 (shortlist + 2 controls, 3 seeds × 4 folds)
```
                              ranking  ranking_std    uplift  best_epoch  control_gap
action-only                  0.596533     0.055936  0.008618         5.0     0.000000
film-d0.4-lr0.0001           0.595900     0.031455  0.008499         5.0    -0.000633
multiplicative-d0.2-lr0.0003 0.577512     0.023324  0.007636         3.0    -0.019021
shuffled-actions             0.563976     0.043446  0.004102         5.0    -0.032557
selected film-d0.4-lr0.0001
```
`control_gap = ranking − max(action_only, shuffled)`. Both conditioned architectures land **at or below** the action_only control. `selected` is only the better of the two shortlisted; it is not a gate pass. Every `ranking_std` (0.02–0.06) exceeds the between-config gaps.

### Notebook 10 — Stage 2 (shortlist, 3 seeds × 4 folds; controls not yet run)
```
                 ranking  ranking_std  uplift_default  uplift_random  long_update  short_epoch
w512-e0.9-r1.0  0.567023     0.023724        0.008004       0.014434       2125.0          5.5
w256-e0.9-r1.0  0.566007     0.022289        0.007142       0.013571       2250.0          3.5
selected w512-e0.9-r1.0
```
Sweep grid was 8 configs: width∈{256,512} × expectile∈{.7,.9} × rank_weight∈{1,3}, dropout .15 fixed. Absolute ranking ≈ 0.567 (just above the 0.5 chance floor). The two shortlisted configs differ by 0.001 (< 1 std). The `action_only` (`zero_context`) and `shuffled_actions` controls run in Section 6, so `control_gap` for 10 is **not yet available**.

---

## 5. Registration gate (`verifier_registration_eligibility`)
All four must hold (each is a **bootstrap CI lower-bound** condition):
1. `ranking_accuracy_ci95[0] > 0.5`
2. `top1_uplift_default_ci95[0] > 0`
3. `beats_action_only`: paired ranking-gap vs action_only control, CI lower bound `> 0`
4. `beats_shuffled_actions`: paired ranking-gap vs shuffled control, CI lower bound `> 0`

Given 09's Stage-2 `control_gap` of −0.0006 (film) and −0.019 (multiplicative), check #3 fails for 09's selected model. Notebook 11 stops if neither model clears these controls.

---

## 6. Notebook lineage (for context)
- **03–07** (archived v1): value+advantage verifier, candidate-pair collection, targeted best-of-8 collection, online selection eval, training diagnostics.
- **08**: canonical LIBERO-PRO collection (240 dev + 160 sealed groups, 12 candidates each).
- **09**: state-conditioned advantage sweep (this dossier).
- **10**: hybrid twin long/short critic (this dossier).
- **11**: arbitrate 09 vs 10 on development; open sealed cohort once for the single winner.

---

## Open question (for the model receiving this)
Given the above, how to make the candidate corrector/verifier better — is the limiting factor architecture, data (quantity/quality/collection method/label structure), model capacity, training scheme, or something fundamentally broken in the setup? Describe concrete, testable directions.
