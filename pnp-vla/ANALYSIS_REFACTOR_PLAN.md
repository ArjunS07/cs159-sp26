# Analysis Refactor Plan

## Objective

Replace the legacy analysis notebooks with one reproducible, configuration-aware offline
analysis package over Supabase. The replacement must reproduce every defensible legacy output,
add the analyses enabled by the new rollout design, and refuse misleading comparisons when data
coverage or statistical assumptions do not match.

The analysis must never aggregate configurations with different identity coverage into a single
comparative bar. In particular, `method` is a taxonomy field, not an experimental condition;
`config_hash` plus cohort membership defines a condition.

## Data contract and validation gates

Every analysis command begins by materializing a versioned local snapshot (Parquet plus a JSON
manifest) from Supabase. The manifest records the experiment label, query time, row counts,
rollout/config hashes, package Git SHA, schema/sampler versions, and validation results.

Validation is fatal unless explicitly marked as an expected unavailable analysis:

1. Require one completed row per `(experiment, identity, config_hash)` and reject duplicate,
   pending, failed-to-log, or error rows from the primary analysis.
2. Verify the standard-LIBERO manifest has 400 identities and the hybrid matrix has exactly:
   400 shared observed arms, 400 16-step controls, 400 `(4,5)` refine-last arms, and 80 rows for
   each remaining full-ablation configuration.
3. Assign `full_ablation` versus `broad_validation` from a versioned manifest, not from row counts
   or an ad hoc SQL `CASE`. Cross-check this assignment against each run's stored driver config.
4. Treat the shared `pnp_uncertainty_only` rollout as the no-op/vanilla behavioral baseline.
   Confirm that uncertainty sampling used the isolated perturbation RNG and did not replace the
   executed base action.
5. Pair outcomes only on `(suite, task_idx, episode_idx, init_state_hash)`, with a one-to-one merge
   validated for each individual `config_hash`.
6. Keep null and unavailable metrics explicit. Never synthesize fake rollout rows for alternate
   uncertainty schedules; derive step subsets from the shared step-indexed telemetry.
7. Verify artifact references before geometry or PCP analyses. Report missing/corrupt artifacts
   and exclude them with counts.
8. Print a compact coverage matrix before calculating results and save it as a table.

## Package structure

Refactor `analysis/` into modules with narrow responsibilities:

| Module | Responsibility |
| --- | --- |
| `snapshot.py` | Paginated Supabase extraction, local Parquet cache, snapshot manifest |
| `validate.py` | Identity/config coverage, uniqueness, status, pairing, artifact checks |
| `conditions.py` | Canonical cohort and human-readable configuration labels |
| `statistics.py` | Wilson intervals, paired bootstrap, McNemar/binomial tests, AUC/PR-AUC |
| `standard_libero.py` | Standard-LIBERO success, schedule, transition, and detector analyses |
| `pro.py` | LIBERO-PRO perturbation/cohort robustness analyses |
| `pcp.py` | PCP dataset, checkpoint, calibration, and three-way evaluation summaries |
| `geometry.py` | Multimodality, directional variance, PCA/isotropy, online features, confounds |
| `report.py` | Stable CSV/JSON/Markdown tables and publication figures |
| `run_analysis.py` | Thin CLI orchestration only |

The CLI supports `snapshot`, `validate`, `standard`, `pro`, `pcp`, and `all`. Re-running analysis
against an unchanged snapshot must not contact Supabase.

## Standard-LIBERO analysis

### Success and schedule effects

Produce configuration-level success rates with Wilson 95% intervals:

- Shared observed/no-op baseline on all 400 identities.
- 16-step matched-compute control and `(4,5)` refinement on all 400 identities.
- All 12 configurations on the 80-identity full-ablation cohort.
- Per-suite and per-task versions with denominators shown.

Primary comparisons are paired within identical identity sets:

- All-identity: observed versus 16-step control versus `(4,5)` refinement.
- Full ablation: observed versus 16/19/25-step controls and each of the eight schedules.
- Schedule families: adjacent two-step versus periodic schedules, while preserving each schedule
  as a separate condition.

For every paired comparison report baseline SR, condition SR, percentage-point delta, `F->S`,
`S->F`, `S->S`, `F->F`, a paired confidence interval, and a two-sided discordant-pair test.
Apply a declared multiple-comparison correction to the eight schedule tests and report raw and
adjusted values. Keep effect sizes visible even when tests are inconclusive.

### Failure prediction

Use only the shared observed arm for prospective failure prediction. Refinement telemetry is
post-treatment and is reported separately, never mixed into the primary detector result.

Report:

- Failure prevalence and score distributions by outcome.
- Pooled ROC-AUC and PR-AUC with identity-bootstrap intervals.
- Per-suite AUC and macro suite-stratified AUC with intervals.
- Per-DOF AUC for all seven dimensions, plus position+gripper and full-vector scores.
- Early-window versus full-episode scores and uncertainty/episode-length correlation.
- Threshold metrics using cross-validation or a held-out calibration split; do not select and
  evaluate a threshold on the same 400 outcomes.
- Performance by suite, task, and the historical hard-task cohort.

The old median-split uncertainty taxonomy is reproduced as a clearly labeled descriptive legacy
table. A calibration/reliability table and risk-coverage curve supersede it for inference.

## LIBERO-PRO analysis

Run when the deduplicated PRO manifest exists; otherwise emit an explicit `not_available` record
without failing the standard-LIBERO report.

This section supersedes legacy Tables 8–9:

- Baseline and refinement SR by suite, perturbation family, axis, strength, distractor, and
  canonical/expanded cohort membership.
- Paired recovery/degradation against the shared observed baseline.
- Pooled, per-suite, and macro-stratified detector ROC-AUC/PR-AUC.
- Seven-DOF, position+gripper, and full-vector uncertainty comparisons.
- In-distribution-to-PRO detector transfer and performance degradation relative to standard
  LIBERO.
- Cross-model analysis when matching pi0.5 and SmolVLA conditions exist.

No PRO analysis may average duplicated canonical/expanded identities; cohort membership is
metadata on one deduplicated identity.

## PCP analysis

This section supersedes legacy Table 6 and the analysis portions of notebook 02:

- Validate PCP feature artifacts and report usable chunks, identities, successes/failures, and
  train/validation/test grouping. Splits must be by identity or task, never by correlated chunk.
- Report training curves, held-out ROC-AUC/PR-AUC, calibration, checkpoint/config hash, and model
  parameter count.
- Reproduce offline gradient sanity checks with effect sizes and uncertainty intervals.
- Report the paired three-way live evaluation: vanilla, P&P-only (`lambda=0`), and PCP using the
  declared deployment lambda.
- Add gate-fire rate, correction norm, Q score, applied corrections, recovery/degradation, and
  compute/latency summaries.

## Geometry and uncertainty structure

This section supersedes all legacy geometry outputs:

- Sarle bimodality coefficient and its failure-prediction AUC.
- Directional parallel/lateral variance and their ratio.
- PCA first-component fraction, isotropy, and Marchenko-Pastur comparison.
- Online feature-gating analysis and episode-length confounding.
- Cross-model geometry comparisons when compatible artifacts exist.

Geometry metrics declare their minimum sample requirement. In particular, Sarle's coefficient is
not valid for the current `K=3` collection; it must be marked unavailable until a larger-K run is
collected rather than producing a number from insufficient samples.

## Legacy-to-new output map

| Legacy output | Replacement |
| --- | --- |
| Table 2: success by method | Configuration/cohort SR tables plus balanced paired comparisons |
| Table 3: recovery/degradation | Per-config paired transition table with intervals/tests |
| Table 4: detector metrics | Pooled, suite-stratified, per-suite, early-window detector report |
| Table 5: median taxonomy | Legacy taxonomy plus calibration and risk-coverage analysis |
| Table 6: PCP three-way | Paired PCP evaluation plus correction and compute diagnostics |
| Tables 8–9: PRO | Deduplicated perturbation analysis and per-DOF detector report |
| Pooled-vs-stratified correction | Both estimates with uncertainty intervals by default |
| Geometry tables | Validated artifact-backed geometry report with minimum-K gates |
| Cross-model transfer | Matched-condition model comparison with explicit coverage |
| Numbers changed vs report | Machine-readable legacy/new delta table with provenance |

## Outputs

Each run writes to `analysis_outputs/<experiment>/<snapshot_id>/`:

- `manifest.json` and `validation.json`.
- Tidy CSV and Parquet tables with stable schemas.
- A concise Markdown summary containing denominators, effect sizes, intervals, and caveats.
- Publication figures whose labels name the exact configuration and cohort.
- `legacy_delta.csv`, comparing recomputed compatible metrics with historical report values.

Figures must never use a truncated axis that visually exaggerates small SR differences, and every
comparative figure must show uncertainty intervals and sample sizes.

## Tests and acceptance criteria

1. Unit-test condition labels, cohort assignment, one-to-one pairing, transitions, AUC edge cases,
   confidence intervals, multiple-testing correction, and artifact validation.
2. Use a synthetic fixture with deliberately unequal configuration coverage to prove that the
   method-level aggregation bug cannot recur.
3. Snapshot tests cover table schemas and deterministic report generation.
4. Integration-test paginated Supabase extraction with mocked pages and duplicate rows.
5. Reproduce all compatible legacy tables from a legacy fixture within documented tolerances.
6. On the completed standard-LIBERO experiment, validation must report exactly 400 identities and
   1,920 rollouts, and the primary all-identity table must compare three 400-row conditions.
7. `python -m analysis.run_analysis --experiment ...` exits nonzero on invalid coverage and exits
   successfully with explicit unavailable sections when PRO, PCP, cross-model, or larger-K data
   have not yet been collected.

## Implementation sequence and commits

1. **Snapshot and validation foundation** — cache Supabase data, formalize conditions/cohorts, and
   add coverage/pairing tests.
2. **Standard-LIBERO report** — replace aggregate method plots with balanced config-level and
   paired analyses; add detector and calibration outputs.
3. **Legacy parity** — implement taxonomy, pooled/stratified comparison, report-delta mapping, and
   regression fixtures.
4. **PRO and cross-model report** — activate when data is present and test deduplication/cohorts.
5. **PCP report** — validate feature splits/checkpoints and generate the three-way evaluation.
6. **Geometry report** — implement artifact-backed metrics and enforce minimum-K requirements.
7. **CLI, documentation, and final audit** — stable output manifest, end-to-end tests, and removal
   of notebook-only analysis paths from the supported workflow.

Each stage receives its own focused commit. Generated analysis artifacts remain ignored; code,
tests, schemas, and this plan are committed.
