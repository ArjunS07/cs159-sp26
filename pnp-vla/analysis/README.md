# Offline analysis workflow

`analysis.run_analysis` is the supported replacement for the legacy analysis notebooks. It
groups experimental conditions by `config_hash`, assigns the historical cohort from the
versioned package manifest, and validates coverage before computing any result.

Create an isolated environment from `pnp-vla/`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-analysis uv sync --extra analysis
```

Load the root credentials into the process without printing them, then create a snapshot and run
all available analyses:

```bash
set -a
source ../.env
set +a
.venv-analysis/bin/python -m analysis.run_analysis snapshot \
  --experiment libero-hybrid-schedules-k3-v1
.venv-analysis/bin/python -m analysis.run_analysis all \
  --experiment libero-hybrid-schedules-k3-v1
```

The second command reuses the latest snapshot and does not contact Supabase. Pass `--refresh`
only when a new snapshot is intended, or `--snapshot PATH` to pin an existing snapshot. Output is
written beneath `analysis_outputs/<experiment>/<snapshot_id>/`. Analysis tables use CSV. Parquet
is reserved for the versioned raw Supabase snapshot cache (`rollouts`, step/vector telemetry, and
run metadata), where preserving nested data and types matters. Availability is recorded in
`manifest.json`, alongside `validation.json` and publication figures.

Commands are `snapshot`, `validate`, `standard`, `pro`, `pro-expanded`, `pcp`, and `all`. Missing optional data is
represented by explicit `not_available` records. Validation failures exit nonzero. In particular,
the current K=3 collection cannot support Sarle's bimodality coefficient, and refinement telemetry
is never admitted to the prospective observed-arm detector analysis.

For the canonical LIBERO-PRO collection, refresh its snapshot once and analyze it against a
validated standard-LIBERO snapshot:

```bash
.venv-analysis/bin/python -m analysis.run_analysis snapshot \
  --experiment libero-pro-canonical-core-k3-v1
.venv-analysis/bin/python -m analysis.run_analysis pro \
  --experiment libero-pro-canonical-core-k3-v1 \
  --reference-experiment libero-hybrid-schedules-k3-v1
```

The canonical PRO result requires exactly 600 identities across the six declared suites and one
row per identity for observed/no-op, 16-step control, and refine-last `(4,5)`. Legacy threshold heatmaps
are labeled exploratory and in-sample; the selective-refinement estimate is reported separately
from five held-out, suite/outcome-stratified folds. Detector thresholds transferred from standard
LIBERO are selected using standard labels only. Storage references are verified during snapshot
creation; `ahats` and observation-frame analyses remain unavailable because those artifacts are
not present.

For the expanded workers 0--5 experiment, use the dedicated command or run
`notebooks/16_analyze_expanded_pro.ipynb`:

```bash
.venv-analysis/bin/python -m analysis.run_analysis pro-expanded \
  --experiment pro-16suite-k5-steps34-v1 --refresh
```

This validates the 13 retained suites, 2,400 paired identities, K=5 and steps `(3,4)`, with the
matched-compute control optional because it was deferred during collection. It also analyzes
`pnp_action_vectors.u_iter`: episode-level contraction features are computed from the observed
arm only, then paired with the refine-last outcome. The primary correctability population is
observed failures, and the target is a paired failure-to-success transition. Full `a_hat` blobs
are not required.

PRO reports also use the complete nine-step denoising telemetry. They report per-step detector
metrics, uncertainty/action-dispersion/action-motion profiles, and regularized logistic models
using either the legacy scalar, the nine-step uncertainty profile, or the full denoising dynamics.
Every model metric comes from held-out predictions in deterministic five-fold splits stratified by
suite and outcome. First-four-chunk and full-episode windows are reported separately so that early
actionable prediction is not conflated with retrospective full-trajectory discrimination.
