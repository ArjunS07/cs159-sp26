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

Commands are `snapshot`, `validate`, `standard`, `pro`, `pcp`, and `all`. Missing optional data is
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
row per identity for observed/no-op, 16-step control, and refine-last `(4,5)`. The expanded
16-suite collection remains explicitly `not_available` until complete. Legacy threshold heatmaps
are labeled exploratory and in-sample; the selective-refinement estimate is reported separately
from five held-out, suite/outcome-stratified folds. Detector thresholds transferred from standard
LIBERO are selected using standard labels only. Storage references are verified during snapshot
creation; `ahats` and observation-frame analyses remain unavailable because those artifacts are
not present.
