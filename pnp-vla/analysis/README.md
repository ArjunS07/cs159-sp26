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
written beneath `analysis_outputs/<experiment>/<snapshot_id>/`; each table is emitted as CSV and
Parquet, with a JSON manifest, validation record, availability record, figures, and `findings.md`.

Commands are `snapshot`, `validate`, `standard`, `pro`, `pcp`, and `all`. Missing optional data is
represented by explicit `not_available` records. Validation failures exit nonzero. In particular,
the current K=3 collection cannot support Sarle's bimodality coefficient, and refinement telemetry
is never admitted to the prospective observed-arm detector analysis.
