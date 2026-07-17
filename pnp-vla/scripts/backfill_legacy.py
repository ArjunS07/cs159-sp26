"""Trust-gated backfill of legacy SQLite result DBs into Supabase (experiment='legacy').

Reads the old DBs (both schema variants: the 01/02/03 RolloutDB schema and the
pnp_pro `pnp_enabled`/`pnp_mode` schema), maps rows onto the canonical schema, and upserts
them under experiment='legacy' with a per-source-DB trust tag.

Excludes:
- synthetic `synthbase_`-prefixed rows (fabricated baselines)
- rows whose config predates the RNG-isolation fix (sampler_algo_version < 2) unless the
  user explicitly whitelists that source DB

Prints a running byte estimate. The per-DB keep/drop decision is the user's (they confirm
which DBs on their Drive are post-fix / trustworthy).

TODO(phase 6): implement once the user lists the trustworthy source DBs.
"""
