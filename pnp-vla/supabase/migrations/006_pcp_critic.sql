-- Immutable PCP critic data snapshots and offline-only checkpoint registry.
-- Apply after 005_pcp_search.sql.  Both tables are service-role control-plane data.

CREATE TABLE IF NOT EXISTS pcp_critic_dataset_snapshots (
    snapshot_id             TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    policy_repo_id          TEXT NOT NULL,
    policy_revision         TEXT NOT NULL,
    artifact_schema_version INTEGER NOT NULL,
    n_rollouts              INTEGER NOT NULL CHECK (n_rollouts > 0),
    n_train_rollouts        INTEGER NOT NULL CHECK (n_train_rollouts > 0),
    n_val_rollouts          INTEGER NOT NULL CHECK (n_val_rollouts >= 0),
    snapshot_path           TEXT NOT NULL,
    snapshot_sha256         TEXT NOT NULL,
    provenance_json         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pcp_critic_models (
    critic_id               TEXT PRIMARY KEY,
    run_id                  UUID REFERENCES experiment_runs(run_id),
    experiment              TEXT,
    snapshot_id             TEXT NOT NULL REFERENCES pcp_critic_dataset_snapshots(snapshot_id),
    policy_repo_id          TEXT NOT NULL,
    policy_revision         TEXT NOT NULL,
    artifact_schema_version INTEGER NOT NULL,
    objective               TEXT NOT NULL CHECK (objective IN ('calql', 'iql')),
    architecture_json       JSONB NOT NULL,
    train_config_json       JSONB NOT NULL,
    metrics_json            JSONB NOT NULL,
    checkpoint_path         TEXT NOT NULL,
    safety_status           TEXT NOT NULL CHECK (safety_status IN ('offline_only')) DEFAULT 'offline_only',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pcp_critic_models_snapshot ON pcp_critic_models(snapshot_id);

ALTER TABLE pcp_critic_dataset_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE pcp_critic_models ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE pcp_critic_dataset_snapshots, pcp_critic_models FROM anon, authenticated;
