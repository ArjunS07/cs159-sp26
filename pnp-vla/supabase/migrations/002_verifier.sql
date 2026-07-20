-- Apply once to an existing v1 database before running verifier notebooks 03/04.
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS generated_chunks_path TEXT;

CREATE TABLE IF NOT EXISTS verifier_models (
    verifier_id TEXT PRIMARY KEY,
    experiment TEXT,
    run_id UUID REFERENCES experiment_runs(run_id),
    model_class TEXT NOT NULL,
    obs_dim INTEGER NOT NULL,
    action_dim INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    prefix_length INTEGER,
    checkpoint_path TEXT NOT NULL,
    split_path TEXT NOT NULL,
    config_json JSONB NOT NULL,
    metrics_json JSONB NOT NULL,
    dataset_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verifier_candidate_groups (
    candidate_group_id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    suite TEXT NOT NULL,
    task_idx INTEGER NOT NULL,
    episode_idx INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    uncertainty_stratum TEXT NOT NULL,
    pairing_mode TEXT NOT NULL,
    prefix_length INTEGER NOT NULL,
    snapshot_validated BOOLEAN NOT NULL,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verifier_candidates (
    candidate_id TEXT PRIMARY KEY,
    candidate_group_id TEXT NOT NULL REFERENCES verifier_candidate_groups(candidate_group_id)
        ON DELETE CASCADE,
    rollout_id TEXT REFERENCES rollouts(rollout_id) ON DELETE SET NULL,
    candidate_kind TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    n_steps INTEGER,
    policy_chunk_path TEXT NOT NULL,
    env_chunk_path TEXT NOT NULL,
    observation_path TEXT NOT NULL,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(candidate_group_id, candidate_kind)
);

CREATE INDEX IF NOT EXISTS idx_verifier_candidates_group
    ON verifier_candidates(candidate_group_id);
