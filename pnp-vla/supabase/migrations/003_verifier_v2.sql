-- Verifier V2 uses multiple policy-noise trajectories per fixed LIBERO identity.
-- Metadata remains the compatibility source of truth; generated columns make
-- progress/audit queries inexpensive without rewriting historical rows.
ALTER TABLE verifier_candidate_groups
    ADD COLUMN IF NOT EXISTS trajectory_seed BIGINT,
    ADD COLUMN IF NOT EXISTS collection_split TEXT,
    ADD COLUMN IF NOT EXISTS manifest_hash TEXT,
    ADD COLUMN IF NOT EXISTS model_revision TEXT;

CREATE INDEX IF NOT EXISTS idx_verifier_groups_experiment_split
    ON verifier_candidate_groups(experiment, collection_split);

CREATE INDEX IF NOT EXISTS idx_verifier_groups_trajectory
    ON verifier_candidate_groups(benchmark, suite, task_idx, episode_idx, trajectory_seed);
