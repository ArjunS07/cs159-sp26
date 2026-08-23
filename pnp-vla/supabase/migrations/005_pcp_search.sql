-- PCP-search immutable collection manifests and Bellman/RL-token artifact readiness.
-- Idempotent: safe to paste into the Supabase SQL editor more than once.

ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS behavior_seed_index INTEGER DEFAULT 0;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_data_path TEXT;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_data_schema_version INTEGER;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_ready BOOLEAN DEFAULT FALSE;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_validation_json JSONB;
CREATE INDEX IF NOT EXISTS idx_rollouts_training_ready
    ON rollouts(training_ready, suite, task_idx);

CREATE TABLE IF NOT EXISTS pcp_search_manifests (
    manifest_id        TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    schema_version     INTEGER NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('draft', 'frozen')),
    parent_manifest_id TEXT REFERENCES pcp_search_manifests(manifest_id),
    n_rollouts         INTEGER NOT NULL CHECK (n_rollouts > 0),
    policy_repo_id     TEXT NOT NULL,
    policy_revision    TEXT NOT NULL,
    collection_config  JSONB NOT NULL,
    provenance_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
    manifest_path      TEXT NOT NULL,
    manifest_sha256    TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    frozen_at          TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS pcp_search_manifest_results (
    manifest_id     TEXT NOT NULL REFERENCES pcp_search_manifests(manifest_id),
    ordinal         INTEGER NOT NULL CHECK (ordinal >= 0),
    rollout_id      TEXT REFERENCES rollouts(rollout_id),
    status          TEXT NOT NULL CHECK (
        status IN ('collected', 'training_ready', 'excluded', 'errored')),
    reason          TEXT,
    validation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (manifest_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_pcp_search_results_rollout
    ON pcp_search_manifest_results(rollout_id);

-- Internal collection-control tables: service-role only. With no anon/authenticated policy,
-- RLS denies all public client access by default.
ALTER TABLE pcp_search_manifests ENABLE ROW LEVEL SECURITY;
ALTER TABLE pcp_search_manifest_results ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE pcp_search_manifests, pcp_search_manifest_results
    FROM anon, authenticated;

CREATE OR REPLACE FUNCTION reject_frozen_pcp_search_manifest_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status = 'frozen' THEN
        RAISE EXCEPTION 'PCP-search manifest % is frozen and immutable', OLD.manifest_id;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS pcp_search_manifest_immutable ON pcp_search_manifests;
CREATE TRIGGER pcp_search_manifest_immutable
BEFORE UPDATE OR DELETE ON pcp_search_manifests
FOR EACH ROW EXECUTE FUNCTION reject_frozen_pcp_search_manifest_mutation();

CREATE OR REPLACE VIEW pcp_search_training_ready_rollouts AS
SELECT r.*, m.manifest_id, m.name AS manifest_name, mr.ordinal AS manifest_ordinal
FROM rollouts r
JOIN pcp_search_manifest_results mr ON mr.rollout_id = r.rollout_id
JOIN pcp_search_manifests m ON m.manifest_id = mr.manifest_id
WHERE r.training_ready IS TRUE AND mr.status = 'training_ready';
ALTER VIEW pcp_search_training_ready_rollouts SET (security_invoker = true);
REVOKE ALL ON TABLE pcp_search_training_ready_rollouts FROM anon, authenticated;
