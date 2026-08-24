-- ============================================================================
-- Canonical Postgres schema for the pnp results store (provenance-first).
--
-- Apply in the Supabase SQL editor (or `supabase db push`). Then create a PRIVATE
-- Storage bucket named `artifacts` (Dashboard -> Storage -> New bucket).
--
-- Design: an `experiment` label is the versioning boundary; `experiment_runs`
-- captures env/model/asset provenance once per driver invocation; rollouts FK to
-- the run that last wrote them. Small uncertainty rows live here; bulky arrays go
-- to Storage (paths recorded as *_path columns).
-- ============================================================================

-- ── experiments: light registry / version boundary ─────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    experiment  TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes       TEXT
);

-- ── experiment_runs: one row per driver invocation; provenance captured once ─
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment                TEXT REFERENCES experiments(experiment),
    label                     TEXT,
    driver                    TEXT,
    benchmark                 TEXT,          -- 'libero' | 'libero_pro'
    status                    TEXT DEFAULT 'running',  -- running | completed | failed
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at               TIMESTAMPTZ,
    n_rollouts                INTEGER DEFAULT 0,
    notes                     TEXT,
    -- provenance
    pnp_git_sha               TEXT,
    git_dirty                 BOOLEAN,
    pnp_version               TEXT,
    sampler_algo_version      INTEGER,
    schema_version            INTEGER,
    policy_model              TEXT,
    model_repo_id             TEXT,
    model_revision            TEXT,
    weights_sha256            TEXT,
    policy_config_hash        TEXT,
    lerobot_version           TEXT,
    torch_version             TEXT,
    cuda_version              TEXT,
    gpu_name                  TEXT,
    mujoco_gl                 TEXT,
    python_version            TEXT,
    hostname                  TEXT,
    libero_version            TEXT,
    libero_pro_asset_revision TEXT,
    libero_pro_asset_sha256   TEXT,
    -- numerics / determinism
    global_seed               BIGINT,
    tf32_enabled              BOOLEAN,
    matmul_precision          TEXT,
    autocast_dtype            TEXT,
    cudnn_deterministic       BOOLEAN,
    config_json               JSONB
);

-- ── rollouts: one episode under one method-config ───────────────────────────
CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id                TEXT PRIMARY KEY,
    experiment                TEXT,
    run_id                    UUID REFERENCES experiment_runs(run_id),
    sampler_algo_version      INTEGER,
    schema_version            INTEGER,
    -- identity
    benchmark                 TEXT,
    suite                     TEXT,
    task_idx                  INTEGER,
    task_desc                 TEXT,
    episode_idx               INTEGER,
    init_state_hash           TEXT,
    bddl_sha256               TEXT,
    -- task descriptors (explicit; no regex-parsing of suite in analysis)
    suite_family              TEXT,          -- base | position_perturb | distractor | swap | task
    perturb_axis              TEXT,          -- 'x' | 'y' | NULL
    perturb_strength          REAL,
    distractor_object         TEXT,
    canonical_member          BOOLEAN,
    expanded_member           BOOLEAN,
    max_steps                 INTEGER,
    chunk_size                INTEGER,
    n_chunks                  INTEGER,
    -- config (denormalized for filtering)
    method                    TEXT,
    pnp_enabled               BOOLEAN,
    pnp_step_indices          JSONB,
    pnp_k                     INTEGER,
    refine_average            BOOLEAN,
    pnp_time_min              REAL,
    action_dim                INTEGER,
    num_inference_steps       INTEGER,
    num_samples               INTEGER,
    -- PCP config (nullable)
    correction_lambda         REAL,
    q_gate                    REAL,
    correction_steps          JSONB,
    q_ckpt_id                 TEXT,
    -- reproducibility
    episode_seed              BIGINT,
    perturb_seed              BIGINT,
    config_json               JSONB,
    config_hash               TEXT,
    -- outcome + health
    success                   BOOLEAN,
    n_steps                   INTEGER,
    elapsed_s                 REAL,
    terminated_reason         TEXT,          -- success | timeout | done
    status                    TEXT,          -- completed | errored
    error_msg                 TEXT,
    nan_action_count          INTEGER,
    started_at                TIMESTAMPTZ,
    finished_at               TIMESTAMPTZ,
    -- compute accounting
    n_vf_evals                INTEGER,
    inference_ms_total        REAL,
    -- summary metrics
    u_mean_episode            REAL,
    u_max_episode             REAL,
    n_pnp_activations         INTEGER,
    u_mean_d0 REAL, u_mean_d1 REAL, u_mean_d2 REAL, u_mean_d3 REAL,
    u_mean_d4 REAL, u_mean_d5 REAL, u_mean_d6 REAL,
    action_delta_l2_mean      REAL,
    action_delta_l2_max       REAL,
    action_var_mean           REAL,
    gripper_flip_count        INTEGER,
    gripper_flip_rate         REAL,
    chunk_disagreement_mean   REAL,
    mm_bc_pc1_episode         REAL,
    -- PCP deployment telemetry (nullable)
    n_corrections_applied     INTEGER,
    gate_fire_rate            REAL,
    mean_correction_norm      REAL,
    mean_q_score              REAL,
    -- multi-sample telemetry (nullable)
    ms_chosen_idx             INTEGER,
    ms_candidate_u            JSONB,
    -- artifacts (Storage keys, nullable)
    video_path                TEXT,
    ahats_path                TEXT,
    pcp_chunks_path           TEXT,
    trajectory_path           TEXT,
    generated_chunks_path     TEXT,
    obs_frames_path           TEXT,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rollouts_experiment ON rollouts(experiment);
CREATE INDEX IF NOT EXISTS idx_rollouts_run        ON rollouts(run_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_filter     ON rollouts(experiment, method, refine_average, suite);

-- PCP-search uses a second policy-noise stream on selected physical states. This belongs to the
-- identity (and rollout hash), while the training artifact/readiness columns are persistence-only.
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS behavior_seed_index INTEGER DEFAULT 0;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_data_path TEXT;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_data_schema_version INTEGER;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_ready BOOLEAN DEFAULT FALSE;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS training_validation_json JSONB;
CREATE INDEX IF NOT EXISTS idx_rollouts_training_ready
    ON rollouts(training_ready, suite, task_idx);

-- ── PCP-search immutable rollout manifests + separate mutable progress ──────
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

-- These are service-role-only experiment-control tables. No anon/authenticated policy is
-- intentionally defined: RLS therefore denies those clients by default.
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

-- ── PCP critic immutable data/checkpoint registry ──────────────────────────
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

-- ── pnp_euler_steps: per P&P-active step (queryable for the detector) ────────
CREATE TABLE IF NOT EXISTS pnp_euler_steps (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rollout_id        TEXT NOT NULL REFERENCES rollouts(rollout_id) ON DELETE CASCADE,
    chunk_idx         INTEGER NOT NULL,
    chunk_noise_seed  BIGINT,
    euler_step        INTEGER NOT NULL,
    s                 REAL,
    u_mean            REAL,
    u_max             REAL,
    a_std_mean        REAL,
    u_d0 REAL, u_d1 REAL, u_d2 REAL, u_d3 REAL, u_d4 REAL, u_d5 REAL, u_d6 REAL,
    a_std_d0 REAL, a_std_d1 REAL, a_std_d2 REAL, a_std_d3 REAL,
    a_std_d4 REAL, a_std_d5 REAL, a_std_d6 REAL
);
CREATE INDEX IF NOT EXISTS idx_pes_rollout ON pnp_euler_steps(rollout_id);

-- ── pnp_action_vectors: small per-step geometry vectors ─────────────────────
CREATE TABLE IF NOT EXISTS pnp_action_vectors (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rollout_id   TEXT NOT NULL REFERENCES rollouts(rollout_id) ON DELETE CASCADE,
    chunk_idx    INTEGER NOT NULL,
    euler_step   INTEGER NOT NULL,
    a_mean_vec   JSONB,
    a_std_vec    JSONB,
    bc_vec       JSONB,
    mm_pc1_frac  REAL,
    mm_bc_pc1    REAL,
    -- Per-ITERATION consecutive disagreement from the K predict-and-perturb iterations:
    -- u_iter is (K-1,) scalars, u_iter_vec is (K-1, action_dim). u_mean/u_vec in
    -- pnp_euler_steps average this axis away; these keep it so decay across the K
    -- perturbations is measurable.
    u_iter       JSONB,
    u_iter_vec   JSONB
);
CREATE INDEX IF NOT EXISTS idx_pav_rollout ON pnp_action_vectors(rollout_id);

-- ── baseline_uncertainty: multi-sample baseline ─────────────────────────────
CREATE TABLE IF NOT EXISTS baseline_uncertainty (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rollout_id       TEXT REFERENCES rollouts(rollout_id) ON DELETE CASCADE,
    chunk_idx        INTEGER,
    n_samples        INTEGER,
    ms_var_vec       JSONB,
    ms_pair_l2       REAL,
    init_state_hash  TEXT,
    suite            TEXT
);

-- NOTE: no qc_rollouts / qc_eval tables. With the action/probe/sinks decomposition, PCP
-- collection is just a rollout carrying a pcp_chunks blob (rollouts.pcp_chunks_path; label =
-- rollouts.success), and the PCP 3-way eval is three ordinary rollouts rows
-- (method IN ('vanilla','pnp_only','pcp')). Corrector training data =
-- SELECT ... FROM rollouts WHERE pcp_chunks_path IS NOT NULL.

-- ── q_correctors: trained-corrector registry ────────────────────────────────
CREATE TABLE IF NOT EXISTS q_correctors (
    q_ckpt_id           TEXT PRIMARY KEY,
    run_id              UUID REFERENCES experiment_runs(run_id),
    experiment          TEXT,
    action_dim          INTEGER,
    obs_dim             INTEGER,
    correction_steps    JSONB,
    hard_lo             REAL,
    hard_hi             REAL,
    train_config        JSONB,
    ckpt_path           TEXT,      -- Storage: q_correctors/{q_ckpt_id}.pt
    -- eval / provenance metrics
    val_auc             REAL,
    val_pr_auc          REAL,
    val_brier           REAL,
    val_ece             REAL,
    per_suite_auc       JSONB,
    n_pos               INTEGER,
    n_neg               INTEGER,
    train_dataset_hash  TEXT,
    source_experiment   TEXT,
    split_path          TEXT,      -- Storage blob of train/val rollout_ids
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── verifier models and exact same-state candidate supervision ─────────────
CREATE TABLE IF NOT EXISTS verifier_models (
    verifier_id        TEXT PRIMARY KEY,
    experiment         TEXT,
    run_id             UUID REFERENCES experiment_runs(run_id),
    model_class        TEXT NOT NULL,
    obs_dim            INTEGER NOT NULL,
    action_dim         INTEGER NOT NULL,
    horizon            INTEGER NOT NULL,
    prefix_length      INTEGER,
    checkpoint_path    TEXT NOT NULL,
    split_path         TEXT NOT NULL,
    config_json        JSONB NOT NULL,
    metrics_json       JSONB NOT NULL,
    dataset_hash       TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verifier_candidate_groups (
    candidate_group_id TEXT PRIMARY KEY,
    experiment         TEXT NOT NULL,
    benchmark          TEXT NOT NULL,
    suite              TEXT NOT NULL,
    task_idx           INTEGER NOT NULL,
    episode_idx        INTEGER NOT NULL,
    chunk_idx          INTEGER NOT NULL,
    uncertainty_stratum TEXT NOT NULL,
    pairing_mode       TEXT NOT NULL, -- snapshot | paired_full_episode
    prefix_length      INTEGER NOT NULL,
    snapshot_validated BOOLEAN NOT NULL,
    metadata_json      JSONB,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verifier_candidates (
    candidate_id       TEXT PRIMARY KEY,
    candidate_group_id TEXT NOT NULL REFERENCES verifier_candidate_groups(candidate_group_id)
                         ON DELETE CASCADE,
    rollout_id         TEXT REFERENCES rollouts(rollout_id) ON DELETE SET NULL,
    candidate_kind     TEXT NOT NULL, -- default | fresh_noise
    success            BOOLEAN NOT NULL,
    n_steps            INTEGER,
    policy_chunk_path  TEXT NOT NULL,
    env_chunk_path     TEXT NOT NULL,
    observation_path   TEXT NOT NULL,
    metadata_json      JSONB,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(candidate_group_id, candidate_kind)
);
CREATE INDEX IF NOT EXISTS idx_verifier_candidates_group
    ON verifier_candidates(candidate_group_id);

-- ── encoding_cache: content-addressed obs+language encodings ─────────────────
CREATE TABLE IF NOT EXISTS encoding_cache (
    cache_key       TEXT PRIMARY KEY,
    model_revision  TEXT,
    obs_hash        TEXT,
    lang_hash       TEXT,
    dims            INTEGER,
    blob_path       TEXT,          -- Storage: encodings/{cache_key}.npz
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migration for databases created before PRO cohort membership was recorded.
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS canonical_member BOOLEAN;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS expanded_member BOOLEAN;
ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS generated_chunks_path TEXT;

-- Idempotent migration for databases created before per-iteration uncertainty was recorded.
ALTER TABLE pnp_action_vectors ADD COLUMN IF NOT EXISTS u_iter JSONB;
ALTER TABLE pnp_action_vectors ADD COLUMN IF NOT EXISTS u_iter_vec JSONB;
