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
    obs_frames_path           TEXT,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rollouts_experiment ON rollouts(experiment);
CREATE INDEX IF NOT EXISTS idx_rollouts_run        ON rollouts(run_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_filter     ON rollouts(experiment, method, refine_average, suite);

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
    mm_bc_pc1    REAL
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

-- ── qc_rollouts: PCP collection (heavy data in Storage) ─────────────────────
CREATE TABLE IF NOT EXISTS qc_rollouts (
    rollout_id       TEXT PRIMARY KEY,
    run_id           UUID REFERENCES experiment_runs(run_id),
    experiment       TEXT,
    suite            TEXT,
    task_idx         INTEGER,
    episode_idx      INTEGER,
    init_state_hash  TEXT,
    success          BOOLEAN,
    n_chunks         INTEGER,
    chunks_path      TEXT,      -- Storage pointer to per-chunk obs_enc + z_hat parquet
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── qc_eval: PCP 3-way eval (lambda: -1 vanilla / 0 pnp-only / 3.0 pcp) ──────
CREATE TABLE IF NOT EXISTS qc_eval (
    rollout_id   TEXT NOT NULL,
    lambda       REAL NOT NULL,
    run_id       UUID REFERENCES experiment_runs(run_id),
    experiment   TEXT,
    suite        TEXT,
    task_idx     INTEGER,
    episode_idx  INTEGER,
    success      BOOLEAN,
    PRIMARY KEY (rollout_id, lambda)
);

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
