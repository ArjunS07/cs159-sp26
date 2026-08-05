-- Apply once to an existing database before collecting an experiment that needs per-iteration
-- uncertainty (the expanded LIBERO-PRO run). Idempotent; matches the tail of schema.sql.
--
-- u_iter     : (K-1,)            mean |a_hat_{i+1} - a_hat_i| per consecutive iteration pair
-- u_iter_vec : (K-1, action_dim) the same, per action dimension
--
-- pnp_euler_steps.u_mean / u_vec average the iteration axis away. These keep it, so
-- "does disagreement decay across the K perturbations?" is answerable from the database.
ALTER TABLE pnp_action_vectors ADD COLUMN IF NOT EXISTS u_iter JSONB;
ALTER TABLE pnp_action_vectors ADD COLUMN IF NOT EXISTS u_iter_vec JSONB;
