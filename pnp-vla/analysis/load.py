"""Pull DataFrames + Storage blobs from Supabase for analysis.

SELECTs (via supabase-py) into pandas: rollouts, pnp_euler_steps, pnp_action_vectors,
baseline_uncertainty, qc_eval — all filterable by experiment/method/refine_average. Fetches
a_hats .npz and PCP chunk parquet from the `artifacts` bucket for geometry/PC1 analyses.

TODO(phase 6): implement.
"""
