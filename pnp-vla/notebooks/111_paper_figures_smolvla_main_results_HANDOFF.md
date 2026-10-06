# Notebook 111 handoff: SmolVLA paper figures

`111_paper_figures_smolvla_main_results.ipynb` is a zero-simulation analysis notebook for the main SmolVLA/LIBERO paper figures. It reads completed rollout metadata and saved uncertainty artifacts through `SupabaseStore`; the underlying rollout data are not embedded in the notebook.

## Evaluation cohort

- 400 exact-matched standard-LIBERO identities: 40 tasks across `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, using state indices 0--9 (100 episodes per suite).
- Exact identity key: `suite`, `task_idx`, `episode_idx`, and `init_state_hash`.
- Every plotted arm must contain exactly 400 completed, unique identities and the notebook rejects cohort mismatches.
- Policy checkpoint: `HuggingFaceVLA/smolvla_libero` through the repository's SmolVLA evaluation stack.

## Rollout sources

The notebook resolves the full method/configuration hash with the repository builders and queries the `rollouts` table by experiment, method, config hash, and `status='completed'`.

| Figure label | Experiment ID | Builder |
|---|---|---|
| Stock, 1 action | `smolvla-libero-stock-a1-v1` | `build_smolvla_a1_method()` |
| Stock, 10 actions | `smolvla-libero-a10-pnp-k5-steps34-v1` | first arm of `build_smolvla_libero_methods()` |
| Refinement, 1 action | `smolvla-libero-a1-pnp-steps123-k311-v1` | `build_smolvla_a1_pnp_method()` |
| Refinement, 10 actions | `smolvla-libero-a10-pnp-steps123-k311-v1` | `build_smolvla_schedule_method()` |
| Naive average | `smolvla-libero-a10-blend-ablation-v1` | first arm of `build_smolvla_blend_ablation_methods()` |
| Refined aggregation | `smolvla-libero-a10-consensus-project-s03-k3-v1` | `build_smolvla_consensus_s03_method()` |

Refinement uses the P&P schedule `steps=(1,2,3)` with comparison counts `K=(3,1,1)`. Refined aggregation averages the stock and refined chunks, perturbs/projects the average at the repository's `s=0.3` setting, performs three projection refinements, and executes ten actions.

## Derived analyses

- Per-suite transitions pair each refinement or aggregation rollout with stock 10-action execution on the exact identity key. `F->S` and `S->F` are episode-outcome changes, not action-level labels.
- The uncertainty figures use artifacts from the 10-action refinement arm. For each chunk and probed flow step, `U10` is the mean uncertainty over the first ten action positions. Values are averaged across episode chunks and then weighted by the `(3,1,1)` number of P&P comparisons.
- Failure is the positive ROC class. AUCs are reported per suite (`n=100`) and pooled (`n=400`) with 3,000 bootstrap draws. This is retrospective episode-level uncertainty, not an online gate.

## Outputs and execution

Run the notebook top to bottom with the analysis extras and Supabase credentials configured. It writes PNG, PDF, SVG, CSV provenance/count tables, and a decoded uncertainty cache under `paper_figures_smolvla/`. The main exports are:

- `smolvla_main_sr_400.*`
- `smolvla_sr_by_suite.*`
- `smolvla_transitions_by_suite.*`
- `smolvla_u10_failure_auc_by_suite.*`
- `smolvla_u10_failure_roc.*`

The notebook does not launch LIBERO environments or require a GPU; the first uncached uncertainty run downloads the stored `.npz` artifacts.
