# Colab notebooks

Thin notebooks — the only place inference runs. Each sets secrets, clones the complete
`cs159-sp26` repository, installs its nested `pnp-vla/` package with `pip install -e`, then
calls drivers from `pnp`. Set `GIT_REF` to the branch or tag to execute; use `main` after the
refactor is merged.

Always use a fresh Colab GPU runtime. The notebooks use the same checked clone/update/install
bootstrap, install the pinned `pnp-vla[sim]` stack, and preserve Colab's native
Torch/TorchVision/CUDA build. Their shared `setup_environment()` validation removes only
known-incompatible optional packages, verifies pi0.5/LIBERO imports, and keeps Hugging Face
model files on local `/content/hf_home` rather than Google Drive FUSE. If it reports a core
stack mismatch, restart with a fresh GPU runtime instead of upgrading Torch in place.

- `01_run_experiments.ipynb` — `run_controlled_slice()` (LIBERO slice) + `run_pro()` (LIBERO-PRO)
- `08_collect_verifier_v2_pro.ipynb` — canonical trajectory-seeded V2 collection notebook;
  normally run its three generated workers instead.
- `09_train_state_conditioned_verifier_v2.ipynb` — conditioned architecture sweep and final
  checkpoint-bundle registration.
- `10_train_hybrid_chunk_critic.ipynb` — twin 50-step IQL critic, distilled twin 10-step
  candidate critic, causal ranking sweep, controls, and checkpoint registration.
- `56_pcp_critic_preflight.ipynb` — creates a frozen train-eligible PCP critic snapshot.
- `57_train_pcp_critic.ipynb` — trains a new Cal-QL or IQL PCP critic checkpoint.
- `58_pcp_critic_offline_eval.ipynb` — reads a registered checkpoint for offline diagnostics.
- `59_pcp_search_adapter_smoke.ipynb` — verifies the guarded, offline-only adapter contract.
- `workers/60_five_step_diversity_pro220_worker_{0,1}.ipynb` — paired three-arm pilot on the
  frozen 220-identity PRO cohort: five-step x1, five-step x3 lowest-U20, and explicit
  select-then-refine with 10 executed actions per 50-action prediction.
- `61_analyze_five_step_diversity_pro220.ipynb` — exact matched SR and per-suite deltas
  versus historical 10-step stock, plus all three candidate-pair cosine/L2 diagnostics
  for the first 10 and full 50 actions; supports interim previews and strict final analysis.
- `workers/62_coarse_single_refinement_pro220_worker_{0,1}.ipynb` — single-query follow-up:
  five-step refine `(2,3)`, three-step refine `(2,)`, and three-step no refinement, all K=5
  and 10 executed actions. Two 110-identity/330-rollout shards; exact matched historical
  comparisons every 25 completed three-arm identities and at the end.
- `63_analyze_coarse_single_refinement_pro220.ipynb` analyzes the exact seven-arm matched
  cohort for workers 62, including
  paired/per-suite SR, U10/U20/U50, PnP contraction, failure detection, and new-arm compute.
  It downloads uncertainty-profile blobs only and needs no GPU or simulator.
- `11_arbitrate_and_confirm_verifier.ipynb` — development-only baseline/hybrid selection,
  followed by one-shot evaluation of exactly one eligible winner on the sealed PRO cohort.

- `16_analyze_expanded_pro.ipynb` — zero-GPU Supabase analysis for expanded PRO workers 0–5,
  including paired rollout results and consecutive-uncertainty contraction.

- `17_train_diverse_pi05.ipynb` — create one shared task-stratified episode-bootstrap manifest
  and full-fine-tune member 0 or 1 on an 80GB A100/H100.
- `workers/18_diversity_signal_model_{0,1}.ipynb` — matched 13-suite PRO signal workers for the
  two trained models (K=5 uncertainty plus exact first-decision chunks).
- `19_analyze_diversity_signal.ipynb` — paired complementarity, uncertainty-selection, and
  first-action diversity analysis for the two model workers.

Superseded verifier notebooks and workers are retained under `archive/verifier_v1/` for
provenance. They should not be used for new collection or model selection.

See the package README for the bootstrap cell.
