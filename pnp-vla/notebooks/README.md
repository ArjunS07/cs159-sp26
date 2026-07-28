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
- `10_confirm_verifier_v2.ipynb` — one-shot evaluation on the sealed PRO cohort.

Superseded verifier notebooks and workers are retained under `archive/verifier_v1/` for
provenance. They should not be used for new collection or model selection.

See the package README for the bootstrap cell.
