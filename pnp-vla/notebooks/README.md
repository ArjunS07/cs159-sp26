# Colab notebooks

Thin notebooks — the only place inference runs. Each sets secrets, clones the complete
`cs159-sp26` repository, installs its nested `pnp-vla/` package with `pip install -e`, then
calls drivers from `pnp`. Set `GIT_REF` to the branch or tag to execute; use `main` after the
refactor is merged.

Always use a fresh Colab GPU runtime. Both notebooks use the same checked clone/update/install
bootstrap, install the pinned `pnp-vla[sim]` stack, and preserve Colab's native
Torch/TorchVision/CUDA build. Their shared `setup_environment()` validation removes only
known-incompatible optional packages, verifies pi0.5/LIBERO imports, and keeps Hugging Face
model files on local `/content/hf_home` rather than Google Drive FUSE. If it reports a core
stack mismatch, restart with a fresh GPU runtime instead of upgrading Torch in place.

- `01_run_experiments.ipynb` — `run_controlled_slice()` (LIBERO slice) + `run_pro()` (LIBERO-PRO)
- `02_pcp_train_eval.ipynb` — `pcp_collect()` → `train_q_corrector()` → `pcp_eval()`
- LIBERO-PRO setup notebook — user-provided; feeds `pnp/libero_pro.py`

See the package README for the bootstrap cell.
