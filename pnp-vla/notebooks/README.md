# Colab notebooks

Thin notebooks — the only place inference runs. Each: set secrets, clone + `pip install -e`
this repo, then call drivers from `pnp`.

- `01_run_experiments.ipynb` — `run_controlled_slice()` (LIBERO slice) + `run_pro()` (LIBERO-PRO)
- `02_pcp_train_eval.ipynb` — `pcp_collect()` → `train_q_corrector()` → `pcp_eval()`
- LIBERO-PRO setup notebook — user-provided; feeds `pnp/libero_pro.py`

See the repo README for the bootstrap cell. TODO(phase 7): author the notebooks.
