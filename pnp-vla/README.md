# pnp — Predict-and-Perturb / Predict-Correct-Perturb for pi0.5

Installable core for running the **pi0.5** VLA policy on **LIBERO / LIBERO-PRO** with
custom inference strategies — **Predict & Perturb (P&P)** and **Predict-Correct-Perturb
(PCP)** — logging every rollout to a hosted **Supabase** project. Inference runs in thin
Colab notebooks that `pip install -e` this package; analysis runs locally against Supabase.
The package is maintained in the `pnp-vla/` subdirectory of the larger `cs159-sp26`
repository.

## Layout
```
pnp/            importable core (config, env_setup, libero_env, libero_pro,
                models, sampler, pnp, pcp, rollout, store)
supabase/       schema.sql — canonical Postgres DDL + storage bucket
scripts/        backfill_legacy.py — trust-gated import of old SQLite DBs
notebooks/      thin Colab notebooks (the only place inference runs)
analysis/       local scripts: read Supabase -> tables/figures
```

## Colab bootstrap (private repo)
```python
import os
from google.colab import userdata
os.environ["SUPABASE_URL"]         = userdata.get("SUPABASE_URL")
os.environ["SUPABASE_SERVICE_KEY"] = userdata.get("SUPABASE_SERVICE_KEY")
os.environ["HF_TOKEN"]             = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"]        = userdata.get("WANDB_API_KEY")
GH_PAT = userdata.get("GH_PAT")
REPO_DIR = "/content/cs159-sp26"
GIT_REF = "main"  # Change to "main" after the refactor is merged.
![ -d "$REPO_DIR/.git" ] || git clone -q --branch "$GIT_REF" https://$GH_PAT@github.com/ArjunS07/cs159-sp26.git "$REPO_DIR"
!git -C "$REPO_DIR" fetch -q origin "$GIT_REF"
!git -C "$REPO_DIR" checkout -q "$GIT_REF"
!git -C "$REPO_DIR" pull -q --ff-only origin "$GIT_REF"
!pip install -q -e "$REPO_DIR/pnp-vla[sim]"
```

## Local analysis
```bash
cd pnp-vla
pip install -e '.[analysis]'
python -m analysis.run_analysis --experiment <label>
```

Required env vars: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (+ `HF_TOKEN`, `WANDB_API_KEY`,
`GH_PAT` in Colab).
