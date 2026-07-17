# pnp — Predict-and-Perturb / Predict-Correct-Perturb for pi0.5

Installable core for running the **pi0.5** VLA policy on **LIBERO / LIBERO-PRO** with
custom inference strategies — **Predict & Perturb (P&P)** and **Predict-Correct-Perturb
(PCP)** — logging every rollout to a hosted **Supabase** project. Inference runs in thin
Colab notebooks that `pip install -e` this package; analysis runs locally against Supabase.

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
!git clone https://{GH_PAT}@github.com/<you>/pnp-vla.git
!pip install -q -e ./pnp-vla[sim]
```

## Local analysis
```bash
pip install -e '.[analysis]'
python -m analysis.run_analysis --experiment <label>
```

Required env vars: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (+ `HF_TOKEN`, `WANDB_API_KEY`,
`GH_PAT` in Colab).
