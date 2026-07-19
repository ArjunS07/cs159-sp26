# Colab rollout workers

These are stable launchers for stock LIBERO and the canonical LIBERO-PRO collection. Mutable
experiment logic lives in `pnp.experiments`; every launcher pulls `main` before importing it.
Do not copy rollout logic into these notebooks.

## Stock LIBERO (completed)

| Shard | Launcher |
| ---: | --- |
| 0 | [Open worker 0 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_worker_0.ipynb) |
| 1 | [Open worker 1 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_worker_1.ipynb) |
| 2 | [Open worker 2 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_worker_2.ipynb) |
| 3 | [Open worker 3 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_worker_3.ipynb) |
| 4 | [Open worker 4 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_worker_4.ipynb) |
| 5 | [Open worker 5 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_worker_5.ipynb) |

## Canonical LIBERO-PRO

The PRO workers install the official six-suite assets automatically, assert a 600-identity
manifest, and run three pi0.5 configurations per identity: shared observed/PCP telemetry at
steps 1–9, a 16-step matched-compute control, and refine-last `(4,5)` with `K=3`. Each worker
owns 100 identities and 300 rollouts; all six together produce 1,800 rollouts under experiment
`libero-pro-canonical-core-k3-v1`.

| Shard | Launcher |
| ---: | --- |
| 0 | [Open PRO worker 0 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro_worker_0.ipynb) |
| 1 | [Open PRO worker 1 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro_worker_1.ipynb) |
| 2 | [Open PRO worker 2 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro_worker_2.ipynb) |
| 3 | [Open PRO worker 3 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro_worker_3.ipynb) |
| 4 | [Open PRO worker 4 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro_worker_4.ipynb) |
| 5 | [Open PRO worker 5 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro_worker_5.ipynb) |

Before launch, add `GH_PAT`, `HF_TOKEN`, `SUPABASE_URL`, and `SUPABASE_SERVICE_KEY` to Colab
Secrets and grant notebook access. Run all cells in every worker that Colab allows concurrently.

All six workers use `SHARD_COUNT=6` and distinct indices. Running only a subset is safe but does
not reassign the absent workers' identities. If a runtime terminates, reopen the same worker and
Run all; deterministic rollout IDs skip completed work. Never change `SHARD_COUNT` after collection
starts unless the database is flushed and the experiment is restarted.

Regenerate the launchers after an intentional bootstrap change with:

```bash
python scripts/generate_colab_workers.py
```
