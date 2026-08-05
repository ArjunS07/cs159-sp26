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

## Expanded LIBERO-PRO (16 suites, K=5)

All 16 expanded-cohort suites at 20 episodes/task, one schedule at `K=5`, refine-last `(3,4)`, and
three configurations per identity: observed no-op with PCP telemetry, a 20-step matched-compute
control (`10 + 5x2`, honest at this K), and refine-last. Experiment
`pro-16suite-k5-steps34-v1`. Assets install per family automatically — `_swap`/`_task` from the
HuggingFace dataset, position-perturbation and distractor suites from the pinned git clone.

This run exists to test whether the **per-iteration** disagreement decay across the K
perturbations predicts correctability, so it records `pnp_action_vectors.u_iter` /
`u_iter_vec` (~112 bytes per probed step) rather than full `a_hats` blobs (~7 KB).

**Apply `supabase/migrations/004_u_iter.sql` before launching** — paste it into the Supabase SQL
Editor and run it. It is idempotent. Skip it and the first rollout's `pnp_action_vectors` insert
fails with an unknown-column error and the worker stops; the failure is loud, but it happens after
that rollout's `rollouts` row is upserted, so the identity would be skipped as already-done on
retry. If that happens, delete the experiment's rows before restarting:

```python
store.client.table('rollouts').delete().eq('experiment', 'pro-16suite-k5-steps34-v1').execute()
```

| Shard | Launcher |
| ---: | --- |
| 0 | [Open expanded worker 0 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro16_worker_0.ipynb) |
| 1 | [Open expanded worker 1 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro16_worker_1.ipynb) |
| 2 | [Open expanded worker 2 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro16_worker_2.ipynb) |
| 3 | [Open expanded worker 3 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro16_worker_3.ipynb) |
| 4 | [Open expanded worker 4 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro16_worker_4.ipynb) |
| 5 | [Open expanded worker 5 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/libero_pro16_worker_5.ipynb) |

Note `analysis/validate.py`'s `validate_pro` asserts the canonical 6-suite/600-identity shape, so
`run_analysis pro` rejects this experiment until a validator for it exists. Collection is
unaffected.

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
