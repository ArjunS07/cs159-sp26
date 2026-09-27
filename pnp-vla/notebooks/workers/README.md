# Colab rollout workers

## SmolVLA success Q10 and RL Token program (prepared, not launched)

Notebooks 109–114 pull branch `codex/smolvla-q-program` and contain only setup and
experiment parameters; implementation lives under `pnp/`. Set Colab Secrets
`GH_PAT`, `HF_TOKEN`, `SUPABASE_URL`, and `SUPABASE_SERVICE_KEY` before use.
Notebook 109 requires **800 complete v4 fresh8 trees** and freezes one snapshot
before training. The root-MC and tree-TD arms use the same roots and architecture;
the training target differs. Compare them on the same held-out roots and run a
nested root-count learning curve before relying on either Q.

| Notebook | Role | Interpretation |
| --- | --- | --- |
| `109_smolvla_success_q10_fresh8_train.ipynb` | Train success Q10, root MC or tree TD | P&P proposal and P&P continuation distribution |
| `110_smolvla_q10_pnp_gate_offline.ipynb` | Paired accept/reject audit on recorded candidates | Offline one-intervention diagnostic, not deployed success |
| `111_smolvla_contextual_rl_token.ipynb` | Extract contextual VLM states, train reconstruction bottleneck, train root-MC Q ablation | Genuine RL Token representation pilot; source roots only |
| `112_smolvla_q10_vanilla_rerank_transfer.ipynb` | Simulator evaluation over ordinary SmolVLA draws | Deployment uses no P&P; Q is still trained on P&P continuation, so this is a transfer test |
| `113_smolvla_q10_pcp_gradient_diagnostic.ipynb` | Bounded offline action-gradient proposal | Score/displacement smoke only; no corrected-action outcome |
| `114_smolvla_q10_latent_guidance_transfer.ipynb` | Live Q-gradient correction during flow denoising | Simulator pilot against paired ordinary-policy stock; not a trained PCP MLP |

The final independent ordinary-policy reranker needs **ordinary-policy training
proposals and ordinary-policy continuation outcomes**. Fresh8 does not provide
those labels. A full PCP claim needs simulator execution of corrected chunks and
paired controls. The RLT ablation needs live contextual-token extraction before
its Q can be deployed; notebook 112 accepts the existing prefill-prefix Q only.
All expensive cells are opt-in. Notebook 112 defaults to one evaluation identity
per shard. Nothing is launched by creating these notebooks.

These are stable launchers for stock LIBERO and the canonical LIBERO-PRO collection. Mutable
experiment logic lives in `pnp.experiments`; every launcher pulls `main` before importing it.
Do not copy rollout logic into these notebooks.

## SmolVLA fresh8 Q10 trees (not launched)

Notebook 108 uses the 800 completed standard-LIBERO source episodes at indices 10–29. It
reuses the frozen depth-1 root manifest: 65% high-U10 roots, 35% uniform roots, with one root
per source episode. Every v4 tree has the exact stored source continuation plus eight
fresh-initial-noise P&P alternatives. At roots with a complete v3 tree, the first four fresh
branches reuse their persisted outcomes and Bellman artifacts; the worker generates and executes
only fresh seeds 5–8. Other roots generate and execute all eight. Each new alternative
executes ten actions and then follows the same frozen P&P continuation. The old v3 trees and
artifacts are untouched; v4 candidate records link to the reused artifacts with provenance.

An L4 is known to run the previous eight-candidate tree collector. An A100 is not required;
T4 fit and speed remain unmeasured. Choose a GPU runtime, grant `GH_PAT`, `HF_TOKEN`, `SUPABASE_URL`, and
`SUPABASE_SERVICE_KEY` notebook access, and run one worker with `TREE_LIMIT = 1` first. Inspect
the stored group and the printed GPU peak memory/tree time. Set `TREE_LIMIT = None` and rerun
the same worker for its full 266- or 267-root shard. Each completed tree is persisted; an interrupted
worker can be restarted with the same shard index. New candidates are generated in a four- or
eight-lane batch; simulator rollouts remain sequential because each branch must start from the exact
saved MuJoCo state. The A100 may therefore be underused; compare completed-tree time and Colab
credit consumption in the smoke run before choosing a GPU. Three independent workers allow
parallelism across separate Colab runtimes when available. Keep the three-shard assignment
fixed after collection begins.

| Shard | Launcher |
| ---: | --- |
| 0 | [Open fresh8 worker 0 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/108_smolvla_fresh8_tree_worker_0.ipynb) |
| 1 | [Open fresh8 worker 1 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/108_smolvla_fresh8_tree_worker_1.ipynb) |
| 2 | [Open fresh8 worker 2 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/108_smolvla_fresh8_tree_worker_2.ipynb) |

This 800-root cohort supports a within-distribution learning curve. A later independent test
of new initial states should collect source episodes from different episode indices and freeze
their roots before looking at their branch outcomes. Keep those identities out of model
selection on the 10–29 cohort.

## SmolVLA standard LIBERO A10 pilot

Notebook 83 restores SmolVLA to the current unified rollout framework and covers the standard
400-identity slice (four suites x ten tasks x ten initial states) in two fixed shards. Each
identity runs two arms: exact stock actions with measurement-only P&P telemetry, and always-on
refine-last P&P. Both arms use 10 Euler integration steps, generate 50 actions, and execute only
the first 10 before replanning. K=5 probes run at zero-based Euler steps (3,4), with U10/U20/U50,
contraction, and per-action-dimension uncertainty persisted. Videos and observation frames are
off; progress prints every 10 completed matched identities.

| Shard | Launcher |
| ---: | --- |
| 0 | [Open SmolVLA worker 0 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/83_smolvla_libero_a10_worker_0.ipynb) |
| 1 | [Open SmolVLA worker 1 in Colab](https://colab.research.google.com/github/ArjunS07/cs159-sp26/blob/main/pnp-vla/notebooks/workers/83_smolvla_libero_a10_worker_1.ipynb) |

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

## Q-Planning critic evaluation

- `66_eval_qplanning_q10_pro220.ipynb`: one Q10 worker, 220 rollouts.
- `67_eval_qplanning_q50_pro220.ipynb`: one Q50 worker, 220 rollouts.

Each uses 64 candidates, top 16 Q-weighted averaging, 3 decode steps, and 10 executed actions.
The workers print exact episode-matched historical stock comparisons every 25 completions.
Run them independently on two GPU runtimes.

### Untouched position-perturbation confirmation

- 68_eval_qplanning_heldout160_worker_0.ipynb
- 68_eval_qplanning_heldout160_worker_1.ipynb
- 68_eval_qplanning_heldout160_worker_2.ipynb
- 68_eval_qplanning_heldout160_worker_3.ipynb

These four fixed shards cover the reserved 160-row position-perturbation PRO manifest. Each
worker runs 40 identities under three newly collected, exactly matched arms: stock PI0.5, Q10,
and Q50 (120 rollouts/worker). The critics stay frozen; this is not continual learning. Progress
prints every 10 complete three-arm identities. Videos, frames, and generated chunks are off.

The older six-worker launchers use `SHARD_COUNT=6` and distinct indices. Running only a subset is safe but does
not reassign the absent workers' identities. If a runtime terminates, reopen the same worker and
Run all; deterministic rollout IDs skip completed work. Never change `SHARD_COUNT` after collection
starts unless the database is flushed and the experiment is restarted.

Regenerate the launchers after an intentional bootstrap change with:

```bash
python scripts/generate_colab_workers.py
```
