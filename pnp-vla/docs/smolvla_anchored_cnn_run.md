# Anchored transformer and temporal CNN comparison

Run from `pnp-vla` with the existing native Mac virtual environment:

```sh
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python scripts/train_smolvla_anchored_local.py --device mps --family both
```

The runner uses the existing preaction cache for snapshot `671b5b211099997fc83d1277` and verifies its 1,280 training roots and 320 disjoint validation roots. It downloads only the already persisted combined manifest if no local copy exists. It does not rebuild the dataset or write to Supabase.

Both arms freeze the original scalar MC update-2,000 checkpoint, including its stock decoder readout. Their score is the frozen stock logit plus `h(context, stock, 100*delta) - h(context, stock, 0)`. The second term retains its parameter gradient, while the reference action is detached. Each final readout starts at zero, so update-zero reranking reproduces stock: 832/1,280 training successes and 209/320 validation successes.

The transformer uses one residual action encoder layer; the CNN uses four temporal convolutions. Each receives the same frozen 128-dimensional stock context, normalized stock action, and scaled normalized candidate delta. Context is computed once per root and shared across its nine candidates.

Each arm runs exactly 2,000 updates with ordinary candidate MC binary cross entropy, four uniformly sampled roots per update (36 candidates), AdamW learning rate `3e-4`, weight decay `1e-4`, and gradient clipping at 1. The two arms share root sampling seeds. This averages 6.25 presentations per training root; it is a fixed compute comparison, not a convergence claim. No pairwise objective, GAE, new data, or policy intervention is used.

Validation is logged every 250 updates. Atomic checkpoints every 100 updates include model, optimizer, full random states, history, architecture, dataset digest, baseline file hash, and the complete experiment contract. Rerun the same command to resume. `--family transformer` or `--family temporal_cnn` resumes one arm. A changed contract is rejected.

Results live under `~/pnp-vla-runs/checkpoints/smolvla_anchored_cnn_v1/671b5b211099997fc83d1277/`, with `latest.pt`, `history.json`, and `final_report.json` per family. Local MLflow uses `~/pnp-vla-runs/mlflow/tracking.db`; run links are printed for the existing UI at `http://127.0.0.1:5001`.

Fixed final reports cover all training and validation roots. They include BCE, nonstock BCE, the frozen stock contribution to total BCE, Brier score, selection rescues/spoils, macro and fresh-only pair accuracy, within-root logit spread, and per-root stock gradients with the reference held fixed. Gradient projections and signed rescue/spoil counts describe alignment with recorded outcomes; they do not establish true policy gradients or intervention success. The comparison is limited to the shared frozen stock readout, rather than the full VLA context.
