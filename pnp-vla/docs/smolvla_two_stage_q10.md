# SmolVLA Q10 trajectory pretraining and root ranking

This experiment asks whether extra recorded trajectories improve *within-state*
action ranking. It uses the immutable 1,600-tree fresh8 snapshot
`671b5b211099997fc83d1277` and the existing 1,280/320 root split.
All data derived from a held-out root's source episode or nine branches stays
out of training.

## Data audit

- The 1,280 training roots have 11,520 recorded candidate outcomes. Only 368
  roots have both successful and failed candidates; 65 can rescue failed stock.
- The 320 validation roots have 84 mixed trees, 17 rescueable stock failures,
  209 stock successes and a hindsight oracle of 226 successes. This oracle is
  an optimistic bound for the *recorded* candidate set; each branch was run
  once and its realized outcome is noisy.
- Every training root has one full SmolVLA source episode. Two fixed
  fresh-noise branches (seeds 1 and 5) are also selected before training.
  Branch root windows are excluded from pretraining, since the root stage
  compares all nine candidates. Branches that terminate within the first
  ten actions have zero continuation windows.
- The stored state features are frozen SmolVLA `embed_prefix` embeddings,
  pooled to at most 128 tokens. They are **not** contextualized VLM final
  hidden states. Actions are 10 by 7 chunks. Source and branch artifacts
  differ in prefix shape and step-index origin; the cache normalizes both.
- The 1,886 training-ready PCP rollouts in Supabase use a pi0.5 checkpoint.
  Their visual features are not mixed into this SmolVLA experiment.

## Training contract

Stage 1 samples initial roots uniformly, then one available trajectory
uniformly, then a decision window uniformly. It predicts eventual episode
success with binary cross entropy. This supplies broader state and action
coverage, but a single trajectory per state cannot by itself identify an
action advantage; the network may learn mostly episode difficulty.

Stage 2 samples half of each root batch from mixed-outcome roots and half from
all roots. Its loss is

```text
root BCE + 0.2 * mean mixed-root softplus(-(Q_positive-Q_negative)/0.1)
         + 0.25 * trajectory replay BCE.
```

The pairwise term cancels a state-only score within each root. Trajectory
replay guards against forgetting. A control starts from random weights and
uses the same stage-2 batches, objective, and replay. The previous 2,000-step
root-only BCE checkpoint is a historical reference.

Both new runs use the same fixed seed, action normalization from training
roots only, and no checkpoint selection on validation. MLflow records data
counts, losses, gradient norms, reranking outcomes, pair accuracy, score
spread, and cohort-specific results. The training script now saves weight
snapshots at every 250-update stage-two validation point; a run started before
that addition retains its stage-one and final checkpoints plus MLflow metrics.
Repeated use of the same 320 validation
roots makes any observed improvement exploratory; a later untouched cohort
would be needed for a confirmatory claim.

## Local commands

```bash
PYTHONDONTWRITEBYTECODE=1 pnp-vla/.venv/bin/python pnp-vla/scripts/train_smolvla_two_stage_local.py --prepare-only --cache-workers 8
PYTHONDONTWRITEBYTECODE=1 pnp-vla/.venv/bin/python pnp-vla/scripts/train_smolvla_two_stage_local.py
PYTHONDONTWRITEBYTECODE=1 pnp-vla/.venv/bin/python pnp-vla/scripts/train_smolvla_two_stage_local.py --skip-pretrain
```

The SQLite MLflow store, artifacts and checkpoints live under
`~/pnp-vla-runs`. The local UI serves at `http://127.0.0.1:5000` when
`mlflow server` is running against that database.

## Smaller-capacity comparison

The same script accepts `--model-scale small`. This uses one decoder block,
width 128, four attention heads and FFN width 512, for 408,073 trainable
parameters. The original uses three blocks, width 256, eight heads and FFN
width 1024. Both keep dropout 0.20, the 101-bin expected-value head, the same
cached data, seed, training schedule and root split. The small model's runs
use a separate MLflow experiment and checkpoint directory. Run it with and
without `--skip-pretrain` to measure the contribution of trajectory
pretraining at the new capacity. Comparisons on the already-used validation
split are exploratory.

At the fixed 1,000-update fine-tuning endpoint, the small two-stage model
selects 201/320 successful chunks versus 209/320 for stock, with held-out
pair accuracy 0.496 and Brier 0.097. Its matched no-pretraining control
selects 195/320, with pair accuracy 0.464 and Brier 0.155. The original
two-stage model selects 201/320, with pair accuracy 0.565 and Brier 0.092.
Thus pretraining helps the small model relative to its control, but reducing
capacity does not improve reranking. All three final models trail stock.

The original two-stage model has training-root pair accuracy 0.498 and Brier
0.094; the small model has 0.499 and 0.104, respectively. These train-versus-
validation comparisons provide no evidence that excess capacity is the main
reason action ranking fails. They instead suggest that the current features
or losses emphasize state success over within-state action effects. The
small final model's gradient at failed stock chunks points toward a recorded
successful alternative on only 8/17 rescueable validation roots; the
diagnostic is in `action_gradients_small.json` under the local run directory.

## Action-gradient diagnostic for PCP

`scripts/diagnose_smolvla_q_gradients_local.py` differentiates the expected
success score with respect to each validation candidate's 10 by 7 action
chunk. It reports gradient norms separately for successful and failed
candidates *within mixed roots*, and asks whether the gradient at a failed
chunk points toward the nearest recorded successful chunk from the same
state. The 17 validation roots where stock fails but an alternative succeeds
receive a separate alignment measure. Gradients are compared in
training-standardized action coordinates. Positive alignment is necessary
but not sufficient for PCP: interpolation can leave the policy distribution,
and the recorded outcome is only one rollout per candidate.
