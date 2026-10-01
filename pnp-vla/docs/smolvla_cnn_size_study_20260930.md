# Local CNN capacity and stopping study

User authorized these studies on the local GPU on September 30, 2026. The
runner is `scripts/run_smolvla_cnn_size_study.py`; it holds an exclusive study
lock and waits for the existing continuation GPU lock before MPS computation.
No Modal jobs, Colab operations or additional data collection are involved.

## Matched matrix

| Temporal CNN width | Trainable parameters | Initialization seeds |
|---|---:|---|
| 32 | 28,736 | 42, 123, 314 |
| 64 | 104,576 | 42, 123, 314 |
| 128 | 397,568 | 42, 123, 314 |

All nine arms start with a fresh zero-output residual and the same frozen
original scalar MC baseline at update 2,000, SHA
`86ddaa7f5cf7fb190a18d88e8a4b1b336a7b57edd2791fcb12b4e0b1a060e597`.
The baseline is outside the trainable parameter counts. The action encoder
retains four residual temporal convolutions with kernel size three, learned
positions, scaled stock-relative action differences, and a trainable context
projection/readout. Both h(candidate) and h(reference) retain parameter gradients.

Each arm receives exactly 8,000 updates, four roots per batch and all nine
candidates, using identical root indices across sizes and seeds. This is 25
average root presentations. The split remains 1,280 training / 320 validation
roots from snapshot `671b5b211099997fc83d1277`. Full proposed actions and masks
based only on the known remaining action budget are used.

The only objective is ordinary root MC binary cross-entropy. AdamW is fresh in
every arm: cosine LR 3e-4 to 3e-5 over the fixed budget, weight decay 1e-4,
gradient norm clipping at one. No ranking loss, TD or GAE is added to this
capacity comparison. The running recorded-policy TD experiment remains separate.

The frozen baseline's stock readout is computed once per root and cached. Live
score checks verify the cache; a focused test verifies scores and every residual
parameter gradient against direct evaluation. No residual features are frozen.

## Chronological stopping study

Evaluate training and validation at update zero and every 250 updates. The sole
selection criterion is nonstock validation BCE: reset patience after an
improvement greater than .001, otherwise count one miss. After eight misses,
freeze the selected checkpoint. Update zero is eligible. Results after the
first stopping event cannot revise that choice. Report selected update and
would-stop update separately; an untriggered stop means the budget was exhausted.

Training continues to update 8,000 to obtain the matched final endpoint. At both
the fixed final and rule-selected endpoints, report train/validation success
selection versus stock, rescues, spoils, all/fresh within-root pair accuracy,
BCE, Brier and fixed-reference action-gradient projection diagnostics. Gradient
projections against recorded outcomes are not ground-truth useful derivatives.

The 320 validation roots have been repeatedly inspected. Selection by this
rule, cross-width comparisons and any small performance gains are exploratory;
they are not independent estimates of deployment improvement. The earlier
10,000-update CNN is not the matched width-128 control because its initialization
history and LR differ.

## Execution and recovery

- MLflow: http://127.0.0.1:5001/#/experiments/17
- Log: `/Users/arjunsharma/pnp-vla-runs/diagnostics/cnn_size_study_v1.log`
- Output: `/Users/arjunsharma/pnp-vla-runs/checkpoints/smolvla_cnn_size_study_v1/671b5b211099997fc83d1277`
- Results: `comparison.json`, `results.md`, and per-arm `history.json` / `final_report.json`.
- Each arm saves latest state every 100 updates and at evaluations. Checkpoints
  include optimizer, history, early-stopping state and CPU/MPS/NumPy/Python RNG.
- Rule-selected versions are immutable, so interruption between writing an
  improvement and committing history cannot destroy the previous choice.
- Re-running the same runner resumes the fixed matrix and skips finished arms.
  The study lock prevents duplicate queued workers.

Eight focused tests passed before launch, including unchanged cached parameter
gradients, exact CPU optimizer recovery and stopping without hindsight.
