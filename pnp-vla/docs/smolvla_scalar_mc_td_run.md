# Matched scalar MC and TD pilot

Launch: ` .venv/bin/python -u scripts/train_smolvla_scalar_returns_local.py `.
Runs on the Mac Metal GPU; no simulator or VLA inference is required.

## Data and target

The frozen 1,600-root snapshot retains its 1,280 / 320 split. Training uses
three trajectories per training root: the original source episode and the
predeclared fresh-noise branches 1 and 5. Both branch intervention roots and
their continuations are included. The other six fresh branches are not used
for training in this pilot. Evaluation uses all nine candidates at each of
the same 320 held-out roots. These are already-used validation roots, not a
new untouched test set.

Context features are reused from the original complete trajectory cache.
Action contents are restored from the full proposed `bellman/action` arrays.
Masks depend only on the known remaining episode budget, never observed
success or termination. Terminal partial chunks retain all proposed actions.
Rewards and termination enter targets only. Remaining budget is an input.

MC target is the trajectory's final binary success. TD target is
`r + discount * sigmoid(target_Q(next_state, next_proposed_chunk))`, with
gamma 1 and discount zero on terminal or deadline transitions. This is
recorded-policy SARSA evaluation, not a maximum over candidate actions.
Both targets use binary cross entropy on a scalar logit. TD uses detached soft
probability targets. No pairwise ranking objective or GAE is used.

## Matched configuration

- Two decoder layers, width 128, four heads, FFN 512, dropout zero.
- Full frozen context prefix pooled to 128 tokens, plus robot and proprioception.
- Absolute proposed actions normalized using training roots only.
- One learned readout token and a single sigmoid output: 659,749 parameters.
- Same seed and deterministic minibatch membership for both arms.
- Uniform training root, then one of its three trajectories, then a window.
- Batch 32, 2,000 updates, AdamW peak learning rate 0.0001, weight decay 0.0001.
- 100-update warmup, cosine decay, gradient clipping at 1.
- TD target parameter EMA rate 0.005; initial target equals initial critic.
- Held-out evaluation and checkpoint every 250 updates, including update zero.

## Outputs

Local files are under `~/pnp-vla-runs/checkpoints/smolvla-q10-scalar-mc-td-v1-preaction/`.
The preparation cache resumes per trajectory. The trainer atomically saves
`latest.pt` every 50 updates with the critic, target critic, optimizer, random
state, update, history and MLflow run ID. Running the same command restores
that checkpoint automatically, resumes the same MLflow run and skips completed
arms. Abrupt power loss can require repeating at most 49 completed updates.
Resume rejects a changed dataset, architecture or training configuration.
MLflow experiment 7 is served at <http://127.0.0.1:5001/#/experiments/7>.
macOS Control Center occupies port 5000, so use port 5001 even if an initial
running-process log printed an older port. No model upload is performed.

Report fixed final-update success selection versus stock, rescues, spoils,
pair accuracy and Brier score. One seed and repeatedly inspected validation
roots make this a diagnostic comparison, not a definitive model selection.
One-step TD can propagate sparse reward slowly; 2,000 updates are a controlled
budget, not evidence that either method has converged.
