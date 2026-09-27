# SmolVLA Q10 critic on Modal

This runs the tree-only experiment from notebook 107 on one NVIDIA T4. It uses
the existing 480-tree v3 snapshot in Supabase. Cache preparation runs on CPU;
training and held-out root scoring use the T4. It does not load SmolVLA or run
new LIBERO episodes.

## Setup

From a fresh machine, clone the repository and enter its root:

```bash
git clone git@github.com:ArjunS07/cs159-sp26.git
cd cs159-sp26
```

The Modal image uploads the local `pnp-vla/pnp` code, including local edits;
it does not upload `.env`, notebooks, or model weights. Run from a checkout
with the same success-Q code as this launcher.

Create a Modal secret from the ignored local `.env` containing
`SUPABASE_URL` and `SUPABASE_SERVICE_KEY`:

```bash
modal secret create pnp-supabase --from-dotenv .env
```

This secret was already created in the `arjuns07` workspace on September 25,
2026. Do not repeat the command unless the Supabase credentials change. No
GitHub or Hugging Face token is needed to train from the cached tree artifacts.

## Run

```bash
# CPU only: download/validate 4,320 branch artifacts once.
modal run pnp-vla/scripts/modal_smolvla_success_q10.py --phase prepare

# CPU only: pack the 20 GiB cache into one archive for fast T4 local staging.
modal run pnp-vla/scripts/modal_smolvla_success_q10.py --phase pack

# Two updates per arm, then score the fixed validation roots.
modal run pnp-vla/scripts/modal_smolvla_success_q10.py --phase pilot

# 2,000 updates per arm, sequential T4 jobs; run after inspecting the pilot.
modal run --detach pnp-vla/scripts/modal_smolvla_success_q10.py --phase full

# Re-score saved full checkpoints without retraining.
modal run pnp-vla/scripts/modal_smolvla_success_q10.py --phase audit
```

The `pilot` and `full` phases automatically run the idempotent prepare and
pack steps first. Use `--arms root_mc` or `--arms tree_td` to run one arm. A pilot uses a
separate checkpoint directory, so it cannot be mistaken for or resumed as a
full training run. Repeating the full command resumes saved full checkpoints.
Each T4 job has a four-hour timeout. Cache preparation has a two-hour timeout.

For a bounded diagnosis on existing v3 data, set `--tree-limit 256` and
`--updates 1000`; this creates a distinct immutable snapshot key. The launcher
also accepts `--dataset fresh8 --tree-limit 800` after all 800 v4 trees are
complete. That dataset uses its own snapshot and candidate order: stored
source plus fresh seeds 1–8. `--train-root-limit 100`, `200`, or `400` selects
nested training subsets while retaining the same full-snapshot validation
roots. Reports for different scales have distinct names.

The persistent Modal Volume is `smolvla-success-q10`. The packed cache is under
`packed/`. Reports are under
`reports/`; checkpoints are under `runs/full/<snapshot digest>/<arm>/`.

```bash
modal volume ls smolvla-success-q10 reports
modal volume get smolvla-success-q10 reports ./modal-reports
```

The audit reports selected successes versus the stored-source candidate,
rescues, spoils, within-tree ordering, Brier score, and the observed oracle
ceiling. It estimates one intervention followed by the frozen P&P continuation.
A fresh branch audit and closed-loop rollouts are still needed to claim a
policy-level gain.

## Run on September 25, 2026

The immutable snapshot digest was `fc647fad3a438bc65c066a59`: 384 training
roots and 96 validation roots. Both full arms completed 2,000 updates on a
Tesla T4. The final report is
`reports/full_fc647fad3a438bc65c066a59.json` in the Volume.

| Choice on the same 96 held-out roots | Observed successes | Rescues | Spoils | Pair accuracy | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stored source | 66 | — | — | — | — |
| Root MC | 60 | 0 | 6 | 0.562 | 0.180 |
| Tree TD | 56 | 2 | 12 | 0.475 | 0.196 |
| Best observed candidate per root | 74 | — | — | — | — |

This is a negative result for argmax selection under both objectives. The
best-observed row is an empirical ceiling among the nine recorded candidates,
not a known optimal policy or a closed-loop evaluation. Earlier checkpoints
also remained below the 66-success stored-source baseline (root MC peaked at
62; tree TD peaked at 57 among scheduled evaluations). The report and latest
checkpoints are on the Volume; no new LIBERO episodes were run.

## 256-root diagnostic on September 26, 2026

Snapshot `d25aa525f602dfac19774105` has 205 training and 51 validation
roots. Stock succeeds on 37 validation roots; a uniformly random recorded
candidate has 33.11 expected successes, and the recorded oracle succeeds on
41. After 1,000 updates, root MC selects 36 successes (2 rescues, 3 spoils);
tree TD selects 28 (1 rescue, 10 spoils). The training TD sampler sees root
windows only 9.13% of the time. These results are a smaller diagnostic and do
not supersede the 480-tree result above. The final report is
`reports/full_d25aa525f602dfac19774105.json` in the Volume.
