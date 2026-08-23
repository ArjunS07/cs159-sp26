# PCP-search collection

This package owns data collection for the new Bellman/RL-token critic. It does not import the
legacy Q-corrector architecture, loaders, checkpoints, or datasets.

1. Build the fixed initial 400-rollout manifest from the pinned historical cohort:

   ```bash
   PYTHONPATH=. python scripts/pcp_search_plan.py --output /tmp/pcp_search_initial.json
   ```

   For an offline audit, add `--input-json /path/to/cached_rollouts.json`.

2. Apply `supabase/migrations/005_pcp_search.sql`, inspect the generated summary, then freeze it
   with `--publish`.
   Publishing is content-addressed and idempotent; membership cannot be updated or deleted after
   it is frozen. Worker progress lives in a separate table.

3. Paste the resulting `pcps-...` ID into the four `52_pcp_search_worker_*.ipynb` launchers. The
   notebooks only bootstrap the repo and invoke `run_pcp_search_worker`. Each exposes one runtime
   throughput knob, `ROLLOUT_BATCH_SIZE` (default: 8), for independent same-task LIBERO
   environments per GPU policy call. It does not alter rollout semantics, lane-specific seeds, or
   saved artifacts; lower it only if a particular Colab GPU runs out of memory.

4. After the first 400 are audited, call `build_next_tranche_manifest` for each 200-rollout
   adaptive tranche. It allocates five-rollout blocks using failure-probability UCB90, expected
   failure chunks, and current training-ready transition coverage, with a 50-rollout task cap.

Every training-ready rollout contains exact H=10 Bellman transitions, T+1 physical/simulator
states, raw and processed observations at all decision boundaries (including terminal), full
frozen VLA prefixes/tokens/masks/positions, stock normalized and environment actions, rewards and
terminal masks, generated chunks, and complete steps-3/4 K=5 P&P traces. The P&P pass is
measurement-only: executed actions are the pinned vanilla policy's actions.

## Partitioned LIBERO-PRO program

`53_pcp_search_initial_survey.ipynb` is read-only: it profiles the historical stock P&P PRO
cohort, confirms that its old 50-action execution artifacts are selection-only, and prints the
approved 640 train / 160 heldout whole-suite partition. `54_pcp_search_manifest_control.ipynb`
freezes the train and heldout PRO manifests and each standard adaptive tranche. The four
`53_pcp_search_pro_worker_*.ipynb` launchers accept either frozen PRO manifest ID. Heldout rows
are fully collected but have `pcp_train_eligible=false`; loaders must never fit on them.
