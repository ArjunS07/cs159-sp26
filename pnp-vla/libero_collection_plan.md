# LIBERO data collection plan

This document tells you exactly how to help collect data. You do not need to understand the
research project first. Do not change the policy settings in these instructions.

## Goal

We are collecting training data for a new action-value critic.

- The policy creates a 50-action chunk.
- The robot executes only the first 10 actions, then makes a new plan.
- Every rollout saves actions, rewards, robot state, P&P uncertainty, and the other data needed
  for Bellman-backup training.
- Do **not** enable refinement, a corrector, or candidate selection.

There are two new collection rounds:

1. **Standard LIBERO adaptive:** 200 rollouts. This adds failures and coverage in the main LIBERO
   tasks.
2. **Fresh-state LIBERO-PRO train:** 640 rollouts. This uses new physical start states where
   LIBERO-PRO provides them. Two small milk suites have only ten states, so they use a new random
   behavior seed instead.

The 160 held-out position-perturbation PRO rollouts are **not part of this run**. Do not collect
or train on them yet.

## Who runs what

Two people should each run two Colab notebooks at the same time. All four notebooks in one phase
use the same manifest ID. Only the shard index changes.

| Person | Notebook | `SHARD_INDEX` |
|---|---|---:|
| Person A | first notebook | 0 |
| Person A | second notebook | 1 |
| Person B | first notebook | 2 |
| Person B | second notebook | 3 |

Use a GPU Colab runtime. In every notebook, run the bootstrap/setup cell first. It pulls the
current `main` branch. Then use this worker cell, changing only `MANIFEST_ID` and `SHARD_INDEX`:

```python
from pnp.pcp_search.collection import run_pcp_search_worker

MANIFEST_ID = "PASTE_THE_MANIFEST_ID_HERE"
SHARD_COUNT = 4
SHARD_INDEX = 0  # 0, 1, 2, or 3: one different value per notebook
ROLLOUT_BATCH_SIZE = 32

assert MANIFEST_ID.startswith("pcps-")
run_pcp_search_worker(
    manifest_id=MANIFEST_ID,
    shard_count=SHARD_COUNT,
    shard_index=SHARD_INDEX,
    rollout_batch_size=ROLLOUT_BATCH_SIZE,
    batch_strategy="mixed_task",
)
```

`mixed_task` is important. It keeps the GPU batch full by running independent LIBERO tasks in the
same model call. It does not change the scientific rollout settings.

If GPU or system RAM runs out, change only `ROLLOUT_BATCH_SIZE` to `24` and restart that same
shard. Do not change the shard count, action horizon, policy, or P&P settings.

## Coordinator steps

One person is the coordinator. The coordinator runs the following in a small Colab notebook after
the bootstrap cell. Save every printed manifest ID in this document's run log.

### Phase 1: publish and collect standard LIBERO adaptive 200

```python
from pnp.pcp_search.control import publish_standard_adaptive

STANDARD_PARENT = "pcps-c810651498933ba955c51560"
STANDARD_MANIFEST_ID = publish_standard_adaptive(
    parent_manifest_id=STANDARD_PARENT,
    tranche_index=1,
)
print(STANDARD_MANIFEST_ID)
```

Give `STANDARD_MANIFEST_ID` to all four workers. Wait until the monitor says all 200 rows are
`training_ready` before moving on.

### Phase 2: publish and collect fresh PRO sentinels

```python
from pnp.pcp_search.control import publish_fresh_pro_train_sentinel

FRESH_PRO_SENTINEL_ID = publish_fresh_pro_train_sentinel()
print(FRESH_PRO_SENTINEL_ID)
```

Give this ID to all four workers. There are eight rollouts total, so each worker runs two. Wait
until all eight are `training_ready`.

### Phase 3: publish and collect fresh PRO train 640

```python
from pnp.pcp_search.control import publish_fresh_pro_train

FRESH_PRO_TRAIN_ID = publish_fresh_pro_train()
print(FRESH_PRO_TRAIN_ID)
```

Give this ID to all four workers. Each worker runs 160 rollouts. This data is allowed for critic
training after it passes validation.

### Monitoring

Use notebook `55_pcp_search_monitor.ipynb`, or run:

```python
from pnp.pcp_search.monitor import collect_manifest_monitor

collect_manifest_monitor("PASTE_THE_MANIFEST_ID_HERE")
```

The phase is complete only when `training_ready` equals the planned rollout count and there are no
`errored` rows.

## If something stops

- A Colab disconnect is okay. Run the same worker again with the same manifest ID, shard count,
  and shard index.
- Do not change from four shards to another number while any old four-shard notebook is still
  running.
- The system skips rollouts already saved and validated. A partially completed batch is safe to
  run again.
- Report an error message, manifest ID, and shard index to the coordinator. Do not delete rows or
  rerun a different manifest to work around an error.

## Run log

| Phase | Manifest ID | Expected | Training-ready | Notes |
|---|---|---:|---:|---|
| Standard adaptive 200 |  | 200 |  |  |
| Fresh PRO sentinel |  | 8 |  |  |
| Fresh PRO train |  | 640 |  |  |
