# Focused critic follow-up — September 30, 2026

## Fixed local training queue

Run `scripts/run_smolvla_local_continuation.py` with the repo's `.venv` Python.
The queue first restores the anchored temporal CNN MC checkpoint at update
2,000, including Adam moments, and adds 8,000 ordinary all-nine root BCE updates.
Then it restores the original scalar five-step recorded-policy TD checkpoint
from the overnight experiment and adds 8,000 TD updates. These use different
architectures and data sampling, so they are not a matched MC-vs-TD comparison.

Both keep the frozen 1,280/320 root split, gamma=1, the existing architectures,
and a cosine learning rate from 1e-4 to 1e-5. TD retains its EMA target rate .005.
Checkpoints save every 100 MC / 50 TD updates; train and validation metrics save
every 1,000 updates. Final gradient diagnostics and MLflow artifacts accompany
the fixed final endpoint. The queue takes an exclusive local lock and can be
rerun to resume. No new collection data enters this experiment midway.

MC reaches 10,000 residual updates (31.25 average root presentations). The TD
checkpoint counter reaches 12,000, representing 14,000 total updates including
the original 2,000 before the overnight counter restarted. Outputs are under
`~/pnp-vla-runs/checkpoints/smolvla_local_continuation_v1/671b5b211099997fc83d1277`.

## Targeted training collection: notebook 123

Three copies of `123_smolvla_training_action_contrast.ipynb`, with shard indices
0/1/2 and shard count 3. Preview first, smoke with tree limit 1, then set the
limit to None for all four assigned roots. Collection requires RUN_COLLECTION.

Twelve original training roots: six historically mixed, three all failures,
three all successes, selected deterministically with suite/task diversity.
Each has fourteen candidates: stored source, one fresh proposal, and twelve
smooth signed x/y/z offsets at .02/.06. This is 156 newly simulated branches,
plus twelve reused originals. Full proposals and frozen-policy continuations
are saved to Supabase under a separate experiment and immutable manifest.
Complete fourteen-artifact groups are skipped after a disconnect.

This probes local action sensitivity. It adds branches at existing training
roots; it does not add twelve independent held-out test situations. All siblings
retain the original training split. Data must be audited and versioned before
a separate training experiment ingests it.

## Focused real flow-step PCP: notebook 122 / Modal launcher

One fixed late-fusion scalar MC checkpoint:
`9b8800bc9f27116afcb1812f93f8a9f6767ac012c7cdac790502032fbb7d0c8e`.
The worker refuses a different checksum or model family.

Select all available historically mixed original validation roots, up to 48,
excluding the previous eighteen pilot source episodes. The local old-cohort
snapshot has 42 mixed validation roots: eight stock failures and 34 stock
successes before prior-pilot exclusion. Actual eligible inventory is reported;
we do not manufacture balance by duplicating failure roots.

Keep correction step 3, s=.7, one paired future continuation seed per root,
and the existing nine-candidate contract: stored source, fresh noise, live zero,
ascent/descent/random at RMS .02/.06. Three disjoint shards. At 42 eligible
roots, this means 336 new simulated branches plus reused stored originals.
Freeze the full cohort manifest before interpreting intervention results.

Report both ascent magnitudes against live zero, rescues/spoils, and random and
descent controls. Root, not branch, is the sampling unit. Previously used
validation and historical mixed-outcome enrichment make results exploratory
and conditional; they do not estimate full-policy benchmark improvement.
Retain final executed-action displacement and Q changes as secondary mechanics
diagnostics; increasing predicted Q is not evidence of better control.

The Modal script uploads only package code/metadata and this exact checkpoint,
uses the existing pnp-supabase secret, and runs L4 workers with two CPU cores
and 16 GiB RAM. Each invocation has a 45-minute timeout, 10-minute startup
timeout, no automatic retries, at most three containers, and single-use workers.
Explicit spending approval is required before the smoke and full calls.
Current allocation rate is about $1.02/worker-hour. Expected total is $1–$3;
the requested $5 operational stop limit is not a provider-enforced billing cap.
## Execution status after explicit approval

The user approved the bounded Modal pilot. The smoke completed one validation
root in 6.3 minutes. All nine candidate artifacts were saved to Supabase under
`smolvla-libero-focused-validation-flow-pcp-v1-9b8800bc9f27116a-d4e3295d4fc3`.
Live zero exactly reproduced the full baseline action chunk. All correction
controls shared initial noise, clean estimate and probe epsilon; the sampler
also checked the final perturbation RNG state. Observed latent correction RMS
was .006/.018 for clean action RMS .02/.06, with finite gradients and Q values.

Live zero failed on this root. Both ascent and descent magnitudes succeeded;
large random and fresh sampling also succeeded, while small random failed.
This demonstrates perturbation sensitivity, not a useful gradient direction.

The cohort audit confirmed 42 unique validation roots, eight historical stock
failures and 34 successes, in three disjoint shards of 14. The full workers
were dispatched at https://modal.com/apps/arjuns07/main/ap-fR5mrDo1v52MGahm5RRi3F
with the approved 45-minute limits. The smoke root is resumed without duplicate
collection. Runtime may yield only a partial cohort; report actual completed
roots and deterministic order/time-limit truncation, without extending spend.

The local CNN extension completed 8,000 additional MC updates: training selected
854/1,280 versus 832 stock (37 rescues, 15 spoils), fresh pair accuracy .7398;
validation selected 198/320 versus 209 stock (6 rescues, 17 spoils), fresh pair
accuracy .5016 and Brier .1353. It overfits and is not promoted into this PCP
experiment. The separately initialized TD continuation is still running;
these arms differ in architecture/data and are not a matched objective test.
