# Jeff parity corrections — October 2, 2026

No GPU evaluation was launched for these changes. The prior Modal app is stopped and its automation remains paused.

## Fixed

- All five worker notebooks (123 worker 0/1, 125 proposal, 126 replication worker 0/1) pin Torch 2.11.0+cu128 and TorchVision 0.26.0+cu128. Installation resolves those pins with the simulation dependencies, avoiding an intermediate cu130 installation. Notebook descriptions agree with batch eight and matmul TF32 disabled.
- Stock generation stays the original VLA sampler: no invasive P&P or Q correction. Jeff's diagnostic wrapper also returned that original stock sample, but wrapper equivalence has not been measured on the historical inputs.
- VLA snapshot is frozen to `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`; vision model and tokenizer to `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Their latest modification dates predate Jeff's historical run. Historical weight hashes were absent, so this is an immutable reproducibility fix, not a claim of recovered historical weight hashes.
- Removed the extra cuDNN TF32 override that Jeff's notebooks did not set. Pinned Torch's default is retained and recorded. Matmul TF32 remains false and matmul precision highest.
- Replication v2 reconstructs source-run ordering, shard membership, batch limits, companion lanes and resume boundaries from surviving Supabase rows. It covers 400 targets in 85 batches: five size-two, two size-one, one size-three, 77 size-five. On this saved history no extra companion rollouts are needed (400 executed lanes).
- Pre-batching-optimization source runs use their original dense camera rendering for multi-lane batches. Historical serial batches retain sparse rendering, as their original serial runner did. Final 8ad4afe batches use sparse rendering with lead two.
- Resuming v2 skips whole reconstructed batches and preserves the lanes of partially completed batches. It never reuses v1 rows. Only unfinished targets are saved; completed targets and companion lanes do not overwrite existing records.
- Source content hash, immutable model revision and weight-file hashes, dependency versions, GPU driver, Python, cuDNN version/flags, deterministic settings and thread count are stored in run metadata. Source revision attribution and batch memberships are saved per episode in diagnostic JSON.
- Opt-in parity traces record actual active lane membership at every chunk, noise seed and tensor hash, generated chunk hash, plus first camera and preprocessed-input hashes. Both serial and batched paths preserve their existing sampling flow. No hashes are computed in ordinary rollouts unless tracing is requested.
- Fixed sweep persistence: stock had executed its no-probe configuration but saved its rows/IDs using the treatment configuration. New rows use the actual arm configuration for both IDs and saved metadata, and include it explicitly in arm telemetry. Historical results are not rewritten.
- New experiment names isolate corrected protocol results: `smolvla-libero-jeff-stock-replication-v2`, `smolvla-pcp-a100-jeff-settings-stock-v6`, `smolvla-pcp-full-proposal-jeff-settings-v5`.
- Rebuilt notebooks and uploaded/downloaded the code/checkpoint bundle to verify its SHA256. Existing critic weights are unchanged.

## What remains unknown or deliberately distinct

The reconstructed plan assumes surviving rows and run creation order reflect historical resume completion. Overwritten attempts and within-run recoveries cannot be recovered from those rows. The plan logs this assumption rather than asserting exact historical lane membership is proven.

Jeff's Python patch version was 3.13.15; Modal's Debian image currently supplies 3.13.3. Exact historical driver, MuJoCo/robosuite versions, cuDNN TF32 setting, camera inputs and model-file hashes were not saved. These cannot be honestly recovered from a notebook with broad dependencies. Current values are now logged; arbitrary simulator-version guesses were not introduced.

PCP treatments continue to use the approved P&P + Q method. Their new stock control is pure VLA. Main PCP sweep batching is same-task/parity and defaults to eight, but is not the seven-run historical reconstruction; that reconstruction belongs to the dedicated replication runner. The strict outcome-comparison gate stays unsatisfied until a future replication establishes its results. No launch is implicit in these fixes.

## Verification

Historical reconstruction was checked against all 400 saved reference identities. Focused tests cover equal totals with mismatched episodes, missing seeds, historical companion preservation, incomplete provenance and serial rendering. All five generated worker notebooks were parsed as Python and checked for Torch/CUDA pins. The focused rollout/PCP suite passes. This is CPU verification; GPU numerical parity remains unverified.
