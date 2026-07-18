# Full LIBERO / LIBERO-PRO Rollout Collection Plan

## Objective

Collect a reusable `K=3` ablation dataset without a schedule-screening phase. Run the complete
schedule matrix on the historical 80-identity failure-prone cohort, then validate the established
refine-last `(4,5)` configuration across every remaining stock LIBERO identity.

## Rollout matrix

The sampler uses zero-based step indices from 0 through 9. The eight schedules are:

| Family | Schedules |
| --- | --- |
| Adjacent | `(2,3)`, `(3,4)`, `(4,5)`, `(5,6)`, `(7,8)` |
| Periodic | `(1,3,5,7,9)`, `(3,6,9)`, `(2,5,8)` |

For every episode identity, collect one observed arm with `pnp_steps=(1,2,3,4,5,6,7,8,9)`
and `K=3`. Its step-indexed telemetry supports every schedule as a downstream view. For each
schedule, collect two refinement arms with the same interval and `K=3`:

1. `refine_last`: refine from the final perturbation estimate.
2. `refine_average`: refine from the mean perturbation estimate.

The unique matched-compute controls are based on `10 + K * len(schedule)`:

| Schedule length | Inference steps |
| ---: | ---: |
| 2 | 16 |
| 3 | 19 |
| 5 | 25 |

This gives 20 configurations per identity: 1 shared observed baseline, 8 refine-last arms,
8 refine-average arms, and 3 matched-compute controls.

## Collection groups

| Group | Episode identities | Observed base | Extra steps | Refine last | Refine average | Planned rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LIBERO full ablation | 80 (8 historical tasks x 10 episodes) | 1 / identity | 3 / identity | 8 / identity | 8 / identity | **1,600** |
| LIBERO broad validation | 320 remaining identities | 1 / identity | 1 / identity (16 steps) | 1 / identity `(4,5)` | 0 | **960** |
| **LIBERO total** | **400** | | | | | **2,560** |
| LIBERO-PRO | `N_PRO` from the deduplicated union manifest | 1 / identity | 3 / identity | 8 / identity | 8 / identity | **20 x N_PRO** |
| **Total** | `400 + N_PRO` | | | | | **2,560 + 20 x N_PRO** |

LIBERO-PRO uses one manifest containing the union of the prior canonical and expanded
cohorts. Deduplicate by `(suite, task_idx, episode_idx, init_state_hash)` and retain explicit
`canonical_member` and `expanded_member` metadata. Derive `N_PRO` from the installed assets
and print it before starting the run rather than hard-coding a historical count.

## Experiment configuration

- Use experiment labels `libero-hybrid-schedules-k3-v1` and
  `pro-union-full-schedules-k3-v1`.
- Build the schedule matrix through one helper shared by the LIBERO and LIBERO-PRO loops.
- Enable uncertainty telemetry and the PCP feature sink (`obs_enc` and `z_hat`) on the shared
  observed arm. Select `(7,8)` or any other schedule downstream from its step-indexed data.
- Keep `compute_multimodal=False` at `K=3`; multimodality statistics require a later run with
  a larger `K`.
- Resume through the existing episode-identity and logical-config hashes. Never change a
  logical experiment in place after collection begins; use a new experiment label.
- Deterministically shard episode identities with common `SHARD_COUNT` and distinct
  `SHARD_INDEX` values; never shard configurations for the same identity across workers.
- Print each worker's identity/config counts and expected rollout count before execution.

## Supported downstream experiments

- Full schedule ablations, including adjacent versus periodic refinement.
- Refine-last versus refine-average comparisons.
- Matched-compute comparisons at 16, 19, and 25 inference steps.
- Failure prediction from observed uncertainty/action telemetry, task metadata, and success.
- PCP training from the shared observed features (including `(7,8)`) and rollout success labels.
- LIBERO-PRO robustness analysis by perturbation family, axis, strength, distractor, and
  legacy cohort membership.
- A later `K` ablation on promising schedules, recorded under separate experiment labels.

## Validation before collection

1. Assert eight unique schedules, three unique compute controls, and 20 unique logical
   configuration hashes.
2. Assert 400 unique standard LIBERO identities, with 80 in the historical ablation cohort and
   320 in broad validation.
3. Assert the PRO union manifest has no duplicate identity keys and valid cohort metadata.
4. Assert the shared observed arm covers the union of every refinement schedule at the same `pnp_k`.
5. Assert only the shared observed arm enables the PCP feature sink.
6. Run one identity through all 20 configurations and verify Supabase rows, uncertainty rows,
   PCP artifacts, resumption, and failure-free environment teardown before the full launch.
