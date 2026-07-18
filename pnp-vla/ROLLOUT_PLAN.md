# Full LIBERO / LIBERO-PRO Rollout Collection Plan

## Objective

Collect a complete, reusable `K=3` ablation dataset without a screening phase. A
schedule-matched uncertainty-only rollout is the observed baseline for each refinement arm;
there is no redundant no-probe vanilla arm in the primary matrix.

## Rollout matrix

The sampler uses zero-based step indices from 0 through 9. The eight schedules are:

| Family | Schedules |
| --- | --- |
| Adjacent | `(2,3)`, `(3,4)`, `(4,5)`, `(5,6)`, `(7,8)` |
| Periodic | `(1,3,5,7,9)`, `(3,6,9)`, `(2,5,8)` |

For every episode identity and schedule, collect three paired arms with exactly the same
measurement/refinement interval and `K=3`:

1. `observed_base`: uncertainty measurement without refinement.
2. `refine_last`: refine from the final perturbation estimate.
3. `refine_average`: refine from the mean perturbation estimate.

The unique matched-compute controls are based on `10 + K * len(schedule)`:

| Schedule length | Inference steps |
| ---: | ---: |
| 2 | 16 |
| 3 | 19 |
| 5 | 25 |

This gives 27 configurations per identity: 8 observed baselines, 8 refine-last arms,
8 refine-average arms, and 3 matched-compute controls.

## Collection groups

| Group | Episode identities | Observed base | Extra steps | Refine last | Refine average | Planned rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LIBERO | 400 (4 suites x 10 tasks x 10 episodes) | 8 / identity | 3 / identity | 8 / identity | 8 / identity | **10,800** |
| LIBERO-PRO | `N_PRO` from the deduplicated union manifest | 8 / identity | 3 / identity | 8 / identity | 8 / identity | **27 x N_PRO** |
| **Total** | `400 + N_PRO` | | | | | **27 x (400 + N_PRO)** |

LIBERO-PRO uses one manifest containing the union of the prior canonical and expanded
cohorts. Deduplicate by `(suite, task_idx, episode_idx, init_state_hash)` and retain explicit
`canonical_member` and `expanded_member` metadata. Derive `N_PRO` from the installed assets
and print it before starting the run rather than hard-coding a historical count.

## Experiment configuration

- Use experiment labels `libero-full-schedules-k3-v1` and
  `pro-union-full-schedules-k3-v1`.
- Build the schedule matrix through one helper shared by the LIBERO and LIBERO-PRO loops.
- Enable uncertainty telemetry on every observed arm.
- Enable the PCP feature sink (`obs_enc` and `z_hat`) only on the observed `(7,8)` arm.
- Keep `compute_multimodal=False` at `K=3`; multimodality statistics require a later run with
  a larger `K`.
- Resume through the existing episode-identity and logical-config hashes. Never change a
  logical experiment in place after collection begins; use a new experiment label.
- Print the identity count, 27 configs per identity, expected rollout count, and method/config
  summary before executing either group.

## Supported downstream experiments

- Full schedule ablations, including adjacent versus periodic refinement.
- Refine-last versus refine-average comparisons.
- Matched-compute comparisons at 16, 19, and 25 inference steps.
- Failure prediction from observed uncertainty/action telemetry, task metadata, and success.
- PCP training from the observed `(7,8)` features and rollout success labels.
- LIBERO-PRO robustness analysis by perturbation family, axis, strength, distractor, and
  legacy cohort membership.
- A later `K` ablation on promising schedules, recorded under separate experiment labels.

## Validation before collection

1. Assert eight unique schedules, three unique compute controls, and 27 unique logical
   configuration hashes.
2. Assert 400 unique standard LIBERO identities.
3. Assert the PRO union manifest has no duplicate identity keys and valid cohort metadata.
4. Assert every refinement arm has an observed arm with identical `pnp_steps` and `pnp_k`.
5. Assert only observed `(7,8)` enables the PCP feature sink.
6. Run one identity through all 27 configurations and verify Supabase rows, uncertainty rows,
   PCP artifacts, resumption, and failure-free environment teardown before the full launch.

