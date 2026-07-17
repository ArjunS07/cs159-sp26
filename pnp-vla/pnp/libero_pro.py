"""LIBERO-PRO environment surgery + episode building (pluggable).

Ported primarily from final/pnp_pro_experiment_averages.ipynb (newer; includes the
refine_average both-variants pass), folding in supplementary setup paths from
'final/pnp_pro_experiment copy.ipynb' (distractor-suite handling, broader init-state gen).

Provides:
- environment patching: copy LIBERO-PRO benchmark/suite-map/object files over the installed
  libero package, rewrite libero_task_map -> .get(..., []), dynamic libero_suites, register
  position-perturb (temp_x/y*) + distractor (_with_*) + swap/task suites, OBJECTS_DICT aliases
- restore_libero_pro_inits(init_src, libero_site): copy .pruned_init files into place
- build_libero_pro_episodes(...): 6 x 10 x 10 = 600-episode list
- suite descriptor decoding (suite_family / perturb_axis / perturb_strength / distractor)

One-time steps (init-state generation, asset tarball backup/restore) are exposed as EXPLICIT
helper functions, not auto-run at import.

TODO(phase 4): port from the two experiment notebooks.
"""
