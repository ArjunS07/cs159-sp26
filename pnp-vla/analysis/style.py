"""Figure styling — one set_style() applied once; plot fns never redefine rcParams.

seaborn + a consistent color-per-method categorical palette, LaTeX serif font
(text.usetex=True), clean/spacious layout (sns.despine, generous margins, restrained
gridlines), savefig at dpi=300 (vector PDF alongside PNG where useful).

Consult the `dataviz` skill before finalizing the palette.

TODO(phase 6): implement set_style() + savefig() helpers + method palette.
"""

# Canonical method order/labels for consistent colors across every figure.
METHOD_ORDER = [
    "vanilla",
    "extra_steps",
    "pnp_uncertainty_only",
    "pnp_refinement",          # refine_average=False (last)
    "pnp_refinement_avg",      # refine_average=True
    "multi_sample_select",
    "pnp_only",                # PCP lambda=0
    "pcp",                     # PCP lambda>0
]
