"""Build the thin Wave-0 model-free screening notebook (nb 13).

Wave 0 is the circuit breaker in front of the expensive verifier rebuild: a batch of
*model-free* diagnostics (no verifier scores) that separate "no signal exists" from "our
deep model is broken", ending in the pre-registered **Gate A** decision. All methodology
lives in ``pnp.verifier.diagnostics``; this notebook only loads candidates, calls the
helpers section by section, and keeps the Gate A thresholds inline (study design, not
reusable machinery). There is deliberately **no MODEL cell** -- nothing here touches the
verifier, so no training defect can contaminate the readout.
"""
from __future__ import annotations

from nb_common import BOOTSTRAP, ROOT, code, md, notebook, write_notebook


SCREENING = notebook([
    md(r"""# 13 — Wave-0 model-free screening (Gate A)

Zero-sim, **zero-model** screening on already-collected candidate groups. None of these cells
score the verifier, so no training defect (starved state encoder, missing action norm) can
contaminate them. The point is to answer, cheaply, *before* funding a rebuild:

1. **Does signal exist at all?** — GBT boring-baseline (H-representation) + prefix-distance /
   label-disagreement curve (H-chaos).
2. **What would a selector need to be worth it?** — selector-quality → uplift simulation
   converts "ranking above 0.5" into "the minimum `r` to beat the 0.650 default at budget `k`".
3. **What is the training-free floor?** — the selector zoo (a learned Q must beat these).

**Pre-registered Gate A (lock before running).** The real bar is the **0.650 default**, not 0.5.

| branch | condition |
|---|---|
| **DEAD → pivot (skip rebuild)** | GBT ranking flat (< 0.54) AND distance curve flat AND no training-free selector beats default AND perfect-selector ceiling clears 0.650 by < 0.02 |
| **ALIVE → Wave 1 (hygiene rebuild + nb 12)** | any of: GBT ranking materially > chance, distance curve rises, or a selector beats default (CI-lo > 0) |
| **low-power guard** | report discordant / group n; treat any thin slice as inconclusive |

Continuation-time U is not in existing data, so it is out of scope here."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Load candidate groups (model-free table + mode structure)"),
    code(r'''import numpy as np
from tqdm.auto import tqdm
from IPython.display import display
from pnp import notebook as nb
from pnp.verifier import *

PREFIX_LENGTH=10
DEFAULT_SUCCESS=0.650  # the deployment bar every selector must beat
ctx=nb.setup("wave0_screening"); store,DEVICE,OUTPUT=ctx.store,ctx.device,ctx.output
DEVELOPMENT=("verifier-clean-pairs-v3","verifier-clean-pairs-v4-dev",
             "verifier-clean-pairs-v4-test","verifier-online-selection-v1",
             "verifier-v2-pro-development")
development=load_candidate_examples(store,DEVELOPMENT,cache_dir=OUTPUT/"candidate_cache_diag")
cand=build_candidate_table(development,None,prefix_length=PREFIX_LENGTH,progress=tqdm)
add_mode_structure(cand)
print("rows:",len(cand),"| groups:",cand.group_id.nunique())
print("multimodal groups (n_modes>=2):",cand[cand.n_modes>=2].group_id.nunique(),
      "/",cand.group_id.nunique())'''),
    md("""## 3. Decidable mass  *(the structural cap on any selector)*

A group is **binary-decidable** only if it holds both a success and a failure — otherwise its
outcome is fixed no matter which candidate you pick. If most groups are unanimous, deployed
uplift is capped no matter how good the selector. The **continuous** columns (`n_steps`,
`return_target`) turn unanimous-success groups into decidable ones — completion speed is signal
the binary label throws away. `speed_selection_uplift` reads off that hidden headroom."""),
    code(r'''print("binary   :",decidable_mass(cand))
print("n_steps  :",decidable_mass(cand,value_col="n_steps"))
print("return   :",decidable_mass(cand,value_col="return_target"))
spd=speed_selection_uplift(cand)
print("speed uplift (steps saved by picking fastest success vs default):")
print("  eligible groups=%d  mean_saved=%.1f  CI=[%.1f,%.1f]  faster-option frac=%.3f"
      %(spd["n_eligible_groups"],spd["mean_step_savings"],
        spd["savings_ci_lo"],spd["savings_ci_hi"],spd["fraction_with_faster_option"]))
dec=decidable_groups(cand)
print("decidable-only groups:",dec.group_id.nunique(),"/",cand.group_id.nunique())'''),
    md("""## 4. Prefix-distance / label-disagreement curve  *(H-chaos)*

Rising `p_disagree` with distance = outcomes vary smoothly with the prefix (learnable);
flat = within-group labels are coin flips. We also run the **continuous** version (mean
`|Δ return_target|` per bin) and the binary curve on **decidable-only** groups, in case the
signal hides where selection actually matters."""),
    code(r'''dd_table,dd_slope=prefix_distance_disagreement(cand)
display(dd_table)
dd_range=float(dd_table.p_disagree.max()-dd_table.p_disagree.min()) if len(dd_table) else float("nan")
print("binary  all groups : slope=%.5f  range=%.3f  (rising => learnable)"%(dd_slope,dd_range))
_,cont_slope=prefix_distance_disagreement(cand,value_col="return_target")
print("return  all groups : slope=%.5f  (continuous quality gap vs prefix distance)"%cont_slope)
_,dec_slope=prefix_distance_disagreement(dec) if len(dec) else (None,float("nan"))
print("binary  decidable  : slope=%.5f  (n_groups=%d)"%(dec_slope,dec.group_id.nunique()))'''),
    md("""## 5. GBT boring-baseline  *(H-representation)*

Gradient-boosted trees on handcrafted geometric features, GroupKFold, scored with the same
`group_pair_accuracy` as the deep model. GBT ≈ chance ⇒ strong H-chaos evidence; GBT wins ⇒
representation is the bottleneck. Its out-of-fold scores feed the zoo below."""),
    code(r'''gbt=fit_gbt_baseline(cand,prefix_length=PREFIX_LENGTH)
print("GBT group-macro ranking: %.3f  (n=%d)"%(gbt["ranking"],gbt["n_groups"]))
print("top features:",list(gbt["importances"].items())[:5])
cand["gbt_score"]=gbt["oof"]  # attach out-of-fold predictions as a score column'''),
    md("""## 6. Selector zoo — training-free floor + GBT pick-best / reject-worst *(vs 0.650)*

A selector beats the default only when **`uplift_ci_lo > 0`** (the paired-uplift CI, not the
deployed-success `ci_lo`, which is always positive)."""),
    code(r'''zoo=selector_zoo(cand,score_cols=("gbt_score",))
display(zoo.sort_values("uplift_vs_default",ascending=False)
        [["selector","score_col","deployed_success","uplift_vs_default",
          "uplift_ci_lo","uplift_ci_hi","n_groups"]])
print("default success (bar): %.3f"%zoo.attrs["default_success"])
print("selectors that beat default (uplift_ci_lo>0):",
      list(zoo[zoo.uplift_ci_lo>0].selector.unique()) or "none")'''),
    md("""## 7. Selector-quality → deployed-uplift simulation

`success(r, k)` under a noisy-oracle selector of ranking accuracy `r`. Read off `r*` = the
minimum ranking needed to clear the 0.650 default at each budget `k`."""),
    code(r'''surf=simulate_selector_uplift(cand)
display(surf.pivot(index="r",columns="k",values="deployed_success").round(3))
for k in sorted(surf.k.unique()):
    ok=surf[(surf.k==k)&(surf.deployed_success>=DEFAULT_SUCCESS)]
    print("k=%d -> r* to beat %.3f: %s"%(k,DEFAULT_SUCCESS,
          "%.2f"%ok.r.min() if len(ok) else "unreachable (even r=1.0)"))'''),
    md("""## 8. Per-step U re-stratification  *(closes the gate-choice question)*

The stored `uncertainty_stratum` is a trajectory-average tercile. Re-stratify on U at a
specific denoising step (endpoint s=0.1 vs mid s=0.5) to check the gate is s-invariant."""),
    code(r'''attach_per_step_uncertainty(cand,development,store,progress=tqdm)
for col in ("u_endpoint","u_mid"):
    if cand[col].notna().any():
        print("=== oracle uplift by %s tercile ==="%col)
        display(stratify_by_u_level(cand,col)[["%s_bucket"%col,"groups","oracle_uplift"]])
    else:
        print("no per-step U attached for",col)'''),
    md("""## 9. Gate A — readout (pre-registered thresholds)

"Beats default" uses the **paired** `uplift_ci_lo > 0`, not the deployed-success `ci_lo`
(the earlier false-ALIVE bug). A continuous-outcome (speed) signal counts as *signal exists*
but does not clear the 0.650 success bar on its own."""),
    code(r'''# Raw numbers come from the package; the thresholds/decision stay visible here.
gbt_rank=gbt["ranking"]
curve_rises=(len(dd_table)>0) and (dd_range>0.10) and (dd_slope>0)
free_beats=bool((zoo[zoo.score_col=="-"].uplift_ci_lo>0).any())
gbt_sel=zoo[(zoo.score_col=="gbt_score")&(zoo.uplift_ci_lo>0)]
ceil=surf[surf.r==1.0].sort_values("k").iloc[-1]
gbt_flat=(not np.isfinite(gbt_rank)) or (gbt_rank<0.54)
speed_signal=bool(np.isfinite(spd["savings_ci_lo"]) and spd["savings_ci_lo"]>0)
print("signals:")
print("  GBT ranking      = %.3f  (flat if <0.54: %s)"%(gbt_rank,gbt_flat))
print("  distance curve   : slope=%.5f range=%.3f  (rises: %s)"%(dd_slope,dd_range,curve_rises))
print("  free selector beats default (uplift_ci_lo>0): %s"%free_beats)
print("  GBT selector beats default (uplift_ci_lo>0) : %s"%(not gbt_sel.empty))
print("  continuous speed signal (savings_ci_lo>0)   : %s"%speed_signal)
print("  perfect-selector ceiling: deployed=%.3f  uplift=%.3f (k=%d)"
      %(ceil.deployed_success,ceil.uplift_vs_default,int(ceil.k)))
print("-"*60)
success_alive=(not gbt_flat) or curve_rises or free_beats or (not gbt_sel.empty)
if success_alive:
    print("Gate A = ALIVE -> Wave 1 (3a hygiene rebuild: overfit check, action norm,",
          "multi-task heads; then nb 12 model-scored D1)")
elif speed_signal:
    print("Gate A = ALIVE (continuous) -> signal exists on completion SPEED, not binary",
          "success. Pursue the speed/return target; still run the commit-horizon study (nb 15).")
elif ceil.uplift_vs_default < 0.02:
    print("Gate A = DEAD -> pivot: honest-negative / mode-level operator (Chunk 6).",
          "Even a perfect selector barely clears the 0.650 default at H=10.",
          "Decisive next step: the commit-horizon re-collection (nb 14/15).")
else:
    print("Gate A = AMBIGUOUS -> no signal fired but the ceiling is non-trivial;",
          "inspect the tables and revisit thresholds")'''),
], "13_wave0_screening.ipynb")


def main():
    write_notebook(ROOT / "notebooks" / "13_wave0_screening.ipynb", SCREENING)


if __name__ == "__main__":
    main()
