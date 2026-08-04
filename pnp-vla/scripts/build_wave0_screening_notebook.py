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
    md("""## 3. Prefix-distance / label-disagreement curve  *(H-chaos)*

Rising `p_disagree` with distance = outcomes vary smoothly with the prefix (learnable);
flat = within-group labels are coin flips."""),
    code(r'''dd_table,dd_slope=prefix_distance_disagreement(cand)
display(dd_table)
dd_range=float(dd_table.p_disagree.max()-dd_table.p_disagree.min()) if len(dd_table) else float("nan")
print("slope=%.5f  p_disagree range=%.3f  (rising curve => learnable)"%(dd_slope,dd_range))'''),
    md("""## 4. GBT boring-baseline  *(H-representation)*

Gradient-boosted trees on handcrafted geometric features, GroupKFold, scored with the same
`group_pair_accuracy` as the deep model. GBT ≈ chance ⇒ strong H-chaos evidence; GBT wins ⇒
representation is the bottleneck. Its out-of-fold scores feed the zoo below."""),
    code(r'''gbt=fit_gbt_baseline(cand,prefix_length=PREFIX_LENGTH)
print("GBT group-macro ranking: %.3f  (n=%d)"%(gbt["ranking"],gbt["n_groups"]))
print("top features:",list(gbt["importances"].items())[:5])
cand["gbt_score"]=gbt["oof"]  # attach out-of-fold predictions as a score column'''),
    md("## 5. Selector zoo — training-free floor + GBT pick-best / reject-worst *(vs 0.650)*"),
    code(r'''zoo=selector_zoo(cand,score_cols=("gbt_score",))
display(zoo.sort_values("uplift_vs_default",ascending=False))
print("default success (bar): %.3f"%zoo.attrs["default_success"])'''),
    md("""## 6. Selector-quality → deployed-uplift simulation

`success(r, k)` under a noisy-oracle selector of ranking accuracy `r`. Read off `r*` = the
minimum ranking needed to clear the 0.650 default at each budget `k`."""),
    code(r'''surf=simulate_selector_uplift(cand)
display(surf.pivot(index="r",columns="k",values="deployed_success").round(3))
for k in sorted(surf.k.unique()):
    ok=surf[(surf.k==k)&(surf.deployed_success>=DEFAULT_SUCCESS)]
    print("k=%d -> r* to beat %.3f: %s"%(k,DEFAULT_SUCCESS,
          "%.2f"%ok.r.min() if len(ok) else "unreachable (even r=1.0)"))'''),
    md("""## 7. Per-step U re-stratification  *(closes the gate-choice question)*

The stored `uncertainty_stratum` is a trajectory-average tercile. Re-stratify on U at a
specific denoising step (endpoint s=0.1 vs mid s=0.5) to check the gate is s-invariant."""),
    code(r'''attach_per_step_uncertainty(cand,development,store,progress=tqdm)
for col in ("u_endpoint","u_mid"):
    if cand[col].notna().any():
        print("=== oracle uplift by %s tercile ==="%col)
        display(stratify_by_u_level(cand,col)[["%s_bucket"%col,"groups","oracle_uplift"]])
    else:
        print("no per-step U attached for",col)'''),
    md("## 8. Gate A — readout (pre-registered thresholds)"),
    code(r'''# Raw numbers come from the package; the thresholds/decision stay visible here.
gbt_rank=gbt["ranking"]
curve_rises=(len(dd_table)>0) and (dd_range>0.10) and (dd_slope>0)
free=zoo[zoo.score_col=="-"]
free_beats=bool((free.uplift_vs_default>0).any() and (free[free.uplift_vs_default>0].ci_lo>0).any())
gbt_sel=zoo[(zoo.score_col=="gbt_score")&(zoo.ci_lo>0)&(zoo.uplift_vs_default>0)]
ceil=surf[surf.r==1.0].sort_values("k").iloc[-1]
gbt_flat=(not np.isfinite(gbt_rank)) or (gbt_rank<0.54)
print("signals:")
print("  GBT ranking      = %.3f  (flat if <0.54: %s)"%(gbt_rank,gbt_flat))
print("  distance curve   : slope=%.5f range=%.3f  (rises: %s)"%(dd_slope,dd_range,curve_rises))
print("  free selector beats default (CI-lo>0): %s"%free_beats)
print("  GBT selector beats default (CI-lo>0) : %s"%(not gbt_sel.empty))
print("  perfect-selector ceiling: deployed=%.3f  uplift=%.3f (k=%d)"
      %(ceil.deployed_success,ceil.uplift_vs_default,int(ceil.k)))
print("-"*60)
alive=(not gbt_flat) or curve_rises or free_beats or (not gbt_sel.empty)
if alive:
    print("Gate A = ALIVE -> Wave 1 (3a hygiene rebuild: overfit check, action norm,",
          "multi-task heads; then nb 12 model-scored D1)")
elif ceil.uplift_vs_default < 0.02:
    print("Gate A = DEAD -> pivot: honest-negative / mode-level operator (Chunk 6).",
          "Even a perfect selector barely clears the 0.650 default.")
else:
    print("Gate A = AMBIGUOUS -> no signal fired but the ceiling is non-trivial;",
          "inspect the tables and revisit thresholds")'''),
], "13_wave0_screening.ipynb")


def main():
    write_notebook(ROOT / "notebooks" / "13_wave0_screening.ipynb", SCREENING)


if __name__ == "__main__":
    main()
