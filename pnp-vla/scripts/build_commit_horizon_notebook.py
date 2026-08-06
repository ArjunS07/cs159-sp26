"""Build the commit-horizon comparison notebook (nb 15) — the Gate B readout.

Loads the candidate tables at each commitment horizon (H=10 from the existing DEVELOPMENT
experiments, H=25/50 from the nb-14 re-collection) and compares the quantities that decide
whether test-time selection is horizon-gated or cooked: decidable mass, disagreement slope, the
perfect-selector oracle ceiling, and the best real-selector paired uplift. All methodology lives
in ``pnp.verifier.diagnostics``; this notebook orchestrates and keeps the pre-registered Gate B
thresholds inline. There is no MODEL cell — the comparison is model-free (GBT is the boring
baseline scorer, not the deep verifier).
"""
from __future__ import annotations

from nb_common import BOOTSTRAP, ROOT, code, md, notebook, write_notebook


COMMIT_HORIZON_COMPARE = notebook([
    md("""# 15 — Commit-horizon comparison (Gate B)

Does the oracle ceiling and decidable mass **rise** as the operator commits more of each candidate
chunk? Existing data is `H=10` (the standard horizon); nb 14 adds `H=25` and `H=50` over the same
states. If the ceiling rises materially and a real selector beats the default at `H=50`, test-time
selection is **horizon-gated** (alive in the high-latency regime), not cooked.

**Pre-registered Gate B.** Real bar is the **0.650 default**. Low-power flag: any decidable slice
< 50 groups.

| branch | condition |
|---|---|
| **POSITIVE → selection is horizon-gated** | perfect-selector uplift rises by **≥ 0.05** from H=10 to H=50 **AND** at H=50 a selector beats default (`uplift_ci_lo > 0`) |
| **COOKED → outcome-selection is dead** | perfect-selector uplift **< 0.02** even at H=50 |
| **AMBIGUOUS** | ceiling moved but neither branch fired |"""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("""## 2. Load the candidate table at each horizon

`H=10` is the union of the existing DEVELOPMENT experiments (same as nb 12/13); `H=25`/`H=50` are
the nb-14 re-collection. Horizons without data yet are skipped so this runs before nb 14."""),
    code(r'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display
from pnp import notebook as nb
from pnp.verifier import *

PREFIX_LENGTH=10
DEFAULT_SUCCESS=0.650  # the deployment bar every selector must beat
ctx=nb.setup("commit_horizon"); store,DEVICE,OUTPUT=ctx.store,ctx.device,ctx.output
HORIZON_EXPERIMENTS={
    10: ("verifier-clean-pairs-v3","verifier-clean-pairs-v4-dev",
         "verifier-clean-pairs-v4-test","verifier-online-selection-v1",
         "verifier-v2-pro-development"),
    25: ("verifier-commit25-v1",),
    50: ("verifier-commit50-v1",)}
tables={}
for h,exps in HORIZON_EXPERIMENTS.items():
    try:
        examples=load_candidate_examples(store,exps,cache_dir=OUTPUT/("cand_h%d"%h))
    except Exception as error:
        print("H=%d load failed:"%h,type(error).__name__,error); continue
    if not examples:
        print("H=%d: no candidates yet (run nb 14)"%h); continue
    cand=build_candidate_table(examples,None,prefix_length=PREFIX_LENGTH,progress=tqdm)
    add_mode_structure(cand)
    tables[h]=cand
    print("H=%d: rows=%d groups=%d"%(h,len(cand),cand.group_id.nunique()))'''),
    md("""## 3. Per-horizon summary

Decidable mass (binary + continuous), disagreement slope, GBT boring-baseline ranking, the
perfect-selector oracle ceiling, and the best training-free/GBT selector that actually beats the
default (`uplift_ci_lo > 0`)."""),
    code(r'''def horizon_summary(h,cand):
    dm=decidable_mass(cand,value_col="return_target")
    _,binary_slope=prefix_distance_disagreement(cand)
    _,return_slope=prefix_distance_disagreement(cand,value_col="return_target")
    surf=simulate_selector_uplift(cand)
    ceil=surf[surf.r==1.0].sort_values("k").iloc[-1]
    gbt=fit_gbt_baseline(cand,prefix_length=PREFIX_LENGTH)
    zoo=selector_zoo(cand.assign(gbt_score=gbt["oof"]),score_cols=("gbt_score",))
    beats=zoo[zoo.uplift_ci_lo>0].sort_values("uplift_vs_default",ascending=False)
    best=beats.iloc[0] if len(beats) else None
    return {"H":h,"groups":cand.group_id.nunique(),
            "decidable":dm["binary_decidable"],
            "decidable_frac":round(dm["binary_fraction"],3),
            "value_decidable_frac":round(dm.get("value_fraction",float("nan")),3),
            "binary_slope":round(binary_slope,5),"return_slope":round(return_slope,5),
            "gbt_ranking":(round(gbt["ranking"],3) if np.isfinite(gbt["ranking"]) else float("nan")),
            "ceiling_deployed":round(ceil.deployed_success,3),
            "ceiling_uplift":round(ceil.uplift_vs_default,3),"ceiling_k":int(ceil.k),
            "best_selector":(best.selector if best is not None else "none"),
            "best_uplift":(round(best.uplift_vs_default,3) if best is not None else 0.0),
            "best_uplift_ci_lo":(round(best.uplift_ci_lo,3) if best is not None else float("nan"))}

summary=pd.DataFrame([horizon_summary(h,c) for h,c in sorted(tables.items())])
display(summary)'''),
    md("## 4. Trend plots — oracle ceiling and decidable mass vs commit horizon"),
    code(r'''if len(summary)>=2:
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(summary.H,summary.ceiling_uplift,marker="o")
    ax[0].axhline(0.05,ls="--",c="grey",label="POSITIVE threshold")
    ax[0].axhline(0.02,ls=":",c="red",label="COOKED threshold")
    ax[0].set(xlabel="commit horizon H",ylabel="perfect-selector uplift vs default",
              title="Oracle ceiling vs H"); ax[0].legend()
    ax[1].plot(summary.H,summary.decidable_frac,marker="o",label="binary")
    ax[1].plot(summary.H,summary.value_decidable_frac,marker="s",label="continuous")
    ax[1].set(xlabel="commit horizon H",ylabel="decidable fraction",
              title="Decidable mass vs H"); ax[1].legend()
    plt.tight_layout(); plt.show()
else:
    print("need >= 2 horizons for the trend plot; run nb 14 first")'''),
    md("## 5. Gate B — readout (pre-registered thresholds)"),
    code(r'''by_h={int(row.H): row for _,row in summary.iterrows()}
have=set(by_h)
low_power=[int(r.H) for _,r in summary.iterrows() if r.decidable<50]
print("low-power horizons (decidable<50):",low_power or "none")
if {10,50}<=have:
    rise=by_h[50].ceiling_uplift-by_h[10].ceiling_uplift
    beats50=bool(np.isfinite(by_h[50].best_uplift_ci_lo) and by_h[50].best_uplift_ci_lo>0)
    print("perfect-selector ceiling uplift: H=10 %.3f -> H=50 %.3f  (rise=%.3f, need>=0.05)"
          %(by_h[10].ceiling_uplift,by_h[50].ceiling_uplift,rise))
    print("H=50 selector beats default (uplift_ci_lo>0): %s  (%s, uplift=%.3f)"
          %(beats50,by_h[50].best_selector,by_h[50].best_uplift))
    print("-"*60)
    if rise>=0.05 and beats50:
        print("Gate B = POSITIVE -> selection is horizon-gated, not cooked.",
              "Write the 'selection helps when you commit / high-latency regime' result;",
              "train a verifier at H=50 (Wave 1) and evaluate deployed uplift.")
    elif by_h[50].ceiling_uplift<0.02:
        print("Gate B = COOKED -> even a perfect selector barely clears 0.650 at H=50.",
              "The fresh-noise candidates are interchangeable; pivot to a correction operator",
              "(per-step) or a task/policy with more decidable mass.")
    else:
        print("Gate B = AMBIGUOUS -> the ceiling moved but neither branch fired;",
              "inspect the summary and per-horizon tables.")
else:
    print("Gate B pending: need both H=10 and H=50 tables (run nb 14 first).")'''),
], "15_commit_horizon_compare.ipynb")


def main():
    write_notebook(ROOT / "notebooks" / "15_commit_horizon_compare.ipynb", COMMIT_HORIZON_COMPARE)


if __name__ == "__main__":
    main()
