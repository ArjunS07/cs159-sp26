"""Build the thin verifier existing-data diagnostics notebook (nb 12).

All methodology lives in ``pnp.verifier.diagnostics``; this notebook only loads
candidates, calls those helpers section by section, and keeps the pre-registered
D1 threshold/decision logic inline (it is a study-design decision, not reusable
machinery).
"""
from __future__ import annotations

from nb_common import BOOTSTRAP, ROOT, code, md, notebook, write_notebook


DIAGNOSTICS = notebook([
    md(r"""# 12 — Verifier existing-data diagnostics (Phase 0)

Zero-sim diagnostics on already-collected candidate groups. No new collection, no training.
Reuses the trained verifier only to *score* existing candidates. Ends with the **D1** gate readout.
All methodology is in `pnp.verifier.diagnostics`; this notebook orchestrates and keeps only the
pre-registered D1 thresholds inline.

**Pre-registered D1 thresholds (lock before running).** Evaluate on the deployment-relevant
stratum = top prefix-mode-spread tercile (proxy for multimodal high-U), discordant groups only:

| branch | condition |
|---|---|
| **ALIVE → Phase 1** | ranking ≥ 0.62 AND CI-lo > 0.50 AND control_gap > 0 AND oracle_uplift ≥ 0.05 |
| **REPRESENTATION → Phase 2a** | ranking CI includes 0.50 (flat) BUT oracle_uplift ≥ 0.10 |
| **NO-SUPPORT → narrow claim** | oracle_uplift < 0.03 |
| **low-power guard** | stratum < 20 discordant groups → INCONCLUSIVE |

Operator B is not pursued: this scores finished (t=1) chunks, matching endpoint best-of-n."""),
    md("## 1. Setup"), code(BOOTSTRAP),
    md("## 2. Load candidate groups"),
    code(r'''import numpy as np
from tqdm.auto import tqdm
from IPython.display import display
from pnp import notebook as nb
from pnp.verifier import *

PREFIX_LENGTH=10
ctx=nb.setup("verifier_diagnostics"); store,DEVICE,OUTPUT=ctx.store,ctx.device,ctx.output
DEVELOPMENT=("verifier-clean-pairs-v3","verifier-clean-pairs-v4-dev",
             "verifier-clean-pairs-v4-test","verifier-online-selection-v1",
             "verifier-v2-pro-development")
development=load_candidate_examples(store,DEVELOPMENT,cache_dir=OUTPUT/"candidate_cache_diag")
print("loaded",len(development),"candidate examples across",
      len({e.candidate_group_id for e in development}),"groups")'''),
    md("""## 3. Trained verifier (needed only for 0a + mode-ranking accuracy)

The model-free cells (0b-ii, 0c, 0d) run regardless. To fill the model-scored columns, set `MODEL`
to a trained `CompactAdvantageVerifier` — either reuse an in-session model from nb 09
(`MODEL = selected_model`) or rebuild from a registered checkpoint with `build_verifier`."""),
    code(r'''# Option A (simplest): run this right after nb 09 and reuse its model:
#     MODEL = selected_model
# Option B: rebuild from a registered checkpoint:
#     checkpoint, row = store.load_verifier(VERIFIER_ID)
#     MODEL = build_verifier(checkpoint["metadata"]["selected_spec"], checkpoint["model"], DEVICE)
MODEL = None  # <-- set this to enable the model-scored sections'''),
    md("## 4. Per-candidate table (conditioned score + action-only control) + mode structure"),
    code(r'''cand=build_candidate_table(development,MODEL,DEVICE,prefix_length=PREFIX_LENGTH,progress=tqdm)
add_mode_structure(cand)
print("rows:",len(cand),"| groups:",cand.group_id.nunique(),"| model scored:",MODEL is not None)
print("strata:",cand.groupby("stratum").group_id.nunique().to_dict())
print("spread terciles (groups):",cand.groupby("spread_tercile").group_id.nunique().to_dict())
print("multimodal groups (n_modes>=2):",cand[cand.n_modes>=2].group_id.nunique(),
      "/",cand.group_id.nunique())'''),
    md("## 5. 0a — Stratified ranking + oracle uplift + control_gap  *(needs MODEL)*"),
    code(r'''if MODEL is not None:
    print("=== 0a by uncertainty stratum ==="); display(stratified(cand,"stratum"))
    print("=== 0a by prefix-mode-spread tercile ==="); display(stratified(cand,"spread_tercile"))
else:
    print("MODEL is None -> skipping 0a. The model-free sections below still run.")'''),
    md("## 6. 0b — Mode-level re-analysis"),
    code(r'''mr=[a for _,g in cand.groupby("group_id") if not np.isnan(a:=mode_ranking_accuracy(g))]
wf=[f for _,g in cand.groupby("group_id") if not np.isnan(f:=within_mode_pair_fraction(g))]
print("mode-ranking accuracy (discordant-mode groups): "
      + ("%.3f  (n=%d)"%(np.mean(mr),len(mr)) if mr else "needs MODEL"))
print("within-mode fraction of BT pairs: %.3f  (n=%d)"%(np.nanmean(wf),len(wf)),
      "  <- high = that fraction of the training gradient is within-mode noise")
if MODEL is not None:
    from scipy.stats import spearmanr
    tmp=cand.copy(); tmp["soft"]=tmp.groupby(["group_id","mode"])["success"].transform("mean")
    sc=tmp.dropna(subset=["score"])
    print("Spearman(score, cluster-success soft label): %.3f"%spearmanr(sc["score"],sc["soft"]).correlation)'''),
    md("## 7. 0c — Oracle ceiling (free)"),
    code(r'''print("overall oracle uplift (max - default): %.4f"
      % np.mean([group_oracle_uplift(g) for _,g in cand.groupby("group_id")]))
for label,mask in [("multimodal (n_modes>=2)",cand.n_modes>=2),
                   ("unimodal   (n_modes==1)",cand.n_modes<2),
                   ("high-spread tercile",     cand.spread_tercile=="high")]:
    sub=cand[mask]
    if sub.group_id.nunique():
        u=np.mean([group_oracle_uplift(g) for _,g in sub.groupby("group_id")])
        print("  %-26s oracle_uplift=%.4f  (groups=%d)"%(label,u,sub.group_id.nunique()))'''),
    md("## 8. 0d — Training-free baselines a learned Q must beat"),
    code(r'''mv=[majority_mode_success(g) for _,g in cand.groupby("group_id")]
dflt=[default_success(g) for _,g in cand.groupby("group_id")]
orc=[group_oracle_uplift(g) for _,g in cand.groupby("group_id")]
print("selector success  | majority-mode: %.4f | default: %.4f | oracle headroom: %.4f"
      % (np.mean(mv),np.mean(dflt),np.mean(orc)))
display(cand.groupby("stratum").agg(groups=("group_id","nunique"),
                                    group_success=("success","mean")).reset_index())'''),
    md("## 9. D1 — gate readout (pre-registered thresholds)"),
    code(r'''# Raw numbers come from the package; the thresholds/decision stay visible here.
stats=deployment_stratum_stats(cand, stratum_value="high")
rank,ci,gap=stats["ranking"],stats["ci"],stats["control_gap"]
orc,n=stats["oracle_uplift"],stats["discordant_n"]
print("deployment stratum (high prefix-mode spread):")
print("  ranking=%.3f  CI[%.3f, %.3f]  control_gap=%.3f  oracle_uplift=%.3f  discordant_n=%d"
      % (rank,ci[0],ci[1],gap,orc,n))
print("-"*60)
if n < 20:
    print("D1 = INCONCLUSIVE (low power: <20 discordant groups in stratum)")
elif MODEL is None:
    print("D1 = need MODEL for ranking/control_gap; oracle_uplift=%.3f already informs Phase-2 fork" % orc)
elif rank >= 0.62 and ci[0] > 0.50 and (np.isnan(gap) or gap > 0) and orc >= 0.05:
    print("D1 = ALIVE -> Phase 1 (mode-level verifier rebuild)")
elif (np.isnan(rank) or ci[0] <= 0.50) and orc >= 0.10:
    print("D1 = REPRESENTATION -> Phase 2a privileged-teacher probe")
elif orc < 0.03:
    print("D1 = NO-SUPPORT -> narrow claim to gate + diagnosis (E1/E3)")
else:
    print("D1 = AMBIGUOUS -> inspect the 0a/0c tables and revisit thresholds")'''),
], "12_verifier_existing_data_diagnostics.ipynb")


def main():
    write_notebook(
        ROOT / "notebooks" / "12_verifier_existing_data_diagnostics.ipynb", DIAGNOSTICS)


if __name__ == "__main__":
    main()
