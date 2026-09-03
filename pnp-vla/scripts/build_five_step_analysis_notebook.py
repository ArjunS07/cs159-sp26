"""Generate notebook 61 in the style of the earlier paired PRO analysis notebooks."""
from nb_common import ROOT, bootstrap, code, md, notebook, write_notebook


def analysis_notebook():
    document = notebook([
        md("""# 61 - Five-step diversity PRO220 analysis

Combines workers 60-0 and 60-1. Like notebooks 49 and 27, this notebook shows matched
success-rate tables, paired per-suite plots, candidate choices, and action diversity.
All four arms execute **10 actions** per 50-action prediction.

No GPU, simulator, Drive mount, or action-artifact downloads are required. The worker-saved
manifest and checkpoint revision are verified before matching. Leave preview mode on while
collection is running; switch to strict mode for the final **220-identity** result."""),
        md("## 1. Setup"),
        code(bootstrap(extras="analysis", setup_env=False)),
        md("## 2. Load and exact-match the cohort"),
        code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

from analysis.five_step_diversity import (
    ARMS, LABELS, KEYS, fetch_five_step_rows, match_cohort, analyze_five_step,
    success_figures, diversity_figures, selection_figures)
from pnp.five_step_diversity_experiment import FIVE_STEP_DIVERSITY_EXPERIMENT
from pnp.store import SupabaseStore

EXPERIMENT = FIVE_STEP_DIVERSITY_EXPERIMENT
REQUIRE_FULL_COHORT = False  # True for the final 220-identity analysis
OUTPUT = Path("five_step_diversity_pro220_outputs")
OUTPUT.mkdir(exist_ok=True)

store = SupabaseStore()
rows, manifest, provenance = fetch_five_step_rows(store, experiment=EXPERIMENT)
paired, matched, coverage = match_cohort(
    rows, manifest, require_complete=REQUIRE_FULL_COHORT)

print("FINAL STRICT ANALYSIS" if REQUIRE_FULL_COHORT else "INTERIM MATCHED PREVIEW")
print(f"{len(paired)}/{len(manifest)} matched identities; "
      f"{3 * len(paired)} new rollouts + {len(paired)} reused baseline rollouts.")
print("Every SR denominator below uses these same matched identities.")
display(coverage)
if not REQUIRE_FULL_COHORT:
    print("Incomplete identities are excluded from EVERY arm, including historical stock. "
          "Rerun with REQUIRE_FULL_COHORT=True after collection finishes.")
print("Verified source:", provenance["source_id"])
paired.to_csv(OUTPUT / "matched_episode_outcomes.csv", index=False)
matched[KEYS + ["arm", "rollout_id", "run_id", "config_hash", "success"]].to_csv(
    OUTPUT / "matched_rollout_ids.csv", index=False)
coverage.to_csv(OUTPUT / "coverage.csv", index=False)
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
tables = analyze_five_step(paired, matched)
for name, frame in tables.items():
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)"""),
        md("""## 3. Primary result: success rates and paired changes

The historical reference is the exact verified 10-step x1 arm from the direct-U20-gradient
pilot, not the older 50-executed-action data. The three new arms are compared to that reference;
selection versus 5-step x1 and refinement versus selection isolate the additional interventions.
F_to_S and S_to_F are paired outcome flips. Intervals resample identities, not chunks."""),
        code("""display(tables["arm_success_rates"][[
    "label", "identities", "successes", "sr_pct", "ci_low_pct", "ci_high_pct"]])
display(tables["paired_effects"][[
    "comparison", "identities", "baseline_sr_pct", "condition_sr_pct", "delta_pp",
    "ci_low_pp", "ci_high_pp", "F_to_S", "S_to_F", "paired_p_value"]])
display(tables["suite_effects"].query("reference == 'baseline'")[[
    "suite", "comparison", "identities", "delta_pp", "F_to_S", "S_to_F"]])
for path in success_figures(tables, OUTPUT / "figures"):
    display(Image(filename=str(path)))"""),
        md("""## 4. How different are the three queried chunks?

Each boundary retains the original, unrefined pairs **0-1, 0-2, 1-2**. Cosine similarity uses
flattened **normalized policy-space actions including the gripper**: 1 means the same direction,
0 means orthogonal, and -1 means opposite. Cosine distance is **1 - similarity**; larger means
more directional difference. L2 also captures magnitude differences.

First-10 metrics describe the executed prefix; full-50 metrics include the unexecuted tail.
Undefined zero-norm cosines remain missing, not zero. Tables average within each episode first;
histograms explicitly pool boundaries. The two x3 arms evolve along different trajectories:
these are within-boundary candidate comparisons, not matched later-state cross-arm comparisons.
The first-boundary table avoids the later trajectory divergence."""),
        code("""display(tables["candidate_diversity"][[
    "arm", "horizon", "pair", "identities", "boundaries", "finite_cosines",
    "cosine_similarity", "cosine_distance", "action_l2_mean", "gripper_disagreement"]])
print("First boundary only (same initialization, before each arm's first action):")
display(tables["first_boundary_diversity"][[
    "arm", "horizon", "pair", "identities", "cosine_similarity", "action_l2_mean"]])
for path in diversity_figures(tables, OUTPUT / "figures"):
    display(Image(filename=str(path)))"""),
        md("""## 5. Selection and refinement diagnostics

Selection frequencies and U summaries are episode-weighted. U50 is the stored full-chunk U
(the worker generates exactly 50 actions). Candidate diagnostics precede refinement in both
x3 arms. The refinement-path U is measured along the changed solve; it is **not** an independent
remeasurement of the final refined action. Lower path-U or greater diversity does not prove
better task success—use the paired SR results above."""),
        code("""display(tables["selection_summary"])
print("U10/U20/U50 for each original candidate slot:")
display(tables["candidate_uncertainty"])
print("Select-then-refine: mean of each episode's boundary diagnostics")
display(tables["refinement_by_episode"].drop(columns="rollout_id").mean().to_frame("mean").T)
for path in selection_figures(tables, OUTPUT / "figures"):
    display(Image(filename=str(path)))

print("New-arm compute diagnostics (episode-mean cost per boundary):")
display(tables["compute"])
print("Historical VF counts used older instrumentation and are intentionally omitted. "
      "Wall time also depends on hardware; these are not controlled speed benchmarks.")"""),
        md("## 6. Compact readout"),
        code("""print(f"Matched identities: {len(paired)}/{len(manifest)}")
for row in tables["arm_success_rates"].itertuples(index=False):
    print(f"{row.label}: {row.sr_pct:.1f}% ({row.successes}/{row.identities})")
for row in tables["paired_effects"].itertuples(index=False):
    print(f"{row.comparison}: {row.delta_pp:+.2f} pp "
          f"[{row.ci_low_pp:+.2f}, {row.ci_high_pp:+.2f}] "
          f"(F→S {row.F_to_S}, S→F {row.S_to_F})")
print()
print("Interpretation order:")
print("1. Check the 5-step x1 control against historical 10-step stock.")
print("2. Check whether choosing among three queries beats 5-step x1.")
print("3. Check whether refining the selected query beats selection alone.")
print("4. Inspect first-10 versus full-50 diversity; tail diversity alone may not help execution.")
print("5. Treat this reused development cohort and within-trajectory diagnostics as exploratory.")
print()
print("Saved CSV tables and PNG plots:", OUTPUT.resolve())"""),
    ], "61_analyze_five_step_diversity_pro220.ipynb")
    document["metadata"].pop("accelerator", None)
    return document


def main():
    path = ROOT / "notebooks" / "61_analyze_five_step_diversity_pro220.ipynb"
    write_notebook(path, analysis_notebook())
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
