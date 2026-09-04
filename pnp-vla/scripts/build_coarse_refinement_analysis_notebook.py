"""Generate notebook 63 for the single-query 5/3-step refinement follow-up."""
from nb_common import ROOT, bootstrap, code, md, notebook, write_notebook


def analysis_notebook():
    document = notebook([
        md("""# 63 - Single-query coarse-refinement PRO220 analysis

Combines workers 62-0 and 62-1 and reuses the four exact historical arms printed by those
workers. All seven arms use the same checkpoint, frozen identities, episode seeds, 50-action
predictions, and **10 executed actions before replanning**.

The primary result is whole-matched-cohort SR with paired transitions and per-suite changes.
The notebook also downloads the new arms' small saved uncertainty-profile blobs to analyze
U10/U20/U50, contraction, failure detection, and compute. It does not download trajectories,
videos, observations, or generated-action blobs. Leave preview mode on during collection; switch
to strict mode for the final 220-identity analysis."""),
        md("## 1. Setup"),
        code(bootstrap(extras="analysis", setup_env=False)),
        md("## 2. Load and exact-match all seven arms"),
        code("""from pathlib import Path
import json
from IPython.display import display, Image
from tqdm.auto import tqdm

from analysis.coarse_refinement import (
    ARMS, LABELS, KEYS, fetch_coarse_refinement_rows, match_cohort, analyze_success,
    load_probe_artifacts, probe_tables, success_figures, probe_figures, compute_figure)
from pnp.coarse_refinement_experiment import COARSE_REFINEMENT_EXPERIMENT
from pnp.store import SupabaseStore

EXPERIMENT = COARSE_REFINEMENT_EXPERIMENT
REQUIRE_FULL_COHORT = False  # True for the final 220-identity result
OUTPUT = Path("coarse_refinement_pro220_outputs")
OUTPUT.mkdir(exist_ok=True)

store = SupabaseStore()
rows, manifest, provenance = fetch_coarse_refinement_rows(store, experiment=EXPERIMENT)
paired, matched, coverage = match_cohort(
    rows, manifest, require_complete=REQUIRE_FULL_COHORT)
tables = analyze_success(paired, matched)

print("FINAL STRICT ANALYSIS" if REQUIRE_FULL_COHORT else "INTERIM MATCHED PREVIEW")
print(f"{len(paired)}/{len(manifest)} matched identities; "
      f"{3 * len(paired)} new + {4 * len(paired)} reused historical rollouts.")
print("Every SR, delta, and historical column uses these exact same identities.")
display(coverage)
if not REQUIRE_FULL_COHORT:
    print("Incomplete identities are excluded from EVERY arm. Set REQUIRE_FULL_COHORT=True "
          "after both workers finish.")
print("Verified source:", provenance["source_id"])

paired.to_csv(OUTPUT / "matched_episode_outcomes.csv", index=False)
matched[KEYS + ["arm", "rollout_id", "run_id", "config_hash", "success"]].to_csv(
    OUTPUT / "matched_rollout_ids.csv", index=False)
coverage.to_csv(OUTPUT / "coverage.csv", index=False)
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
for name, frame in tables.items():
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)"""),
        md("""## 3. Primary result: success rates and paired effects

The 10-step reference is the measurement-only stock arm from the direct-U20-gradient pilot:
no refinement, gradient steering, selection, or correction changed its executed actions.
The targeted paired contrasts isolate five-step refinement, three-step refinement, the three-step
coarse control, and three versus five integration steps. F_to_S and S_to_F count identity-level
outcome flips. Confidence intervals resample identities, not rollout boundaries."""),
        code("""display(tables["arm_success_rates"][[
    "label", "identities", "successes", "sr_pct", "ci_low_pct", "ci_high_pct"]])
display(tables["paired_effects"][[
    "comparison", "identities", "baseline_sr_pct", "condition_sr_pct", "delta_pp",
    "ci_low_pp", "ci_high_pp", "F_to_S", "S_to_F", "paired_p_value"]])
display(tables["suite_effects"][[
    "suite", "comparison", "identities", "delta_pp", "F_to_S", "S_to_F"]])
for path in success_figures(tables, OUTPUT / "figures"):
    display(Image(filename=str(path)))"""),
        md("""## 4. U10/U20/U50 and contraction

This cell downloads only the ahats_path NPZ blobs from the three new arms. Despite the legacy
column name, these workers saved **uncertainty profiles, not a-hat action stacks**. Scores are
averaged within each episode before cross-episode summaries, so longer failures do not receive
extra weight. Each probe is labeled by both zero-based Euler index and flow time because step 2
is not the same flow time in three- and five-step schedules.

Refined-arm U is measured along the invasive refined solve; it is not an independent measurement
of the final action. Cross-arm uncertainty scales are descriptive and should not replace the
paired success result."""),
        code("""probe_records = load_probe_artifacts(store, matched, progress=tqdm)
probe_results = probe_tables(probe_records)
tables.update(probe_results)
for name, frame in probe_results.items():
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)

display(tables["probe_summary"][[
    "probe", "outcome", "identities", "u10", "u20", "u50",
    "contraction10", "contraction20", "contraction50"]])
display(tables["failure_detection_auc"][[
    "probe", "horizon", "n", "failures", "roc_auc", "pr_auc"]])
for path in probe_figures(tables, OUTPUT / "figures"):
    display(Image(filename=str(path)))"""),
        md("## 5. Compute diagnostics"),
        code("""display(tables["compute"])
display(Image(filename=str(compute_figure(tables, OUTPUT / "figures"))))
print("Inference time depends on worker hardware and is descriptive. VF evaluations are the "
      "cleaner algorithmic-compute comparison. Historical VF counts are omitted because their "
      "instrumentation is not directly comparable.")"""),
        md("## 6. Compact readout"),
        code("""print(f"Matched identities: {len(paired)}/{len(manifest)}")
for row in tables["arm_success_rates"].itertuples(index=False):
    print(f"{row.label}: {row.sr_pct:.1f}% ({row.successes}/{row.identities})")
print()
for row in tables["paired_effects"].itertuples(index=False):
    print(f"{row.comparison}: {row.delta_pp:+.2f} pp "
          f"[{row.ci_low_pp:+.2f}, {row.ci_high_pp:+.2f}] "
          f"(F→S {row.F_to_S}, S→F {row.S_to_F})")
print()
print("Interpretation order:")
print("1. Compare 5-step x1 + refine with historical 5-step x1 (refinement contribution).")
print("2. Compare 3-step x1 + refine with 3-step x1 (refinement contribution).")
print("3. Compare 3-step x1 with stock 10-step (coarse-integration damage or benefit).")
print("4. Compare 3-step and 5-step refinement, then inspect compute and U profiles.")
print("5. Treat this reused PRO220 development cohort as exploratory, not sealed confirmation.")
print("Saved CSV tables and PNG plots:", OUTPUT.resolve())"""),
    ], "63_analyze_coarse_single_refinement_pro220.ipynb")
    document["metadata"].pop("accelerator", None)
    return document


def main():
    path = ROOT / "notebooks" / "63_analyze_coarse_single_refinement_pro220.ipynb"
    write_notebook(path, analysis_notebook())
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
