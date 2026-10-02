"""Paired fixed-final comparisons, using saved root predictions."""
import json
from pathlib import Path
import numpy as np

DIRECTORY = Path.home() / "pnp-vla-runs/diagnostics/overnight_20260930"
JOBS = ["mc_roots", "td_n5", "mc_long", "td_long", "mc_roots_late", "mc_roots_contextdrop"]


def paired(first, second, split):
    a = {r["group_id"]: r for r in first["splits"][split]["root_records"]}
    b = {r["group_id"]: r for r in second["splits"][split]["root_records"]}
    if a.keys() != b.keys(): raise ValueError("root cohorts differ")
    values = {"selection": [], "brier": [], "macro_pair": [], "fresh_macro_pair": []}
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["labels"] != y["labels"]: raise ValueError("root labels differ")
        labels = np.asarray(x["labels"])
        values["selection"].append(int(x["selected_success"]) - int(y["selected_success"]))
        values["brier"].append(float(np.mean((np.asarray(x["scores"])-labels)**2) - np.mean((np.asarray(y["scores"])-labels)**2)))
        for field, record in (("macro_pair", "critic_pair_accuracy"), ("fresh_macro_pair", "fresh_pair_accuracy")):
            if x[record] is not None: values[field].append(x[record] - y[record])
    result = {}
    rng = np.random.default_rng(42930)
    for name, rows in values.items():
        array = np.asarray(rows)
        bootstrap = array[rng.integers(0,len(array),size=(10000,len(array)))].mean(1)
        result[name] = {"delta": float(array.mean()), "ci95": np.quantile(bootstrap,[.025,.975]).tolist(), "roots":len(array)}
    return result


def main():
    reports = {name: json.loads((DIRECTORY/(name+"_diagnostics.json")).read_text()) for name in JOBS}
    comparisons = {}
    for a,b in [("mc_roots","mc_long"),("td_n5","td_long"),("mc_roots_late","mc_roots"),("mc_roots_contextdrop","mc_roots")]:
        comparisons[a+"_minus_"+b] = {s:paired(reports[a],reports[b],s) for s in ("train","validation")}
    (DIRECTORY/"paired_comparisons.json").write_text(json.dumps(comparisons,indent=2))
    lines = ["# Overnight critic results — September 30, 2026", "", "All six jobs completed their fixed 4,000-update extensions from the same appropriate 2,000-update MC/TD checkpoint. Stock succeeds on 209/320 validation roots and 832/1,280 training roots. Validation has been repeatedly inspected; comparisons are exploratory.", "", "| Model | Validation successes | Rescues / spoils | Pair accuracy | Brier | Training pair accuracy |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,r in reports.items():
        v=r["splits"]["validation"]["selection"]; t=r["splits"]["train"]["selection"]
        lines.append(f"| {name} | {v['selected_successes']}/320 | {v['rescues']} / {v['spoils']} | {v['within_tree_pair_accuracy']:.3f} | {v['brier']:.4f} | {t['within_tree_pair_accuracy']:.3f} |")
    lines += ["", "## Paired validation differences", "", "Intervals resample entire roots; they do not correct for repeated validation use or multiple model comparisons.", "", "| Comparison | Success delta (percentage points), 95% interval | Brier delta, 95% interval | Macro pair delta, 95% interval |", "| --- | --- | --- | --- |"]
    for name,c in comparisons.items():
        def fmt(field,scale=1):
            v=c["validation"][field]; lo,hi=v["ci95"]
            return f"{v['delta']*scale:+.4f} [{lo*scale:+.4f}, {hi*scale:+.4f}]"
        lines.append(f"| {name} | {fmt('selection',100)} | {fmt('brier')} | {fmt('macro_pair')} |")
    lines += ["", "## Interpretation", "", "No model improves on stock. Training action discrimination stays close to chance despite better return-probability prediction. Five-step TD improves Brier relative to matched one-step TD; it does not improve nearby-action choice. Root-focused MC, late fusion and token dropout give similar held-out ordering but no convincing training action fit.", "", "Astra reviewed the dataset, sampling, targets, matched MC/TD results, and architecture initialization. Its late-fusion audit found an active but weak branch whose contribution varied mainly between roots. Final context-dropout synthesis was not independently reviewed before its usage limit interrupted the last review; subsequent availability is reported in chat.", "", "Next: a matched 32-mixed-training-root MC/BCE conditioning diagnostic. Absolute and stock-relative inputs initially preserve the same predictions and action gradients. Compare fitting against the root-constant entropy floor; do not treat memorization as held-out performance. Notebook 119 controlled action contrasts are a separate data diagnostic. PCP outcome intervention has not yet been executed.", ""]
    (DIRECTORY/"results.md").write_text("\n".join(lines))
    print(json.dumps(comparisons,indent=2))


if __name__ == "__main__": main()
