"""Build the source-only prefix uncertainty / delayed-refinement screening notebook."""
from __future__ import annotations

from nb_common import ROOT, bootstrap, code, md, notebook, write_notebook


DOCUMENT = notebook([
    md(r"""# 28 — Source prefix uncertainty as a delayed refinement gate

This notebook asks one narrow question on the matched source-checkpoint LIBERO-PRO rollouts:

> After observing the first `k` unrefined policy chunks, can their mean PnP uncertainty identify
> episodes for which switching refinement on for the rest of the rollout might help?

It screens both a one-sided high-uncertainty threshold and a bounded uncertainty window for
`k = 1..10`. Every success-rate denominator contains the complete 1,300-episode source cohort.
Episodes ending before `k` cannot fire the gate and retain their baseline outcome.

**Important:** this is a retrospective proxy, not the deployable policy. When a gate fires, the
notebook substitutes the terminal outcome of the paired rollout that used refinement from chunk 0.
A real delayed gate follows the baseline trajectory for `k` chunks and then switches, so its state
at the switch is different. Use this notebook to choose a small number of candidate `(k, gate)`
settings; measure the winner with a new online rollout arm.

The earlier approximate `+2.62 pp` result came from notebook 21: `k=4`, window `0.025–0.080`,
342 selected episodes. Notebook 21 used a whole-episode fallback for short trajectories; this
notebook removes that fallback and keeps those episodes unchanged instead."""),
    md("## 1. Setup"),
    code(bootstrap(extras="analysis", setup_env=False)),
    md("## 2. Load the exact source baseline/refinement cohort"),
    code(r'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from pnp import notebook as nb
from pnp.config import Method, PI05_REPO_ID
from pnp.diversity import DIVERSITY_PAIR_KEYS, analyze_source_prefix_gate_proxy
from pnp.experiments import PRO_EXPANDED_EXPERIMENT, expanded_pro_suites

ctx = nb.setup("source_prefix_gate")
store, OUTPUT = ctx.store, ctx.output
PREFIX_COUNTS = tuple(range(1, 11))
GRID_SIZE = 33
MIN_SELECTED = 25
EPISODES_PER_TASK = 10

source_rows = pd.DataFrame(store.fetch_all(
    "rollouts",
    "rollout_id,suite,task_idx,episode_idx,init_state_hash,method,status,success,"
    "pnp_k,pnp_step_indices,refine_average",
    configure=lambda query: query.eq("experiment", PRO_EXPANDED_EXPERIMENT),
    order_by=("rollout_id",)))
suites = set(expanded_pro_suites())
source_rows = source_rows[
    source_rows.suite.isin(suites) &
    source_rows.episode_idx.astype(int).lt(EPISODES_PER_TASK) &
    source_rows.status.eq("completed") & source_rows.pnp_k.eq(5) &
    source_rows.pnp_step_indices.apply(lambda value: tuple(value or []) == (3, 4)) &
    source_rows.method.isin([Method.UNCERTAINTY, Method.REFINEMENT])].copy()
source_rows = source_rows[
    source_rows.method.eq(Method.UNCERTAINTY) |
    ~source_rows.refine_average.fillna(False).astype(bool)]

assert not source_rows.duplicated(DIVERSITY_PAIR_KEYS + ["method"]).any()
counts = source_rows.groupby(DIVERSITY_PAIR_KEYS).method.nunique()
complete_keys = counts[counts.eq(2)].reset_index()[DIVERSITY_PAIR_KEYS]
source_rows = source_rows.merge(complete_keys, on=DIVERSITY_PAIR_KEYS,
                                validate="many_to_one")
expected = len(suites) * 10 * EPISODES_PER_TASK
assert len(complete_keys) == expected, (
    f"Expected {expected} exact baseline/refinement identities; found {len(complete_keys)}")

observed_ids = source_rows[
    source_rows.method.eq(Method.UNCERTAINTY)].rollout_id.astype(str).tolist()
step_rows = []
for start in range(0, len(observed_ids), 100):
    batch = observed_ids[start:start + 100]
    step_rows.extend(store.fetch_all(
        "pnp_euler_steps", "rollout_id,chunk_idx,euler_step,u_mean",
        configure=lambda query, ids=batch: query.in_("rollout_id", ids),
        order_by=("rollout_id",)))
source_steps = pd.DataFrame(step_rows)
assert not source_steps.empty
print({"experiment": PRO_EXPANDED_EXPERIMENT,
       "matched_episodes": len(complete_keys), "suites": len(suites),
       "methods": sorted(source_rows.method.unique()),
       "probe": "K=5, Euler steps (3,4), last refinement"})'''),
    md("## 3. Full-cohort exploratory sweep"),
    code(r'''tables = analyze_source_prefix_gate_proxy(
    source_rows, source_steps, prefix_counts=PREFIX_COUNTS,
    grid_size=GRID_SIZE, min_selected=MIN_SELECTED)
summary = tables["source_prefix_gate_summary"]
best_threshold = tables["source_prefix_best_threshold"].merge(
    summary[["prefix_count", "baseline_sr", "failure_auc_among_reached"]],
    on="prefix_count", validate="one_to_one")
best_window = tables["source_prefix_best_window"].merge(
    summary[["prefix_count", "baseline_sr"]],
    on="prefix_count", validate="one_to_one")

def percent_table(frame, columns):
    output = frame[columns].copy()
    for column in ("baseline_sr", "proxy_sr_all_episodes"):
        if column in output:
            output[column] = 100 * output[column]
    return output.rename(columns={
        "prefix_count": "first_k_chunks", "n_reached": "episodes_reaching_k",
        "n_selected": "episodes_gate_on", "baseline_sr": "baseline_sr_pct",
        "proxy_sr_all_episodes": "proxy_sr_all_episodes_pct"})

print("Best one-sided rule at each k: switch when prefix U >= threshold")
display(percent_table(best_threshold, [
    "prefix_count", "n_reached", "threshold", "n_selected", "baseline_sr",
    "proxy_sr_all_episodes", "proxy_delta_pp", "selected_F_to_S", "selected_S_to_F",
    "failure_auc_among_reached"]))
print("Best bounded window at each k: switch when lower <= prefix U <= upper")
display(percent_table(best_window, [
    "prefix_count", "n_reached", "lower", "upper", "n_selected", "baseline_sr",
    "proxy_sr_all_episodes", "proxy_delta_pp", "selected_F_to_S", "selected_S_to_F"]))'''),
    md(r"""## 4. Find the sweet spot

The top panel shows the complete one-sided threshold surface. The middle panel compares the best
one-sided threshold and best bounded window at each prefix length. The bottom panel prevents a
misleading late-prefix result: it shows how many episodes still reach the decision and whether
prefix uncertainty predicts baseline failure (AUC 0.5 = chance)."""),
    code(r'''threshold_sweep = tables["source_prefix_threshold_sweep"]
pivot = threshold_sweep.pivot(
    index="prefix_count", columns="threshold", values="proxy_delta_pp").sort_index()
fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
image = axes[0].imshow(pivot.to_numpy(), aspect="auto", origin="lower",
                       cmap="RdYlGn", vmin=-max(1, np.nanmax(abs(pivot.to_numpy()))),
                       vmax=max(1, np.nanmax(abs(pivot.to_numpy()))))
axes[0].set_yticks(range(len(pivot.index)), pivot.index)
tick_idx = np.linspace(0, len(pivot.columns)-1, 9).astype(int)
axes[0].set_xticks(tick_idx, [f"{pivot.columns[i]:.3f}" for i in tick_idx])
axes[0].set(xlabel="High-U threshold", ylabel="First k chunks",
            title="Retrospective whole-cohort SR delta (pp): U >= threshold")
fig.colorbar(image, ax=axes[0], label="Proxy SR change (percentage points)")

x = best_threshold.prefix_count.to_numpy()
axes[1].plot(x, best_threshold.proxy_delta_pp, marker="o", label="best U >= threshold")
axes[1].plot(x, best_window.proxy_delta_pp, marker="o", label="best bounded window")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(xticks=x, xlabel="First k chunks", ylabel="Best proxy SR change (pp)",
            title="Best exploratory gate at each prefix length")
axes[1].legend()

axes[2].plot(summary.prefix_count, 100 * summary.coverage_reached,
             marker="o", color="#4C78A8", label="episodes reaching k (%)")
axes[2].set(xticks=summary.prefix_count, xlabel="First k chunks",
            ylabel="Episodes reaching decision (%)", ylim=(0, 105))
auc_axis = axes[2].twinx()
auc_axis.plot(summary.prefix_count, summary.failure_auc_among_reached,
              marker="s", color="#F58518", label="failure AUC")
auc_axis.axhline(.5, color="#F58518", linestyle="--", alpha=.6)
auc_axis.set(ylabel="AUC: prefix U predicts baseline failure", ylim=(0, 1))
lines = axes[2].lines + auc_axis.lines[:1]
axes[2].legend(lines, [line.get_label() for line in lines], loc="best")
axes[2].set_title("Decision coverage and failure-prediction signal")
plt.show()'''),
    md(r"""## 5. Simple discovery/test split

To reduce the most obvious same-data optimization, the first five initialization indices choose
one rule; the other five estimate that locked rule. This does **not** restore a pristine PRO test
set—we have already inspected this benchmark—but it is more informative than reporting only the
best in-sample cell."""),
    code(r'''def subset_analysis(indices):
    keys = source_rows[source_rows.episode_idx.astype(int).isin(indices)][DIVERSITY_PAIR_KEYS]
    subset_rows = source_rows.merge(keys.drop_duplicates(), on=DIVERSITY_PAIR_KEYS,
                                    validate="many_to_one")
    ids = set(subset_rows[subset_rows.method.eq(Method.UNCERTAINTY)].rollout_id.astype(str))
    subset_steps = source_steps[source_steps.rollout_id.astype(str).isin(ids)]
    return analyze_source_prefix_gate_proxy(
        subset_rows, subset_steps, prefix_counts=PREFIX_COUNTS,
        grid_size=GRID_SIZE, min_selected=max(10, MIN_SELECTED // 2))

discovery = subset_analysis(range(5))
heldout = subset_analysis(range(5, 10))

def best_global(table, rule):
    eligible = table[table.eligible & table.proxy_delta_pp.notna()]
    sort_columns = ["proxy_delta_pp", "n_selected"]
    return eligible.sort_values(sort_columns, ascending=[False, False]).iloc[0]

def evaluate_locked(pairs, rule):
    group = pairs[pairs.prefix_count.eq(int(rule.prefix_count))]
    reached = group.reached_prefix.to_numpy(bool)
    score = group.prefix_u.to_numpy(float)
    if rule.gate_type == "high_threshold":
        selected = reached & (score >= float(rule.threshold))
    else:
        selected = reached & (score >= float(rule.lower)) & (score <= float(rule.upper))
    baseline = group.baseline_success.to_numpy(bool)
    refined = group.always_refined_success.to_numpy(bool)
    policy = np.where(selected, refined, baseline)
    return {"first_k_chunks": int(rule.prefix_count), "gate": rule.gate_type,
            "threshold_or_lower": float(rule.get("threshold", rule.get("lower", np.nan))),
            "upper": float(rule.get("upper", np.nan)), "episodes": len(group),
            "episodes_gate_on": int(selected.sum()),
            "baseline_sr_pct": 100 * baseline.mean(),
            "proxy_sr_pct": 100 * policy.mean(),
            "proxy_delta_pp": 100 * (policy.mean() - baseline.mean()),
            "F_to_S": int((selected & ~baseline & refined).sum()),
            "S_to_F": int((selected & baseline & ~refined).sum())}

locked_rows = []
for table_name, pairs_name in [
        ("source_prefix_threshold_sweep", "source_prefix_gate_pairs"),
        ("source_prefix_window_sweep", "source_prefix_gate_pairs")]:
    rule = best_global(discovery[table_name], table_name)
    result = evaluate_locked(heldout[pairs_name], rule)
    result["chosen_on_discovery_delta_pp"] = float(rule.proxy_delta_pp)
    locked_rows.append(result)
print("Rules chosen only on episode indices 0–4, then evaluated on indices 5–9")
display(pd.DataFrame(locked_rows))'''),
    md(r"""## Interpretation

- Prefer a small `k` whose signal is stable across neighboring thresholds—not a single isolated
  green cell.
- Check `episodes_reaching_k`: large late-prefix deltas can describe only difficult surviving
  episodes even though the SR denominator remains complete.
- Treat the split result as the primary screening readout.
- Once a candidate is chosen, collect a new arm that executes chunks `0..k-1` unrefined and then
  permanently enables `(3,4)` refine-last when the locked gate fires. Only that experiment tests
  the proposed online policy."""),
], "28_analyze_source_prefix_refinement_gate.ipynb")


def main():
    write_notebook(ROOT / "notebooks" / "28_analyze_source_prefix_refinement_gate.ipynb",
                   DOCUMENT)


if __name__ == "__main__":
    main()
