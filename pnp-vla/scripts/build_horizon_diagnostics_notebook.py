"""Build notebook 42: full-cohort worker-41 horizon diagnostics."""
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "notebooks" / "42_analyze_source_horizon_diagnostics.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(r'''# 42 — Full-cohort LIBERO-PRO U10/U20/U50 and contraction analysis

This notebook analyzes the **1,300 diagnostic episodes completed by workers 40/41**. The executed policy always replans after 10 actions; U10, U20, and U50 refer to how many positions of each generated 50-action chunk enter the uncertainty statistic.

It first reports pure failure-detection and contraction results from the worker-41 diagnostic arm. A later, explicitly labeled section exact-matches these diagnostic scores to the previously collected corrected 10-action baseline/refinement outcomes for retrospective window studies. Every window-policy SR uses all 1,300 episodes in its denominator.'''),
    code(r'''EXTRAS = 'analysis'
SETUP_ENV = False
import urllib.request
exec(urllib.request.urlopen('https://raw.githubusercontent.com/ArjunS07/cs159-sp26/main/pnp-vla/scripts/colab_bootstrap.py').read().decode())'''),
    markdown('## Configuration and imports'),
    code(r'''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve

from analysis.horizon_diagnostics import (
    HORIZONS, failure_auc_table, load_horizon_artifacts,
    pair_diagnostics_with_historical, prefix_failure_auc_table,
    prefix_feature_table, quantile_outcome_curve,
    select_historical_10_action_arms, validate_diagnostic_cohort)
from analysis.suffix_sensitivity import (
    apply_window_by_suite, bootstrap_rank_auc, summarize_pair,
    threshold_sweep, top_windows, window_sweep)
from pnp.config import Method, RolloutConfig
from pnp.diversity import (
    SOURCE_ACTION_HORIZON_EXPERIMENT,
    SOURCE_HORIZON_MULTI_QUERY_EXPERIMENT)
from pnp.store import SupabaseStore

EXPECTED_IDENTITIES = 1300
REQUIRE_COMPLETE = True
GRID_SIZE = 25
MIN_SELECTED = 25
OUTPUT = Path('source_horizon_diagnostics_outputs')
CACHE = OUTPUT / 'cache'
OUTPUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
store = SupabaseStore()'''),
    markdown('## Load and validate the exact diagnostic cohort'),
    code(r'''diagnostic_config = RolloutConfig(
    pnp_steps=(3, 4), pnp_k=5, n_action_steps=10)
diagnostic_hash = store.config_hash(
    store._logical_key(Method.UNCERTAINTY, diagnostic_config))
rows = pd.DataFrame(store.fetch_all(
    'rollouts', '*', configure=lambda query: query.eq(
        'experiment', SOURCE_HORIZON_MULTI_QUERY_EXPERIMENT).eq(
        'method', Method.UNCERTAINTY), order_by=('rollout_id',)))
rows = rows[rows.config_hash.eq(diagnostic_hash)].copy()
diagnostic = validate_diagnostic_cohort(
    rows, expected_identities=EXPECTED_IDENTITIES,
    require_complete=REQUIRE_COMPLETE)
coverage = (diagnostic.groupby('suite', sort=True)
    .agg(episodes=('rollout_id', 'size'), successes=('success', 'sum'))
    .reset_index())
coverage['success_rate_pct'] = 100 * coverage.successes / coverage.episodes
print(f'Validated {len(diagnostic)} exact worker-41 diagnostic identities; '
      f'{diagnostic.suite.nunique()} suites; config hash {diagnostic_hash}.')
display(coverage)
assert len(diagnostic) == EXPECTED_IDENTITIES
assert coverage.episodes.eq(100).all(), coverage'''),
    markdown('## Download and decode U/contraction artifacts'),
    code(r'''cache_files = {
    'features': CACHE / 'features.pkl', 'records': CACHE / 'records.pkl',
    'positions': CACHE / 'positions.pkl', 'iterations': CACHE / 'iterations.pkl'}
if all(path.exists() for path in cache_files.values()):
    features = pd.read_pickle(cache_files['features'])
    records = pd.read_pickle(cache_files['records'])
    positions = pd.read_pickle(cache_files['positions'])
    iterations = pd.read_pickle(cache_files['iterations'])
    print('Loaded decoded worker-41 artifacts from local cache.')
else:
    features, records, positions, iterations = load_horizon_artifacts(
        store, diagnostic, progress=tqdm)
    features.to_pickle(cache_files['features'])
    records.to_pickle(cache_files['records'])
    positions.to_pickle(cache_files['positions'])
    iterations.to_pickle(cache_files['iterations'])
    print('Downloaded, decoded, and cached all artifacts.')
assert len(features) == EXPECTED_IDENTITIES
print({'episode_features': features.shape, 'probe_records': records.shape,
       'position_rows': positions.shape, 'iteration_rows': iterations.shape})
features.to_csv(OUTPUT / 'worker41_episode_features.csv', index=False)'''),
    markdown('## Baseline success and uncertainty summaries'),
    code(r'''u_episode_scores = [f'u{h}_episode' for h in HORIZONS]
u_first_scores = [f'u{h}_first_chunk' for h in HORIZONS]
summary = pd.DataFrame([{
    'episodes': len(features),
    'success_rate_pct': 100 * features.success.mean(),
    **{f'{score}_mean': features[score].mean()
       for score in u_episode_scores + u_first_scores}}])
display(summary)

suite_sr = (features.groupby('suite', sort=True).success
            .agg(['size', 'sum', 'mean']).reset_index())
suite_sr['success_rate_pct'] = 100 * suite_sr['mean']
fig, ax = plt.subplots(figsize=(13, 5))
labels = suite_sr.suite.str.removeprefix('libero_')
ax.bar(labels, suite_sr.success_rate_pct, color='#4C78A8')
ax.set(ylabel='Success rate (%)', ylim=(0, 105),
       title='Worker-41 diagnostic baseline: corrected 10-action execution')
ax.tick_params(axis='x', rotation=40); ax.grid(axis='y', alpha=.2)
fig.tight_layout(); fig.savefig(OUTPUT / 'baseline_success_by_suite.png', dpi=180)
plt.show()'''),
    markdown('## U10/U20/U50 failure AUC — full episode and first observation'),
    code(r'''u_scores = u_episode_scores + u_first_scores
u_auc = failure_auc_table(features, u_scores, n_boot=2000)
pooled_u_auc = u_auc[u_auc.suite.eq('pooled')].copy()
print('Pooled failure AUC (larger uncertainty predicts failure)')
display(pooled_u_auc)
print('Per-suite failure AUC')
display(u_auc[~u_auc.suite.eq('pooled')])
u_auc.to_csv(OUTPUT / 'uncertainty_failure_auc.csv', index=False)

labels_by_score = {
    'u10_episode': 'U10, full episode', 'u20_episode': 'U20, full episode',
    'u50_episode': 'U50, full episode',
    'u10_first_chunk': 'U10, first chunk',
    'u20_first_chunk': 'U20, first chunk',
    'u50_first_chunk': 'U50, first chunk'}
plot_auc = u_auc[
    ~u_auc.suite.eq('pooled') & u_auc.score_name.isin(u_episode_scores)
    & u_auc.failure_auc.notna()].copy()
suite_order = sorted(plot_auc.suite.unique())
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
for offset, score in zip((-.18, 0, .18), u_episode_scores):
    group = plot_auc[plot_auc.score_name.eq(score)].set_index('suite').reindex(suite_order)
    valid = group.failure_auc.notna().to_numpy()
    y = np.arange(len(suite_order))[valid] + offset
    axes[0].errorbar(group.failure_auc.to_numpy()[valid], y,
        xerr=np.vstack((group.failure_auc.to_numpy()[valid] - group.auc_ci_low.to_numpy()[valid],
                        group.auc_ci_high.to_numpy()[valid] - group.failure_auc.to_numpy()[valid])),
        fmt='o', capsize=2, label=labels_by_score[score])
axes[0].set_yticks(np.arange(len(suite_order)),
                   [name.removeprefix('libero_') for name in suite_order])
axes[0].axvline(.5, color='black', linestyle='--', linewidth=1)
axes[0].set(xlim=(0, 1), xlabel='Failure ROC-AUC (95% bootstrap CI)',
            title='Failure AUC by suite')
axes[0].legend(fontsize=8); axes[0].grid(axis='x', alpha=.2)
failures = (~features.success.astype(bool)).astype(int).to_numpy()
for score in u_episode_scores:
    fpr, tpr, _ = roc_curve(failures, features[score].to_numpy(float))
    auc = pooled_u_auc[pooled_u_auc.score_name.eq(score)].failure_auc.iloc[0]
    axes[1].plot(fpr, tpr, label=f'{labels_by_score[score]}: AUC {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', label='chance')
axes[1].set(xlabel='False-positive rate', ylabel='True-positive rate',
            title='Pooled failure ROC')
axes[1].legend(); axes[1].grid(alpha=.2)
fig.savefig(OUTPUT / 'uncertainty_failure_auc_and_roc.png', dpi=180)
plt.show()'''),
    markdown('## How many observation chunks are needed?'),
    code(r'''prefix = prefix_feature_table(records, features, max_chunks=8)
prefix_auc = prefix_failure_auc_table(prefix, n_boot=2000)
prefix_auc.to_csv(OUTPUT / 'first_k_chunk_auc.csv', index=False)
print('All rows below retain all 1,300 episodes; early-completed episodes use every available chunk.')
display(prefix_auc)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for axis, score_type, title in (
        (axes[0], 'uncertainty', 'Uncertainty predicts failure'),
        (axes[1], 'negative_contraction', 'Weak/non-contraction predicts failure')):
    selected = prefix_auc[prefix_auc.score_type.eq(score_type)]
    for horizon in HORIZONS:
        group = selected[selected.action_horizon.eq(horizon)]
        axis.plot(group.first_k_chunks, group.failure_auc, marker='o',
                  label=f'first {horizon} actions')
    axis.axhline(.5, color='black', linestyle='--')
    axis.set(xlabel='First k observation chunks included', ylabel='Failure ROC-AUC',
             ylim=(.35, 1), title=title)
    axis.legend(); axis.grid(alpha=.2)
fig.savefig(OUTPUT / 'first_k_chunk_auc.png', dpi=180)
plt.show()'''),
    markdown('## Consecutive uncertainty contraction'),
    code(r'''contraction_scores = [f'contraction{h}_episode' for h in HORIZONS]
for score in contraction_scores:
    features[f'negative_{score}'] = -features[score]
contraction_failure_scores = [f'negative_{score}' for score in contraction_scores]
contraction_auc = failure_auc_table(
    features, contraction_failure_scores, n_boot=3000)
print('Failure AUC: larger score means weaker/more-negative contraction')
display(contraction_auc[contraction_auc.suite.eq('pooled')])
contraction_auc.to_csv(OUTPUT / 'contraction_failure_auc.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for horizon, color in zip(HORIZONS, ('#4C78A8', '#F58518', '#54A24B')):
    curve = quantile_outcome_curve(
        features, score_column=f'contraction{horizon}_episode')
    rate = 100 * curve.outcome_rate
    sem = 100 * np.sqrt(curve.outcome_rate * (1 - curve.outcome_rate) / curve.episodes)
    axes[0].errorbar(curve.score_mean, rate, yerr=sem, marker='o', capsize=3,
                     color=color, label=f'first {horizon} actions')
axes[0].set(xlabel='Mean contraction within score quintile',
            ylabel='Success rate (%)',
            title='Does stronger contraction predict success?')
axes[0].legend(); axes[0].grid(alpha=.2)

trajectory = (iterations.groupby(['success', 'horizon', 'perturbation_pair'])
    .disagreement.agg(['mean', 'std', 'count']).reset_index())
trajectory['sem'] = trajectory['std'] / np.sqrt(trajectory['count'])
for success, linestyle, label_prefix in ((True, '-', 'success'), (False, '--', 'failure')):
    for horizon, color in zip(HORIZONS, ('#4C78A8', '#F58518', '#54A24B')):
        group = trajectory[
            trajectory.success.eq(success) & trajectory.horizon.eq(horizon)]
        axes[1].plot(group.perturbation_pair + 1, group['mean'], marker='o',
                     linestyle=linestyle, color=color,
                     label=f'{label_prefix}, first {horizon}')
axes[1].set(xlabel='Consecutive perturbation pair',
            ylabel='Mean disagreement',
            title='Disagreement across the K=5 perturbations')
axes[1].legend(fontsize=8, ncol=2); axes[1].grid(alpha=.2)
fig.savefig(OUTPUT / 'contraction_success_and_trajectory.png', dpi=180)
plt.show()'''),
    markdown('## Action-position profile'),
    code(r'''position_summary = (positions.groupby(['success', 'action_position'])
    .uncertainty.agg(['mean', 'std', 'count']).reset_index())
position_summary['sem'] = position_summary['std'] / np.sqrt(position_summary['count'])
fig, ax = plt.subplots(figsize=(12, 5))
for success, group in position_summary.groupby('success', sort=False):
    color = '#4C78A8' if bool(success) else '#E45756'
    label = 'successful episodes' if bool(success) else 'failed episodes'
    ax.plot(group.action_position, group['mean'], color=color, label=label)
    ax.fill_between(group.action_position, group['mean'] - group['sem'],
                    group['mean'] + group['sem'], color=color, alpha=.16)
ax.axvline(9.5, color='black', linestyle='--', label='10 executed actions')
ax.axvline(19.5, color='#9467BD', linestyle=':', label='U20 boundary')
ax.set(xlabel='Action position in generated 50-action chunk',
       ylabel='Mean uncertainty',
       title='Uncertainty by predicted action position and episode outcome')
ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUTPUT / 'uncertainty_by_action_position.png', dpi=180)
plt.show()'''),
    markdown(r'''## Retrospective pairing to historical refinement

The cells below are **not additional worker-41 rollouts**. They exact-match the worker-41 scores to the corrected historical 10-action unrefined and always-refined outcomes by suite, task, episode index, and initialization hash. The consistency table explicitly reports whether the separately collected unrefined outcome agrees with worker 41.'''),
    code(r'''historical_rows = pd.DataFrame(store.fetch_all(
    'rollouts', '*', configure=lambda query: query.eq(
        'experiment', SOURCE_ACTION_HORIZON_EXPERIMENT).in_(
        'method', [Method.UNCERTAINTY, Method.REFINEMENT]),
    order_by=('rollout_id',)))
historical_arms = select_historical_10_action_arms(
    store, historical_rows, expected_identities=EXPECTED_IDENTITIES)
paired = pair_diagnostics_with_historical(features, historical_arms)
assert len(paired) == EXPECTED_IDENTITIES
consistency = pd.DataFrame([{
    'matched_identities': len(paired),
    'worker41_vs_historical_baseline_outcome_matches':
        int(paired.diagnostic_matches_historical_baseline.sum()),
    'outcome_match_pct': 100 * paired.diagnostic_matches_historical_baseline.mean(),
    'outcome_mismatches': int((~paired.diagnostic_matches_historical_baseline).sum()),
}])
display(consistency)
consistency.to_csv(OUTPUT / 'historical_baseline_consistency.csv', index=False)
overall_refinement, suite_refinement = summarize_pair(paired)
print('Historical always-on refinement vs corrected historical baseline')
display(overall_refinement)
display(suite_refinement)
overall_refinement.to_csv(OUTPUT / 'historical_refinement_overall.csv', index=False)
suite_refinement.to_csv(OUTPUT / 'historical_refinement_by_suite.csv', index=False)'''),
    markdown('## Exploratory uncertainty windows and one-sided thresholds'),
    code(r'''window_scores = u_episode_scores + u_first_scores
all_windows, all_top, all_thresholds = [], [], []
for score in window_scores:
    score_values = paired[score].dropna().to_numpy(float)
    lower_max = max(.06, float(np.quantile(score_values, .90)))
    upper_max = max(.08, float(np.quantile(score_values, .995)))
    sweep = window_sweep(
        paired, score_column=score, grid_size=GRID_SIZE,
        min_selected=MIN_SELECTED, lower_max=lower_max,
        upper_max=upper_max)
    all_windows.append(sweep)
    all_top.append(top_windows(sweep, n=10))
    all_thresholds.append(threshold_sweep(
        paired, score_column=score, grid_size=41,
        min_selected=MIN_SELECTED, threshold_max=upper_max))
window_results = pd.concat(all_windows, ignore_index=True)
top_windows_table = pd.concat(all_top, ignore_index=True)
threshold_results = pd.concat(all_thresholds, ignore_index=True)
best_windows = (top_windows_table.sort_values(['score_name', 'rank'])
                .groupby('score_name', sort=False).head(1).reset_index(drop=True))
best_thresholds = (threshold_results[threshold_results.eligible]
    .sort_values(['score_name', 'delta_pp', 'episodes_refined'],
                 ascending=[True, False, False])
    .groupby('score_name', sort=False).head(1).reset_index(drop=True))
print('Best bounded window per uncertainty score. delta_pp uses all 1,300 episodes.')
display(best_windows[[
    'score_name', 'episodes_in_sr_denominator', 'lower', 'upper',
    'episodes_refined', 'window_policy_sr', 'delta_pp',
    'selected_F_to_S', 'selected_S_to_F']])
print('Best U >= threshold per uncertainty score. delta_pp uses all 1,300 episodes.')
display(best_thresholds[[
    'score_name', 'episodes_in_sr_denominator', 'threshold',
    'episodes_refined', 'threshold_policy_sr', 'delta_pp',
    'selected_F_to_S', 'selected_S_to_F']])
window_results.to_csv(OUTPUT / 'window_sweep.csv', index=False)
top_windows_table.to_csv(OUTPUT / 'top_windows.csv', index=False)
threshold_results.to_csv(OUTPUT / 'threshold_sweep.csv', index=False)'''),
    code(r'''fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
for axis, score in zip(axes.flat, window_scores):
    sweep = window_results[window_results.score_name.eq(score)]
    pivot = sweep.pivot(index='lower', columns='upper', values='delta_pp').sort_index()
    limit = max(1, np.nanmax(np.abs(pivot.to_numpy())))
    image = axis.imshow(pivot.to_numpy(), aspect='auto', origin='lower',
                        cmap='RdYlGn', vmin=-limit, vmax=limit)
    yi = np.linspace(0, len(pivot.index) - 1, 5).astype(int)
    xi = np.linspace(0, len(pivot.columns) - 1, 6).astype(int)
    axis.set_yticks(yi, [f'{pivot.index[i]:.3f}' for i in yi])
    axis.set_xticks(xi, [f'{pivot.columns[i]:.3f}' for i in xi], rotation=35)
    axis.set(xlabel='Upper U bound', ylabel='Lower U bound', title=score)
    fig.colorbar(image, ax=axis, label='Whole-cohort SR change (pp)')
fig.suptitle('Retrospective refinement-window sweep — all 1,300 episodes in denominator')
fig.savefig(OUTPUT / 'window_sweep_heatmaps.png', dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(best_windows))
ax.bar(x, best_windows.delta_pp, color='#54A24B')
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x, best_windows.score_name, rotation=30, ha='right')
ax.set(ylabel='Best exploratory whole-cohort SR change (pp)',
       title='Best refinement window by uncertainty definition')
ax.grid(axis='y', alpha=.2)
fig.tight_layout(); fig.savefig(OUTPUT / 'best_window_by_score.png', dpi=180)
plt.show()'''),
    markdown('## Optimal window by suite and correction-prediction signals'),
    code(r'''best_episode = (best_windows[best_windows.score_name.isin(u_episode_scores)]
    .sort_values(['delta_pp', 'episodes_refined'], ascending=[False, False]).iloc[0])
optimal_suite = apply_window_by_suite(
    paired, score_column=best_episode.score_name,
    lower=best_episode.lower, upper=best_episode.upper)
print(f"Best full-episode rule: {best_episode.score_name}, "
      f"[{best_episode.lower:.4f}, {best_episode.upper:.4f}], "
      f"whole-cohort delta {best_episode.delta_pp:+.2f} pp")
display(optimal_suite)
fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
labels = optimal_suite.suite.str.removeprefix('libero_')
x = np.arange(len(optimal_suite)); width = .38
axes[0].bar(x - width/2, optimal_suite.baseline_sr_pct, width,
            label='historical unrefined', color='#4C78A8')
axes[0].bar(x + width/2, optimal_suite.window_policy_sr_pct, width,
            label='window-gated refinement', color='#F58518')
axes[0].set_xticks(x, labels, rotation=40, ha='right')
axes[0].set(ylabel='Success rate (%)', ylim=(0, 105), title='SR by suite')
axes[0].legend(); axes[0].grid(axis='y', alpha=.2)
colors = np.where(optimal_suite.window_minus_baseline_pp >= 0, '#54A24B', '#E45756')
axes[1].bar(x, optimal_suite.window_minus_baseline_pp, color=colors)
axes[1].axhline(0, color='black', linewidth=1)
axes[1].axhline(best_episode.delta_pp, color='#9467BD', linestyle='--',
                label=f'overall {best_episode.delta_pp:+.2f} pp')
axes[1].set_xticks(x, labels, rotation=40, ha='right')
axes[1].set(ylabel='Window policy minus baseline (pp)', title='Whole-cohort SR change')
axes[1].legend(); axes[1].grid(axis='y', alpha=.2)
fig.savefig(OUTPUT / 'optimal_window_by_suite.png', dpi=180)
plt.show()
optimal_suite.to_csv(OUTPUT / 'optimal_window_by_suite.csv', index=False)

failed = paired[~paired.baseline_success].copy()
correction_rows = []
correction_scores = u_episode_scores + contraction_scores
for score in correction_scores:
    auc, low, high = bootstrap_rank_auc(
        failed.condition_success, failed[score], n_boot=3000)
    correction_rows.append({
        'score_name': score, 'historical_baseline_failures': len(failed),
        'corrected_by_refinement': int(failed.condition_success.sum()),
        'correction_auc': auc, 'auc_ci_low': low, 'auc_ci_high': high})
correction_auc = pd.DataFrame(correction_rows)
print('Among historical baseline failures, can the worker-41 signal predict F->S correction?')
display(correction_auc)
correction_auc.to_csv(OUTPUT / 'correction_auc.csv', index=False)'''),
    markdown(r'''## Interpretation before revisiting the Q corrector

Use the pooled/per-suite U-horizon AUC and first-k curves to choose the failure-detection signal. Use the correction-AUC table separately to determine whether that signal identifies **correctable** failures rather than merely failures. The Q-corrector collection should then stratify clean training episodes by these predeclared signals; it should not train on LIBERO-PRO outcomes if PRO remains the held-out robustness benchmark.'''),
    code(r'''print('Worker-41 diagnostic rows:', len(features))
print('Historical exact-matched rows:', len(paired))
print('Diagnostic/historical unrefined outcome agreement:',
      f'{100 * paired.diagnostic_matches_historical_baseline.mean():.2f}%')
print('\nPooled uncertainty failure AUC:')
display(pooled_u_auc[['score_name', 'failure_auc', 'auc_ci_low', 'auc_ci_high']])
print('\nBest whole-cohort retrospective windows:')
display(best_windows[['score_name', 'lower', 'upper', 'episodes_refined', 'delta_pp']])
print('\nAll outputs saved to:', OUTPUT.resolve())'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
