"""Build notebook 44: corrected standard-LIBERO horizon diagnostics."""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "notebooks" / "44_analyze_standard_libero_horizon_diagnostics.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    md(r'''# 44 — Corrected standard-LIBERO U10/U20/U50 and contraction analysis

This notebook analyzes the 400 uncertainty-only episodes from workers 43. Every rollout generates 50 actions, executes 10, and uses a no-op K=5 probe at Euler steps (3,4). It measures failure detection and Q-corrector data readiness without mixing in LIBERO-PRO training outcomes.

A final, explicitly retrospective section exact-matches these diagnostic scores to the older corrected 10-action refine-last (4,5) outcomes. Those rows are useful for screening signals, but they are not interventions collected by workers 43.'''),
    code(r'''EXTRAS = 'analysis'
SETUP_ENV = False
import urllib.request
exec(urllib.request.urlopen('https://raw.githubusercontent.com/ArjunS07/cs159-sp26/main/pnp-vla/scripts/colab_bootstrap.py').read().decode())'''),
    md('## Configuration and exact cohort validation'),
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
    validate_diagnostic_cohort)
from analysis.suffix_sensitivity import (
    apply_window_by_suite, bootstrap_rank_auc, summarize_pair,
    top_windows, window_sweep)
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS
from pnp.experiments import (
    LIBERO_10STEP_EXPERIMENT, LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT,
    build_full_methods, build_libero_horizon_diagnostic_methods)
from pnp.store import SupabaseStore

EXPECTED_IDENTITIES = 400
GRID_SIZE = 25
MIN_SELECTED = 20
OUTPUT = Path('standard_libero_horizon_outputs')
CACHE = OUTPUT / 'cache'
OUTPUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
store = SupabaseStore()

method, diagnostic_config = build_libero_horizon_diagnostic_methods()[0]
diagnostic_hash = store.config_hash(store._logical_key(method, diagnostic_config))
rows = pd.DataFrame(store.fetch_all(
    'rollouts', '*', configure=lambda query: query.eq(
        'experiment', LIBERO_HORIZON_DIAGNOSTICS_EXPERIMENT).eq(
        'method', Method.UNCERTAINTY).eq('config_hash', diagnostic_hash),
    order_by=('rollout_id',)))
diagnostic = validate_diagnostic_cohort(
    rows, expected_identities=EXPECTED_IDENTITIES, require_complete=True)
assert diagnostic.suite.nunique() == 4
coverage = (diagnostic.groupby('suite', sort=True)
    .agg(episodes=('rollout_id', 'size'), successes=('success', 'sum'),
         pcp_paths=('pcp_chunks_path', lambda value: value.notna().sum()),
         trajectory_paths=('trajectory_path', lambda value: value.notna().sum()),
         uncertainty_artifacts=('ahats_path', lambda value: value.notna().sum()))
    .reset_index())
coverage['success_rate_pct'] = 100 * coverage.successes / coverage.episodes
print(f'Validated {len(diagnostic)} exact standard-LIBERO diagnostic rows; '
      f'config hash {diagnostic_hash}.')
display(coverage)
assert len(diagnostic) == EXPECTED_IDENTITIES
assert coverage.episodes.eq(100).all()
assert coverage.pcp_paths.eq(100).all(), 'PCP/Q feature coverage is incomplete'
assert coverage.trajectory_paths.eq(100).all(), 'trajectory coverage is incomplete' '''),
    md('## Decode U10/U20/U50 and contraction artifacts'),
    code(r'''cache_files = {
    'features': CACHE / 'features.pkl', 'records': CACHE / 'records.pkl',
    'positions': CACHE / 'positions.pkl', 'iterations': CACHE / 'iterations.pkl'}
if all(path.exists() for path in cache_files.values()):
    features = pd.read_pickle(cache_files['features'])
    records = pd.read_pickle(cache_files['records'])
    positions = pd.read_pickle(cache_files['positions'])
    iterations = pd.read_pickle(cache_files['iterations'])
    print('Loaded decoded artifacts from local cache.')
else:
    features, records, positions, iterations = load_horizon_artifacts(
        store, diagnostic, progress=tqdm)
    features.to_pickle(cache_files['features']); records.to_pickle(cache_files['records'])
    positions.to_pickle(cache_files['positions']); iterations.to_pickle(cache_files['iterations'])
    print('Downloaded, decoded, and cached all 400 artifacts.')
assert len(features) == EXPECTED_IDENTITIES
print({'features': features.shape, 'probe_records': records.shape,
       'positions': positions.shape, 'iterations': iterations.shape})
features.to_csv(OUTPUT / 'episode_features.csv', index=False)'''),
    md('## Success rate and U10/U20/U50 failure AUC'),
    code(r'''u_episode_scores = [f'u{h}_episode' for h in HORIZONS]
u_first_scores = [f'u{h}_first_chunk' for h in HORIZONS]
u_scores = u_episode_scores + u_first_scores
u_auc = failure_auc_table(features, u_scores, n_boot=3000)
pooled_u = u_auc[u_auc.suite.eq('pooled')].copy()
print('Pooled failure AUC; larger U predicts failure')
display(pooled_u)
print('Per-suite failure AUC')
display(u_auc[~u_auc.suite.eq('pooled')])
u_auc.to_csv(OUTPUT / 'uncertainty_failure_auc.csv', index=False)

suite_sr = (features.groupby('suite', sort=True).success
            .agg(['size', 'sum', 'mean']).reset_index())
suite_sr['success_rate_pct'] = 100 * suite_sr['mean']
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
labels = suite_sr.suite.str.removeprefix('libero_')
axes[0].bar(labels, suite_sr.success_rate_pct, color='#4C78A8')
axes[0].set(ylabel='Success rate (%)', ylim=(0, 105),
            title='Standard-LIBERO diagnostic baseline')
axes[0].tick_params(axis='x', rotation=25); axes[0].grid(axis='y', alpha=.2)
failures = (~features.success.astype(bool)).astype(int).to_numpy()
for score, label in zip(u_episode_scores, ('U10', 'U20', 'U50')):
    fpr, tpr, _ = roc_curve(failures, features[score])
    auc = pooled_u[pooled_u.score_name.eq(score)].failure_auc.iloc[0]
    axes[1].plot(fpr, tpr, label=f'{label}: AUC {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', label='chance')
axes[1].set(xlabel='False-positive rate', ylabel='True-positive rate',
            title='Pooled failure ROC')
axes[1].legend(); axes[1].grid(alpha=.2)
fig.savefig(OUTPUT / 'baseline_sr_and_failure_roc.png', dpi=180)
plt.show()'''),
    code(r'''plot_auc = u_auc[
    ~u_auc.suite.eq('pooled') & u_auc.score_name.isin(u_episode_scores)
    & u_auc.failure_auc.notna()].copy()
suites = sorted(plot_auc.suite.unique())
fig, ax = plt.subplots(figsize=(10, 5))
for offset, score, label in zip((-.18, 0, .18), u_episode_scores, ('U10', 'U20', 'U50')):
    group = plot_auc[plot_auc.score_name.eq(score)].set_index('suite').reindex(suites)
    valid = group.failure_auc.notna().to_numpy(); y = np.arange(len(suites))[valid] + offset
    ax.errorbar(group.failure_auc.to_numpy()[valid], y,
        xerr=np.vstack((group.failure_auc.to_numpy()[valid] - group.auc_ci_low.to_numpy()[valid],
                        group.auc_ci_high.to_numpy()[valid] - group.failure_auc.to_numpy()[valid])),
        fmt='o', capsize=3, label=label)
ax.set_yticks(np.arange(len(suites)), [name.removeprefix('libero_') for name in suites])
ax.axvline(.5, color='black', linestyle='--')
ax.set(xlim=(0, 1), xlabel='Failure ROC-AUC (95% bootstrap CI)',
       title='U10/U20/U50 failure AUC by suite')
ax.legend(); ax.grid(axis='x', alpha=.2)
fig.tight_layout(); fig.savefig(OUTPUT / 'failure_auc_by_suite.png', dpi=180)
plt.show()'''),
    md('## First-k observation chunks and consecutive contraction'),
    code(r'''prefix = prefix_feature_table(records, features, max_chunks=8)
prefix_auc = prefix_failure_auc_table(prefix, n_boot=3000)
print('Every first-k row retains all 400 episodes; early successes use all available chunks.')
display(prefix_auc)
prefix_auc.to_csv(OUTPUT / 'first_k_chunk_auc.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for axis, score_type, title in (
        (axes[0], 'uncertainty', 'Uncertainty predicts failure'),
        (axes[1], 'negative_contraction', 'Weak/non-contraction predicts failure')):
    frame = prefix_auc[prefix_auc.score_type.eq(score_type)]
    for horizon in HORIZONS:
        group = frame[frame.action_horizon.eq(horizon)]
        axis.plot(group.first_k_chunks, group.failure_auc, marker='o',
                  label=f'first {horizon} actions')
    axis.axhline(.5, color='black', linestyle='--')
    axis.set(xlabel='First k observation chunks', ylabel='Failure ROC-AUC',
             ylim=(.35, 1), title=title)
    axis.legend(); axis.grid(alpha=.2)
fig.savefig(OUTPUT / 'first_k_chunk_auc.png', dpi=180)
plt.show()'''),
    code(r'''contraction_scores = [f'contraction{h}_episode' for h in HORIZONS]
for score in contraction_scores:
    features[f'negative_{score}'] = -features[score]
contraction_auc = failure_auc_table(
    features, [f'negative_{score}' for score in contraction_scores], n_boot=3000)
print('Failure AUC for weak/more-negative contraction')
display(contraction_auc[contraction_auc.suite.eq('pooled')])
contraction_auc.to_csv(OUTPUT / 'contraction_failure_auc.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for horizon, color in zip(HORIZONS, ('#4C78A8', '#F58518', '#54A24B')):
    curve = quantile_outcome_curve(features, score_column=f'contraction{horizon}_episode')
    sem = 100 * np.sqrt(curve.outcome_rate * (1 - curve.outcome_rate) / curve.episodes)
    axes[0].errorbar(curve.score_mean, 100 * curve.outcome_rate,
                     yerr=sem, marker='o', capsize=3, color=color,
                     label=f'first {horizon} actions')
axes[0].set(xlabel='Mean contraction within score quintile', ylabel='Success rate (%)',
            title='Does stronger contraction predict success?')
axes[0].legend(); axes[0].grid(alpha=.2)
trajectory = (iterations.groupby(['success', 'horizon', 'perturbation_pair'])
    .disagreement.agg(['mean', 'std', 'count']).reset_index())
for success, linestyle, outcome in ((True, '-', 'success'), (False, '--', 'failure')):
    for horizon, color in zip(HORIZONS, ('#4C78A8', '#F58518', '#54A24B')):
        group = trajectory[trajectory.success.eq(success) & trajectory.horizon.eq(horizon)]
        axes[1].plot(group.perturbation_pair + 1, group['mean'], marker='o',
                     linestyle=linestyle, color=color, label=f'{outcome}, {horizon}')
axes[1].set(xlabel='Consecutive perturbation pair', ylabel='Mean disagreement',
            title='Disagreement through K=5 perturbations')
axes[1].legend(fontsize=8, ncol=2); axes[1].grid(alpha=.2)
fig.savefig(OUTPUT / 'contraction_analysis.png', dpi=180)
plt.show()'''),
    md('## Action-position uncertainty profile'),
    code(r'''profile = (positions.groupby(['success', 'action_position']).uncertainty
    .agg(['mean', 'std', 'count']).reset_index())
profile['sem'] = profile['std'] / np.sqrt(profile['count'])
fig, ax = plt.subplots(figsize=(12, 5))
for success, group in profile.groupby('success', sort=False):
    color = '#4C78A8' if bool(success) else '#E45756'
    label = 'success' if bool(success) else 'failure'
    ax.plot(group.action_position, group['mean'], color=color, label=label)
    ax.fill_between(group.action_position, group['mean'] - group['sem'],
                    group['mean'] + group['sem'], color=color, alpha=.16)
ax.axvline(9.5, color='black', linestyle='--', label='executed prefix')
ax.axvline(19.5, color='#9467BD', linestyle=':', label='U20 boundary')
ax.set(xlabel='Action position in generated chunk', ylabel='Mean uncertainty',
       title='Uncertainty by action position and episode outcome')
ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUTPUT / 'uncertainty_position_profile.png', dpi=180)
plt.show()'''),
    md(r'''## Retrospective match to the corrected refine-last (4,5) arm

Workers 43 did not run refinement. This section joins their diagnostic scores to the older corrected 10-action standard-LIBERO baseline and refine-last (4,5) outcomes. Outcome agreement is printed before any window or correction analysis.'''),
    code(r'''historical_rows = pd.DataFrame(store.fetch_all(
    'rollouts', '*', configure=lambda query: query.eq(
        'experiment', LIBERO_10STEP_EXPERIMENT).in_(
        'method', [Method.UNCERTAINTY, Method.REFINEMENT]),
    order_by=('rollout_id',)))
full_methods = build_full_methods()
historical_configs = {
    Method.UNCERTAINTY: next(config for method, config in full_methods
                             if method == Method.UNCERTAINTY),
    Method.REFINEMENT: next(config for method, config in full_methods
                            if method == Method.REFINEMENT
                            and tuple(config.pnp_steps) == (4, 5)),
}
historical_arms = {}
for arm_method, config in historical_configs.items():
    config_hash = store.config_hash(store._logical_key(arm_method, config))
    arm = historical_rows[
        historical_rows.status.eq('completed')
        & historical_rows.method.eq(arm_method)
        & historical_rows.config_hash.eq(config_hash)].copy()
    assert len(arm) == EXPECTED_IDENTITIES, (arm_method, len(arm))
    assert not arm.duplicated(DIVERSITY_PAIR_KEYS).any()
    historical_arms[arm_method] = arm
paired = pair_diagnostics_with_historical(features, historical_arms)
assert len(paired) == EXPECTED_IDENTITIES
consistency = pd.DataFrame([{
    'exact_matched_episodes': len(paired),
    'worker43_vs_historical_baseline_match_pct':
        100 * paired.diagnostic_matches_historical_baseline.mean(),
    'outcome_mismatches': int((~paired.diagnostic_matches_historical_baseline).sum())}])
display(consistency)
overall_refine, suite_refine = summarize_pair(paired)
print('Historical refine-last (4,5) vs historical corrected baseline')
display(overall_refine); display(suite_refine)
consistency.to_csv(OUTPUT / 'historical_consistency.csv', index=False)'''),
    md('## Retrospective windows and Q-corrector screening'),
    code(r'''all_windows, all_top = [], []
for score in u_episode_scores + u_first_scores:
    values = paired[score].dropna().to_numpy(float)
    sweep = window_sweep(
        paired, score_column=score, grid_size=GRID_SIZE,
        min_selected=MIN_SELECTED,
        lower_max=max(.06, float(np.quantile(values, .90))),
        upper_max=max(.08, float(np.quantile(values, .995))))
    all_windows.append(sweep); all_top.append(top_windows(sweep, n=10))
window_results = pd.concat(all_windows, ignore_index=True)
top_window_results = pd.concat(all_top, ignore_index=True)
best_windows = (top_window_results.sort_values(['score_name', 'rank'])
                .groupby('score_name', sort=False).head(1).reset_index(drop=True))
print('Retrospective best windows; every delta uses all 400 episodes.')
display(best_windows[[
    'score_name', 'episodes_in_sr_denominator', 'lower', 'upper',
    'episodes_refined', 'window_policy_sr', 'delta_pp',
    'selected_F_to_S', 'selected_S_to_F']])
window_results.to_csv(OUTPUT / 'retrospective_window_sweep.csv', index=False)
top_window_results.to_csv(OUTPUT / 'retrospective_top_windows.csv', index=False)

failed = paired[~paired.baseline_success].copy()
correction_rows = []
for score in u_episode_scores + contraction_scores:
    auc, low, high = bootstrap_rank_auc(
        failed.condition_success, failed[score], n_boot=3000)
    correction_rows.append({
        'score_name': score, 'baseline_failures': len(failed),
        'corrected_failures': int(failed.condition_success.sum()),
        'correction_auc': auc, 'auc_ci_low': low, 'auc_ci_high': high})
correction_auc = pd.DataFrame(correction_rows)
print('Among baseline failures, can the diagnostic predict refine-last correction?')
display(correction_auc)
correction_auc.to_csv(OUTPUT / 'retrospective_correction_auc.csv', index=False)'''),
    code(r'''task_screen = (features.groupby(['suite', 'task_idx'], sort=True)
    .agg(episodes=('rollout_id', 'size'), success_rate=('success', 'mean'),
         u10=('u10_episode', 'mean'), u20=('u20_episode', 'mean'),
         u50=('u50_episode', 'mean'), contraction20=('contraction20_episode', 'mean'))
    .reset_index())
task_screen['q_hard_task_candidate'] = task_screen.success_rate.between(.10, .90, inclusive='neither')
print('Task-level screening. Q hard-task candidates have empirical SR strictly between 10% and 90%.')
display(task_screen)

q_export = features.merge(
    diagnostic[DIVERSITY_PAIR_KEYS + ['rollout_id', 'pcp_chunks_path', 'trajectory_path']],
    on=DIVERSITY_PAIR_KEYS + ['rollout_id'], validate='one_to_one')
q_export['u20_quartile'] = pd.qcut(
    q_export.u20_episode.rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
q_export.to_csv(OUTPUT / 'q_screening_rollouts.csv', index=False)
task_screen.to_csv(OUTPUT / 'q_screening_tasks.csv', index=False)
print({'q_rows': len(q_export),
       'pcp_feature_rows': int(q_export.pcp_chunks_path.notna().sum()),
       'trajectory_rows': int(q_export.trajectory_path.notna().sum()),
       'logged_correction_steps': [3, 4],
       'important': 'Existing TrainConfig defaults to (7,8); use (3,4) for this dataset.'})'''),
    md(r'''## Q-corrector next step

Use these results to predeclare the failure score and uncertainty stratum. Then rebuild the Q dataset from the stored standard-LIBERO PCP features with `TrainConfig(correction_steps=(3,4))`, splitting at rollout/task level. Keep LIBERO-PRO completely out of Q training and use it only for final robustness evaluation.'''),
    code(r'''print('Diagnostic episodes:', len(features))
print('Historical matched episodes:', len(paired))
print('Outcome agreement:', f'{100 * paired.diagnostic_matches_historical_baseline.mean():.2f}%')
print('\nPooled U failure AUC:')
display(pooled_u[['score_name', 'failure_auc', 'auc_ci_low', 'auc_ci_high']])
print('\nContraction failure AUC:')
display(contraction_auc[contraction_auc.suite.eq('pooled')][
    ['score_name', 'failure_auc', 'auc_ci_low', 'auc_ci_high']])
print('\nAll outputs saved to:', OUTPUT.resolve())'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
