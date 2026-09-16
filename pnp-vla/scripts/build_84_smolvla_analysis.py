"""Build notebook 84 without embedding generated outputs in git."""
from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.strip() + "\n",
    }


cells = [
    markdown("""
# 84 — SmolVLA standard-LIBERO A10 refinement and uncertainty analysis

Exact matched analysis of the two worker-83 arms: stock SmolVLA with measurement-only P&P and
always-on P&P refinement. The notebook validates all 400 identities before reporting paired SR,
failure AUC, prefix diagnostics, contraction, and action-position profiles.

The transition section asks four distinct questions so ordinary failure detection is not confused
with refinement usefulness: does high stock uncertainty predict **any flip**, **F→S rescue among
stock failures**, **S→F harm among stock successes**, or **F→S rather than S→F among discordant
episodes**? Whole-episode scores are retrospective. First-chunk and first-four-chunk scores are
reported separately as candidates for earlier data-collection decisions.
"""),
    code("""
EXTRAS = 'analysis'
SETUP_ENV = False
import urllib.request
exec(urllib.request.urlopen('https://raw.githubusercontent.com/ArjunS07/cs159-sp26/main/pnp-vla/scripts/colab_bootstrap.py').read().decode())
"""),
    markdown("""
## Configuration, exact arm selection, and cohort audit

The config hashes are derived from the same method builder used by workers 83. This prevents stale
rows or similarly named experiments from entering the denominator.
"""),
    code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve

from analysis.horizon_diagnostics import (
    HORIZONS, failure_auc_table, load_horizon_artifacts,
    prefix_failure_auc_table, prefix_feature_table,
    validate_diagnostic_cohort)
from analysis.smolvla_libero import (
    TRANSITION_ORDER, flip_auc_table, pair_stock_refinement,
    uncertainty_quantile_flip_table)
from analysis.suffix_sensitivity import bootstrap_rank_auc, summarize_pair
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS
from pnp.experiments import (
    SMOLVLA_LIBERO_EXPERIMENT, build_smolvla_libero_methods)
from pnp.store import SupabaseStore

EXPECTED_IDENTITIES = 400
N_BOOT = 3000
MAX_PREFIX_CHUNKS = 8
OUTPUT = Path('smolvla_libero_a10_analysis')
CACHE = OUTPUT / 'cache'
OUTPUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
store = SupabaseStore()

method_configs = dict(build_smolvla_libero_methods())
rows = pd.DataFrame(store.fetch_all(
    'rollouts', '*', configure=lambda query: query.eq(
        'experiment', SMOLVLA_LIBERO_EXPERIMENT).in_(
        'method', [Method.UNCERTAINTY, Method.REFINEMENT]),
    order_by=('rollout_id',)))

arms, config_hashes = {}, {}
for method, config in method_configs.items():
    config_hash = store.config_hash(store._logical_key(method, config))
    config_hashes[method] = config_hash
    frame = rows[
        rows.status.eq('completed') & rows.method.eq(method)
        & rows.config_hash.eq(config_hash)].copy()
    if frame.duplicated(DIVERSITY_PAIR_KEYS).any():
        raise ValueError(f'duplicate completed identities for {method}')
    if len(frame) != EXPECTED_IDENTITIES:
        raise ValueError(
            f'expected {EXPECTED_IDENTITIES} completed {method} identities, found {len(frame)}')
    arms[method] = frame.sort_values(DIVERSITY_PAIR_KEYS).reset_index(drop=True)

paired = pair_stock_refinement(
    arms[Method.UNCERTAINTY], arms[Method.REFINEMENT])
if len(paired) != EXPECTED_IDENTITIES:
    raise ValueError(f'expected {EXPECTED_IDENTITIES} exact pairs, found {len(paired)}')

coverage = pd.DataFrame([{
    'arm': method, 'config_hash': config_hashes[method], 'episodes': len(frame),
    'successes': int(frame.success.sum()),
    'success_rate_pct': 100 * frame.success.mean(),
    'uncertainty_artifacts': int(frame.ahats_path.notna().sum()),
    'trajectory_artifacts': int(frame.trajectory_path.notna().sum()),
} for method, frame in arms.items()])
print({'experiment': SMOLVLA_LIBERO_EXPERIMENT,
       'exact_matched_identities': len(paired), 'action_execution_horizon': 10,
       'integration_steps': 10, 'generated_chunk_size': 50})
display(coverage)
assert coverage.uncertainty_artifacts.eq(EXPECTED_IDENTITIES).all()
"""),
    markdown("""
## Decode stock uncertainty artifacts

Only the 400 stock-arm artifacts are required for failure and transition prediction. This is the
signal available before deciding whether an intervention would have been useful; downloading the
refinement arm as well would double analysis time without answering that selection question.
Decoded tables are cached in the Colab runtime.
"""),
    code("""
stock_rows = validate_diagnostic_cohort(
    arms[Method.UNCERTAINTY], expected_identities=EXPECTED_IDENTITIES,
    require_complete=True)
cache_tag = config_hashes[Method.UNCERTAINTY][:12]
cache_files = {
    name: CACHE / f'{name}_{cache_tag}.pkl'
    for name in ('features', 'records', 'positions', 'iterations')}
if all(path.exists() for path in cache_files.values()):
    features = pd.read_pickle(cache_files['features'])
    records = pd.read_pickle(cache_files['records'])
    positions = pd.read_pickle(cache_files['positions'])
    iterations = pd.read_pickle(cache_files['iterations'])
    print('Loaded decoded stock artifacts from local cache.')
else:
    features, records, positions, iterations = load_horizon_artifacts(
        store, stock_rows, progress=tqdm)
    features.to_pickle(cache_files['features'])
    records.to_pickle(cache_files['records'])
    positions.to_pickle(cache_files['positions'])
    iterations.to_pickle(cache_files['iterations'])
    print('Downloaded and cached 400 stock uncertainty artifacts.')
assert len(features) == EXPECTED_IDENTITIES
print({'episode_features': features.shape, 'probe_records': records.shape,
       'action_positions': positions.shape, 'perturbation_pairs': iterations.shape})
"""),
    markdown("""
## Primary result: matched success rate and paired changes
"""),
    code("""
overall_sr, suite_sr = summarize_pair(paired)
print('Overall paired result')
display(overall_sr)
print('Per-suite paired result')
display(suite_sr)
overall_sr.to_csv(OUTPUT / 'overall_paired_sr.csv', index=False)
suite_sr.to_csv(OUTPUT / 'suite_paired_sr.csv', index=False)

labels = suite_sr.suite.str.removeprefix('libero_')
x = np.arange(len(suite_sr)); width = .36
fig, axes = plt.subplots(2, 1, figsize=(13, 10), height_ratios=[1.05, 1],
                         constrained_layout=True)
axes[0].bar(x - width/2, suite_sr.baseline_sr_pct, width,
            label='stock + uncertainty measurement', color='#4C78A8')
axes[0].bar(x + width/2, suite_sr.condition_sr_pct, width,
            label='always-on refinement', color='#F58518')
axes[0].set_xticks(x, labels, rotation=25, ha='right')
axes[0].set(ylabel='Success rate (%)', ylim=(0, 105),
            title='SmolVLA A10 success rate on exact matched identities')
axes[0].legend(); axes[0].grid(axis='y', alpha=.2)

delta = suite_sr.condition_minus_baseline_pp.to_numpy()
lower = delta - suite_sr.delta_ci_low_pp.to_numpy()
upper = suite_sr.delta_ci_high_pp.to_numpy() - delta
axes[1].bar(x, delta, color=np.where(delta >= 0, '#54A24B', '#E45756'))
axes[1].errorbar(x, delta, yerr=np.vstack([lower, upper]), fmt='none',
                 ecolor='black', capsize=3)
axes[1].axhline(0, color='black', linewidth=1)
axes[1].axhline(overall_sr.condition_minus_baseline_pp.iloc[0],
                color='#9467BD', linestyle='--',
                label=f"overall: {overall_sr.condition_minus_baseline_pp.iloc[0]:+.2f} pp")
axes[1].set_xticks(x, labels, rotation=25, ha='right')
axes[1].set(ylabel='Refinement minus stock SR (percentage points)',
            title='Paired SR change by suite')
axes[1].legend(); axes[1].grid(axis='y', alpha=.2)
fig.savefig(OUTPUT / 'matched_sr_and_delta_by_suite.png', dpi=180)
plt.show()
"""),
    markdown("""
## Stock U10/U20/U50 as a failure detector

The left panel uses all stock episodes within each suite. The right panel pools all four suites.
Larger uncertainty is defined as predicting failure.
"""),
    code("""
u_episode_scores = [f'u{h}_episode' for h in HORIZONS]
u_first_scores = [f'u{h}_first_chunk' for h in HORIZONS]
u_auc = failure_auc_table(
    features, u_episode_scores + u_first_scores, n_boot=N_BOOT)
u_auc.to_csv(OUTPUT / 'stock_uncertainty_failure_auc.csv', index=False)
print('Pooled failure AUC')
display(u_auc[u_auc.suite.eq('pooled')])
print('Per-suite failure AUC')
display(u_auc[~u_auc.suite.eq('pooled')])

plot_auc = u_auc[
    ~u_auc.suite.eq('pooled') & u_auc.score_name.isin(u_episode_scores)
    & u_auc.failure_auc.notna()].copy()
suites = sorted(plot_auc.suite.unique())
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
for offset, score, label in zip((-.18, 0, .18), u_episode_scores, ('U10', 'U20', 'U50')):
    group = plot_auc[plot_auc.score_name.eq(score)].set_index('suite').reindex(suites)
    valid = group.failure_auc.notna().to_numpy()
    y = np.arange(len(suites))[valid] + offset
    center = group.failure_auc.to_numpy()[valid]
    axes[0].errorbar(center, y,
        xerr=np.vstack((center - group.auc_ci_low.to_numpy()[valid],
                        group.auc_ci_high.to_numpy()[valid] - center)),
        fmt='o', capsize=3, label=label)
axes[0].set_yticks(np.arange(len(suites)),
                   [name.removeprefix('libero_') for name in suites])
axes[0].axvline(.5, color='black', linestyle='--')
axes[0].set(xlim=(0, 1), xlabel='Failure ROC-AUC (95% bootstrap CI)',
            title='Stock uncertainty failure AUC by suite')
axes[0].legend(); axes[0].grid(axis='x', alpha=.2)

failure = ~features.success.astype(bool).to_numpy()
pooled = u_auc[u_auc.suite.eq('pooled')].set_index('score_name')
for score, label in zip(u_episode_scores, ('U10', 'U20', 'U50')):
    fpr, tpr, _ = roc_curve(failure, features[score])
    axes[1].plot(fpr, tpr, label=f'{label}: {pooled.loc[score, "failure_auc"]:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', label='chance')
axes[1].set(xlabel='False-positive rate', ylabel='True-positive rate',
            title='Pooled stock failure ROC')
axes[1].legend(); axes[1].grid(alpha=.2)
fig.savefig(OUTPUT / 'stock_failure_auc.png', dpi=180)
plt.show()
"""),
    markdown("""
## First-k chunks, contraction, and action-position profile

First-k statistics retain every episode: episodes that terminate early contribute all chunks that
were actually available. Exact-chunk curves would be survivor-biased and are intentionally not
used for the primary prefix claim.
"""),
    code("""
prefix = prefix_feature_table(records, features, max_chunks=MAX_PREFIX_CHUNKS)
prefix_auc = prefix_failure_auc_table(prefix, n_boot=N_BOOT)
prefix_auc.to_csv(OUTPUT / 'stock_first_k_auc.csv', index=False)
display(prefix_auc)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for axis, score_type, title in (
        (axes[0], 'uncertainty', 'Prefix uncertainty predicts stock failure'),
        (axes[1], 'negative_contraction', 'Weak/non-contraction predicts stock failure')):
    frame = prefix_auc[prefix_auc.score_type.eq(score_type)]
    for horizon in HORIZONS:
        group = frame[frame.action_horizon.eq(horizon)]
        axis.plot(group.first_k_chunks, group.failure_auc, marker='o',
                  label=f'first {horizon} actions')
    axis.axhline(.5, color='black', linestyle='--')
    axis.set(xlabel='First k observation chunks averaged', ylabel='Failure ROC-AUC',
             ylim=(.3, 1), title=title)
    axis.legend(); axis.grid(alpha=.2)
fig.savefig(OUTPUT / 'stock_first_k_auc.png', dpi=180)
plt.show()

for horizon in HORIZONS:
    features[f'negative_contraction{horizon}_episode'] = -features[
        f'contraction{horizon}_episode']
contraction_auc = failure_auc_table(
    features, [f'negative_contraction{h}_episode' for h in HORIZONS], n_boot=N_BOOT)
contraction_auc.to_csv(OUTPUT / 'stock_contraction_failure_auc.csv', index=False)
print('Pooled weak/non-contraction failure AUC')
display(contraction_auc[contraction_auc.suite.eq('pooled')])

profile = (positions.groupby(['success', 'action_position']).uncertainty
           .agg(['mean', 'std', 'count']).reset_index())
profile['sem'] = profile['std'] / np.sqrt(profile['count'])
fig, ax = plt.subplots(figsize=(12, 5))
for success, group in profile.groupby('success', sort=False):
    color = '#4C78A8' if bool(success) else '#E45756'
    label = 'stock success' if bool(success) else 'stock failure'
    ax.plot(group.action_position, group['mean'], color=color, label=label)
    ax.fill_between(group.action_position, group['mean'] - group['sem'],
                    group['mean'] + group['sem'], color=color, alpha=.16)
ax.axvline(9.5, color='black', linestyle='--', label='10 executed actions')
ax.axvline(19.5, color='#9467BD', linestyle=':', label='U20 boundary')
ax.set(xlabel='Action position in generated 50-action chunk',
       ylabel='Mean uncertainty', title='Stock uncertainty by action position and outcome')
ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUTPUT / 'stock_action_position_profile.png', dpi=180)
plt.show()
"""),
    markdown("""
## Are outcome flips enriched at high stock uncertainty?

The same initial identity can transition S→F or F→S because the two policies immediately begin
different trajectories. The following tests are associations, not proof that an online gate can
recover the refined outcome after observing the stock trajectory.

Crucially, rescue and harm are evaluated conditionally. Otherwise ordinary U-versus-failure
correlation would make F→S/F→F episodes look high-U even if U says nothing about whether
refinement helps.
"""),
    code("""
score_columns = u_episode_scores + u_first_scores
score_metadata = features[DIVERSITY_PAIR_KEYS + ['rollout_id', 'n_chunks'] + score_columns]
paired_scored = paired.merge(
    score_metadata, on=DIVERSITY_PAIR_KEYS, validate='one_to_one')

first4 = prefix[prefix.first_k_chunks.eq(4)][
    ['rollout_id'] + [f'u{h}' for h in HORIZONS]].copy()
first4 = first4.rename(columns={f'u{h}': f'u{h}_first4_chunks' for h in HORIZONS})
paired_scored = paired_scored.merge(first4, on='rollout_id', validate='one_to_one')
first4_scores = [f'u{h}_first4_chunks' for h in HORIZONS]
flip_scores = u_episode_scores + u_first_scores + first4_scores

transition_counts = (paired_scored.groupby(['suite', 'transition'], observed=False)
    .size().unstack(fill_value=0).reindex(columns=TRANSITION_ORDER, fill_value=0))
transition_counts['episodes'] = transition_counts.sum(axis=1)
transition_counts['flip_rate_pct'] = 100 * (
    transition_counts['F->S'] + transition_counts['S->F']) / transition_counts.episodes
transition_counts['net_refinement_gain_pp'] = 100 * (
    transition_counts['F->S'] - transition_counts['S->F']) / transition_counts.episodes
print('Outcome transitions by suite')
display(transition_counts.reset_index())
print('Overall transition counts')
display(paired_scored.transition.value_counts().reindex(TRANSITION_ORDER, fill_value=0)
        .rename_axis('transition').to_frame('episodes'))

flip_auc = flip_auc_table(paired_scored, flip_scores, n_boot=N_BOOT)
flip_auc.to_csv(OUTPUT / 'uncertainty_flip_auc.csv', index=False)
print('High uncertainty predicts the positive event named in each row.')
display(flip_auc)
transition_counts.reset_index().to_csv(OUTPUT / 'transition_counts_by_suite.csv', index=False)
"""),
    code("""
timings = {
    'whole episode': u_episode_scores,
    'first chunk': u_first_scores,
    'first 4 chunks': first4_scores,
}
cohort_order = ['all', 'stock failures', 'stock successes', 'discordant only']
colors = ['#4C78A8', '#54A24B', '#E45756', '#9467BD']
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True, sharey=True)
for axis, (timing, scores) in zip(axes, timings.items()):
    for offset, cohort, color in zip(np.linspace(-.24, .24, 4), cohort_order, colors):
        group = flip_auc[
            flip_auc.cohort.eq(cohort) & flip_auc.score_name.isin(scores)]
        group = group.set_index('score_name').reindex(scores)
        center = group.auc.to_numpy(float)
        valid = np.isfinite(center)
        x = np.arange(3)[valid] + offset
        axis.errorbar(center[valid], x,
            xerr=np.vstack((center[valid] - group.auc_ci_low.to_numpy(float)[valid],
                            group.auc_ci_high.to_numpy(float)[valid] - center[valid])),
            fmt='o', capsize=3, color=color, label=cohort)
    axis.axvline(.5, color='black', linestyle='--')
    axis.set_yticks(np.arange(3), ['U10', 'U20', 'U50'])
    axis.set(xlim=(0, 1), xlabel='ROC-AUC (95% bootstrap CI)', title=timing)
    axis.grid(axis='x', alpha=.2)
axes[0].set_ylabel('Uncertainty horizon')
axes[-1].legend(fontsize=8, loc='best')
fig.suptitle('Does high stock uncertainty identify outcome flips, rescue, or harm?')
fig.savefig(OUTPUT / 'uncertainty_flip_auc.png', dpi=180)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
for axis, score, label in zip(axes, u_episode_scores, ('U10', 'U20', 'U50')):
    groups = [paired_scored.loc[paired_scored.transition.eq(t), score].dropna().to_numpy()
              for t in TRANSITION_ORDER]
    axis.boxplot(groups, tick_labels=TRANSITION_ORDER, showfliers=False)
    axis.set(xlabel='Stock → refinement outcome', ylabel='Stock uncertainty',
             title=f'{label}, whole episode')
    axis.grid(axis='y', alpha=.2)
fig.savefig(OUTPUT / 'uncertainty_by_transition.png', dpi=180)
plt.show()
"""),
    markdown("""
## High-U quantile enrichment for guidance-data collection

Positive paired SR change in the highest-U quartile means high-U identities contain more F→S
than S→F outcomes. A high flip rate with near-zero or negative paired change instead means U finds
instability without identifying a beneficial correction direction. Agreement between whole-episode
and first-four-chunk results is especially relevant for selecting roots during collection.
"""),
    code("""
quantile_scores = ['u20_episode', 'u20_first_chunk', 'u20_first4_chunks']
quantile_tables = pd.concat([
    uncertainty_quantile_flip_table(paired_scored, score_column=score, bins=4)
    for score in quantile_scores], ignore_index=True)
display(quantile_tables)
quantile_tables.to_csv(OUTPUT / 'uncertainty_quantile_flip_enrichment.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
for axis, score in zip(axes, quantile_scores):
    group = quantile_tables[quantile_tables.score_name.eq(score)]
    axis.plot(group.uncertainty_quantile, group.flip_rate_pct, marker='o',
              label='any flip rate')
    axis.plot(group.uncertainty_quantile, group.refinement_minus_stock_pp, marker='o',
              label='paired SR change')
    axis.axhline(0, color='black', linewidth=1)
    axis.set_xticks([1, 2, 3, 4])
    axis.set(xlabel='Uncertainty quartile (4 = highest)', ylabel='Percentage points',
             title=score.replace('_', ' '))
    axis.grid(alpha=.2); axis.legend(fontsize=8)
fig.savefig(OUTPUT / 'u20_quantile_flip_enrichment.png', dpi=180)
plt.show()

high_u = quantile_tables[quantile_tables.uncertainty_quantile.eq(4)][[
    'score_name', 'episodes', 'flip_rate_pct', 'F_to_S', 'S_to_F',
    'refinement_minus_stock_pp']]
print('Highest-U quartile summary')
display(high_u)
"""),
    markdown("""
## Reading the result

- **Failure AUC** asks whether U detects that stock SmolVLA will fail.
- **Any-flip AUC** asks whether U identifies identities sensitive to refinement.
- **F→S AUC among stock failures** asks whether high U identifies *recoverable* failures rather
  than failures that remain failures.
- **S→F AUC among stock successes** asks whether high U warns that refinement will destroy a
  stock success.
- **Discordant-only AUC** asks whether high U favors rescue over harm once a flip occurs.

For guidance-data collection, the strongest result would be above-chance rescue AUC and positive
net F→S−S→F enrichment in the high-U first-four-chunk group. High failure AUC alone is useful for
choosing difficult roots, but does not show that the current refinement mechanism can improve them.
"""),
    code("""
summary = {
    'exact_pairs': len(paired_scored),
    'stock_sr_pct': 100 * paired_scored.baseline_success.mean(),
    'refinement_sr_pct': 100 * paired_scored.condition_success.mean(),
    'paired_delta_pp': 100 * (
        paired_scored.condition_success.mean() - paired_scored.baseline_success.mean()),
    'F_to_S': int((paired_scored.transition == 'F->S').sum()),
    'S_to_F': int((paired_scored.transition == 'S->F').sum()),
}
print(summary)
print('\\nPooled stock failure AUC:')
display(u_auc[u_auc.suite.eq('pooled')][
    ['score_name', 'failure_auc', 'auc_ci_low', 'auc_ci_high']])
print('\\nConditional flip AUCs:')
display(flip_auc[['cohort', 'positive_event', 'score_name', 'episodes', 'positives',
                  'auc', 'auc_ci_low', 'auc_ci_high']])
print('\\nOutputs:', OUTPUT.resolve())
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "84_analyze_smolvla_libero_a10.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path(__file__).resolve().parents[1] / "notebooks" / "84_analyze_smolvla_libero_a10.ipynb"
path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(path)
