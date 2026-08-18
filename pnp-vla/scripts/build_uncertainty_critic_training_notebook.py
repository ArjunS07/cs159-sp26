"""Build notebook 46: train the same-observation U20 gradient critic."""
from __future__ import annotations

from pathlib import Path

from nb_common import bootstrap, code, md, notebook, write_notebook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "46_train_uncertainty_gradient_critic.ipynb"


cells = [
    md(r'''# 46 — Train the same-observation U20 gradient critic

This notebook consumes the **800 completed standard-LIBERO trajectories from workers 45**. It does not use LIBERO-PRO outcomes. Each stored observation has six independently sampled ordinary action predictions and U10/U20/U50 labels.

The primary model learns `U20 = f(observation encoding, ordinary initial action prediction)` using:

- regression to continuous U20; and
- same-observation ranking, so lower-U candidates must score lower.

Two explicit ablations train under the identical split: action-only (no observation context) and observation + the later collected `z_hat`. Init-state indices 20–35 are training data; 36–39 are held-out validation. Indices 0–19 remain untouched for later standard-LIBERO evaluation.

Passing this notebook shows that the score is predictable and differentiable with respect to actions. It does **not** show that taking the gradient improves success rate; that requires a subsequent online steering evaluation.'''),
    code(bootstrap(extras="analysis", setup_env=False)),
    md("## Configuration"),
    code(r'''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from tqdm.auto import tqdm

from pnp.store import SupabaseStore
from pnp.uncertainty_critic_train import (
    CriticTrainConfig, dataset_audit, evaluate_critic, fetch_candidate_rows,
    gradient_diagnostic, load_candidate_groups, load_checkpoint,
    save_checkpoint, train_uncertainty_critic)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_ROOT = Path('/content/drive/MyDrive/pnp_uncertainty_critic')
except ImportError:
    OUTPUT_ROOT = Path('uncertainty_critic_outputs')

CACHE_PATH = OUTPUT_ROOT / 'worker45_candidate_groups_v1.npz'
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOWNLOAD_WORKERS = 12
TRAIN_CONFIG = CriticTrainConfig(
    epochs=120, patience=18, batch_groups=64,
    learning_rate=3e-4, regression_weight=1.0, ranking_weight=0.5)
REPRESENTATIONS = ('initial_obs', 'initial_action_only', 'z_hat_obs')

print({
    'device': str(DEVICE),
    'cache': str(CACHE_PATH),
    'checkpoints': str(CHECKPOINT_DIR),
    'train_states': '20-35',
    'validation_states': '36-39',
    'untouched_eval_states': '0-19',
    'primary_input': 'ordinary pre-refinement action prediction + observation',
    'target': 'mean U20 over probe steps (3,4)',
})
if DEVICE.type == 'cpu':
    print('GPU is recommended but not required; this is a small critic, not pi0.5 fine-tuning.')'''),
    md("## Validate all 800 database rows and decode artifacts"),
    code(r'''store = SupabaseStore()
rows = fetch_candidate_rows(store, require_complete=True)
print(f'Validated {len(rows)} exact worker-45 trajectories.')

groups = load_candidate_groups(
    store, rows, cache_path=CACHE_PATH,
    workers=DOWNLOAD_WORKERS, progress=tqdm)
audit = dataset_audit(groups)
display(pd.DataFrame([audit]).drop(columns=['groups_by_chunk']))
display(pd.DataFrame([
    {'chunk_idx': int(chunk), 'groups': int(count)}
    for chunk, count in audit['groups_by_chunk'].items()]))

train_groups = groups.subset(groups.train_mask)
validation_groups = groups.subset(groups.validation_mask)
assert set(np.unique(train_groups.episode_idx)) == set(range(20, 36))
assert set(np.unique(validation_groups.episode_idx)) == set(range(36, 40))
assert not set(train_groups.rollout_id) & set(validation_groups.rollout_id)
print({
    'train_groups': len(train_groups),
    'train_candidates': len(train_groups) * 6,
    'validation_groups': len(validation_groups),
    'validation_candidates': len(validation_groups) * 6,
})'''),
    md("## Label distribution and same-observation signal"),
    code(r'''label_rows = []
for split, frame in [('train', train_groups), ('validation', validation_groups)]:
    for horizon in (10, 20, 50):
        values = getattr(frame, f'targets_u{horizon}').reshape(-1)
        label_rows.append({
            'split': split, 'horizon': f'U{horizon}', 'n': len(values),
            'mean': values.mean(), 'std': values.std(),
            'p05': np.quantile(values, .05), 'median': np.median(values),
            'p95': np.quantile(values, .95),
        })
label_summary = pd.DataFrame(label_rows)
display(label_summary)

spread = validation_groups.targets_u20.max(1) - validation_groups.targets_u20.min(1)
print({
    'validation_groups': len(spread),
    'mean_within_observation_U20_range': float(spread.mean()),
    'median_within_observation_U20_range': float(np.median(spread)),
    'groups_with_nontrivial_range': int((spread > 1e-6).sum()),
})

fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
for horizon, color in zip((10, 20, 50), ('#4C78A8', '#F58518', '#54A24B')):
    axes[0].hist(getattr(train_groups, f'targets_u{horizon}').reshape(-1),
                 bins=50, density=True, alpha=.35, label=f'U{horizon}', color=color)
axes[0].set(title='Training-label distributions', xlabel='uncertainty', ylabel='density')
axes[0].legend(); axes[0].grid(alpha=.2)
axes[1].hist(spread, bins=40, color='#9467BD', alpha=.8)
axes[1].set(title='Candidate variation at the same observation',
            xlabel='max U20 − min U20 among six candidates', ylabel='groups')
axes[1].grid(alpha=.2)
plt.show()'''),
    md("## Train the primary critic and two ablations"),
    code(r'''models, histories, training_meta, metrics = {}, {}, {}, {}
for representation in REPRESENTATIONS:
    print(f'\n=== {representation} ===')
    model, history, meta = train_uncertainty_critic(
        train_groups, validation_groups, DEVICE,
        representation=representation, config=TRAIN_CONFIG, progress=True)
    models[representation] = model
    histories[representation] = pd.DataFrame(history)
    training_meta[representation] = meta
    metrics[representation] = evaluate_critic(
        model, validation_groups, DEVICE, representation, TRAIN_CONFIG)

metrics_table = (pd.DataFrame(metrics).T.reset_index()
                 .rename(columns={'index': 'representation'}))
display(metrics_table[[
    'representation', 'groups', 'candidates', 'mae_u20', 'rmse_u20',
    'pearson', 'spearman', 'within_group_ranking_accuracy',
    'default_candidate_u20', 'critic_selected_u20',
    'oracle_candidate_u20', 'selected_minus_default_u20']])
metrics_table.to_csv(OUTPUT_ROOT / 'validation_metrics.csv', index=False)'''),
    md("## Learning curves and held-out predictions"),
    code(r'''fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
for representation, history in histories.items():
    axes[0].plot(history.epoch, history.train_loss, alpha=.7,
                 label=f'{representation}: train')
    axes[0].plot(history.epoch, history.validation_loss, linestyle='--',
                 label=f'{representation}: validation')
axes[0].set(title='Training history', xlabel='epoch', ylabel='combined loss')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)

from pnp.uncertainty_critic_train import predict_groups
for representation, color in zip(REPRESENTATIONS, ('#4C78A8', '#F58518', '#54A24B')):
    prediction = predict_groups(
        models[representation], validation_groups, DEVICE,
        representation, TRAIN_CONFIG).reshape(-1)
    target = validation_groups.targets_u20.reshape(-1)
    sample = np.linspace(0, len(target) - 1, min(len(target), 3000), dtype=int)
    axes[1].scatter(target[sample], prediction[sample], s=8, alpha=.18,
                    color=color, label=representation)
limit = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
axes[1].plot([0, limit], [0, limit], 'k--', linewidth=1)
axes[1].set(title='Held-out U20 predictions', xlabel='measured U20',
            ylabel='predicted U20', xlim=(0, limit), ylim=(0, limit))
axes[1].legend(); axes[1].grid(alpha=.2)
fig.savefig(OUTPUT_ROOT / 'training_and_validation.png', dpi=180)
plt.show()'''),
    md("## Gradient sanity check"),
    code(r'''gradient_rows = []
for representation, model in models.items():
    row = gradient_diagnostic(
        model, validation_groups, DEVICE, representation, max_groups=32)
    gradient_rows.append({'representation': representation, **row})
gradient_table = pd.DataFrame(gradient_rows)
display(gradient_table)
assert gradient_table.all_finite.all()
assert (gradient_table.nonzero_fraction > 0).all()
print('The score has finite, nonzero action gradients. This verifies mechanics only; '
      'the gradient direction still needs an online rollout test.')'''),
    md("## Save and reload reproducible checkpoints"),
    code(r'''checkpoint_rows = []
for representation, model in models.items():
    path = CHECKPOINT_DIR / f'u20_critic_{representation}_v1.pt'
    save_checkpoint(
        path, model, training_meta[representation], metrics[representation],
        groups, representation, TRAIN_CONFIG)
    restored, payload = load_checkpoint(path, DEVICE)
    before = evaluate_critic(
        model, validation_groups.subset(np.arange(len(validation_groups)) < 16),
        DEVICE, representation, TRAIN_CONFIG)['mae_u20']
    after = evaluate_critic(
        restored, validation_groups.subset(np.arange(len(validation_groups)) < 16),
        DEVICE, representation, TRAIN_CONFIG)['mae_u20']
    assert abs(before - after) < 1e-9
    checkpoint_rows.append({
        'representation': representation, 'path': str(path),
        'size_mb': path.stat().st_size / 2**20,
        'dataset_hash': payload['dataset_hash'],
        'validation_ranking_accuracy':
            payload['validation_metrics']['within_group_ranking_accuracy'],
    })

checkpoint_table = pd.DataFrame(checkpoint_rows)
display(checkpoint_table)
summary = {
    'dataset_audit': audit,
    'train_config': TRAIN_CONFIG.__dict__,
    'validation_metrics': metrics,
    'checkpoints': checkpoint_rows,
    'recommended_deployable_checkpoint':
        str(CHECKPOINT_DIR / 'u20_critic_initial_obs_v1.pt'),
    'next_required_test':
        'online held-out action-gradient steering; do not infer SR improvement from this fit',
}
(OUTPUT_ROOT / 'training_summary.json').write_text(json.dumps(summary, indent=2))
print('Saved:', OUTPUT_ROOT / 'training_summary.json')
print('Primary deployable input is initial_obs. z_hat_obs is an ablation because it '
      'requires running the P&P probe before the critic can score the action.')'''),
]


def main():
    write_notebook(OUTPUT, notebook(cells, OUTPUT.name))
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
