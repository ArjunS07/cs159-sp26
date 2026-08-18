"""Build notebook 47: step-aligned residual U20 critic and direct gradient pilot."""
from pathlib import Path

from nb_common import bootstrap, code, md, notebook, write_notebook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "47_train_residual_u20_critic_v2.ipynb"


cells = [
    md(r'''# 47 — Step-aligned residual U20 critic + direct-gradient test

Notebook 46 showed strong global U20 prediction but chance-level ranking among six actions from the **same observation**. This notebook addresses that exact failure:

1. each `z_hat` at Euler step 3 or 4 is paired with U20 measured at that same step;
2. `V(observation, step)` predicts only the state-level mean;
3. `A(observation, action, step)` predicts group-centered log-U20 residuals;
4. same-observation ranking receives the dominant training weight; and
5. checkpoint selection uses held-out candidate ranking, not global regression.

It then runs a separate direct test on untouched standard-LIBERO state index 0. Frozen π0.5 weights are differentiated to obtain the true local gradient of measured K=5 U20 with respect to the live flow latent. Common perturbation randomness compares gradient descent, ascent, and a matched random direction. The diagnostic never executes the modified action and uses no LIBERO-PRO data.

The direct test establishes only whether U20 is locally steerable. Neither a lower local U20 nor a good critic ranking proves an SR improvement.'''),
    code(bootstrap(extras="sim,analysis", setup_env=True)),
    md("## Configuration"),
    code(r'''from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from tqdm.auto import tqdm

from pnp.store import SupabaseStore
from pnp.uncertainty_critic_train import fetch_candidate_rows
from pnp.uncertainty_critic_v2 import (
    ResidualTrainConfig, evaluate_residual_critic, load_step_aligned_groups,
    residual_gradient_diagnostic, save_residual_checkpoint,
    train_residual_critic)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_ROOT = Path('/content/drive/MyDrive/pnp_uncertainty_critic_v2')
except ImportError:
    OUTPUT_ROOT = Path('uncertainty_critic_v2_outputs')

CACHE_PATH = OUTPUT_ROOT / 'worker45_step_aligned_groups_v2.npz'
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOWNLOAD_WORKERS = 12
TRAIN_CONFIG = ResidualTrainConfig(
    epochs=120, patience=20, batch_groups=64, learning_rate=3e-4,
    value_weight=.25, residual_weight=1.0, ranking_weight=3.0)
REPRESENTATIONS = ('z_hat_obs', 'z_hat_action_only', 'initial_obs')

# The direct test requires backward passes through frozen pi0.5. Activation
# checkpointing reduces memory, but an A100 is preferable. Reduce MAX_EPISODES
# to 1 if an L4 runs out of memory; do not reduce K=5.
RUN_DIRECT_TEST = True
DIRECT_MAX_EPISODES = 4       # one task-0 state from each standard suite
DIRECT_PROBE_STEPS = (3, 4)
DIRECT_EPSILONS = (.0025, .005, .01)

print({
    'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'train_states': '20-35', 'validation_states': '36-39',
    'direct_test_state': 0, 'direct_test_uses_pro': False,
    'primary_critic': 'same-step obs + z_hat residual',
    'loss_weights': TRAIN_CONFIG.__dict__,
    'direct_k': 5, 'direct_u_horizon': 20,
})'''),
    md("## Rebuild the step-aligned dataset"),
    code(r'''store = SupabaseStore()
rows = fetch_candidate_rows(store, require_complete=True)
groups = load_step_aligned_groups(
    store, rows, cache_path=CACHE_PATH, workers=DOWNLOAD_WORKERS, progress=tqdm)
train_groups = groups.subset(groups.train_mask)
validation_groups = groups.subset(groups.validation_mask)

assert len(rows) == 800
assert set(np.unique(train_groups.episode_idx)) == set(range(20, 36))
assert set(np.unique(validation_groups.episode_idx)) == set(range(36, 40))
assert not set(train_groups.rollout_id) & set(validation_groups.rollout_id)
coverage = (pd.DataFrame({
    'suite': groups.suite, 'task_idx': groups.task_idx,
    'episode_idx': groups.episode_idx, 'chunk_idx': groups.chunk_idx,
    'probe_step': groups.probe_step, 'probe_s': groups.probe_s})
    .groupby(['probe_step', 'probe_s']).size().rename('groups').reset_index())
print({
    'source_trajectories': len(rows), 'step_aligned_groups': len(groups),
    'candidate_examples': len(groups) * 6,
    'train_groups': len(train_groups), 'validation_groups': len(validation_groups),
})
display(coverage)

spread_rows = []
for split, frame in [('train', train_groups), ('validation', validation_groups)]:
    for step in (3, 4):
        values = frame.targets_u20[frame.probe_step == step]
        spread_rows.append({
            'split': split, 'probe_step': step, 'groups': len(values),
            'mean_u20': values.mean(),
            'mean_within_group_range': (values.max(1) - values.min(1)).mean(),
        })
display(pd.DataFrame(spread_rows))'''),
    md("## Train residual/ranking critics"),
    code(r'''models, histories, metadata, metrics = {}, {}, {}, {}
for representation in REPRESENTATIONS:
    print(f'\n=== {representation} ===')
    model, history, meta = train_residual_critic(
        train_groups, validation_groups, DEVICE,
        representation=representation, config=TRAIN_CONFIG, progress=True)
    models[representation] = model
    histories[representation] = pd.DataFrame(history)
    metadata[representation] = meta
    metrics[representation] = evaluate_residual_critic(
        model, validation_groups, DEVICE, representation, TRAIN_CONFIG)

metrics_table = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'representation'})
display(metrics_table[[
    'representation', 'groups', 'candidates', 'mae_u20',
    'within_group_ranking_accuracy', 'default_candidate_u20',
    'critic_selected_u20', 'oracle_candidate_u20',
    'selected_minus_default_u20',
    'step3_selected_minus_default_u20', 'step4_selected_minus_default_u20']])
metrics_table.to_csv(OUTPUT_ROOT / 'validation_metrics_v2.csv', index=False)

print('Primary pass criteria: ranking clearly above 0.50 and '
      'selected_minus_default_u20 materially below zero on both steps.')'''),
    md("## Learning curves and candidate-selection effect"),
    code(r'''fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
for representation, history in histories.items():
    axes[0].plot(history.epoch, history.validation_ranking,
                 label=representation)
axes[0].axhline(.5, color='black', linestyle='--', label='chance')
axes[0].set(title='Held-out same-observation ranking', xlabel='epoch',
            ylabel='pairwise ranking accuracy')
axes[0].legend(); axes[0].grid(alpha=.2)

x = np.arange(len(metrics_table)); width = .24
axes[1].bar(x - width, metrics_table.default_candidate_u20, width,
            label='ordinary candidate 0')
axes[1].bar(x, metrics_table.critic_selected_u20, width,
            label='critic chooses lowest score')
axes[1].bar(x + width, metrics_table.oracle_candidate_u20, width,
            label='measured-U20 oracle')
axes[1].set_xticks(x, metrics_table.representation, rotation=20)
axes[1].set(title='Held-out candidate selection', ylabel='mean measured U20')
axes[1].legend(); axes[1].grid(axis='y', alpha=.2)
fig.savefig(OUTPUT_ROOT / 'residual_critic_validation.png', dpi=180)
plt.show()'''),
    md("## Critic gradient checks and checkpoints"),
    code(r'''gradient_rows, checkpoint_rows = [], []
for representation, model in models.items():
    diagnostic = residual_gradient_diagnostic(
        model, validation_groups, DEVICE, representation)
    gradient_rows.append({'representation': representation, **diagnostic})
    path = CHECKPOINT_DIR / f'residual_u20_{representation}_v2.pt'
    save_residual_checkpoint(
        path, model, metadata[representation], metrics[representation],
        groups, representation, TRAIN_CONFIG)
    checkpoint_rows.append({
        'representation': representation, 'path': str(path),
        'size_mb': path.stat().st_size / 2**20})

gradient_table = pd.DataFrame(gradient_rows)
display(gradient_table); display(pd.DataFrame(checkpoint_rows))
assert gradient_table.all_finite.all()
assert (gradient_table.nonzero_fraction > 0).all()

summary = {
    'metrics': metrics, 'gradient_diagnostics': gradient_rows,
    'checkpoints': checkpoint_rows, 'train_config': TRAIN_CONFIG.__dict__,
    'primary_checkpoint': str(CHECKPOINT_DIR / 'residual_u20_z_hat_obs_v2.pt'),
}
(OUTPUT_ROOT / 'residual_training_summary.json').write_text(json.dumps(summary, indent=2))'''),
    md(r'''## Direct differentiation through measured U20

This is deliberately separate from the learned critic. At the selected live Euler state:

- freeze every π0.5 parameter;
- compute K=5 U20 with autograd enabled only for the flow latent;
- normalize the true gradient to a specified RMS update size;
- recompute U20 after descent, ascent, and a random direction;
- reuse exactly the same P&P perturbation seed for every comparison.

No modified action is executed. A useful result is a consistently negative descent delta that also beats the random control. Because the same states choose and evaluate the epsilon sweep, treat individual epsilon values as diagnostics, not a tuned benchmark result.'''),
    code(r'''direct_results = pd.DataFrame()
if RUN_DIRECT_TEST:
    # Free critic GPU memory before loading pi0.5.
    for key in list(models):
        models[key] = models[key].cpu()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    from pnp import models as pnp_models
    from pnp.config import LIBERO_DUMMY_ACTION, NUM_STEPS_WAIT
    from pnp.libero_env import obs_to_policy
    from pnp.rollout import (
        _draw_chunk_noise, chunk_noise_seed, episode_seed, iter_task_envs)
    from pnp.uncertainty_critic import SOURCE_MODEL_REVISION, prepare_episodes
    from pnp.uncertainty_critic_v2 import direct_uncertainty_gradient_test

    policy, preprocess, _ = pnp_models.load_pi05(revision=SOURCE_MODEL_REVISION)
    policy_device = pnp_models.default_device()
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)

    # One untouched init-state-0 task from each standard suite.
    episodes = [episode for episode in prepare_episodes((0,))
                if int(episode['task_idx']) == 0][:DIRECT_MAX_EPISODES]
    result_rows = []
    for env, task_episodes in iter_task_envs(episodes):
        for episode in task_episodes:
            env.reset(); policy.reset()
            observation = env.set_init_state(episode['init_state'])
            for _ in range(NUM_STEPS_WAIT):
                observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            batch = preprocess(obs_to_policy(observation, episode['task_desc']))
            seed = episode_seed(episode['init_state'], episode['ep_idx'])
            noise = _draw_chunk_noise(policy, policy_device, chunk_noise_seed(seed, 0))
            policy.model._pnp.chunk_pos = 0.0
            for probe_step in DIRECT_PROBE_STEPS:
                records = direct_uncertainty_gradient_test(
                    policy, batch, noise.clone(), probe_step=probe_step, k=5, horizon=20,
                    epsilons=DIRECT_EPSILONS,
                    perturb_seed=seed + 100 * probe_step,
                    random_seed=seed + 10_000 + probe_step)
                result_rows.extend({
                    'suite': episode['suite'], 'task_idx': episode['task_idx'],
                    'episode_idx': episode['ep_idx'], **record}
                    for record in records)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    direct_results = pd.DataFrame(result_rows)
    direct_results.to_csv(OUTPUT_ROOT / 'direct_u20_gradient_test.csv', index=False)
    display(direct_results)
else:
    print('Direct test skipped by RUN_DIRECT_TEST=False.')'''),
    code(r'''if not direct_results.empty:
    summary = (direct_results.groupby(['probe_step', 'epsilon_rms'])
        .agg(n=('suite', 'size'), baseline_u20=('baseline_u20', 'mean'),
             descent_delta_u20=('descent_delta_u20', 'mean'),
             ascent_delta_u20=('ascent_delta_u20', 'mean'),
             random_delta_u20=('random_delta_u20', 'mean'),
             descent_lowers_u=('descent_delta_u20', lambda x: np.mean(np.asarray(x) < 0)))
        .reset_index())
    display(summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label, color in (
            ('descent_delta_u20', 'true-gradient descent', '#54A24B'),
            ('ascent_delta_u20', 'gradient ascent', '#E45756'),
            ('random_delta_u20', 'matched random direction', '#4C78A8')):
        curve = direct_results.groupby('epsilon_rms')[column].agg(['mean', 'sem']).reset_index()
        ax.errorbar(curve.epsilon_rms, curve['mean'], yerr=curve['sem'],
                    marker='o', capsize=3, label=label, color=color)
    ax.axhline(0, color='black', linewidth=1)
    ax.set(title='Does direct differentiation lower measured K=5 U20?',
           xlabel='latent update RMS', ylabel='new U20 − baseline U20')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / 'direct_u20_gradient_test.png', dpi=180)
    plt.show()

    descent_rate = float((direct_results.descent_delta_u20 < 0).mean())
    beats_random = float((direct_results.descent_delta_u20
                          < direct_results.random_delta_u20).mean())
    print({
        'fraction_descent_lowers_measured_u20': descent_rate,
        'fraction_descent_beats_random_direction': beats_random,
        'interpretation': (
            'mechanically promising; next run a separately predeclared online intervention'
            if descent_rate >= .75 and beats_random >= .75 else
            'weak local causal control; do not proceed directly to SR claims'),
    })'''),
]


def main():
    write_notebook(OUTPUT, notebook(cells, OUTPUT.name))
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
