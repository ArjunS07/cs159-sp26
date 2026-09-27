"""Generate thin Colab launchers for the SmolVLA Q / RL Token program.

The notebooks only set experiment parameters.  All model and rollout logic is
in pnp/.  Regenerate after changing this file; outputs contain no cell state.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "notebooks" / "workers"
BRANCH = "codex/smolvla-q-program"
RAW_BOOTSTRAP = (
    "https://raw.githubusercontent.com/ArjunS07/cs159-sp26/"
    f"{BRANCH}/pnp-vla/scripts/colab_bootstrap.py")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def notebook(name: str, cells: list[dict]) -> None:
    payload = {"cells": cells, "metadata": {"colab": {"name": name},
               "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
               "nbformat": 4, "nbformat_minor": 5}
    (ROOT / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


BOOT = f'''from google.colab import drive
drive.mount('/content/drive')
EXTRAS = 'sim'
SETUP_ENV = False
REPO_REF = '{BRANCH}'
import urllib.request
exec(urllib.request.urlopen('{RAW_BOOTSTRAP}').read().decode())
'''


def main() -> None:
    notebook("109_smolvla_success_q10_fresh8_train.ipynb", [
        markdown("""# 109 — Q10 success critic on the fresh8 P&P trees

**Estimand:** success after one recorded ten-action intervention, followed by the frozen SmolVLA P&P policy. The eight alternatives in this dataset are **P&P chunks**. This notebook does not train or validate a P&P-independent reranker.

Before training, complete and freeze all 800 v4 roots. Use a GPU runtime for training, but snapshot preparation is CPU work. No run starts until `RUN_TRAIN` is changed.
"""), code(BOOT),
        code("""from pathlib import Path
import torch
from pnp.smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS, fresh8_snapshot_key
from pnp.smolvla_tree_bellman_finetune import load_or_create_tree_snapshot, prepare_tree_bellman_cache
from pnp.store import SupabaseStore

TREE_LIMIT = 800
ARM = 'root_mc'  # change to 'tree_td' for matched TD arm
TRAIN_ROOT_LIMIT = None  # optional nested learning curve; max ~640 after split
UPDATES = 2000
RUN_TRAIN = False
CACHE_ROOT = Path('/content/smolvla_q10_fresh8_cache')
OUTPUT_ROOT = Path('/content/drive/MyDrive/pnp_smolvla_q10_fresh8')
assert torch.cuda.is_available(), 'Select a Colab GPU runtime.'
store = SupabaseStore()
snapshot = load_or_create_tree_snapshot(
    store=store, snapshot_key=fresh8_snapshot_key(TREE_LIMIT),
    tree_limit=TREE_LIMIT, experiment=FRESH8_EXPERIMENT,
    candidate_kinds=FRESH8_KINDS)
cache = prepare_tree_bellman_cache(
    snapshot=snapshot, cache_root=CACHE_ROOT, gamma=1.0,
    success_reward=True, store=store)
assert len(snapshot['groups']) == TREE_LIMIT
print({'snapshot': snapshot['snapshot_digest'], 'train_roots': len(cache['train_group_ids']),
       'validation_roots': len(cache['validation_group_ids']),
       'train_windows': cache['train_windows'], 'gpu': torch.cuda.get_device_name(0)})
"""),
        code("""if RUN_TRAIN:
    from pnp.smolvla_success_critic import train_smolvla_success_q10
    report = train_smolvla_success_q10(
        arm=ARM, output_root=OUTPUT_ROOT, cache_root=CACHE_ROOT,
        tree_limit=TREE_LIMIT, train_root_limit=TRAIN_ROOT_LIMIT,
        snapshot_key=fresh8_snapshot_key(TREE_LIMIT), experiment=FRESH8_EXPERIMENT,
        candidate_kinds=FRESH8_KINDS, updates=UPDATES, resume=True,
        device='cuda', store=store)
    print(report)
else:
    print('Training not started. Set RUN_TRAIN=True after reviewing the frozen snapshot.')
""")])

    notebook("110_smolvla_q10_pnp_gate_offline.ipynb", [
        markdown("""# 110 — Offline P&P accept/reject audit

Scores the exact nine recorded v4 branches at each held-out root. Candidate 0 is the stored P&P source; candidates 1–8 are fresh-noise P&P proposals. This measures **one intervention under P&P continuation**, not an online deployed selector. Threshold selection on part of the validation roots is exploratory; use new episode indices for a final claim.
"""), code(BOOT),
        code("""from pathlib import Path
import hashlib
import numpy as np
import torch
from pnp.smolvla_q_selection import (
    FRESH8_EXPERIMENT, FRESH8_KINDS, fresh8_snapshot_key,
    load_success_q10, score_root_trees, paired_counts)
from pnp.smolvla_success_critic import TimedRoots
from pnp.smolvla_tree_bellman_finetune import load_or_create_tree_snapshot, prepare_tree_bellman_cache
from pnp.store import SupabaseStore

TREE_LIMIT = 800
CHECKPOINT = Path('REPLACE_WITH_NOTEBOOK_109_FINAL_CHECKPOINT.pt')
ROOT_TOKEN_DIR = None  # set only for notebook 111's RLT-Q checkpoint
CACHE_ROOT = Path('/content/smolvla_q10_fresh8_cache')
assert CHECKPOINT.is_file(), 'Set CHECKPOINT to a saved success-Q10 checkpoint.'
store = SupabaseStore()
snapshot = load_or_create_tree_snapshot(
    store=store, snapshot_key=fresh8_snapshot_key(TREE_LIMIT), tree_limit=TREE_LIMIT,
    experiment=FRESH8_EXPERIMENT, candidate_kinds=FRESH8_KINDS)
cache = prepare_tree_bellman_cache(
    snapshot=snapshot, cache_root=CACHE_ROOT, gamma=1.0,
    success_reward=True, store=store)
model, payload = load_success_q10(
    CHECKPOINT, expected_snapshot=snapshot['snapshot_digest'], device='cuda')
groups = {g['candidate_group_id']: g for g in snapshot['groups']}
roots = TimedRoots(cache, cache['validation_group_ids'], groups, ROOT_TOKEN_DIR)
scores, outcomes = score_root_trees(model, roots, device='cuda')
ids = [e['candidate_group_id'] for e in roots.entries]
assert outcomes.shape == scores.shape == (len(roots), 9)
print({'representation': payload.get('representation'), 'roots': len(roots),
       'baseline': paired_counts(outcomes, np.zeros(len(roots), dtype=int))})
"""),
        code("""# Development/test division is fixed by root identity. Neither part is a fresh final test:
# training checkpoints have already been inspected on this validation cohort.
order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
dev = np.asarray(order[:len(order)//2]); test = np.asarray(order[len(order)//2:])
best = scores[:, 1:].argmax(1) + 1
delta = scores[np.arange(len(scores)), best] - scores[:, 0]
thresholds = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
def selected(tau):
    return np.where(delta > tau, best, 0)
development = [(tau, paired_counts(outcomes[dev], selected(tau)[dev])) for tau in thresholds]
for tau, metrics in development:
    print('development', tau, metrics)
tau = max(thresholds, key=lambda t: (
    paired_counts(outcomes[dev], selected(t)[dev])['selected_successes'], t))
print('chosen development threshold:', tau)
print('exploratory held-out half:', paired_counts(outcomes[test], selected(tau)[test]))
print('ungated argmax:', paired_counts(outcomes, scores.argmax(1)))
""")])

    notebook("111_smolvla_contextual_rl_token.ipynb", [
        markdown("""# 111 — Contextual RL Token reconstruction and Q10 ablation

This implements the **representation-learning stage** of [RL Token](https://arxiv.org/html/2604.23073v2): extract the frozen VLM's **final contextualized prefix states**, train an encoder readout token to autoregressively reconstruct them, freeze it, then train a root-MC Q10 on the resulting single state token. It is distinct from the learned readout token already inside `QPlanningCritic`.

Local adaptation: final states are pooled to 128 tokens; the VLA is frozen and there is no separate RL actor. The source artifacts are rich enough for final-layer extraction; compact v4 branch artifacts are not, so this ablation currently supports **root MC only**, not tree TD. Start with `EXTRACT_LIMIT=1` to verify the pinned LeRobot API and shapes. Full feature extraction and training are opt-in.
"""), code(BOOT),
        code("""from pathlib import Path
import torch
from pnp.smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS, fresh8_snapshot_key
from pnp.smolvla_rl_token import (
    prepare_contextual_features, train_rl_token, materialize_root_tokens)
from pnp.smolvla_tree_bellman_finetune import load_or_create_tree_snapshot, prepare_tree_bellman_cache
from pnp.smolvla_tree_collection import _source_rows
from pnp.store import SupabaseStore

TREE_LIMIT = 800
FEATURE_ROOT = Path('/content/drive/MyDrive/pnp_smolvla_rlt/contextual_features')
RLT_CHECKPOINT = Path('/content/drive/MyDrive/pnp_smolvla_rlt/rlt_reconstruction.pt')
ROOT_TOKEN_DIR = Path('/content/drive/MyDrive/pnp_smolvla_rlt/root_tokens')
Q_OUTPUT_ROOT = Path('/content/drive/MyDrive/pnp_smolvla_q10_fresh8')
CACHE_ROOT = Path('/content/smolvla_q10_fresh8_cache')
EXTRACT_LIMIT = 1  # smoke; set to None for all 800 source rollouts
RUN_EXTRACTION = False
RUN_RLT_TRAIN = False
RUN_RLT_Q_TRAIN = False
assert torch.cuda.is_available(), 'Use a GPU runtime for VLM feature extraction.'
store = SupabaseStore()
snapshot = load_or_create_tree_snapshot(
    store=store, snapshot_key=fresh8_snapshot_key(TREE_LIMIT), tree_limit=TREE_LIMIT,
    experiment=FRESH8_EXPERIMENT, candidate_kinds=FRESH8_KINDS)
cache = prepare_tree_bellman_cache(
    snapshot=snapshot, cache_root=CACHE_ROOT, gamma=1.0,
    success_reward=True, store=store)
source_rows = _source_rows(store)
source_by_path = {str(r['training_data_path']): r['rollout_id'] for r in source_rows}
groups = {g['candidate_group_id']: g for g in snapshot['groups']}
def source_ids(group_ids):
    return {source_by_path[groups[g]['candidates'][0]['training_data_path']] for g in group_ids}
train_ids = source_ids(cache['train_group_ids'])
validation_ids = source_ids(cache['validation_group_ids'])
assert not train_ids & validation_ids
assert len(train_ids | validation_ids) == TREE_LIMIT
print({'train_source_episodes': len(train_ids), 'validation_source_episodes': len(validation_ids)})
"""),
        code("""if RUN_EXTRACTION:
    print(prepare_contextual_features(
        output_root=FEATURE_ROOT, train_rollout_ids=train_ids,
        validation_rollout_ids=validation_ids, limit=EXTRACT_LIMIT,
        store=store))
else:
    print('No extraction. Set RUN_EXTRACTION=True; keep EXTRACT_LIMIT=1 for first smoke.')
"""),
        code("""if RUN_RLT_TRAIN:
    assert len(list((FEATURE_ROOT/'train').glob('*.npz'))) == len(train_ids)
    assert len(list((FEATURE_ROOT/'validation').glob('*.npz'))) == len(validation_ids)
    print(train_rl_token(feature_root=FEATURE_ROOT, output_path=RLT_CHECKPOINT,
                         epochs=20, batch_size=16, device='cuda'))
else:
    print('RLT reconstruction not started.')
"""),
        code("""if RUN_RLT_Q_TRAIN:
    assert RLT_CHECKPOINT.is_file()
    feature_manifest = materialize_root_tokens(
        snapshot=snapshot, cache=cache, source_rows=source_rows,
        checkpoint=RLT_CHECKPOINT, feature_root=FEATURE_ROOT,
        output_root=ROOT_TOKEN_DIR, device='cuda')
    print(feature_manifest)
    from pnp.smolvla_success_critic import train_smolvla_success_q10
    report = train_smolvla_success_q10(
        arm='root_mc', output_root=Q_OUTPUT_ROOT, cache_root=CACHE_ROOT,
        tree_limit=TREE_LIMIT, snapshot_key=fresh8_snapshot_key(TREE_LIMIT),
        experiment=FRESH8_EXPERIMENT, candidate_kinds=FRESH8_KINDS,
        root_token_dir=ROOT_TOKEN_DIR, updates=2000, resume=True,
        device='cuda', store=store)
    print(report)
else:
    print('RL Token Q ablation not started.')
""")])

    notebook("112_smolvla_q10_vanilla_rerank_transfer.ipynb", [
        markdown("""# 112 — Ordinary SmolVLA reranker, independent of P&P at deployment

This evaluates **ordinary fresh-noise SmolVLA chunks** and chooses the highest-Q chunk. P&P is not called in candidate generation or continuation. The available Q was trained on P&P-root trees, so the Q estimate has a proposal/continuation mismatch. Treat this as an empirical transfer diagnostic, **not** a clean stock-policy Q experiment. A confirmatory stock reranker needs stock-generated training roots and stock continuation outcomes.

Uses LIBERO indices 0–9, disjoint from the 10–29 training cohort. Fixed two shards; each defaults to one episode smoke. This is an actual simulator experiment and is opt-in.
"""), code(BOOT),
        code("""from pathlib import Path
import torch
from pnp.smolvla_q_online_experiments import run_vanilla_rerank_transfer_worker

CHECKPOINT = Path('REPLACE_WITH_NOTEBOOK_109_PREFILL_Q_CHECKPOINT.pt')
SHARD_INDEX = 0  # make a copy with 1 for the other half
SHARD_COUNT = 2
EPISODE_LIMIT = 1  # smoke; set None for the fixed 200-identity shard
CANDIDATE_GENERATION_BATCH_SIZE = 1  # prioritize stock-slot fidelity in the pilot
RUN_EVAL = False
assert torch.cuda.is_available(), 'Use a GPU runtime.'
assert CHECKPOINT.is_file(), 'Set CHECKPOINT to notebook 109 root-MC checkpoint.'
print({'checkpoint': str(CHECKPOINT), 'shard': f'{SHARD_INDEX}/{SHARD_COUNT}',
       'episode_limit': EPISODE_LIMIT, 'run': RUN_EVAL})
"""),
        code("""if RUN_EVAL:
    run_vanilla_rerank_transfer_worker(
        checkpoint_path=str(CHECKPOINT), shard_index=SHARD_INDEX,
        shard_count=SHARD_COUNT, episode_limit=EPISODE_LIMIT,
        candidate_count=9,
        candidate_generation_batch_size=CANDIDATE_GENERATION_BATCH_SIZE)
else:
    print('No simulator run. Set RUN_EVAL=True after reviewing the checkpoint and cohort.')
""")])

    notebook("114_smolvla_q10_latent_guidance_transfer.ipynb", [
        markdown("""# 114 — Q-guided latent correction in the simulator

This is a **live correction experiment**: at one Euler step, take one bounded Q-ascent step through the SmolVLA flow latent, then finish denoising and execute the first ten actions. The reference is the exact historical ordinary SmolVLA A10 outcome on the same initial identity. P&P is not invoked at deployment. The Q is trained on P&P-root data, so this is explicitly a **transfer pilot**. It is not the previously proposed supervised PCP MLP.

Run one identity first. Inspect the `q_guidance_telemetry` including Q movement and action RMS. A Q rise alone is not a control improvement; compare simulator successes and failure pairs on the full fixed shard after the smoke test. The two shards cover indices 0–9 and are opt-in.
"""), code(BOOT),
        code("""from pathlib import Path
import torch
from pnp.smolvla_q_online_experiments import run_q10_latent_guidance_transfer_worker

CHECKPOINT = Path('REPLACE_WITH_NOTEBOOK_109_PREFILL_Q_CHECKPOINT.pt')
SHARD_INDEX = 0  # use 1 in a second runtime for the other half
SHARD_COUNT = 2
EPISODE_LIMIT = 1  # smoke; set None for the fixed full shard
EULER_STEP = 4
LATENT_STEP_RMS = 0.05  # in flow-latent units; pilot value, not tuned
RUN_EVAL = False
assert torch.cuda.is_available(), 'Use a GPU runtime.'
assert CHECKPOINT.is_file(), 'Set CHECKPOINT to notebook 109 root-MC checkpoint.'
print({'checkpoint': str(CHECKPOINT), 'shard': f'{SHARD_INDEX}/{SHARD_COUNT}',
       'episode_limit': EPISODE_LIMIT, 'euler_step': EULER_STEP,
       'latent_step_rms': LATENT_STEP_RMS, 'run': RUN_EVAL})
"""),
        code("""if RUN_EVAL:
    run_q10_latent_guidance_transfer_worker(
        checkpoint_path=str(CHECKPOINT), shard_index=SHARD_INDEX,
        shard_count=SHARD_COUNT, episode_limit=EPISODE_LIMIT,
        euler_step=EULER_STEP, latent_step_rms=LATENT_STEP_RMS)
else:
    print('No simulator run. Set RUN_EVAL=True after reviewing the pilot parameters.')
""")])

    notebook("113_smolvla_q10_pcp_gradient_diagnostic.ipynb", [
        markdown("""# 113 — Q-gradient PCP proposal diagnostic (offline only)

At held-out v4 root states, compute a bounded gradient step on the stored P&P action chunk and measure its Q-score change and normalized action displacement. **No corrected action has an observed outcome.** A higher predicted Q is not evidence of better control. The next stage is a matched simulator evaluation of these new chunks, with a stock/P&P fallback and a fixed trust radius. This notebook does not claim to implement or evaluate the old `pcp.py` MLP corrector.
"""), code(BOOT),
        code("""from pathlib import Path
import numpy as np
import torch
from pnp.smolvla_q_selection import (
    FRESH8_EXPERIMENT, FRESH8_KINDS, fresh8_snapshot_key,
    load_success_q10, bounded_q_ascent)
from pnp.smolvla_success_critic import TimedRoots
from pnp.smolvla_tree_bellman_finetune import load_or_create_tree_snapshot, prepare_tree_bellman_cache
from pnp.store import SupabaseStore

CHECKPOINT = Path('REPLACE_WITH_NOTEBOOK_109_PREFILL_Q_CHECKPOINT.pt')
TREE_LIMIT = 800
MAX_ROOTS = 16
TRUST_RADIUS_STD = 0.05
RUN_DIAGNOSTIC = False
assert CHECKPOINT.is_file(), 'Set CHECKPOINT to a saved prefill-prefix Q10 checkpoint.'
store = SupabaseStore()
snapshot = load_or_create_tree_snapshot(
    store=store, snapshot_key=fresh8_snapshot_key(TREE_LIMIT), tree_limit=TREE_LIMIT,
    experiment=FRESH8_EXPERIMENT, candidate_kinds=FRESH8_KINDS)
cache = prepare_tree_bellman_cache(
    snapshot=snapshot, cache_root=Path('/content/smolvla_q10_fresh8_cache'),
    gamma=1.0, success_reward=True, store=store)
model, payload = load_success_q10(CHECKPOINT,
    expected_snapshot=snapshot['snapshot_digest'], device='cuda')
assert payload.get('representation', 'frozen_prefill_prefix_v1') == 'frozen_prefill_prefix_v1'
groups = {g['candidate_group_id']: g for g in snapshot['groups']}
roots = TimedRoots(cache, cache['validation_group_ids'], groups)
"""),
        code("""if RUN_DIAGNOSTIC:
    records = []
    for i in range(min(MAX_ROOTS, len(roots))):
        row = roots[i]
        prefix = torch.as_tensor(row['prefix'][None], device='cuda')
        pad = torch.as_tensor(row['pad'][None], device='cuda')
        robot = torch.as_tensor(row['robot'][None], device='cuda')
        proprio = torch.as_tensor(row['proprio'][None], device='cuda')
        actions = torch.as_tensor(row['actions'][0:1], device='cuda')
        valid = torch.as_tensor(row['action_valid'][0:1], device='cuda')
        def score(candidate):
            return model.expected_value(prefix, pad, robot, proprio, candidate, valid)
        proposed, telemetry = bounded_q_ascent(
            score, actions, max_normalized_rms=TRUST_RADIUS_STD,
            action_std=model.action_std)
        with torch.no_grad():
            after = score(proposed)
        records.append({'root': roots.entries[i]['candidate_group_id'],
                        'q_before': float(telemetry['q_before'][0]),
                        'q_after': float(after[0]),
                        'normalized_update_rms': float(telemetry['normalized_update_rms'][0])})
    for record in records:
        print(record)
    print('NO SIMULATOR OUTCOMES FOR CORRECTED ACTIONS; this is a gradient smoke test only.')
else:
    print('No Q-gradient diagnostic. Set RUN_DIAGNOSTIC=True to run on held-out roots.')
""")])


if __name__ == "__main__":
    main()
