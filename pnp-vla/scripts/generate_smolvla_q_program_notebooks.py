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

SOURCE_BOOT = BOOT.replace('SETUP_ENV = False', 'SETUP_ENV = True')
EGL = '''# Install NVIDIA EGL before importing MuJoCo.
import re, subprocess
driver = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
    text=True).splitlines()[0].strip()
package = f"libnvidia-gl-{re.match(r'\\d+', driver).group()}"
subprocess.run(['apt-get', 'update', '-qq'], check=True)
policy = subprocess.check_output(['apt-cache', 'policy', package], text=True)
if 'Candidate: (none)' in policy:
    raise RuntimeError(f'{package} is unavailable in this runtime')
subprocess.run(['apt-get', 'install', '-y', '--no-install-recommends', package], check=True)
subprocess.run('ldconfig -p | grep libEGL_nvidia', shell=True, check=True)
'''


def main() -> None:
    notebook("109_smolvla_success_q10_fresh8_train.ipynb", [
        markdown("""# 109 — Q10 success critic on the fresh8 P&P trees

**Estimand:** success after one recorded ten-action intervention, followed by the frozen SmolVLA P&P policy. The eight alternatives in this dataset are **P&P chunks**. This notebook does not train or validate a P&P-independent reranker.

Before training, complete and freeze all 800 v4 roots. Use a GPU runtime for training, but snapshot preparation is CPU work. No run starts until `RUN_TRAIN` is changed.

`TREE_LIMIT` counts distinct roots; `UPDATES` counts optimizer steps. To extend an existing 2,000-step checkpoint, set `UPDATES = 5000`, `EXTENSION_LEARNING_RATE = 3e-5`, and keep all other training settings unchanged. This resumes model, optimizer, and RNG state and uses the explicit constant learning rate after step 2,000. For a new sampling or model-hyperparameter experiment, set a new `EXPERIMENT_TAG` so its checkpoints do not overwrite the original run. `TD_ROOT_FRACTION` controls how often TD samples a fork-root window instead of a continuation window.
"""), code(BOOT),
        code("""from pathlib import Path
import torch
from pnp.smolvla_q_selection import FRESH8_EXPERIMENT, FRESH8_KINDS, fresh8_snapshot_key
from pnp.smolvla_tree_bellman_finetune import load_or_create_tree_snapshot, prepare_tree_bellman_cache
from pnp.store import SupabaseStore

TREE_LIMIT = 800
ARM = 'root_mc'  # change to 'tree_td' for matched TD arm
TRAIN_ROOT_LIMIT = None  # optional nested learning curve; max ~640 after split
UPDATES = 2000  # optimizer steps; TREE_LIMIT is distinct root states
BATCH_SIZE = 8  # MC: roots/update; TD: branches/update, eight windows per branch
LEARNING_RATE = 1e-4
WARMUP_UPDATES = 100
EVAL_INTERVAL = 250
CHECKPOINT_INTERVAL = 500
TARGET_RATE = 0.005  # TD target-network EMA
GRAD_CLIP = 1.0
TD_ROOT_FRACTION = 0.0  # 0 preserves uniform-window sampling; new runs only
RESUME = True
EXTENSION_LEARNING_RATE = None  # e.g. 3e-5 when extending a completed 2000-step run
RUN_TRAIN = False
CACHE_ROOT = Path('/content/smolvla_q10_fresh8_cache')
OUTPUT_ROOT = Path('/content/drive/MyDrive/pnp_smolvla_q10_fresh8')
EXPERIMENT_TAG = ''  # e.g. 'td_root50_v1' for a fresh, separately stored run
RUN_OUTPUT_ROOT = OUTPUT_ROOT if not EXPERIMENT_TAG else OUTPUT_ROOT / EXPERIMENT_TAG
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
        arm=ARM, output_root=RUN_OUTPUT_ROOT, cache_root=CACHE_ROOT,
        tree_limit=TREE_LIMIT, train_root_limit=TRAIN_ROOT_LIMIT,
        snapshot_key=fresh8_snapshot_key(TREE_LIMIT), experiment=FRESH8_EXPERIMENT,
        candidate_kinds=FRESH8_KINDS, updates=UPDATES, resume=RESUME,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        warmup_updates=WARMUP_UPDATES, eval_interval=EVAL_INTERVAL,
        checkpoint_interval=CHECKPOINT_INTERVAL, target_rate=TARGET_RATE,
        grad_clip=GRAD_CLIP, td_root_fraction=TD_ROOT_FRACTION,
        extension_learning_rate=EXTENSION_LEARNING_RATE,
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

    notebook("115_smolvla_q10_checkpoint_analysis_and_small_q.ipynb", [
        markdown("""# 115 — Diagnose checkpoint 109 and compare a small scalar Q

Run alongside the two new source-data workers. This notebook uses the **existing immutable 800-root fresh8 snapshot** and the same 640/160 root split as notebook 109. The checkpoint analysis reports oracle headroom, root-weighted pair accuracy, score spread, and exploratory Q-margin gates. Optional controls train a compact sigmoid Q10 and a state-only version on the same root-MC labels. Neither a gate threshold nor a checkpoint chosen on these already-inspected validation roots is a confirmatory result. No simulator run occurs here.
"""), code(BOOT),
        code("""from pathlib import Path
import torch
from pnp.smolvla_small_q import (
    load_fresh8_roots, analyze_existing_checkpoint,
    train_small_q, SmallTrainConfig)

SNAPSHOT_DIGEST = '23241dbe9dfeaec615e1f52d'
LARGE_CHECKPOINT = Path('/content/drive/MyDrive/pnp_smolvla_q10_fresh8') / SNAPSHOT_DIGEST / 'root_mc/checkpoint_step_002000.pt'
CACHE_ROOT = Path('/content/smolvla_q10_fresh8_cache')
OUTPUT_ROOT = Path('/content/drive/MyDrive/pnp_smolvla_small_q_fresh8') / SNAPSHOT_DIGEST
RUN_SMALL_Q = False
RUN_STATE_ONLY = False
assert LARGE_CHECKPOINT.is_file(), 'Mount Drive containing notebook 109 checkpoint.'
assert torch.cuda.is_available(), 'Select a GPU runtime for the model comparisons.'
snapshot, cache, train, validation = load_fresh8_roots(cache_root=CACHE_ROOT)
assert snapshot['snapshot_digest'] == SNAPSHOT_DIGEST
assert len(train) == 640 and len(validation) == 160
print({'snapshot': SNAPSHOT_DIGEST, 'train_roots': len(train),
       'validation_roots': len(validation), 'gpu': torch.cuda.get_device_name(0)})
"""),
        code("""large_report = analyze_existing_checkpoint(
    checkpoint=LARGE_CHECKPOINT, validation=validation,
    snapshot=snapshot, device='cuda')
for key, value in large_report.items():
    print(key, value)
"""),
        code("""if RUN_SMALL_Q:
    small_report = train_small_q(
        train=train, validation=validation,
        snapshot_digest=SNAPSHOT_DIGEST,
        output_path=OUTPUT_ROOT/'small_action_q.pt',
        state_only=False, device='cuda',
        config=SmallTrainConfig(updates=2000))
    print(small_report)
else:
    print('Small action-conditioned Q not started; set RUN_SMALL_Q=True.')
"""),
        code("""if RUN_STATE_ONLY:
    state_report = train_small_q(
        train=train, validation=validation,
        snapshot_digest=SNAPSHOT_DIGEST,
        output_path=OUTPUT_ROOT/'small_state_only.pt',
        state_only=True, device='cuda',
        config=SmallTrainConfig(updates=2000))
    print(state_report)
else:
    print('State-only control not started; set RUN_STATE_ONLY=True.')
""")])

    for shard in range(2):
        notebook(f"116_smolvla_scaling_source_worker_{shard}.ipynb", [
            markdown(f"""# 116 — New SmolVLA P&P source cohort, worker {shard}/2

Collect disjoint LIBERO episode indices **30–49** under the same frozen P&P policy as the original indices 10–29. The 40-task, 800-identity manifest is validated before simulator work; the two workers own 400 identities each and resume completed rows. These are **source episodes**, not yet nine-candidate trees. Once both source shards finish, freeze a new 65% high-U10 / 35% uniform root manifest and run the fresh8 tree workers. Keep evaluation indices 0–9 untouched.

Start with one episode. The full shard is opt-in. Run this concurrently with notebook 115 and the other source worker; no code here starts automatically.
"""), code(EGL), code(SOURCE_BOOT),
            code(f"""from pnp.smolvla_scaling_source import (
    SOURCE_EXPERIMENT, prepare_scaling_source_episodes,
    run_scaling_source_worker)

SHARD_INDEX = {shard}
EPISODE_LIMIT = 1  # first smoke; set None for all 400 identities in this shard
RUN_COLLECTION = False
episodes = prepare_scaling_source_episodes()
print({{'experiment': SOURCE_EXPERIMENT,
       'full_manifest': len(episodes), 'shard': f'{{SHARD_INDEX}}/2',
       'episode_limit': EPISODE_LIMIT, 'run': RUN_COLLECTION}})
"""),
            code("""if RUN_COLLECTION:
    report = run_scaling_source_worker(
        shard_index=SHARD_INDEX, episode_limit=EPISODE_LIMIT)
    print(report)
else:
    print('No collection. Set RUN_COLLECTION=True after reviewing the manifest.')
""")])

    for shard in range(3):
        notebook(f"117_smolvla_scaling_fresh8_tree_worker_{shard}.ipynb", [
            markdown(f"""# 117 — New fresh8 trees, worker {shard}/3

**Run after both notebook 116 source shards finish.** The existing immutable root manifest selects one decision boundary from each of the 800 indices-30–49 source episodes: 65% prioritized by weighted U10 and 35% uniform. Each tree records the exact source plus eight newly generated P&P chunks. No v3 branch reuse is possible in this new cohort. Every branch executes ten actions and then follows the same frozen P&P continuation. Workers use `SHARD_COUNT` and `SHARD_INDEX` to partition the same manifest; completed trees from the earlier two-worker run are reused. Use the same `SHARD_COUNT` in every concurrently running notebook and a distinct `SHARD_INDEX` for each.

Start with one tree. Full collection is opt-in. A future combined-data training run must freeze a new **combined** snapshot; notebook 109 remains tied to the original 800-root snapshot.
"""), code(EGL), code(SOURCE_BOOT),
            code(f"""from pnp.smolvla_scaling_trees import (
    TREE_EXPERIMENT, load_or_build_scaling_manifest,
    run_scaling_fresh8_tree_worker)

SHARD_INDEX = {shard}
SHARD_COUNT = 3  # use the same total in every concurrently running 117 notebook
TREE_LIMIT = 1  # smoke; set None for all roots assigned to this worker
RUN_COLLECTION = False
assert 0 <= SHARD_INDEX < SHARD_COUNT
print({{'experiment': TREE_EXPERIMENT, 'shard': f'{{SHARD_INDEX}}/{{SHARD_COUNT}}',
       'tree_limit': TREE_LIMIT, 'run': RUN_COLLECTION,
       'branches_per_root': 8}})
"""),
            code("""if RUN_COLLECTION:
    report = run_scaling_fresh8_tree_worker(
        shard_index=SHARD_INDEX, shard_count=SHARD_COUNT,
        tree_limit=TREE_LIMIT)
    print(report)
else:
    print('Collection disabled (RUN_COLLECTION=False). Set it to True to validate the 800 source episodes, build the root manifest, and run the smoke tree.')
""")])


if __name__ == "__main__":
    main()
