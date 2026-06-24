#!/usr/bin/env python3
"""Generate the three self-contained final/ notebooks.

01_run_experiments.ipynb  - controlled 80-ep slice + LIBERO-PRO 600-ep (pi0.5 + SmolVLA)
02_pcp_train_eval.ipynb   - Predict-Correct-Perturb Q-function train + 3-way eval
03_analysis.ipynb         - all tables/figures + 6/19 fixes (stratified AUC + geometry)

Each notebook embeds an RNG-isolated copy of the P&P core (no cross-notebook /
no smolvla_eval_core import) so it is fully self-contained.

Run:  python3 scripts/build_final_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RNG-isolation transform applied to smolvla_eval_core.py to build the shared
# "core" cell embedded (verbatim) into notebooks 01 and 02.
# ─────────────────────────────────────────────────────────────────────────────
GENERATOR_INFRA = '''

# === RNG-ISOLATION FIX (addresses 6/19 PDF section 1.1) ======================
# Perturbation noise now uses a dedicated per-device torch.Generator that NEVER
# touches the global RNG. This makes mode="uncertainty" a TRUE full-rollout
# no-op (its executed trajectory is identical to vanilla on the same seed) and
# makes mode="both"/refine perturbations reproducible and paired across methods.
# The old code used torch.randn_like(x_acc), which advanced the global RNG and
# silently desynchronised uncertainty/refinement rollouts from vanilla after the
# first chunk -- the bug the 6/19 report claimed (but did not actually) fix.
_PNP_GENS = {}
_PNP_LAST_SEED = [0]


def _pnp_gen(device):
    d = torch.device(device)
    g = _PNP_GENS.get(d)
    if g is None:
        g = torch.Generator(device=d)
        g.manual_seed(int(_PNP_LAST_SEED[0]) ^ 0x9E3779B9)
        _PNP_GENS[d] = g
    return g


def _pnp_seed_perturb(seed):
    """Seed the dedicated perturbation stream (independent of the global RNG)."""
    _PNP_LAST_SEED[0] = int(seed)
    for g in _PNP_GENS.values():
        g.manual_seed(int(seed) ^ 0x9E3779B9)


# a_hats persistence for the geometry analyses (PCA isotropy / multimodality).
if 'AHATS_DIR' not in dir():
    AHATS_DIR = None


def _save_ahats_npz(rollout_id, all_step_recs):
    """Persist raw per-iteration clean-action stacks when record_per_iteration is on."""
    if AHATS_DIR is None or rollout_id is None:
        return
    arrs = {}
    for ci, st in all_step_recs:
        a = st.get('a_hats')
        if a is not None:
            arrs[f'chunk{ci}_step{st["step"]}'] = np.asarray(a, dtype=np.float32)
    if arrs:
        os.makedirs(AHATS_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(AHATS_DIR, f'{rollout_id}.npz'), **arrs)


def assert_pnp_noop(policy, batch, step_indices=(1, 2), seed=0):
    """Real no-op check (NON-empty step_indices): uncertainty mode must equal vanilla.

    Replaces the old vacuous smoke test that used step_indices=() so the
    perturbation never fired and could not detect RNG contamination.
    """
    saved = (PNP_CONFIG.enabled, PNP_CONFIG.mode, PNP_CONFIG.step_indices,
             PNP_CONFIG.num_iterations)
    PNP_CONFIG.enabled = False
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); _pnp_seed_perturb(seed)
    with torch.no_grad():
        a1 = policy.predict_action_chunk(batch, noise=None).clone()
    PNP_CONFIG.enabled = True
    PNP_CONFIG.mode = 'uncertainty'
    PNP_CONFIG.step_indices = tuple(step_indices)
    PNP_CONFIG.num_iterations = 3
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); _pnp_seed_perturb(seed)
    with torch.no_grad():
        a2 = policy.predict_action_chunk(batch, noise=None).clone()
    (PNP_CONFIG.enabled, PNP_CONFIG.mode, PNP_CONFIG.step_indices,
     PNP_CONFIG.num_iterations) = saved
    d = float((a1 - a2).abs().max().item())
    verdict = 'PASS (true no-op)' if d == 0.0 else 'FAIL (RNG contamination!)'
    print(f'P&P no-op check: max|baseline - uncertainty| = {d:.3e}  ->  {verdict}')
    return d
# === end RNG-ISOLATION FIX ===================================================
'''


def build_core_src():
    src = (ROOT / "smolvla_eval_core.py").read_text()

    def sub(old, new, n):
        assert src.count(old) == n, f"expected {n} of:\n{old!r}\ngot {src.count(old)}"
        return src.replace(old, new)

    # 1) insert generator infra after the config constants block
    anchor = "FINAL_STEP_CONFIGS = [(2, 3), (3, 4), (4, 5)]\nPNP_K = 3\nBASELINE_STEPS = 10\n"
    src = sub(anchor, anchor + GENERATOR_INFRA, 1)

    # 2) dedicated-generator perturbation draw
    src = sub("        eps = torch.randn_like(x_acc)\n",
              "        eps = torch.empty_like(x_acc).normal_(0.0, 1.0, generator=_pnp_gen(x_acc.device))\n",
              1)

    # 3) seed the dedicated stream alongside the global seed (two-line form x2)
    two = "    torch.manual_seed(_seed)\n    torch.cuda.manual_seed(_seed)"
    src = sub(two, two + "\n    _pnp_seed_perturb(_seed)", 2)

    # 3b) semicolon form (run_episode_backfill) x1
    semi = "    torch.manual_seed(_seed); torch.cuda.manual_seed(_seed)"
    src = sub(semi, semi + "; _pnp_seed_perturb(_seed)", 1)

    # 3c) multi-sample per-candidate seed x1
    ms = ("        torch.manual_seed(base_seed + chunk_idx * 1000 + si)\n"
          "        torch.cuda.manual_seed(base_seed + chunk_idx * 1000 + si)")
    src = sub(ms, ms + "\n        _pnp_seed_perturb(base_seed + chunk_idx * 1000 + si)", 1)

    # 4) persist a_hats inside log_episode
    commit = "                 for ci, st in all_step_recs])\n        self._con.commit()"
    src = sub(commit,
              "                 for ci, st in all_step_recs])\n"
              "        _save_ahats_npz(rollout_id, all_step_recs)\n"
              "        self._con.commit()",
              1)
    return src


CORE_SRC = build_core_src()


# ─────────────────────────────────────────────────────────────────────────────
# Shared cells (config + environment) reused across notebooks.
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_SRC = r'''# === Unified config: single Drive root, one schema per artifact =============
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive'
except Exception:
    DRIVE = os.path.expanduser('~')

FINAL_ROOT  = os.path.join(DRIVE, 'cs159-sp26-final')
RESULTS_DIR = os.path.join(FINAL_ROOT, 'results')
VIDEO_DIR   = os.path.join(RESULTS_DIR, 'videos')
AHATS_DIR   = os.path.join(RESULTS_DIR, 'ahats')      # raw a_hats stacks (geometry)
FIGURES_DIR = os.path.join(FINAL_ROOT, 'figures')
TABLES_DIR  = os.path.join(FINAL_ROOT, 'tables')

SLICE_DB = os.path.join(RESULTS_DIR, 'rollouts_final.db')  # controlled 80-ep slice
PRO_DB   = os.path.join(RESULTS_DIR, 'rollouts_pro.db')    # LIBERO-PRO 600-ep
QC_DB    = os.path.join(RESULTS_DIR, 'qc.db')              # PCP chunks + 3-way eval
QC_CKPT  = os.path.join(RESULTS_DIR, 'q_corrector.pt')

# Curated LIBERO-PRO init files (.pruned_init) must be placed here by the user
# (exported from the LIBERO-Pro dataset). See the LIBERO-PRO markdown cell.
PRO_INIT_SRC = os.path.join(FINAL_ROOT, 'libero_pro_init_files')

# Optional package snapshot tarball (speeds up Colab restore).
CACHE_DIR = os.path.join(DRIVE, 'smolvla_colab_cache')

for _d in (RESULTS_DIR, VIDEO_DIR, AHATS_DIR, FIGURES_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)

MODELS = ['pi05', 'smolvla']
print('FINAL_ROOT =', FINAL_ROOT)
'''


ENV_SETUP_SRC = r'''# === Environment: GPU + LIBERO/MuJoCo + lerobot (pi0.5 & SmolVLA) ===========
import subprocess, sys, os, importlib

print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
      or 'No GPU detected (CPU mode will be extremely slow / infeasible).')

# Fast path: restore a prebuilt site-packages snapshot from Drive if present.
SNAPSHOT = os.path.join(CACHE_DIR, 'site_packages.tar.gz')
if os.path.exists(SNAPSHOT):
    import shutil
    local = '/content/site_packages_restore.tar.gz'
    shutil.copy(SNAPSHOT, local)
    subprocess.run(['tar', '-xzf', local, '-C', '/'], check=True)
    os.remove(local)
    importlib.invalidate_caches()
    print('Restored package snapshot.')

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HOME', os.path.join(CACHE_DIR, 'hf_models'))

def _ensure(mod, pip_spec):
    try:
        importlib.import_module(mod)
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pip_spec.split()])
        importlib.invalidate_caches()

_ensure('mujoco', 'mujoco')
_ensure('libero', 'libero')
try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'lerobot[pi0,smolvla]'])
    importlib.invalidate_caches()

# Authenticate for gated checkpoints (set HF_TOKEN env var beforehand).
from huggingface_hub import login
_tok = os.getenv('HF_TOKEN')
login(token=_tok) if _tok else None
print('Environment ready.')
'''


LIBERO_PRO_MD = r'''## LIBERO-PRO data preparation (required for the PRO and PCP runs)

The LIBERO-PRO suites (`libero_object_temp_x0.1/x0.2/y0.1/y0.2`,
`libero_spatial_with_milk`, `libero_goal_with_yellow_book`) are **not** part of
the stock LIBERO install. Before running the PRO cells:

1. Download the LIBERO-Pro init/BDDL assets (HF dataset `zhouxueyang/LIBERO-Pro`).
2. Place the per-suite `*.pruned_init` files under `PRO_INIT_SRC`
   (`<FINAL_ROOT>/libero_pro_init_files/<suite>/...`) and the matching BDDL files
   into the installed libero package's `bddl_files/<suite>/` directory.
3. The helper `restore_libero_pro_inits(PRO_INIT_SRC, LIBERO_SITE)` copies the
   init files into place; `build_libero_pro_episodes(LIBERO_SITE, ...)` then
   builds the 6 x 10 x 10 = 600 episode list.

The controlled 80-episode slice (Section 6) uses only stock `libero_goal` /
`libero_spatial`, so it runs without this preparation.
'''


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01: run_experiments
# ─────────────────────────────────────────────────────────────────────────────
NOOP_SRC = r'''# === Verify the RNG-isolation fix is a TRUE no-op before spending GPU time ====
policy, preprocess, postprocess = load_pi05_session(video_dir=VIDEO_DIR)
globals()['CURRENT_POLICY_MODEL'] = 'pi05'

eps = build_final_episodes(None)[:1]
_ep = eps[0]
_env = OffScreenRenderEnv(bddl_file_name=_ep['bddl_path'], camera_names=CAMERAS,
                          camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                          has_offscreen_renderer=True, use_camera_obs=True,
                          has_renderer=False, reward_shaping=False)
try:
    _env.reset(); policy.reset()
    _obs = _env.set_init_state(_ep['init_state'])
    for _ in range(NUM_STEPS_WAIT):
        _obs, _, _, _ = _env.step(LIBERO_DUMMY_ACTION)
    _batch = preprocess(obs_to_policy(_obs, _ep['task_desc'], device))
    d = assert_pnp_noop(policy, _batch, step_indices=(2, 3), seed=0)
    assert d == 0.0, 'RNG isolation failed -- do not trust downstream numbers.'
finally:
    _env.close()
'''

SLICE_DRIVER_SRC = r'''# === Controlled 80-episode slice: 4 methods, paired, RNG-isolated ============
# Re-runs ALL four methods natively (no v1 import) so every row uses the fixed
# RNG. uncertainty_only is now a TRUE no-op of vanilla (expected SR ~= vanilla);
# refinement (mode="both") is the genuine intervention.
import json as _json
from itertools import groupby
from tqdm.auto import tqdm

SLICE_STEP_CONFIGS = [(2, 3), (3, 4), (4, 5)]
SLICE_K = 3
BASELINE_STEPS = {'pi05': 10, 'smolvla': 10}
EXTRA_STEPS    = {'pi05': 16, 'smolvla': 16}   # matched-compute baseline
METHODS = ['vanilla', 'extra_steps', 'pnp_uncertainty_only', 'pnp_refinement']


def _load_model(model_name):
    if model_name == 'pi05':
        return load_pi05_session(video_dir=VIDEO_DIR)
    return load_smolvla_session(video_dir=VIDEO_DIR)


def run_controlled_slice(model_name, db):
    policy, preprocess, postprocess = _load_model(model_name)
    globals()['CURRENT_POLICY_MODEL'] = model_name
    episodes = build_final_episodes(None)              # 8 tasks x 10 eps = 80
    base, extra = BASELINE_STEPS[model_name], EXTRA_STEPS[model_name]
    done = db.existing_keys(final_eval_slice=1, policy_model=model_name)
    eps_sorted = sorted(episodes, key=lambda x: (x['suite'], x['task_idx']))
    for (suite, task_idx), grp in groupby(eps_sorted, key=lambda x: (x['suite'], x['task_idx'])):
        grp = list(grp)
        env = OffScreenRenderEnv(bddl_file_name=grp[0]['bddl_path'], camera_names=CAMERAS,
                                 camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                                 has_offscreen_renderer=True, use_camera_obs=True,
                                 has_renderer=False, reward_shaping=False)
        try:
            for method in METHODS:
                configs = SLICE_STEP_CONFIGS if method in (
                    'pnp_uncertainty_only', 'pnp_refinement') else [None]
                for step_indices in configs:
                    PNP_CONFIG.enabled = method in ('pnp_uncertainty_only', 'pnp_refinement')
                    PNP_CONFIG.mode = 'both' if method == 'pnp_refinement' else 'uncertainty'
                    PNP_CONFIG.step_indices = tuple(step_indices) if step_indices else (1,)
                    PNP_CONFIG.num_iterations = SLICE_K
                    PNP_CONFIG.time_min = None
                    PNP_CONFIG.record_per_iteration = False
                    nis = extra if method == 'extra_steps' else (base if method == 'vanilla' else None)
                    step_key = _json.dumps(list(step_indices)) if step_indices else None
                    for ep in tqdm(grp, leave=False,
                                   desc=f'{model_name}/{method}/{step_key} {suite} T{task_idx}'):
                        key = (ep['suite'], ep['task_idx'], ep['ep_idx'],
                               ep['init_state_hash'], method, step_key)
                        if key in done:
                            continue
                        run_episode_pnp(env, ep['init_state'], policy, ep['task_desc'],
                                        ep['max_steps'], device,
                                        suite=ep['suite'], task_idx=ep['task_idx'],
                                        episode_idx=ep['ep_idx'], db=db,
                                        save_video='failures_only', method=method,
                                        final_eval_slice=1, num_inference_steps=nis)
        finally:
            env.close()
    db.sync_to_path(SLICE_DB)


SLICE_DB_HANDLE = RolloutDB(SLICE_DB)
for _m in MODELS:
    print(f'\n===== controlled slice: {_m} =====')
    run_controlled_slice(_m, SLICE_DB_HANDLE)
SLICE_DB_HANDLE.summary()
'''

PRO_DRIVER_SRC = r'''# === LIBERO-PRO 600-ep: baseline / uncertainty(+a_hats) / both ==============
from itertools import groupby
from tqdm.auto import tqdm

PRO_STEP_INDICES = (1, 2)
PRO_K = 3
LIBERO_SITE = os.path.dirname(get_libero_path('bddl_files'))
PRO_PHASES = [
    # (method,                 mode,          enabled, record_per_iteration)
    ('vanilla',                'uncertainty', False,   False),
    ('pnp_uncertainty_only',   'uncertainty', True,    True),   # saves a_hats
    ('pnp_refinement',         'both',        True,    False),
]


def run_pro(model_name, db):
    policy, preprocess, postprocess = _load_model(model_name)
    globals()['CURRENT_POLICY_MODEL'] = model_name
    try:
        restore_libero_pro_inits(PRO_INIT_SRC, LIBERO_SITE)
    except Exception as e:
        print('WARNING: could not restore PRO init files:', e)
    episodes = build_libero_pro_episodes(LIBERO_SITE, benchmark_dict_=benchmark_dict)
    if not episodes:
        print('No PRO episodes found -- complete the LIBERO-PRO data prep first.')
        return
    eps_sorted = sorted(episodes, key=lambda x: (x['suite'], x['task_idx']))
    for (suite, task_idx), grp in groupby(eps_sorted, key=lambda x: (x['suite'], x['task_idx'])):
        grp = list(grp)
        env = OffScreenRenderEnv(bddl_file_name=grp[0]['bddl_path'], camera_names=CAMERAS,
                                 camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                                 has_offscreen_renderer=True, use_camera_obs=True,
                                 has_renderer=False, reward_shaping=False)
        try:
            for method, mode, enabled, rec in PRO_PHASES:
                PNP_CONFIG.enabled = enabled
                PNP_CONFIG.mode = mode
                PNP_CONFIG.step_indices = PRO_STEP_INDICES
                PNP_CONFIG.num_iterations = PRO_K
                PNP_CONFIG.time_min = None
                PNP_CONFIG.record_per_iteration = rec
                for ep in tqdm(grp, leave=False, desc=f'{model_name}/{method} {suite} T{task_idx}'):
                    run_episode_pnp(env, ep['init_state'], policy, ep['task_desc'],
                                    ep['max_steps'], device,
                                    suite=ep['suite'], task_idx=ep['task_idx'],
                                    episode_idx=ep['ep_idx'], db=db,
                                    method=method, final_eval_slice=0,
                                    num_inference_steps=None)
        finally:
            env.close()
    db.sync_to_path(PRO_DB)


PRO_DB_HANDLE = RolloutDB(PRO_DB)
for _m in MODELS:
    print(f'\n===== LIBERO-PRO: {_m} =====')
    run_pro(_m, PRO_DB_HANDLE)
PRO_DB_HANDLE.summary()
'''


def build_nb01():
    cells = [
        md("# 01 - Run Experiments (self-contained)\n\n"
           "Controlled 80-episode LIBERO slice + LIBERO-PRO 600-episode stretch for "
           "**pi0.5** and **SmolVLA**, with the **RNG-isolation fix** applied to the "
           "P&P sampler.\n\n"
           "**Outputs:** `rollouts_final.db` (slice), `rollouts_pro.db` (PRO), and "
           "`ahats/*.npz` (raw per-iteration clean-action stacks for the geometry "
           "analyses in notebook 03).\n\n"
           "> Requires a GPU, the LIBERO/MuJoCo simulator, and gated pi0.5/SmolVLA "
           "checkpoints (set `HF_TOKEN`). Run top to bottom."),
        md("## 1. Config"),
        code(CONFIG_SRC),
        md("## 2. Environment"),
        code(ENV_SETUP_SRC),
        md("## 3. P&P core (RNG-isolated) + RolloutDB + rollout helpers\n\n"
           "Self-contained copy of the evaluation core with `torch.randn_like` "
           "replaced by a dedicated `torch.Generator` (see the RNG-ISOLATION FIX "
           "block) and `a_hats` persistence wired into `RolloutDB.log_episode`."),
        code(CORE_SRC),
        md("## 4. No-op verification"),
        code(NOOP_SRC),
        md("## 5. Controlled 80-episode slice"),
        code(SLICE_DRIVER_SRC),
        md("## 6. LIBERO-PRO 600-episode stretch"),
        LIBERO_PRO_MD_CELL,
        code(PRO_DRIVER_SRC),
    ]
    return nb(cells)


LIBERO_PRO_MD_CELL = md(LIBERO_PRO_MD)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02: PCP (Predict-Correct-Perturb) train + eval
# ─────────────────────────────────────────────────────────────────────────────
PCP_COLLECT_SRC = r'''# === PCP data collection: labeled (z_hat, obs_enc) chunks on LIBERO-PRO ======
# Measurement-only collection: executes vanilla actions but records, at every
# denoising step, the mean clean-action prediction z_hat (over K P&P iters) and
# the mean-pooled prefix embedding obs_enc. Perturbation noise uses the dedicated
# generator (RNG isolation), so collection does not perturb the executed rollout.
import sqlite3, json as _json, types as _types
from itertools import groupby
from tqdm.auto import tqdm

RUN_COLLECT = True            # set False once qc.db is populated
N_COLLECT_EPS = 10            # episodes per PRO task
COLLECT_STEPS = tuple(range(10))
PNP_K = 3
_CURRENT_OBS_ENC = [None]
_COLLECT_BUF = []


@torch.no_grad()
def _sample_actions_collect(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kw):
    if num_steps is None:
        num_steps = self.config.num_inference_steps
    bsize = tokens.shape[0]; dev = tokens.device
    if noise is None:
        noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), dev)
    measure = self._orig_sample_actions(
        images, img_masks, tokens, masks, noise=noise.clone(), num_steps=num_steps, **kw).clone()
    prefix_embs, ppm, pam = self.embed_prefix(images, img_masks, tokens, masks)
    _CURRENT_OBS_ENC[0] = prefix_embs.mean(dim=1)[0].detach().float().cpu().numpy()
    att4d = self._prepare_attention_masks_4d(make_att_2d_masks(ppm, pam))
    pos = torch.cumsum(ppm, dim=1) - 1
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, pkv = self.paligemma_with_expert.forward(
        attention_mask=att4d, position_ids=pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True)
    dt = -1.0 / num_steps; x_t = noise
    for step in range(num_steps):
        s = 1.0 + step * dt
        tt = torch.tensor(s, dtype=torch.float32, device=dev).expand(bsize)
        def vf(inx, ts=tt):
            return self.denoise_step(prefix_pad_masks=ppm, past_key_values=pkv, x_t=inx, timestep=ts)
        if step in COLLECT_STEPS:
            x_acc = x_t; ah = []
            for _ in range(PNP_K):
                v = vf(x_acc); a_hat = x_acc - s * v; ah.append(a_hat)
                x_acc = (1.0 - s) * a_hat + s * torch.empty_like(x_acc).normal_(
                    0.0, 1.0, generator=_pnp_gen(x_acc.device))
            z_hat = torch.stack(ah, 0).mean(0)[0, :, :7].detach().float().cpu().numpy()
            _COLLECT_BUF.append({'step_idx': step, 's': float(s), 'z_hat': z_hat.tolist()})
        v_t = vf(x_t); x_t = x_t + dt * v_t
    return measure


def _qc_init_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS qc_rollouts ("
                "rollout_id TEXT PRIMARY KEY, suite TEXT, task_idx INTEGER, episode_idx INTEGER, "
                "init_state_hash TEXT, success INTEGER, n_chunks INTEGER, chunks TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS qc_eval ("
                "rollout_id TEXT, suite TEXT, task_idx INTEGER, episode_idx INTEGER, "
                "lambda REAL, success INTEGER, PRIMARY KEY (rollout_id, lambda))")
    con.commit(); return con


def collect_episode(env, ep, policy, con):
    rid = f"{ep['suite']}:{ep['task_idx']}:{ep['ep_idx']}:{ep['init_state_hash']}"
    if con.execute('SELECT 1 FROM qc_rollouts WHERE rollout_id=?', (rid,)).fetchone():
        return
    env.reset(); policy.reset()
    obs = env.set_init_state(ep['init_state'])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    _seed = _episode_seed(ep['init_state'], ep['ep_idx'])
    torch.manual_seed(_seed); torch.cuda.manual_seed(_seed); _pnp_seed_perturb(_seed)
    chunks, queue, success, ci = [], [], False, 0
    for step in range(ep['max_steps']):
        if not queue:
            _COLLECT_BUF.clear()
            batch = preprocess(obs_to_policy(obs, ep['task_desc'], device))
            chunk = policy.predict_action_chunk(batch, noise=None)
            chunks.append({'chunk_idx': ci, 'obs_enc': _CURRENT_OBS_ENC[0].tolist(),
                           'steps': list(_COLLECT_BUF)})
            ci += 1
            arr = chunk.squeeze(0).cpu().numpy()
            queue = [arr[i].copy() for i in range(arr.shape[0])]
        a = queue.pop(0); obs, _, done, _ = env.step(a)
        if env.check_success():
            success = True; break
        if done:
            break
    n = len(chunks)
    for c in chunks:
        c['chunk_pos'] = c['chunk_idx'] / max(n, 1)
    con.execute('INSERT OR REPLACE INTO qc_rollouts VALUES (?,?,?,?,?,?,?,?)',
                (rid, ep['suite'], ep['task_idx'], ep['ep_idx'], ep['init_state_hash'],
                 int(success), n, _json.dumps(chunks)))
    con.commit()


if RUN_COLLECT:
    policy, preprocess, postprocess = load_pi05_session(video_dir=VIDEO_DIR)
    globals()['CURRENT_POLICY_MODEL'] = 'pi05'
    m = policy.model
    if not hasattr(m, '_orig_sample_actions'):
        m._orig_sample_actions = m.sample_actions
    m.sample_actions = _types.MethodType(_sample_actions_collect, m)
    LIBERO_SITE = os.path.dirname(get_libero_path('bddl_files'))
    try:
        restore_libero_pro_inits(PRO_INIT_SRC, LIBERO_SITE)
    except Exception as e:
        print('init restore:', e)
    eps = build_libero_pro_episodes(LIBERO_SITE, episode_idxs=list(range(N_COLLECT_EPS)),
                                    benchmark_dict_=benchmark_dict)
    con = _qc_init_db(QC_DB)
    for (suite, ti), grp in groupby(sorted(eps, key=lambda x: (x['suite'], x['task_idx'])),
                                    key=lambda x: (x['suite'], x['task_idx'])):
        grp = list(grp)
        env = OffScreenRenderEnv(bddl_file_name=grp[0]['bddl_path'], camera_names=CAMERAS,
                                 camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                                 has_offscreen_renderer=True, use_camera_obs=True,
                                 has_renderer=False, reward_shaping=False)
        try:
            for ep in tqdm(grp, leave=False, desc=f'collect {suite} T{ti}'):
                collect_episode(env, ep, policy, con)
        finally:
            env.close()
    print('qc_rollouts:', con.execute('SELECT COUNT(*) FROM qc_rollouts').fetchone()[0])
    con.close()
else:
    print('RUN_COLLECT=False -- using existing qc.db')
'''

PCP_MODEL_SRC = r'''# === QCorrector model + dataset (correction-step samples, hard-task filter) ==
import sqlite3, json as _json
import torch.nn as nn, torch.optim as optim
from collections import defaultdict

CORRECTION_STEPS = (7, 8)     # where PCP applies the gradient at deploy time
HARD_LO, HARD_HI = 0.10, 0.90 # hard-task filter (avoids observation-identity shortcut)


class QCorrector(nn.Module):
    """Q(z_hat, obs, [chunk_pos, s]) ~= P(success). 3-layer MLP + residual."""
    def __init__(self, action_feat_dim=350, obs_feat_dim=2048, pos_dim=2, hidden=256, dropout=0.2):
        super().__init__()
        self.action_norm = nn.LayerNorm(action_feat_dim)
        self.obs_proj = nn.Linear(obs_feat_dim, 128)
        self.obs_norm = nn.LayerNorm(128)
        self.net = nn.Sequential(
            nn.Linear(action_feat_dim + 128 + pos_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.res = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)

    def forward(self, action, chunk_pos, obs):
        obs_h = self.obs_norm(self.obs_proj(obs))
        h = self.net(torch.cat([self.action_norm(action), obs_h, chunk_pos], dim=-1))
        return self.out(h + self.res(h)).squeeze(-1)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / self.log_t.exp()


def _load_qc_samples(qc_db):
    con = sqlite3.connect(qc_db)
    rows = con.execute(
        'SELECT rollout_id, suite, task_idx, success, chunks FROM qc_rollouts').fetchall()
    con.close()
    task_succ = defaultdict(list)
    for rid, suite, ti, succ, ch in rows:
        task_succ[(suite, ti)].append(succ)
    hard = {k for k, v in task_succ.items() if HARD_LO < (sum(v) / len(v)) < HARD_HI}
    print(f'hard tasks (SR in {HARD_LO}-{HARD_HI}): {len(hard)} / {len(task_succ)}')
    by_rollout = defaultdict(list)
    for rid, suite, ti, succ, ch in rows:
        if (suite, ti) not in hard:
            continue
        for c in _json.loads(ch):
            for st in c['steps']:
                if st['step_idx'] not in CORRECTION_STEPS:
                    continue
                z = np.asarray(st['z_hat'], dtype=np.float32).reshape(-1)
                by_rollout[(rid, succ)].append(
                    (z, np.asarray(c['obs_enc'], dtype=np.float32),
                     np.asarray([c['chunk_pos'], st['s']], dtype=np.float32), int(succ)))
    return by_rollout


QC_SAMPLES = _load_qc_samples(QC_DB)
ACTION_DIM = next(iter(QC_SAMPLES.values()))[0][0].shape[0] if QC_SAMPLES else 350
OBS_DIM = next(iter(QC_SAMPLES.values()))[0][1].shape[0] if QC_SAMPLES else 2048
print(f'rollouts={len(QC_SAMPLES)}  action_dim={ACTION_DIM}  obs_dim={OBS_DIM}')
'''

PCP_TRAIN_SRC = r'''# === Train Q (AdamW, cosine, label smoothing) + temperature calibration ======
SEED = 42; TRAIN_FRAC = 0.80; LR = 3e-4; WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05; EPOCHS = 100; PATIENCE = 20; BATCH = 256
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

rng = np.random.default_rng(SEED)
pos = [k for k in QC_SAMPLES if k[1] == 1]
neg = [k for k in QC_SAMPLES if k[1] == 0]
rng.shuffle(pos); rng.shuffle(neg)
tr_keys = set(pos[:int(len(pos) * TRAIN_FRAC)]) | set(neg[:int(len(neg) * TRAIN_FRAC)])


def _stack(keys):
    Z, O, P, Y = [], [], [], []
    for k in keys:
        for z, o, p, y in QC_SAMPLES[k]:
            Z.append(z); O.append(o); P.append(p); Y.append(y)
    return (torch.tensor(np.array(Z)), torch.tensor(np.array(O)),
            torch.tensor(np.array(P)), torch.tensor(np.array(Y), dtype=torch.float32))


Ztr, Otr, Ptr, Ytr = _stack([k for k in QC_SAMPLES if k in tr_keys])
Zva, Ova, Pva, Yva = _stack([k for k in QC_SAMPLES if k not in tr_keys])
print(f'train chunks={len(Ytr)}  val chunks={len(Yva)}')

Q_MODEL = QCorrector(ACTION_DIM, OBS_DIM).to(device)
print('QCorrector params:', sum(p.numel() for p in Q_MODEL.parameters()))
opt = optim.AdamW(Q_MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR / 50)
pos_w = torch.tensor([(Ytr == 0).sum() / max((Ytr == 1).sum(), 1)], device=device)


def _smooth_bce(logits, y):
    y = y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
    return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)


def _batches(n):
    idx = torch.randperm(n)
    for i in range(0, n, BATCH):
        yield idx[i:i + BATCH]


best_auc, best_state, bad = -1.0, None, 0
for ep in range(EPOCHS):
    Q_MODEL.train()
    for b in _batches(len(Ytr)):
        opt.zero_grad()
        logit = Q_MODEL(Ztr[b].to(device), Ptr[b].to(device), Otr[b].to(device))
        loss = _smooth_bce(logit, Ytr[b].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Q_MODEL.parameters(), 1.0)
        opt.step()
    sched.step()
    Q_MODEL.eval()
    with torch.no_grad():
        vlogit = Q_MODEL(Zva.to(device), Pva.to(device), Ova.to(device)).cpu()
    auc = (roc_auc_score(Yva.numpy(), torch.sigmoid(vlogit).numpy())
           if roc_auc_score and Yva.unique().numel() > 1 else float('nan'))
    if auc > best_auc:
        best_auc, best_state, bad = auc, {k: v.cpu().clone() for k, v in Q_MODEL.state_dict().items()}, 0
    else:
        bad += 1
    if ep % 10 == 0 or bad == 0:
        print(f'epoch {ep:3d}  val AUC={auc:.4f}  best={best_auc:.4f}')
    if bad >= PATIENCE:
        print(f'early stop @ {ep}'); break

if best_state:
    Q_MODEL.load_state_dict(best_state)
print('best val AUC:', best_auc)

# Temperature calibration on val.
Q_SCALER = TemperatureScaler().to(device)
copt = optim.LBFGS(Q_SCALER.parameters(), lr=0.05, max_iter=100)
Q_MODEL.eval()
with torch.no_grad():
    vlogit = Q_MODEL(Zva.to(device), Pva.to(device), Ova.to(device)).detach()
yv = Yva.to(device)

def _cl():
    copt.zero_grad()
    l = nn.functional.binary_cross_entropy_with_logits(Q_SCALER(vlogit), yv)
    l.backward(); return l

if len(yv):
    copt.step(_cl)
print('temperature:', Q_SCALER.log_t.exp().item())
torch.save({'model': Q_MODEL.state_dict(), 'scaler': Q_SCALER.state_dict(),
            'action_dim': ACTION_DIM, 'obs_dim': OBS_DIM, 'val_auc': best_auc}, QC_CKPT)
print('saved', QC_CKPT)
'''

PCP_SANITY_SRC = r'''# === Offline gradient sanity check: delta-f on failure vs success chunks ======
import matplotlib.pyplot as plt

Q_MODEL.eval()
lams = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
df_succ, df_fail = [], []
Zv = Zva.to(device); Ov = Ova.to(device); Pv = Pva.to(device)
for lam in lams:
    z = Zv.clone().requires_grad_(True)
    with torch.enable_grad():
        f0 = torch.sigmoid(Q_SCALER(Q_MODEL(z, Pv, Ov)))
        f0.sum().backward()
    g = z.grad.detach()
    with torch.no_grad():
        fz = torch.sigmoid(Q_SCALER(Q_MODEL((z.detach() + lam * g), Pv, Ov)))
    delta = (fz - f0.detach()).cpu().numpy()
    y = Yva.numpy()
    df_succ.append(float(delta[y == 1].mean()) if (y == 1).any() else 0.0)
    df_fail.append(float(delta[y == 0].mean()) if (y == 0).any() else 0.0)

plt.figure(figsize=(5, 4))
plt.plot(lams, df_succ, 'o-', label='success chunks')
plt.plot(lams, df_fail, 'o-', label='failure chunks')
plt.xlabel('lambda'); plt.ylabel('mean delta-f after correction'); plt.legend()
plt.title('Offline gradient sanity (failure >> success expected)')
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'pcp_gradient_sanity.png'), dpi=150)
plt.show()
for lam, s, f in zip(lams, df_succ, df_fail):
    print(f'lambda={lam:>4}  df_succ={s:+.4f}  df_fail={f:+.4f}')
'''


PCP_EVAL_SRC = r'''# === Three-way live eval: vanilla / PnP-only (lambda=0) / PCP (lambda=3) ======
# Gated correction at steps (7,8): only steps with Q<0.5 receive the gradient.
import types as _types
from itertools import groupby
from tqdm.auto import tqdm

RUN_EVAL = True
EVAL_EPS = 10
LAMBDA_PCP = 3.0
Q_GATE = 0.5
CHUNK_SIZE = 50
_EVAL_CP = [0.0]
LAMBDA_LIVE = 0.0


def _refine_with_correction(x_t, s, vf, obs_enc, chunk_pos):
    x_acc = x_t; ah = []
    for _ in range(PNP_K):
        with torch.no_grad():
            v = vf(x_acc); a_hat = x_acc - s * v; ah.append(a_hat)
            x_acc = (1.0 - s) * a_hat + s * torch.empty_like(x_acc).normal_(
                0.0, 1.0, generator=_pnp_gen(x_acc.device))
    z_hat = torch.stack(ah, 0).mean(0)
    if LAMBDA_LIVE > 0 and Q_MODEL is not None:
        zc = z_hat[0, :, :7].reshape(1, -1).detach().clone().requires_grad_(True)
        cp = torch.tensor([[float(chunk_pos), float(s)]], dtype=torch.float32, device=zc.device)
        ob = obs_enc.reshape(1, -1).float()
        with torch.enable_grad():
            score = torch.sigmoid(Q_SCALER(Q_MODEL(zc, cp, ob)))
            if score.item() < Q_GATE:
                score.backward()
                g = zc.grad.detach().reshape(z_hat[0, :, :7].shape)
                a_star = z_hat.clone()
                a_star[0, :, :7] = z_hat[0, :, :7] + LAMBDA_LIVE * g
                with torch.no_grad():
                    return (1.0 - s) * a_star + s * torch.empty_like(x_acc).normal_(
                        0.0, 1.0, generator=_pnp_gen(x_acc.device))
    return x_acc


@torch.no_grad()
def _sample_actions_eval(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kw):
    if num_steps is None:
        num_steps = self.config.num_inference_steps
    bsize = tokens.shape[0]; dev = tokens.device
    if noise is None:
        noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), dev)
    if not PNP_CONFIG.enabled:
        return self._orig_sample_actions(images, img_masks, tokens, masks,
                                          noise=noise, num_steps=num_steps, **kw)
    prefix_embs, ppm, pam = self.embed_prefix(images, img_masks, tokens, masks)
    obs_enc = prefix_embs.mean(dim=1)[0].detach()
    att4d = self._prepare_attention_masks_4d(make_att_2d_masks(ppm, pam))
    pos = torch.cumsum(ppm, dim=1) - 1
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, pkv = self.paligemma_with_expert.forward(
        attention_mask=att4d, position_ids=pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True)
    dt = -1.0 / num_steps; x_t = noise
    for step in range(num_steps):
        s = 1.0 + step * dt
        tt = torch.tensor(s, dtype=torch.float32, device=dev).expand(bsize)
        def vf(inx, ts=tt):
            return self.denoise_step(prefix_pad_masks=ppm, past_key_values=pkv, x_t=inx, timestep=ts)
        if step in CORRECTION_STEPS:
            x_t = _refine_with_correction(x_t, s, vf, obs_enc, _EVAL_CP[0])
        v_t = vf(x_t); x_t = x_t + dt * v_t
    return x_t


def eval_episode(env, ep, policy, lam):
    global LAMBDA_LIVE
    LAMBDA_LIVE = lam
    PNP_CONFIG.enabled = lam is not None
    env.reset(); policy.reset()
    obs = env.set_init_state(ep['init_state'])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    _seed = _episode_seed(ep['init_state'], ep['ep_idx'])
    torch.manual_seed(_seed); torch.cuda.manual_seed(_seed); _pnp_seed_perturb(_seed)
    est_chunks = max(1, round(ep['max_steps'] / CHUNK_SIZE))
    queue, ci, success = [], 0, False
    for step in range(ep['max_steps']):
        if not queue:
            _EVAL_CP[0] = min(ci / est_chunks, 1.0)
            batch = preprocess(obs_to_policy(obs, ep['task_desc'], device))
            chunk = policy.predict_action_chunk(batch, noise=None)
            ci += 1
            arr = chunk.squeeze(0).cpu().numpy()
            queue = [arr[i].copy() for i in range(arr.shape[0])]
        a = queue.pop(0); obs, _, done, _ = env.step(a)
        if env.check_success():
            success = True; break
        if done:
            break
    return success


if RUN_EVAL:
    if 'Q_MODEL' not in dir():
        ckpt = torch.load(QC_CKPT, map_location=device)
        Q_MODEL = QCorrector(ckpt['action_dim'], ckpt['obs_dim']).to(device)
        Q_MODEL.load_state_dict(ckpt['model']); Q_MODEL.eval()
        Q_SCALER = TemperatureScaler().to(device); Q_SCALER.load_state_dict(ckpt['scaler'])
    policy, preprocess, postprocess = load_pi05_session(video_dir=VIDEO_DIR)
    globals()['CURRENT_POLICY_MODEL'] = 'pi05'
    m = policy.model
    if not hasattr(m, '_orig_sample_actions'):
        m._orig_sample_actions = m.sample_actions
    m.sample_actions = _types.MethodType(_sample_actions_eval, m)
    PNP_CONFIG.mode = 'both'; PNP_CONFIG.step_indices = CORRECTION_STEPS; PNP_CONFIG.num_iterations = PNP_K

    LIBERO_SITE = os.path.dirname(get_libero_path('bddl_files'))
    con = _qc_init_db(QC_DB)
    # restrict eval to hard PRO tasks identified during training
    hard_rows = con.execute('SELECT suite, task_idx, AVG(success) FROM qc_rollouts '
                            'GROUP BY suite, task_idx').fetchall()
    hard = {(s, t) for s, t, sr in hard_rows if HARD_LO < sr < HARD_HI}
    eps = [e for e in build_libero_pro_episodes(LIBERO_SITE, episode_idxs=list(range(EVAL_EPS)),
                                                benchmark_dict_=benchmark_dict)
           if (e['suite'], e['task_idx']) in hard]
    print(f'eval episodes (hard tasks): {len(eps)}')
    PASSES = [('vanilla', None), ('pnp_only', 0.0), ('pcp', LAMBDA_PCP)]
    for (suite, ti), grp in groupby(sorted(eps, key=lambda x: (x['suite'], x['task_idx'])),
                                    key=lambda x: (x['suite'], x['task_idx'])):
        grp = list(grp)
        env = OffScreenRenderEnv(bddl_file_name=grp[0]['bddl_path'], camera_names=CAMERAS,
                                 camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                                 has_offscreen_renderer=True, use_camera_obs=True,
                                 has_renderer=False, reward_shaping=False)
        try:
            for name, lam in PASSES:
                lab = -1.0 if lam is None else lam
                for ep in tqdm(grp, leave=False, desc=f'{name} {suite} T{ti}'):
                    rid = f"{ep['suite']}:{ep['task_idx']}:{ep['ep_idx']}:{ep['init_state_hash']}"
                    if con.execute('SELECT 1 FROM qc_eval WHERE rollout_id=? AND lambda=?',
                                   (rid, lab)).fetchone():
                        continue
                    succ = eval_episode(env, ep, policy, lam)
                    con.execute('INSERT OR REPLACE INTO qc_eval VALUES (?,?,?,?,?,?)',
                                (rid, ep['suite'], ep['task_idx'], ep['ep_idx'], lab, int(succ)))
                    con.commit()
        finally:
            env.close()
    for name, lam in PASSES:
        lab = -1.0 if lam is None else lam
        r = con.execute('SELECT AVG(success), COUNT(*) FROM qc_eval WHERE lambda=?', (lab,)).fetchone()
        print(f'{name:10s} lambda={lab:>4}  SR={(r[0] or 0)*100:.1f}%  (n={r[1]})')
    con.close()
'''


def build_nb02():
    cells = [
        md("# 02 - PCP (Predict-Correct-Perturb) train + eval\n\n"
           "Consolidated, self-contained Q-function corrector pipeline for **pi0.5** "
           "on LIBERO-PRO: (1) collect labeled clean-action chunks, (2) train the "
           "`QCorrector`, (3) calibrate + gradient sanity, (4) three-way live eval "
           "(vanilla / PnP-only `lambda=0` / PCP `lambda=3`) -> **Table 6**.\n\n"
           "Uses the same RNG-isolated perturbation generator as notebook 01.\n\n"
           "**Cleanups applied vs the report:** the perturbation path uses the "
           "dedicated generator; `lambda` is a single deploy-time constant "
           "(`LAMBDA_PCP=3.0`, `lambda=0` = PnP-only) instead of the inconsistent "
           "0.5/3.0 values; the QCorrector param count is printed so the paper's "
           "~450K figure can be reconciled with the actual model."),
        md("## 1. Config"),
        code(CONFIG_SRC),
        md("## 2. Environment"),
        code(ENV_SETUP_SRC),
        md("## 3. P&P core (RNG-isolated) + loaders"),
        code(CORE_SRC),
        LIBERO_PRO_MD_CELL,
        md("## 4. Data collection"),
        code(PCP_COLLECT_SRC),
        md("## 5. QCorrector model + dataset"),
        code(PCP_MODEL_SRC),
        md("## 6. Training + temperature calibration"),
        code(PCP_TRAIN_SRC),
        md("## 7. Offline gradient sanity check"),
        code(PCP_SANITY_SRC),
        md("## 8. Three-way live eval (Table 6)"),
        code(PCP_EVAL_SRC),
    ]
    return nb(cells)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03: analysis (tables, figures, 6/19 fixes, geometry)
# ─────────────────────────────────────────────────────────────────────────────
AN_CONFIG_SRC = r'''# === Analysis config (runs anywhere the DBs / ahats are reachable) ===========
import os
try:
    from google.colab import drive
    drive.mount('/content/drive'); DRIVE = '/content/drive/MyDrive'
except Exception:
    DRIVE = os.path.expanduser('~')

FINAL_ROOT  = os.path.join(DRIVE, 'cs159-sp26-final')
RESULTS_DIR = os.path.join(FINAL_ROOT, 'results')
AHATS_DIR   = os.path.join(RESULTS_DIR, 'ahats')
FIGURES_DIR = os.path.join(FINAL_ROOT, 'figures')
TABLES_DIR  = os.path.join(FINAL_ROOT, 'tables')
SLICE_DB = os.path.join(RESULTS_DIR, 'rollouts_final.db')
PRO_DB   = os.path.join(RESULTS_DIR, 'rollouts_pro.db')
QC_DB    = os.path.join(RESULTS_DIR, 'qc.db')
for _d in (FIGURES_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)
print('FINAL_ROOT =', FINAL_ROOT)
'''

AN_HELPERS_SRC = r'''# === Shared analysis helpers ================================================
import sqlite3, json, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
except Exception:
    roc_auc_score = average_precision_score = f1_score = None

pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40)

# Known report (Aviato Dynamics) headline numbers, for the delta-vs-report CSV.
REPORT_NUMBERS = {
    'slice_SR_vanilla': 76.2,
    'slice_SR_extra_steps': None,
    'slice_SR_pnp_uncertainty_only': 90.0,
    'slice_SR_pnp_refinement': None,
}
DELTAS = []   # rows: metric, report_value, recomputed_value


def load_rollouts(path):
    if not os.path.exists(path):
        print('MISSING:', path); return pd.DataFrame()
    con = sqlite3.connect(path)
    df = pd.read_sql_query('SELECT * FROM rollouts', con); con.close()
    return df


def detector_metrics(score, fail):
    score = np.asarray(score, float); fail = np.asarray(fail, int)
    m = np.isfinite(score); score, fail = score[m], fail[m]
    out = {'n': int(len(score)), 'n_fail': int(fail.sum())}
    if roc_auc_score is None or len(np.unique(fail)) < 2 or len(score) < 3:
        out.update(roc_auc=np.nan, pr_auc=np.nan, spearman=np.nan, f1=np.nan, tau=np.nan)
        return out
    out['roc_auc'] = roc_auc_score(fail, score)
    out['pr_auc'] = average_precision_score(fail, score)
    out['spearman'] = stats.spearmanr(score, fail).correlation
    best_f1, best_t = -1.0, np.nan
    for t in np.quantile(score, np.linspace(0.05, 0.95, 19)):
        f = f1_score(fail, (score >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    out['f1'], out['tau'] = best_f1, best_t
    return out


def stratified_auc(df, score_col, by='suite', fail_col='fail'):
    """6/19 section 1.3 fix: AUC computed within each suite, then averaged."""
    aucs = []
    for s, g in df.groupby(by):
        if roc_auc_score is None or g[fail_col].nunique() < 2:
            continue
        aucs.append(roc_auc_score(g[fail_col], g[score_col]))
    return float(np.mean(aucs)) if aucs else np.nan, len(aucs)


def savefig(name):
    p = os.path.join(FIGURES_DIR, name)
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.show()
    print('wrote', p)


def savetable(df, name):
    p = os.path.join(TABLES_DIR, name)
    df.to_csv(p, index=False); print('wrote', p)
'''

AN_SLICE_SRC = r'''# === Tables 2-5: controlled 80-ep slice =====================================
slice_df = load_rollouts(SLICE_DB)
if not slice_df.empty:
    slice_df['fail'] = 1 - slice_df['success']
    keys = ['policy_model', 'method', 'suite', 'task_idx', 'episode_idx', 'init_state_hash']

    # Best-of-three step-config collapse (success = max over the 3 P&P step configs).
    collapsed = (slice_df.groupby(keys, dropna=False)
                 .agg(success=('success', 'max'),
                      u_mean_episode=('u_mean_episode', 'mean'),
                      n_steps=('n_steps', 'mean')).reset_index())

    # Table 2: SR by method/model.
    t2 = (collapsed.groupby(['policy_model', 'method'])
          .agg(SR=('success', 'mean'), n=('success', 'size')).reset_index())
    t2['SR'] = (100 * t2['SR']).round(1)
    print('=== Table 2: success rate by method ===')
    print(t2.to_string(index=False))
    savetable(t2, 'table2_success_rate.csv')
    for _, r in t2.iterrows():
        key = f"slice_SR_{r['method']}"
        if r['policy_model'] == 'pi05' and key in REPORT_NUMBERS:
            DELTAS.append(dict(metric=f"{key} (pi05)", report_value=REPORT_NUMBERS[key],
                               recomputed_value=r['SR']))

    # Table 3: recovery / degradation vs vanilla.
    t3rows = []
    for model, md_ in collapsed.groupby('policy_model'):
        piv = md_.pivot_table(index=['suite', 'task_idx', 'episode_idx', 'init_state_hash'],
                              columns='method', values='success')
        if 'vanilla' not in piv:
            continue
        v = piv['vanilla']
        for method in [m for m in piv.columns if m != 'vanilla']:
            mth = piv[method]; mask = v.notna() & mth.notna()
            vv, mm = v[mask], mth[mask]
            t3rows.append(dict(policy_model=model, method=method, n=int(mask.sum()),
                               recovery=int(((vv == 0) & (mm == 1)).sum()),
                               degradation=int(((vv == 1) & (mm == 0)).sum())))
    t3 = pd.DataFrame(t3rows)
    print('\n=== Table 3: recovery / degradation vs vanilla ===')
    print(t3.to_string(index=False))
    savetable(t3, 'table3_recovery_degradation.csv')

    # Table 4: detector metrics per step config (pnp_uncertainty_only, U->failure).
    det = slice_df[slice_df['method'] == 'pnp_uncertainty_only'].copy()
    t4rows = []
    for (model, cfg), g in det.groupby(['policy_model', 'pnp_step_indices']):
        m = detector_metrics(g['u_mean_episode'], g['fail'])
        t4rows.append(dict(policy_model=model, step_config=cfg, **m))
    t4 = pd.DataFrame(t4rows)
    print('\n=== Table 4: P&P detector metrics per step config ===')
    print(t4.to_string(index=False))
    savetable(t4, 'table4_detector_metrics.csv')

    # Table 5: median-split uncertainty taxonomy (low/high U x success/fail).
    t5rows = []
    for model, g in det.groupby('policy_model'):
        gg = g.dropna(subset=['u_mean_episode'])
        if gg.empty:
            continue
        med = gg['u_mean_episode'].median()
        gg = gg.assign(U=np.where(gg['u_mean_episode'] >= med, 'high_U', 'low_U'))
        tab = pd.crosstab(gg['U'], np.where(gg['success'] == 1, 'success', 'fail'))
        print(f'\n=== Table 5: taxonomy ({model}) median U={med:.4f} ===')
        print(tab)
        tab2 = tab.reset_index(); tab2.insert(0, 'policy_model', model)
        t5rows.append(tab2)
    if t5rows:
        savetable(pd.concat(t5rows, ignore_index=True), 'table5_taxonomy.csv')
else:
    print('No slice data yet -- run notebook 01 first.')
'''

AN_STRATIFY_SRC = r'''# === 6/19 section 1.3: stratified (per-suite) AUC vs pooled AUC ==============
strat_rows = []
for label, path, method in [('slice', SLICE_DB, 'pnp_uncertainty_only'),
                            ('pro', PRO_DB, 'pnp_uncertainty_only')]:
    df = load_rollouts(path)
    if df.empty:
        continue
    df = df[df['method'] == method].copy()
    df['fail'] = 1 - df['success']
    df = df.dropna(subset=['u_mean_episode'])
    for model, g in df.groupby('policy_model'):
        if roc_auc_score is None or g['fail'].nunique() < 2:
            continue
        pooled = roc_auc_score(g['fail'], g['u_mean_episode'])      # old (pooled) method
        strat, n_suites = stratified_auc(g, 'u_mean_episode')       # 6/19 fix
        strat_rows.append(dict(dataset=label, policy_model=model, pooled_auc=round(pooled, 4),
                               stratified_auc=round(strat, 4), n_suites=n_suites,
                               delta=round(pooled - strat, 4)))
        DELTAS.append(dict(metric=f'{label} detector AUC pooled ({model})',
                           report_value='pooled', recomputed_value=round(pooled, 4)))
        DELTAS.append(dict(metric=f'{label} detector AUC stratified ({model})',
                           report_value='per-suite-mean', recomputed_value=round(strat, 4)))
strat = pd.DataFrame(strat_rows)
print('=== Pooled vs per-suite-stratified detector AUC ===')
print(strat.to_string(index=False) if not strat.empty else 'no detector rows')
if not strat.empty:
    savetable(strat, 'stratified_vs_pooled_auc.csv')
'''


AN_PRO_SRC = r'''# === Tables 8-9: LIBERO-PRO aggregate + per-DOF ============================
def load_perdof(path, method='pnp_uncertainty_only'):
    if not os.path.exists(path):
        return pd.DataFrame()
    con = sqlite3.connect(path)
    dims = ", ".join([f"AVG(e.u_d{i}) ud{i}" for i in range(7)])
    df = pd.read_sql_query(
        "SELECT r.rollout_id, r.suite, r.task_idx, r.episode_idx, r.success, r.n_steps, "
        "r.policy_model, AVG(e.u_mean) u_mean, " + dims + " "
        "FROM rollouts r JOIN pnp_euler_steps e ON r.rollout_id=e.rollout_id "
        "WHERE r.method=? GROUP BY r.rollout_id", con, params=(method,))
    con.close()
    return df

pro_df = load_rollouts(PRO_DB)
if not pro_df.empty:
    pro_df['fail'] = 1 - pro_df['success']
    # Table 8: baseline SR + per-suite detector AUC.
    base = pro_df[pro_df['method'] == 'vanilla']
    unc = pro_df[pro_df['method'] == 'pnp_uncertainty_only'].dropna(subset=['u_mean_episode'])
    t8rows = []
    for model in pro_df['policy_model'].dropna().unique():
        for suite in sorted(pro_df['suite'].unique()):
            b = base[(base.policy_model == model) & (base.suite == suite)]
            u = unc[(unc.policy_model == model) & (unc.suite == suite)]
            auc = (roc_auc_score(u['fail'], u['u_mean_episode'])
                   if roc_auc_score and u['fail'].nunique() > 1 else np.nan)
            t8rows.append(dict(policy_model=model, suite=suite,
                               baseline_SR=round(100 * b['success'].mean(), 1) if len(b) else np.nan,
                               n=len(u), detector_auc=round(auc, 4) if auc == auc else np.nan))
    t8 = pd.DataFrame(t8rows)
    print('=== Table 8: LIBERO-PRO baseline SR + per-suite detector AUC ===')
    print(t8.to_string(index=False))
    savetable(t8, 'table8_pro_aggregate.csv')

    # Table 9: 7-dim vs pos+grip subset AUC + per-DOF AUC.
    pdf = load_perdof(PRO_DB)
    if not pdf.empty:
        pdf['fail'] = 1 - pdf['success']
        pos_grip = ['ud0', 'ud1', 'ud2', 'ud6']
        all_dims = [f'ud{i}' for i in range(7)]
        pdf['score_full'] = np.sqrt((pdf[all_dims] ** 2).sum(axis=1))
        pdf['score_posgrip'] = np.sqrt((pdf[pos_grip] ** 2).sum(axis=1))
        t9rows = []
        for model, g in pdf.groupby('policy_model'):
            full_s, _ = stratified_auc(g, 'score_full')
            pg_s, _ = stratified_auc(g, 'score_posgrip')
            row = dict(policy_model=model, auc_7dim=round(full_s, 4), auc_pos_grip=round(pg_s, 4))
            for i in range(7):
                a, _ = stratified_auc(g, f'ud{i}')
                row[f'auc_ud{i}'] = round(a, 4)
            t9rows.append(row)
        t9 = pd.DataFrame(t9rows)
        print('\n=== Table 9: per-DOF detector AUC (per-suite averaged) ===')
        print(t9.to_string(index=False))
        savetable(t9, 'table9_per_dof.csv')
else:
    print('No PRO data yet -- run notebook 01 PRO driver first.')
'''

AN_PCP_SRC = r'''# === Table 6: PCP three-way eval ===========================================
if os.path.exists(QC_DB):
    con = sqlite3.connect(QC_DB)
    try:
        qe = pd.read_sql_query('SELECT * FROM qc_eval', con)
    except Exception:
        qe = pd.DataFrame()
    con.close()
    if not qe.empty:
        name_map = {-1.0: 'vanilla', 0.0: 'PnP-only (lambda=0)', 3.0: 'PCP (lambda=3)'}
        qe['arm'] = qe['lambda'].map(lambda x: name_map.get(x, f'lambda={x}'))
        t6 = (qe.groupby('arm').agg(SR=('success', 'mean'), n=('success', 'size')).reset_index())
        t6['SR'] = (100 * t6['SR']).round(1)
        print('=== Table 6: PCP three-way (LIBERO-PRO hard tasks) ===')
        print(t6.to_string(index=False))
        savetable(t6, 'table6_pcp.csv')
        for _, r in t6.iterrows():
            DELTAS.append(dict(metric=f"PCP {r['arm']} SR", report_value=None, recomputed_value=r['SR']))
    else:
        print('qc_eval empty -- run notebook 02 eval first.')
else:
    print('No qc.db -- run notebook 02 first.')
'''

AN_GEOMETRY_SRC = r'''# === Geometry of uncertainty (net-new, 6/19 sections 2.2-2.6) ===============
def _sarle_bc(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return np.nan
    g = stats.skew(x); k = stats.kurtosis(x, fisher=True)
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g ** 2 + 1.0) / denom if denom else np.nan


def load_ahats_index(ahats_dir, ref_df):
    """Map each persisted ahats npz (named <rollout_id>.npz) to its rollout row."""
    if not os.path.isdir(ahats_dir):
        print('No ahats dir:', ahats_dir); return []
    ref = ref_df.set_index('rollout_id') if 'rollout_id' in ref_df else None
    items = []
    for fp in glob.glob(os.path.join(ahats_dir, '*.npz')):
        rid = os.path.splitext(os.path.basename(fp))[0]
        if ref is None or rid not in ref.index:
            continue
        items.append((fp, ref.loc[rid]))
    return items


pro_unc = pro_df[pro_df['method'] == 'pnp_uncertainty_only'].copy() if not pro_df.empty else pd.DataFrame()
ah_items = load_ahats_index(AHATS_DIR, pro_unc) if not pro_unc.empty else []
print(f'ahats episodes available: {len(ah_items)}')

if ah_items:
    bc_rows, dir_rows = [], []
    suite_vecs = {}    # suite -> list of per-episode mean-action vectors (for PCA)
    for fp, row in ah_items:
        npz = np.load(fp)
        # each entry: (K, B, chunk, adim); pool deviations across stored steps.
        devs, means, parr, latv = [], [], [], []
        for key in npz.files:
            A = npz[key]                       # (K,B,chunk,adim)
            A = A.reshape(A.shape[0], -1)      # (K, D)
            m = A.mean(0)
            d = A - m                          # (K, D)
            devs.append(d.reshape(-1))
            means.append(m)
            nrm = np.linalg.norm(m) + 1e-8
            u = m / nrm
            proj = d @ u                       # (K,)  parallel component
            parr.append(float((proj ** 2).mean()))
            latv.append(float((np.sum(d ** 2, axis=1) - proj ** 2).mean() / max(d.shape[1] - 1, 1)))
        bc = _sarle_bc(np.concatenate(devs)) if devs else np.nan
        bc_rows.append(dict(suite=row['suite'], success=int(row['success']), fail=int(1 - row['success']),
                            bc=bc, var_par=float(np.mean(parr)), var_lat=float(np.mean(latv)),
                            par_lat_ratio=float(np.mean(parr) / (np.mean(latv) + 1e-8))))
        suite_vecs.setdefault(row['suite'], []).append(np.mean(means, axis=0))

    bc_df = pd.DataFrame(bc_rows)

    # 2.2 Multimodality: BC by outcome + AUC(BC -> failure), per suite then averaged.
    bc_summary = (bc_df.groupby('suite')
                  .apply(lambda g: pd.Series({
                      'BC|succ': g.loc[g.success == 1, 'bc'].mean(),
                      'BC|fail': g.loc[g.fail == 1, 'bc'].mean(),
                      'n': len(g)})).reset_index())
    bc_auc, _ = stratified_auc(bc_df.dropna(subset=['bc']), 'bc', fail_col='fail')
    print('=== Multimodality (Sarle BC, 0.555 = bimodal threshold) ===')
    print(bc_summary.to_string(index=False))
    print(f'AUC(BC -> failure), per-suite mean = {bc_auc:.4f}')
    savetable(bc_summary, 'geometry_multimodality_bc.csv')

    # 2.4 Directional: Tier-2 var_par / var_lat ratio + AUC.
    ratio_auc, _ = stratified_auc(bc_df.dropna(subset=['par_lat_ratio']), 'par_lat_ratio', fail_col='fail')
    dir_summary = (bc_df.groupby('suite')
                   .agg(var_par=('var_par', 'mean'), var_lat=('var_lat', 'mean'),
                        ratio=('par_lat_ratio', 'mean')).reset_index())
    print('\n=== Directional (Tier-2 var_par/var_lat) ===')
    print(dir_summary.to_string(index=False))
    print(f'AUC(par/lat ratio -> failure), per-suite mean = {ratio_auc:.4f}')
    savetable(dir_summary, 'geometry_directional.csv')

    # 2.3 PCA isotropy on per-episode mean-action vectors (Marchenko-Pastur reference).
    iso_rows = []
    for suite, vecs in suite_vecs.items():
        X = np.vstack(vecs)
        if X.shape[0] < 3:
            continue
        Xc = X - X.mean(0)
        N, D = Xc.shape
        sv = np.linalg.svd(Xc, compute_uv=False)
        ev = (sv ** 2) / max(N - 1, 1)
        pc1_frac = float(ev[0] / ev.sum()) if ev.sum() else np.nan
        sigma2 = float(ev.mean())
        ratio = D / N
        mp_max = sigma2 * (1 + np.sqrt(ratio)) ** 2      # MP upper edge (noise reference)
        iso_rows.append(dict(suite=suite, N=N, D=D, pc1_frac=round(pc1_frac, 4),
                             lambda_max=round(float(ev[0]), 6), mp_upper_edge=round(mp_max, 6),
                             above_mp=bool(ev[0] > mp_max)))
    iso = pd.DataFrame(iso_rows)
    print('\n=== PCA isotropy (PC1 fraction vs Marchenko-Pastur) ===')
    print(iso.to_string(index=False))
    savetable(iso, 'geometry_pca_isotropy.csv')
else:
    print('No ahats persisted -- run notebook 01 PRO uncertainty pass (record_per_iteration=True).')

# 2.5 Online features + 2.6 length confound (from pnp_euler_steps; no ahats needed).
if os.path.exists(PRO_DB):
    con = sqlite3.connect(PRO_DB)
    steps = pd.read_sql_query(
        "SELECT e.rollout_id, e.chunk_idx, AVG(e.u_mean) u "
        "FROM pnp_euler_steps e JOIN rollouts r ON r.rollout_id=e.rollout_id "
        "WHERE r.method='pnp_uncertainty_only' GROUP BY e.rollout_id, e.chunk_idx", con)
    meta = pd.read_sql_query(
        "SELECT rollout_id, suite, success, n_steps, u_mean_episode FROM rollouts "
        "WHERE method='pnp_uncertainty_only'", con)
    con.close()
    if not steps.empty and not meta.empty:
        meta = meta.set_index('rollout_id')
        thr = steps['u'].quantile(0.75)
        feat_rows = []
        for rid, g in steps.groupby('rollout_id'):
            if rid not in meta.index:
                continue
            seq = g.sort_values('chunk_idx')['u'].to_numpy()
            ema = seq[0]
            for v in seq[1:]:
                ema = 0.7 * ema + 0.3 * v
            run = np.maximum.accumulate(np.cumsum(seq) / (np.arange(len(seq)) + 1))
            m = meta.loc[rid]
            feat_rows.append(dict(
                rollout_id=rid, suite=m['suite'], fail=int(1 - m['success']),
                n_steps=int(m['n_steps']),
                f_instant=float(np.mean(seq)),
                f_ema=float(ema),
                f_dev_runmean=float(np.max(seq - run)),
                f_cum_highU=float(np.sum(seq >= thr)),
                f_chunk_idx=float(len(seq)),
                f_progress=float(m['n_steps']),
                f_early=float(np.mean(seq[:2])),
            ))
        F = pd.DataFrame(feat_rows)
        online_feats = ['f_instant', 'f_ema', 'f_dev_runmean', 'f_cum_highU',
                        'f_chunk_idx', 'f_progress', 'f_early']
        on_rows = [dict(feature=f, auc=round(stratified_auc(F.dropna(subset=[f]), f, fail_col='fail')[0], 4))
                   for f in online_feats]
        on_df = pd.DataFrame(on_rows)
        print('\n=== Online feature gating: per-suite-mean AUC(feature -> failure) ===')
        print(on_df.to_string(index=False))
        savetable(on_df, 'geometry_online_features.csv')

        # Length confound: r(U, episode length) + early-window vs full AUC.
        r_ul = stats.pearsonr(F['f_instant'], F['n_steps'])[0] if len(F) > 2 else np.nan
        auc_full, _ = stratified_auc(F, 'f_instant', fail_col='fail')
        auc_early, _ = stratified_auc(F, 'f_early', fail_col='fail')
        print(f'\n=== Length confound ===\nr(U, length) = {r_ul:.4f}  '
              f'AUC(full U)={auc_full:.4f}  AUC(early-window U)={auc_early:.4f}')
        savetable(pd.DataFrame([dict(r_U_length=round(r_ul, 4), auc_full_U=round(auc_full, 4),
                                     auc_early_U=round(auc_early, 4))]),
                  'geometry_length_confound.csv')
'''

AN_CROSSMODEL_SRC = r'''# === Cross-model detector transfer + delta-vs-report CSV ====================
cm_rows = []
for label, path in [('slice', SLICE_DB), ('pro', PRO_DB)]:
    df = load_rollouts(path)
    if df.empty:
        continue
    df = df[df['method'] == 'pnp_uncertainty_only'].dropna(subset=['u_mean_episode']).copy()
    df['fail'] = 1 - df['success']
    for model, g in df.groupby('policy_model'):
        auc, n = stratified_auc(g, 'u_mean_episode')
        cm_rows.append(dict(dataset=label, policy_model=model, detector_auc=round(auc, 4), n_suites=n))
cm = pd.DataFrame(cm_rows)
print('=== Cross-model detector AUC (per-suite averaged) ===')
print(cm.to_string(index=False) if not cm.empty else 'no data')
if not cm.empty:
    savetable(cm, 'crossmodel_detector_auc.csv')
    piv = cm.pivot_table(index='dataset', columns='policy_model', values='detector_auc')
    if {'pi05', 'smolvla'}.issubset(piv.columns):
        ax = piv.plot(kind='bar', figsize=(6, 4))
        ax.set_ylabel('per-suite-mean detector AUC'); ax.set_title('pi0.5 vs SmolVLA P&P detector')
        savefig('crossmodel_detector_auc.png')

delta_df = pd.DataFrame(DELTAS)
print('\n=== Numbers changed vs report ===')
print(delta_df.to_string(index=False) if not delta_df.empty else 'no deltas recorded')
if not delta_df.empty:
    savetable(delta_df, 'numbers_changed_vs_report.csv')
print('\nAnalysis complete. Tables ->', TABLES_DIR, ' Figures ->', FIGURES_DIR)
'''


def build_nb03():
    cells = [
        md("# 03 - Analysis (tables, figures, 6/19 fixes, geometry)\n\n"
           "Analysis-only: reads the unified DBs from notebooks 01/02 and regenerates "
           "every report artifact plus the 6/19 corrections. **No GPU/simulator "
           "needed.**\n\n"
           "- **Tables 2-5** (controlled slice): SR, recovery/degradation, detector "
           "metrics, taxonomy (best-of-three step-config collapse).\n"
           "- **6/19 section 1.3**: per-suite-stratified AUC reported alongside pooled.\n"
           "- **Tables 8-9** (LIBERO-PRO): baseline SR + per-suite/per-DOF detector AUC.\n"
           "- **Table 6**: PCP three-way eval.\n"
           "- **Geometry (net-new)**: Sarle bimodality, PCA isotropy + Marchenko-Pastur, "
           "directional var_par/var_lat, online-feature gating, length confound.\n"
           "- **Cross-model** pi0.5 vs SmolVLA transfer + a *numbers-changed-vs-report* CSV."),
        md("## 1. Config + helpers"),
        code(AN_CONFIG_SRC),
        code(AN_HELPERS_SRC),
        md("## 2. Tables 2-5 (controlled slice)"),
        code(AN_SLICE_SRC),
        md("## 3. Stratified vs pooled detector AUC (6/19 section 1.3)"),
        code(AN_STRATIFY_SRC),
        md("## 4. Tables 8-9 (LIBERO-PRO + per-DOF)"),
        code(AN_PRO_SRC),
        md("## 5. Table 6 (PCP)"),
        code(AN_PCP_SRC),
        md("## 6. Geometry of uncertainty"),
        code(AN_GEOMETRY_SRC),
        md("## 7. Cross-model transfer + delta-vs-report"),
        code(AN_CROSSMODEL_SRC),
    ]
    return nb(cells)


def main():
    FINAL.mkdir(exist_ok=True)
    out = {
        "01_run_experiments.ipynb": build_nb01(),
        "02_pcp_train_eval.ipynb": build_nb02(),
        "03_analysis.ipynb": build_nb03(),
    }
    for name, doc in out.items():
        p = FINAL / name
        p.write_text(json.dumps(doc, indent=1))
        print(f"wrote {p}  ({len(doc['cells'])} cells)")


if __name__ == "__main__":
    main()
