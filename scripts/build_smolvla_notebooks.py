#!/usr/bin/env python3
"""Generate test_smolvla_jennifer.ipynb and pnp_smolvla_jennifer_analysis.ipynb."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


SMOLVLA_PNP = r'''
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

@torch.no_grad()
def _sample_actions_pnp_smolvla(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
    """Drop-in replacement for SmolVLAPytorch.sample_actions with optional P&P."""
    cfg = PNP_CONFIG
    if (not cfg.enabled) or self._rtc_enabled():
        return self._orig_sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)

    num_steps = INFERENCE_NUM_STEPS_OVERRIDE or self.config.num_steps
    bsize = state.shape[0]
    device = state.device
    if noise is None:
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)

    measure_only_output = None
    if cfg.mode == 'uncertainty':
        measure_only_output = self._orig_sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone(), **kwargs
        ).clone()

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_key_values = self.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=self.config.use_cache,
        fill_kv_cache=True,
    )

    dt = -1.0 / num_steps
    x_t = noise
    chunk_rec = {'num_steps': num_steps, 'steps': []}

    for step in range(num_steps):
        time = 1.0 + step * dt
        s = time
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            )

        if cfg.step_selected(step, s):
            x_t, rec = _pnp_refine_at_step(x_t, s, denoise_step_partial_call, cfg)
            rec['step'] = step
            chunk_rec['steps'].append(rec)

        if self._rtc_enabled():
            v_t = self.rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=kwargs.get('prev_chunk_left_over'),
                inference_delay=kwargs.get('inference_delay'),
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=kwargs.get('execution_horizon'),
            )
        else:
            v_t = denoise_step_partial_call(x_t)
        x_t = x_t + dt * v_t

    _pnp_log_chunk(chunk_rec)
    return measure_only_output if measure_only_output is not None else x_t
'''

ROLLoutDB = r'''
_ADIM = 7
_U_DIM_COLS    = [f'u_d{i}'     for i in range(_ADIM)]
_ASTD_DIM_COLS = [f'a_std_d{i}' for i in range(_ADIM)]
_DIM_COLS      = _U_DIM_COLS + _ASTD_DIM_COLS
_EXTRA_ROLLOUT_COLS = [
    'method', 'final_eval_slice', 'num_inference_steps', 'num_samples',
    'action_delta_l2_mean', 'action_delta_l2_max', 'action_var_mean',
    'gripper_flip_count', 'gripper_flip_rate', 'chunk_disagreement_mean',
    'policy_model',
]


class RolloutDB:
    """SQLite store for rollout outcomes, P&P uncertainty, and experiment metadata."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS rollouts (
        rollout_id        TEXT PRIMARY KEY,
        suite             TEXT,
        task_idx          INTEGER,
        task_desc         TEXT,
        episode_idx       INTEGER,
        init_state_hash   TEXT,
        success           INTEGER,
        n_steps           INTEGER,
        elapsed_s         REAL,
        pnp_enabled       INTEGER,
        pnp_k             INTEGER,
        pnp_step_indices  TEXT,
        pnp_mode          TEXT,
        u_mean_episode    REAL,
        u_max_episode     REAL,
        n_pnp_activations INTEGER,
        timestamp         TEXT,
        video_path        TEXT,
        method                    TEXT,
        final_eval_slice          INTEGER,
        num_inference_steps       INTEGER,
        num_samples               INTEGER,
        action_delta_l2_mean      REAL,
        action_delta_l2_max       REAL,
        action_var_mean           REAL,
        gripper_flip_count        INTEGER,
        gripper_flip_rate         REAL,
        chunk_disagreement_mean   REAL,
        policy_model              TEXT
    );
    CREATE TABLE IF NOT EXISTS pnp_euler_steps (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        rollout_id   TEXT    NOT NULL REFERENCES rollouts(rollout_id),
        chunk_idx    INTEGER NOT NULL,
        euler_step   INTEGER NOT NULL,
        s            REAL,
        u_mean       REAL,
        u_max        REAL,
        a_std_mean   REAL,
        u_d0 REAL, u_d1 REAL, u_d2 REAL, u_d3 REAL, u_d4 REAL, u_d5 REAL, u_d6 REAL,
        a_std_d0 REAL, a_std_d1 REAL, a_std_d2 REAL, a_std_d3 REAL,
        a_std_d4 REAL, a_std_d5 REAL, a_std_d6 REAL
    );
    CREATE INDEX IF NOT EXISTS idx_pes_rollout ON pnp_euler_steps(rollout_id);
    """

    def __init__(self, db_path):
        self.db_path = str(db_path)
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        self._con.executescript(self._DDL)
        self._migrate_schema()
        self._con.commit()
        n = self._con.execute('SELECT COUNT(*) FROM rollouts').fetchone()[0]
        print(f"RolloutDB: {self.db_path}  ({n} existing rollouts)")

    def _migrate_schema(self):
        pes_cols = {row[1] for row in self._con.execute("PRAGMA table_info(pnp_euler_steps)")}
        for col in _DIM_COLS:
            if col not in pes_cols:
                self._con.execute(f'ALTER TABLE pnp_euler_steps ADD COLUMN {col} REAL')
        rollout_cols = {row[1] for row in self._con.execute("PRAGMA table_info(rollouts)")}
        type_map = {
            'video_path': 'TEXT', 'method': 'TEXT', 'final_eval_slice': 'INTEGER',
            'num_inference_steps': 'INTEGER', 'num_samples': 'INTEGER',
            'action_delta_l2_mean': 'REAL', 'action_delta_l2_max': 'REAL',
            'action_var_mean': 'REAL', 'gripper_flip_count': 'INTEGER',
            'gripper_flip_rate': 'REAL', 'chunk_disagreement_mean': 'REAL',
            'policy_model': 'TEXT',
        }
        for col, typ in type_map.items():
            if col not in rollout_cols:
                self._con.execute(f'ALTER TABLE rollouts ADD COLUMN {col} {typ}')

    @staticmethod
    def init_state_hash(init_state):
        return hashlib.md5(np.asarray(init_state).tobytes()).hexdigest()[:12]

    @staticmethod
    def make_rollout_id(suite, task_idx, episode_idx, init_state, pnp_cfg,
                        method=None, num_inference_steps=None, num_samples=None,
                        policy_model=None):
        cfg_str = _json.dumps({
            'enabled': pnp_cfg.enabled,
            'k': pnp_cfg.num_iterations,
            'step_indices': list(pnp_cfg.step_indices) if pnp_cfg.step_indices else None,
            'time_min': pnp_cfg.time_min,
            'mode': pnp_cfg.mode,
            'method': method,
            'num_inference_steps': num_inference_steps,
            'num_samples': num_samples,
            'policy_model': policy_model,
        }, sort_keys=True)
        key = f'{suite}:{task_idx}:{episode_idx}:{RolloutDB.init_state_hash(init_state)}:{cfg_str}'
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def log_episode(self, rollout_id, suite, task_idx, task_desc, episode_idx,
                    init_state, success, n_steps, elapsed_s, pnp_cfg, episode_rec,
                    video_path=None, method=None, final_eval_slice=0,
                    num_inference_steps=None, num_samples=None, instability=None,
                    policy_model=None):
        instability = instability or {}
        all_step_recs = [
            (ci, st)
            for ci, chunk in enumerate(episode_rec.get('chunks', []))
            for st in chunk.get('steps', [])
        ]
        u_vals = [st['u_mean'] for _, st in all_step_recs]
        u_mean_ep = float(np.mean(u_vals)) if u_vals else None
        u_max_ep  = float(np.max(u_vals))  if u_vals else None

        if pnp_cfg.step_indices is not None:
            step_idx_str = _json.dumps(list(pnp_cfg.step_indices))
        else:
            step_idx_str = f'time_min:{pnp_cfg.time_min}' if pnp_cfg.time_min is not None else None

        dim_col_str = ', '.join(_DIM_COLS)
        dim_ph_str  = ', '.join(['?'] * len(_DIM_COLS))

        def _dim_vals(st):
            u_vec     = st.get('u_vec',     [None] * _ADIM)
            a_std_vec = st.get('a_std_vec', [None] * _ADIM)
            return [float(v) if v is not None else None for v in list(u_vec)[:_ADIM]] + \
                   [float(v) if v is not None else None for v in list(a_std_vec)[:_ADIM]]

        with self._con:
            self._con.execute('DELETE FROM pnp_euler_steps WHERE rollout_id = ?', (rollout_id,))
            self._con.execute(
                'INSERT OR REPLACE INTO rollouts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (rollout_id, suite, task_idx, task_desc, episode_idx,
                 self.init_state_hash(init_state),
                 int(success), n_steps, round(elapsed_s, 3),
                 int(pnp_cfg.enabled), pnp_cfg.num_iterations,
                 step_idx_str, pnp_cfg.mode,
                 u_mean_ep, u_max_ep, len(all_step_recs),
                 _time.strftime('%Y-%m-%dT%H:%M:%S'), video_path,
                 method, int(final_eval_slice), num_inference_steps, num_samples,
                 instability.get('action_delta_l2_mean'),
                 instability.get('action_delta_l2_max'),
                 instability.get('action_var_mean'),
                 instability.get('gripper_flip_count'),
                 instability.get('gripper_flip_rate'),
                 instability.get('chunk_disagreement_mean'),
                 policy_model))
            self._con.executemany(
                f'INSERT INTO pnp_euler_steps '
                f'(rollout_id, chunk_idx, euler_step, s, u_mean, u_max, a_std_mean, {dim_col_str}) '
                f'VALUES (?,?,?,?,?,?,?,{dim_ph_str})',
                [(rollout_id, ci, st['step'], st['s'],
                  st['u_mean'], st['u_max'], st['a_std_mean'],
                  *_dim_vals(st))
                 for ci, st in all_step_recs])
        self._con.commit()

    def update_instability(self, rollout_id, instability):
        """Backfill executed-action instability metrics on an existing row."""
        sql = (
            "UPDATE rollouts SET action_delta_l2_mean=?, action_delta_l2_max=?, "
            "action_var_mean=?, gripper_flip_count=?, gripper_flip_rate=?, "
            "chunk_disagreement_mean=? WHERE rollout_id=?"
        )
        with self._con:
            self._con.execute(
                sql,
                (instability.get('action_delta_l2_mean'),
                 instability.get('action_delta_l2_max'),
                 instability.get('action_var_mean'),
                 instability.get('gripper_flip_count'),
                 instability.get('gripper_flip_rate'),
                 instability.get('chunk_disagreement_mean'),
                 rollout_id))
        self._con.commit()

    def vanilla_backfill_targets(self):
        return self.query(
            "SELECT rollout_id, suite, task_idx, episode_idx, init_state_hash "
            "FROM rollouts WHERE method='vanilla' AND final_eval_slice=1 "
            "AND action_delta_l2_mean IS NULL"
        )

    def existing_keys(self, final_eval_slice=1, policy_model=None):
        rows = self.query(
            'SELECT suite, task_idx, episode_idx, init_state_hash, method, pnp_step_indices, policy_model '
            'FROM rollouts WHERE final_eval_slice = ?',
            (int(final_eval_slice),),
        )
        keys = set()
        for r in rows:
            if policy_model is not None and r.get('policy_model') != policy_model:
                continue
            keys.add((r['suite'], r['task_idx'], r['episode_idx'], r['init_state_hash'],
                      r.get('method'), r.get('pnp_step_indices')))
        return keys

    def sync_to_path(self, dst_path):
        import shutil
        tmp = dst_path + '.tmp'
        dst = sqlite3.connect(tmp)
        self._con.backup(dst)
        dst.close()
        shutil.move(tmp, dst_path)
        print(f'Synced DB -> {dst_path}')

    def verify_disk(self, dst_path):
        mem = self._con.execute('SELECT COUNT(*) FROM rollouts').fetchone()[0]
        disk = sqlite3.connect(dst_path).execute('SELECT COUNT(*) FROM rollouts').fetchone()[0]
        print(f'verify: mem={mem} disk={disk}')

    def query(self, sql, params=()):
        cur = self._con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary(self):
        rows = self.query("""
            SELECT suite, task_idx, method, policy_model,
                   COUNT(*) AS n_ep,
                   ROUND(AVG(success)*100, 1) AS sr_pct,
                   ROUND(AVG(u_mean_episode), 5) AS u_mean_all
            FROM rollouts
            GROUP BY suite, task_idx, method, policy_model
            ORDER BY suite, task_idx, method
        """)
        print(f"{'suite':<18} {'task':>4} {'method':<22} {'model':<8} {'n':>4} {'sr%':>6} {'u_all':>10}")
        print('-' * 82)
        for r in rows:
            print(f"{r['suite']:<18} {r['task_idx']:>4} {str(r.get('method','')):<22} "
                  f"{str(r.get('policy_model') or ''):<8} {r['n_ep']:>4} {r['sr_pct']:>6} {str(r['u_mean_all']):>10}")
        return rows
'''


def build_test_notebook():
    pi05_pnp = Path('/tmp/cell_16.py').read_text()
    rollout_helpers = Path('/tmp/cell_19.py').read_text()

    cells = [
        md("""# SmolVLA Replication + π0.5 Vanilla Instability Backfill

**Prerequisites:** Run [`test_pi05_jennifer.ipynb`](test_pi05_jennifer.ipynb) and [`test_pi05_jennifer_v2.ipynb`](test_pi05_jennifer_v2.ipynb) first so Drive has the package snapshot and π0.5 v2 DB.

**Every session:** Sections 1 → 2 → 3b → 4 → 5 → 6

**Part A (Section 6b):** Re-run π0.5 `vanilla` on the fixed v2 slice and **UPDATE** `action_delta_l2_mean` (and related instability cols) in the existing [`results_v2/rollouts_v2.db`](results_v2/rollouts_v2.db). `SKIP_COMPLETED=True` skips rows that already have instability logged.

**Part B (Section 6c):** Evaluate **SmolVLA** (`HuggingFaceVLA/smolvla_libero`, 450M) on the **same 80 episodes** with `vanilla` + `pnp_uncertainty_only` → [`results_smolvla/rollouts_smolvla.db`](results_smolvla/rollouts_smolvla.db).

**Analysis:** [`pnp_smolvla_jennifer_analysis.ipynb`](pnp_smolvla_jennifer_analysis.ipynb)

> MuJoCo **3.3.2** is recommended for SmolVLA LIBERO eval ([lerobot#1369](https://github.com/huggingface/lerobot/issues/1369)).
"""),
        md("---\n## Section 1: Drive mount\n\nRun every session."),
        code("""from google.colab import drive
import os
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive'
SHARED = f'{DRIVE}/cs159-sp26'
CACHE_DIR = f'{DRIVE}/smolvla_colab_cache'
HF_HOME = f'{CACHE_DIR}/hf_models'

PI05_RESULTS_DIR = f'{SHARED}/results_v2'
PI05_V2_DB = f'{PI05_RESULTS_DIR}/rollouts_v2.db'
PI05_VIDEO_DIR = f'{PI05_RESULTS_DIR}/videos_v2'

SMOLVLA_RESULTS_DIR = f'{SHARED}/results_smolvla'
SMOLVLA_DB = f'{SMOLVLA_RESULTS_DIR}/rollouts_smolvla.db'
SMOLVLA_VIDEO_DIR = f'{SMOLVLA_RESULTS_DIR}/videos_smolvla'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(PI05_RESULTS_DIR, exist_ok=True)
os.makedirs(SMOLVLA_RESULTS_DIR, exist_ok=True)
os.makedirs(SMOLVLA_VIDEO_DIR, exist_ok=True)

os.environ['HF_HOME'] = HF_HOME
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'

SNAPSHOT = f'{CACHE_DIR}/site_packages.tar.gz'
print(f'Shared:          {SHARED}')
print(f'π0.5 v2 DB:      {PI05_V2_DB}  (exists={os.path.isfile(PI05_V2_DB)})')
print(f'SmolVLA DB:      {SMOLVLA_DB}')
print(f'Snapshot:        {"FOUND" if os.path.exists(SNAPSHOT) else "NOT FOUND — run test_pi05_jennifer Section 3 first"}')
"""),
        md("---\n## Section 2: GPU check"),
        code("""import subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(out)
"""),
        md("---\n## Section 3b: Restore packages from Drive snapshot"),
        code("""import subprocess, sys, os, time, shutil, importlib

if not os.path.exists(SNAPSHOT):
    raise FileNotFoundError('No snapshot — run test_pi05_jennifer Section 3 first.')

LOCAL_SNAPSHOT = '/content/site_packages_restore.tar.gz'
print(f'Copying snapshot ({os.path.getsize(SNAPSHOT)/1e6:.0f} MB)...')
shutil.copy(SNAPSHOT, LOCAL_SNAPSHOT)
subprocess.run(['tar', '-xzf', LOCAL_SNAPSHOT, '-C', '/'], check=True)
os.remove(LOCAL_SNAPSHOT)
importlib.invalidate_caches()

# torch/diffusers compat
try:
    import torch
    from torch.ao.quantization import CUSTOM_KEY  # noqa
    print(f'torch {torch.__version__} OK')
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'torch', 'torchvision', 'torchaudio'])
    importlib.invalidate_caches()
    import torch
    print(f'torch upgraded to {torch.__version__}')

for pkg in ['mujoco', 'libero', 'lerobot']:
    importlib.import_module(pkg)
    print(f'  {pkg} OK')

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print('SmolVLAPolicy import: OK')
except ImportError:
    print('SmolVLAPolicy missing — installing lerobot[smolvla]...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'lerobot[smolvla]'])
    importlib.invalidate_caches()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print('SmolVLAPolicy import: OK after install')

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print('PI05Policy import: OK')
"""),
        md("---\n## Section 4: LIBERO helpers (policy-agnostic)"),
        code("""import os, time, json, math
import torch, numpy as np
from huggingface_hub import login
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.factory import make_pre_post_processors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hf_token = os.getenv('HF_TOKEN')
login(token=hf_token) if hf_token else login()

MAX_STEPS_MAP = {
    'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300,
    'libero_10': 520, 'libero_90': 400,
}
CAMERAS = ['agentview', 'robot0_eye_in_hand']
IMG_SIZE = 360
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10


def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_policy(obs_dict, task_desc, device):
    agentview = np.ascontiguousarray(obs_dict['agentview_image'][::-1, ::-1])
    wrist     = np.ascontiguousarray(obs_dict['robot0_eye_in_hand_image'][::-1, ::-1])
    img_agent = torch.from_numpy(agentview / 255.0).permute(2, 0, 1).float()
    img_wrist = torch.from_numpy(wrist / 255.0).permute(2, 0, 1).float()
    state = np.concatenate([
        obs_dict['robot0_eef_pos'],
        _quat2axisangle(obs_dict['robot0_eef_quat']),
        obs_dict['robot0_gripper_qpos'],
    ])
    return {
        'observation.images.image':  img_agent,
        'observation.images.image2': img_wrist,
        'observation.state': torch.from_numpy(state).float(),
        'task': task_desc,
    }

benchmark_dict = benchmark.get_benchmark_dict()
print('Section 4 ready.')
"""),
        md("---\n## Section 5: P&P sampler, RolloutDB, rollout helpers"),
        code(pi05_pnp.replace(
            'print("PnP defined: PnPConfig, PnPRecorder, PNP_CONFIG, PNP_RECORDER, _sample_actions_pnp")',
            '_sample_actions_pnp_pi05 = _sample_actions_pnp\n' + SMOLVLA_PNP + '\nprint("PnP defined (pi05 + smolvla variants)")'
        )),
        code("""import types

def _infer_action_dim(policy, default=7):
    feats = getattr(policy.config, 'output_features', {})
    if 'action' in feats:
        return int(feats['action'].shape[0])
    return default


def apply_pnp_patch(policy, flavor='pi05'):
    \"\"\"Monkey-patch policy.model.sample_actions with P&P wrapper.\"\"\"
    model = policy.model
    if not hasattr(model, '_orig_sample_actions'):
        model._orig_sample_actions = model.sample_actions
    else:
        model.sample_actions = model._orig_sample_actions

    if flavor == 'pi05':
        fn = _sample_actions_pnp_pi05
        if getattr(model.config, 'compile_model', False):
            compile_mode = _pnp_compile_mode(model.config)
            def _unwrap(fn):
                while hasattr(fn, '_orig_mod'):
                    fn = fn._orig_mod
                return fn
            model._orig_sample_actions = torch.compile(_unwrap(model._orig_sample_actions), mode=compile_mode)
            model.sample_actions = torch.compile(types.MethodType(fn, model), mode=compile_mode)
        else:
            model.sample_actions = types.MethodType(fn, model)
    elif flavor == 'smolvla':
        model.sample_actions = types.MethodType(_sample_actions_pnp_smolvla, model)
    else:
        raise ValueError(flavor)

    PNP_CONFIG.action_dim = _infer_action_dim(policy)
    PNP_CONFIG.enabled = False
    print(f'Patched {flavor} sample_actions (action_dim={PNP_CONFIG.action_dim})')


def load_pi05():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained('lerobot/pi05_libero_finetuned').to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, 'lerobot/pi05_libero_finetuned',
        preprocessor_overrides={'device_processor': {'device': str(device)}},
    )
    apply_pnp_patch(policy, 'pi05')
    return policy, preprocess, postprocess


def load_smolvla():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model_id = 'HuggingFaceVLA/smolvla_libero'
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, model_id,
        preprocessor_overrides={'device_processor': {'device': str(device)}},
    )
    apply_pnp_patch(policy, 'smolvla')
    return policy, preprocess, postprocess
"""),
        code("""import os, sqlite3, hashlib, json as _json, time as _time
""" + ROLLoutDB),
        code(rollout_helpers.replace(
            "rollout_id = RolloutDB.make_rollout_id(\n            suite, task_idx or 0, episode_idx or 0, init_state, PNP_CONFIG,\n            method=method, num_inference_steps=num_inference_steps, num_samples=num_samples)",
            "rollout_id = RolloutDB.make_rollout_id(\n            suite, task_idx or 0, episode_idx or 0, init_state, PNP_CONFIG,\n            method=method, num_inference_steps=num_inference_steps, num_samples=num_samples,\n            policy_model=globals().get('CURRENT_POLICY_MODEL'))"
        ).replace(
            "            instability=instability,\n        )",
            "            instability=instability,\n            policy_model=globals().get('CURRENT_POLICY_MODEL'),\n        )",
        ) + """

CURRENT_POLICY_MODEL = None

def run_episode_backfill(env, init_state, policy, task_desc, max_steps, device,
                         episode_idx=None):
    # Run vanilla episode without DB write; return instability dict only.
    env.reset(); policy.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    _seed = _episode_seed(init_state, episode_idx)
    torch.manual_seed(_seed); torch.cuda.manual_seed(_seed)
    PNP_CONFIG.enabled = False
    PNP_RECORDER.new_episode({})
    executed_actions, chunk_boundary_actions = [], []
    last_n_chunks = 0
    for step in range(max_steps):
        raw_obs = obs_to_policy(obs, task_desc, device)
        batch = preprocess(raw_obs)
        _pnp_mark_cuda_graph_step()
        with torch.no_grad():
            action = policy.select_action(batch)
        action_np = _to_numpy_action(action)
        executed_actions.append(action_np.copy())
        n_chunks = len(PNP_RECORDER._cur['chunks']) if PNP_RECORDER._cur else 0
        if n_chunks > last_n_chunks:
            chunk_boundary_actions.append(action_np.copy())
            last_n_chunks = n_chunks
        obs, _, done, _ = env.step(action_np)
        if env.check_success() or done:
            break
    return _compute_action_instability(executed_actions, chunk_boundary_actions)
"""),
        md("""---
## Section 6: Controlled experiment

**Flags:**
- `RUN_PI05_BACKFILL` — Part A: fill π0.5 vanilla instability in existing v2 DB
- `RUN_SMOLVLA_EVAL` — Part B: SmolVLA vanilla + detector on same slice
- `SKIP_COMPLETED` — skip episode×method combos already present (SmolVLA DB) or rows with non-null instability (backfill)
"""),
        code("""import sqlite3

FINAL_STEP_CONFIGS = [(2, 3), (3, 4), (4, 5)]
FINAL_EPISODE_IDXS = list(range(10))
PNP_K = 3
BASELINE_STEPS = 10

RUN_PI05_BACKFILL = True
RUN_SMOLVLA_EVAL = True
SKIP_COMPLETED = True

SMOLVLA_METHODS = ['vanilla', 'pnp_uncertainty_only']
RUN_SMOLVLA_METHODS = ['vanilla', 'pnp_uncertainty_only']

# Hardcoded fallback (neurips appendix v2 slice)
FALLBACK_TASKS = [
    ('libero_spatial', 5), ('libero_spatial', 8),
    ('libero_goal', 0), ('libero_goal', 1), ('libero_goal', 2),
    ('libero_goal', 3), ('libero_goal', 5), ('libero_goal', 6),
]

FINAL_EPISODES = []
EPISODE_KEYS = set()
_db_keys = set()

if os.path.isfile(PI05_V2_DB):
    _con = sqlite3.connect(PI05_V2_DB)
    rows = _con.execute(
        "SELECT DISTINCT suite, task_idx, episode_idx, init_state_hash "
        "FROM rollouts WHERE final_eval_slice=1 AND method='vanilla'"
    ).fetchall()
    _con.close()
    _db_keys = {(r[0], r[1], r[2], r[3]) for r in rows}
    task_set = sorted({(r[0], r[1]) for r in rows}) if rows else FALLBACK_TASKS
    print(f'Loaded {len(rows)} vanilla keys from π0.5 v2 DB -> {len(task_set)} tasks')
else:
    task_set = FALLBACK_TASKS
    print('π0.5 v2 DB not found — using fallback task list')

for suite, task_idx in task_set:
    task_suite = benchmark_dict[suite]()
    task = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    bddl_path = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    max_steps = MAX_STEPS_MAP.get(suite, 300)
    for ep_idx in FINAL_EPISODE_IDXS:
        if ep_idx >= len(init_states):
            continue
        init_state = init_states[ep_idx]
        ish = RolloutDB.init_state_hash(init_state)
        key = (suite, task_idx, ep_idx, ish)
        if _db_keys and key not in _db_keys:
            continue
        EPISODE_KEYS.add(key)
        FINAL_EPISODES.append(dict(
            suite=suite, task_idx=task_idx, task_desc=task.language,
            ep_idx=ep_idx, init_state=init_state, bddl_path=bddl_path,
            max_steps=max_steps, init_state_hash=ish,
        ))

print(f'FINAL_EPISODES: {len(FINAL_EPISODES)}')
"""),
        code("""# === Part A: π0.5 vanilla instability backfill ===
from itertools import groupby
from tqdm.notebook import tqdm

if RUN_PI05_BACKFILL:
    if not os.path.isfile(PI05_V2_DB):
        print('Skip backfill — π0.5 v2 DB not found')
    else:
        pi05_db = RolloutDB(PI05_V2_DB)
        targets = pi05_db.vanilla_backfill_targets()
        print(f'Backfill targets (NULL action_delta_l2_mean): {len(targets)}')
        if not targets:
            print('Nothing to backfill — all vanilla rows have instability metrics.')
        else:
            target_map = {(r['suite'], r['task_idx'], r['episode_idx'], r['init_state_hash']): r['rollout_id']
                          for r in targets}
            policy, preprocess, postprocess = load_pi05()
            CURRENT_POLICY_MODEL = 'pi05'
            PNP_CONFIG.enabled = False
            sorted_eps = sorted(FINAL_EPISODES, key=lambda x: (x['suite'], x['task_idx']))
            n_updated = 0
            for (suite, task_idx), group_iter in groupby(sorted_eps, key=lambda x: (x['suite'], x['task_idx'])):
                episodes = list(group_iter)
                env = OffScreenRenderEnv(
                    bddl_file_name=episodes[0]['bddl_path'], camera_names=CAMERAS,
                    camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                    has_offscreen_renderer=True, use_camera_obs=True,
                    has_renderer=False, reward_shaping=False,
                )
                try:
                    for ep in tqdm(episodes, desc=f'backfill {suite} T{task_idx}', leave=False):
                        key = (ep['suite'], ep['task_idx'], ep['ep_idx'], ep['init_state_hash'])
                        if key not in target_map:
                            continue
                        rid = target_map[key]
                        inst = run_episode_backfill(
                            env, ep['init_state'], policy, ep['task_desc'], ep['max_steps'],
                            device, episode_idx=ep['ep_idx'])
                        pi05_db.update_instability(rid, inst)
                        n_updated += 1
                finally:
                    env.close()
            pi05_db.sync_to_path(PI05_V2_DB)
            print(f'Updated instability on {n_updated} vanilla rows')
            pi05_db.summary()
else:
    print('RUN_PI05_BACKFILL=False — skipped')
"""),
        code("""# === Part B: SmolVLA eval (vanilla + pnp_uncertainty_only) ===
import json as _json
from itertools import groupby
from tqdm.notebook import tqdm

if RUN_SMOLVLA_EVAL:
    policy, preprocess, postprocess = load_smolvla()
    CURRENT_POLICY_MODEL = 'smolvla'
    VIDEO_DIR = SMOLVLA_VIDEO_DIR
    DB = RolloutDB(SMOLVLA_DB)

    if not FINAL_EPISODES:
        print('No FINAL_EPISODES — run slice cell first')
    else:
        sorted_eps = sorted(FINAL_EPISODES, key=lambda x: (x['suite'], x['task_idx']))
        for method in RUN_SMOLVLA_METHODS:
            print(f'\\n{"#"*60}\\nSmolVLA method: {method}\\n{"#"*60}')
            configs = list(FINAL_STEP_CONFIGS) if method == 'pnp_uncertainty_only' else [None]
            for step_indices in configs:
                PNP_CONFIG.enabled = (method == 'pnp_uncertainty_only')
                PNP_CONFIG.mode = 'uncertainty'
                PNP_CONFIG.step_indices = step_indices
                PNP_CONFIG.num_iterations = PNP_K
                PNP_CONFIG.time_min = None
                PNP_RECORDER.reset()
                step_key = _json.dumps(list(step_indices)) if step_indices else None
                completed = DB.existing_keys(policy_model='smolvla') if SKIP_COMPLETED else set()

                for (suite, task_idx), group_iter in groupby(sorted_eps, key=lambda x: (x['suite'], x['task_idx'])):
                    episodes = list(group_iter)
                    env = OffScreenRenderEnv(
                        bddl_file_name=episodes[0]['bddl_path'], camera_names=CAMERAS,
                        camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                        has_offscreen_renderer=True, use_camera_obs=True,
                        has_renderer=False, reward_shaping=False,
                    )
                    try:
                        for ep in tqdm(episodes, desc=f'  {method} {suite} T{task_idx}', leave=False):
                            ep_key = (ep['suite'], ep['task_idx'], ep['ep_idx'], ep['init_state_hash'],
                                      method, step_key)
                            if ep_key in completed:
                                continue
                            run_episode_pnp(
                                env, ep['init_state'], policy, ep['task_desc'], ep['max_steps'], device,
                                suite=ep['suite'], task_idx=ep['task_idx'], episode_idx=ep['ep_idx'],
                                db=DB, save_video='failures_only', method=method, final_eval_slice=1,
                                num_inference_steps=BASELINE_STEPS,
                            )
                    finally:
                        env.close()
        DB.sync_to_path(SMOLVLA_DB)
        DB.summary()
else:
    print('RUN_SMOLVLA_EVAL=False — skipped')
"""),
        code("""# Optional: SmolVLA equivalence smoke test (P&P off)
if RUN_SMOLVLA_EVAL and FINAL_EPISODES:
    policy, preprocess, postprocess = load_smolvla()
    ep = FINAL_EPISODES[0]
    env = OffScreenRenderEnv(
        bddl_file_name=ep['bddl_path'], camera_names=CAMERAS,
        camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
        has_offscreen_renderer=True, use_camera_obs=True, has_renderer=False, reward_shaping=False,
    )
    try:
        env.reset(); policy.reset()
        obs = env.set_init_state(ep['init_state'])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        batch = preprocess(obs_to_policy(obs, ep['task_desc'], device))
        PNP_CONFIG.enabled = False
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.no_grad():
            a1 = policy.predict_action_chunk(batch, noise=None).clone()
        PNP_CONFIG.enabled = True
        PNP_CONFIG.step_indices = ()
        PNP_CONFIG.mode = 'uncertainty'
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.no_grad():
            a2 = policy.predict_action_chunk(batch, noise=None).clone()
        print(f'SmolVLA loop equiv max|Δ|={ (a1-a2).abs().max().item():.3e}')
    finally:
        env.close()
"""),
    ]
    return nb(cells)


def build_analysis_notebook():
    cells = [
        md("""# SmolVLA Replication Analysis

Reads:
- [`results_smolvla/rollouts_smolvla.db`](results_smolvla/rollouts_smolvla.db) from [`test_smolvla_jennifer.ipynb`](test_smolvla_jennifer.ipynb)
- [`results_v2/rollouts_v2.db`](results_v2/rollouts_v2.db) (π0.5, read-only for cross-model comparison)

**Goal:** Test whether P&P detector signal transfers to the 450M-parameter SmolVLA policy on the same fixed 80-episode slice.

Sections: **0** design · **A** SmolVLA SR + instability · **C** detector metrics · **E** cross-model table · **D** findings
"""),
        code("""from google.colab import drive
drive.mount('/content/drive')
"""),
        code("""import os, sqlite3, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from IPython.display import display

SHARED = '/content/drive/MyDrive/cs159-sp26'
PI05_RESULTS = f'{SHARED}/results_v2'
PI05_DB = f'{PI05_RESULTS}/rollouts_v2.db'
SMOLVLA_RESULTS = f'{SHARED}/results_smolvla'
SMOLVLA_DB = f'{SMOLVLA_RESULTS}/rollouts_smolvla.db'
FIGURES_DIR = f'{SMOLVLA_RESULTS}/figures_smolvla'
FINAL_FIGURES_DIR = f'{FIGURES_DIR}/final'

for path, fallback in [(SMOLVLA_DB, 'results_smolvla/rollouts_smolvla.db'),
                       (PI05_DB, 'results_v2/rollouts_v2.db')]:
    if not os.path.isfile(path):
        local = Path(fallback)
        if local.is_file():
            if 'smolvla' in fallback:
                SMOLVLA_RESULTS = str(local.parent)
                SMOLVLA_DB = str(local)
            else:
                PI05_RESULTS = str(local.parent)
                PI05_DB = str(local)

os.makedirs(FINAL_FIGURES_DIR, exist_ok=True)

def save_fig(name, final=True):
    d = FINAL_FIGURES_DIR if final else FIGURES_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'final_smolvla_{name}.png' if final else f'smolvla_{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Wrote {path}')

FINAL_STEP_CONFIGS = [(2, 3), (3, 4), (4, 5)]
EPISODE_KEYS = ['suite', 'task_idx', 'episode_idx', 'init_state_hash']
INSTABILITY_COLS = ['action_delta_l2_mean', 'action_delta_l2_max', 'action_var_mean',
                    'gripper_flip_count', 'gripper_flip_rate', 'chunk_disagreement_mean']

con_smol = sqlite3.connect(SMOLVLA_DB)
smol_rollouts = pd.read_sql('SELECT * FROM rollouts', con_smol)
con_pi = sqlite3.connect(PI05_DB)
pi05_rollouts = pd.read_sql('SELECT * FROM rollouts', con_pi)

smol_df = smol_rollouts[smol_rollouts['final_eval_slice'] == 1].copy()
pi05_df = pi05_rollouts[pi05_rollouts['final_eval_slice'] == 1].copy()
print(f'SmolVLA slice rows: {len(smol_df)}')
print(f'π0.5 slice rows:    {len(pi05_df)}')
"""),
        md("""---
## Section 0: Experiment design

Same **80 episodes** as π0.5 v2 (`libero_goal` + `libero_spatial`, top-8 failure-prone tasks).

| Method | SmolVLA run | Purpose |
|--------|-------------|---------|
| `vanilla` | 10 Euler steps, P&P off | Baseline SR + instability |
| `pnp_uncertainty_only` | P&P at steps (2,3), (3,4), (4,5), K=3 | Detector signal |

**Hypothesis:** If P&P uncertainty is architecture-agnostic, SmolVLA should show similar U↔failure and U↔jerk correlations as π0.5 on this slice.
"""),
        code("""def describe_slice(df, label):
    if df.empty:
        print(f'{label}: no rows')
        return
    tasks = df.groupby(['suite', 'task_idx']).size()
    methods = df['method'].value_counts().to_dict()
    print(f'=== {label} ===')
    print(f'  Episodes (unique keys): {df[EPISODE_KEYS].drop_duplicates().shape[0]}')
    print(f'  Methods: {methods}')
    if 'vanilla' in methods:
        v = df[df['method'] == 'vanilla']
        null_inst = v['action_delta_l2_mean'].isna().sum()
        if null_inst:
            print(f'  WARNING: {null_inst} vanilla rows missing action_delta_l2_mean')
        else:
            print(f'  vanilla SR: {v["success"].mean():.1%}')

describe_slice(smol_df, 'SmolVLA')
describe_slice(pi05_df, 'π0.5')

vanilla_null_pi = pi05_df.query("method=='vanilla'")['action_delta_l2_mean'].isna().sum()
if vanilla_null_pi:
    print(f'\\nπ0.5 backfill incomplete: {vanilla_null_pi} vanilla rows still NULL — run test_smolvla Section 6b')
else:
    print('\\nπ0.5 vanilla instability backfill: OK')
"""),
        md("---\n## Section A: SmolVLA vanilla SR + instability"),
        code("""def collapse_step_configs(df, success_agg='max'):
    agg = {'success': success_agg}
    for col in ['u_mean_episode', 'n_steps'] + INSTABILITY_COLS:
        if col in df.columns:
            agg[col] = 'mean'
    return df.groupby(['method'] + EPISODE_KEYS, as_index=False).agg(agg)

smol_episode_df = collapse_step_configs(smol_df)
method_summary = (
    smol_episode_df.groupby('method')
    .agg(n_episodes=('success', 'count'), success_rate=('success', 'mean'),
         action_delta_l2_mean=('action_delta_l2_mean', 'mean'),
         u_mean_episode=('u_mean_episode', 'mean'))
    .reset_index()
)
print('=== SmolVLA method summary ===')
display(method_summary.round(4))
method_summary.to_csv(os.path.join(SMOLVLA_RESULTS, 'final_smolvla_method_comparison.csv'), index=False)

if not method_summary.empty:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(method_summary['method'], method_summary['success_rate'])
    ax.set_ylabel('Success rate')
    ax.set_title('SmolVLA success by method (v2 slice)')
    ax.set_ylim(0, 1)
    save_fig('success_by_method')
    plt.show()
"""),
        md("---\n## Section C: SmolVLA detector analysis"),
        code("""try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-learn'])
    from sklearn.metrics import roc_auc_score, average_precision_score


def threshold_curve(df, u_col='u_mean_episode', label_col='success'):
    y_true = 1 - df[label_col].astype(int)
    scores = df[u_col].astype(float)
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best = {'f1': -1}
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 > best.get('f1', -1):
            best = {'threshold': t, 'precision': prec, 'recall': rec, 'f1': f1}
    return best


def compute_detector_metrics(df, model_name):
    det = df[df['method'] == 'pnp_uncertainty_only'].copy()
    if det.empty:
        return pd.DataFrame()
    rows = []
    for step_cfg in sorted(det['pnp_step_indices'].dropna().unique()):
        sub = det[det['pnp_step_indices'] == step_cfg]
        if sub['success'].nunique() < 2:
            continue
        u = sub['u_mean_episode'].astype(float)
        y_fail = 1 - sub['success'].astype(int)
        r_p, _ = stats.pearsonr(u, sub['success'])
        r_s, _ = stats.spearmanr(u, sub['success'])
        row = {
            'model': model_name,
            'step_config': step_cfg,
            'n': len(sub),
            'pearson_r': r_p,
            'spearman_u_vs_success': r_s,
            'roc_auc': roc_auc_score(y_fail, u),
            'pr_auc': average_precision_score(y_fail, u),
        }
        if 'action_delta_l2_mean' in sub.columns and sub['action_delta_l2_mean'].notna().any():
            row['spearman_u_vs_action_delta_l2_mean'] = stats.spearmanr(
                u, sub['action_delta_l2_mean'].astype(float))[0]
        best = threshold_curve(sub)
        row.update({f'best_{k}': v for k, v in best.items()})
        rows.append(row)
    return pd.DataFrame(rows)

smol_det = compute_detector_metrics(smol_df, 'smolvla')
pi05_det = compute_detector_metrics(pi05_df, 'pi0.5')

if not smol_det.empty:
    print('=== SmolVLA detector metrics ===')
    display(smol_det.round(4))
    smol_det.to_csv(os.path.join(SMOLVLA_RESULTS, 'final_smolvla_detector_metrics.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sub = collapse_step_configs(smol_df[smol_df['method'] == 'pnp_uncertainty_only'])
    ax = axes[0]
    for s in [0, 1]:
        m = sub['success'] == s
        ax.scatter(sub.loc[m, 'u_mean_episode'], sub.loc[m, 'success'], alpha=0.5,
                   label='success' if s else 'fail')
    ax.set_xlabel('u_mean_episode'); ax.set_ylabel('success'); ax.set_title('Uncertainty vs outcome'); ax.legend()
    save_fig('uncertainty_success_failure')
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(sub['u_mean_episode'], sub['action_delta_l2_mean'], c=sub['success'], cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('u_mean_episode'); ax.set_ylabel('action_delta_l2_mean'); ax.set_title('Uncertainty vs jerk')
    plt.colorbar(sc, ax=ax, label='success')
    save_fig('uncertainty_vs_instability')
    plt.show()

    if len(smol_det) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(smol_det))
        ax.bar(x, smol_det['roc_auc'], tick_label=smol_det['step_config'])
        ax.set_ylabel('ROC-AUC (predict failure)')
        ax.set_title('SmolVLA detector by step config')
        save_fig('step_config_comparison')
        plt.show()
"""),
        md("---\n## Section E: Cross-model transfer table"),
        code("""def aggregate_detector(det_df, model_name, pi05_vanilla_sr=None, smol_vanilla_sr=None):
    if det_df.empty:
        return {'model': model_name, 'n_episodes': 0}
    vanilla_sr = None
    if model_name == 'pi0.5':
        v = pi05_df[pi05_df['method'] == 'vanilla']
        vanilla_sr = v['success'].mean() if len(v) else None
    else:
        v = smol_df[smol_df['method'] == 'vanilla']
        vanilla_sr = v['success'].mean() if len(v) else None
    return {
        'model': model_name,
        'n_episodes': int(det_df['n'].max()) if 'n' in det_df else 0,
        'vanilla_sr': vanilla_sr,
        'roc_auc_mean': det_df['roc_auc'].mean(),
        'pr_auc_mean': det_df['pr_auc'].mean(),
        'spearman_u_vs_success': det_df['spearman_u_vs_success'].mean(),
        'spearman_u_vs_action_delta_l2_mean': det_df.get(
            'spearman_u_vs_action_delta_l2_mean', pd.Series(dtype=float)).mean(),
        'best_f1_mean': det_df.get('best_f1', pd.Series(dtype=float)).mean(),
    }

cross_rows = []
if not pi05_det.empty:
    cross_rows.append(aggregate_detector(pi05_det, 'pi0.5'))
if not smol_det.empty:
    cross_rows.append(aggregate_detector(smol_det, 'smolvla'))

cross_df = pd.DataFrame(cross_rows)
print('=== Cross-model detector transfer (neurips placeholder table) ===')
display(cross_df.round(4))
cross_df.to_csv(os.path.join(SMOLVLA_RESULTS, 'final_smolvla_cross_model.csv'), index=False)

if len(cross_df) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    metrics = ['roc_auc_mean', 'spearman_u_vs_action_delta_l2_mean']
    titles = ['ROC-AUC (mean over step configs)', 'Spearman(U, action_delta_l2_mean)']
    for ax, m, t in zip(axes, metrics, titles):
        if m in cross_df.columns:
            ax.bar(cross_df['model'], cross_df[m])
            ax.set_title(t)
            ax.set_ylim(-1, 1 if 'spearman' in m else None)
    save_fig('cross_model_detector')
    plt.show()
"""),
        md("---\n## Section D: Failure taxonomy + findings"),
        code("""def classify_failure_modes(episode_df, u_col='u_mean_episode', inst_col='action_delta_l2_mean'):
    det = episode_df[episode_df['method'] == 'pnp_uncertainty_only'].copy()
    if det.empty:
        return pd.DataFrame()
    det = collapse_step_configs(det)
    u_med = det[u_col].median()
    inst_med = det[inst_col].median() if inst_col in det and det[inst_col].notna().any() else 0
    labels = []
    for _, r in det.iterrows():
        hi_u = r[u_col] >= u_med
        hi_i = r.get(inst_col, 0) >= inst_med if pd.notna(r.get(inst_col)) else False
        if not r['success']:
            if hi_u and hi_i:
                tax = 'high_U_high_instability_failure'
            elif not hi_u and not hi_i:
                tax = 'low_U_low_instability_failure'
            elif hi_u:
                tax = 'high_U_low_instability_failure'
            else:
                tax = 'low_U_high_instability_failure'
        else:
            tax = 'high_U_success' if hi_u else 'low_U_success'
        labels.append({**{k: r[k] for k in EPISODE_KEYS}, 'taxonomy': tax, 'success': r['success'],
                       'u_mean_episode': r[u_col]})
    return pd.DataFrame(labels)

taxonomy_df = classify_failure_modes(smol_df)
if not taxonomy_df.empty:
    print('=== SmolVLA failure taxonomy ===')
    display(taxonomy_df['taxonomy'].value_counts())
    taxonomy_df.to_csv(os.path.join(SMOLVLA_RESULTS, 'final_smolvla_failure_taxonomy.csv'), index=False)


def summarize_smolvla_findings(cross_df, smol_det, method_summary):
    print('=' * 72)
    print('SMOLVLA REPLICATION FINDINGS')
    print('=' * 72)
    if not method_summary.empty and 'vanilla' in method_summary['method'].values:
        sr = method_summary.loc[method_summary['method'] == 'vanilla', 'success_rate'].iloc[0]
        print(f'1. SmolVLA vanilla SR on v2 slice: {sr:.1%}')
    if not smol_det.empty:
        print(f'2. SmolVLA detector: mean ROC-AUC={smol_det["roc_auc"].mean():.3f}, '
              f'mean Spearman(U,success)={smol_det["spearman_u_vs_success"].mean():.3f}')
        if 'spearman_u_vs_action_delta_l2_mean' in smol_det:
            print(f'   mean Spearman(U,jerk)={smol_det["spearman_u_vs_action_delta_l2_mean"].mean():.3f}')
    if len(cross_df) >= 2:
        pi = cross_df[cross_df['model'] == 'pi0.5'].iloc[0]
        sm = cross_df[cross_df['model'] == 'smolvla'].iloc[0]
        print('3. Transfer vs π0.5:')
        print(f'   π0.5     ROC-AUC={pi.get("roc_auc_mean", float("nan")):.3f}  vanilla SR={pi.get("vanilla_sr", float("nan")):.1%}')
        print(f'   SmolVLA  ROC-AUC={sm.get("roc_auc_mean", float("nan")):.3f}  vanilla SR={sm.get("vanilla_sr", float("nan")):.1%}')
        delta_auc = sm.get('roc_auc_mean', 0) - pi.get('roc_auc_mean', 0)
        print(f'   Δ ROC-AUC (SmolVLA - π0.5) = {delta_auc:+.3f}')
    print('\\n4. CAVEATS: same slice, different absolute SR; detector measures instability not correctness.')
    print('=' * 72)

summarize_smolvla_findings(cross_df, smol_det, method_summary)
"""),
    ]
    return nb(cells)


def main():
    # Ensure temp cells exist for pi05 P&P source
    import subprocess
    subprocess.run([
        'python3', '-c',
        "import json; from pathlib import Path;"
        "nb=json.loads(Path('test_pi05_jennifer_v2.ipynb').read_text());"
        "Path('/tmp/cell_16.py').write_text(''.join(nb['cells'][16]['source']));"
        "Path('/tmp/cell_19.py').write_text(''.join(nb['cells'][19]['source']));"
    ], cwd=ROOT, check=True)

    test_path = ROOT / 'test_smolvla_jennifer.ipynb'
    analysis_path = ROOT / 'pnp_smolvla_jennifer_analysis.ipynb'
    test_path.write_text(json.dumps(build_test_notebook(), indent=1))
    analysis_path.write_text(json.dumps(build_analysis_notebook(), indent=1))
    print(f'Wrote {test_path}')
    print(f'Wrote {analysis_path}')


if __name__ == '__main__':
    main()
