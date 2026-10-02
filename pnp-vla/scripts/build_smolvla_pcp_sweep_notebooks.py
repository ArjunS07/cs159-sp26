"""Build portable worker notebooks and a code/checkpoint bundle, without dispatch."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=Path.home()/'pnp-vla-runs'
DIGEST='671b5b211099997fc83d1277'
CHECKPOINTS={
 'frozen_mc':BASE/'checkpoints/smolvla-overnight-20260930-mc_roots_late'/DIGEST/'mc/latest.pt',
 'cnn_mc':BASE/'checkpoints/smolvla_local_continuation_v1'/DIGEST/'mc_anchored_cnn/latest.pt',
 'td':BASE/'checkpoints/smolvla_local_continuation_v1'/DIGEST/'td_n5/latest.pt'}
BUNDLE=ROOT/'output/smolvla_pcp_a100_trial_bundle.zip'


def cell(kind,text):
 return {'cell_type':kind,'id':hashlib.sha256(text.encode()).hexdigest()[:12],'metadata':{},'source':text.splitlines(keepends=True),**({'execution_count':None,'outputs':[]} if kind=='code' else {})}


def save(path,cells):
 path.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{
  'colab':{'provenance':[]},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'},
  'language_info':{'name':'python'},'gpuClass':'premium'},'cells':cells},indent=1)+'\n')


setup='''import os, sys, subprocess, zipfile
from pathlib import Path
os.environ.update(MUJOCO_GL='egl', NVIDIA_DRIVER_CAPABILITIES='compute,utility,graphics',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
from google.colab import userdata
for key in ('SUPABASE_URL','SUPABASE_SERVICE_KEY','HF_TOKEN'):
    try:
        value=userdata.get(key)
        if value: os.environ[key]=value
    except Exception: pass
for key in ('SUPABASE_URL','SUPABASE_SERVICE_KEY'):
    if not os.environ.get(key): raise RuntimeError(f'Add {key} in Colab Secrets and allow notebook access.')
if not BUNDLE_PATH:
    subprocess.run([sys.executable,'-m','pip','install','-q','supabase'],check=True)
    from supabase import create_client
    import hashlib
    client=create_client(os.environ['SUPABASE_URL'],os.environ['SUPABASE_SERVICE_KEY'])
    cached=Path('/content')/f'pcp_bundle_{BUNDLE_SHA256}.zip'
    if not cached.exists() or hashlib.sha256(cached.read_bytes()).hexdigest()!=BUNDLE_SHA256:
        data=client.storage.from_(BUNDLE_BUCKET).download(BUNDLE_STORAGE_KEY)
        if hashlib.sha256(data).hexdigest()!=BUNDLE_SHA256:
            raise RuntimeError('Downloaded bundle SHA256 mismatch')
        cached.write_bytes(data)
    BUNDLE_PATH=str(cached)
if __import__('hashlib').sha256(Path(BUNDLE_PATH).read_bytes()).hexdigest()!=BUNDLE_SHA256:
    raise RuntimeError('Bundle differs from the exact version pinned in this notebook')
PACKAGE=Path('/content/smolvla-pcp-trial')
PACKAGE.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE_PATH) as archive:
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in archive.namelist()):
        raise ValueError('Unsafe bundle path')
    archive.extractall(PACKAGE)
print({'python':sys.version,'executable':sys.executable},flush=True)
subprocess.run([sys.executable,'-m','pip','--version'],check=True)
install=subprocess.run([sys.executable,'-m','pip','install','-e',str(PACKAGE)+'[sim]',
                        'torch==2.11.0+cu128','torchvision==0.26.0+cu128',
                        '--extra-index-url','https://download.pytorch.org/whl/cu128'],
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
install_log=Path('/content/pcp_dependency_install.log')
install_log.write_text(install.stdout)
if install.returncode:
    print('\\n'.join(install.stdout.splitlines()[-120:]),flush=True)
    raise RuntimeError(f'Dependency installation failed (exit {install.returncode}). '
                       f'Full log: {install_log}. Paste the ERROR lines above; no experiment has started.')
print('Simulation/model dependencies installed.',flush=True)
subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=False,stdout=subprocess.DEVNULL)
if str(PACKAGE) not in sys.path: sys.path.insert(0,str(PACKAGE))
subprocess.run([sys.executable,'-m','pip','install',
                'torch==2.11.0+cu128','torchvision==0.26.0+cu128',
                '--index-url','https://download.pytorch.org/whl/cu128'],check=True)
if 'torch' in sys.modules and sys.modules['torch'].__version__ != '2.11.0+cu128':
    raise RuntimeError('Torch was already imported. Restart the runtime and rerun with the pinned version.')
from pnp import env_setup
if env_setup._find_nvidia_egl_library() is None:
    import re
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0]
    major=re.match(r'\\d+',driver.strip()).group()
    subprocess.run(['apt-get','update','-qq'],check=True)
    subprocess.run(['apt-get','install','-y','--no-install-recommends',f'libnvidia-gl-{major}'],check=True)
env_setup._ensure_nvidia_egl_vendor()
if env_setup._find_nvidia_egl_library() is None: raise RuntimeError('NVIDIA EGL missing; hardware rendering is required.')
env_setup._ensure_libero_config()
env_setup._fix_torch_quant_compat()
env_setup._remove_broken_optional_torchaudio()
env_setup._ensure_libero_assets(token=os.getenv('HF_TOKEN'))
import torch, hashlib, json
if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime.')
metadata=json.loads((PACKAGE/'bundle_manifest.json').read_text())
if metadata['checkpoints']!=CHECKPOINT_SHA256: raise RuntimeError('Unexpected checkpoint manifest')
CHECKPOINT_PATHS={name:str(PACKAGE/'checkpoints'/f'{name}.pt') for name in metadata['checkpoints']}
for name,path in CHECKPOINT_PATHS.items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=metadata['checkpoints'][name]:
        raise RuntimeError(f'Checkpoint mismatch: {name}')
print({'gpu':torch.cuda.get_device_name(0),'batch_size':BATCH_SIZE,'worker':WORKER_INDEX,'simulator_processes':SIMULATOR_PROCESSES})
'''


def main(publish=False):
 BUNDLE.parent.mkdir(exist_ok=True)
 manifest={'checkpoints':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in CHECKPOINTS.items()},
           'scope':'pnp Python sources, package metadata, and three frozen critic checkpoints; no secrets or optimizer changes'}
 with zipfile.ZipFile(BUNDLE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for path in sorted((ROOT/'pnp').rglob('*.py')):z.write(path,path.relative_to(ROOT))
  for name in ('pyproject.toml','README.md'):z.write(ROOT/name,name)
  for name,path in CHECKPOINTS.items():z.write(path,f'checkpoints/{name}.pt')
  z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
 bundle_sha=hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
 bundle_key=f'smolvla_pcp_sweep/bundles/{bundle_sha}/trial_bundle.zip'
 if publish:
  import sys
  sys.path.insert(0,str(ROOT))
  from scripts.train_smolvla_combined_local import _load_local_credentials
  from pnp.store import SupabaseStore
  _load_local_credentials()
  store=SupabaseStore()
  for name,path in CHECKPOINTS.items():
   sha=manifest['checkpoints'][name]
   key=f'smolvla_pcp_sweep/checkpoints/{sha}/{name}.pt'
   store._upload(key,path.read_bytes())
   if hashlib.sha256(store._download(key)).hexdigest()!=sha:raise RuntimeError(f'Remote checkpoint verification failed: {name}')
  store._upload(bundle_key,BUNDLE.read_bytes())
  if hashlib.sha256(store._download(bundle_key)).hexdigest()!=bundle_sha:raise RuntimeError('Remote bundle verification failed')
  store._upload(f'smolvla_pcp_sweep/bundles/{bundle_sha}/manifest.json',json.dumps({**manifest,'bundle_sha256':bundle_sha,'bundle_storage_key':bundle_key},indent=2).encode())
  print('Supabase bundle and all three checkpoints uploaded and hash-verified.')
 for worker in (0,1):
  config=f'''WORKER_INDEX = {worker}
WORKER_COUNT = 2
BATCH_SIZE = 8  # Jeff-matched replication setting; do not enlarge before comparison
SIMULATOR_PROCESSES = min(8, max(1, (__import__('os').cpu_count() or 2) - 2))
EXPERIMENT = 'smolvla-pcp-a100-jeff-settings-stock-v6'
BUNDLE_PATH = ''  # empty automatically downloads the exact pinned Supabase bundle
BUNDLE_BUCKET = 'artifacts'
BUNDLE_SHA256 = {bundle_sha!r}
BUNDLE_STORAGE_KEY = {bundle_key!r}
CHECKPOINT_SHA256 = {manifest['checkpoints']!r}
OUTPUT_DIR = f'/content/pcp-search-worker-{{WORKER_INDEX}}'
'''
  cells=[cell('markdown',f'''# PCP focused evaluation — A100 worker {worker}

The run cell first requires the separately logged 400-episode Jeff stock replication to match every historical outcome. Recorded matmul settings match Jeff: matmul TF32 off, highest precision; cuDNN TF32 retains the Torch default because Jeff did not record it; initial seed stream 0; batches grouped by task and historical shard parity. Existing results remain under their old experiment. Open worker 0 and worker 1 in **separate A100 Colab runtimes**. Add `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` to Colab Secrets, enable notebook access, then **Run all**. The code bundle and three frozen checkpoints download automatically from Supabase; there is no file-upload prompt. `HF_TOKEN` is optional for public assets. No GitHub checkout or manual file upload is needed. The notebook pins the bundle and each checkpoint by SHA256, verifies them before loading, and caches the download for reruns in this runtime.

CNN MC is excluded based on weaker previous gradient-direction evidence. These two critics are not established as universally best. The fixed grid has **18 configurations, 9 per worker**, plus each worker's paired **stock SmolVLA baseline: no P&P, no re-noising, no Q correction**. Treatments retain their P&P parent (steps 1/2/3, K 3/1/1) plus Q correction; this measures the whole treatment against stock, not Q's marginal benefit alone. All configurations apply PCP at sampler step 3, every action chunk throughout the full episode. Two modest standardized RMS radii (.02/.06): ascent, descent, and Q<0.5 gated ascent for frozen MC and continued TD (12 arms); matched random controls (2 arms); three-inner-step ascent for frozen MC and TD (4 arms). Most settings are fixed to prioritize depth over breadth. 

No wall-clock budget or cutoff, and no adaptive elimination. Every arm receives the same cases: up to 50 released starting states on each of 40 tasks, with three shared generation-noise streams (0/1/2; stream 0 matches Jeff). Normally **6,000 episodes per configuration**, or **60,000 full-episode rollouts per worker** including its zero baseline. Progress milestones are 400 / 2,000 / 6,000 cases; if the benchmark exposes fewer states, the plan adjusts automatically. ETA is measured from throughput, not guaranteed to fit 12 hours. Rerunning the same worker resumes completed Supabase rows.

Leaderboards include successes, rescues/spoils, conditional rescue/spoil rates, paired success change and standard errors clustered by task/starting state across repeated seeds. They also count actual PCP interventions. Chunks and repeated seeds are not counted as independent starting states. Both workers use the same cohort and shared baseline seeds; their baselines must not be pooled as independent replicates.

Batching starts at eight, grouped by task and historical shard parity, with parallel simulator processes, reused environments, matmul TF32 disabled, sparse camera rendering, and four background upload clients. Torch is pinned to 2.11.0+cu128. This does not reproduce every historical resume boundary; the dedicated replication runner reconstructs those separately. No critic classification/Bellman-validation phase, smoke rollout or video encoding. Actor CPU work and rendering still limit utilization; reported GPU utilization distinguishes that from inference limits.

Completed outcomes go to Supabase `rollouts` under `{{EXPERIMENT_PLACEHOLDER}}`, with configuration/checkpoint identity, errors, success and steps. Storage `artifacts/smolvla_pcp_sweep/...` contains the manifest, per-chunk correction telemetry, executed trajectories and worker leaderboards. Use the companion monitor notebook from any CPU Colab session while these workers run.

This is fixed, equal-exposure exploratory evaluation on released benchmark states, some of which may have been used previously. Many interventions within an episode do not create independent evidence. Standard errors are conditional on these 40 benchmark tasks and group seed repeats by starting state; they do not establish generalization to new tasks or remove multiple-comparison bias. Correction radii use the frozen MC training action standard deviations as a common coordinate system for every critic and random control. CNN's anchor is the live probe clean estimate, held fixed during its inner search.
'''.replace('{EXPERIMENT_PLACEHOLDER}','smolvla-pcp-a100-jeff-settings-stock-v6')),
         cell('code',config),cell('code',setup),cell('code', '''from pnp.smolvla_jeff_replication import require_replication
print(require_replication())
from pnp.smolvla_pcp_sweep import run_sweep, sweep_grid
# Retain shared random controls; divide the surviving 18 arms evenly.
GRID = [arm for arm in sweep_grid() if arm['critic'] in ('frozen_mc', 'td', 'none')]
for ordinal, arm in enumerate(GRID): arm['ordinal'] = ordinal
report=run_sweep(checkpoint_paths=CHECKPOINT_PATHS, worker_index=WORKER_INDEX,
                 worker_count=WORKER_COUNT, batch_size=BATCH_SIZE,
                 simulator_processes=SIMULATOR_PROCESSES,
                 output_dir=OUTPUT_DIR, experiment=EXPERIMENT, grid_override=GRID)
''')]
  save(ROOT/f'notebooks/workers/123_smolvla_pcp_a100_search_worker_{worker}.ipynb',cells)
 config=f"""WORKER_INDEX = 0  # this standalone experiment has one worker (your third Colab)
WORKER_COUNT = 1
BATCH_SIZE = 8
SIMULATOR_PROCESSES = min(8, max(1, (__import__('os').cpu_count() or 2) - 2))
BUNDLE_PATH = ''
BUNDLE_BUCKET = 'artifacts'
BUNDLE_SHA256 = {bundle_sha!r}
BUNDLE_STORAGE_KEY = {bundle_key!r}
CHECKPOINT_SHA256 = {manifest['checkpoints']!r}
OUTPUT_DIR = '/content/pcp-full-proposal'
"""
 cells=[cell('markdown',"""# Full PCP proposal — third A100 worker

Run this alongside workers 0 and 1 in a **third separate A100 runtime**. Add the same Supabase Secrets and Run all. Code and frozen checkpoints download automatically and are SHA256-verified. This has its own experiment, `smolvla-pcp-full-proposal-jeff-settings-v5`, and does not alter running workers.

**Algorithm at each action chunk:** predict a clean chunk at early/middle/late sampler indices 2/5/8 (80%/50%/20% noise remaining); use three independently re-noised one-step predictions to measure action disagreement; optionally gate on standardized RMS disagreement >= .06; apply one normalized Q-gradient correction inside radius .06 to the first 10 actions; re-noise into three independent branches; complete the remaining flow integration for each branch; rank their fully denoised chunks by Q; execute the best. The exact uncorrected continuation is included as candidate zero and is used when the gate is closed. No backpropagation through the flow and no model fine-tuning.

**18 fixed configurations:** frozen MC/continued TD x early/middle/late correction x always-on/disagreement-gated (12); reranking without correction for each critic at each timing (6). Radius is fixed at .06 to focus replication on timing. The exact parent P&P schedule and random draws are retained in the baseline continuation at every timing. CNN is excluded because prior directional evidence was weaker. All arms receive equal exposure: up to 2,000 starting states x three seeds = 6,000 full episodes each, plus the **stock SmolVLA baseline (no P&P or Q correction)**. The treatments and their candidate-zero fallback retain P&P; the fallback is distinct from the stock evaluation control. No time cutoff or adaptive promotion. This can exceed 12 hours; runtime estimates come from measured throughput.

The .06 uncertainty threshold is a fixed heuristic. Probe disagreement measures local action instability, not calibrated critic uncertainty or failure probability. Regeneration does not guarantee actions remain in distribution; reranking does not guarantee real success. The baseline fallback only enforces nondecreasing **predicted Q** when scores are finite. The correction radius bounds the injected clean-action displacement, not the final regenerated chunk.

Each branch uses the full episode batch on GPU and shares the encoded visual/language prefix; branches are evaluated sequentially to control memory. Parallel simulator processes retain only one environment per active lane. Batch eight preserves the task/parity grouping; CUDA OOM halves it automatically. Torch is pinned to 2.11.0+cu128.

Supabase stores every completed/error episode, executed actions, checkpoint/configuration identity, uncertainty, gate decisions, all terminal candidate Q scores, selected branch, correction magnitude and intervention/search counts. Progress uses `artifacts/smolvla_pcp_sweep/smolvla-pcp-full-proposal-jeff-settings-v5/worker_0_latest.json`. Success changes, rescues/spoils and standard errors group repeated seeds by starting state. Benchmark-state evaluation remains exploratory.
"""),cell('code',config),cell('code',setup),cell('code',"""from pnp.smolvla_jeff_replication import require_replication
print(require_replication())
from pnp.smolvla_pcp_full_proposal import run_full_proposal
report=run_full_proposal(checkpoint_paths=CHECKPOINT_PATHS, worker_index=WORKER_INDEX,
                        worker_count=WORKER_COUNT, batch_size=BATCH_SIZE,
                        simulator_processes=SIMULATOR_PROCESSES, output_dir=OUTPUT_DIR)
""")]
 save(ROOT/'notebooks/workers/125_smolvla_pcp_full_proposal_worker_2.ipynb',cells)
 # Stock-only parity check: two disjoint shards of Jeff's exact 400 identities.
 replication_setup=setup
 for worker in (0,1):
  cfg=f"""WORKER_INDEX = {worker}
BATCH_SIZE = 8
SIMULATOR_PROCESSES = 1  # parity runner uses Jeff's direct environment stepping
BUNDLE_PATH = ''
BUNDLE_BUCKET = 'artifacts'
BUNDLE_SHA256 = {bundle_sha!r}
BUNDLE_STORAGE_KEY = {bundle_key!r}
CHECKPOINT_SHA256 = {manifest['checkpoints']!r}
"""
  text=f"""# Jeff stock replication — Colab worker {worker}

Use this notebook and the other worker in your two separate **fresh GPU runtimes**. Each reproduces its historical shard, totaling 400 target identities. The runner reconstructs the seven recorded source runs and their resume boundaries; batches may include companion episodes needed to preserve historical membership. Add Supabase Secrets and Run all. Stop any previous sweep in that runtime first.

This is **stock SmolVLA only: no P&P, no re-noising, no critic and no reranking**. It uses Jeff's stream-0 episode seeds, ten Euler steps, 50 generated actions, ten executed actions, historical same-task batch limits (two or eight), historical rendering settings, matmul TF32 disabled and highest precision. cuDNN TF32 retains its Torch default because its historical value is unknown. Torch is pinned to Jeff's 2.11.0+cu128 build. Your Colab GPU may differ from Jeff's L4; outcome agreement is checked rather than assumed.

Results use a new v2 experiment; v1 rows are never reused. Resume skips whole reconstructed batches, preserving companion lanes rather than shrinking partially completed batches. At the end each notebook reads the latest combined results and compares every episode against Jeff's saved 255/400 stock arm. It reports successes, outcome flips, seed mismatches and step-count mismatches. A partial report is expected if the other worker is still running. Equal aggregate totals alone do not establish replication. No PCP sweep starts from this notebook.
"""
  run=f"""import torch
if torch.__version__ != '2.11.0+cu128':
    raise RuntimeError('Restart the runtime so the pinned Torch build loads.')
from pnp.smolvla_jeff_replication import run_replication
report = run_replication(shard_index=WORKER_INDEX)
print(json.dumps({{k:v for k,v in report.items() if k!='outcome_flips'}},indent=2))
if report['outcome_flips']:
    print('Episode outcome differences:',json.dumps(report['outcome_flips'],indent=2))
"""
  read="""from pnp.smolvla_jeff_replication import comparison
report = comparison()
print(json.dumps(report,indent=2))
"""
  save(ROOT/f'notebooks/workers/126_smolvla_jeff_stock_replication_worker_{worker}.ipynb',
       [cell('markdown',text),cell('code',cfg),cell('code',replication_setup),cell('code',run),
        cell('markdown','Rerun the next cell after both workers finish to refresh the combined comparison.'),cell('code',read)])

 monitor=[cell('markdown','''# PCP search — live Supabase monitor

CPU runtime is enough. Add the same Supabase Secrets, then run the cells whenever you want a fresh view. This reads the two worker progress artifacts and can pull all completed/error rollout rows with pagination. It launches no experiments.
'''),cell('code','''import sys, subprocess, os, json
subprocess.run([sys.executable,'-m','pip','install','-q','supabase','pandas'],check=True)
from google.colab import userdata
from supabase import create_client
import pandas as pd
client=create_client(userdata.get('SUPABASE_URL'),userdata.get('SUPABASE_SERVICE_KEY'))
EXPERIMENT='smolvla-pcp-a100-jeff-settings-stock-v6'
'''),cell('code','''reports=[]
for current_experiment,worker in [(EXPERIMENT,0),(EXPERIMENT,1),('smolvla-pcp-full-proposal-jeff-settings-v5',0)]:
    key=f'smolvla_pcp_sweep/{current_experiment}/worker_{worker}_latest.json'
    try:
        report=json.loads(client.storage.from_('artifacts').download(key));reports.append(report)
        print({k:v for k,v in report.items() if k!='leaderboard'})
        display(pd.DataFrame(report['leaderboard']).head(20))
    except Exception as error:
        if not any(t in str(error).lower() for t in ('404','not found','does not exist')):raise
        print(f'{current_experiment} worker {worker}: first progress artifact is not available yet.')
'''),cell('code','''# Optional: pull raw episodes, including errors. Completed reruns upsert stable IDs.
PULL_RAW_EPISODES=False
if PULL_RAW_EPISODES:
    rows=[]
    for start in range(0,1000000,1000):
        batch=(client.table('rollouts').select('rollout_id,method,suite,task_idx,episode_idx,behavior_seed_index,success,status,error_msg,n_steps,elapsed_s,ms_candidate_u')
               .eq('experiment',EXPERIMENT).order('rollout_id').range(start,start+999).execute().data or [])
        rows.extend(batch)
        if len(batch)<1000:break
    episodes=pd.DataFrame(rows)
    print('Logged rows:',len(episodes))
    display(episodes.head())
    episodes.to_json('/content/pcp_search_episodes.jsonl',orient='records',lines=True)
''')]
 save(ROOT/'notebooks/124_smolvla_pcp_a100_search_monitor.ipynb',monitor)
 print(json.dumps({'bundle':str(BUNDLE),'bytes':BUNDLE.stat().st_size,'workers':3}))


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--publish',action='store_true',help='Upload and verify the exact bundle/checkpoints in Supabase')
 main(publish=parser.parse_args().publish)
