"""Replicate Jeff's 400 stock episodes on two L4 shards; no PCP dispatch."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.13')
 .apt_install('git','ffmpeg','libegl1','libgl1','libglib2.0-0','cmake','build-essential',
              'libegl1-mesa-dev','libgl1-mesa-dev')
 .env({'PYTHONPATH':'/root/pnp-vla','MUJOCO_GL':'egl',
       'NVIDIA_DRIVER_CAPABILITIES':'compute,utility,graphics',
       'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2'})
 .add_local_file(str(ROOT/'pyproject.toml'),'/root/pnp-vla/pyproject.toml',copy=True)
 .add_local_file(str(ROOT/'README.md'),'/root/pnp-vla/README.md',copy=True)
 .add_local_dir(str(ROOT/'pnp'),'/root/pnp-vla/pnp',copy=True,
                ignore=['**/__pycache__/**','**/*.pyc'])
 .run_commands("python -m pip install '/root/pnp-vla[sim]' torch==2.11.0+cu128 torchvision==0.26.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128",
               'python -m pip uninstall -y torchao',
               'python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128'))
app=modal.App('smolvla-jeff-stock-replication')

@app.function(image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),
              timeout=10800,startup_timeout=600,retries=0,max_containers=2,
              single_use_containers=True,scaledown_window=2,
              secrets=[modal.Secret.from_name('pnp-supabase')])
def worker(shard_index):
    import os,torch
    from pnp import env_setup
    env_setup._ensure_nvidia_egl_vendor()
    env_setup._ensure_libero_config()
    env_setup._ensure_libero_assets(token=os.getenv('HF_TOKEN'))
    env_setup._fix_torch_quant_compat()
    if not torch.cuda.is_available() or env_setup._find_nvidia_egl_library() is None:
        raise RuntimeError('L4 and hardware EGL required')
    if torch.__version__!='2.11.0+cu128':raise RuntimeError('Torch differs from Jeff')
    torch.set_num_threads(2)
    from pnp.smolvla_jeff_replication import run_replication
    return run_replication(shard_index)

@app.local_entrypoint()
def main():
    reports=list(worker.starmap([(0,),(1,)]))
    for r in reports:print({k:v for k,v in r.items() if k!='outcome_flips'})
