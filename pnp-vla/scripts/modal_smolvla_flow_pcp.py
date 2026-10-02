"""Bounded focused PCP evaluation. Launch only after explicit spending approval.

Uploads only pnp source, package metadata and the named frozen critic checkpoint.
Uses the existing pnp-supabase secret; never uploads a local dotenv file.
"""
from pathlib import Path
import hashlib
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
SHA = "9b8800bc9f27116afcb1812f93f8a9f6767ac012c7cdac790502032fbb7d0c8e"
CHECKPOINT = (Path.home() / "pnp-vla-runs/checkpoints"
              / "smolvla-overnight-20260930-mc_roots_late"
              / "671b5b211099997fc83d1277/mc/latest.pt")
if modal.is_local() and (not CHECKPOINT.is_file() or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != SHA):
    raise ValueError("The approved frozen late-fusion checkpoint is missing or changed")

image = (modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "ffmpeg", "libegl1", "libgl1", "libglib2.0-0",
                 "cmake", "build-essential", "libegl1-mesa-dev", "libgl1-mesa-dev")
    .env({"PYTHONPATH": "/root/pnp-vla", "MUJOCO_GL": "egl",
          "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
          "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2"})
    .add_local_file(str(ROOT / "pyproject.toml"), "/root/pnp-vla/pyproject.toml", copy=True)
    .add_local_file(str(ROOT / "README.md"), "/root/pnp-vla/README.md", copy=True)
    .add_local_dir(str(ROOT / "pnp"), "/root/pnp-vla/pnp", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .run_commands("python -m pip install '/root/pnp-vla[sim]'", "python -m pip uninstall -y torchao")
    .add_local_file(str(CHECKPOINT), "/root/critic.pt"))
app = modal.App("smolvla-focused-flow-pcp")


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(16384, 16384),
              timeout=5400, startup_timeout=600, retries=0,
              max_containers=3, single_use_containers=True, scaledown_window=2,
              secrets=[modal.Secret.from_name("pnp-supabase")])
def worker(shard_index: int, root_limit: int | None = None) -> dict:
    import os
    import torch
    from pnp import env_setup
    # SmolVLA and LIBERO assets are public; no gated pi0.5 token is required.
    env_setup._ensure_nvidia_egl_vendor()
    env_setup._ensure_libero_config()
    env_setup._ensure_libero_assets(token=os.getenv("HF_TOKEN"))
    env_setup._fix_torch_quant_compat()
    if not torch.cuda.is_available():
        raise RuntimeError("Requested L4 is unavailable")
    if env_setup._find_nvidia_egl_library() is None:
        raise RuntimeError("NVIDIA EGL missing; refuse slow software rendering")
    torch.set_num_threads(2)
    from pnp.smolvla_focused_pcp_eval import run_focused_flow_pcp_eval
    return run_focused_flow_pcp_eval(
        checkpoint_path="/root/critic.pt", expected_checkpoint_sha=SHA,
        shard_index=shard_index, shard_count=3, root_limit=root_limit,
        run_intervention=True, cache_root="/tmp/focused-pcp",
        output_path=f"/tmp/flow_pcp_shard_{shard_index}.json", device="cuda")


@app.local_entrypoint()
def main(smoke: bool = True, approve_spend: bool = False):
    """A smoke call collects one root; full calls exactly three disjoint shards."""
    if not approve_spend:
        raise ValueError("Explicit spending approval required: --approve-spend")
    if smoke:
        results = [worker.remote(0, root_limit=1)]
    else:
        # Blocking gather preserves visibility of failures; no automatic retries.
        results = list(worker.starmap([(0, None), (1, None), (2, None)]))
    directory = Path.home() / "pnp-vla-runs/diagnostics/focused_flow_pcp_v1"
    directory.mkdir(parents=True, exist_ok=True)
    for report in results:
        path = directory / f"{report['experiment']}_shard_{report['shard_index']}.json"
        path.write_text(json.dumps(report, indent=2))
        print({"report": str(path), "new_roots": report["new_roots"]})
