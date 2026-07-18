"""Colab environment bootstrap (pi0.5 only).

Ported from build_final_notebooks.py ENV_SETUP_SRC: torch/quantization compat repair,
lazy pip ensures, MUJOCO_GL/HF_HOME env, optional site-packages snapshot restore, HF login.
Call `setup_environment()` once at the top of a Colab notebook (before importing libero/lerobot).
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys


def _ensure(mod: str, pip_spec: str) -> None:
    try:
        importlib.import_module(mod)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pip_spec.split()])
        importlib.invalidate_caches()


def _fix_torch_quant_compat() -> None:
    """Provide quantization constants expected by the LeRobot/diffusers import chain.

    Some Colab Torch builds contain quantization modules whose ``fx.convert`` module imports
    these names from ``torch.ao.quantization`` even though that package does not export them.
    The constants are dictionary keys, so supplying the canonical string values is sufficient
    and avoids replacing Colab's matched Torch/CUDA stack at runtime.
    """
    import torch
    import torch.ao.quantization as taoq

    patched = []
    for name, value in (("CUSTOM_KEY", "custom"),
                        ("NUMERIC_DEBUG_HANDLE_KEY", "numeric_debug_handle")):
        if not hasattr(taoq, name):
            setattr(taoq, name, value)
            patched.append(name)

    suffix = f" (patched {', '.join(patched)})" if patched else ""
    print(f"torch {torch.__version__} — quantization compat OK{suffix}")


def _restore_snapshot(cache_dir: str) -> None:
    snapshot = os.path.join(cache_dir, "site_packages.tar.gz")
    if os.path.exists(snapshot):
        import shutil
        local = "/content/site_packages_restore.tar.gz"
        shutil.copy(snapshot, local)
        subprocess.run(["tar", "-xzf", local, "-C", "/"], check=True)
        os.remove(local)
        importlib.invalidate_caches()
        print("Restored package snapshot.")


def setup_environment(cache_dir: str | None = None, hf_home: str = "/content/hf_home") -> None:
    """Prepare a Colab session for pi0.5 + LIBERO/MuJoCo inference."""
    try:
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
              or "No GPU detected (CPU will be infeasible).")
    except FileNotFoundError:
        print("nvidia-smi not found (no GPU).")

    if cache_dir:
        _restore_snapshot(cache_dir)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Keep the HF cache on fast LOCAL disk; the Drive FUSE mount truncates multi-GB shards.
    os.environ["HF_HOME"] = hf_home

    _fix_torch_quant_compat()
    _ensure("mujoco", "mujoco")
    _ensure("libero", "libero")
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lerobot[pi0]"])
        importlib.invalidate_caches()

    from huggingface_hub import login
    tok = os.getenv("HF_TOKEN")
    login(token=tok) if tok else None
    print("Environment ready.")
