"""Deterministic Colab bootstrap for the pinned pi0.5 simulation stack.

This module deliberately does not install or upgrade the core Torch stack.  Colab ships a
matched Torch/TorchVision/CUDA build; replacing part of it in a live process is unsafe.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys


_QUANTIZATION_CONSTANTS = (
    ("CUSTOM_KEY", "custom"),
    ("NUMERIC_DEBUG_HANDLE_KEY", "numeric_debug_handle"),
    ("FROM_NODE_KEY", "from_node"),
)


def _fix_torch_quant_compat() -> tuple[str, ...]:
    """Add constants expected by the LeRobot import chain, idempotently."""
    quantization = importlib.import_module("torch.ao.quantization")

    patched = []
    for name, value in _QUANTIZATION_CONSTANTS:
        if not hasattr(quantization, name):
            setattr(quantization, name, value)
            patched.append(name)
    return tuple(patched)


def _uninstall_optional(package: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", package]
    )
    for name in tuple(sys.modules):
        if name == package or name.startswith(f"{package}."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _remove_torchao() -> bool:
    """Remove torchao when present; pi0.5 does not need this incompatible extra."""
    if importlib.util.find_spec("torchao") is None:
        return False
    print("Removing incompatible optional torchao (not required by pi0.5).")
    _uninstall_optional("torchao")
    return True


def _remove_broken_optional_torchaudio() -> bool:
    """Retain a working TorchAudio and remove only an unloadable optional build."""
    if importlib.util.find_spec("torchaudio") is None:
        return False
    try:
        importlib.import_module("torchaudio")
    except Exception as exc:
        print(f"Removing incompatible optional torchaudio ({exc}).")
        _uninstall_optional("torchaudio")
        return True
    return False


def _require_core_runtime():
    """Return Torch/TorchVision after validating Colab's native GPU stack."""
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        torchvision = importlib.import_module("torchvision")
        has_ops = getattr(getattr(torchvision, "extension", None), "_has_ops", None)
        if callable(has_ops) and not has_ops():
            raise RuntimeError("TorchVision native operators failed to load")
    except Exception as exc:
        raise RuntimeError(
            "Colab's Torch/TorchVision/CUDA stack is incompatible or no usable CUDA GPU "
            "is attached. Start a fresh GPU runtime and reinstall only pnp-vla[sim]; do "
            "not upgrade or hot-swap torch, torchvision, or CUDA in the running runtime. "
            f"Original error: {exc}"
        ) from exc
    return torch, torchvision


def _require_hf_credentials() -> str:
    """Accept HF_TOKEN or a token already stored by huggingface_hub."""
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    if not token:
        raise RuntimeError(
            "Hugging Face credentials are missing. Set the HF_TOKEN Colab secret (with "
            "access to the gated pi0.5 checkpoint) or authenticate once with huggingface-cli."
        )
    return token


def _verify_model_and_sim_imports() -> dict[str, object]:
    """Import every API needed by model loading and LIBERO before declaring readiness."""
    try:
        transformers = importlib.import_module("transformers")
        getattr(transformers, "AutoProcessor")
        pi05 = importlib.import_module("lerobot.policies.pi05.modeling_pi05")
        getattr(pi05, "PI05Policy")
        factory = importlib.import_module("lerobot.policies.factory")
        getattr(factory, "make_pre_post_processors")
        libero = importlib.import_module("libero")
        mujoco = importlib.import_module("mujoco")
        _require_robosuite_mujoco_api(mujoco)
        lerobot = importlib.import_module("lerobot")
    except Exception as exc:
        raise RuntimeError(
            "The pinned pi0.5/LIBERO stack failed its import check. Start a fresh Colab "
            "GPU runtime, clone the requested revision, and run `pip install -e "
            "'/content/cs159-sp26/pnp-vla[sim]'` once. "
            f"Original error: {exc}"
        ) from exc
    return {
        "Transformers": transformers,
        "LeRobot": lerobot,
        "LIBERO": libero,
        "MuJoCo": mujoco,
    }


def _require_robosuite_mujoco_api(mujoco: object) -> None:
    """Reject MuJoCo's 3.10 API, which legacy robosuite cannot call."""
    version = _version(mujoco, "mujoco")
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match and tuple(map(int, match.groups())) >= (3, 10):
        raise RuntimeError(
            f"MuJoCo {version} uses mj_fullM(model, data, dst), but LIBERO's robosuite "
            "requires the pre-3.10 mj_fullM(model, dst, qM) API. Install the pinned "
            "pnp-vla[sim] dependencies in a fresh runtime (mujoco>=3.1.6,<3.10)."
        )


def _version(module: object, distribution: str) -> str:
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def setup_environment(hf_home: str = "/content/hf_home") -> None:
    """Validate and prepare a fresh Colab GPU runtime for pi0.5 + LIBERO."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Google Drive FUSE has corrupted multi-GB model shards; always use local disk.
    os.environ["HF_HOME"] = hf_home

    torch, torchvision = _require_core_runtime()
    patched = _fix_torch_quant_compat()
    if patched:
        print(f"Patched Torch quantization constants: {', '.join(patched)}")
    _remove_torchao()
    _remove_broken_optional_torchaudio()
    _require_hf_credentials()
    modules = _verify_model_and_sim_imports()

    try:
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = "unknown CUDA device"
    versions = {
        "GPU": gpu,
        "Torch": _version(torch, "torch"),
        "CUDA": str(getattr(getattr(torch, "version", None), "cuda", "unknown")),
        "TorchVision": _version(torchvision, "torchvision"),
        **{name: _version(module, name.lower()) for name, module in modules.items()},
    }
    print("Environment versions: " + ", ".join(f"{key}={value}" for key, value in versions.items()))
    print("Environment ready.")
