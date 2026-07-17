"""pi0.5 model loading + P&P patch application (pi0.5 only).

Ported from smolvla_eval_core.py (_ensure_hf_weights, load_pi05, apply_pnp_patch). The
SmolVLA loader, RTC, and torch.compile dual-model branches are dropped.
"""
from __future__ import annotations

import glob
import os

import torch

from .config import PI05_REPO_ID
from . import sampler as _sampler


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_hf_weights(repo_id: str) -> str:
    """Fully download repo_id and validate every safetensors shard.

    lerobot's from_pretrained silently falls back to RANDOM weights when a cached .safetensors
    is truncated (which happens on the Drive FUSE mount). Force a clean re-download on integrity
    failure, and raise loudly rather than evaluating an untrained model.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    tok = os.getenv("HF_TOKEN")
    last_err = None
    for force in (False, True):
        path = snapshot_download(repo_id, token=tok, force_download=force)
        shards = glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True)
        if not shards:
            return path
        try:
            for f in shards:
                with safe_open(f, framework="pt"):
                    pass
            return path
        except Exception as e:                       # truncated/corrupt -> force re-download
            last_err = e
            print(f"[weights] integrity check failed: {e}; re-downloading {repo_id}...")
    raise RuntimeError(f"Invalid/incomplete weights for {repo_id}: {last_err}. "
                       "Set HF_TOKEN (gated repo) and check disk space.")


def _infer_action_dim(policy, default=7) -> int:
    feats = getattr(policy.config, "output_features", {})
    if "action" in feats:
        return int(feats["action"].shape[0])
    return default


def apply_pnp_patch(policy) -> None:
    """Install the unified hooked sampler on the pi0.5 policy model."""
    _sampler.install_patch(policy.model)
    adim = _infer_action_dim(policy)
    policy.model._pnp_action_dim = adim   # informational; RolloutConfig.action_dim defaults to ADIM=7
    print(f"Patched pi05 sample_actions (action_dim={adim})")


def load_pi05(device=None, repo_id: str = PI05_REPO_ID):
    """Load the pi0.5 LIBERO-finetuned policy and its pre/post processors, patched for P&P."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    device = device or default_device()
    _ensure_hf_weights(repo_id)
    policy = PI05Policy.from_pretrained(repo_id).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, repo_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    apply_pnp_patch(policy)
    return policy, preprocess, postprocess
