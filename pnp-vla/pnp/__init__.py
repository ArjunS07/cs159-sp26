"""pnp — Predict-and-Perturb / Predict-Correct-Perturb inference for pi0.5.

Import config freely anywhere; the sim/model-heavy modules (models, sampler,
rollout, libero_env, libero_pro) pull torch/lerobot/libero lazily so this package
imports without a GPU stack (e.g. for local analysis or store round-trips).
"""
from . import config  # noqa: F401
from .config import (  # noqa: F401
    RolloutConfig, TrainConfig, Method, ALL_METHODS, PCP_3WAY)

__version__ = "0.1.0"
