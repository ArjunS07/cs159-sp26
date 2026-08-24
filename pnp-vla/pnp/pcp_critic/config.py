"""Configuration contracts for the PCP-search offline critic.

This module deliberately does not import the simulator or PI05.  A checkpoint is
therefore inspectable and its data contract can be checked before a GPU runtime
loads a VLA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ObjectiveKind = Literal["calql", "iql"]


@dataclass(frozen=True)
class PCPCriticModelConfig:
    """Architecture for a clean-50-action, physical-state-conditioned critic."""
    action_horizon: int = 50
    action_dim: int = 7
    width: int = 256
    n_rl_tokens: int = 4
    n_blocks: int = 4
    dropout: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PCPCriticTrainConfig:
    """Shared TD configuration; ``objective`` selects the ablation."""
    objective: ObjectiveKind = "calql"
    seed: int = 42
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    updates: int = 10_000
    eval_interval: int = 250
    patience: int = 10
    target_rate: float = 0.005
    grad_clip: float = 1.0
    # IQL branch.
    expectile: float = 0.7
    # Cal-QL branch. The candidates are generated in normalized action space.
    conservative_weight: float = 1.0
    n_local_actions: int = 4
    n_broad_actions: int = 4
    local_action_std: float = 0.10
    mc_calibration_weight: float = 1.0

    def __post_init__(self):
        if self.objective not in ("calql", "iql"):
            raise ValueError(f"unsupported PCP critic objective {self.objective!r}")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if self.batch_size < 1 or self.updates < 1:
            raise ValueError("batch_size and updates must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PCPSearchAdapterConfig:
    """Runtime guard and normalized clean-action correction settings."""
    mode: Literal["offline_only", "online_enabled"] = "offline_only"
    gradient_step: float = 0.10
    gradient_epsilon: float = 1e-6
    candidate_count: int = 5

    def __post_init__(self):
        if self.gradient_step < 0 or self.candidate_count < 1:
            raise ValueError("invalid PCP-search adapter configuration")

    def to_dict(self) -> dict:
        return asdict(self)
