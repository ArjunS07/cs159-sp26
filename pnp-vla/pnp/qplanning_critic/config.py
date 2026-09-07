"""Configuration for the paper-style Q50 and causal Q10 critic tests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class QPlanningModelConfig:
    action_horizon: int
    action_dim: int = 7
    width: int = 768
    n_layers: int = 8
    n_heads: int = 12
    ffn_width: int = 3072
    dropout: float = 0.10
    n_bins: int = 101
    value_min: float = 0.0
    value_max: float = 1.0
    hl_gauss_sigma: float = 0.01

    def __post_init__(self):
        if self.action_horizon not in (10, 50):
            raise ValueError("the declared experiment supports only Q10 and Q50")
        if self.width % self.n_heads:
            raise ValueError("width must be divisible by n_heads")
        if self.n_bins < 2 or not self.value_min < self.value_max:
            raise ValueError("invalid categorical value support")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QPlanningTrainConfig:
    seed: int = 42
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    effective_batch_size: int = 64
    micro_batch_size: int = 16
    updates: int = 8_000
    warmup_updates: int = 500
    print_interval: int = 100
    eval_interval: int = 500
    checkpoint_interval: int = 1_000
    target_rate: float = 0.005
    grad_clip: float = 1.0
    max_validation_transitions: int = 4096
    use_bf16: bool = True

    def __post_init__(self):
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if min(self.effective_batch_size, self.micro_batch_size, self.updates) < 1:
            raise ValueError("batch sizes and updates must be positive")
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        if self.warmup_updates < 0 or self.warmup_updates > self.updates:
            raise ValueError("warmup_updates must lie in [0, updates]")
        for value in (self.print_interval, self.eval_interval, self.checkpoint_interval):
            if value < 1:
                raise ValueError("logging/checkpoint intervals must be positive")

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.micro_batch_size

    def learning_rate_at(self, update: int) -> float:
        if self.warmup_updates and update <= self.warmup_updates:
            return self.learning_rate * update / self.warmup_updates
        span = max(1, self.updates - self.warmup_updates)
        progress = min(1.0, max(0.0, (update - self.warmup_updates) / span))
        return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> dict:
        return asdict(self)
