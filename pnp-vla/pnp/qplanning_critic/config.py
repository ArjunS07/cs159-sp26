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
    # Optional deterministic compression of the frozen VLA prefix before the
    # critic projects it.  ``None`` preserves every historical checkpoint's
    # architecture and behaviour.  New demonstration-pretrained SmolVLA
    # critics use a bounded token count so thousands of cached visual prefixes
    # are practical to store and train on.
    prefix_pool_tokens: int | None = None

    def __post_init__(self):
        if self.action_horizon not in (10, 50):
            raise ValueError("the declared experiment supports only Q10 and Q50")
        if self.width % self.n_heads:
            raise ValueError("width must be divisible by n_heads")
        if self.n_bins < 2 or not self.value_min < self.value_max:
            raise ValueError("invalid categorical value support")
        if self.prefix_pool_tokens is not None and self.prefix_pool_tokens < 1:
            raise ValueError("prefix_pool_tokens must be positive or None")

    def to_dict(self) -> dict:
        result = asdict(self)
        # Keep the serialized architecture of old checkpoints byte-for-byte
        # compatible.  A missing field reconstructs the default ``None``.
        if result["prefix_pool_tokens"] is None:
            result.pop("prefix_pool_tokens")
        return result


@dataclass(frozen=True)
class QPlanningTrainConfig:
    seed: int = 42
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    effective_batch_size: int = 64
    micro_batch_size: int = 16
    updates: int = 8_000
    lr_schedule_updates: int | None = None
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
        if self.lr_schedule_updates is not None and self.lr_schedule_updates < self.updates:
            raise ValueError("lr_schedule_updates must be at least updates")
        for value in (self.print_interval, self.eval_interval, self.checkpoint_interval):
            if value < 1:
                raise ValueError("logging/checkpoint intervals must be positive")

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.micro_batch_size

    def learning_rate_at(self, update: int) -> float:
        if self.warmup_updates and update <= self.warmup_updates:
            return self.learning_rate * update / self.warmup_updates
        schedule_updates = self.lr_schedule_updates or self.updates
        span = max(1, schedule_updates - self.warmup_updates)
        progress = min(1.0, max(0.0, (update - self.warmup_updates) / span))
        return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> dict:
        return asdict(self)
