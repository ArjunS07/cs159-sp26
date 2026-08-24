"""Frozen-prefix RL-token encoder and action-reinjected twin PCP critic."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import PCPCriticModelConfig


class ResidualFiLMBlock(nn.Module):
    """Action is injected at every block, preventing an action-agnostic Q head."""
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(width, width * 2)
        self.net = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width * 2, width))

    def forward(self, value: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(action_context).chunk(2, -1)
        h = self.norm(value) * (1 + scale) + shift
        return value + self.net(h)


class RLTokenEncoder(nn.Module):
    """Learned latent tokens pooled from frozen VLA prefixes and physical state."""
    def __init__(self, prefix_dim: int, robot_dim: int, proprio_dim: int, width: int, n_tokens: int):
        super().__init__()
        self.prefix = nn.Sequential(nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, width))
        self.physical = nn.Sequential(
            nn.LayerNorm(robot_dim + proprio_dim), nn.Linear(robot_dim + proprio_dim, width), nn.GELU())
        self.tokens = nn.Parameter(torch.empty(n_tokens, width))
        nn.init.normal_(self.tokens, std=.02)
        self.cross = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, prefix_embeddings: torch.Tensor, prefix_pad_mask: torch.Tensor,
                robot_state: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if prefix_embeddings.ndim != 3:
            raise ValueError("prefix_embeddings must be [batch,tokens,width]")
        encoded = self.prefix(prefix_embeddings.float())
        physical = self.physical(torch.cat([robot_state.float(), proprio.float()], -1))
        query = self.tokens.unsqueeze(0).expand(len(encoded), -1, -1) + physical[:, None, :]
        # MultiheadAttention uses True for ignored positions, our artifact pad mask uses True valid.
        pooled, _ = self.cross(query, encoded, encoded, key_padding_mask=~prefix_pad_mask.bool(),
                               need_weights=False)
        return self.norm(pooled + query)


class ActionConditionedQ(nn.Module):
    def __init__(self, width: int, horizon: int, action_dim: int, n_blocks: int, dropout: float):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.action_tokens = nn.Sequential(nn.Linear(action_dim, width), nn.GELU(), nn.LayerNorm(width))
        self.action_pool = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.state_pool = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([ResidualFiLMBlock(width, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, rl_tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected action [batch,{self.horizon},{self.action_dim}], got {tuple(action.shape)}")
        tokens = self.action_tokens(action.float())
        # Mean + max forces the action pathway to retain sharp late-chunk deviations.
        action_context = self.action_pool(torch.cat([tokens.mean(1), tokens.amax(1)], -1))
        value = self.state_pool(rl_tokens.mean(1))
        for block in self.blocks:
            value = block(value, action_context)
        return self.head(value).squeeze(-1)


class PCPCritic(nn.Module):
    """Twin clean-action Q critic; PI05 prefix embeddings are inputs, never parameters."""
    def __init__(self, *, prefix_dim: int, robot_dim: int, proprio_dim: int,
                 config: PCPCriticModelConfig | None = None):
        super().__init__()
        self.config = config or PCPCriticModelConfig()
        self.prefix_dim, self.robot_dim, self.proprio_dim = prefix_dim, robot_dim, proprio_dim
        c = self.config
        self.rl_encoder = RLTokenEncoder(prefix_dim, robot_dim, proprio_dim, c.width, c.n_rl_tokens)
        self.q_heads = nn.ModuleList([
            ActionConditionedQ(c.width, c.action_horizon, c.action_dim, c.n_blocks, c.dropout)
            for _ in range(2)])
        self.value_head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width), nn.GELU(),
                                        nn.Linear(c.width, 1))
        self.register_buffer("action_mean", torch.zeros(c.action_dim))
        self.register_buffer("action_std", torch.ones(c.action_dim))

    def architecture_config(self) -> dict:
        return {"prefix_dim": self.prefix_dim, "robot_dim": self.robot_dim,
                "proprio_dim": self.proprio_dim, **self.config.to_dict()}

    def set_action_statistics(self, mean, std) -> None:
        self.action_mean.copy_(torch.as_tensor(mean, dtype=self.action_mean.dtype))
        self.action_std.copy_(torch.as_tensor(std, dtype=self.action_std.dtype).clamp_min(1e-6))

    def encode_state(self, prefix_embeddings, prefix_pad_mask, robot_state, proprio):
        return self.rl_encoder(prefix_embeddings, prefix_pad_mask, robot_state, proprio)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def q_values_from_tokens(self, rl_tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_action(action)
        return torch.stack([head(rl_tokens, normalized) for head in self.q_heads])

    def q_values(self, prefix_embeddings, prefix_pad_mask, robot_state, proprio, action) -> torch.Tensor:
        return self.q_values_from_tokens(self.encode_state(
            prefix_embeddings, prefix_pad_mask, robot_state, proprio), action)

    def minimum_q(self, prefix_embeddings, prefix_pad_mask, robot_state, proprio, action) -> torch.Tensor:
        return self.q_values(prefix_embeddings, prefix_pad_mask, robot_state, proprio, action).amin(0)

    def state_value_from_tokens(self, rl_tokens: torch.Tensor) -> torch.Tensor:
        return self.value_head(rl_tokens.mean(1)).squeeze(-1)

    def state_value(self, prefix_embeddings, prefix_pad_mask, robot_state, proprio) -> torch.Tensor:
        return self.state_value_from_tokens(self.encode_state(
            prefix_embeddings, prefix_pad_mask, robot_state, proprio))
