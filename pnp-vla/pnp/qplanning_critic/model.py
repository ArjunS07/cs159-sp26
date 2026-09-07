"""Single-Q RL-token decoder with an HL-Gauss value head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QPlanningModelConfig


class QPlanningCritic(nn.Module):
    """Score an action sequence using frozen PI context plus physical state."""

    def __init__(self, *, prefix_dim: int, robot_dim: int, proprio_dim: int,
                 config: QPlanningModelConfig):
        super().__init__()
        self.config = config
        self.prefix_dim = int(prefix_dim)
        self.robot_dim = int(robot_dim)
        self.proprio_dim = int(proprio_dim)
        c = config
        self.prefix_projection = nn.Sequential(nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, c.width))
        self.robot_projection = nn.Sequential(nn.LayerNorm(robot_dim), nn.Linear(robot_dim, c.width))
        self.proprio_projection = nn.Sequential(nn.LayerNorm(proprio_dim), nn.Linear(proprio_dim, c.width))
        self.action_projection = nn.Sequential(nn.Linear(c.action_dim, c.width), nn.LayerNorm(c.width))
        self.action_positions = nn.Parameter(torch.empty(c.action_horizon, c.width))
        self.rl_token = nn.Parameter(torch.empty(1, 1, c.width))
        layer = nn.TransformerDecoderLayer(
            d_model=c.width, nhead=c.n_heads, dim_feedforward=c.ffn_width,
            dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=c.n_layers, norm=nn.LayerNorm(c.width))
        self.value_head = nn.Linear(c.width, c.n_bins)
        self.register_buffer("action_mean", torch.zeros(c.action_dim))
        self.register_buffer("action_std", torch.ones(c.action_dim))
        self.register_buffer("value_bins", torch.linspace(c.value_min, c.value_max, c.n_bins))
        nn.init.normal_(self.rl_token, std=0.02)
        nn.init.normal_(self.action_positions, std=0.02)

    def architecture_config(self) -> dict:
        return {"prefix_dim": self.prefix_dim, "robot_dim": self.robot_dim,
                "proprio_dim": self.proprio_dim, **self.config.to_dict()}

    def set_action_statistics(self, mean, std) -> None:
        self.action_mean.copy_(torch.as_tensor(mean, dtype=self.action_mean.dtype))
        self.action_std.copy_(torch.as_tensor(std, dtype=self.action_std.dtype).clamp_min(1e-6))

    def forward(self, prefix: torch.Tensor, prefix_valid: torch.Tensor,
                robot: torch.Tensor, proprio: torch.Tensor,
                action: torch.Tensor, action_valid: torch.Tensor) -> torch.Tensor:
        c = self.config
        if action.shape[-2:] != (c.action_horizon, c.action_dim):
            raise ValueError(
                f"expected action [batch,{c.action_horizon},{c.action_dim}], got {tuple(action.shape)}")
        if action_valid.shape != action.shape[:2]:
            raise ValueError("action_valid must be [batch,horizon]")
        memory = torch.cat([
            self.prefix_projection(prefix.float()),
            self.robot_projection(robot.float())[:, None, :],
            self.proprio_projection(proprio.float())[:, None, :],
        ], dim=1)
        memory_valid = torch.cat([
            prefix_valid.bool(),
            torch.ones((len(prefix), 2), dtype=torch.bool, device=prefix.device),
        ], dim=1)
        normalized = (action.float() - self.action_mean) / self.action_std
        action_tokens = self.action_projection(normalized) + self.action_positions[None]
        rl = self.rl_token.expand(len(action), -1, -1)
        target = torch.cat([rl, action_tokens], dim=1)
        target_ignored = torch.cat([
            torch.zeros((len(action), 1), dtype=torch.bool, device=action.device),
            ~action_valid.bool(),
        ], dim=1)
        decoded = self.decoder(
            target, memory, tgt_key_padding_mask=target_ignored,
            memory_key_padding_mask=~memory_valid)
        return self.value_head(decoded[:, 0])

    def expected_value(self, *args, **kwargs) -> torch.Tensor:
        logits = self(*args, **kwargs)
        return (logits.softmax(-1) * self.value_bins).sum(-1)

    def hl_gauss_targets(self, target: torch.Tensor) -> torch.Tensor:
        """Project scalar returns onto the fixed support with Gaussian CDF bins."""
        target = target.float().clamp(self.config.value_min, self.config.value_max)
        centers = self.value_bins.float()
        edges = (centers[:-1] + centers[1:]) * 0.5
        lower = torch.cat([torch.full_like(edges[:1], -torch.inf), edges])
        upper = torch.cat([edges, torch.full_like(edges[:1], torch.inf)])
        scale = self.config.hl_gauss_sigma * (2.0 ** 0.5)
        cdf_upper = 0.5 * (1.0 + torch.erf((upper[None] - target[:, None]) / scale))
        cdf_lower = 0.5 * (1.0 + torch.erf((lower[None] - target[:, None]) / scale))
        probabilities = (cdf_upper - cdf_lower).clamp_min(0)
        return probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)

    def categorical_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distribution = self.hl_gauss_targets(target)
        return -(distribution * F.log_softmax(logits.float(), -1)).sum(-1).mean()
