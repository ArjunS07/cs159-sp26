"""Twin long-horizon and short-horizon critics for candidate action selection."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 2, width),
        )

    def forward(self, x):
        return x + self.net(x)


class ChunkCriticMember(nn.Module):
    """Scores one fixed-horizon action chunk conditioned on a decision state."""

    def __init__(self, obs_dim: int, action_dim: int, horizon: int,
                 width: int = 256, dropout: float = 0.10):
        super().__init__()
        self.obs_dim, self.action_dim, self.horizon = obs_dim, action_dim, horizon
        self.state = nn.Sequential(
            nn.LayerNorm(obs_dim), nn.Linear(obs_dim, width), nn.GELU(),
            nn.LayerNorm(width))
        self.position = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, width))
        self.action = nn.Sequential(
            nn.Linear(horizon * (action_dim + 1), width), nn.GELU(), nn.LayerNorm(width))
        self.fusion = nn.Linear(width * 4, width)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(width, dropout) for _ in range(3)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def set_action_statistics(self, mean, std):
        self.action_mean.copy_(torch.as_tensor(mean, dtype=self.action_mean.dtype))
        self.action_std.copy_(torch.as_tensor(std, dtype=self.action_std.dtype).clamp_min(1e-6))

    def forward(self, obs, position, actions, mask, *, zero_context=False):
        if actions.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"expected actions [..., {self.horizon}, {self.action_dim}], "
                f"got {tuple(actions.shape)}")
        state = self.state(obs)
        if zero_context:
            state = torch.zeros_like(state)
        position = self.position(position.reshape(-1, 1).float())
        valid = mask.float().unsqueeze(-1)
        normalized = ((actions - self.action_mean) / self.action_std) * valid
        action = self.action(torch.cat([normalized, valid], -1).flatten(1))
        fused = self.fusion(torch.cat([state, position, action, state * action], -1))
        return self.head(self.blocks(fused)).squeeze(-1)


class StateValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim + 1), nn.Linear(obs_dim + 1, width), nn.GELU(),
            ResidualMLPBlock(width, dropout), ResidualMLPBlock(width, dropout),
            nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, obs, position):
        return self.net(torch.cat([obs, position.reshape(-1, 1).float()], -1)).squeeze(-1)


class HybridChunkCritic(nn.Module):
    """IQL-style 50-step critic plus a deployable 10-step twin critic."""

    def __init__(self, obs_dim: int = 2048, action_dim: int = 7,
                 long_horizon: int = 50, short_horizon: int = 10,
                 width: int = 256, dropout: float = 0.10):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.long_horizon, self.short_horizon = long_horizon, short_horizon
        self.width, self.dropout = width, dropout
        self.long_critics = nn.ModuleList([
            ChunkCriticMember(obs_dim, action_dim, long_horizon, width, dropout)
            for _ in range(2)])
        self.short_critics = nn.ModuleList([
            ChunkCriticMember(obs_dim, action_dim, short_horizon, width, dropout)
            for _ in range(2)])
        self.value_network = StateValueNetwork(obs_dim, width, dropout)

    def architecture_config(self):
        return {
            "obs_dim": self.obs_dim, "action_dim": self.action_dim,
            "long_horizon": self.long_horizon, "short_horizon": self.short_horizon,
            "width": self.width, "dropout": self.dropout,
        }

    def set_action_statistics(self, mean, std):
        for critic in (*self.long_critics, *self.short_critics):
            critic.set_action_statistics(mean, std)

    def long_values(self, obs, position, actions, mask, *, zero_context=False):
        return torch.stack([
            critic(obs, position, actions, mask, zero_context=zero_context)
            for critic in self.long_critics])

    def short_values(self, obs, position, actions, mask, *, zero_context=False):
        return torch.stack([
            critic(obs, position, actions, mask, zero_context=zero_context)
            for critic in self.short_critics])

    def state_value(self, obs, position):
        return self.value_network(obs, position)

    def encode_context(self, obs_enc, chunk_position):
        return obs_enc, chunk_position

    def rank_candidates(self, context, actions, action_mask, prefix_length=10,
                        zero_context=False):
        if int(prefix_length) != self.short_horizon:
            raise ValueError(f"hybrid critic requires prefix_length={self.short_horizon}")
        if actions.ndim == 3:
            actions, action_mask = actions[:, None], action_mask[:, None]
        batch, candidates = actions.shape[:2]
        obs, position = context
        obs = obs[:, None].expand(batch, candidates, -1).reshape(batch * candidates, -1)
        position = position[:, None].expand(batch, candidates).reshape(-1)
        prefix = actions[..., :self.short_horizon, :].reshape(
            batch * candidates, self.short_horizon, self.action_dim)
        mask = action_mask[..., :self.short_horizon].reshape(
            batch * candidates, self.short_horizon)
        values = self.short_values(
            obs, position, prefix, mask, zero_context=zero_context).amin(0)
        return values.reshape(batch, candidates)

    score_candidates = rank_candidates
