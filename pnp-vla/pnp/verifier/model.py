"""Temporal clean-action verifier with independently reusable state context."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VerifierOutput:
    state_logit: torch.Tensor
    joint_logit: torch.Tensor

    @property
    def action_residual(self):
        return self.joint_logit - self.state_logit


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 1), nn.GroupNorm(8, width),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MaskedAttentionPool(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, x, mask):
        logits = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1) * mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class CleanChunkVerifier(nn.Module):
    """Score clean environment-space candidate chunks.

    ``score_candidates`` is permutation-equivariant over the candidate dimension and caches the
    observation pathway through ``encode_context``.
    """
    def __init__(self, obs_dim: int = 2048, action_dim: int = 7, width: int = 64,
                 context_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim), nn.Linear(obs_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, context_dim), nn.LayerNorm(context_dim),
        )
        self.position_encoder = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.state_head = nn.Sequential(
            nn.Linear(context_dim + 32, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.action_norm = nn.LayerNorm(action_dim)
        self.action_in = nn.Linear(action_dim, width)
        self.temporal = nn.ModuleList([
            ResidualTemporalBlock(width, dilation, dropout) for dilation in (1, 2, 4)])
        self.prefix_pool = MaskedAttentionPool(width)
        self.future_pool = MaskedAttentionPool(width)
        self.film = nn.Linear(context_dim, width * 4)
        joint_in = context_dim + 32 + width * 2
        self.joint_head = nn.Sequential(
            nn.Linear(joint_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def encode_context(self, obs_enc, chunk_position):
        obs = self.obs_encoder(obs_enc)
        position = self.position_encoder(chunk_position.reshape(-1, 1).float())
        return obs, position

    def _encode_actions(self, actions, action_mask, prefix_length):
        valid = action_mask.float().unsqueeze(-1)
        x = self.action_in(self.action_norm(actions)) * valid
        x = x.transpose(1, 2)
        for block in self.temporal:
            x = block(x) * valid.transpose(1, 2)
        x = x.transpose(1, 2)
        horizon = actions.shape[-2]
        indices = torch.arange(horizon, device=actions.device).reshape(1, -1)
        if not torch.is_tensor(prefix_length):
            prefix_length = torch.full((actions.shape[0],), int(prefix_length), device=actions.device)
        prefix_length = prefix_length.reshape(-1, 1)
        prefix_mask = action_mask & (indices < prefix_length)
        future_mask = action_mask & (indices >= prefix_length)
        return self.prefix_pool(x, prefix_mask), self.future_pool(x, future_mask)

    def score_candidates(self, context, actions, action_mask, prefix_length=10):
        obs, position = context
        if actions.ndim == 3:
            actions, action_mask = actions[:, None], action_mask[:, None]
        batch, candidates, horizon, adim = actions.shape
        flat_actions = actions.reshape(batch * candidates, horizon, adim)
        flat_mask = action_mask.reshape(batch * candidates, horizon).bool()
        if torch.is_tensor(prefix_length):
            prefix_length = prefix_length.reshape(-1)
            if prefix_length.numel() == batch:
                prefix_length = prefix_length[:, None].expand(batch, candidates).reshape(-1)
        prefix, future = self._encode_actions(flat_actions, flat_mask, prefix_length)
        obs_rep = obs[:, None].expand(batch, candidates, -1).reshape(batch * candidates, -1)
        pos_rep = position[:, None].expand(batch, candidates, -1).reshape(batch * candidates, -1)
        gamma, beta = self.film(obs_rep).chunk(2, dim=-1)
        action_features = torch.cat([prefix, future], dim=-1)
        action_features = action_features * (1.0 + gamma) + beta
        joint = self.joint_head(torch.cat([obs_rep, pos_rep, action_features], dim=-1))
        return joint.reshape(batch, candidates)

    def forward(self, obs_enc, actions, action_mask, chunk_position, prefix_length=10):
        context = self.encode_context(obs_enc, chunk_position)
        state = self.state_head(torch.cat(context, dim=-1)).squeeze(-1)
        joint = self.score_candidates(context, actions, action_mask, prefix_length).squeeze(-1)
        return VerifierOutput(state_logit=state, joint_logit=joint)
