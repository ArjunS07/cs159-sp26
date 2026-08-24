"""Guarded live PCP-search scoring and Q-gradient correction adapter."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PCPSearchAdapterConfig


@dataclass
class PCPSearchTelemetry:
    q_min: list[float]
    q_spread: list[float]
    gradient_norm: list[float]
    uncertainty: list[float | None]


class PCPSearchAdapter:
    """Pure tensor adapter. Online action mutation requires explicit opt-in."""
    def __init__(self, critic, *, config: PCPSearchAdapterConfig | None = None):
        self.critic = critic.eval()
        self.config = config or PCPSearchAdapterConfig()

    def _q(self, prefix_embeddings, prefix_pad_mask, robot_state, proprio, action):
        return self.critic.q_values(prefix_embeddings, prefix_pad_mask, robot_state, proprio, action)

    def score_candidates(self, prefix_embeddings, prefix_pad_mask, robot_state, proprio,
                         candidates: torch.Tensor, *, uncertainty: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        if candidates.ndim == 3:
            candidates = candidates[:, None]
        batch, count = candidates.shape[:2]
        tiled = lambda value: value[:, None].expand(batch, count, *value.shape[1:]).reshape(batch * count, *value.shape[1:])
        q = self._q(tiled(prefix_embeddings), tiled(prefix_pad_mask), tiled(robot_state), tiled(proprio),
                    candidates.reshape(batch * count, *candidates.shape[2:])).reshape(2, batch, count)
        minimum = q.amin(0)
        return minimum, {"q_min": minimum.detach(), "q_spread": (q[0] - q[1]).abs().detach(),
                          "uncertainty": uncertainty.detach() if uncertainty is not None else None}

    def corrected_action(self, prefix_embeddings, prefix_pad_mask, robot_state, proprio,
                         action: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.config.mode != "online_enabled":
            raise RuntimeError("PCP-search adapter is offline_only; explicitly opt in before action mutation")
        candidate = action.detach().clone().requires_grad_(True)
        q = self._q(prefix_embeddings, prefix_pad_mask, robot_state, proprio, candidate).amin(0).sum()
        gradient, = torch.autograd.grad(q, candidate)
        norm = gradient.flatten(1).norm(dim=1, keepdim=True).clamp_min(self.config.gradient_epsilon)
        corrected = candidate + self.config.gradient_step * gradient / norm[:, None]
        return corrected.detach(), {"q": q.detach(), "gradient_norm": norm.squeeze(1).detach()}
