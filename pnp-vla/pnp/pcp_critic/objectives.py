"""Shared Bellman, Cal-QL, and IQL losses for :mod:`pnp.pcp_critic`."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def expectile_loss(residual: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(residual >= 0, expectile, 1 - expectile)
    return (weight * residual.square()).mean()


def td_target(target_model, batch: dict[str, torch.Tensor], *, use_iql_value: bool = False) -> torch.Tensor:
    with torch.no_grad():
        if use_iql_value:
            next_value = target_model.state_value(
                batch["next_prefix"], batch["next_pad"], batch["next_robot"], batch["next_proprio"])
        else:
            next_value = target_model.q_values(
                batch["next_prefix"], batch["next_pad"], batch["next_robot"], batch["next_proprio"],
                batch["next_action"]).amin(0)
        return batch["reward"] + batch["discount"] * next_value


def local_and_broad_actions(action: torch.Tensor, *, n_local: int, n_broad: int,
                            local_std: float, generator=None) -> torch.Tensor:
    """Behavior-local and broad clipped actions, all in stored normalized action space."""
    candidates = [action]
    for _ in range(n_local):
        noise = torch.randn(action.shape, dtype=action.dtype, device=action.device, generator=generator)
        candidates.append((action + noise * local_std).clamp(-1, 1))
    for _ in range(n_broad):
        candidates.append(torch.empty_like(action).uniform_(-1, 1, generator=generator))
    return torch.stack(candidates, 1)


def calibrated_conservative_loss(model, batch: dict[str, torch.Tensor], *, n_local: int,
                                 n_broad: int, local_std: float, weight: float) -> torch.Tensor:
    candidates = local_and_broad_actions(batch["action"], n_local=n_local, n_broad=n_broad,
                                         local_std=local_std)
    count = candidates.shape[1]
    tokens = model.encode_state(batch["prefix"], batch["pad"], batch["robot"], batch["proprio"])
    tiled_tokens = tokens[:, None].expand(-1, count, -1, -1).reshape(-1, *tokens.shape[1:])
    values = model.q_values_from_tokens(tiled_tokens, candidates.flatten(0, 1)).amin(0).reshape(len(tokens), count)
    # Cal-QL: only discourage candidate estimates above the trajectory MC return.
    return weight * F.relu(torch.logsumexp(values, 1) - batch["mc_return"]).mean()
