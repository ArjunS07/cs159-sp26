"""Matched anchored action-effect critics trained with ordinary MC or TD targets."""
import copy

import torch
from torch import nn

from .qplanning_critic.model import pool_prefix_tokens


class AnchoredCritic(nn.Module):
    """Frozen reference prediction plus a nonlinear, exactly centered residual.

    Both action encoders receive the same frozen stock decoder readout. References
    are always detached; the anchor residual remains differentiable in parameters.
    No outcome or observed termination is an input.
    """
    def __init__(self, base, family, width=128, delta_scale=100.0):
        super().__init__()
        if family not in ("transformer", "temporal_cnn"):
            raise ValueError("unknown residual action encoder")
        if base.config.action_dim != 7 or base.config.action_horizon != 10:
            raise ValueError("anchored critic requires seven-dimensional Q10 inputs")
        if base.config.dropout != 0:
            raise ValueError("common baseline must have dropout zero")
        self.base = copy.deepcopy(base).requires_grad_(False).eval()
        self.family, self.width, self.delta_scale = family, width, float(delta_scale)
        self.input_projection = nn.Linear(14, width)
        self.positions = nn.Parameter(torch.randn(10, width) * .02)
        self.context_projection = nn.Linear(base.config.width, width)
        # Initialize these common layers before architecture-specific layers.
        self.readout = nn.Sequential(nn.Linear(11 * width, width), nn.GELU(),
                                     nn.Linear(width, 1, bias=False))
        nn.init.zeros_(self.readout[-1].weight)
        if family == "transformer":
            layer = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            self.encoder = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1)
                                          for _ in range(4)])
        self.output_norm = nn.LayerNorm(width)

    @property
    def action_mean(self):
        return self.base.action_mean

    @property
    def action_std(self):
        return self.base.action_std

    def architecture_config(self):
        return {"model_family": "anchored_" + self.family,
                "base_architecture": self.base.architecture_config(),
                "width": self.width, "delta_scale": self.delta_scale,
                "encoder_layers": 1 if self.family == "transformer" else 4,
                "kernel_size": None if self.family == "transformer" else 3,
                "context": "frozen_original_stock_decoder_readout",
                "dropout": 0, "baseline_frozen": True}

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def reference_features(self, prefix, prefix_valid, robot, proprio, reference, valid):
        """Exactly reproduce the original scalar baseline's stock readout/logit."""
        base = self.base
        reference = reference.detach()
        prefix, prefix_valid = pool_prefix_tokens(prefix, prefix_valid, base.config.prefix_pool_tokens)
        memory = torch.cat([base.prefix_projection(prefix.float()),
                            base.robot_projection(robot.float())[:, None],
                            base.proprio_projection(proprio.float())[:, None]], dim=1)
        memory_valid = torch.cat([prefix_valid.bool(), torch.ones((len(prefix), 2),
                                  device=prefix.device, dtype=torch.bool)], dim=1)
        z = (reference.float() - base.action_mean) / base.action_std
        action_tokens = base.action_projection(z) + base.action_positions[None]
        target = torch.cat([base.rl_token.expand(len(prefix), -1, -1), action_tokens], dim=1)
        ignored = torch.cat([torch.zeros((len(prefix), 1), device=prefix.device,
                                        dtype=torch.bool), ~valid.bool()], dim=1)
        features = base.decoder(target, memory, tgt_key_padding_mask=ignored,
                                memory_key_padding_mask=~memory_valid)[:, 0]
        return features.detach(), base.value_head(features).detach()

    def _h(self, features, reference, action, valid):
        mask = valid.bool()[..., None]
        ref_z = (reference - self.action_mean) / self.action_std
        delta = self.delta_scale * (action - reference) / self.action_std
        x = self.input_projection(torch.cat([ref_z, delta], dim=-1) * mask)
        context = self.context_projection(features)
        x = (x + self.positions[None] + context[:, None]) * mask
        if self.family == "transformer":
            x = self.encoder(x, src_key_padding_mask=~valid.bool()) * mask
        else:
            for convolution in self.encoder:
                x = (x + torch.nn.functional.gelu(convolution(x.transpose(1, 2)).transpose(1, 2))) * mask
        x = self.output_norm(x) * mask
        return self.readout(torch.cat([x.flatten(1), context], dim=-1))

    def logits_from_features(self, features, baseline, action, valid, reference):
        if action.shape[-2:] != (10, 7) or reference.shape != action.shape or valid.shape != action.shape[:2]:
            raise ValueError("candidate, reference and known-budget mask shapes disagree")
        if not valid.bool().any(1).all():
            raise ValueError("each chunk requires at least one valid action")
        reference = reference.detach()
        # h(anchor) MUST retain its parameter gradient, including on stock rows.
        residual = self._h(features.detach(), reference, action, valid) - self._h(
            features.detach(), reference, reference, valid)
        return baseline.detach() + residual

    def forward(self, prefix, prefix_valid, robot, proprio, action, action_valid, reference):
        features, baseline = self.reference_features(prefix, prefix_valid, robot, proprio,
                                                     reference, action_valid)
        return self.logits_from_features(features, baseline, action, action_valid, reference)

    def expected_value(self, *args, **kwargs):
        return self(*args, **kwargs).squeeze(-1).sigmoid()
