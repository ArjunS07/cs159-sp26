"""ONE pi0.5 Euler-loop sampler with a per-step hook (Dedup #1).

Replaces the three near-identical `sample_actions` monkeypatches
(_sample_actions_pnp / _sample_actions_collect / _sample_actions_eval) with a single patched
method that replicates the pi0.5 flow-matching Euler loop and yields to a *strategy* at each
selected step. Strategies (RecordStrategy in pnp.py; Collect/CorrectStrategy in pcp.py) are
duck-typed: `.invasive: bool`, `.selected(step, s)`, `.step(x_t, s, vf, ctx) -> x_t`,
`.finish(ctx)`.

The initial noise is NOT drawn here — the rollout engine passes it explicitly via `noise=`
so the per-(episode,chunk) noise contract holds. Every velocity-field evaluation increments
`model._pnp_vf_evals` for compute accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .pnp import PNP_CONFIG, PNP_RECORDER  # noqa: F401  (PNP_RECORDER used by strategies)


@dataclass
class ChunkContext:
    """Per-chunk state handed to a strategy at each selected Euler step."""
    step: int = 0
    num_steps: int = 0
    obs_enc: Any = None          # (obs_dim,) mean-pooled prefix embedding
    chunk_pos: float = 0.0       # chunk_idx / est_chunks (set by the rollout engine)
    device: Any = None
    records: list = field(default_factory=list)


def install_patch(model) -> None:
    """Swap model.sample_actions for the hooked version (idempotent). pi0.5 only."""
    if not hasattr(model, "_orig_sample_actions"):
        model._orig_sample_actions = model.sample_actions
    model._pnp_strategy = None
    model._pnp_chunk_pos = 0.0
    model._pnp_vf_evals = 0
    import types
    model.sample_actions = types.MethodType(_sample_actions_hooked, model)


def set_strategy(model, strategy) -> None:
    model._pnp_strategy = strategy


@torch.no_grad()
def _sample_actions_hooked(self, images, img_masks, tokens, masks, noise=None,
                           num_steps=None, **kwargs):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    cfg = PNP_CONFIG
    strat = getattr(self, "_pnp_strategy", None)
    rtc = getattr(self, "_rtc_enabled", lambda: False)()
    if strat is None or not cfg.enabled or rtc:
        return self._orig_sample_actions(
            images, img_masks, tokens, masks, noise=noise, num_steps=num_steps, **kwargs)

    if num_steps is None:
        num_steps = self.config.num_inference_steps
    bsize = tokens.shape[0]
    device = tokens.device
    if noise is None:
        noise = self.sample_noise(
            (bsize, self.config.chunk_size, self.config.max_action_dim), device)

    # Non-invasive strategies (uncertainty / collect) execute vanilla actions; the custom
    # loop below only measures. Compute the returned action from the saved original sampler
    # on a clone of the SAME noise so it is exactly a vanilla rollout.
    measure_only = None
    if not strat.invasive:
        measure_only = self._orig_sample_actions(
            images, img_masks, tokens, masks, noise=noise.clone(), num_steps=num_steps, **kwargs
        ).clone()

    # ---- prefix / KV cache: replicated verbatim from the original sample_actions ----
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)

    ctx = ChunkContext(num_steps=num_steps, device=device,
                       obs_enc=prefix_embs.mean(dim=1)[0].detach(),
                       chunk_pos=float(getattr(self, "_pnp_chunk_pos", 0.0)))

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        s = 1.0 + step * dt
        time_tensor = torch.tensor(s, dtype=torch.float32, device=device).expand(bsize)

        def vfield(inp, _ts=time_tensor):
            self._pnp_vf_evals = getattr(self, "_pnp_vf_evals", 0) + 1
            return self.denoise_step(
                prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                x_t=inp, timestep=_ts)

        if strat.selected(step, s):
            ctx.step = step
            x_t = strat.step(x_t, s, vfield, ctx)
        x_t = x_t + dt * vfield(x_t)

    strat.finish(ctx)
    return measure_only if measure_only is not None else x_t


def measure_chunk_uncertainty(policy, batch, noise, probe_steps=(1, 2), num_iterations=3):
    """Run one measurement-only uncertainty pass; return (action_chunk, u_mean_scalar)."""
    saved = (PNP_CONFIG.enabled, PNP_CONFIG.mode, PNP_CONFIG.step_indices,
             PNP_CONFIG.num_iterations)
    from .pnp import RecordStrategy
    model = policy.model
    prev_strat = getattr(model, "_pnp_strategy", None)
    try:
        PNP_CONFIG.enabled = True
        PNP_CONFIG.mode = "uncertainty"
        PNP_CONFIG.step_indices = tuple(probe_steps)
        PNP_CONFIG.num_iterations = num_iterations
        model._pnp_strategy = RecordStrategy(PNP_CONFIG)
        PNP_RECORDER.new_episode()
        action = policy.predict_action_chunk(batch, noise=noise)
        ep = PNP_RECORDER._cur
        us = [st["u_mean"] for c in ep["chunks"] for st in c["steps"]] if ep else []
        u = float(sum(us) / len(us)) if us else 0.0
        return action, u
    finally:
        (PNP_CONFIG.enabled, PNP_CONFIG.mode, PNP_CONFIG.step_indices,
         PNP_CONFIG.num_iterations) = saved
        model._pnp_strategy = prev_strat
