"""ONE pi0.5 Euler-loop sampler with a per-step hook (Dedup #1).

Replaces the three near-identical `sample_actions` monkeypatches
(_sample_actions_pnp / _sample_actions_collect / _sample_actions_eval) with a single patched
method that replicates the pi0.5 flow-matching Euler loop and yields to the ONE spec-driven
`RolloutTap` (tap.py) at each selected step. The tap is duck-typed:
`.invasive: bool`, `.selected(step, s)`, `.step(x_t, s, vf, ctx) -> x_t`, `.finish(ctx)`.

The initial noise is NOT drawn here — the rollout engine passes it explicitly via `noise=`
so the per-(episode,chunk) noise contract holds. Every velocity-field evaluation increments
`model._pnp.vf_evals` for compute accounting.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import ADIM


class _EncodingCache:
    """In-session LRU of pi0.5 prefix encodings, keyed by a content hash of the prefix inputs.

    Holds on-device tensors, so a cache hit skips `embed_prefix` (the SigLIP image tower) entirely.
    The win is the first chunk, which is identical across the paired methods at one `init_state`.
    Opt-in (OFF by default). The store's Storage-backed get/put_encoding is a separate, optional
    persistent tier — not used here (bf16 prefixes don't round-trip through numpy cheaply).
    """

    def __init__(self, size: int = 8, model_revision: str = ""):
        self.size = size
        self.model_revision = model_revision
        self._lru: "OrderedDict[str, tuple]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, *inputs) -> str:
        h = hashlib.sha1(self.model_revision.encode())

        def update(value) -> None:
            if torch.is_tensor(value):
                h.update(b"tensor")
                array = (value.detach().to("cpu", torch.float32)
                         if value.is_floating_point() else value.detach().cpu()).numpy()
                h.update(repr(array.shape).encode())
                h.update(array.tobytes())
                return
            if isinstance(value, (list, tuple)):
                h.update(f"{type(value).__name__}:{len(value)}".encode())
                for item in value:
                    update(item)
                return
            raise TypeError(
                f"encoding-cache inputs must be tensors or nested lists/tuples; "
                f"got {type(value).__name__}")

        for value in inputs:
            update(value)
        return h.hexdigest()

    def get(self, key: str):
        v = self._lru.get(key)
        if v is None:
            self.misses += 1
            return None
        self._lru.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key: str, value) -> None:
        self._lru[key] = value
        self._lru.move_to_end(key)
        while len(self._lru) > self.size:
            self._lru.popitem(last=False)


def enable_encoding_cache(model, size: int = 8, model_revision: str = "") -> None:
    """Turn on the in-session prefix-encoding LRU on a patched model (opt-in; OFF by default)."""
    model._pnp.enc_cache = _EncodingCache(size, model_revision)


@dataclass
class ChunkContext:
    """Per-chunk state handed to a strategy at each selected Euler step."""
    step: int = 0
    num_steps: int = 0
    obs_enc: Any = None          # (obs_dim,) mean-pooled prefix embedding
    chunk_pos: float = 0.0       # chunk_idx / est_chunks (set by the rollout engine)
    device: Any = None
    records: list = field(default_factory=list)


@dataclass
class _SamplerState:
    """Per-model sampler state stamped on the patched pi0.5 model by install_patch — one namespace
    instead of scattered _pnp_* attributes. install_patch guarantees it exists, so readers access
    it directly (no defensive getattr)."""
    strategy: object = None      # the active RolloutTap, or None = vanilla (delegate to orig)
    chunk_pos: float = 0.0       # chunk_idx / est_chunks (set by run_episode)
    vf_evals: int = 0            # velocity-field eval counter (compute accounting)
    num_steps: object = None     # per-call num_inference_steps override (extra_steps)
    enc_cache: object = None     # opt-in prefix-encoding LRU (see enable_encoding_cache)
    action_dim: int = ADIM       # real (un-padded) action dims (set by apply_pnp_patch)


def install_patch(model) -> None:
    """Swap model.sample_actions for the hooked version (idempotent). pi0.5 only."""
    if not hasattr(model, "_orig_sample_actions"):
        model._orig_sample_actions = model.sample_actions
    model._pnp = _SamplerState()
    import types
    model.sample_actions = types.MethodType(_sample_actions_hooked, model)


def set_strategy(model, strategy) -> None:
    model._pnp.strategy = strategy


@contextmanager
def _temp_strategy(model, strategy):
    """Install a strategy for the duration of a block, restoring the previous one afterward."""
    prev = model._pnp.strategy
    model._pnp.strategy = strategy
    try:
        yield
    finally:
        model._pnp.strategy = prev


@torch.no_grad()
def _sample_actions_hooked(self, images, img_masks, tokens, masks, noise=None,
                           num_steps=None, **kwargs):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    # Resolve the step count: explicit arg > per-call override (extra_steps) > model default.
    num_steps = num_steps or self._pnp.num_steps or self.config.num_inference_steps
    strat = self._pnp.strategy
    if strat is None:
        return self._orig_sample_actions(
            images, img_masks, tokens, masks, noise=noise, num_steps=num_steps, **kwargs)

    bsize = tokens.shape[0]
    device = tokens.device
    if noise is None:
        noise = self.sample_noise(
            (bsize, self.config.chunk_size, self.config.max_action_dim), device)

    # Non-invasive strategies (uncertainty / collect) execute vanilla actions; the custom
    # loop below only measures. Compute the returned action from the saved original sampler
    # on a clone of the SAME noise so it is exactly a vanilla rollout.
    baseline_action = None
    needs_baseline_fallback = bool(
        getattr(strat, "needs_baseline_fallback", False))
    if not strat.invasive or needs_baseline_fallback:
        baseline_action = self._orig_sample_actions(
            images, img_masks, tokens, masks, noise=noise.clone(), num_steps=num_steps, **kwargs
        ).clone()

    begin_chunk = getattr(strat, "begin_chunk", None)
    if begin_chunk is not None:
        begin_chunk()

    # ---- prefix / KV cache: replicated verbatim from the original sample_actions ----
    # (opt-in) reuse the prefix encoding across paired methods at the same obs — skips embed_prefix.
    cache = self._pnp.enc_cache
    ckey = cache.key(images, img_masks, tokens, masks) if cache is not None else None
    prefix = cache.get(ckey) if cache is not None else None
    if prefix is None:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks)
        if cache is not None:
            cache.put(ckey, (prefix_embs, prefix_pad_masks, prefix_att_masks))
    else:
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)

    ctx = ChunkContext(num_steps=num_steps, device=device,
                       obs_enc=prefix_embs.mean(dim=1)[0].detach(),
                       chunk_pos=float(self._pnp.chunk_pos))

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        s = 1.0 + step * dt
        time_tensor = torch.tensor(s, dtype=torch.float32, device=device).expand(bsize)

        def vfield(inp, _ts=time_tensor):
            self._pnp.vf_evals += 1
            return self.denoise_step(
                prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                x_t=inp, timestep=_ts)

        if strat.selected(step, s):
            ctx.step = step
            x_t = strat.step(x_t, s, vfield, ctx)
        x_t = x_t + dt * vfield(x_t)

    strat.finish(ctx)
    if not strat.invasive:
        return baseline_action
    if needs_baseline_fallback and not bool(getattr(strat, "chunk_intervened", False)):
        return baseline_action
    return x_t


def measure_chunk_uncertainty(policy, batch, noise, probe_steps=(1, 2), num_iterations=3,
                              uncertainty_horizon=None, return_details=False,
                              return_features=False):
    """Run one measurement-only probe pass and score a selectable action horizon.

    Historical callers receive ``(action_chunk, full_chunk_u)``. Diagnostic callers may
    request a leading action horizon and U10/U20/Ufull plus contraction summaries.
    """
    from .config import RolloutConfig
    from .pnp import PnPRecorder
    from .tap import RolloutTap
    model = policy.model
    cfg = RolloutConfig(
        pnp_steps=tuple(probe_steps), pnp_k=num_iterations,
        save_trajectory=False, save_pcp_features=bool(return_features),
        save_ahats=bool(return_features))
    rec = PnPRecorder(); rec.new_episode()
    tap = RolloutTap(cfg, rec, device=None, adim=model._pnp.action_dim)
    with _temp_strategy(model, tap):
        action = policy.predict_action_chunk(batch, noise=noise)
    records = [st for c in rec.current_chunks() for st in c["steps"]]
    if not records:
        details = {"u10": 0.0, "u20": 0.0, "u_full": 0.0,
                   "contraction10": 0.0, "contraction20": 0.0,
                   "contraction_full": 0.0}
        output = (action, 0.0, details) if return_details else (action, 0.0)
        if return_features:
            return (*output, {
                "obs_enc": np.empty((0,), dtype=np.float32),
                "step_indices": np.empty((0,), dtype=np.int64),
                "s": np.empty((0,), dtype=np.float32),
                "z_hat": np.empty((0, 0, model._pnp.action_dim), dtype=np.float32),
                "first_a_hat": np.empty(
                    (0, 0, model._pnp.action_dim), dtype=np.float32),
                "last_a_hat": np.empty(
                    (0, 0, model._pnp.action_dim), dtype=np.float32),
                "u_time": np.empty((0, 0), dtype=np.float32),
                "u_iter_time": np.empty((0, 0, 0), dtype=np.float32),
            })
        return output
    u_time = torch.as_tensor(
        np.stack([np.asarray(st["u_time"], dtype=float) for st in records]),
        dtype=torch.float32).mean(dim=0)

    def prefix_mean(horizon):
        width = len(u_time) if horizon is None else min(int(horizon), len(u_time))
        return float(u_time[:width].mean())

    contractions = {}
    for horizon, name in ((10, "contraction10"), (20, "contraction20"),
                          (None, "contraction_full")):
        values = []
        for record in records:
            profile = record.get("u_iter_time")
            if profile is None:
                continue
            profile = torch.as_tensor(profile, dtype=torch.float32)
            if profile.ndim != 2 or len(profile) < 2:
                continue
            width = profile.shape[1] if horizon is None else min(int(horizon), profile.shape[1])
            sequence = profile[:, :width].mean(dim=1)
            values.append(float(sequence[0] - sequence[-1]))
        contractions[name] = float(np.mean(values)) if values else 0.0
    details = {"u10": prefix_mean(10), "u20": prefix_mean(20),
               "u_full": prefix_mean(None), **contractions}
    if uncertainty_horizon is not None and int(uncertainty_horizon) > len(u_time):
        raise ValueError(
            f"uncertainty_horizon={uncertainty_horizon} exceeds chunk length {len(u_time)}")
    score = prefix_mean(uncertainty_horizon)
    output = (action, score, details) if return_details else (action, score)
    if return_features:
        if not tap.pcp_chunks or len(tap.pcp_chunks) != 1:
            raise RuntimeError("candidate feature capture expected exactly one PCP chunk")
        pcp = tap.pcp_chunks[0]
        pcp_steps = sorted(pcp["steps"], key=lambda item: int(item["step_idx"]))
        record_by_step = {int(record["step"]): record for record in records}
        ordered_records = [record_by_step[int(item["step_idx"])] for item in pcp_steps]
        features = {
            "obs_enc": np.asarray(pcp["obs_enc"], dtype=np.float32),
            "step_indices": np.asarray(
                [item["step_idx"] for item in pcp_steps], dtype=np.int64),
            "s": np.asarray([item["s"] for item in pcp_steps], dtype=np.float32),
            "z_hat": np.stack(
                [np.asarray(item["z_hat"], dtype=np.float32) for item in pcp_steps]),
            "first_a_hat": np.stack([
                np.asarray(item["a_hats"], dtype=np.float32)[0, 0]
                for item in ordered_records]),
            "last_a_hat": np.stack([
                np.asarray(item["a_hats"], dtype=np.float32)[-1, 0]
                for item in ordered_records]),
            "u_time": np.stack(
                [np.asarray(item["u_time"], dtype=np.float32)
                 for item in ordered_records]),
            "u_iter_time": np.stack(
                [np.asarray(item["u_iter_time"], dtype=np.float32)
                 for item in ordered_records]),
        }
        return (*output, features)
    return output
