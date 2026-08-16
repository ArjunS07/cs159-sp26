"""RolloutTap — the ONE spec-driven sampler strategy (replaces the three strategy classes).

Built from a `RolloutConfig`, the tap is what the unified Euler loop in `sampler.py` yields to
at each selected step. It runs the probe ONCE, feeds whichever sinks are enabled, then applies
the configured action. This is where the action / probe / sinks decomposition is assembled:

    .selected(step, s)  -> step in probe.steps
    .step(x_t, s, vf, ctx) -> run_probe -> feed sinks -> apply action -> next x_t
    .invasive           -> action changes inference (refine/correction) vs measure-only
    .finish(ctx)        -> flush the per-chunk uncertainty record + pcp features

The sampler's strategy interface is unchanged, so only what BUILDS the tap lives here.
"""
from __future__ import annotations

import torch

from .config import RolloutConfig
from .pnp import (run_probe, apply_refine, apply_fractional_refine, PnPRecorder,
                  temporal_decay_weights, temporal_prefix_weights)
from .pcp import apply_correct, CorrectTelemetry


class RolloutTap:
    """Assemble action / probe / sinks for one rollout. `recorder` collects per-chunk
    uncertainty; `_pcp_chunks` collects per-chunk (obs_enc, z_hat) features when that sink is on.
    """

    def __init__(self, config: RolloutConfig, recorder: PnPRecorder, device, adim: int):
        self.config = config
        self.recorder = recorder
        self.device = device
        self.adim = adim
        self.records_uncertainty = config.records_uncertainty
        self.save_ahats = config.save_ahats
        self.save_pcp = config.save_pcp_features
        self._correcting = config.correction_lambda is not None
        self._refine_considered = 0
        self._refine_applied = 0
        self._chunk_refine_applied = 0
        self._chunk_idx = -1
        # correction telemetry (only when the action is a PCP correction)
        self._corr = CorrectTelemetry() if self._correcting else None
        # pcp-features accumulation (only when save_pcp_features)
        self._pcp_chunks: list = []
        self._pcp_buf: list = []

    # ── sampler-facing interface ────────────────────────────────────────────
    @property
    def invasive(self) -> bool:
        """Whether the action changes inference (drives the sampler's measure-only path)."""
        return self.config.refine or self._correcting

    @property
    def needs_baseline_fallback(self) -> bool:
        """Whether a no-fire chunk must return the exact stock sampler output."""
        return self.config.refine and (
            self.config.refine_threshold is not None
            or self.config.refine_start_chunk is not None)

    def begin_chunk(self) -> None:
        self._chunk_idx += 1
        self._chunk_refine_applied = 0

    @property
    def chunk_intervened(self) -> bool:
        return self._chunk_refine_applied > 0

    def selected(self, step: int, s: float) -> bool:
        return self.config.has_probe and self.config.probe_selected(step, s)

    def step(self, x_t, s, vf, ctx):
        cfg = self.config
        temporal_weights = None
        if cfg.refine_prefix_only:
            temporal_weights = temporal_prefix_weights(
                x_t.shape[-2], int(cfg.n_action_steps),
                device=x_t.device, dtype=x_t.dtype)
        elif cfg.refine_tail_decay_end is not None:
            temporal_weights = temporal_decay_weights(
                x_t.shape[-2], int(cfg.n_action_steps), int(cfg.refine_tail_decay_end),
                device=x_t.device, dtype=x_t.dtype)
        if cfg.refine_inner_strength is not None:
            strength = float(cfg.refine_inner_strength)
            temporal_weights = (
                torch.full((x_t.shape[-2],), strength, device=x_t.device, dtype=x_t.dtype)
                if temporal_weights is None else temporal_weights * strength)
        pr = run_probe(x_t, s, vf, k=cfg.pnp_k, adim=cfg.action_dim,
                       compute_multimodal=cfg.compute_multimodal,
                       suffix_probe_samples=cfg.suffix_probe_samples,
                       prefix_horizon=(cfg.n_action_steps if cfg.suffix_probe_samples else None),
                       temporal_update_weights=temporal_weights)

        # sink: per-step uncertainty (+ optional a_hats geometry stack)
        if self.records_uncertainty or cfg.compute_multimodal:
            rec = dict(pr.rec)
            rec["step"] = ctx.step
            if self.save_ahats:
                rec["a_hats"] = pr.a_hats.detach().float().cpu().numpy()
            ctx.records.append(rec)

        # sink: pcp features (mean z_hat at this step; obs_enc flushed per-chunk in finish)
        if self.save_pcp:
            self._pcp_buf.append({
                "step_idx": ctx.step, "s": float(s),
                "z_hat": pr.z_hat_full[0, :, :self.adim].detach().float().cpu().numpy(),
            })

        # action: at most one feedback path (measure-only when no action is set)
        if cfg.refine:
            # Delayed refinement deliberately preserves the exact stock sampler output before
            # this zero-based prediction-chunk index. The sampler's baseline fallback guarantees
            # those early chunks are not merely an approximation from the instrumented loop.
            if (cfg.refine_start_chunk is not None
                    and self._chunk_idx < cfg.refine_start_chunk):
                return x_t
            self._refine_considered += 1
            if cfg.refine_threshold is not None:
                gate_score = float(pr.rec["u_mean"])
                if cfg.refine_uncertainty_horizon is not None:
                    horizon = int(cfg.refine_uncertainty_horizon)
                    if horizon > len(pr.u_time):
                        raise ValueError(
                            f"refine_uncertainty_horizon={horizon} exceeds chunk length "
                            f"{len(pr.u_time)}")
                    gate_score = float(pr.u_time[:horizon].mean())
                if gate_score < float(cfg.refine_threshold):
                    return x_t
            self._refine_applied += 1
            self._chunk_refine_applied += 1
            if cfg.refine_horizon_m is not None:
                return apply_fractional_refine(
                    pr, x_t, cfg.refine_average,
                    horizon_m=cfg.refine_horizon_m,
                    num_steps=ctx.num_steps, step=ctx.step)
            return apply_refine(pr, cfg.refine_average)
        if self._correcting:
            return apply_correct(pr, ctx, cfg.correction_lambda, cfg.q_gate,
                                 cfg.q_model, cfg.q_scaler, self.adim, self._corr)
        return x_t

    def finish(self, ctx):
        if self.records_uncertainty or self.config.compute_multimodal:
            self.recorder.log_chunk({"num_steps": ctx.num_steps, "steps": ctx.records})
        if self.save_pcp:
            self._pcp_chunks.append({
                "chunk_idx": len(self._pcp_chunks),
                "obs_enc": ctx.obs_enc.detach().float().cpu().numpy(),
                "chunk_pos": ctx.chunk_pos,
                "steps": list(self._pcp_buf),
            })
            self._pcp_buf = []

    # ── result extraction (read by run_episode after the episode) ───────────
    @property
    def pcp_chunks(self):
        return self._pcp_chunks

    @property
    def pcp_telemetry(self):
        return self._corr.telemetry() if self._corr is not None else None

    @property
    def refinement_gate_telemetry(self):
        if not self.config.refine or self.config.refine_threshold is None:
            return None
        return {
            "n_corrections_applied": self._refine_applied,
            "gate_fire_rate": self._refine_applied / max(self._refine_considered, 1),
        }
