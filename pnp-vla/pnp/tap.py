"""RolloutTap — the ONE spec-driven sampler strategy (replaces the three strategy classes).

Built from a `RolloutConfig`, the tap is what the unified Euler loop in `sampler.py` yields to
at each selected step. It runs the probe ONCE, feeds whichever sinks are enabled, then applies
the configured action. This is where the action / probe / sinks decomposition is assembled:

    .selected(step, s)  -> step in probe.steps
    .step(x_t, s, vf, ctx) -> run_probe -> feed sinks -> apply action -> next x_t
    .invasive           -> action changes inference (Refine/Correct) vs measure-only
    .finish(ctx)        -> flush the per-chunk uncertainty record + pcp features

The sampler's strategy interface is unchanged, so only what BUILDS the tap lives here.
"""
from __future__ import annotations

from .config import RolloutConfig, Refine, Correct
from .pnp import run_probe, apply_refine, PnPRecorder
from .pcp import apply_correct, CorrectTelemetry


class RolloutTap:
    """Assemble action / probe / sinks for one rollout. `recorder` collects per-chunk
    uncertainty; `_pcp_chunks` collects per-chunk (obs_enc, z_hat) features when that sink is on.
    """

    def __init__(self, config: RolloutConfig, recorder: PnPRecorder, device, adim: int):
        self.config = config
        self.probe = config.probe
        self.action = config.action
        self.recorder = recorder
        self.device = device
        self.adim = adim
        self.records_uncertainty = config.records_uncertainty
        self.save_ahats = config.save_ahats
        self.save_pcp = config.save_pcp_features
        # correction telemetry (only when the action is Correct)
        self._corr = CorrectTelemetry() if isinstance(self.action, Correct) else None
        # pcp-features accumulation (only when save_pcp_features)
        self._pcp_chunks: list = []
        self._pcp_buf: list = []

    # ── sampler-facing interface ────────────────────────────────────────────
    @property
    def invasive(self) -> bool:
        """Whether the action changes inference (drives the sampler's measure-only path)."""
        return isinstance(self.action, (Refine, Correct))

    def selected(self, step: int, s: float) -> bool:
        return self.probe is not None and self.probe.selected(step, s)

    def step(self, x_t, s, vf, ctx):
        pr = run_probe(x_t, s, vf, self.probe)

        # sink: per-step uncertainty (+ optional a_hats geometry stack)
        if self.records_uncertainty or self.probe.compute_multimodal:
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

        # action: at most one feedback path (measure-only when action is None)
        if isinstance(self.action, Refine):
            return apply_refine(pr, self.action.average)
        if isinstance(self.action, Correct):
            return apply_correct(pr, ctx, self.action, self.adim, self._corr)
        return x_t

    def finish(self, ctx):
        if self.records_uncertainty or (self.probe and self.probe.compute_multimodal):
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
