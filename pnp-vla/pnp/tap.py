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

    def __init__(self, config: RolloutConfig, recorder: PnPRecorder, device, adim: int,
                 action_postprocess=None):
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
        self._gradient_records: list[dict] = []
        self._gradient_action_gate_records: list[dict] = []
        self._action_postprocess = action_postprocess
        # correction telemetry (only when the action is a PCP correction)
        self._corr = CorrectTelemetry() if self._correcting else None
        # pcp-features accumulation (only when save_pcp_features)
        self._pcp_chunks: list = []
        self._pcp_buf: list = []

    # ── sampler-facing interface ────────────────────────────────────────────
    @property
    def invasive(self) -> bool:
        """Whether the action changes inference (drives the sampler's measure-only path)."""
        return (self.config.refine or self._correcting
                or self.config.uncertainty_gradient_mode is not None)

    @property
    def needs_baseline_fallback(self) -> bool:
        """Whether a no-fire chunk must return the exact stock sampler output."""
        return bool(
            (self.config.refine and (
                self.config.refine_threshold is not None
                or self.config.refine_start_chunk is not None))
            or self.config.uncertainty_gradient_action_rms_max is not None)

    def begin_chunk(self) -> None:
        self._chunk_idx += 1
        self._chunk_refine_applied = 0

    @property
    def chunk_intervened(self) -> bool:
        return self._chunk_refine_applied > 0

    def _environment_action(self, action, horizon):
        """Decode each action exactly as the rollout queue does at environment.step."""
        if self._action_postprocess is None:
            return action[..., :horizon, :]
        decoded_actions = []
        for index in range(horizon):
            # Third-party processor steps may operate in-place. Never give them a view into
            # the live chunks: rejecting a candidate must return byte-for-byte stock actions.
            processor_input = action[..., index, :self.adim].detach().clone()
            decoded = self._action_postprocess(processor_input)
            if torch.is_tensor(decoded):
                decoded = decoded.to(device=action.device, dtype=action.dtype)
            else:
                decoded = torch.as_tensor(
                    decoded, device=action.device, dtype=action.dtype)
            decoded_actions.append(decoded)
        return torch.stack(decoded_actions, dim=-2)

    def finalize_action(self, baseline_action, candidate_action):
        """Choose between exact-stock and candidate after both chunks are fully decoded."""
        threshold = self.config.uncertainty_gradient_action_rms_max
        if threshold is None:
            return candidate_action if self.chunk_intervened else baseline_action

        horizon = min(
            int(self.config.n_action_steps), baseline_action.shape[-2], candidate_action.shape[-2])
        stock = self._environment_action(baseline_action, horizon)
        candidate = self._environment_action(candidate_action, horizon)
        motion_dim = min(6, self.adim, stock.shape[-1], candidate.shape[-1])
        if horizon < 1 or motion_dim < 1:
            raise ValueError("action displacement gate received an empty action prefix")
        delta = candidate[..., :horizon, :motion_dim] - stock[..., :horizon, :motion_dim]
        action_rms = float(torch.sqrt(torch.mean(delta.float().square())).item())
        first_action_l2 = float(torch.linalg.vector_norm(
            delta[..., 0, :].float(), dim=-1).mean().item())
        max_abs = float(delta.float().abs().max().item())
        gripper_disagreement = False
        if min(stock.shape[-1], candidate.shape[-1]) > 6:
            stock_gripper = stock[..., :horizon, 6]
            candidate_gripper = candidate[..., :horizon, 6]
            gripper_disagreement = bool(
                torch.any(torch.signbit(stock_gripper) != torch.signbit(candidate_gripper)).item())
        accepted = action_rms <= float(threshold)
        self._gradient_action_gate_records.append({
            "chunk_idx": self._chunk_idx,
            "action_rms": action_rms,
            "first_action_l2": first_action_l2,
            "max_abs_action_change": max_abs,
            "gripper_sign_disagreement": gripper_disagreement,
            "threshold": float(threshold),
            "accepted": accepted,
            "actions_compared": horizon,
            "motion_dimensions": motion_dim,
        })
        return candidate_action if accepted else baseline_action

    def selected(self, step: int, s: float) -> bool:
        return self.config.has_probe and self.config.probe_selected(step, s)

    def step(self, x_t, s, vf, ctx):
        cfg = self.config
        gradient_updated = None
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
        if cfg.uncertainty_gradient_mode is not None:
            from .uncertainty_critic_v2 import direct_latent_uncertainty_update
            random_seed = (int(torch.initial_seed())
                           ^ ((self._chunk_idx + 1) * 1_000_003)
                           ^ (int(ctx.step) * 9_176)) & ((1 << 63) - 1)
            gradient_updated, pr, telemetry = direct_latent_uncertainty_update(
                x_t, s, vf, k=cfg.pnp_k,
                horizon=cfg.uncertainty_gradient_horizon,
                step_size=cfg.uncertainty_gradient_step_size,
                mode=cfg.uncertainty_gradient_mode, random_seed=random_seed)
            telemetry.update({
                "chunk_idx": self._chunk_idx, "euler_step": int(ctx.step), "s": float(s)})
            self._gradient_records.append(telemetry)
        else:
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
        if gradient_updated is not None:
            return gradient_updated
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
    def uncertainty_gradient_telemetry(self):
        if self.config.uncertainty_gradient_mode is None:
            return None
        records = list(self._gradient_records)
        if not records:
            return {"records": [], "n_updates": 0}
        return {
            "records": records,
            "n_updates": len(records),
            "mean_pre_u20": sum(row["pre_u20"] for row in records) / len(records),
            "mean_post_u20": sum(row["post_u20"] for row in records) / len(records),
            "mean_delta_u20": sum(row["delta_u20"] for row in records) / len(records),
            "mean_update_rms": sum(row["update_rms"] for row in records) / len(records),
            **({
                "action_gate_records": list(self._gradient_action_gate_records),
                "n_action_gate_considered": len(self._gradient_action_gate_records),
                "n_action_gate_accepted": sum(
                    row["accepted"] for row in self._gradient_action_gate_records),
            } if self.config.uncertainty_gradient_action_rms_max is not None else {}),
        }

    @property
    def refinement_gate_telemetry(self):
        if not self.config.refine or self.config.refine_threshold is None:
            return None
        return {
            "n_corrections_applied": self._refine_applied,
            "gate_fire_rate": self._refine_applied / max(self._refine_considered, 1),
        }
