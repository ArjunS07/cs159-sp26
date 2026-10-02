"""Flow-step PCP pilot on SmolVLA's actual hooked Euler/P&P sampler.

The terminal-action scalar critic is extrapolated to an intermediate clean
estimate. This tests the correction mechanism, not a validated policy claim.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from . import smolvla_tree_collection as tree
from .pnp import PnPRecorder, run_probe
from .sampler import _temp_strategy
from .smolvla_pcp_pilot import run_pcp_pilot, root_score
from .tap import BatchedRolloutTap

EXPERIMENT = "smolvla-libero-q10-flow-step-pcp-pilot-v1"
CORRECTION_STEP = 3  # actual zero-based hook: s = 1 - 3/10 = .7


def _tensor_digest(value):
    array = value.detach().float().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def apply_flow_q_correction(probe, score, valid, *, mode, action_rms, random_seed, correction_mode="rms"):
    """Nudge the probe's clean estimate, re-noise at its s using shared last_eps.

    K=1 is required: mean and last clean estimates then coincide, making the
    zero correction a true no-op of the existing last-estimate P&P baseline.
    """
    if probe.a_hats.shape[0] != 1 or probe.z_hat_full.shape[0] != 1:
        raise ValueError("matched scalar flow correction requires a singleton K=1 probe")
    if mode not in ("zero", "q_plus", "q_minus", "random") or not np.isfinite(action_rms) or action_rms < 0 or correction_mode not in ("rms", "lambda"):
        raise ValueError("invalid correction mode or RMS")
    full = probe.z_hat_full.detach()
    if (full.ndim != 3 or full.shape[1] < 10 or full.shape[2] < 7 or not 0 < probe.s < 1
            or not torch.isfinite(full).all() or not torch.isfinite(probe.x_acc).all()
            or not torch.isfinite(probe.last_eps).all()):
        raise ValueError("invalid clean estimate/noise level")
    valid = torch.as_tensor(valid, dtype=torch.bool, device=full.device)
    if valid.shape != (10,) or not valid.any():
        raise ValueError("need a fixed nonempty preaction Q10 mask")
    original = full[:, :10, :7].clone().requires_grad_(True)
    with torch.enable_grad():
        q = score(original)
        gradient = torch.autograd.grad(q.sum(), original)[0].detach()
    gradient *= valid[None, :, None]
    rms = gradient[:, valid].square().mean().sqrt()
    if not torch.isfinite(q).all() or not torch.isfinite(gradient).all() or not torch.isfinite(rms):
        raise ValueError("nonfinite scalar Q/gradient")
    telemetry = {"method": "flow_step_clean_estimate_q_correction",
                 "s": float(probe.s), "probe_k": 1, "control": mode,
                 "q_before": float(q.detach().reshape(-1)[0]),
                 "gradient_rms": float(rms), "clean_estimate_sha256": _tensor_digest(full),
                 "shared_last_eps_sha256": _tensor_digest(probe.last_eps),
                 "correction_mode": correction_mode,
                 "requested_correction_strength": float(action_rms),
                 "preaction_valid_mask": valid.cpu().tolist(), "random_seed": int(random_seed)}
    if mode == "zero" or action_rms == 0:
        telemetry.update(q_after=telemetry["q_before"], realized_clean_action_rms=0.,
                         realized_latent_action_rms=0., zero_correction_exact_noop=True)
        return probe.x_acc, telemetry
    if float(rms) <= 1e-12:
        raise ValueError("flow clean-estimate gradient is zero; correction refused")
    if mode == "random":
        generator = torch.Generator(device=full.device).manual_seed(int(random_seed))
        unit = torch.empty_like(gradient).normal_(generator=generator) * valid[None, :, None]
        unit /= unit[:, valid].square().mean().sqrt()
    else:
        unit = gradient / rms * (1 if mode == "q_plus" else -1)
    radius = action_rms if correction_mode == "rms" else action_rms * float(rms)
    corrected = full.clone()
    corrected[:, :10, :7] += radius * unit
    # Full clean tail and padded dimensions are preserved, then all dimensions
    # are re-noised consistently. Never mix already-noisy x_acc into a clean tail.
    delta_full = corrected - full
    # Algebraically (1-s)*corrected+s*last_eps. The incremental form keeps
    # every untouched padded/tail latent bit-identical to the P&P reference.
    updated = probe.x_acc + (1.0 - probe.s) * delta_full
    if not torch.isfinite(updated).all():
        raise ValueError("nonfinite corrected flow state")
    with torch.no_grad():
        after = score(corrected[:, :10, :7])
    delta = corrected[:, :10, :7] - full[:, :10, :7]
    telemetry.update(q_after=float(after.reshape(-1)[0]),
        effective_lambda=float(radius / float(rms)),
        injected_clean_tail_max_abs=float(delta_full[:, 10:].abs().max()),
        realized_clean_action_rms=float(delta[:, valid].square().mean().sqrt()),
        realized_latent_action_rms=float((updated - probe.x_acc)[:, :10, :7][:, valid].square().mean().sqrt()),
        zero_correction_exact_noop=False)
    return updated, telemetry


class FlowPCPTap(BatchedRolloutTap):
    """Keep existing P&P, replacing the chosen K=1 step's feedback with PCP."""
    def __init__(self, *args, score, valid, mode, action_rms, random_seed, correction_step, correction_mode="rms", **kwargs):
        super().__init__(*args, **kwargs)
        self.score, self.valid = score, valid
        self.mode, self.action_rms, self.random_seed = mode, action_rms, random_seed
        self.correction_step, self.correction_mode = correction_step, correction_mode
        self.correction_records = []
        if correction_step not in (2, 3) or self.config.probe_k(correction_step) != 1:
            raise ValueError("choose existing zero-based K=1 P&P step 2 or 3")

    def step(self, x_t, s, vf, ctx):
        if ctx.step != self.correction_step:
            return super().step(x_t, s, vf, ctx)
        if abs(float(s) - (1 - self.correction_step / ctx.num_steps)) > 1e-6:
            raise ValueError("sampler noise level differs from configured correction hook")
        probe = run_probe(x_t, s, vf, k=1, adim=self.adim, generators=self.generators)
        if self._pending_variable_probe is not None:
            raise RuntimeError("pending probe was not finalized")
        self._pending_variable_probe = (probe, int(ctx.step))
        updated, telemetry = apply_flow_q_correction(probe, self.score, self.valid,
            mode=self.mode, action_rms=self.action_rms, random_seed=self.random_seed,
            correction_mode=self.correction_mode)
        telemetry.update(euler_step=int(ctx.step), integration_steps=int(ctx.num_steps))
        self.correction_records.append(telemetry)
        return updated


def generate_flow_controls(*, policy, preprocess, item, bundle, source, row, critic,
                           valid, random_seed, correction_step=CORRECTION_STEP, correction_mode="rms",
                           task_description=None):
    """Generate complete chunks through real sampler hooks; never edit final chunks."""
    device = next(policy.model.parameters()).device
    critic_device = next(critic.parameters()).device
    # The source observation helper needs task text; recover it from the episode
    # manifest instead of inventing or reconstructing a different prompt.
    if task_description is None:
        from .smolvla_tree_source_experiment import prepare_smolvla_tree_source_episodes
        ep = next(e for e in prepare_smolvla_tree_source_episodes() if
                  (e["suite"], e["task_idx"], e["ep_idx"]) ==
                  (item["suite"], item["task_idx"], item["episode_idx"]))
        task_description = ep["task_desc"]
    batch = preprocess(tree._source_policy_observation(bundle["arrays"], source["boundary_index"], task_description))
    noise = tree._draw_chunk_noise(policy, device, int(source["noise_seed"]))
    generator = tree._source_perturb_generator(policy, device,
        perturb_seed=int(np.asarray(bundle["arrays"]["perturb_seed"])), completed_chunks=int(item["chunk_idx"]))
    cfg = tree._pnp_config()
    model = policy.model
    previous_steps, previous_position = model._pnp.num_steps, model._pnp.chunk_pos
    model._pnp.num_steps = tree.SMOLVLA_TREE_INTEGRATION_STEPS
    estimated_chunks = max(1, round(item["source_max_steps"] / int(policy.config.chunk_size)))
    model._pnp.chunk_pos = [min(item["chunk_idx"] / estimated_chunks, 1.)]
    score = lambda candidate: root_score(critic, row, candidate.to(critic_device))
    extras = {}
    reference_eps = reference_clean = reference_rng = None
    def generate(tap):
        with _temp_strategy(model, tap), torch.no_grad():
            return policy.predict_action_chunk(batch, noise=noise.clone()).detach()
    try:
        baseline_rec = PnPRecorder(); baseline_rec.new_episode()
        baseline_tap = BatchedRolloutTap(cfg, [baseline_rec], [tree._clone_generator(generator, device)], device, model._pnp.action_dim)
        baseline = generate(baseline_tap)
        reference_rng = baseline_tap.generators[0].get_state()
        controls = [("stock_replay", "zero", 0.)] + [
            (f"{mode}_{label}", mode, radius) for label, radius in (("small", .02), ("large", .06))
            for mode in ("q_plus", "q_minus", "random")]
        for name, mode, radius in controls:
            rec = PnPRecorder(); rec.new_episode()
            tap = FlowPCPTap(cfg, [rec], [tree._clone_generator(generator, device)], device,
                model._pnp.action_dim, score=score, valid=valid, mode=mode, action_rms=radius,
                random_seed=random_seed, correction_step=correction_step, correction_mode=correction_mode)
            chunk = generate(tap)
            if len(tap.correction_records) != 1:
                raise RuntimeError("flow correction hook did not fire exactly once")
            telemetry = tap.correction_records[0]
            if mode == "zero":
                if not torch.equal(chunk, baseline):
                    raise RuntimeError("zero correction changed baseline P&P final actions")
                reference_eps = telemetry["shared_last_eps_sha256"]
                reference_clean = telemetry["clean_estimate_sha256"]
            if (telemetry["shared_last_eps_sha256"] != reference_eps or
                    telemetry["clean_estimate_sha256"] != reference_clean or
                    not torch.equal(tap.generators[0].get_state(), reference_rng)):
                raise RuntimeError("controls do not share pre-correction clean state/probe noise/RNG position")
            with torch.no_grad():
                final_q = float(score(chunk[:, :10, :7]).reshape(-1)[0])
                zero_final_q = float(score(baseline[:, :10, :7]).reshape(-1)[0])
            telemetry.update(final_chunk_q=final_q, zero_final_chunk_q=zero_final_q,
                             root_only_intervention=True,
                             initial_noise_sha256=_tensor_digest(noise),
                             baseline_final_chunk_sha256=_tensor_digest(baseline),
                             final_chunk_sha256=_tensor_digest(chunk),
                             baseline_zero_final_actions_identical=True,
                             final_first10_rms_vs_zero=float((chunk[:, :10, :7] - baseline[:, :10, :7])[:, valid].square().mean().sqrt()),
                             final_full_chunk_rms_vs_zero=float((chunk - baseline).square().mean().sqrt()),
                             zero_vs_historical_first10_rms=float(np.sqrt(np.square(
                                 (baseline[0, :10, :7].float().cpu().numpy() - np.asarray(source["policy_chunk"])[:10, :7])[valid]).mean())),
                             critic_domain="terminal_action_Q_extrapolated_to_flow_clean_estimate")
            extras[name] = (chunk[0].float().cpu().numpy(), telemetry)
    finally:
        model._pnp.num_steps, model._pnp.chunk_pos = previous_steps, previous_position
    return extras


def run_flow_pcp_pilot(*, correction_step=CORRECTION_STEP, correction_mode="rms", **kwargs):
    if correction_step not in (2, 3):
        raise ValueError("correction_step must be zero-based existing K=1 P&P hook 2 or 3")
    if correction_mode not in ("rms", "lambda"):
        raise ValueError("correction_mode must be rms or lambda")
    def propose(**arguments):
        return generate_flow_controls(**arguments, correction_step=correction_step, correction_mode=correction_mode)
    return run_pcp_pilot(**kwargs, _proposal_generator=propose, _experiment_base=EXPERIMENT,
        _method_config={"method": "flow_step_clean_estimate_q_correction",
            "correction_step": correction_step, "correction_mode": correction_mode,
            "scales_rms": [.02, .06] if correction_mode == "rms" else None,
            "lambda_values": [.02, .06] if correction_mode == "lambda" else None,
            "root_only_intervention": True, "correction_noise_level": 1 - correction_step / 10,
            "integration_steps": 10, "pnp_steps": [1, 2, 3], "pnp_k_by_step": [3, 1, 1],
            "correction_probe_k": 1, "gate": "always_on_at_predeclared_step",
            "critic_domain": "terminal_action_Q_extrapolated_to_flow_clean_estimate"})
