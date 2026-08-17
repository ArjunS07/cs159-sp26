"""P&P core: recorder, isolated perturbation RNG, the probe, refinement, no-op contract test.

Design invariants (the whole point):
- The perturbation noise draws use a DEDICATED per-device torch.Generator (`_pnp_gen`) seeded
  via `_pnp_seed_perturb`. It NEVER touches the global RNG or the initial-noise stream that
  `rollout.py` owns. So a measure-only probe is a true no-op of vanilla, and refinement/PCP
  perturbations are reproducible and paired across methods.
- `run_probe` runs the K predict-and-perturb block ONCE and returns a structured `ProbeResult`;
  `apply_refine` / `pcp.apply_correct` derive the next state from it. The tap (tap.py) decides
  which sinks consume the result and which action feeds back — the probe itself is pure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import PERTURB_SEED_MASK, ADIM

# No module-level config/recorder globals: config travels with the strategy, and the
# recorder is created per run_episode call. Only the mechanical perturbation-generator state
# below is module-level (reseeded per episode).

# ─────────────────────────────────────────────────────────────────────────────
# Isolated perturbation generators (never advance the global / noise RNG).
# ─────────────────────────────────────────────────────────────────────────────
_PNP_GENS: dict[str, torch.Generator] = {}
_DIAGNOSTIC_GENS: dict[str, torch.Generator] = {}
_PNP_LAST_SEED = [0]
_DIAGNOSTIC_SEED_MASK = 0xD1A69057


def _pnp_gen(device) -> torch.Generator:
    key = str(device)
    g = _PNP_GENS.get(key)
    if g is None:
        g = torch.Generator(device=torch.device(device))
        g.manual_seed(int(_PNP_LAST_SEED[0]) ^ PERTURB_SEED_MASK)
        _PNP_GENS[key] = g
    return g


def _diagnostic_gen(device) -> torch.Generator:
    """Independent stream so suffix diagnostics never change the established P&P sequence."""
    key = str(device)
    g = _DIAGNOSTIC_GENS.get(key)
    if g is None:
        g = torch.Generator(device=torch.device(device))
        g.manual_seed(int(_PNP_LAST_SEED[0]) ^ _DIAGNOSTIC_SEED_MASK)
        _DIAGNOSTIC_GENS[key] = g
    return g


def _pnp_seed_perturb(seed: int) -> None:
    """Seed the dedicated perturbation stream (independent of the global RNG).

    Call once per episode so perturbations are reproducible and paired across methods.
    """
    _PNP_LAST_SEED[0] = int(seed)
    for g in _PNP_GENS.values():
        g.manual_seed(int(seed) ^ PERTURB_SEED_MASK)
    for g in _DIAGNOSTIC_GENS.values():
        g.manual_seed(int(seed) ^ _DIAGNOSTIC_SEED_MASK)


# ─────────────────────────────────────────────────────────────────────────────
# Multimodality helpers (numpy, cheap; run on the K stacked predictions)
# ─────────────────────────────────────────────────────────────────────────────
def _bc_1d(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient on a 1-D array of K samples. >0.555 ~ bimodal."""
    n = x.shape[0]
    if n < 4:
        return float("nan")
    d = x - x.mean()
    s = np.sqrt((d ** 2).mean()) + 1e-12
    g = (d ** 3).mean() / s ** 3                       # skewness
    k = (d ** 4).mean() / s ** 4 - 3.0                 # excess kurtosis
    return float((g ** 2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def _multimodal_stats(A0: np.ndarray):
    """A0: (K, chunk, adim). Returns (per-dim BC vec, PC1 variance fraction, BC of PC1)."""
    K = A0.shape[0]
    d = A0 - A0.mean(axis=0, keepdims=True)
    s2 = (d ** 2).mean(axis=0); sd = np.sqrt(s2) + 1e-12
    g = (d ** 3).mean(axis=0) / sd ** 3               # (chunk, adim) skew
    k = (d ** 4).mean(axis=0) / sd ** 4 - 3.0         # (chunk, adim) excess kurt
    bc = (g ** 2 + 1.0) / (k + 3.0 * (K - 1) ** 2 / ((K - 2) * (K - 3)))
    bc_vec = np.nanmean(bc, axis=0)                   # (adim,)
    F = A0.reshape(K, -1).astype(np.float64)
    F = F - F.mean(axis=0, keepdims=True)
    try:
        U, S, _ = np.linalg.svd(F, full_matrices=False)
        pc1_frac = float(S[0] ** 2 / (S ** 2).sum())
        bc_pc1 = _bc_1d(U[:, 0] * S[0])               # modality along dominant direction
    except np.linalg.LinAlgError:
        pc1_frac, bc_pc1 = float("nan"), float("nan")
    return bc_vec, pc1_frac, bc_pc1


def temporal_decay_weights(length: int, prefix: int, decay_end: int, *, device, dtype):
    """One through the executed prefix, linear decay through the near tail, then zero."""
    if not 0 < int(prefix) < int(decay_end) <= int(length):
        raise ValueError("expected 0 < prefix < decay_end <= action-chunk length")
    positions = torch.arange(int(length), device=device, dtype=dtype)
    weights = (float(decay_end) - positions) / float(decay_end - prefix)
    return weights.clamp_(0.0, 1.0)


def temporal_prefix_weights(length: int, prefix: int, *, device, dtype):
    """One on the executed prefix and zero on every discarded action position."""
    if not 0 < int(prefix) < int(length):
        raise ValueError("expected 0 < prefix < action-chunk length")
    weights = torch.zeros(int(length), device=device, dtype=dtype)
    weights[:int(prefix)] = 1.0
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# The shared K predict-and-perturb block — the PROBE (pure measurement).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ProbeResult:
    """Structured output of one K predict-and-perturb block at noise level s.

    Everything downstream (the uncertainty sink, apply_refine, pcp.apply_correct) is derived
    from this — the probe itself changes no state and makes no feedback decision.
    """
    a_hats: torch.Tensor      # (K, B, chunk, adim) — clean estimates, action-dim sliced
    z_hat_full: torch.Tensor  # (B, chunk, max_adim) — mean of the K FULL-width clean estimates
    x_acc: torch.Tensor       # re-noised state after the LAST iteration (full width)
    last_eps: torch.Tensor    # the last iteration's perturbation noise (for refine-average)
    u_time: torch.Tensor      # (chunk,) disagreement, averaged over K-pairs/batch/action dims
    s: float
    rec: dict                 # uncertainty record (u_mean/u_max/u_vec/a_std_vec/... + multimodal)


def run_probe(x_t, s, vfield, *, k: int, adim: int = ADIM,
              compute_multimodal: bool = False, suffix_probe_samples: int = 0,
              prefix_horizon: int | None = None,
              temporal_update_weights: torch.Tensor | None = None) -> ProbeResult:
    """Run K predict-and-perturb iterations at fixed noise level s and measure uncertainty.

        predict:  a_hat = x - s * v(x, s)
        perturb:  x'    = (1 - s) * a_hat + s * eps,   eps ~ N(0, I)  (isolated generator)

    Pure: no global state touched (perturbation noise comes from the dedicated generator).
    """
    if temporal_update_weights is not None:
        temporal_update_weights = torch.as_tensor(
            temporal_update_weights, device=x_t.device, dtype=x_t.dtype)
        if temporal_update_weights.ndim != 1 or len(temporal_update_weights) != x_t.shape[-2]:
            raise ValueError("temporal_update_weights must contain one value per action position")
        if not bool(((temporal_update_weights >= 0) & (temporal_update_weights <= 1)).all()):
            raise ValueError("temporal_update_weights must lie in [0, 1]")
        temporal_update_weights = temporal_update_weights.view(1, -1, 1)
    if suffix_probe_samples:
        if prefix_horizon is None or not 0 < int(prefix_horizon) < x_t.shape[-2]:
            raise ValueError("suffix diagnostics require a prefix inside the generated chunk")
        if isinstance(suffix_probe_samples, bool) or int(suffix_probe_samples) < 1:
            raise ValueError("suffix_probe_samples must be a positive integer")

    original_x = x_t
    x_acc = x_t
    a_hats, a_hats_full = [], []
    last_eps = None
    gen = _pnp_gen(x_acc.device)
    for _ in range(k):
        v = vfield(x_acc)
        a_hat = x_acc - s * v
        a_hats.append(a_hat[..., :adim])
        a_hats_full.append(a_hat)
        last_eps = torch.empty_like(x_acc).normal_(generator=gen)   # dedicated stream, not global
        proposal = (1.0 - s) * a_hat + s * last_eps
        x_acc = (proposal if temporal_update_weights is None else
                 x_acc + temporal_update_weights * (proposal - x_acc))

    A = torch.stack(a_hats, dim=0)                     # (K, B, chunk, adim)
    z_hat_full = torch.stack(a_hats_full, dim=0).mean(dim=0)   # (B, chunk, max_adim)
    if A.shape[0] >= 2:
        d_consecutive = (A[1:] - A[:-1]).abs()                 # (K-1, B, chunk, adim)
        u_consecutive = d_consecutive.mean(dim=0)              # (B, chunk, adim)
        a_std = A.std(dim=0)
    else:
        d_consecutive = torch.zeros_like(A[:1])
        u_consecutive = torch.zeros_like(A[0]); a_std = torch.zeros_like(A[0])
    u_time = u_consecutive.mean(dim=(0, 2))

    rec = {
        "s": float(s),
        "u_mean": float(u_consecutive.mean()),
        "u_max": float(u_consecutive.max()),
        "a_std_mean": float(a_std.mean()),
        "u_vec": u_consecutive.mean(dim=(0, 1)).detach().float().cpu().numpy(),
        "a_std_vec": a_std.mean(dim=(0, 1)).detach().float().cpu().numpy(),
        "a_mean_vec": A.mean(dim=(0, 1, 2)).detach().float().cpu().numpy(),
        "u_time": u_time.detach().float().cpu().numpy(),
        # Keep the perturbation-pair and action-position axes so contraction can be measured
        # separately on the executed prefix, near tail, and complete generated chunk.
        "u_iter_time": d_consecutive.mean(dim=(1, 3)).detach().float().cpu().numpy(),
        # Per-ITERATION disagreement, i.e. |a_hat_{i+1} - a_hat_i| for i in 0..K-2, kept instead
        # of collapsed into u_mean. This is what makes "does disagreement DECAY across the K
        # perturbations?" answerable; u_mean/u_vec average that axis away. ~112 bytes per probed
        # step at K=5, so it is recorded unconditionally rather than behind a save_* sink.
        "u_iter": d_consecutive.mean(dim=(1, 2, 3)).detach().float().cpu().numpy(),
        "u_iter_vec": d_consecutive.mean(dim=(1, 2)).detach().float().cpu().numpy(),
    }
    for horizon in (10, 20):
        if horizon <= len(u_time):
            rec[f"u_prefix_{horizon}"] = float(u_time[:horizon].mean())

    if suffix_probe_samples:
        boundary = int(prefix_horizon)
        reference = a_hats_full[0][..., :boundary, :adim]
        suffix_predictions = []
        diagnostic_gen = _diagnostic_gen(original_x.device)
        for _ in range(int(suffix_probe_samples)):
            eps = torch.empty_like(original_x).normal_(generator=diagnostic_gen)
            variant = original_x.clone()
            variant[..., boundary:, :] = (
                (1.0 - s) * a_hats_full[0][..., boundary:, :]
                + s * eps[..., boundary:, :])
            variant_clean = variant - s * vfield(variant)
            suffix_predictions.append(variant_clean[..., :boundary, :adim])
        suffix_predictions = torch.stack(suffix_predictions)
        delta = suffix_predictions - reference.unsqueeze(0)
        candidate_stack = torch.cat([reference.unsqueeze(0), suffix_predictions], dim=0)
        rec.update({
            "suffix_prefix_horizon": boundary,
            "suffix_probe_samples": int(suffix_probe_samples),
            "suffix_prefix_abs_mean": float(delta.abs().mean()),
            "suffix_prefix_l2_mean": float(torch.linalg.vector_norm(delta, dim=-1).mean()),
            "suffix_prefix_std_mean": float(candidate_stack.std(dim=0).mean()),
            "suffix_prefix_gripper_flip_rate": float(
                (torch.sign(suffix_predictions[..., 6])
                 != torch.sign(reference[..., 6]).unsqueeze(0)).float().mean())
                if adim > 6 else float("nan"),
            "suffix_prefix_predictions": suffix_predictions.detach().float().cpu().numpy(),
            "suffix_prefix_reference": reference.detach().float().cpu().numpy(),
        })
    if compute_multimodal and A.shape[0] >= 4:
        bc_vec, pc1_frac, bc_pc1 = _multimodal_stats(A.detach().float().cpu().numpy()[:, 0])
        rec["bc_vec"] = bc_vec
        rec["mm_pc1_frac"] = float(pc1_frac)
        rec["mm_bc_pc1"] = float(bc_pc1)
    return ProbeResult(a_hats=A, z_hat_full=z_hat_full, x_acc=x_acc,
                       last_eps=last_eps, u_time=u_time, s=float(s), rec=rec)


def apply_refine(pr: ProbeResult, average: bool) -> torch.Tensor:
    """Refinement action: re-noise from the probe's clean estimate.

    average=True  -> re-noise the MEAN of the K clean estimates (same last eps);
    average=False -> re-noise the LAST prediction (== pr.x_acc).
    """
    if average:
        return (1.0 - pr.s) * pr.z_hat_full + pr.s * pr.last_eps
    return pr.x_acc


def apply_fractional_refine(pr: ProbeResult, x_t: torch.Tensor, average: bool, *,
                            horizon_m: int, num_steps: int, step: int) -> torch.Tensor:
    """Move part-way toward ordinary P&P without changing the sampler time index.

    Ordinary P&P at zero-based Euler ``step`` makes a full clean-predict/re-noise excursion whose
    remaining horizon is ``num_steps - step``.  Scaling that update by
    ``horizon_m / (num_steps - step)`` gives the same effective excursion ``horizon_m / num_steps``
    at every selected step.  ``horizon_m == remaining`` is exactly ordinary full P&P.
    """
    if isinstance(horizon_m, bool) or int(horizon_m) != horizon_m or horizon_m < 1:
        raise ValueError("horizon_m must be a positive integer")
    if isinstance(num_steps, bool) or int(num_steps) != num_steps or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if isinstance(step, bool) or int(step) != step or not 0 <= step < num_steps:
        raise ValueError("step must be a zero-based index within num_steps")
    remaining = int(num_steps) - int(step)
    if horizon_m > remaining:
        raise ValueError(
            f"horizon_m={horizon_m} exceeds remaining sampler horizon {remaining} at step {step}")
    full = apply_refine(pr, average)
    alpha = float(horizon_m) / remaining
    return x_t + alpha * (full - x_t)


def summarize_probe_diagnostics(rec_ep: dict | None) -> dict:
    """Episode means used only for readable worker progress; full data stays in the artifact."""
    steps = [step for chunk in (rec_ep or {}).get("chunks", [])
             for step in chunk.get("steps", [])]
    mapping = {
        "u_first10": "u_prefix_10",
        "u_first20": "u_prefix_20",
        "u_full": "u_mean",
        "suffix_to_prefix_l2": "suffix_prefix_l2_mean",
        "suffix_to_prefix_abs": "suffix_prefix_abs_mean",
        "suffix_to_prefix_std": "suffix_prefix_std_mean",
        "suffix_gripper_flip": "suffix_prefix_gripper_flip_rate",
    }
    summary = {}
    for output, source in mapping.items():
        values = [float(step[source]) for step in steps
                  if step.get(source) is not None and np.isfinite(step[source])]
        if values:
            summary[output] = float(np.mean(values))
    for horizon, output in ((10, "contraction_first10"),
                            (20, "contraction_first20"),
                            (None, "contraction_full")):
        contractions = []
        for step in steps:
            profile = step.get("u_iter_time")
            if profile is None:
                continue
            profile = np.asarray(profile, dtype=float)
            if profile.ndim != 2 or len(profile) < 2:
                continue
            width = profile.shape[1] if horizon is None else min(int(horizon), profile.shape[1])
            sequence = profile[:, :width].mean(axis=1)
            contractions.append(float(sequence[0] - sequence[-1]))
        if contractions:
            summary[output] = float(np.mean(contractions))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode uncertainty recorder.
# ─────────────────────────────────────────────────────────────────────────────
class PnPRecorder:
    """Collects per-episode P&P uncertainty (chunks -> steps) for later persistence."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episodes = []
        self._cur = None
        self._chunk_idx = 0

    def new_episode(self, meta: dict | None = None):
        self._cur = {"meta": dict(meta or {}), "chunks": [], "success": None, "n_steps": None}
        self._chunk_idx = 0

    def current_chunks(self) -> list:
        """Chunks logged in the in-progress episode (before close_episode)."""
        return (self._cur or {}).get("chunks", [])

    def log_chunk(self, chunk_rec: dict):
        if self._cur is None:
            return
        chunk_rec = dict(chunk_rec)
        chunk_rec["chunk_idx"] = self._chunk_idx
        self._cur["chunks"].append(chunk_rec)
        self._chunk_idx += 1

    def close_episode(self, success: bool, n_steps: int):
        if self._cur is None:
            return
        self._cur["success"] = bool(success)
        self._cur["n_steps"] = int(n_steps)
        self.episodes.append(self._cur)
        self._cur = None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sample selection: sample N chunks, keep the lowest-uncertainty one.
# ─────────────────────────────────────────────────────────────────────────────
def multi_sample_select(policy, batch, base_seed, chunk_idx, num_samples, probe_steps,
                        noise_of, num_iterations=3, uncertainty_horizon=None,
                        return_details=False):
    """Return (chosen_action_chunk, chosen_idx, per_candidate_u).

    `noise_of(si)` yields the initial noise for candidate si (from the rollout's per-episode
    noise generator family). Uncertainty is probed in measurement mode.
    """
    from .sampler import measure_chunk_uncertainty  # local import avoids a cycle
    cand_u, cand_actions, candidate_profiles = [], [], []
    for si in range(num_samples):
        noise = noise_of(si)
        _pnp_seed_perturb(base_seed + chunk_idx * 1000 + si)
        measured = measure_chunk_uncertainty(
            policy, batch, noise=noise, probe_steps=probe_steps,
            num_iterations=num_iterations, uncertainty_horizon=uncertainty_horizon,
            return_details=return_details)
        action, u = measured[:2]
        cand_actions.append(action)
        cand_u.append(float(u))
        if return_details:
            candidate_profiles.append(measured[2])
    chosen = int(np.argmin(cand_u))
    output = (cand_actions[chosen], chosen, cand_u)
    return (*output, candidate_profiles) if return_details else output


def multi_policy_select(candidates, noises, probe_steps, num_iterations=3,
                        perturb_seeds=None, uncertainty_horizon=None,
                        return_details=False):
    """Select the lowest-uncertainty action from policy/batch candidates.

    ``candidates`` is an ordered sequence of ``(policy, preprocessed_batch)`` pairs and
    ``noises`` supplies the matched initial noise for each slot. Keeping slot order explicit lets
    source+m1 and source+source controls use the same candidate seeds at every decision point.
    """
    from .sampler import measure_chunk_uncertainty

    if len(candidates) != len(noises) or not candidates:
        raise ValueError("candidates and noises must have the same non-zero length")
    if perturb_seeds is not None and len(perturb_seeds) != len(candidates):
        raise ValueError("perturb_seeds must match the candidate count")
    actions, uncertainties, candidate_profiles = [], [], []
    for index, ((policy, batch), noise) in enumerate(zip(candidates, noises)):
        if perturb_seeds is not None:
            _pnp_seed_perturb(int(perturb_seeds[index]))
        measured = measure_chunk_uncertainty(
            policy, batch, noise=noise, probe_steps=tuple(probe_steps),
            num_iterations=int(num_iterations), uncertainty_horizon=uncertainty_horizon,
            return_details=return_details)
        action, uncertainty = measured[:2]
        actions.append(action)
        uncertainties.append(float(uncertainty))
        if return_details:
            candidate_profiles.append(measured[2])
    chosen = int(np.argmin(uncertainties))
    output = (actions[chosen], chosen, uncertainties, actions)
    return (*output, candidate_profiles) if return_details else output


# ─────────────────────────────────────────────────────────────────────────────
# No-op / noise-pairing contract test (upgraded assert_pnp_noop).
# ─────────────────────────────────────────────────────────────────────────────
def assert_pnp_noop(policy, batch, step_indices=(1, 2), seed=0, raise_on_fail=True, tol=None):
    """Verify the two contract properties on real weights, via predict_action_chunk(noise=None):

      1. RNG ISOLATION (exact): with the global RNG re-seeded identically before each call,
         the global RNG state AFTER a measure-only probe chunk equals the state after a vanilla
         chunk — the dedicated perturbation generator never advanced it.
      2. OUTPUT EQUALITY (within a tolerance): a measure-only probe matches vanilla up to the
         bf16/flow-matching nondeterminism floor, measured by running vanilla twice.
    """
    from .config import RolloutConfig
    from .tap import RolloutTap
    model = policy.model
    prev = model._pnp.strategy

    def _seeded_chunk():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        _pnp_seed_perturb(seed)
        return policy.predict_action_chunk(batch, noise=None).detach().float().cpu().numpy()

    try:
        model._pnp.strategy = None                       # vanilla (delegates to orig sampler)
        a_van1 = _seeded_chunk()
        rng_after_vanilla = torch.random.get_rng_state()
        a_van2 = _seeded_chunk()
        floor = float(np.abs(a_van1 - a_van2).max())

        cfg = RolloutConfig(pnp_steps=tuple(step_indices), pnp_k=3, save_trajectory=False)
        model._pnp.strategy = RolloutTap(cfg, PnPRecorder(), device=None,
                                         adim=model._pnp.action_dim)
        a_unc = _seeded_chunk()
        rng_after_unc = torch.random.get_rng_state()

        rng_ok = torch.equal(rng_after_vanilla, rng_after_unc)
        gap = float(np.abs(a_unc - a_van1).max())
        thresh = tol if tol is not None else max(floor * 3.0, 1e-3)
        ok = rng_ok and gap <= thresh
        msg = (f"assert_pnp_noop: rng_isolated={rng_ok}  output_gap={gap:.2e} "
               f"(floor={floor:.2e}, thresh={thresh:.2e})  -> {'OK' if ok else 'FAIL'}")
        if not ok and raise_on_fail:
            raise AssertionError(msg)
        print(msg)
        return ok
    finally:
        model._pnp.strategy = prev
