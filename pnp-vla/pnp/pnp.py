"""P&P core: recorder, isolated perturbation RNG, the probe, refinement, no-op contract test.

Ported and unified from smolvla_eval_core.py + the GENERATOR_INFRA block and the newer
`refine_average` / multimodality additions in pnp_pro_experiment_averages.ipynb.

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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import Probe, PERTURB_SEED_MASK, ADIM

# No module-level config/recorder globals: config travels with the strategy, and the
# recorder is created per run_episode call. Only the mechanical perturbation-generator state
# below is module-level (reseeded per episode).

# ─────────────────────────────────────────────────────────────────────────────
# Isolated perturbation generators (never advance the global / noise RNG).
# ─────────────────────────────────────────────────────────────────────────────
_PNP_GENS: dict[str, torch.Generator] = {}
_PNP_LAST_SEED = [0]


def _pnp_gen(device) -> torch.Generator:
    key = str(device)
    g = _PNP_GENS.get(key)
    if g is None:
        g = torch.Generator(device=torch.device(device))
        g.manual_seed(int(_PNP_LAST_SEED[0]) ^ PERTURB_SEED_MASK)
        _PNP_GENS[key] = g
    return g


def _pnp_seed_perturb(seed: int) -> None:
    """Seed the dedicated perturbation stream (independent of the global RNG).

    Call once per episode so perturbations are reproducible and paired across methods.
    """
    _PNP_LAST_SEED[0] = int(seed)
    for g in _PNP_GENS.values():
        g.manual_seed(int(seed) ^ PERTURB_SEED_MASK)


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
    s: float
    rec: dict                 # uncertainty record (u_mean/u_max/u_vec/a_std_vec/... + multimodal)


def run_probe(x_t, s, vfield, probe: Probe) -> ProbeResult:
    """Run K predict-and-perturb iterations at fixed noise level s and measure uncertainty.

        predict:  a_hat = x - s * v(x, s)
        perturb:  x'    = (1 - s) * a_hat + s * eps,   eps ~ N(0, I)  (isolated generator)

    Pure: no global state touched (perturbation noise comes from the dedicated generator).
    """
    adim = probe.action_dim
    x_acc = x_t
    a_hats, a_hats_full = [], []
    last_eps = None
    gen = _pnp_gen(x_acc.device)
    for _ in range(probe.k):
        v = vfield(x_acc)
        a_hat = x_acc - s * v
        a_hats.append(a_hat[..., :adim])
        a_hats_full.append(a_hat)
        last_eps = torch.empty_like(x_acc).normal_(generator=gen)   # dedicated stream, not global
        x_acc = (1.0 - s) * a_hat + s * last_eps

    A = torch.stack(a_hats, dim=0)                     # (K, B, chunk, adim)
    z_hat_full = torch.stack(a_hats_full, dim=0).mean(dim=0)   # (B, chunk, max_adim)
    if A.shape[0] >= 2:
        u_consecutive = (A[1:] - A[:-1]).abs().mean(dim=0)     # (B, chunk, adim)
        a_std = A.std(dim=0)
    else:
        u_consecutive = torch.zeros_like(A[0]); a_std = torch.zeros_like(A[0])

    rec = {
        "s": float(s),
        "u_mean": float(u_consecutive.mean()),
        "u_max": float(u_consecutive.max()),
        "a_std_mean": float(a_std.mean()),
        "u_vec": u_consecutive.mean(dim=(0, 1)).detach().float().cpu().numpy(),
        "a_std_vec": a_std.mean(dim=(0, 1)).detach().float().cpu().numpy(),
        "a_mean_vec": A.mean(dim=(0, 1, 2)).detach().float().cpu().numpy(),
    }
    if probe.compute_multimodal and A.shape[0] >= 4:
        bc_vec, pc1_frac, bc_pc1 = _multimodal_stats(A.detach().float().cpu().numpy()[:, 0])
        rec["bc_vec"] = bc_vec
        rec["mm_pc1_frac"] = float(pc1_frac)
        rec["mm_bc_pc1"] = float(bc_pc1)
    return ProbeResult(a_hats=A, z_hat_full=z_hat_full, x_acc=x_acc,
                       last_eps=last_eps, s=float(s), rec=rec)


def apply_refine(pr: ProbeResult, average: bool) -> torch.Tensor:
    """Refinement action: re-noise from the probe's clean estimate.

    average=True  -> re-noise the MEAN of the K clean estimates (same last eps);
    average=False -> re-noise the LAST prediction (== pr.x_acc).
    """
    if average:
        return (1.0 - pr.s) * pr.z_hat_full + pr.s * pr.last_eps
    return pr.x_acc


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
                        noise_of):
    """Return (chosen_action_chunk, chosen_idx, per_candidate_u).

    `noise_of(si)` yields the initial noise for candidate si (from the rollout's per-episode
    noise generator family). Uncertainty is probed in measurement mode.
    """
    from .sampler import measure_chunk_uncertainty  # local import avoids a cycle
    cand_u, cand_actions = [], []
    for si in range(num_samples):
        noise = noise_of(si)
        _pnp_seed_perturb(base_seed + chunk_idx * 1000 + si)
        action, u = measure_chunk_uncertainty(policy, batch, noise=noise, probe_steps=probe_steps)
        cand_actions.append(action)
        cand_u.append(float(u))
    chosen = int(np.argmin(cand_u))
    return cand_actions[chosen], chosen, cand_u


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
    prev = getattr(model, "_pnp_strategy", None)

    def _seeded_chunk():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        _pnp_seed_perturb(seed)
        return policy.predict_action_chunk(batch, noise=None).detach().float().cpu().numpy()

    try:
        model._pnp_strategy = None                       # vanilla (delegates to orig sampler)
        a_van1 = _seeded_chunk()
        rng_after_vanilla = torch.random.get_rng_state()
        a_van2 = _seeded_chunk()
        floor = float(np.abs(a_van1 - a_van2).max())

        cfg = RolloutConfig(probe=Probe(steps=tuple(step_indices), k=3), save_trajectory=False)
        model._pnp_strategy = RolloutTap(cfg, PnPRecorder(), device=None,
                                         adim=getattr(model, "_pnp_action_dim", ADIM))
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
        model._pnp_strategy = prev
