"""P&P core: recorder, isolated perturbation RNG, refine step, no-op contract test.

Ported and unified from smolvla_eval_core.py + the GENERATOR_INFRA block and the newer
`refine_average` / multimodality additions in pnp_pro_experiment_averages.ipynb.

Design invariants (the whole point):
- The perturbation noise draws use a DEDICATED per-device torch.Generator (`_pnp_gen`) seeded
  via `_pnp_seed_perturb`. It NEVER touches the global RNG or the initial-noise stream that
  `rollout.py` owns. So uncertainty mode is a true no-op of vanilla, and refinement/PCP
  perturbations are reproducible and paired across methods.
- `PNP_CONFIG` is the single mutable config the patched sampler reads; drivers set it per pass.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import PnPConfig, PERTURB_SEED_MASK, ADIM

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
# The shared K predict-and-perturb block.
# ─────────────────────────────────────────────────────────────────────────────
def pnp_refine_at_step(x_t, s, vfield, cfg: PnPConfig):
    """Run K predict-and-perturb iterations at fixed noise level s.

        predict:  a_hat = x - s * v(x, s)
        perturb:  x'    = (1 - s) * a_hat + s * eps,   eps ~ N(0, I)  (isolated generator)

    Returns (x_out, rec). x_out is the refined re-noised state if cfg.do_refine, else the
    original x_t (uncertainty/collect are non-invasive). Honors cfg.refine_average (re-noise
    the MEAN of the K clean estimates vs the LAST) and cfg.compute_multimodal.
    `rec` always carries the uncertainty measured across iterations.
    """
    adim = cfg.action_dim
    x_acc = x_t
    a_hats, a_hats_full = [], []
    last_eps = None
    gen = _pnp_gen(x_acc.device)
    for _ in range(cfg.num_iterations):
        v = vfield(x_acc)
        a_hat = x_acc - s * v
        a_hats.append(a_hat[..., :adim])
        a_hats_full.append(a_hat)
        eps = torch.empty_like(x_acc).normal_(generator=gen)   # dedicated stream, not global
        last_eps = eps
        x_acc = (1.0 - s) * a_hat + s * eps

    A = torch.stack(a_hats, dim=0)                     # (K, B, chunk, adim)
    if A.shape[0] >= 2:
        u_consecutive = (A[1:] - A[:-1]).abs().mean(dim=0)     # (B, chunk, adim)
        a_std = A.std(dim=0)
    else:
        u_consecutive = torch.zeros_like(A[0]); a_std = torch.zeros_like(A[0])

    u_vec = u_consecutive.mean(dim=(0, 1)).detach().float().cpu().numpy()
    a_std_vec = a_std.mean(dim=(0, 1)).detach().float().cpu().numpy()
    a_mean_vec = A.mean(dim=(0, 1, 2)).detach().float().cpu().numpy()

    rec = {
        "s": float(s),
        "u_mean": float(u_consecutive.mean()),
        "u_max": float(u_consecutive.max()),
        "a_std_mean": float(a_std.mean()),
        "u_vec": u_vec,                                # np (adim,)
        "a_std_vec": a_std_vec,
        "a_mean_vec": a_mean_vec,
        "u_consecutive": u_consecutive.detach().float().cpu().numpy(),
        "a_std": a_std.detach().float().cpu().numpy(),
    }
    if cfg.compute_multimodal and A.shape[0] >= 4:
        A0 = A.detach().float().cpu().numpy()[:, 0]    # (K, chunk, adim)
        bc_vec, pc1_frac, bc_pc1 = _multimodal_stats(A0)
        rec["bc_vec"] = bc_vec
        rec["mm_pc1_frac"] = float(pc1_frac)
        rec["mm_bc_pc1"] = float(bc_pc1)
    if cfg.record_per_iteration:
        rec["a_hats"] = A.detach().float().cpu().numpy()

    if cfg.do_refine and cfg.refine_average and len(a_hats_full) >= 1:
        a_bar = torch.stack(a_hats_full, dim=0).mean(dim=0)    # full-width mean estimate
        x_out = (1.0 - s) * a_bar + s * last_eps              # re-noise the MEAN (same eps)
    else:
        x_out = x_acc                                         # re-noise the LAST prediction
    return (x_out if cfg.do_refine else x_t), rec


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
# Strategy (duck-typed; the sampler calls .selected / .step / .invasive / .finish).
# ─────────────────────────────────────────────────────────────────────────────
class RecordStrategy:
    """P&P uncertainty/refinement. Non-invasive when do_refine is False (pure measurement).

    Holds its config and the per-episode recorder explicitly — no module globals.
    """

    def __init__(self, cfg: PnPConfig, recorder: PnPRecorder):
        self.cfg = cfg
        self.recorder = recorder

    @property
    def invasive(self) -> bool:
        return self.cfg.do_refine

    def selected(self, step: int, s: float) -> bool:
        return self.cfg.step_selected(step, s)

    def step(self, x_t, s, vf, ctx):
        x_out, rec = pnp_refine_at_step(x_t, s, vf, self.cfg)
        rec["step"] = ctx.step
        ctx.records.append(rec)
        return x_out

    def finish(self, ctx):
        self.recorder.log_chunk({"num_steps": ctx.num_steps, "steps": ctx.records})


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
         the global RNG state AFTER an uncertainty-mode chunk equals the state after a vanilla
         chunk — the dedicated perturbation generator never advanced it.
      2. OUTPUT EQUALITY (within a tolerance): uncertainty mode matches vanilla up to the
         bf16/flow-matching nondeterminism floor, measured by running vanilla twice.
    """
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

        cfg = PnPConfig(enabled=True, mode="uncertainty", step_indices=tuple(step_indices),
                        num_iterations=3)
        model._pnp_strategy = RecordStrategy(cfg, PnPRecorder())
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
