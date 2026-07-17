"""Central configuration: constants + typed config dataclasses.

Everything that used to be a scattered notebook global lives here. Import from
`pnp.config` rather than redefining. No torch/sim imports at module load so this
is safe to import locally without a GPU stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Versioning levers (see the plan's provenance design)
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1
# Bump whenever P&P/PCP sampling SEMANTICS change (e.g. the RNG-isolation fix), so
# results produced by different algorithm versions are separable *by data*.
SAMPLER_ALGO_VERSION = 2  # v2 == post-RNG-isolation (dedicated perturbation generator)

# ─────────────────────────────────────────────────────────────────────────────
# LIBERO / simulator constants
# ─────────────────────────────────────────────────────────────────────────────
ADIM = 7  # real (un-padded) LIBERO action dims: 0-2 xyz, 3-5 axis-angle, 6 gripper
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

CAMERAS = ["agentview", "robot0_eye_in_hand"]
IMG_SIZE = 360
NUM_STEPS_WAIT = 10  # dummy steps after set_init_state to let the sim settle
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
VIDEO_FPS = 10

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
LIBERO_PRO_MAX_STEPS = 280  # default horizon for non-stock PRO suites

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
PI05_REPO_ID = "lerobot/pi05_libero_finetuned"

# ─────────────────────────────────────────────────────────────────────────────
# RNG
# ─────────────────────────────────────────────────────────────────────────────
# XOR mask used to derive the isolated perturbation-generator seed from the
# episode seed. Kept identical to the legacy value so paired reproduction holds.
PERTURB_SEED_MASK = 0x9E3779B9


# ─────────────────────────────────────────────────────────────────────────────
# P&P (Predict-and-Perturb) sampling config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PnPConfig:
    """Self-refining (Predict-and-Perturb) sampling config for pi0.5 flow matching.

    LeRobot time runs s=1.0 (noise) -> s=0.0 (clean), so the early/high-noise steps are
    the FIRST Euler steps (large s). Select them with step_indices=(1,)/(1,2) or time_min.
    """
    enabled: bool = False
    step_indices: Optional[Sequence[int]] = (1,)   # which Euler steps run P&P (unless time_min)
    time_min: Optional[float] = None               # alt selector: run P&P when s >= time_min
    num_iterations: int = 3                         # K predict-and-perturb iterations
    mode: str = "both"                              # "uncertainty" | "refine" | "both"
    refine_average: bool = False                    # re-noise from mean(z_hat) (True) vs last a_hat
    action_dim: int = ADIM                          # real (un-padded) dims used for uncertainty
    record_per_iteration: bool = False              # also persist the full (K,B,chunk,adim) stack
    compute_multimodal: bool = False                # per-dim Sarle BC + PC1 stats (needs K>=4)

    def step_selected(self, step: int, s: float) -> bool:
        if not self.enabled:
            return False
        if self.time_min is not None:
            return s >= self.time_min
        return self.step_indices is not None and step in tuple(self.step_indices)

    @property
    def do_refine(self) -> bool:
        return self.mode in ("refine", "both")


# ─────────────────────────────────────────────────────────────────────────────
# PCP (Predict-Correct-Perturb) config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PCPConfig:
    """Predict-Correct-Perturb config.

    `mode` selects the sampler strategy: 'collect' (measurement-only, stash z_hat/obs_enc for
    training) or 'correct' (gated Q-gradient nudge at deploy time). None = not a PCP pass.
    For 'correct', the notebook loads the trained corrector and attaches it via the runtime
    handles (q_model/q_scaler/q_ckpt_id) — these are NOT serialized into config_json.
    """
    mode: Optional[str] = None                  # 'collect' | 'correct' | None
    correction_steps: Sequence[int] = (7, 8)   # Euler steps where the gradient nudge applies
    pnp_k: int = 3                              # predict/perturb iterations before correcting
    lambda_pcp: float = 3.0                     # correction step size
    q_gate: float = 0.5                         # only correct chunks with predicted P(success) < gate
    # data collection / hard-task filter
    collect_steps: Sequence[int] = tuple(range(10))
    hard_lo: float = 0.10
    hard_hi: float = 0.90
    # runtime handles for mode='correct' (set by the notebook after load_q_corrector; not serialized)
    q_model: object = field(default=None, repr=False, compare=False)
    q_scaler: object = field(default=None, repr=False, compare=False)
    q_ckpt_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Per-rollout config — the flag bundle the notebook composes and passes to run_episode
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RolloutConfig:
    """Everything that defines ONE rollout's behavior + what to record.

    The notebook builds a dict of these (one per method) and drives run_episode with them;
    nothing about the experiment (which methods, the sweep, the loop) lives in the package.
    """
    pnp: Optional[PnPConfig] = None             # P&P record/refine; None = vanilla
    pcp: Optional[PCPConfig] = None             # PCP collect/correct; None = not PCP
    num_inference_steps: Optional[int] = None   # override (e.g. matched-compute extra_steps)
    num_samples: Optional[int] = None           # multi-sample-select candidate count
    # recording toggles (see the plan's Storage bucket section)
    record_trajectory: bool = True              # executed actions + robot state -> Storage (cheap)
    record_obs_frames: bool = False             # low-res decision-point frames (largest cost)
    video: str = "off"                          # "off" | "failures_only" | "all"
    record_per_iteration: bool = False          # full a_hats stacks -> Storage (geometry)
