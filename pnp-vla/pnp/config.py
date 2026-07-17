"""Central configuration: constants + typed config dataclasses.

Everything that used to be a scattered notebook global lives here. Import from
`pnp.config` rather than redefining. No torch/sim imports at module load so this
is safe to import locally without a GPU stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

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
# Rollout decomposition: three ORTHOGONAL axes on RolloutConfig.
#   (A) action  — how the executed action is produced (None / Refine / Correct / MultiSample)
#   (B) probe   — the P&P predict-and-perturb MEASUREMENT (where + how hard)
#   (C) sinks   — independent save_* booleans (what gets persisted)
# A "method" is no longer a class; it's a point in this switch space.
# ─────────────────────────────────────────────────────────────────────────────


# ── (B) Probe — the predict-and-perturb measurement ──────────────────────────
@dataclass
class Probe:
    """The P&P measurement at selected Euler steps. `steps` is the single source of truth for
    WHERE anything happens — the uncertainty read AND any Refine/Correct feedback act there.

    LeRobot flow time runs s=1.0 (noise) -> s=0.0 (clean), so early/high-noise steps are the
    FIRST Euler steps (large s). Select them with steps=(1,)/(1,2) or the time_min selector.
    """
    steps: Optional[Sequence[int]] = (1,)   # Euler steps to probe (unless time_min set)
    k: int = 3                              # K predict-and-perturb iterations
    time_min: Optional[float] = None        # alt selector: probe when s >= time_min
    compute_multimodal: bool = False        # per-dim Sarle BC + PC1 stats (needs k>=4)
    action_dim: int = ADIM                  # real (un-padded) dims used for uncertainty

    def selected(self, step: int, s: float) -> bool:
        if self.time_min is not None:
            return s >= self.time_min
        return self.steps is not None and step in tuple(self.steps)


# ── (A) Action sources — at most one per rollout ─────────────────────────────
@dataclass
class Refine:
    """Re-noise inference from the probe's clean estimate (self-refinement)."""
    average: bool = False    # re-noise from mean of K a_hats (True) vs the last a_hat (False)


@dataclass
class Correct:
    """PCP: gated Q-gradient nudge on the probe's z_hat. q_model/q_scaler are RUNTIME handles
    the notebook attaches after load_q_corrector — they are NOT serialized into config_json."""
    lam: float = 3.0                        # correction step size
    gate: float = 0.5                       # only correct chunks with predicted P(success) < gate
    q_ckpt_id: Optional[str] = None
    q_model: object = field(default=None, repr=False, compare=False)
    q_scaler: object = field(default=None, repr=False, compare=False)


@dataclass
class MultiSample:
    """Sample n chunks and keep the lowest-uncertainty one. Applied at the CHUNK level in
    run_episode (not inside the Euler loop), so it carries its own probe_steps."""
    n: int = 5
    probe_steps: Sequence[int] = (1, 2)


Action = Union[Refine, Correct, MultiSample]


# ─────────────────────────────────────────────────────────────────────────────
# (C) Per-rollout config — action + probe + independent save_* sinks.
# The notebook composes these; nothing about the experiment lives in the package.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RolloutConfig:
    """Everything that defines ONE rollout's behavior + what to record.

    Three orthogonal axes: `action` (how the executed action is produced), `probe` (the P&P
    measurement), and the `save_*` sinks (what to persist). e.g. training data for the
    corrector is just `RolloutConfig(probe=Probe((7,8), k=3), save_pcp_features=True)` — a
    vanilla action source with the pcp-features sink on.
    """
    action: Optional[Action] = None             # None (vanilla) | Refine | Correct | MultiSample
    probe: Optional[Probe] = None               # the P&P measurement; None = no probe
    num_inference_steps: Optional[int] = None   # base sampler step override (matched-compute)
    # sinks (each persists one thing independently)
    save_uncertainty: Optional[bool] = None     # default: on iff a probe is set
    save_pcp_features: bool = False             # per-chunk obs_enc + z_hat -> Storage (training)
    save_ahats: bool = False                    # full K a_hats stacks -> Storage (geometry)
    save_observations: bool = False             # low-res decision-point frames -> Storage
    save_trajectory: bool = True                # executed actions + robot state -> Storage (cheap)
    video: str = "off"                          # "off" | "failures_only" | "all"

    def __post_init__(self):
        a = self.action
        if a is not None and not isinstance(a, (Refine, Correct, MultiSample)):
            raise TypeError(f"action must be None/Refine/Correct/MultiSample, got {type(a).__name__}")
        # Refine/Correct feed off the probe, so a probe is mandatory for them.
        if isinstance(a, (Refine, Correct)) and self.probe is None:
            raise ValueError(f"{type(a).__name__} action requires a probe (it acts at probe.steps)")
        # MultiSample carries its own probe_steps and runs at the chunk level.
        if isinstance(a, MultiSample) and self.probe is not None:
            raise ValueError("MultiSample carries its own probe_steps; leave RolloutConfig.probe=None")
        # Probe-derived sinks need a probe.
        for sink in ("save_pcp_features", "save_ahats"):
            if getattr(self, sink) and self.probe is None:
                raise ValueError(f"{sink} requires a probe (its data comes from the probe)")
        if self.save_uncertainty and self.probe is None:
            raise ValueError("save_uncertainty=True requires a probe")

    @property
    def records_uncertainty(self) -> bool:
        """Whether to persist per-step uncertainty rows (default: on iff a probe is set)."""
        return self.save_uncertainty if self.save_uncertainty is not None else (self.probe is not None)


# ─────────────────────────────────────────────────────────────────────────────
# Q-corrector training hyperparameters (training-time, not a rollout concern).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Hyperparameters for train_q_corrector + the hard-task/step filter for its dataset."""
    correction_steps: Sequence[int] = (7, 8)    # deploy steps to train on (subset of collected)
    hard_lo: float = 0.10                        # keep tasks with SR in (hard_lo, hard_hi)
    hard_hi: float = 0.90
    seed: int = 42
    train_frac: float = 0.80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smooth: float = 0.05
    epochs: int = 100
    patience: int = 20
    batch: int = 256
