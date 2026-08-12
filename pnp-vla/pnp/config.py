"""Central configuration: constants + typed config dataclasses.

Everything that used to be a scattered notebook global lives here. Import from
`pnp.config` rather than redefining. No torch/sim imports at module load so this
is safe to import locally without a GPU stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Versioning levers (see the plan's provenance design)
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1
# Bump whenever P&P/PCP sampling SEMANTICS change (e.g. the RNG-isolation fix), so
# results produced by different algorithm versions are separable *by data*.
SAMPLER_ALGO_VERSION = 1

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
LIBERO_PRO_MAX_STEPS = 280  # fallback when a suite matches no stock base

# Longest base-suite prefix wins, so libero_object_temp_x0.1 -> libero_object, not a partial match.
_BASE_SUITE_PREFIXES = tuple(sorted(MAX_STEPS_MAP, key=len, reverse=True))


def resolve_max_steps(suite: str) -> int:
    """Episode horizon for a suite, stock or perturbed.

    LIBERO-PRO's own TASK_MAX_STEPS gives every perturbed suite its BASE suite's canonical limit
    (goal 300, spatial 220, libero_10 520, object 280 -- the OpenVLA values, each the suite's
    longest training demo plus margin). So resolve by longest base prefix rather than exact key,
    or `libero_10_swap` silently runs at 280 instead of 520 and `libero_goal_*` at 280 instead
    of 300.
    """
    for prefix in _BASE_SUITE_PREFIXES:
        if suite.startswith(prefix):
            return MAX_STEPS_MAP[prefix]
    return LIBERO_PRO_MAX_STEPS

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
# Canonical method taxonomy — the SINGLE source of the `method` strings written to
# rollouts.method. Both the Colab notebooks (which write them) and local analysis/ (which
# filters/colors by them) import these, so the label never drifts across the two. The
# refinement last/avg variant is NOT a separate method — it lives in the refine_average column.
# ─────────────────────────────────────────────────────────────────────────────
class Method:
    VANILLA = "vanilla"
    EXTRA_STEPS = "extra_steps"
    UNCERTAINTY = "pnp_uncertainty_only"    # probe set, no action (RNG-isolated no-op)
    REFINEMENT = "pnp_refinement"           # refine; last vs avg = refine_average column
    THRESHOLD_REFINEMENT = "pnp_threshold_refinement"  # refine selected steps iff U >= threshold
    MULTI_SAMPLE = "multi_sample_select"
    CHUNK_SOURCE_SOURCE = "chunk_select_source_source"
    CHUNK_SOURCE_MULTI_QUERY = "chunk_select_source_multi_query"
    CHUNK_SOURCE_M1 = "chunk_select_source_m1"
    PNP_ONLY = "pnp_only"                   # PCP correction, lambda == 0
    PCP = "pcp"                             # PCP correction, lambda > 0
    COLLECT = "collect"                     # vanilla rollout w/ save_pcp_features (training data)


ALL_METHODS = (Method.VANILLA, Method.EXTRA_STEPS, Method.UNCERTAINTY, Method.REFINEMENT,
               Method.THRESHOLD_REFINEMENT,
               Method.MULTI_SAMPLE, Method.CHUNK_SOURCE_SOURCE, Method.CHUNK_SOURCE_MULTI_QUERY,
               Method.CHUNK_SOURCE_M1,
               Method.PNP_ONLY, Method.PCP)
PCP_3WAY = (Method.VANILLA, Method.PNP_ONLY, Method.PCP)   # the paired 3-way eval arms


# ─────────────────────────────────────────────────────────────────────────────
# Rollout config: three ORTHOGONAL axes, FLAT on one dataclass.
#   (A) action  — how the executed action is produced. At most one of: refine /
#                 correction_lambda (PCP) / num_samples (multi-sample); None => vanilla.
#   (B) probe   — the P&P predict-and-perturb MEASUREMENT: pnp_steps (where) + pnp_k (how hard).
#   (C) sinks   — independent save_* booleans (what gets persisted).
# A "method" is not a class; it's a point in this flat switch space. The fields map 1:1 to the
# `rollouts` columns, so store._denorm is near-identity (no build-then-flatten round-trip).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RolloutConfig:
    """Everything that defines ONE rollout's behavior + what to record, as flat fields.

    e.g. training data for the corrector is just
    `RolloutConfig(pnp_steps=(7,8), pnp_k=3, save_pcp_features=True)` — a vanilla action source
    (no refine/correction/num_samples) with the pcp-features sink on.

    LeRobot flow time runs s=1.0 (noise) -> s=0.0 (clean), so early/high-noise steps are the
    FIRST Euler steps (large s). Select them with pnp_steps=(1,)/(1,2) or the pnp_time_min selector.
    """
    # ── (B) probe — the P&P measurement (pnp_steps is None => no probe) ──
    pnp_steps: Optional[Sequence[int]] = None   # Euler steps to probe (unless pnp_time_min set)
    pnp_k: int = 3                              # K predict-and-perturb iterations
    pnp_time_min: Optional[float] = None        # alt selector: probe when s >= pnp_time_min
    compute_multimodal: bool = False            # per-dim Sarle BC + PC1 stats (needs pnp_k>=4)
    action_dim: int = ADIM                      # real (un-padded) dims used for uncertainty
    # ── (A) action — at most one of refine / correction_lambda / num_samples ──
    refine: bool = False                        # re-noise from the probe's clean estimate
    refine_average: bool = False                # refine from mean of K a_hats (True) vs last (False)
    refine_threshold: Optional[float] = None    # refine selected Euler step iff u_mean >= this
    correction_lambda: Optional[float] = None   # set => PCP correction action (0.0 == P&P, no grad)
    q_gate: float = 0.5                         # only correct chunks with predicted P(success) < gate
    q_ckpt_id: Optional[str] = None
    q_model: object = field(default=None, repr=False, compare=False)   # runtime handle, not serialized
    q_scaler: object = field(default=None, repr=False, compare=False)  # runtime handle, not serialized
    num_samples: Optional[int] = None           # set => multi-sample-select (chunk level)
    ms_probe_steps: Sequence[int] = (1, 2)      # probe steps for multi-sample uncertainty
    candidate_set_id: Optional[str] = None       # immutable model/revision set for online selection
    # ── base + sinks (each persists one thing independently) ──
    num_inference_steps: Optional[int] = None   # base sampler step override (matched-compute)
    save_uncertainty: Optional[bool] = None     # default: on iff a probe is set
    save_pcp_features: bool = False             # per-chunk obs_enc + z_hat -> Storage (training)
    save_ahats: bool = False                    # full K a_hats stacks -> Storage (geometry)
    save_observations: bool = False             # low-res decision-point frames -> Storage
    save_trajectory: bool = True                # executed actions + robot state -> Storage (cheap)
    save_generated_chunks: bool = False         # exact policy-space clean chunks at t=1
    video: str = "off"                          # "off" | "failures_only" | "all"
    # ── performance (must not change the trajectory; excluded from LOGICAL_FIELDS) ──
    # Render the cameras only on steps whose observation the policy actually consumes (chunk
    # boundaries). Rendering is ~90% of a LIBERO step and 49 of 50 renders are discarded, so this
    # is worth ~5x wall clock. Ignored when a sink needs every frame (save_observations/video).
    skip_unused_renders: bool = False
    # Steps of lead time before a consumed observation to re-enable the cameras. Robosuite
    # returns the LAST cached image on a freshly re-enabled observable rather than a fresh
    # render, so a lead of 1 hands the policy a stale frame (verified on the real simulator:
    # assert_render_skip_equivalent failed at the first decision). Determine the true lag with
    # the render-lag probe and set this from data, never by guessing.
    render_lead: int = 2

    def __post_init__(self):
        n_actions = int(self.refine) + int(self.correction_lambda is not None) \
            + int(self.num_samples is not None)
        if n_actions > 1:
            raise ValueError("at most one action: refine / correction_lambda / num_samples")
        # Refine/correction feed off the probe, so a probe is mandatory for them.
        if (self.refine or self.correction_lambda is not None) and not self.has_probe:
            raise ValueError("refine/correction requires a probe (set pnp_steps or pnp_time_min)")
        # MultiSample carries its own ms_probe_steps and runs at the chunk level.
        if self.num_samples is not None and self.has_probe:
            raise ValueError("num_samples carries ms_probe_steps; leave pnp_steps/pnp_time_min unset")
        if self.refine_average and not self.refine:
            raise ValueError("refine_average=True requires refine=True")
        if self.refine_threshold is not None:
            if not self.refine:
                raise ValueError("refine_threshold requires refine=True")
            if not math.isfinite(float(self.refine_threshold)) or self.refine_threshold < 0:
                raise ValueError("refine_threshold must be finite and non-negative")
        # Probe-derived sinks need a probe.
        for sink in ("save_pcp_features", "save_ahats"):
            if getattr(self, sink) and not self.has_probe:
                raise ValueError(f"{sink} requires a probe (its data comes from the probe)")
        if self.save_uncertainty and not self.has_probe:
            raise ValueError("save_uncertainty=True requires a probe")
        if self.render_lead < 1:
            raise ValueError("render_lead must be >= 1 (the consumed step itself)")

    @property
    def has_probe(self) -> bool:
        """Whether a P&P probe is active (drives measurement + any refine/correct feedback)."""
        return self.pnp_steps is not None or self.pnp_time_min is not None

    def probe_selected(self, step: int, s: float) -> bool:
        if self.pnp_time_min is not None:
            return s >= self.pnp_time_min
        return self.pnp_steps is not None and step in tuple(self.pnp_steps)

    @property
    def records_uncertainty(self) -> bool:
        """Whether to persist per-step uncertainty rows (default: on iff a probe is set)."""
        return self.save_uncertainty if self.save_uncertainty is not None else self.has_probe

    def logical_dict(self) -> dict:
        """The behavior-defining config — everything that changes the executed trajectory.

        This (not a hand-picked subset) is what the rollout_id / config_hash are built from, so
        sweeping ANY of these axes (e.g. correction_lambda, q_gate) yields distinct rollouts even
        under one method name. Excludes: sinks + video (persistence-only), compute_multimodal
        (recording-only), and q_model/q_scaler (runtime handles — q_ckpt_id captures their identity).
        """
        logical = {f: getattr(self, f) for f in LOGICAL_FIELDS}
        # Added after historical experiments were collected. Omitting its default preserves every
        # old config hash/rollout ID; online multi-policy runs set it explicitly and are revision-
        # bound in their IDs.
        if logical.get("candidate_set_id") is None:
            logical.pop("candidate_set_id")
        if logical.get("refine_threshold") is None:
            logical.pop("refine_threshold")
        return logical


# Behavior-defining fields of RolloutConfig (see RolloutConfig.logical_dict).
LOGICAL_FIELDS = ("pnp_steps", "pnp_k", "pnp_time_min", "action_dim",
                  "refine", "refine_average", "refine_threshold",
                  "correction_lambda", "q_gate", "q_ckpt_id",
                  "num_samples", "ms_probe_steps", "candidate_set_id", "num_inference_steps")


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
