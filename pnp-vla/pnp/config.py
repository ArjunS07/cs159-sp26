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
    FRACTIONAL_M2 = "pnp_fractional_m2"     # fractional P&P, effective horizon m=2
    FRACTIONAL_M4 = "pnp_fractional_m4"     # fractional P&P, effective horizon m=4
    SUFFIX_SENSITIVITY = "pnp_suffix_sensitivity"  # stock action + tail-to-prefix diagnostic
    TAPERED_REFINEMENT = "pnp_tapered_refinement"  # temporal P&P feedback decays in the tail
    PREFIX_REFINEMENT = "pnp_prefix_only_refinement"  # inner P&P updates executed positions only
    REDUCED_STRENGTH_REFINEMENT = "pnp_reduced_strength_refinement"  # smaller inner P&P moves
    U20_GRADIENT = "pnp_u20_gradient_descent"  # descend exact P&P U20 through the live latent
    LATENT_RANDOM_CONTROL = "pnp_latent_random_control"  # equal-RMS/equal-compute random update
    U20_GRADIENT_GATE_015 = "pnp_u20_gradient_action_gate_015"
    U20_GRADIENT_GATE_020 = "pnp_u20_gradient_action_gate_020"
    THRESHOLD_REFINEMENT = "pnp_threshold_refinement"  # refine selected steps iff U >= threshold
    DELAYED_REFINEMENT = "pnp_delayed_refinement"  # refine every chunk from a fixed chunk index
    MULTI_SAMPLE = "multi_sample_select"
    CHUNK_SOURCE_SOURCE = "chunk_select_source_source"
    CHUNK_SOURCE_MULTI_QUERY = "chunk_select_source_multi_query"
    CHUNK_SOURCE_M1 = "chunk_select_source_m1"
    FIVE_STEP_SINGLE_QUERY = "five_step_single_query"
    FIVE_STEP_LOWEST_U20 = "five_step_x3_lowest_u20"
    FIVE_STEP_LOWEST_U20_REFINE = "five_step_x3_lowest_u20_then_refine"
    FIVE_STEP_SINGLE_REFINE = "five_step_single_refine"
    THREE_STEP_SINGLE_REFINE = "three_step_single_refine"
    THREE_STEP_SINGLE_QUERY = "three_step_single_query"
    PNP_ONLY = "pnp_only"                   # PCP correction, lambda == 0
    PCP = "pcp"                             # PCP correction, lambda > 0
    COLLECT = "collect"                     # vanilla rollout w/ save_pcp_features (training data)
    PCP_SEARCH_COLLECT = "pcp_search_collect"  # Bellman/RL-token-ready vanilla collection


ALL_METHODS = (Method.VANILLA, Method.EXTRA_STEPS, Method.UNCERTAINTY, Method.REFINEMENT,
               Method.FRACTIONAL_M2, Method.FRACTIONAL_M4,
               Method.SUFFIX_SENSITIVITY, Method.TAPERED_REFINEMENT,
               Method.PREFIX_REFINEMENT, Method.REDUCED_STRENGTH_REFINEMENT,
               Method.U20_GRADIENT, Method.LATENT_RANDOM_CONTROL,
               Method.U20_GRADIENT_GATE_015, Method.U20_GRADIENT_GATE_020,
               Method.THRESHOLD_REFINEMENT, Method.DELAYED_REFINEMENT,
               Method.MULTI_SAMPLE, Method.CHUNK_SOURCE_SOURCE, Method.CHUNK_SOURCE_MULTI_QUERY,
               Method.CHUNK_SOURCE_M1, Method.FIVE_STEP_SINGLE_QUERY,
               Method.FIVE_STEP_LOWEST_U20, Method.FIVE_STEP_LOWEST_U20_REFINE,
               Method.FIVE_STEP_SINGLE_REFINE, Method.THREE_STEP_SINGLE_REFINE,
               Method.THREE_STEP_SINGLE_QUERY,
               Method.PNP_ONLY, Method.PCP, Method.COLLECT, Method.PCP_SEARCH_COLLECT)
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
    # Fractional P&P moves only m/(num_steps-step) of the way from the current state toward the
    # ordinary full P&P state.  At a 10-step sampler this makes the effective perturbation scale
    # m/10 at every selected Euler step while keeping the sampler at the same time index.
    refine_horizon_m: Optional[int] = None
    refine_threshold: Optional[float] = None    # refine selected Euler step iff u_mean >= this
    # None gates on full-chunk uncertainty; an integer gates on that many leading actions.
    refine_uncertainty_horizon: Optional[int] = None
    refine_start_chunk: Optional[int] = None    # leave chunks [0, this) exact-stock; then refine
    # Full P&P feedback through n_action_steps, then a linear decay to zero at this action index.
    # Applied inside every P&P iteration, not merely to the final sampler state.
    refine_tail_decay_end: Optional[int] = None
    # Update only the generated positions that will actually execute during each inner P&P loop.
    refine_prefix_only: bool = False
    # Scale every inner predict/perturb proposal toward its destination. Unlike
    # refine_horizon_m, this changes the intermediate states seen by all K velocity evaluations.
    refine_inner_strength: Optional[float] = None
    # Differentiate exact P&P uncertainty through the frozen VLA and update only the live latent.
    # The random mode still computes the gradient, then applies an equal-RMS random direction.
    uncertainty_gradient_mode: Optional[str] = None  # None | "descent" | "random"
    uncertainty_gradient_step_size: Optional[float] = None  # RMS of the latent update
    uncertainty_gradient_horizon: int = 20
    # Accept the decoded U-gradient chunk only when its first executed arm actions remain this
    # close to the exact stock chunk (RMS in postprocessed LIBERO action coordinates).
    uncertainty_gradient_action_rms_max: Optional[float] = None
    correction_lambda: Optional[float] = None   # set => PCP correction action (0.0 == P&P, no grad)
    q_gate: float = 0.5                         # only correct chunks with predicted P(success) < gate
    q_ckpt_id: Optional[str] = None
    q_model: object = field(default=None, repr=False, compare=False)   # runtime handle, not serialized
    q_scaler: object = field(default=None, repr=False, compare=False)  # runtime handle, not serialized
    num_samples: Optional[int] = None           # set => multi-sample-select (chunk level)
    ms_probe_steps: Sequence[int] = (1, 2)      # probe steps for multi-sample uncertainty
    # Candidate-selection score averages disagreement over this many leading action positions.
    # None preserves the historical full-generated-chunk score.
    selection_uncertainty_horizon: Optional[int] = None
    candidate_set_id: Optional[str] = None       # immutable model/revision set for online selection
    # Immutable policy/checkpoint source for any arm, including single-query controls.
    policy_source_id: Optional[str] = None
    # New same-policy experiments opt into a resume-safe candidate seed scheme. None retains
    # historical rollout IDs and their legacy sample-slot mapping; stock_slot0_v1 makes
    # candidate 0 exactly reuse the ordinary per-(episode, chunk) policy-noise seed.
    candidate_seed_scheme: Optional[str] = None
    # Explicit two-stage action: measure all ordinary candidates, choose the lowest-uncertainty
    # one, then rerun only that same initial noise under refine-last P&P feedback.
    multi_sample_refine_selected: bool = False
    # ── base + sinks (each persists one thing independently) ──
    num_inference_steps: Optional[int] = None   # base sampler step override (matched-compute)
    # None intentionally preserves the legacy full-generated-chunk behavior. Production drivers
    # should set this explicitly; it does not inherit policy.config.n_action_steps.
    n_action_steps: Optional[int] = None        # generated-chunk prefix executed before replanning
    save_uncertainty: Optional[bool] = None     # default: on iff a probe is set
    save_pcp_features: bool = False             # per-chunk obs_enc + z_hat -> Storage (training)
    save_ahats: bool = False                    # full K a_hats stacks -> Storage (geometry)
    # Per-action uncertainty profile; uses the existing ahats_path artifact without storing
    # the much larger K x chunk x action a-hat stack.
    save_time_uncertainty: bool = False
    save_observations: bool = False             # low-res decision-point frames -> Storage
    save_trajectory: bool = True                # executed actions + robot state -> Storage (cheap)
    save_generated_chunks: bool = False         # exact policy-space clean chunks at t=1
    # PCP-search's lossless decision/transition artifact. This is persistence-only: enabling it
    # must not change the executed trajectory or rollout ID.
    save_training_data: bool = False
    # Measurement-only diagnostic: independently re-noise the unused suffix while preserving
    # the executed-prefix latent, then record how the prefix prediction changes.
    suffix_probe_samples: int = 0
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
            + int(self.num_samples is not None) + int(self.uncertainty_gradient_mode is not None)
        if n_actions > 1:
            raise ValueError("at most one action: refine / correction / samples / U-gradient")
        # Refine/correction feed off the probe, so a probe is mandatory for them.
        if (self.refine or self.correction_lambda is not None) and not self.has_probe:
            raise ValueError("refine/correction requires a probe (set pnp_steps or pnp_time_min)")
        if self.uncertainty_gradient_mode is not None:
            if not self.has_probe:
                raise ValueError("uncertainty-gradient feedback requires a P&P probe")
            if self.uncertainty_gradient_mode not in {"descent", "random"}:
                raise ValueError("uncertainty_gradient_mode must be 'descent' or 'random'")
            if (self.uncertainty_gradient_step_size is None
                    or not math.isfinite(float(self.uncertainty_gradient_step_size))
                    or float(self.uncertainty_gradient_step_size) <= 0):
                raise ValueError("uncertainty_gradient_step_size must be finite and positive")
            if (isinstance(self.uncertainty_gradient_horizon, bool)
                    or int(self.uncertainty_gradient_horizon) != self.uncertainty_gradient_horizon
                    or self.uncertainty_gradient_horizon < 1):
                raise ValueError("uncertainty_gradient_horizon must be a positive integer")
        elif self.uncertainty_gradient_step_size is not None:
            raise ValueError("uncertainty_gradient_step_size requires uncertainty_gradient_mode")
        if self.uncertainty_gradient_action_rms_max is not None:
            if self.uncertainty_gradient_mode != "descent":
                raise ValueError(
                    "uncertainty_gradient_action_rms_max requires descent mode")
            if self.n_action_steps is None:
                raise ValueError(
                    "uncertainty_gradient_action_rms_max requires explicit n_action_steps")
            if (not math.isfinite(float(self.uncertainty_gradient_action_rms_max))
                    or float(self.uncertainty_gradient_action_rms_max) <= 0):
                raise ValueError(
                    "uncertainty_gradient_action_rms_max must be finite and positive")
        # MultiSample carries its own ms_probe_steps and runs at the chunk level.
        if self.num_samples is not None and self.has_probe:
            raise ValueError("num_samples carries ms_probe_steps; leave pnp_steps/pnp_time_min unset")
        if self.selection_uncertainty_horizon is not None:
            if self.num_samples is None:
                raise ValueError("selection_uncertainty_horizon requires num_samples")
            if (isinstance(self.selection_uncertainty_horizon, bool)
                    or int(self.selection_uncertainty_horizon) != self.selection_uncertainty_horizon
                    or self.selection_uncertainty_horizon < 1):
                raise ValueError("selection_uncertainty_horizon must be a positive integer")
        if self.candidate_seed_scheme not in {None, "stock_slot0_v1"}:
            raise ValueError("candidate_seed_scheme must be None or 'stock_slot0_v1'")
        if self.policy_source_id is not None and not str(self.policy_source_id).strip():
            raise ValueError("policy_source_id must be a non-empty string or None")
        if self.candidate_seed_scheme is not None and self.num_samples is None:
            raise ValueError("candidate_seed_scheme requires num_samples")
        if self.multi_sample_refine_selected:
            if self.num_samples is None:
                raise ValueError("multi_sample_refine_selected requires num_samples")
            if self.selection_uncertainty_horizon is None:
                raise ValueError(
                    "multi_sample_refine_selected requires selection_uncertainty_horizon")
            if self.candidate_seed_scheme != "stock_slot0_v1":
                raise ValueError(
                    "multi_sample_refine_selected requires candidate_seed_scheme='stock_slot0_v1'")
        if self.refine_average and not self.refine:
            raise ValueError("refine_average=True requires refine=True")
        if self.refine_horizon_m is not None:
            if not self.refine:
                raise ValueError("refine_horizon_m requires refine=True")
            if (isinstance(self.refine_horizon_m, bool)
                    or int(self.refine_horizon_m) != self.refine_horizon_m
                    or self.refine_horizon_m < 1):
                raise ValueError("refine_horizon_m must be a positive integer")
        if self.refine_threshold is not None:
            if not self.refine:
                raise ValueError("refine_threshold requires refine=True")
            if not math.isfinite(float(self.refine_threshold)) or self.refine_threshold < 0:
                raise ValueError("refine_threshold must be finite and non-negative")
        if self.refine_uncertainty_horizon is not None:
            if not self.refine or self.refine_threshold is None:
                raise ValueError(
                    "refine_uncertainty_horizon requires thresholded refinement")
            if (isinstance(self.refine_uncertainty_horizon, bool)
                    or int(self.refine_uncertainty_horizon) != self.refine_uncertainty_horizon
                    or self.refine_uncertainty_horizon < 1):
                raise ValueError("refine_uncertainty_horizon must be a positive integer")
        if self.refine_start_chunk is not None:
            if not self.refine:
                raise ValueError("refine_start_chunk requires refine=True")
            if (isinstance(self.refine_start_chunk, bool)
                    or int(self.refine_start_chunk) != self.refine_start_chunk
                    or self.refine_start_chunk < 0):
                raise ValueError("refine_start_chunk must be a non-negative integer")
        if self.refine_tail_decay_end is not None:
            if not self.refine or self.n_action_steps is None:
                raise ValueError(
                    "refine_tail_decay_end requires refine=True and explicit n_action_steps")
            if self.refine_average:
                raise ValueError("refine_tail_decay_end currently requires refine_average=False")
            if (isinstance(self.refine_tail_decay_end, bool)
                    or int(self.refine_tail_decay_end) != self.refine_tail_decay_end
                    or self.refine_tail_decay_end <= self.n_action_steps):
                raise ValueError("refine_tail_decay_end must be an integer above n_action_steps")
        if self.refine_prefix_only:
            if not self.refine or self.n_action_steps is None:
                raise ValueError(
                    "refine_prefix_only requires refine=True and explicit n_action_steps")
            if self.refine_average:
                raise ValueError("refine_prefix_only currently requires refine_average=False")
            if self.refine_tail_decay_end is not None:
                raise ValueError(
                    "refine_prefix_only and refine_tail_decay_end are mutually exclusive")
        if self.refine_inner_strength is not None:
            if not self.refine:
                raise ValueError("refine_inner_strength requires refine=True")
            if self.refine_average:
                raise ValueError(
                    "refine_inner_strength currently requires refine_average=False")
            if (not math.isfinite(float(self.refine_inner_strength))
                    or not 0 < float(self.refine_inner_strength) <= 1):
                raise ValueError("refine_inner_strength must lie in (0, 1]")
        # Probe-derived sinks need a probe.
        for sink in ("save_pcp_features", "save_ahats", "save_time_uncertainty"):
            if getattr(self, sink) and not self.has_probe:
                raise ValueError(f"{sink} requires a probe (its data comes from the probe)")
        if self.save_training_data:
            if not self.has_probe:
                raise ValueError("save_training_data requires a P&P probe")
            if not self.save_generated_chunks:
                raise ValueError("save_training_data requires save_generated_chunks=True")
            if self.n_action_steps != 10:
                raise ValueError("save_training_data requires explicit n_action_steps=10")
        if self.save_uncertainty and not self.has_probe:
            raise ValueError("save_uncertainty=True requires a probe")
        if self.render_lead < 1:
            raise ValueError("render_lead must be >= 1 (the consumed step itself)")
        if self.n_action_steps is not None:
            if (isinstance(self.n_action_steps, bool)
                    or int(self.n_action_steps) != self.n_action_steps
                    or self.n_action_steps < 1):
                raise ValueError("n_action_steps must be a positive integer")
        if (isinstance(self.suffix_probe_samples, bool)
                or int(self.suffix_probe_samples) != self.suffix_probe_samples
                or self.suffix_probe_samples < 0):
            raise ValueError("suffix_probe_samples must be a non-negative integer")
        if self.suffix_probe_samples:
            if not self.has_probe:
                raise ValueError("suffix_probe_samples requires a P&P probe")
            if self.n_action_steps is None:
                raise ValueError(
                    "suffix_probe_samples requires explicit n_action_steps as the prefix boundary")

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
        if logical.get("policy_source_id") is None:
            logical.pop("policy_source_id")
        if logical.get("candidate_seed_scheme") is None:
            logical.pop("candidate_seed_scheme")
        if not logical.get("multi_sample_refine_selected"):
            logical.pop("multi_sample_refine_selected")
        if logical.get("selection_uncertainty_horizon") is None:
            logical.pop("selection_uncertainty_horizon")
        if logical.get("refine_threshold") is None:
            logical.pop("refine_threshold")
        if logical.get("refine_uncertainty_horizon") is None:
            logical.pop("refine_uncertainty_horizon")
        if logical.get("refine_start_chunk") is None:
            logical.pop("refine_start_chunk")
        if logical.get("refine_horizon_m") is None:
            logical.pop("refine_horizon_m")
        if logical.get("refine_tail_decay_end") is None:
            logical.pop("refine_tail_decay_end")
        if not logical.get("refine_prefix_only"):
            logical.pop("refine_prefix_only")
        if logical.get("refine_inner_strength") is None:
            logical.pop("refine_inner_strength")
        if logical.get("uncertainty_gradient_mode") is None:
            logical.pop("uncertainty_gradient_mode")
            logical.pop("uncertainty_gradient_step_size")
            logical.pop("uncertainty_gradient_horizon")
        if logical.get("uncertainty_gradient_action_rms_max") is None:
            logical.pop("uncertainty_gradient_action_rms_max")
        if logical.get("n_action_steps") is None:
            logical.pop("n_action_steps")
        if not logical.get("suffix_probe_samples"):
            logical.pop("suffix_probe_samples")
        return logical


# Behavior-defining fields of RolloutConfig (see RolloutConfig.logical_dict).
LOGICAL_FIELDS = ("pnp_steps", "pnp_k", "pnp_time_min", "action_dim",
                  "refine", "refine_average", "refine_horizon_m", "refine_threshold",
                  "refine_uncertainty_horizon", "refine_start_chunk", "refine_tail_decay_end",
                  "refine_prefix_only", "refine_inner_strength",
                  "uncertainty_gradient_mode", "uncertainty_gradient_step_size",
                  "uncertainty_gradient_horizon",
                  "uncertainty_gradient_action_rms_max",
                  "correction_lambda", "q_gate", "q_ckpt_id",
                  "num_samples", "ms_probe_steps", "selection_uncertainty_horizon",
                  "candidate_set_id", "policy_source_id", "candidate_seed_scheme",
                  "multi_sample_refine_selected",
                  "num_inference_steps",
                  "n_action_steps", "suffix_probe_samples")


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
