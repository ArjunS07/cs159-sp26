"""Independent, Jeff-matched SmolVLA critic-gradient evaluation.

This module deliberately does not import the PCP sweep/proposal drivers.  It uses the
shared policy, critic architecture definitions, rollout engine, and storage schema, but
reimplements checkpoint loading, bounded action-space gradients, arm construction, and
the experiment loop.  The separation makes it useful for detecting implementation drift
between the established SmolVLA experiments and the newer PCP search code.
"""
from __future__ import annotations

import collections
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .config import PERTURB_SEED_MASK, RolloutConfig, SMOLVLA_REPO_ID
from .pnp import run_probe
from .qplanning_critic.config import QPlanningModelConfig
from .qplanning_critic.model import pool_prefix_tokens
from .smolvla_anchored_critic import AnchoredCritic
from .smolvla_scalar_returns import LateFusionScalarCritic, ScalarCritic
from .tap import BatchedRolloutTap


EXPERIMENT = "smolvla-critic-guidance-independent-jeff400-v1"
PARENT_STEPS = (1, 2, 3)
PARENT_K_BY_STEP = (3, 1, 1)
GUIDANCE_STEP = 3
EXECUTED_ACTIONS = 10
INTEGRATION_STEPS = 10


ARM_LIBRARY = {
    "stock_live": dict(kind="stock", critic=None, mode="none", radius=0.0),
    "pnp_parent": dict(kind="pnp", critic=None, mode="none", radius=0.0),
    "frozen_mc_ascent_r002": dict(
        kind="pnp", critic="frozen_mc", mode="ascent", radius=0.02),
    "frozen_mc_descent_r002": dict(
        kind="pnp", critic="frozen_mc", mode="descent", radius=0.02),
    "td_ascent_r002": dict(kind="pnp", critic="td", mode="ascent", radius=0.02),
    "random_r002": dict(kind="pnp", critic=None, mode="random", radius=0.02),
    "frozen_mc_ascent_r006": dict(
        kind="pnp", critic="frozen_mc", mode="ascent", radius=0.06),
    "td_ascent_r006": dict(kind="pnp", critic="td", mode="ascent", radius=0.06),
    "cnn_mc_ascent_r002": dict(
        kind="pnp", critic="cnn_mc", mode="ascent", radius=0.02),
}

DEFAULT_ARMS = (
    "stock_live",
    "pnp_parent",
    "frozen_mc_ascent_r002",
    "td_ascent_r002",
    "frozen_mc_descent_r002",
    "random_r002",
)


def _method_name(arm_name: str) -> str:
    aliases = {
        "stock_live": "cg_stock",
        "pnp_parent": "cg_parent",
        "frozen_mc_ascent_r002": "cg_mc_up_002",
        "frozen_mc_descent_r002": "cg_mc_down_002",
        "td_ascent_r002": "cg_td_up_002",
        "random_r002": "cg_random_002",
        "frozen_mc_ascent_r006": "cg_mc_up_006",
        "td_ascent_r006": "cg_td_up_006",
        "cnn_mc_ascent_r002": "cg_cnn_up_002",
    }
    return aliases[arm_name]


def resolve_arms(names: Iterable[str]) -> list[dict]:
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("arm names must be nonempty and unique")
    unknown = sorted(set(names) - set(ARM_LIBRARY))
    if unknown:
        raise ValueError(f"unknown critic-guidance arms: {unknown}")
    return [dict(name=name, method=_method_name(name), **ARM_LIBRARY[name]) for name in names]


def load_critic_checkpoint(path: str | Path, device: str):
    """Load one frozen scalar/late-fusion/anchored critic without sweep code."""
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)

    def scalar(architecture):
        architecture = dict(architecture)
        family = architecture.pop("model_family", "scalar_decoder")
        dimensions = {
            key: architecture.pop(key)
            for key in ("prefix_dim", "robot_dim", "proprio_dim")
        }
        if family == "late_fusion_scalar":
            cls = LateFusionScalarCritic
        elif family == "scalar_decoder":
            cls = ScalarCritic
        else:
            raise ValueError(f"unsupported scalar critic family: {family}")
        return cls(**dimensions, config=QPlanningModelConfig(**architecture))

    architecture = saved["architecture"]
    family = architecture.get("model_family", "scalar_decoder")
    if family.startswith("anchored"):
        residual_family = "temporal_cnn" if "cnn" in family else "transformer"
        model = AnchoredCritic(
            scalar(architecture["base_architecture"]), residual_family,
            width=architecture["width"], delta_scale=architecture["delta_scale"])
    else:
        model = scalar(architecture)
    model.load_state_dict(saved["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, sha256


def bounded_standardized_move(current, anchor, gradient, valid, std, radius):
    """Take one normalized ascent direction inside a standardized-RMS trust ball.

    ``gradient`` is dQ/d(action) in native policy coordinates.  The direction is
    normalized after converting it to standardized action coordinates, then the final
    displacement is projected into the same radius around ``anchor``.
    """
    mask = valid[..., None]
    direction = gradient * std * mask
    denominator = (valid.sum(1) * current.shape[-1]).clamp_min(1).to(current.dtype)
    norm = (direction.square().sum((1, 2)) / denominator).sqrt()
    usable = torch.isfinite(direction).all((1, 2)) & (norm > 1e-12)
    safe_norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-12)
    direction = torch.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
    direction = direction / safe_norm[:, None, None]
    delta = (current - anchor) / std + float(radius) * direction * usable[:, None, None]
    delta = delta * mask
    size = (delta.square().sum((1, 2)) / denominator).sqrt()
    delta = delta * torch.minimum(
        torch.ones_like(size), float(radius) / size.clamp_min(1e-12))[:, None, None]
    return anchor + delta * std, usable


class IndependentCriticGuidanceTap(BatchedRolloutTap):
    """P&P parent plus one clean-action Q-gradient update at Euler step 3."""

    def __init__(self, *, arm, critic, correction_std, emit, episodes, observations,
                 steps, chunk_indices, lane_ids, **kwargs):
        super().__init__(**kwargs)
        from .libero_env import obs_to_policy
        from .rollout import _raw_robot_state

        self.arm = arm
        self.critic = critic
        self.correction_std = correction_std
        self.emit = emit
        self.chunk_indices = list(map(int, chunk_indices))
        self.lane_ids = list(map(int, lane_ids))
        self.remaining = [ep["max_steps"] - step for ep, step in zip(episodes, steps)]
        self.robot = torch.as_tensor(np.stack([
            np.r_[_raw_robot_state(obs), remaining / ep["max_steps"]]
            for obs, remaining, ep in zip(observations, self.remaining, episodes)
        ]), device=kwargs["device"], dtype=torch.float32)
        self.proprio = torch.stack([
            obs_to_policy(obs, ep["task_desc"])["observation.state"]
            for obs, ep in zip(observations, episodes)
        ]).to(kwargs["device"])

    def _score(self, prefix, prefix_valid, anchor, valid):
        if isinstance(self.critic, AnchoredCritic):
            features, baseline = self.critic.reference_features(
                prefix, prefix_valid, self.robot, self.proprio, anchor, valid)
            return lambda action: self.critic.logits_from_features(
                features, baseline, action, valid, anchor).reshape(-1).sigmoid()
        return lambda action: self.critic(
            prefix, prefix_valid, self.robot, self.proprio, action, valid
        ).reshape(-1).sigmoid()

    def step(self, x_t, s, vf, ctx):
        probe = run_probe(
            x_t, s, vf, k=self.config.probe_k(ctx.step), adim=self.adim,
            generators=self.generators, record_telemetry=False)
        if self._pending_variable_probe is not None:
            raise RuntimeError("a variable-K probe was not finalized")
        self._pending_variable_probe = (probe, int(ctx.step))

        if int(ctx.step) != GUIDANCE_STEP or self.arm["mode"] == "none":
            return probe.x_acc

        full = probe.z_hat_full.detach()
        anchor = full[:, :EXECUTED_ACTIONS, :7].clone()
        valid = torch.arange(EXECUTED_ACTIONS, device=full.device)[None] < torch.tensor(
            self.remaining, device=full.device)[:, None]
        before = after = None
        bad = torch.zeros(len(anchor), dtype=torch.bool, device=full.device)

        if self.arm["mode"] in ("ascent", "descent"):
            prefix, prefix_valid = pool_prefix_tokens(
                ctx.prefix_embeddings.detach(), ctx.prefix_pad_masks.detach(), 128)
            score = self._score(prefix, prefix_valid, anchor, valid)
            with torch.enable_grad():
                candidate = anchor.clone().requires_grad_(True)
                before = score(candidate)
                gradient, = torch.autograd.grad(before.sum(), candidate)
            sign = 1.0 if self.arm["mode"] == "ascent" else -1.0
            candidate, usable = bounded_standardized_move(
                anchor, anchor, sign * gradient.detach(), valid,
                self.correction_std, self.arm["radius"])
            bad = ~usable | ~torch.isfinite(before.detach())
            candidate = torch.where(bad[:, None, None], anchor, candidate)
            with torch.no_grad():
                after = score(candidate)
        elif self.arm["mode"] == "random":
            random_direction = torch.empty_like(anchor)
            for lane, chunk_index in enumerate(self.chunk_indices):
                seed = (int(self.generators[lane].initial_seed()) ^ PERTURB_SEED_MASK
                        ^ ((chunk_index + 1) * 1_000_003))
                generator = torch.Generator(device=full.device).manual_seed(seed % (2**63 - 1))
                random_direction[lane].normal_(generator=generator)
            candidate, usable = bounded_standardized_move(
                anchor, anchor, random_direction / self.correction_std,
                valid, self.correction_std, self.arm["radius"])
            bad = ~usable
            candidate = torch.where(bad[:, None, None], anchor, candidate)
        else:
            raise ValueError(f"unsupported guidance mode: {self.arm['mode']}")

        delta = candidate.detach() - anchor
        if not torch.isfinite(delta).all():
            raise RuntimeError("critic guidance produced a non-finite action displacement")
        update = torch.zeros_like(full)
        update[:, :EXECUTED_ACTIONS, :7] = delta
        denominator = (valid.sum(1) * 7).clamp_min(1)
        native_rms = (delta.square().sum((1, 2)) / denominator).sqrt()
        standardized_rms = (
            ((delta / self.correction_std).square().sum((1, 2)) / denominator).sqrt())
        before_cpu = before.detach().cpu().numpy() if before is not None else [None] * len(delta)
        after_cpu = after.detach().cpu().numpy() if after is not None else [None] * len(delta)
        diagnostics = torch.stack([native_rms, standardized_rms, bad.float()], 1).cpu().numpy()
        for lane, values in enumerate(diagnostics):
            clean = lambda value: (
                float(value) if value is not None and math.isfinite(float(value)) else None)
            self.emit(self.lane_ids[lane], {
                "chunk_index": self.chunk_indices[lane],
                "euler_step": int(ctx.step),
                "mode": self.arm["mode"],
                "radius": float(self.arm["radius"]),
                "q_before": clean(before_cpu[lane]),
                "q_after": clean(after_cpu[lane]),
                "native_rms": float(values[0]),
                "standardized_rms": float(values[1]),
                "bad_gradient": bool(values[2]),
            })
        # A clean-action displacement maps back to the current latent with coefficient (1-s).
        return probe.x_acc + (1.0 - float(s)) * update

    def after_selected_vfield(self, x_t, s, velocity, ctx):
        # The reproduction does not need to copy P&P traces to CPU.  Clearing the pending
        # variable-K probe changes no sampler arithmetic.
        if self._pending_variable_probe is not None:
            if self._pending_variable_probe[1] != int(ctx.step):
                raise RuntimeError("variable-K probe step mismatch")
            self._pending_variable_probe = None


def _identity(ep_or_row) -> tuple:
    return (
        str(ep_or_row["suite"]),
        int(ep_or_row["task_idx"]),
        int(ep_or_row.get("ep_idx", ep_or_row.get("episode_idx", 0))),
        str(ep_or_row.get("init_state_hash") or ""),
    )


def _historical_reference(store, episodes, *, experiment, method, config):
    from .smolvla_followup_experiments import _completed_rows

    wanted = {_identity(ep) for ep in episodes}
    rows = _completed_rows(store, experiment=experiment, method=method, config=config)
    matched = {_identity(row): bool(row["success"]) for row in rows if _identity(row) in wanted}
    missing = wanted - set(matched)
    if missing:
        raise ValueError(
            f"historical experiment {experiment} is missing {len(missing)} identities; "
            f"first missing={sorted(missing)[0]}")
    return matched


def _arm_config(arm):
    common = dict(
        n_action_steps=EXECUTED_ACTIONS,
        num_inference_steps=INTEGRATION_STEPS,
        skip_unused_renders=True,
        render_lead=2,
        save_trajectory=True,
        video="off",
    )
    if arm["kind"] == "stock":
        return RolloutConfig(**common)
    return RolloutConfig(
        pnp_steps=PARENT_STEPS,
        pnp_k=max(PARENT_K_BY_STEP),
        pnp_k_by_step=PARENT_K_BY_STEP,
        refine=True,
        save_uncertainty=False,
        **common,
    )


def _telemetry_summary(arm, records):
    def mean(key):
        values = [row[key] for row in records if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "implementation": "independent_critic_guidance_v1",
        "arm": arm,
        "records": records,
        "n_updates": len(records),
        "mean_pre_q": mean("q_before"),
        "mean_post_q": mean("q_after"),
        "mean_delta_q": (
            mean("q_after") - mean("q_before")
            if mean("q_after") is not None and mean("q_before") is not None else None),
        "mean_native_rms": mean("native_rms"),
        "mean_standardized_rms": mean("standardized_rms"),
        "n_bad_gradients": sum(bool(row["bad_gradient"]) for row in records),
    }


def run_independent_critic_guidance_worker(
        *, checkpoint_paths: dict[str, str], shard_count: int = 2, shard_index: int = 0,
        arm_names: Iterable[str] = DEFAULT_ARMS, rollout_batch_size: int = 8,
        episode_limit: int | None = None, report_every_identities: int = 10,
        experiment: str = EXPERIMENT):
    """Run exact-state, seed-stream-0 critic guidance with matched historical columns."""
    from . import models
    from .experiments import (
        SMOLVLA_LIBERO_EXPERIMENT, _prepare_libero_episodes,
        build_smolvla_libero_methods, format_matched_progress_table, identity_shard)
    from .libero_env import make_env
    from .rollout import run_episode_batch
    from .smolvla_followup_experiments import (
        SMOLVLA_SCHEDULE_EXPERIMENT, build_smolvla_schedule_method)
    from .smolvla_jeff_replication import configure_precision
    from .store import SupabaseStore, gather_provenance

    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be positive")
    arm_names = tuple(arm_names)
    arms = resolve_arms(arm_names)
    needed = {arm["critic"] for arm in arms if arm["critic"] is not None}
    # Every radius is expressed in the frozen-MC action standardization, including TD,
    # random controls, and runs that otherwise do not score with frozen MC.
    needed.add("frozen_mc")
    missing_paths = sorted(needed - set(checkpoint_paths))
    if missing_paths:
        raise ValueError(f"missing critic checkpoint paths: {missing_paths}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")

    configure_precision()
    all_episodes = _prepare_libero_episodes()
    for episode in all_episodes:
        episode["behavior_seed_index"] = 0
    episodes = identity_shard(all_episodes, shard_count, shard_index)
    if episode_limit is not None:
        if episode_limit < 1:
            raise ValueError("episode_limit must be positive or None")
        episodes = episodes[:int(episode_limit)]

    store = SupabaseStore()
    stock_method, stock_config = build_smolvla_libero_methods()[0]
    parent_method, parent_config = build_smolvla_schedule_method()
    references = {
        "hist stock": _historical_reference(
            store, all_episodes, experiment=SMOLVLA_LIBERO_EXPERIMENT,
            method=stock_method, config=stock_config),
        "hist k311": _historical_reference(
            store, all_episodes, experiment=SMOLVLA_SCHEDULE_EXPERIMENT,
            method=parent_method, config=parent_config),
    }

    critics = {}
    checkpoint_shas = {}
    for name in sorted(needed):
        critics[name], checkpoint_shas[name] = load_critic_checkpoint(
            checkpoint_paths[name], "cuda")
    correction_std = critics["frozen_mc"].action_std.detach()
    policy, preprocess, postprocess = models.load_smolvla(device="cuda")

    methods = [(arm["method"], _arm_config(arm)) for arm in arms]
    method_names = [method for method, _ in methods]
    arm_by_method = {arm["method"]: arm for arm in arms}
    done = store.existing_keys(experiment, status="completed")
    identity_outcomes = collections.defaultdict(dict)
    existing = store.fetch_all(
        "rollouts", "rollout_id,suite,task_idx,episode_idx,init_state_hash,status,success,method",
        configure=lambda query: query.eq("experiment", experiment).eq("status", "completed"),
        order_by=("rollout_id",),
    )
    wanted = {_identity(ep) for ep in episodes}
    for row in existing:
        key = _identity(row)
        if key in wanted and row["method"] in method_names:
            identity_outcomes[key][row["method"]] = bool(row["success"])

    provenance = gather_provenance(model_repo_id=SMOLVLA_REPO_ID)
    provenance["policy_model"] = "smolvla"
    store.start_run(
        driver="independent_smolvla_critic_guidance",
        benchmark="libero",
        experiment=experiment,
        provenance=provenance,
        config={
            "implementation": "independent_critic_guidance_v1",
            "shard_count": shard_count,
            "shard_index": shard_index,
            "arm_names": list(arm_names),
            "identities": len(episodes),
            "checkpoint_sha256": checkpoint_shas,
            "integration_steps": INTEGRATION_STEPS,
            "n_action_steps": EXECUTED_ACTIONS,
            "pnp_steps": list(PARENT_STEPS),
            "pnp_k_by_step": list(PARENT_K_BY_STEP),
            "guidance_step": GUIDANCE_STEP,
            "precision": {"tf32": False, "matmul": "highest"},
        },
    )

    complete_count = sum(set(method_names).issubset(values) for values in identity_outcomes.values())
    next_report = (complete_count // report_every_identities + 1) * report_every_identities
    completed_new = 0
    print({
        "experiment": experiment,
        "worker": f"{shard_index}/{shard_count}",
        "identities_in_shard": len(episodes),
        "arms": [arm["name"] for arm in arms],
        "checkpoint_sha256": checkpoint_shas,
        "parent": "P&P steps 1,2,3 / K 3,1,1; Q update at step 3",
        "radius_units": "frozen-MC standardized RMS over first 10 policy actions",
        "historical_references": list(references),
    }, flush=True)

    def report():
        print(format_matched_progress_table(
            identity_outcomes, method_names,
            {name: {key: value for key, value in ref.items() if key in wanted}
             for name, ref in references.items()}), flush=True)

    try:
        task_keys = sorted({(ep["suite"], ep["task_idx"]) for ep in episodes})
        for task_key in task_keys:
            task_episodes = [ep for ep in episodes
                             if (ep["suite"], ep["task_idx"]) == task_key]
            envs = [make_env(task_episodes[0]["bddl_path"])
                    for _ in range(min(rollout_batch_size, len(task_episodes)))]
            try:
                for method, config in methods:
                    arm = arm_by_method[method]
                    todo = list(store.iter_todo(
                        experiment, task_episodes, [(method, config)], done=done))
                    for start in range(0, len(todo), rollout_batch_size):
                        group = todo[start:start + rollout_batch_size]
                        if not group:
                            continue
                        telemetry = [[] for _ in group]
                        tap_factory = None
                        if arm["kind"] == "pnp":
                            def tap_factory(*, _arm=arm, _telemetry=telemetry, **kwargs):
                                critic = critics.get(_arm["critic"])
                                return IndependentCriticGuidanceTap(
                                    arm=_arm, critic=critic, correction_std=correction_std,
                                    emit=lambda lane, item: _telemetry[lane].append(item),
                                    **kwargs)
                        results = run_episode_batch(
                            envs[:len(group)], [item[0] for item in group], policy,
                            preprocess, postprocess, "cuda", config,
                            tap_factory=tap_factory)
                        for lane, ((episode, _, _, rollout_id), result) in enumerate(
                                zip(group, results)):
                            if arm["kind"] == "pnp":
                                result["q_guidance_telemetry"] = _telemetry_summary(
                                    arm, telemetry[lane])
                            store.log_result(rollout_id, episode, method, config, result)
                            completed_new += 1
                            if result["status"] == "completed":
                                key = _identity(episode)
                                was_complete = set(method_names).issubset(identity_outcomes[key])
                                identity_outcomes[key][method] = bool(result["success"])
                                is_complete = set(method_names).issubset(identity_outcomes[key])
                                if is_complete and not was_complete:
                                    complete_count += 1
                            else:
                                print(
                                    f"ERROR {episode['suite']} task={episode['task_idx']} "
                                    f"ep={episode['ep_idx']} {method}: {result['error_msg']}",
                                    flush=True)
                            print(
                                f"[critic-repro] {completed_new} new | {method} | "
                                f"{episode['suite']} task {episode['task_idx']} "
                                f"ep {episode['ep_idx']} | success={bool(result['success'])}",
                                flush=True)
                            if complete_count >= next_report:
                                while complete_count >= next_report:
                                    next_report += report_every_identities
                                report()
            finally:
                for env in envs:
                    env.close()
    except BaseException:
        store.finish_run(status="failed", n_rollouts=completed_new)
        raise
    store.finish_run(status="completed", n_rollouts=completed_new)
    report()
    return {
        "experiment": experiment,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "identities": len(episodes),
        "new_rollouts": completed_new,
        "complete_identities": complete_count,
        "arms": [arm["name"] for arm in arms],
        "checkpoint_sha256": checkpoint_shas,
    }
