"""Model-only calibration of notebook-87 SmolVLA source-chunk reconstruction.

This deliberately performs no simulator rollout and writes no candidate trees.  It separates
the seven active LIBERO action dimensions from SmolVLA's padded output width, measures GPU batch
shape sensitivity, and checks the assumed P&P perturbation-stream offset.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .pnp import PnPRecorder
from .qplanning_fork_pilot import _source_boundary, _source_policy_observation
from .rollout import _draw_chunk_noise, _stack_policy_batches
from .sampler import _temp_strategy
from .smolvla_tree_collection import (
    SMOLVLA_SCHEDULE_K_BY_STEP,
    SMOLVLA_TREE_ACTIONS,
    SMOLVLA_TREE_INTEGRATION_STEPS,
    _clone_generator,
    _load_source_bundle,
    _pnp_config,
    _source_perturb_generator,
    _source_rows,
    load_or_build_smolvla_tree_manifest,
)
from .store import SupabaseStore
from .tap import BatchedRolloutTap


DEFAULT_BATCH_SIZES = (1, 8, 9)
DEFAULT_DRAW_OFFSETS = tuple(range(-5, 6))


def _calibration_items(items: list[dict], count: int) -> list[dict]:
    """Deterministic round-robin sample across suite, source outcome, and root strategy."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        buckets[(item["suite"], bool(item["source_success"]),
                 item["selection_strategy"])].append(item)
    for values in buckets.values():
        values.sort(key=lambda item: item["selection_tiebreak"])
    selected = []
    keys = sorted(buckets, key=str)
    while len(selected) < int(count):
        advanced = False
        for key in keys:
            if buckets[key] and len(selected) < int(count):
                selected.append(buckets[key].pop(0))
                advanced = True
        if not advanced:
            break
    if len(selected) != int(count):
        raise ValueError(f"could select only {len(selected)}/{count} calibration roots")
    return selected


def _generator_with_draw_offset(policy, device, source: dict, chunk_idx: int,
                                draw_offset: int):
    draws_per_chunk = sum(map(int, SMOLVLA_SCHEDULE_K_BY_STEP))
    total_draws = int(chunk_idx) * draws_per_chunk + int(draw_offset)
    if total_draws < 0:
        raise ValueError("P&P draw offset precedes the start of the source stream")
    # Build the stream at chunk zero, then advance by the exact requested tensor draws.
    generator = _source_perturb_generator(
        policy, device, perturb_seed=int(np.asarray(source["perturb_seed"])),
        completed_chunks=0)
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    for _ in range(total_draws):
        torch.empty(shape, device=device).normal_(generator=generator)
    return generator


def _predict(policy, batch, device, *, item: dict, source: dict,
             draw_offsets: list[int], positions: list[float]) -> np.ndarray:
    if len(draw_offsets) != len(positions) or not draw_offsets:
        raise ValueError("draw_offsets and positions must be nonempty and aligned")
    lane_count = len(draw_offsets)
    noises = torch.cat([
        _draw_chunk_noise(policy, device, int(source["noise_seed"]))
        for _ in range(lane_count)], dim=0)
    batches = _stack_policy_batches([batch for _ in range(lane_count)])
    generators = [
        _generator_with_draw_offset(
            policy, device, source, int(item["chunk_idx"]), offset)
        for offset in draw_offsets]
    recorders = [PnPRecorder() for _ in range(lane_count)]
    for recorder in recorders:
        recorder.new_episode()
    tap = BatchedRolloutTap(
        _pnp_config(), recorders, generators, device, policy.model._pnp.action_dim)
    previous_steps, previous_position = (
        policy.model._pnp.num_steps, policy.model._pnp.chunk_pos)
    policy.model._pnp.num_steps = SMOLVLA_TREE_INTEGRATION_STEPS
    policy.model._pnp.chunk_pos = list(map(float, positions))
    try:
        with _temp_strategy(policy.model, tap), torch.no_grad():
            chunks = policy.predict_action_chunk(batches, noise=noises)
    finally:
        policy.model._pnp.num_steps = previous_steps
        policy.model._pnp.chunk_pos = previous_position
    return chunks.detach().float().cpu().numpy()


def _error_row(prediction: np.ndarray, target: np.ndarray, *, active_dim: int,
               item: dict, mode: str, batch_size: int, draw_offset: int,
               position_scheme: str, duplicate_lane_rms: float = float("nan")) -> dict:
    delta = np.asarray(prediction, np.float32) - np.asarray(target, np.float32)
    first = delta[:SMOLVLA_TREE_ACTIONS]
    active = first[:, :active_dim]
    padded = first[:, active_dim:]

    def rms(value):
        value = np.asarray(value, np.float32)
        return float(np.sqrt(np.mean(value ** 2))) if value.size else float("nan")

    def max_abs(value):
        value = np.asarray(value, np.float32)
        return float(np.max(np.abs(value))) if value.size else float("nan")

    return {
        "source_rollout_id": item["source_rollout_id"],
        "suite": item["suite"], "task_idx": int(item["task_idx"]),
        "episode_idx": int(item["episode_idx"]), "chunk_idx": int(item["chunk_idx"]),
        "source_success": bool(item["source_success"]),
        "selection_strategy": item["selection_strategy"],
        "mode": mode, "batch_size": int(batch_size), "draw_offset": int(draw_offset),
        "position_scheme": position_scheme,
        "active_dim": int(active_dim), "output_dim": int(delta.shape[-1]),
        "first10_active_rms": rms(active),
        "first10_active_max_abs": max_abs(active),
        "first10_padded_rms": rms(padded),
        "first10_padded_max_abs": max_abs(padded),
        "first10_all_rms": rms(first),
        "first10_all_max_abs": max_abs(first),
        "full50_active_rms": rms(delta[:, :active_dim]),
        "full50_active_max_abs": max_abs(delta[:, :active_dim]),
        "duplicate_lane_rms": float(duplicate_lane_rms),
    }


def _print_summary(frame: pd.DataFrame) -> None:
    standard = frame[frame["mode"].eq("batch_shape")]
    summary = (standard.groupby("batch_size")[
        ["first10_active_rms", "first10_active_max_abs", "first10_padded_rms",
         "first10_all_rms", "duplicate_lane_rms"]]
        .quantile([0.5, 0.9, 0.95, 1.0]).round(6))
    print("\nBatch-shape reconstruction quantiles (active = the seven executed LIBERO dims):")
    print(summary.to_string(), flush=True)

    offsets = frame[frame["mode"].eq("draw_offset")].copy()
    best = offsets.loc[offsets.groupby("source_rollout_id")[
        "first10_active_rms"].idxmin()]
    print("\nBest P&P draw offset per root (zero should win if stream accounting is right):")
    print(best[["suite", "task_idx", "episode_idx", "chunk_idx", "draw_offset",
                "first10_active_rms"]].to_string(index=False), flush=True)
    print("best-offset counts:", best["draw_offset"].value_counts().sort_index().to_dict(),
          flush=True)

    positions = frame[frame["mode"].eq("position_scheme")]
    print("\nTime-conditioning comparison:")
    print(positions.groupby("position_scheme")["first10_active_rms"].agg(
        ["median", "mean", "max"]).round(6).to_string(), flush=True)


def run_smolvla_reconstruction_calibration(*, n_roots: int = 20,
                                           batch_sizes=DEFAULT_BATCH_SIZES,
                                           draw_offsets=DEFAULT_DRAW_OFFSETS,
                                           output_csv: str | None = None,
                                           store=None) -> pd.DataFrame:
    """Calibrate source reconstruction without creating environments or database rows."""
    from . import models

    if not 1 <= int(n_roots) <= 100:
        raise ValueError("n_roots must lie in [1, 100]")
    batch_sizes = tuple(map(int, batch_sizes))
    draw_offsets = tuple(map(int, draw_offsets))
    if not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("batch_sizes must contain positive integers")
    if 0 not in draw_offsets:
        raise ValueError("draw_offsets must include zero")

    store = store or SupabaseStore()
    document = load_or_build_smolvla_tree_manifest(store)
    items = _calibration_items(document["payload"]["items"], int(n_roots))
    source_rows = {str(row["rollout_id"]): row for row in _source_rows(store)}
    policy, preprocess, _ = models.load_smolvla()
    device = models.default_device()
    active_dim = int(policy.model._pnp.action_dim)
    rows = []

    print({
        "calibration_roots": len(items), "batch_sizes": batch_sizes,
        "draw_offsets": draw_offsets, "active_action_dim": active_dim,
        "generated_output_dim": int(policy.config.max_action_dim),
        "writes_trees": False,
    }, flush=True)
    for ordinal, item in enumerate(items, 1):
        bundle = _load_source_bundle(store, source_rows[item["source_rollout_id"]])
        source = _source_boundary(bundle, item)
        source["perturb_seed"] = np.asarray(bundle["arrays"]["perturb_seed"])
        instruction = str(np.asarray(bundle["arrays"]["boundary/instruction"])[
            int(source["boundary_index"])])
        policy_observation = _source_policy_observation(
            bundle["arrays"], source["boundary_index"], instruction)
        batch = preprocess(policy_observation)
        target = np.asarray(source["policy_chunk"], np.float32)
        generated_denominator = max(1, round(
            int(item["source_max_steps"]) / int(policy.config.chunk_size)))
        executed_denominator = max(1, round(
            int(item["source_max_steps"]) / SMOLVLA_TREE_ACTIONS))
        source_position = min(int(item["chunk_idx"]) / generated_denominator, 1.0)
        executed_position = min(int(item["chunk_idx"]) / executed_denominator, 1.0)

        for batch_size in batch_sizes:
            predictions = _predict(
                policy, batch, device, item=item, source=source,
                draw_offsets=[0] * batch_size,
                positions=[source_position] * batch_size)
            lane_delta = predictions - predictions[0:1]
            duplicate_rms = float(np.sqrt(np.mean(
                lane_delta[:, :SMOLVLA_TREE_ACTIONS, :active_dim] ** 2)))
            rows.append(_error_row(
                predictions[0], target, active_dim=active_dim, item=item,
                mode="batch_shape", batch_size=batch_size, draw_offset=0,
                position_scheme="generated_chunk_width", duplicate_lane_rms=duplicate_rms))

        valid_offsets = [offset for offset in draw_offsets
                         if int(item["chunk_idx"]) * sum(
                             map(int, SMOLVLA_SCHEDULE_K_BY_STEP)) + offset >= 0]
        offset_predictions = _predict(
            policy, batch, device, item=item, source=source,
            draw_offsets=valid_offsets,
            positions=[source_position] * len(valid_offsets))
        for lane, offset in enumerate(valid_offsets):
            rows.append(_error_row(
                offset_predictions[lane], target, active_dim=active_dim, item=item,
                mode="draw_offset", batch_size=len(valid_offsets), draw_offset=offset,
                position_scheme="generated_chunk_width"))

        position_predictions = _predict(
            policy, batch, device, item=item, source=source,
            draw_offsets=[0, 0], positions=[source_position, executed_position])
        for lane, scheme in enumerate(("generated_chunk_width", "executed_action_width")):
            rows.append(_error_row(
                position_predictions[lane], target, active_dim=active_dim, item=item,
                mode="position_scheme", batch_size=2, draw_offset=0,
                position_scheme=scheme))

        current = rows[-(len(valid_offsets) + len(batch_sizes) + 2):]
        exact = next(row for row in current
                     if row["mode"] == "batch_shape" and row["batch_size"] == batch_sizes[-1])
        best_offset = min((row for row in current if row["mode"] == "draw_offset"),
                          key=lambda row: row["first10_active_rms"])
        print(
            f"[calibration] {ordinal}/{len(items)} {item['suite']} task={item['task_idx']} "
            f"ep={item['episode_idx']} chunk={item['chunk_idx']} | "
            f"B{batch_sizes[-1]} active RMS={exact['first10_active_rms']:.6f}, "
            f"padded RMS={exact['first10_padded_rms']:.6f} | "
            f"best draw offset={best_offset['draw_offset']:+d} "
            f"({best_offset['first10_active_rms']:.6f})",
            flush=True)

    frame = pd.DataFrame(rows)
    _print_summary(frame)
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"saved {path}", flush=True)
    return frame
