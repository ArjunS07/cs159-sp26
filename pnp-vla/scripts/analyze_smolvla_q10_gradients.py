"""Audit stock-action Q gradients against recorded SmolVLA Q10 branch outcomes.

This does not execute gradient-edited actions. Candidate outcomes are one rollout
each, so agreement is a directional diagnostic rather than a true gradient label.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.train_smolvla_combined_local import _load_local_credentials  # noqa: E402
from pnp.qplanning_critic.config import QPlanningModelConfig  # noqa: E402
from pnp.qplanning_critic.model import QPlanningCritic  # noqa: E402
from pnp.smolvla_combined_success import (  # noqa: E402
    ROOT_CACHE_FORMAT, load_or_create_combined_snapshot, prepare_combined_root_cache,
)
from pnp.smolvla_success_critic import TimedRoots  # noqa: E402
from pnp.smolvla_tree_bellman_finetune import _root_batch, _to  # noqa: E402
from pnp.store import SupabaseStore  # noqa: E402


CHECKPOINT_KEY = (
    "smolvla_q10/checkpoints/"
    "delta_ranker_v1_preaction_671b5b211099997fc83d1277_step2000.pt")
CHECKPOINT_SHA256 = "ec6bea84a824a6ab424f2d5d2a8bc3e08173b350653bc962c8cb954938a506ef"


def load_ranker(data: bytes, device: torch.device):
    if hashlib.sha256(data).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("ranker checkpoint SHA-256 mismatch")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    architecture = dict(payload["architecture"])
    prefix_dim = int(architecture.pop("prefix_dim"))
    robot_dim = int(architecture.pop("robot_dim"))
    proprio_dim = int(architecture.pop("proprio_dim"))
    if (architecture["action_horizon"] != 10 or architecture["action_dim"] != 14
            or float(payload["delta_scale"]) != 100):
        raise ValueError("expected the fixed-anchor Q10 delta ranker")
    model = QPlanningCritic(
        prefix_dim=prefix_dim, robot_dim=robot_dim,
        proprio_dim=proprio_dim, config=QPlanningModelConfig(**architecture))
    model.value_head = nn.Linear(model.config.width, 1)
    model.load_state_dict(payload["model"])
    model.to(device).eval().requires_grad_(False)
    mean = torch.as_tensor(payload["action_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(payload["action_std"], device=device, dtype=torch.float32)
    if mean.shape != (7,) or std.shape != (7,) or not bool((std > 0).all()):
        raise ValueError("checkpoint has invalid action normalization")
    return model, payload, mean, std


def fixed_anchor_logits(model, batch, candidate, mean, std, scale):
    """Differentiate candidate only; the stored stock chunk is a fixed anchor."""
    stock = batch["action"][:, 0].detach()
    if candidate.shape != stock.shape:
        raise ValueError("candidate shape must match stock [roots,10,7]")
    features = torch.cat([
        (stock - mean) / std,
        (candidate - stock) / std * scale,
    ], dim=-1)
    return model(batch["prefix"], batch["pad"], batch["robot"],
                 batch["proprio"], features,
                 batch["action_valid"][:, 0]).squeeze(-1)


def score_all_candidates(model, batch, mean, std, scale):
    roots, candidates = batch["action"].shape[:2]
    repeat = lambda x: x.repeat_interleave(candidates, 0)
    stock = batch["action"][:, :1]
    features = torch.cat([
        ((stock - mean) / std).expand_as(batch["action"]),
        (batch["action"] - stock) / std * scale,
    ], dim=-1)
    return model(
        repeat(batch["prefix"]), repeat(batch["pad"]),
        repeat(batch["robot"]), repeat(batch["proprio"]),
        features.reshape(roots * candidates, 10, 14),
        batch["action_valid"].reshape(roots * candidates, 10),
    ).reshape(roots, candidates)


def batch_signals(model, batch, mean, std, scale):
    stock = batch["action"][:, 0].detach()
    # The candidate is a separate leaf; otherwise stock-anchor subtraction
    # gives an identically zero derivative for candidate zero.
    candidate = stock.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = fixed_anchor_logits(model, batch, candidate, mean, std, scale)
        gradient, = torch.autograd.grad(logits.sum(), candidate)
    with torch.no_grad():
        scores = score_all_candidates(model, batch, mean, std, scale)
    mask = batch["action_valid"][:, 0, :, None].float()
    displacement = (batch["action"] - stock.detach()[:, None]) * mask[:, None]
    gradient = gradient * mask
    projection = (gradient[:, None] * displacement).sum((-1, -2))
    # Express norms and cosines in training-standardized action coordinates.
    grad_standard = gradient * std
    delta_standard = displacement / std
    grad_norm = grad_standard.square().sum((-1, -2)).sqrt()
    delta_norm = delta_standard.square().sum((-1, -2)).sqrt()
    cosine = projection / (grad_norm[:, None] * delta_norm).clamp_min(1e-12)
    return {"projection": projection.detach().cpu().numpy(),
            "scores": (scores - scores[:, :1]).detach().cpu().numpy(),
            "cosine": cosine.detach().cpu().numpy(),
            "distance": delta_norm.detach().cpu().numpy(),
            "gradient_norm": grad_norm.detach().cpu().numpy(),
            "gradient_by_dim": grad_standard.abs().mean(1).detach().cpu().numpy()}


def _pair_accuracy(values: np.ndarray, labels: np.ndarray) -> float | None:
    good, bad = values[labels], values[~labels]
    if not len(good) or not len(bad):
        return None
    difference = good[:, None] - bad[None, :]
    return float(((difference > 0).sum() + .5 * (difference == 0).sum())
                 / difference.size)


def summarize(rows: list[dict], *, seed: int = 42) -> dict:
    mixed = [row for row in rows if row["gradient_pair_accuracy"] is not None]
    macro = np.asarray([row["gradient_pair_accuracy"] for row in mixed], np.float64)
    finite = np.asarray([row["critic_pair_accuracy"] for row in mixed], np.float64)
    rng = np.random.default_rng(seed)
    if len(macro):
        bootstrap = np.mean(macro[rng.integers(0, len(macro), size=(2000, len(macro)))], axis=1)
        interval = [float(x) for x in np.quantile(bootstrap, [.025, .975])]
    else:
        interval = None
    rescue = [int(value) for row in rows for value in row["rescue_positive"]]
    avoid = [int(value) for row in rows for value in row["spoil_negative"]]
    return {
        "roots": len(rows), "mixed_roots": len(mixed),
        "stock_successes": sum(row["stock_success"] for row in rows),
        "successful_candidates": sum(row["candidate_successes"] for row in rows),
        "candidate_outcomes": sum(row["candidate_count"] for row in rows),
        "gradient_macro_pair_accuracy": float(macro.mean()) if len(macro) else None,
        "gradient_macro_pair_accuracy_ci95": interval,
        "critic_macro_pair_accuracy": float(finite.mean()) if len(finite) else None,
        "rescue_direction_positive": sum(rescue), "rescue_directions": len(rescue),
        "spoil_direction_negative": sum(avoid), "spoil_directions": len(avoid),
        "median_stock_gradient_norm": float(np.median([
            row["gradient_norm"] for row in rows])) if rows else None,
        "mean_abs_gradient_by_action_dim": np.mean([
            row["gradient_by_dim"] for row in rows], axis=0).tolist() if rows else None,
    }


def analyze_roots(model, roots, device, mean, std, scale,
                  *, max_roots: int | None = None) -> tuple[dict, list[dict]]:
    rows = []
    count = len(roots) if max_roots is None else min(max_roots, len(roots))
    for start in range(0, count, 8):
        batch = _to(_root_batch([roots[i] for i in range(start, min(start + 8, count))]),
                    device)
        signals = batch_signals(model, batch, mean, std, scale)
        labels = batch["success"].cpu().numpy().astype(bool)
        for j, outcome in enumerate(labels):
            projection = signals["projection"][j]
            scores = signals["scores"][j]
            stock_success = bool(outcome[0])
            rows.append({
                "group_id": roots.entries[start + j]["candidate_group_id"],
                "source_experiment": roots.entries[start + j]["source_experiment"],
                "stock_success": stock_success,
                "candidate_successes": int(outcome.sum()),
                "candidate_count": len(outcome),
                "gradient_pair_accuracy": _pair_accuracy(projection, outcome),
                "critic_pair_accuracy": _pair_accuracy(scores, outcome),
                "rescue_positive": (projection[1:][outcome[1:]] > 0).tolist()
                    if not stock_success else [],
                "spoil_negative": (projection[1:][~outcome[1:]] < 0).tolist()
                    if stock_success else [],
                "gradient_norm": float(signals["gradient_norm"][j]),
                "gradient_by_dim": signals["gradient_by_dim"][j].tolist(),
                "mean_abs_cosine_to_alternatives": float(np.mean(np.abs(
                    signals["cosine"][j, 1:]))),
                "median_candidate_distance": float(np.median(
                    signals["distance"][j, 1:])),
            })
        if (start + len(labels)) % 80 == 0 or start + len(labels) == count:
            print({"analyzed_roots": start + len(labels), "total": count}, flush=True)
    return summarize(rows), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path,
                        help="local copy of the exact signed checkpoint")
    parser.add_argument("--checkpoint-key", default=CHECKPOINT_KEY)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"),
                        default="cuda" if torch.cuda.is_available() else
                                "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--split", choices=("train", "validation", "both"),
                        default="both")
    parser.add_argument("--max-roots", type=int,
                        help="first N roots per split for smoke tests")
    parser.add_argument("--cache-root", type=Path,
                        default=Path.home() / "pnp-vla-runs" / "cache")
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "pnp-vla-runs" / "diagnostics" /
                                "smolvla_q10_gradient_benchmark_v1.json")
    args = parser.parse_args()
    if args.max_roots is not None and args.max_roots < 1:
        parser.error("--max-roots must be positive")
    _load_local_credentials()
    store = SupabaseStore()
    data = (args.checkpoint_path.read_bytes() if args.checkpoint_path
            else store._download(args.checkpoint_key))
    device = torch.device(args.device)
    model, payload, mean, std = load_ranker(data, device)
    snapshot = load_or_create_combined_snapshot(store)
    if payload["snapshot_digest"] != snapshot["snapshot_digest"]:
        raise ValueError("checkpoint and root snapshot differ")
    cache = prepare_combined_root_cache(
        snapshot=snapshot, cache_root=args.cache_root, store=store)
    if cache["format"] != ROOT_CACHE_FORMAT:
        raise ValueError("gradient benchmark requires leak-safe pre-action cache")
    groups = {g["candidate_group_id"]: g for g in snapshot["groups"]}
    report = {"checkpoint_sha256": CHECKPOINT_SHA256,
              "snapshot_digest": snapshot["snapshot_digest"],
              "input_contract": ROOT_CACHE_FORMAT,
              "gradient_definition": "d logit(candidate | fixed stock) / d proposed Q10 action",
              "candidate_family": "stock plus eight fresh-initial-noise P&P chunks",
              "splits": {}}
    for name, ids in (("train", cache["train_group_ids"]),
                      ("validation", cache["validation_group_ids"])):
        if args.split not in (name, "both"):
            continue
        roots = TimedRoots(cache, ids, groups)
        summary, rows = analyze_roots(
            model, roots, device, mean, std, float(payload["delta_scale"]),
            max_roots=args.max_roots)
        by_cohort = {cohort: summarize([row for row in rows
                        if row["source_experiment"] == cohort])
                     for cohort in sorted({r["source_experiment"] for r in rows})}
        report["splits"][name] = {"summary": summary,
                                  "by_cohort": by_cohort, "roots": rows}
        print({"split": name, "summary": summary}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print({"report": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
