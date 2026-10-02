"""Matched scalar MC/SARSA data, with proposals independent of termination.

Reuse only context features from the old complete trajectory cache. Restore
every action from the immutable proposal artifact, including terminal chunks.
Observed termination affects targets only, never masks or action contents.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import torch
from torch import nn

from .config import resolve_max_steps
from .pcp_critic.resumable_snapshot import load_training_fields_with_retry
from .qplanning_critic.model import QPlanningCritic
from .smolvla_tree_bellman_finetune import _atomic_json, _atomic_npz

FORMAT = "smolvla_scalar_returns_v1_preaction"
INPUTS = ("prefix", "pad", "robot", "proprio", "action", "action_valid")
FIELDS = ("bellman/action", "boundary/step", "terminated", "truncated", "step_success")


class ScalarCritic(QPlanningCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = nn.Linear(self.config.width, 1)

    def expected_value(self, *args, **kwargs):
        return self(*args, **kwargs).squeeze(-1).sigmoid()


class LateFusionScalarCritic(ScalarCritic):
    """A direct state/action path initialized to preserve the base predictor."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width = self.config.width
        self.late_action = nn.Sequential(nn.Linear(70, width), nn.GELU())
        self.late_head = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.late_head[-1].weight)
        nn.init.zeros_(self.late_head[-1].bias)

    def architecture_config(self):
        return {**super().architecture_config(), "model_family": "late_fusion_scalar"}

    def forward(self, prefix, prefix_valid, robot, proprio, action, action_valid):
        base = super().forward(prefix, prefix_valid, robot, proprio, action, action_valid)
        memory = self.prefix_projection(prefix.float())
        mask = prefix_valid.float()[..., None]
        context = (memory * mask).sum(1) / mask.sum(1).clamp_min(1)
        context = context + self.robot_projection(robot.float()) + self.proprio_projection(proprio.float())
        normalized = ((action.float() - self.action_mean) / self.action_std) * action_valid[..., None]
        embedded = self.late_action(normalized.flatten(1))
        return base + self.late_head(torch.cat([context, embedded, context * embedded], dim=-1))


def restore_transitions(old, raw, spec, root=None):
    actions = np.asarray(raw["bellman/action"], np.float32)[:, :10, :7]
    steps = np.asarray(raw["boundary/step"], np.int64)
    success = np.asarray(raw["step_success"], bool)
    done = np.asarray(raw["terminated"], bool) | np.asarray(raw["truncated"], bool)
    n = len(actions)
    start = int(spec["start"])
    maximum = resolve_max_steps(spec["suite"])
    if (len(steps) != n + 1 or steps[0] != 0 or steps[-1] != len(success)
            or len(done) != len(success) or not np.all((np.diff(steps) > 0) & (np.diff(steps) <= 10))
            or bool(success.any()) != bool(spec["success"])):
        raise ValueError("raw transition boundaries/outcomes disagree")
    if len(old["action"]) != n - start:
        raise ValueError("legacy context cache omitted a transition")
    values = {key: np.asarray(old[key]).copy() for key in INPUTS[:4]}
    absolute = steps[:-1] + spec["step_offset"]
    expected_time = (maximum - absolute[start:]) / maximum
    if not np.allclose(values["robot"][:, -1], expected_time, atol=1e-6):
        raise ValueError("legacy context time alignment disagrees")
    if start:
        if start != 1 or root is None:
            raise ValueError("branch root context required")
        values = {key: np.concatenate([np.asarray(root[key])[None], values[key]], axis=0)
                  for key in INPUTS[:4]}
    if np.any(absolute < 0) or np.any(absolute >= maximum):
        raise ValueError("out-of-budget decision state")
    values["action"] = actions.copy()
    values["action_valid"] = np.arange(10)[None] < (maximum - absolute)[:, None]
    reward = np.zeros(n, np.float32)
    discount = np.ones(n, np.float32)
    for i, (lo, hi) in enumerate(zip(steps[:-1], steps[1:])):
        reward[i] = float(success[lo:hi].any())
        terminal = bool(done[lo:hi].any() or reward[i] or hi + spec["step_offset"] >= maximum)
        discount[i] = 0.0 if terminal else 1.0
        if terminal != (i == n - 1):
            raise ValueError("episode terminal boundary missing or interior terminal found")
    values.update(reward=reward, discount=discount,
                  success=np.full(n, spec["success"], np.float32))
    return values


def prepare_cache(root_cache, roots, cache_root, store, workers=4):
    """Repair all three predeclared trajectories per training root, resumably."""
    cache_root = Path(cache_root)
    legacy = cache_root / "smolvla_two_stage_trajectory_v1" / root_cache["snapshot_digest"]
    index = json.loads((legacy / "cache_index.json").read_text())
    if (set(index["train_group_ids"]) != set(root_cache["train_group_ids"])
            or set(index["validation_group_ids"]) != set(root_cache["validation_group_ids"])):
        raise ValueError("legacy trajectory split differs")
    destination = cache_root / FORMAT / index["snapshot_digest"]
    destination.mkdir(parents=True, exist_ok=True)
    root_indices = {row["candidate_group_id"]: i for i, row in enumerate(roots.entries)}
    local = threading.local()

    def build(spec):
        key = hashlib.sha256((spec["group_id"] + "|" + spec["kind"]).encode()).hexdigest()[:24]
        path = destination / f"trajectory_{key}.npz"
        if not path.exists():
            worker = getattr(local, "store", None)
            if worker is None:
                worker = store.fork_for_thread()
                local.store = worker
            raw = load_training_fields_with_retry(worker, spec["path"], FIELDS)
            root = roots[root_indices[spec["group_id"]]] if spec["start"] else None
            if spec["file"] is None:
                old = {k: np.empty((0, *np.asarray(root[k]).shape), np.asarray(root[k]).dtype)
                       for k in INPUTS[:4]}
                old["action"] = np.empty((0, 10, 7), np.float32)
                arrays = restore_transitions(old, raw, spec, root)
            else:
                with np.load(legacy / spec["file"], allow_pickle=False) as old:
                    arrays = restore_transitions(old, raw, spec, root)
            _atomic_npz(path, {"format": np.asarray(FORMAT), **arrays})
        with np.load(path, allow_pickle=False) as arrays:
            if str(arrays["format"]) != FORMAT:
                raise ValueError("cache format mismatch")
            count = len(arrays["success"])
        return {**spec, "file": path.name, "windows": count}

    entries = []
    print({"transition_cache": str(destination), "trajectories": len(index["entries"])}, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(build, spec) for spec in index["entries"]]
        for future in as_completed(futures):
            entries.append(future.result())
            if len(entries) % 100 == 0:
                print({"repaired_trajectories": len(entries), "total": len(futures)}, flush=True)
    result = {**index, "format": FORMAT, "cache_dir": str(destination),
              "entries": sorted(entries, key=lambda x: (x["group_id"], x["kind"]))}
    _atomic_json(destination / "cache_index.json", result)
    return result


class ReturnSampler:
    def __init__(self, index, max_open=64):
        self.directory = Path(index["cache_dir"])
        self.groups = sorted(index["train_group_ids"])
        self.entries = {gid: [e for e in index["entries"] if e["group_id"] == gid]
                        for gid in self.groups}
        self.open = OrderedDict()
        self.max_open = max_open

    def batch(self, rng, size, nstep=1):
        rows = []
        for _ in range(size):
            gid = self.groups[int(rng.integers(len(self.groups)))]
            entries = self.entries[gid]
            entry = entries[int(rng.integers(len(entries)))]
            filename = entry["file"]
            if filename not in self.open:
                with np.load(self.directory / filename, allow_pickle=False) as archive:
                    self.open[filename] = {k: archive[k] for k in (*INPUTS, "reward", "discount", "success")}
                if len(self.open) > self.max_open:
                    self.open.popitem(last=False)
            self.open.move_to_end(filename)
            data = self.open[filename]
            i = int(rng.integers(entry["windows"]))
            reward, discount = 0., 1.
            for k in range(i, min(i + nstep, entry["windows"])):
                reward += discount * float(data["reward"][k])
                discount *= float(data["discount"][k])
                if not discount:
                    break
            j = min(i + nstep, entry["windows"] - 1)
            rows.append({**{k: data[k][i] for k in INPUTS},
                         **{"next_" + k: data[k][j] for k in INPUTS},
                         "reward": np.float32(reward), "discount": np.float32(discount),
                         "success": data["success"][i]})
        return {k: torch.from_numpy(np.stack([row[k] for row in rows])) for k in rows[0]}


def logits(model, batch, prefix=""):
    return model(*(batch[prefix + key] for key in INPUTS)).squeeze(-1)
