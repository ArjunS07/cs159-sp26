"""Run the SmolVLA success-Q10 experiment on Modal.

From a checkout with the current pnp code:

    modal secret create pnp-supabase --from-dotenv .env
    modal run pnp-vla/scripts/modal_smolvla_success_q10.py --phase pilot
    modal run --detach pnp-vla/scripts/modal_smolvla_success_q10.py --phase full
    modal run pnp-vla/scripts/modal_smolvla_success_q10.py --phase audit

Only pnp/ is uploaded to the image, so uncommitted experiment code works and
local .env files are never uploaded with source. The v3 artifacts are read from
Supabase; checkpoints, cache, and reports persist in a named Modal Volume.
"""
from __future__ import annotations

from pathlib import Path

import modal


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data")
SECRET_NAME = "pnp-supabase"
VOLUME_NAME = "smolvla-success-q10"
ARMS = ("root_mc", "tree_td")


def snapshot_key_for(tree_limit: int) -> str:
    """Keep the historical 480-tree contract while isolating smaller pilots."""
    if tree_limit == 480:
        return "smolvla_trees/manifests/v3_bellman_finetune_480_20260919.json"
    return f"smolvla_trees/manifests/v3_success_q10_diagnostic_{tree_limit}_v1.json"


def snapshot_contract(dataset: str, tree_limit: int) -> tuple[str, str, tuple[str, ...]]:
    if dataset == "v3":
        return (snapshot_key_for(tree_limit),
                "smolvla-libero-depth1-hybrid-trees-v3-bellman",
                ("stored_source", *(f"fresh_seed_{i}" for i in range(1, 5)),
                 *(f"pnp_perturb_{i}" for i in range(1, 5))))
    if dataset == "fresh8":
        return (f"smolvla_trees/manifests/v4_fresh8_success_q10_{tree_limit}_v1.json",
                "smolvla-libero-depth1-fresh8-trees-v4-bellman",
                ("stored_source", *(f"fresh_seed_{i}" for i in range(1, 9))))
    raise ValueError("dataset must be v3 or fresh8")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.6.0", "numpy<3", "supabase>=2,<3", "httpx",
        "pandas", "pyarrow", "imageio", "imageio-ffmpeg", "tqdm", "wandb",
    )
    .env({"PYTHONPATH": "/root/pnp-vla", "OMP_NUM_THREADS": "4"})
    .add_local_dir(str(PACKAGE_ROOT / "pnp"), remote_path="/root/pnp-vla/pnp")
)
app = modal.App("smolvla-success-q10")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
supabase_secret = modal.Secret.from_name(SECRET_NAME)


@app.function(
    image=image, cpu=4, memory=8192, timeout=2 * 60 * 60,
    secrets=[supabase_secret], volumes={str(DATA_ROOT): volume},
)
def prepare(tree_limit: int = 480, download_workers: int = 8,
            dataset: str = "v3") -> dict:
    """Download and validate tree windows on CPU before paying for a T4."""
    import numpy as np
    import torch
    from pnp.smolvla_tree_bellman_finetune import (
        RootTreeDataset, load_or_create_tree_snapshot,
        prepare_tree_bellman_cache,
    )
    from pnp.store import SupabaseStore

    torch.set_num_threads(4)
    store = SupabaseStore()
    snapshot_key, experiment, kinds = snapshot_contract(dataset, tree_limit)
    snapshot = load_or_create_tree_snapshot(
        store=store, snapshot_key=snapshot_key, tree_limit=tree_limit,
        experiment=experiment, candidate_kinds=kinds)
    cache = prepare_tree_bellman_cache(
        snapshot=snapshot, cache_root=DATA_ROOT / "cache", gamma=1.0,
        success_reward=True, download_workers=download_workers, store=store)
    train_roots = RootTreeDataset(cache, cache["train_group_ids"])
    val_roots = RootTreeDataset(cache, cache["validation_group_ids"])
    train_outcomes = np.stack([train_roots[i]["success"] for i in range(len(train_roots))])
    val_outcomes = np.stack([val_roots[i]["success"] for i in range(len(val_roots))])
    train_ids = set(cache["train_group_ids"])
    train_branches = sum(row["candidate_group_id"] in train_ids
                         for row in cache["candidate_entries"])
    volume.commit()
    summary = {
        "snapshot_digest": snapshot["snapshot_digest"],
        "dataset": dataset,
        "trees": len(snapshot["groups"]),
        "train_trees": len(cache["train_group_ids"]),
        "validation_trees": len(cache["validation_group_ids"]),
        "train_windows": cache["train_windows"],
        "validation_windows": cache["validation_windows"],
        "train_mixed_roots": int((train_outcomes.any(1) & ~train_outcomes.all(1)).sum()),
        "validation_mixed_roots": int((val_outcomes.any(1) & ~val_outcomes.all(1)).sum()),
        "train_stock_successes": int(train_outcomes[:, 0].sum()),
        "train_candidate_success_rate": round(float(train_outcomes.mean()), 4),
        "validation_candidate_success_rate": round(float(val_outcomes.mean()), 4),
        "validation_stock_successes": int(val_outcomes[:, 0].sum()),
        "validation_oracle_successes": int(val_outcomes.any(1).sum()),
        "validation_random_candidate_successes_expected": round(
            float(val_outcomes.mean(1).sum()), 2),
        "td_root_window_fraction": round(train_branches / cache["train_windows"], 4),
        "cuda_used": False,
    }
    print(summary, flush=True)
    return summary


@app.function(
    image=image, cpu=4, memory=8192, timeout=2 * 60 * 60,
    secrets=[supabase_secret], volumes={str(DATA_ROOT): volume},
)
def pack_cache(tree_limit: int = 480, dataset: str = "v3") -> dict:
    """Bundle many NPZ files on CPU so the T4 can read one archive."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json
    import os
    import shutil
    import tarfile
    import tempfile
    import time

    from pnp.smolvla_tree_bellman_finetune import (
        DEFAULT_SNAPSHOT_KEY, load_or_create_tree_snapshot,
        prepare_tree_bellman_cache,
    )
    from pnp.store import SupabaseStore

    store = SupabaseStore()
    snapshot_key, experiment, kinds = snapshot_contract(dataset, tree_limit)
    snapshot = load_or_create_tree_snapshot(
        store=store, snapshot_key=snapshot_key, tree_limit=tree_limit,
        experiment=experiment, candidate_kinds=kinds)
    cache = prepare_tree_bellman_cache(
        snapshot=snapshot, cache_root=DATA_ROOT / "cache", gamma=1.0,
        success_reward=True, store=store)
    digest = snapshot["snapshot_digest"]
    packed = DATA_ROOT / "packed" / f"{digest}.tar"
    if packed.is_file():
        print({"packed_cache": str(packed), "bytes": packed.stat().st_size,
               "already_present": True}, flush=True)
        return {"packed_cache": str(packed), "bytes": packed.stat().st_size}

    source = Path(cache["cache_dir"])
    names = sorted({row["path"] for row in
                    cache["candidate_entries"] + cache["group_entries"]})
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="smolvla_pack_") as temporary:
        local_root = Path(temporary) / "success_probability_v1" / digest
        local_root.mkdir(parents=True)

        def copy_one(name: str) -> None:
            shutil.copyfile(source / name, local_root / name)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(copy_one, name) for name in names]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 200 == 0 or count == len(names):
                    print(f"[modal-pack] staged {count}/{len(names)} files "
                          f"in {(time.perf_counter() - started) / 60:.1f}m", flush=True)
        index = dict(cache)
        index["cache_dir"] = str(Path("/tmp/smolvla-cache") /
                                 "success_probability_v1" / digest)
        (local_root / "cache_index.json").write_text(
            json.dumps(index, sort_keys=True, indent=2))
        local_tar = Path(temporary) / "cache.tar"
        with tarfile.open(local_tar, "w") as archive:
            archive.add(local_root, arcname=f"success_probability_v1/{digest}")
        packed.parent.mkdir(parents=True, exist_ok=True)
        pending = packed.with_suffix(".tar.tmp")
        shutil.copyfile(local_tar, pending)
        os.replace(pending, packed)
    volume.commit()
    result = {"packed_cache": str(packed), "files": len(names),
              "bytes": packed.stat().st_size,
              "elapsed_minutes": (time.perf_counter() - started) / 60}
    print(result, flush=True)
    return result


@app.function(
    image=image, gpu="T4", cpu=2, memory=8192, timeout=4 * 60 * 60,
    secrets=[supabase_secret], volumes={str(DATA_ROOT): volume},
)
def train_arm(arm: str, phase: str, updates: int, tree_limit: int = 480,
              dataset: str = "v3", train_root_limit: int = 0) -> dict:
    """Train one arm, with a separate pilot or full checkpoint directory."""
    import json
    import tarfile
    import torch
    from pnp.smolvla_success_critic import train_smolvla_success_q10
    from pnp.smolvla_tree_bellman_finetune import (
        DEFAULT_SNAPSHOT_KEY, load_or_create_tree_snapshot)
    from pnp.store import SupabaseStore

    if arm not in ARMS or phase not in ("pilot", "full"):
        raise ValueError("invalid arm or training phase")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this experiment requires the requested NVIDIA T4")
    torch.set_num_threads(2)
    snapshot_key, experiment, kinds = snapshot_contract(dataset, tree_limit)
    snapshot = load_or_create_tree_snapshot(
        store=SupabaseStore(), snapshot_key=snapshot_key,
        tree_limit=tree_limit, experiment=experiment, candidate_kinds=kinds)
    packed = DATA_ROOT / "packed" / f"{snapshot['snapshot_digest']}.tar"
    if not packed.is_file():
        raise FileNotFoundError(f"run CPU pack_cache first: {packed}")
    local_cache = Path("/tmp/smolvla-cache")
    local_cache.mkdir(parents=True, exist_ok=True)
    with tarfile.open(packed, "r") as archive:
        archive.extractall(local_cache, filter="data")
    index_path = (local_cache / "success_probability_v1" /
                  snapshot["snapshot_digest"] / "cache_index.json")
    index = json.loads(index_path.read_text())
    if (index.get("snapshot_digest") != snapshot["snapshot_digest"] or
            index.get("cache_dir") != str(index_path.parent) or
            index.get("success_reward") is not True):
        raise ValueError("staged training cache contract mismatch")
    print({"staged_cache": str(index_path.parent),
           "bytes": packed.stat().st_size}, flush=True)
    print({"gpu": torch.cuda.get_device_name(0), "arm": arm,
           "phase": phase, "updates": updates}, flush=True)
    result = train_smolvla_success_q10(
        arm=arm, output_root=DATA_ROOT / "runs" / phase,
        cache_root=local_cache, tree_limit=tree_limit,
        train_root_limit=train_root_limit or None,
        snapshot_key=snapshot_key, experiment=experiment, candidate_kinds=kinds,
        updates=updates, resume=True, device="cuda",
        checkpoint_interval=100 if phase == "full" else 500)
    volume.commit()
    return {
        "arm": arm, "phase": phase,
        "snapshot_digest": result["snapshot_digest"],
        "checkpoint": result["final_checkpoint"],
        "validation": result["validation"],
    }


@app.function(
    image=image, gpu="T4", cpu=2, memory=8192, timeout=45 * 60,
    secrets=[supabase_secret], volumes={str(DATA_ROOT): volume},
)
def audit(phase: str = "full", tree_limit: int = 480,
          dataset: str = "v3", train_root_limit: int = 0) -> dict:
    """Score both saved arms on the same held-out forks and persist JSON."""
    import json
    import numpy as np
    import torch
    from pnp.qplanning_critic.config import QPlanningModelConfig
    from pnp.qplanning_critic.model import QPlanningCritic
    from pnp.smolvla_success_critic import FORMAT, TimedRoots, evaluate_roots
    from pnp.smolvla_tree_bellman_finetune import (
        DEFAULT_SNAPSHOT_KEY, load_or_create_tree_snapshot,
        prepare_tree_bellman_cache,
    )
    from pnp.store import SupabaseStore

    if phase not in ("pilot", "full"):
        raise ValueError("phase must be pilot or full")
    torch.set_num_threads(2)
    store = SupabaseStore()
    snapshot_key, experiment, kinds = snapshot_contract(dataset, tree_limit)
    snapshot = load_or_create_tree_snapshot(
        store=store, snapshot_key=snapshot_key, tree_limit=tree_limit,
        experiment=experiment, candidate_kinds=kinds)
    cache = prepare_tree_bellman_cache(
        snapshot=snapshot, cache_root=DATA_ROOT / "cache", gamma=1.0,
        success_reward=True, store=store)
    groups = {group["candidate_group_id"]: group for group in snapshot["groups"]}
    roots = TimedRoots(cache, cache["validation_group_ids"], groups)
    outcomes = np.stack([roots[i]["success"] for i in range(len(roots))])
    report = {
        "phase": phase, "dataset": dataset,
        "train_root_limit": train_root_limit or None,
        "snapshot_digest": snapshot["snapshot_digest"],
        "validation_trees": len(roots),
        "stock_successes": int(outcomes[:, 0].sum()),
        "oracle_successes": int(outcomes.any(axis=1).sum()),
        "arms": {},
    }
    for arm in ARMS:
        directory = DATA_ROOT / "runs" / phase / snapshot["snapshot_digest"]
        if train_root_limit:
            directory = directory / f"train_roots_{train_root_limit}"
        directory = directory / arm
        paths = list(directory.glob("checkpoint_step_*.pt"))
        if not paths:
            report["arms"][arm] = {"status": "no_checkpoint"}
            continue
        path = max(paths, key=lambda item: int(item.stem.rsplit("_", 1)[-1]))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("format") != FORMAT or
                payload.get("snapshot_digest") != snapshot["snapshot_digest"] or
                payload.get("gamma") != 1.0 or
                payload.get("reward") != "step_success"):
            raise ValueError(f"checkpoint contract mismatch: {path}")
        architecture = dict(payload["architecture"])
        dimensions = {key: architecture.pop(key) for key in
                      ("prefix_dim", "robot_dim", "proprio_dim")}
        model = QPlanningCritic(
            config=QPlanningModelConfig(**architecture), **dimensions).cuda()
        model.load_state_dict(payload["model"])
        report["arms"][arm] = {
            "update": int(payload["update"]),
            "checkpoint": str(path),
            **evaluate_roots(model, roots, torch.device("cuda")),
        }
        del model
    scale_suffix = f"_train{train_root_limit}" if train_root_limit else ""
    path = DATA_ROOT / "reports" / f"{phase}_{snapshot['snapshot_digest']}{scale_suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    volume.commit()
    print(report, flush=True)
    return report


@app.local_entrypoint()
def main(phase: str = "pilot", arms: str = "root_mc,tree_td",
         tree_limit: int = 480, updates: int = 0, dataset: str = "v3",
         train_root_limit: int = 0):
    """Run CPU cache prep, sequential T4 jobs, then a held-out audit."""
    if phase not in ("prepare", "pack", "pilot", "full", "audit"):
        raise ValueError("phase must be prepare, pack, pilot, full, or audit")
    selected = tuple(item.strip() for item in arms.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(ARMS):
        raise ValueError(f"arms must be a nonempty subset of {ARMS}")
    if tree_limit < 1 or updates < 0 or train_root_limit < 0:
        raise ValueError("tree_limit must be positive; updates and train_root_limit nonnegative")
    snapshot_contract(dataset, tree_limit)
    if phase == "audit":
        audit.remote(phase="full", tree_limit=tree_limit, dataset=dataset,
                     train_root_limit=train_root_limit)
        return
    prepare.remote(tree_limit=tree_limit, dataset=dataset)
    if phase == "prepare":
        return
    pack_cache.remote(tree_limit=tree_limit, dataset=dataset)
    if phase == "pack":
        return
    steps = updates or (2 if phase == "pilot" else 2_000)
    for arm in selected:
        train_arm.remote(arm=arm, phase=phase, updates=steps,
                         tree_limit=tree_limit, dataset=dataset,
                         train_root_limit=train_root_limit)
    audit.remote(phase=phase, tree_limit=tree_limit, dataset=dataset,
                 train_root_limit=train_root_limit)
