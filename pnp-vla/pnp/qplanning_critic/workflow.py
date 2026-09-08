"""Notebook-facing workflow for the isolated Q10/Q50 training tests."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from ..pcp_critic.registry import PCPCriticRegistry
from ..store import SupabaseStore
from .config import QPlanningModelConfig, QPlanningTrainConfig
from .data import QPlanningWindowDataset, prepare_qplanning_cache
from .model import QPlanningCritic
from .train import train_qplanning_critic


def _print_preflight(snapshot, cache) -> None:
    train_ids, val_ids = set(snapshot.train_rollout_ids), set(snapshot.val_rollout_ids)
    if train_ids & val_ids:
        raise AssertionError("snapshot train/validation rollout IDs overlap")
    if train_ids | val_ids != set(snapshot.rollout_ids):
        raise AssertionError("snapshot split does not cover its immutable rollout set")
    counts = Counter((entry["benchmark"], entry["suite"],
                      "train" if entry["rollout_id"] in train_ids else "validation")
                     for entry in cache.rollouts)
    print("\nImmutable data contract")
    print(f"  snapshot: {snapshot.snapshot_id}")
    print(f"  source PI: {snapshot.policy_repo_id}@{snapshot.policy_revision}")
    print(f"  horizon: Q{cache.horizon} executed actions")
    print(f"  rollouts: {len(train_ids)} train + {len(val_ids)} validation")
    print(f"  windows: {cache.n_train_windows} train + {cache.n_val_windows} validation")
    print(f"  nonterminal bootstrap windows: {cache.n_bootstrap_windows}/{cache.n_windows} "
          f"({cache.n_bootstrap_windows/cache.n_windows:.1%})")
    print("  normalization: train executed actions only")
    print("  uncertainty in model/loss/sampling: NO")
    print("  first-10 generated/executed check: "
          f"MAE={cache.generated_executed_first10_mae:.3g}, "
          f"max={cache.generated_executed_first10_max_abs:.3g}")
    if cache.generated_executed_first10_max_abs > 1e-3:
        print("  WARNING: generated and executed first-10 actions differ; "
              "training still uses the executed trajectory only.")
    print("\nRollouts by benchmark / suite / split")
    for (benchmark, suite, split), count in sorted(counts.items()):
        print(f"  {split:10s} {benchmark:12s} {suite:36s} {count:4d}")


def run_qplanning_training_test(*, snapshot_id: str, horizon: int,
                                run_mode: str = "smoke",
                                cache_root: str | Path = "/content/qplanning_cache",
                                output_root: str | Path = "/content/qplanning_checkpoints",
                                micro_batch_size: int = 16,
                                cache_download_workers: int = 4,
                                device=None, resume: bool = True,
                                store: SupabaseStore | None = None) -> dict:
    """Validate/cache data and run either a short pipeline test or fixed full run."""
    if run_mode not in ("smoke", "full"):
        raise ValueError("run_mode must be 'smoke' or 'full'")
    if horizon not in (10, 50):
        raise ValueError("horizon must be 10 or 50")
    if not snapshot_id.startswith("pcpcds-"):
        raise ValueError("paste the immutable pcpcds-* snapshot ID produced by notebook 56")
    store = store or SupabaseStore()
    snapshot = PCPCriticRegistry(store).load_snapshot(snapshot_id)
    cache = prepare_qplanning_cache(
        store, snapshot, horizon=horizon, gamma=.99, cache_root=cache_root,
        download_workers=cache_download_workers)
    _print_preflight(snapshot, cache)
    train = QPlanningWindowDataset(cache, snapshot.train_rollout_ids)
    validation_limit = 512 if run_mode == "smoke" else 4096
    validation = QPlanningWindowDataset(
        cache, snapshot.val_rollout_ids, max_windows=validation_limit)
    model_config = QPlanningModelConfig(action_horizon=horizon, action_dim=cache.action_dim)
    if run_mode == "smoke":
        train_config = QPlanningTrainConfig(
            effective_batch_size=64, micro_batch_size=micro_batch_size,
            updates=10, warmup_updates=2, print_interval=2, eval_interval=5,
            checkpoint_interval=10, max_validation_transitions=validation_limit)
    else:
        train_config = QPlanningTrainConfig(
            effective_batch_size=64, micro_batch_size=micro_batch_size,
            updates=8_000, warmup_updates=500, print_interval=100,
            eval_interval=500, checkpoint_interval=1_000,
            max_validation_transitions=validation_limit)
    model = QPlanningCritic(
        prefix_dim=cache.prefix_dim, robot_dim=cache.robot_dim,
        proprio_dim=cache.proprio_dim, config=model_config)
    model.set_action_statistics(cache.action_mean, cache.action_std)
    output_dir = Path(output_root) / snapshot_id / f"q{horizon}_{run_mode}"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\nTraining contract")
    print(f"  mode={run_mode}; device={device}; updates={train_config.updates}")
    print(f"  effective batch={train_config.effective_batch_size}; "
          f"microbatch={train_config.micro_batch_size}; "
          f"accumulation={train_config.accumulation_steps}")
    print(f"  checkpoint directory: {output_dir}")
    _, _, report = train_qplanning_critic(
        model, train, validation, device, snapshot_id=snapshot_id,
        cache_digest=cache.digest,
        source_policy={"repo_id": snapshot.policy_repo_id,
                       "revision": snapshot.policy_revision},
        output_dir=output_dir,
        config=train_config, resume=resume)
    report["source_policy"] = {
        "repo_id": snapshot.policy_repo_id, "revision": snapshot.policy_revision}
    report["run_mode"] = run_mode
    return report
