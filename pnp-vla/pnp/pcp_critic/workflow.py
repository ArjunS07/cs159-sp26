"""Notebook-facing production workflows for PCP critic snapshots and training."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import torch

from ..store import SupabaseStore, gather_provenance
from .config import PCPCriticModelConfig, PCPCriticTrainConfig, PCPSearchAdapterConfig
from .data import create_snapshot, prepare_compact_cache
from .deploy import PCPSearchAdapter
from .model import PCPCritic
from .registry import PCPCriticRegistry
from .train import CompactTransitionDataset, checkpoint_bytes, evaluate_critic, train_critic


def create_dataset_snapshot(*, name: str, seed: int = 42, train_fraction: float = .8,
                            gamma: float = .99, store: SupabaseStore | None = None) -> str:
    store = store or SupabaseStore()
    snapshot = create_snapshot(store, name=name, seed=seed, train_fraction=train_fraction, gamma=gamma)
    PCPCriticRegistry(store).publish_snapshot(snapshot, name=name)
    print(f"[pcp-critic] snapshot {snapshot.snapshot_id}: {len(snapshot.rollout_ids)} rollouts, "
          f"{snapshot.provenance['n_transitions']} transitions")
    return snapshot.snapshot_id


def _snapshot_transitions(store, snapshot, *, gamma: float, cache_root: str | Path):
    cache = prepare_compact_cache(store, snapshot, gamma=gamma, cache_root=cache_root)
    train_ids, val_ids = set(snapshot.train_rollout_ids), set(snapshot.val_rollout_ids)
    train = CompactTransitionDataset(cache, train_ids)
    val = CompactTransitionDataset(cache, val_ids)
    if len(train) + len(val) != cache.n_transitions:
        raise AssertionError("critic split assignment escaped snapshot")
    return cache, train, val


def run_pcp_critic_train(*, snapshot_id: str, objective: str = "calql", device=None,
                         model_config: PCPCriticModelConfig | None = None,
                         train_config: PCPCriticTrainConfig | None = None,
                         experiment: str = "pcp-critic-v1", cache_root: str | Path = "pcp_critic_cache",
                         store: SupabaseStore | None = None) -> str:
    store = store or SupabaseStore()
    registry = PCPCriticRegistry(store)
    snapshot = registry.load_snapshot(snapshot_id)
    cfg = train_config or PCPCriticTrainConfig(objective=objective)
    if cfg.objective != objective:
        raise ValueError("objective argument and train_config disagree")
    cache, train, val = _snapshot_transitions(store, snapshot, gamma=cfg.gamma, cache_root=cache_root)
    if len(train) == 0:
        raise ValueError("snapshot has no train transitions")
    model = PCPCritic(prefix_dim=cache.prefix_dim, robot_dim=cache.robot_dim,
                      proprio_dim=cache.proprio_dim, config=model_config)
    model.set_action_statistics(snapshot.action_mean, snapshot.action_std)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    store.start_run("pcp_critic_train", "offline", experiment=experiment, provenance=gather_provenance(),
                    config={"snapshot_id": snapshot_id, "objective": objective,
                            "model": model.architecture_config(), "train": cfg.to_dict(),
                            "compact_cache": {"path": str(cache_root), "digest": cache.digest}})
    try:
        model, metrics = train_critic(model, train, val, device, config=cfg)
        metrics.update({"train_transitions": len(train), "val_transitions": len(val),
                        "dataset_hash": cache.digest, "safety_status": "offline_only"})
        critic_id = "pcpc-" + hashlib.sha256(
            f"{snapshot_id}|{objective}|{uuid.uuid4()}".encode()).hexdigest()[:24]
        registry.register_model(
            critic_id, checkpoint_bytes(model, snapshot_id, cfg, metrics), snapshot=snapshot,
            architecture=model.architecture_config(), train_config=cfg.to_dict(), metrics=metrics,
            objective=objective)
    except BaseException:
        store.finish_run(status="failed", n_rollouts=0)
        raise
    store.finish_run(status="completed", n_rollouts=0)
    print(f"[pcp-critic] registered {critic_id} ({objective}, offline_only)")
    return critic_id


def load_pcp_critic(critic_id: str, *, store: SupabaseStore | None = None, device="cpu"):
    store = store or SupabaseStore()
    payload, row = PCPCriticRegistry(store).load_model_payload(critic_id)
    architecture = payload["architecture"]
    config = PCPCriticModelConfig(**{key: architecture[key] for key in
                                     PCPCriticModelConfig.__dataclass_fields__})
    model = PCPCritic(prefix_dim=architecture["prefix_dim"], robot_dim=architecture["robot_dim"],
                      proprio_dim=architecture["proprio_dim"], config=config)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), row, payload


def make_pcp_search_adapter(*, critic_id: str, policy_repo_id: str, policy_revision: str,
                            config: PCPSearchAdapterConfig | None = None,
                            store: SupabaseStore | None = None, device="cpu") -> PCPSearchAdapter:
    """Load an adapter only when its frozen PI05 contract exactly matches runtime."""
    model, row, _ = load_pcp_critic(critic_id, store=store, device=device)
    if (row["policy_repo_id"], row["policy_revision"]) != (policy_repo_id, policy_revision):
        raise ValueError("PCP critic policy contract does not match the live PI05 policy")
    return PCPSearchAdapter(model, config=config)


def run_pcp_critic_offline_eval(*, critic_id: str, device=None,
                                cache_root: str | Path = "pcp_critic_cache",
                                store: SupabaseStore | None = None) -> dict:
    store = store or SupabaseStore()
    model, row, payload = load_pcp_critic(critic_id, store=store, device=device or "cpu")
    snapshot = PCPCriticRegistry(store).load_snapshot(row["snapshot_id"])
    cfg = PCPCriticTrainConfig(**payload["train_config"])
    _, _, val = _snapshot_transitions(store, snapshot, gamma=cfg.gamma, cache_root=cache_root)
    return {"critic_id": critic_id, "snapshot_id": snapshot.snapshot_id,
            "safety_status": row["safety_status"],
            "validation": evaluate_critic(model, model, val, next(model.parameters()).device, config=cfg)}
