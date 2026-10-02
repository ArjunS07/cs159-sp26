"""Exploratory post-generation Q-gradient pilot; not flow-step PCP deployment.

Benchmark directions against recorded same-state outcomes first, then explicitly
opt in to simulator controls. These reused roots overlap training/validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import smolvla_tree_collection as tree
from .config import resolve_max_steps
from .pcp_critic.resumable_snapshot import _download_with_retry
from .qplanning_critic.config import QPlanningModelConfig
from .smolvla_action_contrast_pilot import _pilot_items, SHARD_COUNT, SCALES
from .smolvla_combined_success import (
    COMBINED_SNAPSHOT_KEY, ROOT_CACHE_FORMAT, SOURCE_EXPERIMENTS, _read_root_group)
from .smolvla_scalar_returns import FORMAT, ScalarCritic, LateFusionScalarCritic
from .smolvla_success_critic import TimedRoots
from .smolvla_tree_bellman_finetune import _json_digest, _split_groups
from .store import SupabaseStore

EXPERIMENT = "smolvla-libero-q10-postgeneration-pcp-pilot-v1"
KINDS = {"stored_source", "fresh_seed_1", "stock_replay", *(
    f"{control}_{scale}" for control in ("q_plus", "q_minus", "random")
    for scale in ("small", "large"))}


def load_scalar_checkpoint(path, snapshot, device="cpu"):
    """Strict architecture/state loading; checkpoints stay on local disk/Drive."""
    path = Path(path).expanduser()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("format") != FORMAT or saved.get("scalar_head") is not True
            or saved.get("snapshot_digest") != snapshot["snapshot_digest"]):
        raise ValueError("need a preaction scalar-return checkpoint for this exact snapshot")
    arch = dict(saved["architecture"])
    family = arch.pop("model_family", "scalar_decoder")
    if family not in ("scalar_decoder", "late_fusion_scalar"):
        raise ValueError(f"unsupported model family {family}")
    dims = {k: arch.pop(k) for k in ("prefix_dim", "robot_dim", "proprio_dim")}
    config = QPlanningModelConfig(**arch)
    if config.action_horizon != 10 or config.action_dim != 7:
        raise ValueError("pilot requires Q10 seven-dimensional normalized actions")
    cls = LateFusionScalarCritic if family == "late_fusion_scalar" else ScalarCritic
    model = cls(**dims, config=config)
    model.load_state_dict(saved["model"], strict=True)
    if not torch.isfinite(model.action_std).all() or not (model.action_std > 0).all():
        raise ValueError("invalid checkpoint action normalization")
    model.to(device).eval().requires_grad_(False)
    metadata = {"checkpoint_path": str(path), "critic_sha256": digest,
                "architecture": saved["architecture"], "critic_arm": saved.get("arm"),
                "critic_update": saved["update"], "snapshot_digest": saved["snapshot_digest"],
                "checkpoint_input_contract": FORMAT, "root_input_contract": ROOT_CACHE_FORMAT,
                "method": "postgeneration_single_step_q_gradient", "scales_rms": list(SCALES),
                "interpretation": "exploratory reused roots; not independent policy evaluation"}
    return model, metadata


def root_score(model, row, action):
    device = next(model.parameters()).device
    context = [torch.as_tensor(row[k][None], device=device).detach()
               for k in ("prefix", "pad", "robot", "proprio")]
    valid = torch.as_tensor(row["action_valid"][0:1], device=device).bool()
    return model(*context, action, valid).squeeze(-1).sigmoid()


def direction_and_benchmark(model, row):
    """Only proposed actions determine inputs; outcomes enter report metrics only."""
    actions = np.asarray(row["actions"], np.float32)
    mask = np.asarray(row["action_valid"], bool)
    if (actions.shape != (9, 10, 7) or mask.shape != (9, 10)
            or not np.array_equal(mask, np.broadcast_to(mask[0], mask.shape))
            or not np.isfinite(actions).all()):
        raise ValueError("invalid same-state preaction root contract")
    if (row["prefix"].shape[-1] != model.prefix_dim or len(row["robot"]) != model.robot_dim
            or len(row["proprio"]) != model.proprio_dim):
        raise ValueError("root features do not match complete checkpoint architecture")
    device = next(model.parameters()).device
    stock = torch.tensor(actions[0:1], device=device, requires_grad=True)
    with torch.enable_grad():
        q = root_score(model, row, stock)
        grad = torch.autograd.grad(q.sum(), stock)[0][0].detach().cpu().numpy()
    grad *= mask[0, :, None]
    if not np.isfinite(grad).all():
        raise ValueError("nonfinite action gradient")
    rms = float(np.sqrt(np.square(grad[mask[0]]).mean()))
    unit = grad / rms if rms > 1e-12 else np.zeros_like(grad)
    projection = ((actions - actions[0]) * grad).sum((1, 2))
    with torch.no_grad():
        scores = [float(root_score(model, row, torch.tensor(a[None], device=device))[0])
                  for a in actions]
    outcomes = np.asarray(row["success"], bool)
    mixed = bool(outcomes.any() and (~outcomes).any())
    def paired(values):
        if not mixed:
            return None
        delta = np.asarray(values)[outcomes, None] - np.asarray(values)[None, ~outcomes]
        return float(((delta > 0) + .5 * (delta == 0)).mean())
    return unit, {"q_stock": float(q.detach()[0]), "gradient_rms": rms,
                  "gradient_usable": rms > 1e-12, "q_scores": scores,
                  "recorded_outcomes": outcomes.tolist(), "linearized_q_delta": projection.tolist(),
                  "mixed_root": mixed, "score_pair_accuracy": paired(scores),
                  "gradient_projection_pair_accuracy": paired(projection)}


def controlled_chunks(stock, direction, valid, seed):
    stock = np.asarray(stock, np.float32)
    if stock.ndim != 2 or stock.shape[0] < 10 or stock.shape[1] < 7:
        raise ValueError("invalid source proposal")
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError("zero/nonfinite gradient: diagnose before attempting intervention")
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(10, 7)).astype(np.float32) * valid[:, None]
    random /= np.sqrt(np.square(random[valid]).mean())
    extras = {"stock_replay": (stock.copy(), {"control": "stock_replay"})}
    for label, magnitude in zip(("small", "large"), SCALES):
        for name, unit in (("q_plus", direction), ("q_minus", -direction), ("random", random)):
            proposal = stock.copy()
            # Trust radius is RMS in raw normalized policy units; no clipping or
            # termination-dependent masking. Unexecuted tail is left intact.
            proposal[:10, :7] += magnitude * unit * valid[:, None]
            delta = proposal[:10, :7] - stock[:10, :7]
            extras[f"{name}_{label}"] = (proposal, {
                "control": name, "requested_normalized_action_rms": magnitude,
                "realized_normalized_action_rms": float(np.sqrt(np.square(delta[valid]).mean())),
                "realized_max_abs_delta": float(np.abs(delta).max()),
                "random_seed": int(seed), "preaction_valid_mask": valid.tolist(),
                "method": "postgeneration_single_step_q_gradient"})
    return extras



def complete_candidate_contract(candidates, *, expected_critic_sha=None):
    if len(candidates) != len(KINDS) or {c["candidate_kind"] for c in candidates} != KINDS:
        return False
    return all((c.get("metadata_json") or {}).get("training_data_path")
               and (expected_critic_sha is None or
                    (c.get("metadata_json") or {}).get("critic_sha256") == expected_critic_sha)
               for c in candidates)


def run_pcp_pilot(*, checkpoint_path, shard_index=0, root_limit=1,
                  run_intervention=False, cache_root="/content/pcp_pilot_cache",
                  output_path="/content/pcp_pilot_report.json", device="cpu", store=None,
                  _proposal_generator=None, _method_config=None, _experiment_base=EXPERIMENT,
                  _cohort_provider=None, _shard_count=SHARD_COUNT, _expected_checkpoint_sha=None):
    if _shard_count < 1 or shard_index not in range(_shard_count) or (root_limit is not None and root_limit < 1):
        raise ValueError("invalid shard/root limit")
    store = store or SupabaseStore()
    # Read an already frozen manifest; never create a new training snapshot here.
    snapshot = json.loads(_download_with_retry(store, COMBINED_SNAPSHOT_KEY))
    claimed = snapshot["snapshot_digest"]
    if _json_digest({k: v for k, v in snapshot.items() if k != "snapshot_digest"}) != claimed:
        raise ValueError("frozen snapshot digest mismatch")
    model, metadata = load_scalar_checkpoint(checkpoint_path, snapshot, device)
    if _expected_checkpoint_sha and metadata["critic_sha256"] != _expected_checkpoint_sha:
        raise ValueError("checkpoint SHA256 differs from the predeclared focused protocol")
    if _method_config:
        metadata.update(_method_config)
    cohort_document = None
    if _cohort_provider:
        cohort_document = _cohort_provider(snapshot, store, metadata)
        items, source_manifest = cohort_document["items"], cohort_document["manifest_hash"]
        metadata["selection_audit"] = cohort_document["audit"]
    else:
        items, source_manifest = _pilot_items(store)
    items = [i for i in items if i["ordinal"] % _shard_count == shard_index][:root_limit]
    source_rows = {r["rollout_id"]: r for r in tree._source_rows(store)}
    groups_by_key = {(g["suite"], g["task_idx"], g["episode_idx"], g["chunk_idx"],
                      g["source_training_data_path"]): g for g in snapshot["groups"]}
    split = {}
    for source in SOURCE_EXPERIMENTS:
        train, validation = _split_groups([g for g in snapshot["groups"] if g["source_experiment"] == source])
        split.update({gid: "train" for gid in train})
        split.update({gid: "validation" for gid in validation})
    experiment = _experiment_base + "-" + metadata["critic_sha256"][:16]
    if _method_config:
        experiment += "-" + tree._digest(_method_config)[:12]
    manifest = tree._digest({"source_manifest": source_manifest, "critic": {k: v for k, v in metadata.items() if k != "checkpoint_path"},
                            "kinds": sorted(KINDS)})
    destination = Path(cache_root) / ROOT_CACHE_FORMAT / claimed
    destination.mkdir(parents=True, exist_ok=True)
    planned, diagnostics = [], []
    for item in items:
        source_row = source_rows[item["source_rollout_id"]]
        key = (item["suite"], item["task_idx"], item["episode_idx"], item["chunk_idx"],
               source_row["training_data_path"])
        if key not in groups_by_key:
            raise ValueError("pilot root is absent from checkpoint's frozen training snapshot")
        group = groups_by_key[key]
        entry = _read_root_group(store, group, destination)
        cache = {"cache_dir": str(destination), "group_entries": [entry], "snapshot_digest": claimed}
        row = TimedRoots(cache, [group["candidate_group_id"]], {group["candidate_group_id"]: group})[0]
        if str(row["input_contract"]) != ROOT_CACHE_FORMAT:
            raise ValueError("legacy termination-dependent root input rejected")
        preaction_mask = np.arange(10) < min(10, resolve_max_steps(group["suite"]) - 10 * int(group["chunk_idx"]))
        if not np.array_equal(row["action_valid"], np.broadcast_to(preaction_mask, (9, 10))):
            raise ValueError("root mask differs from known preaction action budget")
        direction, report = direction_and_benchmark(model, row)
        report.update(source_rollout_id=item["source_rollout_id"],
                      source_group_id=group["candidate_group_id"], original_split=split[group["candidate_group_id"]])
        diagnostics.append(report)
        planned.append((item, source_row, row, direction, report))
    result = {"experiment": experiment, "manifest_hash": manifest, **metadata,
              "shard_index": shard_index, "shard_count": _shard_count,
              "cohort_manifest": cohort_document,
              "benchmark": diagnostics, "run_intervention": run_intervention, "new_roots": 0,
              "intervention_outcomes": []}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not run_intervention:
        return result
    if cohort_document:
        path = f"smolvla_pcp_pilot/manifests/{experiment}/{source_manifest}.json"
        encoded = json.dumps(cohort_document, sort_keys=True).encode()
        try:
            existing_manifest = json.loads(_download_with_retry(store, path))
        except Exception as error:
            if not any(t in str(error).lower() for t in ("404", "not found", "does not exist")):
                raise
            store._upload(path, encoded)
            existing_manifest = json.loads(_download_with_retry(store, path))
        if existing_manifest != cohort_document:
            raise ValueError("immutable focused cohort manifest differs")
    if _proposal_generator is None and not all(r["gradient_usable"] for r in diagnostics):
        raise ValueError("unusable gradient; simulator collection refused")
    from . import libero_env, models
    from .smolvla_tree_source_experiment import prepare_smolvla_tree_source_episodes
    if not torch.cuda.is_available():
        raise RuntimeError("intervention requires a CUDA runtime with NVIDIA EGL")
    episodes = {(e["suite"], e["task_idx"], e["ep_idx"]): e for e in prepare_smolvla_tree_source_episodes()}
    policy, preprocess, postprocess = models.load_smolvla()
    driver = "smolvla_flow_step_pcp_pilot" if _proposal_generator else "smolvla_postgeneration_pcp_pilot"
    store.start_run(driver, "libero", experiment, config=result)
    status = "failed"
    started = time.monotonic()
    try:
        for item, source_row, row, direction, report in planned:
            from .verifier.collection import candidate_group_id
            gid = candidate_group_id("libero", item["suite"], item["task_idx"], item["episode_idx"],
                                     item["chunk_idx"], namespace=experiment, trajectory_seed=item["source_episode_seed"])
            existing = store.fetch_all("verifier_candidate_groups", "candidate_group_id,manifest_hash",
                                      configure=lambda q: q.eq("candidate_group_id", gid), order_by=("candidate_group_id",))
            if existing:
                if any(g["manifest_hash"] != manifest for g in existing):
                    raise ValueError("existing pilot group has a different contract")
                candidates = store.fetch_all("verifier_candidates", "candidate_kind,metadata_json,success,n_steps",
                                            configure=lambda q: q.eq("candidate_group_id", gid), order_by=("candidate_kind",))
                if complete_candidate_contract(candidates, expected_critic_sha=_expected_checkpoint_sha):
                    result["intervention_outcomes"].append({"candidate_group_id": gid,
                        "original_split": report["original_split"], "resumed_complete": True,
                        "success_by_control": {c["candidate_kind"]: c["success"] for c in candidates},
                        "steps_by_control": {c["candidate_kind"]: c["n_steps"] for c in candidates}})
                    continue
            bundle = tree._load_source_bundle(store, source_row)
            source = tree._source_boundary(bundle, item)
            valid = np.arange(10) < min(10, resolve_max_steps(item["suite"]) - int(source["root_step"]))
            if not np.array_equal(valid, row["action_valid"][0]) or not np.allclose(source["policy_chunk"][:10, :7][valid], row["actions"][0][valid], atol=1e-6):
                raise ValueError("cached proposal and immutable replay source disagree")
            seed = tree._seed("pcp-pilot-random", item["source_rollout_id"], item["chunk_idx"])
            if _proposal_generator is None:
                extras = controlled_chunks(source["policy_chunk"], direction, valid, seed)
            else:
                print({"phase": "flow_candidate_generation", "source_group_id": report["source_group_id"],
                       "completed_roots": len(result["intervention_outcomes"]), "requested_roots": len(planned)}, flush=True)
                extras = _proposal_generator(policy=policy, preprocess=preprocess,
                    item=item, bundle=bundle, source=source, row=row, critic=model,
                    valid=valid, random_seed=seed)
            ep = episodes[(item["suite"], item["task_idx"], item["episode_idx"])]
            env = libero_env.make_env(ep["bddl_path"])
            try:
                group, candidates = tree.collect_smolvla_depth1_tree(
                    env, ep, policy, preprocess, postprocess, models.default_device(), item=item,
                    bundle=bundle, manifest_hash=manifest, experiment=experiment,
                    fresh_count=1, perturb_count=0, extra_policy_chunks=extras)
            finally:
                env.close()
            if group["candidate_group_id"] != gid or {c["candidate_kind"] for c in candidates} != KINDS:
                raise RuntimeError("intervention candidate contract mismatch")
            group["metadata_json"].update(metadata, diagnostic=report, independent_test=False)
            for candidate in candidates:
                candidate["metadata_json"].update(metadata, original_split=report["original_split"])
            store.register_candidate_group(group, candidates)
            result["intervention_outcomes"].append({"candidate_group_id": gid,
                "original_split": report["original_split"],
                "success_by_control": {c["candidate_kind"]: c["success"] for c in candidates},
                "steps_by_control": {c["candidate_kind"]: c["n_steps"] for c in candidates}})
            result["new_roots"] += 1
            print({"phase": "root_completed", "completed_roots": len(result["intervention_outcomes"]),
                   "requested_roots": len(planned), "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
                   "success_by_control": result["intervention_outcomes"][-1]["success_by_control"]}, flush=True)
        store._upload(f"smolvla_pcp_pilot/reports/{experiment}/shard_{shard_index}.json",
                      json.dumps(result, sort_keys=True).encode())
        status = "completed"
        print({"phase": "intervention_completed", "new_roots": result["new_roots"],
               "complete_roots": len(result["intervention_outcomes"]),
               "successes_by_control": {kind: sum(bool(r["success_by_control"].get(kind, False))
                                                 for r in result["intervention_outcomes"]) for kind in sorted(KINDS)}}, flush=True)
    finally:
        store.finish_run(status=status, n_rollouts=result["new_roots"] * (len(KINDS) - 1))
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
