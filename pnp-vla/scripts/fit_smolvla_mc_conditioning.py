"""Matched 32-training-root MC/BCE conditioning diagnostic; no ranking loss."""
import copy
import json
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_smolvla_combined_local import _load_local_credentials
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_scalar_returns import ScalarCritic
from pnp.smolvla_combined_success import load_or_create_combined_snapshot, prepare_combined_root_cache
from pnp.smolvla_success_critic import TimedRoots
from pnp.smolvla_tree_bellman_finetune import _root_batch, _to
from pnp.store import SupabaseStore

ROOT = Path.home() / "pnp-vla-runs"
DIGEST = "671b5b211099997fc83d1277"
OUTPUT = ROOT / "diagnostics/mc_bce_conditioning_32_v1"


def scores(model, batch, relative, mean, std):
    action = batch["action"]
    if relative:
        reference = action[:, :1].detach()
        action = torch.cat([((reference - mean) / std).expand_as(action),
                            100 * (action - reference) / std], -1)
    repeat = lambda x: x.repeat_interleave(9, 0)
    return model(repeat(batch["prefix"]), repeat(batch["pad"]), repeat(batch["robot"]),
                 repeat(batch["proprio"]), action.flatten(0, 1),
                 batch["action_valid"].flatten(0, 1)).reshape(-1, 9)


def expanded(base):
    arch = dict(base.architecture_config())
    dims = {k: arch.pop(k) for k in ("prefix_dim", "robot_dim", "proprio_dim")}
    arch["action_dim"] = 14
    model = ScalarCritic(**dims, config=QPlanningModelConfig(**arch))
    state = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    copied = {k: v for k, v in state.items() if k not in ("action_mean", "action_std", "action_projection.0.weight")}
    model.load_state_dict(copied, strict=False)
    with torch.no_grad():
        model.action_projection[0].weight.copy_(torch.cat([state["action_projection.0.weight"], state["action_projection.0.weight"] / 100], 1))
    return model


def verify_initial_equivalence(base, batch, mean, std, device):
    absolute = copy.deepcopy(base).to(device).eval()
    relative = expanded(base).to(device).eval()
    a = scores(absolute, batch, False, mean, std)
    b = scores(relative, batch, True, mean, std)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    F.binary_cross_entropy_with_logits(a, batch["success"].float()).backward()
    F.binary_cross_entropy_with_logits(b, batch["success"].float()).backward()
    gradient_error = 0.
    for (name, p), (other_name, q) in zip(absolute.named_parameters(), relative.named_parameters()):
        assert name == other_name
        mapped = q.grad
        if name == "action_projection.0.weight":
            mapped = mapped[:, :7] + mapped[:, 7:] / 100
        torch.testing.assert_close(p.grad, mapped, rtol=2e-3, atol=3e-5)
        gradient_error = max(gradient_error, float((p.grad - mapped).abs().max()))
    return {"initial_logit_max_error": float((a.detach() - b.detach()).abs().max()),
            "initial_mapped_gradient_max_error": gradient_error}


@torch.no_grad()
def evaluate(model, batch, relative, mean, std):
    model.eval()
    logits = scores(model, batch, relative, mean, std)
    labels = batch["success"].bool()
    macro, fresh = [], []
    for values, outcomes in zip(logits.cpu().numpy(), labels.cpu().numpy()):
        for offset, destination in ((0, macro), (1, fresh)):
            good = values[offset:][outcomes[offset:]]; bad = values[offset:][~outcomes[offset:]]
            if len(good) and len(bad):
                delta = good[:, None] - bad[None]
                destination.append(float(((delta > 0).sum() + .5 * (delta == 0).sum()) / delta.size))
    selected = labels[torch.arange(len(labels), device=labels.device), logits.argmax(1)]
    return {"bce": float(F.binary_cross_entropy_with_logits(logits, labels.float())),
            "macro_pair_accuracy": float(np.mean(macro)), "fresh_macro_pair_accuracy": float(np.mean(fresh)),
            "selected_successes": int(selected.sum()), "roots": len(labels),
            "within_root_logit_std": float(logits.std(1, unbiased=False).mean())}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    _load_local_credentials()
    store = SupabaseStore(); snapshot = load_or_create_combined_snapshot(store)
    cache = prepare_combined_root_cache(snapshot=snapshot, cache_root=ROOT / "cache", store=store)
    groups = {g["candidate_group_id"]: g for g in snapshot["groups"]}
    roots = TimedRoots(cache, cache["train_group_ids"], groups)
    chosen = np.random.default_rng(42930).choice([i for i, e in enumerate(roots.entries) if e["mixed"]], 32, replace=False)
    device = torch.device("mps")
    all_batch = _to(_root_batch([roots[int(i)] for i in chosen]), device)
    checkpoint = ROOT / "checkpoints/smolvla-q10-scalar-mc-td-v1-preaction" / DIGEST / "mc/latest.pt"
    initial = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if initial["snapshot_digest"] != cache["snapshot_digest"]:
        raise ValueError("initial checkpoint and root cache snapshot differ")
    arch = dict(initial["architecture"]); dims = {k: arch.pop(k) for k in ("prefix_dim", "robot_dim", "proprio_dim")}
    base = ScalarCritic(**dims, config=QPlanningModelConfig(**arch)); base.load_state_dict(initial["model"])
    mean, std = base.action_mean.detach().clone().to(device), base.action_std.detach().clone().to(device)
    labels = all_batch["success"].float().cpu().numpy()
    p = labels.mean(1); floor = float(np.mean(-p * np.log(p) - (1 - p) * np.log(1 - p)))
    # Identical state at a root; exact duplicate proposed chunks must share one score.
    duplicate_floor = []
    actions = all_batch["action"].cpu().numpy()
    valid_masks = all_batch["action_valid"].cpu().numpy()
    for action, valid, outcome in zip(actions, valid_masks, labels):
        effective = np.concatenate([(action * valid[..., None]).reshape(9, -1), valid], axis=1)
        _, assignment = np.unique(effective, axis=0, return_inverse=True)
        loss = 0.
        for key in np.unique(assignment):
            mask = assignment == key; q = outcome[mask].mean()
            if 0 < q < 1: loss += mask.sum() / 9 * (-q * np.log(q) - (1 - q) * np.log(1 - q))
        duplicate_floor.append(loss)
    audit = {"snapshot_digest": cache["snapshot_digest"], "initial_checkpoint": str(checkpoint),
             "initial_checkpoint_update": initial["update"], "root_selection_seed": 42930,
             "batch_rng_formula": "42 * 4000003 + update", "relative_delta_scale": 100,
             "updates": 2000, "roots_per_batch": 4, "candidates_per_root": 9,
             "root_ids": [roots.entries[int(i)]["candidate_group_id"] for i in chosen],
             "root_constant_bce_floor": floor, "duplicate_input_bce_floor": float(np.mean(duplicate_floor)),
             "stock_successes": int(labels[:, 0].sum()), "oracle_successes": int(labels.any(1).sum()),
             "objective": "ordinary per-candidate MC BCE; training-only memorization diagnostic"}
    audit.update(verify_initial_equivalence(base, {k: v[:4] for k, v in all_batch.items()}, mean, std, device))
    print({"initial_equivalence": audit}, flush=True)
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2))
    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow/tracking.db"))
    client = MlflowClient(); name = "smolvla-mc-bce-conditioning-32-v1"
    experiment = client.get_experiment_by_name(name)
    exp_id = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=(ROOT / "mlflow/artifacts").as_uri())
    for relative in (False, True):
        label = "relative100" if relative else "absolute"
        torch.manual_seed(42)
        model = (expanded(base) if relative else copy.deepcopy(base)).to(device)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(scores(model, all_batch, relative, mean, std), scores(copy.deepcopy(base).to(device), all_batch, False, mean, std), rtol=1e-4, atol=1e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        latest = OUTPUT / (label + ".pt"); saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
        history = saved["history"] if saved else []; start = saved["update"] if saved else 0
        if saved:
            if saved["audit"]["root_ids"] != audit["root_ids"] or saved["relative"] != relative:
                raise ValueError("resume checkpoint does not match diagnostic arm/root selection")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        with mlflow.start_run(experiment_id=exp_id, run_name=label, run_id=saved["run_id"] if saved else None) as run:
            if not saved: mlflow.log_params({"roots":32, "updates":2000, "lr":3e-4, "root_batch":4, "objective":"MC BCE", "relative":relative, "root_constant_bce_floor":floor, "duplicate_input_bce_floor":audit["duplicate_input_bce_floor"], "paired_update_rng":True})
            print({"arm":label, "mlflow_run":f"http://127.0.0.1:5001/#/experiments/{exp_id}/runs/{run.info.run_id}", "resumed":start}, flush=True)
            if not saved:
                metrics = evaluate(model, all_batch, relative, mean, std)
                history.append({"update": 0, **metrics})
                mlflow.log_metrics(metrics, step=0)
                print({"arm": label, "update": 0, **metrics}, flush=True)
            for step in range(start + 1, 2001):
                torch.manual_seed(42 * 4000003 + step)
                model.train(); positions=np.random.default_rng(42 * 4000003 + step).choice(32,4,replace=False)
                batch={k:v[positions] for k,v in all_batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(scores(model,batch,relative,mean,std),batch["success"].float())
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
                if step % 100 == 0:
                    metrics=evaluate(model,all_batch,relative,mean,std); history.append({"update":step,**metrics})
                    mlflow.log_metrics(metrics,step=step);print({"arm":label,"update":step,**metrics},flush=True)
                    temporary=latest.with_suffix(".tmp")
                    torch.save({"update":step,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":optimizer.state_dict(),"history":history,"run_id":run.info.run_id,"architecture":model.architecture_config(),"relative":relative,"mean":mean.cpu(),"std":std.cpu(),"audit":audit},temporary);temporary.replace(latest)
                    (OUTPUT/(label+"_report.json")).write_text(json.dumps({"history":history,"audit":audit},indent=2))
            mlflow.log_artifact(str(OUTPUT/(label+"_report.json")))


if __name__ == "__main__": main()
