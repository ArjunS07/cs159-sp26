"""Common offline outcome/gradient audit. No training or simulator calls.

Cached scalar reports plus CPU inference on existing CNN frozen features.
Unsupported/missing evidence remains explicit in the coverage table.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_smolvla_q_predictions import BASE, DIGEST, REPORTS, summarize
from scripts.run_smolvla_local_continuation import scalar_from_architecture
from scripts.run_smolvla_cnn_size_study import cached_scores, take
from pnp.smolvla_anchored_critic import AnchoredCritic
from scripts.train_smolvla_anchored_local import load_data
from scripts.diagnose_smolvla_scalar_returns import diagnose


def records_from(split):
    return split.get("root_records") or split.get("gradients", {}).get("root_records", [])


def gradient_summary(records):
    usable = [r for r in records if "gradient_projection" in r]
    pair = []
    for r in usable:
        y = np.asarray(r["labels"], bool)[1:]
        projection = np.asarray(r["gradient_projection"], float)[1:]
        if y.any() and (~y).any():
            d = projection[y, None] - projection[None, ~y]
            pair.append(float(((d > 0) + .5*(d == 0)).mean()))
    rng = np.random.default_rng(73026)
    interval = (np.quantile(np.asarray(pair)[rng.integers(len(pair), size=(4000, len(pair)))].mean(1),
                           [.025, .975]).tolist() if pair else None)
    norms = [r.get("gradient_norm", r.get("normalized_gradient_norm")) for r in usable]
    norms = [n for n in norms if n is not None]
    dims = [r["gradient_by_dim"] for r in usable if "gradient_by_dim" in r]
    return {"roots_with_projection": len(usable), "fresh_mixed_roots": len(pair),
            "fresh_gradient_pair_accuracy": float(np.mean(pair)) if pair else None,
            "fresh_gradient_pair_episode_bootstrap_95": interval,
            "gradient_norm_quantiles": np.quantile(norms, [0, .25, .5, .75, .95, 1]).tolist() if norms else None,
            "cached_mean_absolute_standardized_gradient_by_coordinate": np.mean(dims, axis=0).tolist() if dims else None,
            "interpretation": "Observed displacement alignment; no gradient intervention outcomes."}


def endpoint_summary(records):
    metrics = summarize(records)
    y = np.asarray([r["labels"] for r in records], bool)
    z = np.asarray([r["logits"] for r in records], float)
    # Argmax on logits preserves ordering even when sigmoid saturates numerically.
    selected = y[np.arange(len(y)), z.argmax(1)]
    oracle = y.any(1)
    top = z == z.max(1, keepdims=True)
    metrics.update(oracle_successes=int(oracle.sum()),
                   oracle_missed_successes=int((oracle & ~selected).sum()),
                   oracle_improvement_over_stock=int(oracle.sum()-y[:, 0].sum()),
                   selected_successes_logit_argmax=int(selected.sum()),
                   tied_top_roots=int((top.sum(1)>1).sum()),
                   random_tie_selection_expected_successes=float(((y*top).sum(1)/top.sum(1)).sum()),
                   selection_tie_rule="First candidate, stock first; logit argmax.")
    metrics["gradient"] = gradient_summary(records)
    return metrics


def validate_alignment(records, reference):
    ids = [r["group_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate root IDs")
    labels = {r["group_id"]: r["labels"] for r in records}
    if labels != reference:
        raise ValueError("Root IDs or ordered candidate outcomes differ")


def cnn_records(model, data, ids):
    rows, gradients = [], []
    model.eval().requires_grad_(False)
    for start in range(0, len(ids), 32):
        b = take(data, slice(start, start+32))
        with torch.no_grad():
            z = cached_scores(model, b).numpy()
        reference = b["action"][:, 0].detach()
        action = reference.clone().requires_grad_(True)
        stock_z = model.logits_from_features(b["features"], b["baseline"], action,
                                            b["valid"][:, 0], reference)
        g, = torch.autograd.grad(stock_z.sum(), action)
        normalized_g = g.detach()*model.action_std*b["valid"][:, 0, :, None]
        gradients.extend(normalized_g.numpy())
        delta = (b["action"]-reference[:, None])/model.action_std
        projections = (normalized_g[:, None]*delta).sum((-1, -2)).numpy()
        for j, labels in enumerate(b["labels"].numpy().astype(bool)):
            rows.append({"group_id": ids[start+j], "labels": labels.tolist(), "logits": z[j].tolist(),
                         "gradient_projection": projections[j].tolist(),
                         "normalized_gradient_norm": float(normalized_g[j].norm()),
                         "gradient_by_dim": normalized_g[j].abs().mean(0).tolist()})
    g = np.asarray(gradients)
    return rows, {"mean_signed_standardized_gradient_10x7": g.mean(0).tolist(),
                  "mean_absolute_standardized_gradient_10x7": np.abs(g).mean(0).tolist(),
                  "gradient_energy_fraction_10x7": ((g*g).sum(0)/max(float((g*g).sum()), 1e-30)).tolist()}


def finite_difference_check(model, data):
    # Eight roots in immutable cache order; no label-based selection.
    b = take(data, slice(0, 8))
    reference = b["action"][:, 0].detach()
    a = reference.clone().requires_grad_(True)
    def score(action):
        return model.logits_from_features(b["features"], b["baseline"], action,
                                         b["valid"][:, 0], reference).squeeze(-1)
    z = score(a)
    gradient, = torch.autograd.grad(z.sum(), a)
    rng = np.random.default_rng(73026)
    direction = torch.tensor(rng.normal(size=a.shape), dtype=a.dtype)
    direction *= b["valid"][:, 0, :, None]
    direction /= direction.square().mean((-1, -2), keepdim=True).sqrt()
    direction *= model.action_std
    analytic_z = (gradient*direction).sum((-1, -2)).detach()
    analytic_q = analytic_z*z.detach().sigmoid()*(1-z.detach().sigmoid())
    results = []
    with torch.no_grad():
        for eps in (.001, .003, .01):
            plus, minus = score(reference+eps*direction), score(reference-eps*direction)
            numeric_z = (plus-minus)/(2*eps)
            numeric_q = (plus.sigmoid()-minus.sigmoid())/(2*eps)
            results.append({"normalized_epsilon": eps,
                            "logit_absolute_error_mean": float((numeric_z-analytic_z).abs().mean()),
                            "probability_absolute_error_mean": float((numeric_q-analytic_q).abs().mean()),
                            "analytic_logit_derivative": analytic_z.tolist(),
                            "finite_difference_logit_derivative": numeric_z.tolist()})
    return {"roots": 8, "direction_seed": 73026, "checks": results,
            "interpretation": "Numerical consistency only; fixed detached reference."}


def main():
    torch.set_num_threads(2)
    destination = BASE/"diagnostics/critic_evaluation_audit_v1"
    destination.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REPORTS["pcp_frozen_late_mc"].read_text())
    labels = {s:{r["group_id"]:r["labels"] for r in records_from(d)} for s,d in reference["splits"].items()}
    if set(labels["train"]) & set(labels["validation"]):
        raise ValueError("Train/validation overlap")
    result = {"protocol": "docs/smolvla_critic_evaluation_protocol.md",
              "dataset_digest": DIGEST, "validation_status": "exploratory, repeatedly examined",
              "unique_train_episodes":1280, "unique_validation_episodes":320, "models":{}, "coverage":[]}
    prior_path = destination/"report.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    if prior and prior.get("dataset_digest") != DIGEST:
        raise ValueError("Prior report uses a different dataset")
    paths = dict(REPORTS)
    for path in sorted((BASE/"diagnostics/overnight_20260930").glob("*_diagnostics.json")):
        if "mc_roots_late" not in path.name:
            paths["overnight_"+path.name.replace("_diagnostics.json", "")] = path
    for name, path in paths.items():
        report = json.loads(path.read_text())
        entry = {"source_report":str(path), "source_report_sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "splits":{}}
        for split, data in report["splits"].items():
            records = records_from(data)
            validate_alignment(records, labels[split])
            entry["splits"][split] = endpoint_summary(records)
        result["models"][name] = entry
        result["coverage"].append({"model":name,"outcome":"complete","gradient_alignment":"complete",
                                   "numerical_derivatives":"not retained in cached report",
                                   "validation_bellman":"unavailable: held-out transition cache missing"})
    study = BASE/"checkpoints/smolvla_cnn_size_study_v1"/DIGEST
    cache = torch.load(study/"frozen_features.pt", map_location="cpu", weights_only=False)
    if cache["contract"]["snapshot_digest"] != DIGEST:
        raise ValueError("Frozen feature snapshot differs")
    anchored_directory = BASE/"checkpoints/smolvla_anchored_cnn_v1"/DIGEST
    jobs = []
    for directory in sorted(study.glob("width*_seed*")):
        final = json.loads((directory/"final_report.json").read_text())
        for endpoint in ("fixed_final", "early_selected"):
            jobs.append((directory.name+"/"+endpoint, endpoint,
                         directory/("fixed_final.pt" if endpoint == "fixed_final" else f"early_step_{final['early_checkpoint_update']:06d}.pt")))
    jobs.extend(("anchored_initial/"+family,"fixed_final",anchored_directory/family/"latest.pt")
                for family in ("transformer","temporal_cnn"))
    for name,endpoint,path in jobs:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            previous = prior.get("models",{}).get(name,{})
            if previous.get("checkpoint_sha256") == sha:
                result["models"][name] = previous
                result["coverage"].append(next(row for row in prior["coverage"] if row["model"] == name))
                continue
            saved = torch.load(path,map_location="cpu",weights_only=False)
            if saved["snapshot_digest"] != DIGEST:
                raise ValueError("Checkpoint snapshot differs")
            arch = saved["architecture"]
            family = "transformer" if arch["model_family"] == "anchored_transformer" else "temporal_cnn"
            model = AnchoredCritic(scalar_from_architecture(arch["base_architecture"]), family,
                                   width=arch["width"],delta_scale=arch["delta_scale"])
            model.load_state_dict(saved["model"])
            entry = {"checkpoint":str(path),"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                     "update":saved["update"],"selection":endpoint,"splits":{}}
            for split,data in cache["data"].items():
                records,dimensions = cnn_records(model,data,cache["contract"]["group_ids"][split])
                validate_alignment(records,labels[split])
                entry["splits"][split] = endpoint_summary(records)
                entry["splits"][split]["gradient"].update(dimensions)
            entry["finite_difference"] = finite_difference_check(model,cache["data"]["validation"])
            result["models"][name] = entry
            result["coverage"].append({"model":name,"outcome":"complete","gradient_alignment":"complete",
                                       "numerical_derivatives":"8-root probe complete",
                                       "validation_bellman":"unavailable: held-out transition cache missing"})
            print({"completed":name},flush=True)
    (destination/"report.json").write_text(json.dumps(result,indent=2))
    roots = load_data(BASE,anchored_directory)
    for arm in ("mc","td"):
        path = BASE/"checkpoints/smolvla-q10-scalar-mc-td-v1-preaction"/DIGEST/arm/"latest.pt"
        saved = torch.load(path,map_location="cpu",weights_only=False)
        if saved["snapshot_digest"] != DIGEST or saved["update"] != 2000:
            raise ValueError("Original scalar baseline differs")
        model = scalar_from_architecture(saved["architecture"])
        model.load_state_dict(saved["model"])
        entry = {"checkpoint":str(path),"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                 "update":saved["update"],"splits":{}}
        name = "original_scalar_"+arm+"_update2000"
        for split,data in roots.items():
            _, records = diagnose(model,data,torch.device("cpu"))
            validate_alignment(records,labels[split])
            entry["splits"][split] = endpoint_summary(records)
            print({"completed":name,"split":split},flush=True)
        result["models"][name] = entry
        result["coverage"].append({"model":name,"outcome":"complete","gradient_alignment":"complete",
                                   "numerical_derivatives":"pending",
                                   "validation_bellman":"unavailable: held-out transition cache missing"})
    for name,reason in {
        "legacy_distributional_ranking_and_conditioning_variants":"Require explicit target/input adapters and leakage/provenance review before probability comparisons."
    }.items():
        result["coverage"].append({"model":name,"outcome":"pending","reason":reason})
    (destination/"report.json").write_text(json.dumps(result,indent=2))
    lines = ["# Common critic audit", "", "Existing reused validation: 320 episodes, nine correlated candidate outcomes each. No new policy rollouts. Training results and full calibration/gradient diagnostics are in report.json.", "",
             "| Model / endpoint | Accuracy | AUROC | Brier ↓ | MC log loss ↓ | Fresh gradient pair | Selected / oracle |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name,entry in result["models"].items():
        m = entry["splits"]["validation"]; g = m["gradient"]["fresh_gradient_pair_accuracy"]
        gs = f"{g:.3f}" if g is not None else "unavailable"
        lines.append(f"| {name} | {m['classification_accuracy_at_fixed_half']:.3f} | {m['auroc']:.3f} | {m['brier']:.4f} | {m['mc_log_loss']:.4f} | {gs} | {m['selected_successes_logit_argmax']} / {m['oracle_successes']} |")
    lines += ["", "Stock: 209/320; expected uniform random: 198.11/320; oracle: 226/320. Fresh gradient ordering uses only the 81 fresh-mixed validation roots; chance ordering is .5.", "", "## Coverage limitations", "",
              "Held-out Bellman error is unavailable: existing full transition cache includes training roots only. Cached scalar derivatives have observational alignment evidence but no new numerical derivative test. CNN endpoints have an eight-root numerical probe and 10×7 gradient summaries; those do not establish useful intervention directions.", ""]
    for row in result["coverage"]:
        if row.get("outcome") == "pending":
            lines.append(f"- {row['model']}: {row['reason']}")
    (destination/"results.md").write_text("\n".join(lines)+"\n")
    print({"report":str(destination/"results.md"),"evaluated_endpoints":len(result["models"])},flush=True)


if __name__ == "__main__":
    main()
