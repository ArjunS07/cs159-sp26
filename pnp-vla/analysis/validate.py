"""Fatal validation gates for configuration-aware analyses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pnp.config import Method
from pnp.experiments import SCHEDULES
from .conditions import PAIR_KEYS, assign_standard_cohorts, condition_label


class ValidationError(ValueError):
    pass


def identity_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[PAIR_KEYS].drop_duplicates()


def pair_one_to_one(baseline: pd.DataFrame, condition: pd.DataFrame) -> pd.DataFrame:
    cols = PAIR_KEYS + ["success"]
    return baseline[cols].merge(condition[cols], on=PAIR_KEYS,
                                suffixes=("_baseline", "_condition"),
                                validate="one_to_one")


def coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_hash, group in df.groupby("config_hash", sort=True):
        rows.append({"config_hash": config_hash, "condition_label": condition_label(group.iloc[0]),
                     "method": group.iloc[0]["method"], "n_rollouts": len(group),
                     "n_identities": len(identity_key_frame(group)),
                     "full_ablation_n": int(group["full_ablation_member"].sum()),
                     "broad_validation_n": int((~group["full_ablation_member"]).sum())})
    return pd.DataFrame(rows)


def _single(df: pd.DataFrame, *, method: str, steps=None, inference_steps=None) -> pd.DataFrame:
    out = df[df["method"] == method]
    if steps is not None:
        out = out[out["pnp_step_indices"].apply(
            lambda x: tuple(x) == tuple(steps) if x is not None else False)]
    if inference_steps is not None:
        out = out[out["num_inference_steps"] == inference_steps]
    return out


def _crosscheck_run_cohorts(df: pd.DataFrame, runs: pd.DataFrame) -> list[str]:
    warnings = []
    if runs.empty or "config_json" not in runs:
        return ["run metadata unavailable for cohort cross-check"]
    declared = {}
    for row in runs.to_dict("records"):
        cfg = row.get("config_json") or {}
        if isinstance(cfg, str):
            try: cfg = json.loads(cfg)
            except json.JSONDecodeError: cfg = {}
        if cfg.get("cohort"):
            declared[str(row.get("run_id"))] = cfg["cohort"]
    for run_id, group in df.groupby("run_id"):
        expected = declared.get(str(run_id))
        if expected == "full_ablation" and not group["full_ablation_member"].all():
            raise ValidationError(f"run {run_id} declares full_ablation but contains other tasks")
        if expected == "broad_validation" and group["full_ablation_member"].any():
            raise ValidationError(f"run {run_id} declares broad_validation but contains full-ablation tasks")
    if not declared:
        warnings.append("no cohort declarations found in run metadata")
    return warnings


def validate_standard(rollouts: pd.DataFrame, runs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    required = set(PAIR_KEYS + ["experiment", "rollout_id", "config_hash", "method", "status", "success"])
    missing = sorted(required - set(rollouts.columns))
    if missing:
        raise ValidationError(f"missing rollout columns: {missing}")
    if rollouts.empty:
        raise ValidationError("snapshot has no rollouts")
    bad = rollouts[rollouts["status"] != "completed"]
    if len(bad):
        raise ValidationError(f"primary analysis contains {len(bad)} non-completed rows")
    if rollouts["config_hash"].isna().any():
        raise ValidationError("null config_hash in primary analysis")
    duplicate = rollouts.duplicated(["experiment", *PAIR_KEYS, "config_hash"], keep=False)
    if duplicate.any():
        raise ValidationError(f"duplicate identity/config rows: {int(duplicate.sum())}")
    df = assign_standard_cohorts(rollouts)
    identities = identity_key_frame(df)
    if len(df) != 1920 or len(identities) != 400:
        raise ValidationError(f"expected 1,920 rollouts/400 identities, got {len(df)}/{len(identities)}")

    expected = [(Method.UNCERTAINTY, None, None, 400),
                (Method.EXTRA_STEPS, None, 16, 400),
                (Method.EXTRA_STEPS, None, 19, 80),
                (Method.EXTRA_STEPS, None, 25, 80)]
    expected += [(Method.REFINEMENT, schedule, None, 400 if schedule == (4, 5) else 80)
                 for schedule in SCHEDULES]
    selected_hashes = set()
    for method, steps, inference, n in expected:
        group = _single(df, method=method, steps=steps, inference_steps=inference)
        hashes = group["config_hash"].unique()
        if len(group) != n or len(hashes) != 1:
            raise ValidationError(f"coverage mismatch for {method}/{steps or inference}: rows={len(group)}, configs={len(hashes)}")
        selected_hashes.add(hashes[0])
    if len(selected_hashes) != 12 or df["config_hash"].nunique() != 12:
        raise ValidationError("expected exactly 12 behavior-defining configurations")
    if int(df["full_ablation_member"].groupby(df[PAIR_KEYS].astype(str).agg("|".join, axis=1)).first().sum()) != 80:
        raise ValidationError("full-ablation manifest does not identify exactly 80 identities")
    observed = _single(df, method=Method.UNCERTAINTY)
    if not observed["pnp_step_indices"].apply(
            lambda x: tuple(x) == tuple(range(1, 10)) if x is not None else False).all():
        raise ValidationError("observed arm does not contain step-indexed telemetry schedule 1-9")
    if "episode_seed" in observed and "perturb_seed" in observed and not (observed["episode_seed"] == observed["perturb_seed"]).all():
        raise ValidationError("observed arm perturbation seed is not the isolated episode seed")
    warnings = _crosscheck_run_cohorts(df, runs if runs is not None else pd.DataFrame())
    matrix = coverage_matrix(df)
    return df, {"status": "valid", "n_rollouts": len(df), "n_identities": len(identities),
                "n_configurations": df["config_hash"].nunique(), "warnings": warnings,
                "coverage": matrix.to_dict("records")}


def validate_artifact_references(df: pd.DataFrame, columns=("ahats_path", "pcp_chunks_path")) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df:
            rows.append({"artifact_type": column, "referenced": 0, "status": "not_available"})
        else:
            n = int(df[column].notna().sum())
            rows.append({"artifact_type": column, "referenced": n,
                         "status": "available_unverified" if n else "not_available"})
    return pd.DataFrame(rows)
