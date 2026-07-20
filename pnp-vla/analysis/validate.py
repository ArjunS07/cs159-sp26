"""Fatal validation gates for configuration-aware analyses."""
from __future__ import annotations

import json

import pandas as pd

from pnp.config import Method
from pnp.experiments import SCHEDULES
from pnp.libero_pro import CANONICAL_PRO_SUITES, EXPANDED_PRO_SUITES, LIBERO_PRO_REVISION
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


def pro_coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_hash, group in df.groupby("config_hash", sort=True):
        rows.append({"config_hash": config_hash, "condition_label": condition_label(group.iloc[0]),
                     "method": group.iloc[0]["method"], "n_rollouts": len(group),
                     "n_identities": len(identity_key_frame(group)),
                     "n_suites": group["suite"].nunique(),
                     "canonical_n": int(group["canonical_member"].fillna(False).sum()),
                     "expanded_n": int(group["expanded_member"].fillna(False).sum())})
    return pd.DataFrame(rows)


def _run_config(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


def validate_pro(rollouts: pd.DataFrame, runs: pd.DataFrame | None = None,
                 artifact_validation: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Validate the canonical six-suite, three-condition LIBERO-PRO collection."""
    required = set(PAIR_KEYS + ["experiment", "rollout_id", "run_id", "benchmark",
                   "config_hash", "method", "status", "success", "suite_family",
                   "canonical_member", "expanded_member"])
    missing = sorted(required - set(rollouts.columns))
    if missing:
        raise ValidationError(f"missing PRO rollout columns: {missing}")
    if rollouts.empty:
        raise ValidationError("PRO snapshot has no rollouts")
    if set(rollouts["benchmark"].dropna()) != {"libero_pro"}:
        raise ValidationError(f"expected benchmark=libero_pro, got {sorted(rollouts['benchmark'].dropna().unique())}")
    bad = rollouts[rollouts["status"] != "completed"]
    if len(bad):
        raise ValidationError(f"PRO primary analysis contains {len(bad)} non-completed rows")
    if rollouts["config_hash"].isna().any():
        raise ValidationError("null config_hash in PRO primary analysis")
    duplicate = rollouts.duplicated(["experiment", *PAIR_KEYS, "config_hash"], keep=False)
    if duplicate.any():
        raise ValidationError(f"duplicate PRO identity/config rows: {int(duplicate.sum())}")
    df = rollouts.copy()
    df["condition_label"] = [condition_label(r) for r in df.to_dict("records")]
    identities = identity_key_frame(df)
    if len(df) != 1800 or len(identities) != 600:
        raise ValidationError(f"expected 1,800 PRO rollouts/600 identities, got {len(df)}/{len(identities)}")
    if set(df["suite"].unique()) != set(CANONICAL_PRO_SUITES):
        raise ValidationError("PRO suites do not match the canonical six-suite manifest")
    per_suite_identities = identity_key_frame(df).groupby("suite").size()
    if not (per_suite_identities == 100).all():
        raise ValidationError(f"expected 100 identities per PRO suite, got {per_suite_identities.to_dict()}")
    if not df["canonical_member"].fillna(False).all():
        raise ValidationError("canonical PRO collection contains non-canonical membership")
    expected_expanded = df["suite"].isin(EXPANDED_PRO_SUITES)
    if not (df["expanded_member"].fillna(False).astype(bool) == expected_expanded).all():
        raise ValidationError("expanded membership flags disagree with the versioned manifest")
    expected = [(Method.UNCERTAINTY, None, None),
                (Method.EXTRA_STEPS, None, 16),
                (Method.REFINEMENT, (4, 5), None)]
    selected_hashes = set()
    for method, steps, inference in expected:
        group = _single(df, method=method, steps=steps, inference_steps=inference)
        if len(group) != 600 or group["config_hash"].nunique() != 1:
            raise ValidationError(f"PRO coverage mismatch for {method}/{steps or inference}: {len(group)} rows")
        if not (group.groupby("suite").size() == 100).all():
            raise ValidationError(f"PRO condition {method}/{steps or inference} lacks 100 rows per suite")
        selected_hashes.add(group["config_hash"].iloc[0])
    if len(selected_hashes) != 3 or df["config_hash"].nunique() != 3:
        raise ValidationError("expected exactly three PRO configurations")
    observed = _single(df, method=Method.UNCERTAINTY)
    if not observed["pnp_step_indices"].apply(
            lambda x: tuple(x) == tuple(range(1, 10)) if x is not None else False).all():
        raise ValidationError("PRO observed arm does not record steps 1-9")
    if "episode_seed" in observed and "perturb_seed" in observed and not (
            observed["episode_seed"] == observed["perturb_seed"]).all():
        raise ValidationError("PRO observed perturbation seed is not isolated")
    warnings = []
    artifact_validation = artifact_validation or {}
    pcp_artifacts = artifact_validation.get("pcp_chunks_path")
    if pcp_artifacts is not None:
        if pcp_artifacts.get("status") != "valid" or pcp_artifacts.get("verified") != 600:
            raise ValidationError("expected 600 verified PRO PCP feature artifacts")
    else:
        warnings.append("PCP feature artifact references were not storage-verified")

    run_summary = {"completed_shards": [], "failed_runs": 0}
    runs = runs if runs is not None else pd.DataFrame()
    if runs.empty:
        warnings.append("PRO run metadata unavailable")
    else:
        completed_shards = set()
        for row in runs.to_dict("records"):
            cfg = _run_config(row.get("config_json"))
            if row.get("status") == "failed":
                run_summary["failed_runs"] += 1
                continue
            if row.get("status") != "completed" or row.get("driver") != "canonical_pro_core":
                continue
            if cfg.get("cohort") != "canonical" or cfg.get("n_configs") != 3:
                raise ValidationError(f"invalid canonical PRO run config for {row.get('run_id')}")
            if cfg.get("libero_pro_revision") != LIBERO_PRO_REVISION:
                raise ValidationError(f"LIBERO-PRO revision mismatch for {row.get('run_id')}")
            if cfg.get("shard_count") != 6:
                raise ValidationError(f"PRO run {row.get('run_id')} does not declare six shards")
            completed_shards.add(int(cfg["shard_index"]))
        if completed_shards != set(range(6)):
            raise ValidationError(f"completed PRO shard coverage mismatch: {sorted(completed_shards)}")
        run_summary["completed_shards"] = sorted(completed_shards)
    matrix = pro_coverage_matrix(df)
    return df, {"status": "valid", "benchmark": "libero_pro", "n_rollouts": len(df),
                "n_identities": len(identities), "n_configurations": df["config_hash"].nunique(),
                "n_suites": df["suite"].nunique(), "canonical_complete": True,
                "expanded_complete": False, "run_summary": run_summary,
                "warnings": warnings, "artifact_validation": artifact_validation,
                "coverage": matrix.to_dict("records")}
