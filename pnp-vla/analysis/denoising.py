"""Prospective failure analysis using the full denoising trajectory."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pnp.config import Method
from .statistics import auc_metrics, bootstrap_auc


def _folds(frame: pd.DataFrame, n_folds: int) -> pd.Series:
    folds = pd.Series(-1, index=frame.index, dtype=int)
    for _, indices in frame.groupby(["suite", "fail"]).groups.items():
        for offset, index in enumerate(sorted(indices)):
            folds.loc[index] = offset % n_folds
    return folds


def episode_features(rollouts: pd.DataFrame, steps: pd.DataFrame,
                     vectors: pd.DataFrame, *,
                     max_chunk_exclusive: int | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one prospective feature row per observed rollout."""
    observed = rollouts[rollouts.method == Method.UNCERTAINTY][
        ["rollout_id", "suite", "success"]].copy()
    observed["fail"] = (~observed.success.astype(bool)).astype(int)
    telemetry = steps.merge(observed[["rollout_id"]], on="rollout_id", how="inner")
    if max_chunk_exclusive is not None:
        telemetry = telemetry[telemetry.chunk_idx < max_chunk_exclusive]
    profile = telemetry.groupby(["rollout_id", "euler_step"]).agg(
        u_mean=("u_mean", "mean"), u_max=("u_max", "mean"),
        a_std_mean=("a_std_mean", "mean")).reset_index()

    action = vectors.merge(observed[["rollout_id"]], on="rollout_id", how="inner").copy()
    if max_chunk_exclusive is not None:
        action = action[action.chunk_idx < max_chunk_exclusive]
    action = action.sort_values(["rollout_id", "chunk_idx", "euler_step"])
    action["_vec"] = action.a_mean_vec.map(lambda value: np.asarray(value, dtype=float))
    action["action_motion"] = action.groupby(
        ["rollout_id", "chunk_idx"], sort=False)["_vec"].transform(
            lambda values: pd.Series(
                [np.nan] + [np.linalg.norm(values.iloc[i] - values.iloc[i - 1])
                            for i in range(1, len(values))],
                index=values.index))
    motion = action.groupby(["rollout_id", "euler_step"]).action_motion.mean().reset_index()
    profile = profile.merge(motion, on=["rollout_id", "euler_step"], how="left")

    features = observed.set_index("rollout_id")
    for metric in ("u_mean", "u_max", "a_std_mean", "action_motion"):
        wide = profile.pivot(index="rollout_id", columns="euler_step", values=metric)
        wide.columns = [f"{metric}_e{int(step)}" for step in wide.columns]
        features = features.join(wide)
    u_columns = [f"u_mean_e{step}" for step in range(1, 10)]
    features["u_profile_slope"] = features[u_columns].apply(
        lambda row: np.polyfit(np.arange(1, 10), row, 1)[0], axis=1)
    features["u_late_early_ratio"] = (
        features[[f"u_mean_e{x}" for x in (7, 8, 9)]].mean(axis=1) /
        features[[f"u_mean_e{x}" for x in (1, 2, 3)]].mean(axis=1).replace(0, np.nan))
    features["action_path_length"] = features[
        [f"action_motion_e{x}" for x in range(2, 10)]].sum(axis=1)
    features["legacy_episode_mean"] = telemetry.groupby("rollout_id").u_mean.mean()
    return features.reset_index(), profile.merge(
        observed[["rollout_id", "suite", "fail"]], on="rollout_id", how="left")


def step_metrics(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("u_mean", "u_max", "a_std_mean", "action_motion"):
        first_step = 2 if metric == "action_motion" else 1
        for step in range(first_step, 10):
            score = features[f"{metric}_e{step}"]
            rows.append({"metric": metric, "euler_step": step,
                         **auc_metrics(features.fail, score)})
    return pd.DataFrame(rows)


def profile_by_outcome(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.groupby(["euler_step", "fail"]).agg(
        n=("rollout_id", "nunique"),
        u_mean=("u_mean", "mean"), u_mean_sd=("u_mean", "std"),
        a_std_mean=("a_std_mean", "mean"), a_std_mean_sd=("a_std_mean", "std"),
        action_motion=("action_motion", "mean"),
        action_motion_sd=("action_motion", "std")).reset_index()
    for metric in ("u_mean", "a_std_mean", "action_motion"):
        out[f"{metric}_sem"] = out[f"{metric}_sd"] / np.sqrt(out.n)
    return out


def out_of_fold_models(features: pd.DataFrame, n_folds: int = 5
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare scalar and trajectory feature sets using held-out predictions only."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_sets = {
        "legacy_scalar": ["legacy_episode_mean"],
        "uncertainty_step_profile": [f"u_mean_e{x}" for x in range(1, 10)],
        "full_denoising_dynamics": (
            [f"{metric}_e{x}" for metric in ("u_mean", "u_max", "a_std_mean")
             for x in range(1, 10)] +
            [f"action_motion_e{x}" for x in range(2, 10)] +
            ["u_profile_slope", "u_late_early_ratio", "action_path_length"]),
    }
    frame = features.copy()
    frame["fold"] = _folds(frame, n_folds)
    predictions = []
    for model_name, columns in feature_sets.items():
        for fold in range(n_folds):
            train, test = frame[frame.fold != fold], frame[frame.fold == fold]
            model = make_pipeline(SimpleImputer(), StandardScaler(),
                                  LogisticRegression(C=1, max_iter=2000))
            model.fit(train[columns], train.fail)
            probability = model.predict_proba(test[columns])[:, 1]
            predictions.extend(
                {"rollout_id": row.rollout_id, "suite": row.suite, "fail": row.fail,
                 "fold": fold, "model": model_name, "failure_probability": probability}
                for row, probability in zip(test.itertuples(), probability))
    oof = pd.DataFrame(predictions)
    summary = []
    for model_name, group in oof.groupby("model"):
        summary.append({"scope": "pooled", "model": model_name,
                        **bootstrap_auc(group.fail, group.failure_probability)})
        for suite, suite_group in group.groupby("suite"):
            summary.append({"scope": suite, "model": model_name,
                            **auc_metrics(suite_group.fail, suite_group.failure_probability)})
    return pd.DataFrame(summary), oof


def analyze(rollouts: pd.DataFrame, steps: pd.DataFrame,
            vectors: pd.DataFrame, *, prefix: str = "pro_") -> dict[str, pd.DataFrame]:
    features, profile = episode_features(rollouts, steps, vectors)
    early_features, _ = episode_features(
        rollouts, steps, vectors, max_chunk_exclusive=4)
    model_tables = []
    for window, frame in (("first_4_chunks", early_features),
                          ("full_episode", features)):
        model_summary, _ = out_of_fold_models(frame)
        model_tables.append(model_summary.assign(window=window))
    feature_auc = []
    for window, frame in (("first_4_chunks", early_features),
                          ("full_episode", features)):
        for name in ("u_profile_slope", "u_late_early_ratio", "action_path_length"):
            feature_auc.append({"window": window, "feature": name,
                                **auc_metrics(frame.fail, frame[name])})
    return {
        f"{prefix}denoising_step_metrics": step_metrics(features),
        f"{prefix}denoising_profile_by_outcome": profile_by_outcome(profile),
        f"{prefix}denoising_feature_metrics": pd.DataFrame(feature_auc),
        f"{prefix}denoising_oof_models": pd.concat(model_tables, ignore_index=True),
    }
