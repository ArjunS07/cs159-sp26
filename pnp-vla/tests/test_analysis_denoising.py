import numpy as np
import pandas as pd

from analysis.denoising import analyze, episode_features, out_of_fold_models, profile_by_outcome
from pnp.config import Method


def _fixture(n=40):
    rollouts, steps, vectors = [], [], []
    for episode in range(n):
        rollout_id = f"r{episode}"
        fail = episode % 2
        rollouts.append({"rollout_id": rollout_id, "suite": f"suite_{episode % 2}",
                         "success": not fail, "method": Method.UNCERTAINTY})
        for chunk in range(2):
            for step in range(1, 10):
                value = .01 + fail * step * .002
                vector = np.full(7, fail * step * .01)
                steps.append({"rollout_id": rollout_id, "chunk_idx": chunk,
                              "euler_step": step, "u_mean": value,
                              "u_max": value * 2, "a_std_mean": value / 2})
                vectors.append({"rollout_id": rollout_id, "chunk_idx": chunk,
                                "euler_step": step, "a_mean_vec": vector})
    return pd.DataFrame(rollouts), pd.DataFrame(steps), pd.DataFrame(vectors)


def test_episode_features_preserve_one_row_per_observed_rollout():
    rollouts, steps, vectors = _fixture()
    features, profile = episode_features(rollouts, steps, vectors)
    assert len(features) == 40
    assert features.rollout_id.is_unique
    assert set(profile.euler_step) == set(range(1, 10))
    assert features.action_motion_e2.notna().all()
    summary = profile_by_outcome(profile)
    assert {"u_mean_sem", "a_std_mean_sem", "action_motion_sem"} <= set(summary)
    early, _ = episode_features(rollouts, steps, vectors, max_chunk_exclusive=1)
    assert len(early) == 40


def test_denoising_models_emit_held_out_predictions_for_each_feature_set():
    rollouts, steps, vectors = _fixture()
    features, _ = episode_features(rollouts, steps, vectors)
    summary, predictions = out_of_fold_models(features)
    assert predictions.groupby("model").rollout_id.nunique().eq(40).all()
    assert predictions.groupby(["model", "rollout_id"]).size().eq(1).all()
    assert set(predictions.fold) == set(range(5))
    assert set(summary.model) == {
        "legacy_scalar", "uncertainty_step_profile", "full_denoising_dynamics"}


def test_analysis_prefix_supports_standard_and_pro_reports():
    rollouts, steps, vectors = _fixture()
    tables = analyze(rollouts, steps, vectors, prefix="")
    assert "denoising_oof_models" in tables
    assert "pro_denoising_oof_models" not in tables
