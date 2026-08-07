import numpy as np
import pandas as pd

from analysis.pro_expanded import (episode_contraction,
                                   refinement_effect_by_uncertainty_bin,
                                   uncertainty_window_sweep)
from analysis.validate import validate_pro_expanded
from analysis.report import expanded_pro_figures
from pnp.config import Method
from pnp.experiments import (PRO_EXPANDED_EXPERIMENT, PRO_EXPANDED_K,
                             PRO_EXPANDED_STEPS, expanded_pro_suites)
from pnp.libero_pro import CANONICAL_PRO_SUITES, LIBERO_PRO_REVISION


def _expanded_fixture():
    rows = []
    for suite in expanded_pro_suites():
        n = 100 if suite.endswith("_with_milk") else 200
        for episode in range(n):
            identity = {
                "experiment": PRO_EXPANDED_EXPERIMENT,
                "run_id": f"run-{episode % 6}",
                "benchmark": "libero_pro", "suite": suite,
                "task_idx": episode % 10, "episode_idx": episode // 10,
                "init_state_hash": f"{suite}-{episode}", "status": "completed",
                "suite_family": "distractor" if "_with_" in suite else "position_perturb",
                "canonical_member": suite in CANONICAL_PRO_SUITES,
                "expanded_member": True, "pnp_step_indices": list(PRO_EXPANDED_STEPS),
                "pnp_k": PRO_EXPANDED_K, "u_mean_episode": .04,
                "n_steps": 100, **{f"u_mean_d{i}": .04 for i in range(7)},
            }
            for method, config_hash in ((Method.UNCERTAINTY, "observed-k5"),
                                        (Method.REFINEMENT, "refine-k5")):
                rows.append({**identity, "rollout_id": f"{method}-{suite}-{episode}",
                             "method": method, "config_hash": config_hash,
                             "success": episode % 3 != 0,
                             "num_inference_steps": 10, "refine_average": False})
    runs = pd.DataFrame([
        {"run_id": f"run-{shard}", "status": "completed",
         "driver": "expanded_pro_16suite",
         "config_json": {"cohort": "expanded", "shard_count": 6,
                         "shard_index": shard, "libero_pro_revision": LIBERO_PRO_REVISION,
                         "pnp_k": PRO_EXPANDED_K,
                         "pnp_steps": list(PRO_EXPANDED_STEPS)}}
        for shard in range(6)
    ])
    return pd.DataFrame(rows), runs


def test_validate_complete_expanded_pro_fixture():
    rollouts, runs = _expanded_fixture()
    validated, result = validate_pro_expanded(rollouts, runs)
    assert result["status"] == "valid"
    assert result["n_identities"] == 2400
    assert result["n_rollouts"] == 4800
    assert result["n_suites"] == 13
    assert result["control_complete"] is False
    assert validated.config_hash.nunique() == 2


def test_validate_expanded_pro_ignores_pre_final_suites_and_partial_control():
    rollouts, runs = _expanded_fixture()
    template = rollouts.iloc[0].to_dict()
    stale = []
    for i, method in enumerate((Method.UNCERTAINTY, Method.REFINEMENT,
                                Method.EXTRA_STEPS)):
        stale.append({**template, "suite": "libero_10_swap",
                      "rollout_id": f"stale-suite-{i}", "init_state_hash": f"stale-{i}",
                      "method": method, "config_hash": f"stale-{method}",
                      "num_inference_steps": 20 if method == Method.EXTRA_STEPS else 10})
    stale.append({**template, "rollout_id": "partial-control",
                  "method": Method.EXTRA_STEPS, "config_hash": "control-20",
                  "num_inference_steps": 20, "pnp_step_indices": None, "pnp_k": None})
    mixed = pd.concat([rollouts, pd.DataFrame(stale)], ignore_index=True)
    validated, result = validate_pro_expanded(mixed, runs)
    assert len(validated) == 4800
    assert result["n_snapshot_rollouts"] == 4804
    assert result["n_ignored_rollouts"] == 4
    assert not result["control_complete"]
    assert set(validated.suite) == set(expanded_pro_suites())
    assert any("pre-final pilot rows" in warning for warning in result["warnings"])


def test_episode_contraction_uses_observed_telemetry_and_paired_refine_outcome():
    rows, vectors = [], []
    profiles = ([.4, .3, .2, .1], [.1, .2, .3, .4])
    for episode, profile in enumerate(profiles):
        identity = {"suite": "libero_goal_swap", "task_idx": 0,
                    "episode_idx": episode, "init_state_hash": str(episode),
                    "u_mean_episode": .05}
        rows += [
            {**identity, "rollout_id": f"observed-{episode}",
             "method": Method.UNCERTAINTY, "success": False},
            {**identity, "rollout_id": f"refine-{episode}",
             "method": Method.REFINEMENT, "success": episode == 0},
        ]
        for chunk in range(5):
            vectors.append({"rollout_id": f"observed-{episode}", "chunk_idx": chunk,
                            "euler_step": 3, "u_iter": list(profile)})
            # Deliberately opposite intervention telemetry: it must never enter the feature.
            vectors.append({"rollout_id": f"refine-{episode}", "chunk_idx": chunk,
                            "euler_step": 3, "u_iter": list(profile[::-1])})

    result = episode_contraction(pd.DataFrame(rows), pd.DataFrame(vectors))
    full = result[result.window == "full_episode"].set_index("episode_idx")
    assert full.loc[0, "corrected"]
    assert not full.loc[1, "corrected"]
    assert full.loc[0, "contraction_normalized_slope"] > 0
    assert full.loc[1, "contraction_normalized_slope"] < 0
    assert np.isclose(full.loc[0, "u_iter_0"], .4)
    assert np.isclose(full.loc[0, "u_iter_3"], .1)


def test_uncertainty_threshold_tables_are_paired_and_cover_every_identity():
    rows = []
    for episode in range(20):
        identity = {"suite": "libero_goal_swap", "task_idx": episode // 10,
                    "episode_idx": episode % 10, "init_state_hash": str(episode),
                    "u_mean_episode": .01 + episode * .001}
        observed_success = episode % 2 == 0
        refined_success = observed_success if episode < 10 else not observed_success
        rows += [
            {**identity, "method": Method.UNCERTAINTY, "success": observed_success},
            {**identity, "method": Method.REFINEMENT, "success": refined_success},
        ]
    rollouts = pd.DataFrame(rows)
    sweep = uncertainty_window_sweep(rollouts, grid_size=5)
    assert len(sweep) == 15
    assert sweep.n_refined.max() == 20
    assert set(sweep.analysis_type) == {"exploratory_in_sample"}
    effect = refinement_effect_by_uncertainty_bin(rollouts, n_bins=5)
    assert len(effect) == 5
    assert effect.n.sum() == 20
    assert {"F_to_S", "S_to_F", "delta_pp"}.issubset(effect.columns)


def test_expanded_report_renders_threshold_auc_and_contraction_figures(tmp_path):
    labels = ["observed/no-op (steps 3,4, K=5)", "refine-last (3,4)"]
    suite = "libero_goal_swap"
    success = pd.DataFrame([
        {"suite": suite, "condition_label": label, "n": 20, "sr": .2 + i * .05,
         "ci_low": .1, "ci_high": .4} for i, label in enumerate(labels)
    ])
    paired = pd.DataFrame([{"suite": suite, "condition_label": labels[1],
                            "F_to_F": 12, "F_to_S": 4, "S_to_F": 2, "S_to_S": 2}])
    episodes = pd.DataFrame([
        {"window": "full_episode", "observed_failure": True, "corrected": i >= 4,
         **{f"u_iter_{j}": .4 - .05 * j + .01 * i for j in range(4)}}
        for i in range(8)
    ])
    quartiles = pd.DataFrame([
        {"window": window, "contraction_quartile": q, "correction_rate": .1 * q,
         "ci_low": .05 * q, "ci_high": .15 * q}
        for window in ("first_4_chunks", "full_episode") for q in range(1, 5)
    ])
    effect = pd.DataFrame([
        {"uncertainty_bin": i, "delta_pp": i - 5,
         "delta_ci_low_pp": i - 7, "delta_ci_high_pp": i - 3}
        for i in range(1, 11)
    ])
    sweep = pd.DataFrame([
        {"lower_grid_index": low, "upper_grid_index": high,
         "delta_pp": high - low, "n_refined": 5 * (high - low + 1)}
        for low in range(3) for high in range(low, 3)
    ])
    tables = {
        "pro_success_by_suite": success, "pro_paired_by_suite": paired,
        "expanded_contraction_episode": episodes,
        "expanded_contraction_quartiles": quartiles,
        "pro_detector_by_suite": pd.DataFrame([{
            "suite": suite, "roc_auc": .7, "roc_ci_low": .6, "roc_ci_high": .8}]),
        "pro_detector_summary": pd.DataFrame([{
            "estimate_scope": "pooled", "roc_auc": .72,
            "roc_ci_low": .65, "roc_ci_high": .79}]),
        "expanded_refinement_effect_by_uncertainty_bin": effect,
        "expanded_uncertainty_window_sweep": sweep,
    }
    expanded_pro_figures(tables, tmp_path)
    expected = {
        "expanded_pro_success_by_suite", "expanded_pro_paired_transitions",
        "expanded_contraction_profiles", "expanded_contraction_quartiles",
        "expanded_uncertainty_failure_auc",
        "expanded_refinement_effect_by_uncertainty",
        "expanded_uncertainty_window_sweep",
    }
    assert expected == {path.stem for path in (tmp_path / "figures").glob("*.png")}
