import pandas as pd
import pytest

from analysis.pro import base_suite, cross_validated_policy, standard_threshold_transfer
from analysis.validate import ValidationError, validate_pro
from pnp.config import Method
from pnp.libero_pro import CANONICAL_PRO_SUITES, LIBERO_PRO_REVISION


def _pro_fixture():
    rows = []
    configs = [
        (Method.UNCERTAINTY, "observed", list(range(1, 10)), None),
        (Method.EXTRA_STEPS, "control", None, 16),
        (Method.REFINEMENT, "refine", [4, 5], None),
    ]
    for suite in CANONICAL_PRO_SUITES:
        family = "position_perturb" if "_temp_" in suite else "distractor"
        for task in range(10):
            for episode in range(10):
                identity = f"{suite}-{task}-{episode}"
                for method, config_hash, steps, inference in configs:
                    rows.append({"experiment": "pro", "rollout_id": f"{identity}-{config_hash}",
                                 "run_id": f"run-{episode % 6}", "benchmark": "libero_pro",
                                 "suite": suite, "task_idx": task, "episode_idx": episode,
                                 "init_state_hash": identity, "config_hash": config_hash,
                                 "method": method, "status": "completed", "success": episode % 2 == 0,
                                 "suite_family": family, "canonical_member": True,
                                 "expanded_member": True, "pnp_step_indices": steps,
                                 "num_inference_steps": inference, "episode_seed": episode,
                                 "perturb_seed": episode})
    runs = pd.DataFrame([
        {"run_id": f"run-{shard}", "status": "completed", "driver": "canonical_pro_core",
         "config_json": {"cohort": "canonical", "n_configs": 3, "shard_count": 6,
                         "shard_index": shard, "libero_pro_revision": LIBERO_PRO_REVISION}}
        for shard in range(6)
    ])
    return pd.DataFrame(rows), runs


def test_validate_complete_pro_fixture():
    rollouts, runs = _pro_fixture()
    validated, result = validate_pro(rollouts, runs)
    assert result["status"] == "valid"
    assert result["n_rollouts"] == 1800
    assert result["n_identities"] == 600
    assert result["expanded_complete"] is False
    assert validated.config_hash.nunique() == 3


def test_validate_pro_rejects_duplicate_identity_config():
    rollouts, runs = _pro_fixture()
    rollouts = pd.concat([rollouts, rollouts.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError, match="duplicate"):
        validate_pro(rollouts, runs)


def test_base_suite_mapping_is_explicit():
    assert base_suite("libero_object_temp_x0.2") == "libero_object"
    assert base_suite("libero_goal_with_yellow_book") == "libero_goal"
    assert base_suite("unknown") is None


def test_cross_validated_policy_selects_on_training_folds_only():
    rows = []
    for episode in range(100):
        common = {"suite": "libero_goal_with_yellow_book", "task_idx": episode // 10,
                  "episode_idx": episode % 10, "init_state_hash": str(episode)}
        success = episode % 3 != 0
        score = episode / 1000
        rows.append({**common, "method": Method.UNCERTAINTY, "success": success,
                     "u_mean_episode": score, **{f"u_mean_d{i}": score for i in range(7)}})
        rows.append({**common, "method": Method.REFINEMENT, "success": not success,
                     "u_mean_episode": score, **{f"u_mean_d{i}": score for i in range(7)}})
    result = cross_validated_policy(pd.DataFrame(rows))
    assert set(result.analysis_type) == {"cross_validated_held_out"}
    assert set(result.fold) == set(range(5))
    assert result.n.sum() == 100
    assert result[["lower_train_only", "upper_train_only"]].notna().all().all()


def test_standard_threshold_transfer_selects_only_on_standard_labels():
    standard = pd.DataFrame({
        "method": [Method.UNCERTAINTY] * 6,
        "success": [True, True, True, False, False, False],
        "u_mean_episode": [.01, .02, .03, .07, .08, .09],
    })
    pro = pd.DataFrame({
        "method": [Method.UNCERTAINTY] * 4,
        "suite": ["libero_object_temp_x0.1"] * 4,
        "success": [False, True, False, True],
        "u_mean_episode": [.01, .02, .08, .09],
    })
    first = standard_threshold_transfer(pro, standard)
    second = standard_threshold_transfer(pro.assign(success=~pro.success), standard)
    assert first.threshold_standard_only.nunique() == 1
    assert first.threshold_standard_only.iloc[0] == second.threshold_standard_only.iloc[0]
    assert set(first.selection_dataset) == {"standard_libero"}
    assert set(first.evaluation_dataset) == {"libero_pro"}
