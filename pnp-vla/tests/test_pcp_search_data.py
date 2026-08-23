import numpy as np

from pnp.pcp_search.data import build_training_artifact, validate_training_artifact


def _prefix(value):
    return {
        "processed_images": [np.full((3, 2, 2), value, np.float32)],
        "processed_image_masks": [np.ones((1,), bool)],
        "token_ids": np.asarray([1, 2], np.int64),
        "token_masks": np.asarray([True, True]),
        "prefix_embeddings": np.full((3, 4), value, np.float16),
        "prefix_pad_masks": np.ones((3,), bool),
        "prefix_attention_masks": np.zeros((3,), bool),
        "prefix_attention_2d_masks": np.zeros((3, 3), bool),
        "prefix_position_ids": np.arange(3),
    }


def test_training_artifact_materializes_partial_terminal_transition_and_t_plus_one_state():
    decisions = [{
        "step": step,
        "raw_agentview": np.zeros((2, 2, 3), np.uint8),
        "raw_wrist": np.zeros((2, 2, 3), np.uint8),
        "raw_robot_state": np.zeros(9, np.float32),
        "policy_proprio": np.zeros(8, np.float32),
        "sim_state": np.zeros(6, np.float64),
        "instruction": "do task",
    } for step in (0, 10, 13)]
    artifact = build_training_artifact(
        decisions=decisions,
        prefixes=[_prefix(0), _prefix(1), _prefix(2)],
        generated_chunks=np.zeros((3, 50, 7), np.float32),
        normalized_actions=np.zeros((13, 7), np.float32),
        env_actions=np.zeros((13, 7), np.float32),
        rewards=np.arange(13, dtype=np.float32),
        terminated=np.asarray([False] * 12 + [True]),
        truncated=np.zeros(13, bool),
        step_success=np.asarray([False] * 12 + [True]),
        robot_states=np.zeros((14, 9), np.float32),
        sim_states=np.zeros((14, 6), np.float64),
        chunk_start_steps=[0, 10], chunk_noise_seeds=[11, 12, 13],
        episode_seed=1, perturb_seed=2, initial_state=np.zeros(6),
    )
    validation = validate_training_artifact(artifact)
    assert validation == {
        "training_ready": True, "n_transitions": 2, "n_steps": 13, "n_boundaries": 3}
    assert artifact["bellman/validity_mask"].sum(axis=1).tolist() == [10, 3]
    assert artifact["bellman/terminated"][1, 2]
    assert artifact["prefix/prefix_embeddings"].shape == (3, 3, 4)
    assert len(artifact["robot_state_t_plus_1"]) == 14
