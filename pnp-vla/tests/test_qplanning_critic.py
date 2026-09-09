import io
import json

import numpy as np
import torch

from pnp.pcp_critic.data import DatasetSnapshot
from pnp.pcp_search.data import build_training_artifact
from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.qplanning_critic.data import (
    QPLANNING_ARTIFACT_FIELDS, QPlanningWindowDataset, collate_windows,
    prepare_qplanning_cache, prepare_qplanning_streaming_cache,
    qplanning_windows_from_artifact)
from pnp.qplanning_critic.model import QPlanningCritic
from pnp.store import TRAINING_DATA_MULTIPART_FORMAT


def _prefix(value):
    return {
        "processed_images": [np.zeros((3, 2, 2), np.float32)],
        "processed_image_masks": [np.ones((1,), bool)],
        "token_ids": np.asarray([1, 2], np.int64),
        "token_masks": np.asarray([True, True]),
        "prefix_embeddings": np.full((5, 8), value, np.float16),
        "prefix_pad_masks": np.ones(5, bool),
        "prefix_attention_masks": np.zeros(5, bool),
        "prefix_attention_2d_masks": np.zeros((5, 5), bool),
        "prefix_position_ids": np.arange(5),
    }


def _artifact(n_steps=120):
    boundaries = list(range(0, n_steps + 1, 10))
    decisions = [{
        "step": step, "raw_agentview": np.zeros((2, 2, 3), np.uint8),
        "raw_wrist": np.zeros((2, 2, 3), np.uint8),
        "raw_robot_state": np.full(3, index, np.float32),
        "policy_proprio": np.full(2, index, np.float32),
        "sim_state": np.zeros(4), "instruction": "task",
    } for index, step in enumerate(boundaries)]
    actions = np.arange(n_steps * 7, dtype=np.float32).reshape(n_steps, 7) / 1000
    generated = np.zeros((len(boundaries), 50, 7), np.float32)
    for index, start in enumerate(boundaries[:-1]):
        generated[index, :min(10, n_steps - start)] = actions[start:start + 10]
    rewards = np.zeros(n_steps, np.float32); rewards[-1] = 1
    terminated = np.zeros(n_steps, bool); terminated[-1] = True
    success = terminated.copy()
    return build_training_artifact(
        decisions=decisions, prefixes=[_prefix(i) for i in range(len(boundaries))],
        generated_chunks=generated, normalized_actions=actions, env_actions=actions,
        rewards=rewards, terminated=terminated, truncated=np.zeros(n_steps, bool),
        step_success=success, robot_states=np.zeros((n_steps + 1, 3), np.float32),
        sim_states=np.zeros((n_steps + 1, 4)), chunk_start_steps=boundaries[:-1],
        chunk_noise_seeds=list(range(len(boundaries))), episode_seed=1,
        perturb_seed=2, initial_state=np.zeros(4))


def test_q50_uses_executed_trajectory_across_five_replans():
    arrays = _artifact()
    packed = qplanning_windows_from_artifact(
        {"rollout_id": "r"}, arrays, horizon=50, gamma=.99)
    np.testing.assert_allclose(packed["action"][0], arrays["actions_normalized"][:50])
    np.testing.assert_allclose(packed["next_action"][0], arrays["actions_normalized"][50:100])
    assert packed["discount"][0] == np.float32(.99 ** 50)
    assert packed["next_robot"][0, 0] == 5
    assert packed["action_valid"][-1].sum() == 10
    assert packed["discount"][-1] == 0


def test_q10_bootstraps_at_the_next_planning_boundary():
    arrays = _artifact()
    packed = qplanning_windows_from_artifact(
        {"rollout_id": "r"}, arrays, horizon=10, gamma=.99)
    np.testing.assert_allclose(packed["action"][0], arrays["actions_normalized"][:10])
    np.testing.assert_allclose(packed["next_action"][0], arrays["actions_normalized"][10:20])
    assert packed["next_robot"][0, 0] == 1
    assert packed["discount"][0] == np.float32(.99 ** 10)
    assert packed["discount"][-1] == 0


def test_single_q_rl_token_decoder_and_hl_gauss_head():
    arrays = _artifact(20)
    packed = qplanning_windows_from_artifact(
        {"rollout_id": "r"}, arrays, horizon=10, gamma=.99)
    items = [{key: value[i] for key, value in packed.items()} for i in range(len(packed["reward"]))]
    batch = collate_windows(items)
    model = QPlanningCritic(
        prefix_dim=8, robot_dim=3, proprio_dim=2,
        config=QPlanningModelConfig(
            action_horizon=10, width=32, n_layers=1, n_heads=4,
            ffn_width=64, dropout=0, n_bins=11, hl_gauss_sigma=.1))
    logits = model(
        batch["prefix"], batch["pad"], batch["robot"], batch["proprio"],
        batch["action"], batch["action_valid"])
    targets = model.hl_gauss_targets(torch.tensor([0.0, 1.0]))
    assert logits.shape == (2, 11)
    assert model.rl_token.shape == (1, 1, 32)
    assert len(model.value_head.weight) == 11
    assert torch.allclose(targets.sum(-1), torch.ones(2), atol=1e-6)
    assert model.categorical_loss(logits, torch.tensor([0.0, 1.0])).isfinite()


def test_parallel_source_cache_is_reused_across_horizons_and_repeated_runs(
        tmp_path, monkeypatch):
    artifact = _artifact(60)
    rows = [{
        "rollout_id": rollout_id, "training_data_path": f"remote/{rollout_id}",
        "benchmark": "libero", "suite": "libero_goal", "task_idx": index,
        "run_id": "run",
    } for index, rollout_id in enumerate(("r0", "r1"))]
    snapshot = DatasetSnapshot(
        snapshot_id="pcpcds-test", rollout_ids=("r0", "r1"),
        train_rollout_ids=("r0",), val_rollout_ids=("r1",),
        policy_repo_id="pi", policy_revision="revision", artifact_schema_version=1,
        action_mean=tuple(np.zeros(7)), action_std=tuple(np.ones(7)), provenance={})
    downloads = []

    class Store:
        def __init__(self, parent=False):
            self.parent = parent

        def fork_for_thread(self):
            return Store()

    parent_store = Store(parent=True)

    monkeypatch.setattr(
        "pnp.qplanning_critic.data.eligible_rollout_rows",
        lambda _store, rollout_ids: rows)

    def load(_store, path, fields):
        assert not _store.parent
        downloads.append(path)
        assert tuple(fields) == QPLANNING_ARTIFACT_FIELDS
        return {name: np.asarray(artifact[name]).copy() for name in fields}

    monkeypatch.setattr("pnp.qplanning_critic.data.load_training_fields_with_retry", load)
    q50 = prepare_qplanning_cache(
        parent_store, snapshot, horizon=50, gamma=.99, cache_root=tmp_path,
        download_workers=2)
    assert sorted(downloads) == ["remote/r0", "remote/r1"]
    assert len(q50.rollouts) == 2

    repeated_q50 = prepare_qplanning_cache(
        parent_store, snapshot, horizon=50, gamma=.99, cache_root=tmp_path,
        download_workers=2)
    q10 = prepare_qplanning_cache(
        parent_store, snapshot, horizon=10, gamma=.99, cache_root=tmp_path,
        download_workers=2)
    assert downloads == ["remote/r0", "remote/r1"]
    assert repeated_q50.digest == q50.digest
    assert len(q10.rollouts) == 2
    assert (tmp_path / snapshot.snapshot_id / "source_parts_v2" / "index.json").is_file()
    assert (tmp_path / snapshot.snapshot_id / "q50_v2" / "index.json").is_file()
    assert (tmp_path / snapshot.snapshot_id / "q10_v2" / "index.json").is_file()


def test_source_cache_checkpoints_completed_rollouts_after_failure(tmp_path, monkeypatch):
    artifact = _artifact(60)
    rollout_ids = ("r0", "r1", "r2")
    rows = [{
        "rollout_id": rollout_id, "training_data_path": f"remote/{rollout_id}",
        "benchmark": "libero", "suite": "libero_goal", "task_idx": index,
        "run_id": "run",
    } for index, rollout_id in enumerate(rollout_ids)]
    snapshot = DatasetSnapshot(
        snapshot_id="pcpcds-resume", rollout_ids=rollout_ids,
        train_rollout_ids=("r0", "r1"), val_rollout_ids=("r2",),
        policy_repo_id="pi", policy_revision="revision", artifact_schema_version=1,
        action_mean=tuple(np.zeros(7)), action_std=tuple(np.ones(7)), provenance={})
    monkeypatch.setattr(
        "pnp.qplanning_critic.data.eligible_rollout_rows",
        lambda _store, rollout_ids: rows)
    downloads = []
    should_fail = {"r1": True}

    def load(_store, path, fields):
        rollout_id = path.rsplit("/", 1)[-1]
        downloads.append(rollout_id)
        if should_fail.get(rollout_id):
            raise RuntimeError("simulated transport exhaustion")
        return {name: np.asarray(artifact[name]).copy() for name in fields}

    monkeypatch.setattr("pnp.qplanning_critic.data.load_training_fields_with_retry", load)
    try:
        prepare_qplanning_cache(
            object(), snapshot, horizon=50, gamma=.99, cache_root=tmp_path,
            download_workers=1)
    except RuntimeError as error:
        assert "simulated transport exhaustion" in str(error)
    else:
        raise AssertionError("the simulated first pass should fail")

    should_fail["r1"] = False
    prepare_qplanning_cache(
        object(), snapshot, horizon=50, gamma=.99, cache_root=tmp_path,
        download_workers=1)
    assert downloads.count("r0") == 1


def test_streaming_windows_use_only_shared_source_disk_cache(tmp_path, monkeypatch):
    artifact = _artifact(60)
    rows = [{
        "rollout_id": "r0", "training_data_path": "remote/r0",
        "benchmark": "libero", "suite": "libero_goal", "task_idx": 0,
        "run_id": "run",
    }]
    snapshot = DatasetSnapshot(
        snapshot_id="pcpcds-stream", rollout_ids=("r0",),
        train_rollout_ids=("r0",), val_rollout_ids=(),
        policy_repo_id="pi", policy_revision="revision", artifact_schema_version=1,
        action_mean=tuple(np.zeros(7)), action_std=tuple(np.ones(7)), provenance={})
    downloads = []
    monkeypatch.setattr(
        "pnp.qplanning_critic.data.eligible_rollout_rows",
        lambda _store, rollout_ids: rows)

    def load(_store, path, fields):
        downloads.append(path)
        return {name: np.asarray(artifact[name]).copy() for name in fields}

    monkeypatch.setattr("pnp.qplanning_critic.data.load_training_fields_with_retry", load)
    q50 = prepare_qplanning_streaming_cache(
        object(), snapshot, horizon=50, gamma=.97, cache_root=tmp_path,
        download_workers=1)
    dataset = QPlanningWindowDataset(q50, snapshot.train_rollout_ids)
    expected = qplanning_windows_from_artifact(
        rows[0], artifact, horizon=50, gamma=.97)
    np.testing.assert_allclose(dataset[0]["action"], expected["action"][0])
    assert dataset[0]["reward"] == expected["reward"][0]

    q10 = prepare_qplanning_streaming_cache(
        object(), snapshot, horizon=10, gamma=.97, cache_root=tmp_path,
        download_workers=1)
    assert q10.storage_mode == "source_stream"
    assert downloads == ["remote/r0"]
    assert not (tmp_path / snapshot.snapshot_id / "q50_v2").exists()
    assert not (tmp_path / snapshot.snapshot_id / "q10_v2").exists()


def test_multipart_source_cache_mirrors_clean_parts_and_filters_contaminated_parts(
        tmp_path, monkeypatch):
    artifact = _artifact(60)
    rollout_id = "r0"
    manifest_path = "pcp_search/training_data/r0/manifest.json"
    clean_path = "pcp_search/training_data/r0/parts/0000.npz"
    contaminated_path = "pcp_search/training_data/r0/parts/0001.npz"
    clean_buffer = io.BytesIO()
    np.savez_compressed(clean_buffer, **{
        name: np.asarray(artifact[name])
        for name in QPLANNING_ARTIFACT_FIELDS if name != "rewards"})
    contaminated_buffer = io.BytesIO()
    np.savez_compressed(
        contaminated_buffer, rewards=np.asarray(artifact["rewards"]),
        unrelated_camera=np.zeros((20, 20, 3), np.uint8))
    manifest = {
        "format": TRAINING_DATA_MULTIPART_FORMAT,
        "arrays": {
            name: {
                "dtype": np.asarray(artifact[name]).dtype.str,
                "shape": list(np.asarray(artifact[name]).shape),
                "parts": [{
                    "path": contaminated_path if name == "rewards" else clean_path,
                    "start": None, "stop": None,
                }],
            }
            for name in QPLANNING_ARTIFACT_FIELDS
        },
    }
    manifest["arrays"]["unrelated_camera"] = {
        "dtype": np.dtype(np.uint8).str, "shape": [20, 20, 3],
        "parts": [{"path": contaminated_path, "start": None, "stop": None}],
    }
    payloads = {
        manifest_path: json.dumps(manifest).encode(),
        clean_path: clean_buffer.getvalue(),
        contaminated_path: contaminated_buffer.getvalue(),
    }
    downloads = []

    class Store:
        def fork_for_thread(self):
            return Store()

        def _download(self, path):
            downloads.append(path)
            return payloads[path]

    rows = [{
        "rollout_id": rollout_id, "training_data_path": manifest_path,
        "benchmark": "libero", "suite": "libero_goal", "task_idx": 0,
        "run_id": "run",
    }]
    snapshot = DatasetSnapshot(
        snapshot_id="pcpcds-parts", rollout_ids=(rollout_id,),
        train_rollout_ids=(rollout_id,), val_rollout_ids=(),
        policy_repo_id="pi", policy_revision="revision", artifact_schema_version=1,
        action_mean=tuple(np.zeros(7)), action_std=tuple(np.ones(7)), provenance={})
    monkeypatch.setattr(
        "pnp.qplanning_critic.data.eligible_rollout_rows",
        lambda _store, rollout_ids: rows)

    q10 = prepare_qplanning_streaming_cache(
        Store(), snapshot, horizon=10, gamma=.99, cache_root=tmp_path,
        download_workers=1)
    entry = q10.rollouts[0]
    assert entry["storage_format"] == "multipart_mirror"
    assert downloads.count(manifest_path) == 1
    assert downloads.count(clean_path) == 1
    assert downloads.count(contaminated_path) == 1
    local_manifest = (
        tmp_path / snapshot.snapshot_id / "source_parts_v2" / entry["path"])
    assert local_manifest.is_file()
    local = json.loads(local_manifest.read_text())
    mirrored_clean = local_manifest.parent / local["local_parts"][clean_path]
    assert mirrored_clean.read_bytes() == payloads[clean_path]
    mirrored_filtered = local_manifest.parent / local["local_parts"][contaminated_path]
    with np.load(mirrored_filtered, allow_pickle=False) as archive:
        assert archive.files == ["rewards"]

    dataset = QPlanningWindowDataset(q10, snapshot.train_rollout_ids)
    expected = qplanning_windows_from_artifact(
        rows[0], artifact, horizon=10, gamma=.99)
    np.testing.assert_allclose(dataset[0]["action"], expected["action"][0])

    q50 = prepare_qplanning_streaming_cache(
        Store(), snapshot, horizon=50, gamma=.99, cache_root=tmp_path,
        download_workers=1)
    assert q50.rollouts[0]["storage_format"] == "multipart_mirror"
    assert downloads.count(manifest_path) == 1
    assert downloads.count(clean_path) == 1
    assert downloads.count(contaminated_path) == 1
