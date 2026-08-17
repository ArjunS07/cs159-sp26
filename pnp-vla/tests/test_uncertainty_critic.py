import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pnp.sampler import _EncodingCache, measure_chunk_uncertainty
from pnp.uncertainty_critic import (
    CANDIDATE_COUNT,
    COLLECTION_EPISODE_INDICES,
    EPISODES_PER_TASK,
    N_ACTION_STEPS,
    PNP_K,
    PROBE_STEPS,
    SHARD_COUNT,
    TARGET_CHUNKS,
    TRAIN_EPISODE_INDICES,
    VALIDATION_EPISODE_INDICES,
    _empty_artifact,
    _stack_artifact,
    logical_config,
)


ROOT = Path(__file__).parents[1]


def _notebook_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell["cell_type"] == "code":
            ast.parse(source)
    return "\n".join(sources)


def test_collection_plan_is_large_standard_libero_and_noninvasive():
    config = logical_config()
    assert EPISODES_PER_TASK == 20
    assert SHARD_COUNT == 4
    assert CANDIDATE_COUNT == 6
    assert TARGET_CHUNKS == (0, 2, 4, 8)
    assert COLLECTION_EPISODE_INDICES == tuple(range(20, 40))
    assert TRAIN_EPISODE_INDICES == tuple(range(20, 36))
    assert VALIDATION_EPISODE_INDICES == tuple(range(36, 40))
    assert set(COLLECTION_EPISODE_INDICES).isdisjoint(range(20))
    assert PROBE_STEPS == (3, 4)
    assert PNP_K == 5
    assert N_ACTION_STEPS == 10
    assert config["candidate_zero_executed"] is True
    assert config["n_action_steps"] == 10


def test_artifact_preserves_deployable_input_and_pnp_ablations():
    empty = _empty_artifact(obs_dim=8)
    required = {
        "candidate_initial_action_chunk",
        "candidate_z_hat",
        "candidate_first_a_hat",
        "candidate_last_a_hat",
        "candidate_u_time",
        "candidate_u_iter_time",
        "obs_enc",
    }
    assert required <= set(empty)
    assert empty["candidate_initial_action_chunk"].shape == (
        0, CANDIDATE_COUNT, 50, 7)

    group = {
        "group_chunk_idx": np.asarray(0, np.int16),
        "group_chunk_pos": np.asarray(0.0, np.float32),
        "candidate_noise_seeds": np.arange(CANDIDATE_COUNT),
        "candidate_initial_action_chunk": np.zeros((CANDIDATE_COUNT, 50, 7)),
        "candidate_z_hat": np.zeros((CANDIDATE_COUNT, 2, 50, 7)),
        "candidate_first_a_hat": np.zeros((CANDIDATE_COUNT, 2, 50, 7)),
        "candidate_last_a_hat": np.zeros((CANDIDATE_COUNT, 2, 50, 7)),
        "candidate_u_time": np.zeros((CANDIDATE_COUNT, 2, 50)),
        "candidate_u_iter_time": np.zeros((CANDIDATE_COUNT, 2, 4, 50)),
        "obs_enc": np.zeros(8),
    }
    artifact = _stack_artifact([group])
    assert artifact["candidate_initial_action_chunk"].shape == (1, 6, 50, 7)
    assert artifact["candidate_z_hat"].shape == (1, 6, 2, 50, 7)
    assert artifact["candidate_first_a_hat"].shape == (1, 6, 2, 50, 7)
    assert artifact["candidate_last_a_hat"].shape == (1, 6, 2, 50, 7)
    assert artifact["candidate_u_time"].shape == (1, 6, 2, 50)


def test_measurement_returns_initial_action_and_intermediate_ablations():
    records = []
    pcp_steps = []
    for step, offset in ((3, 0.0), (4, 10.0)):
        a_hats = np.stack([
            np.full((1, 50, 7), offset + candidate, np.float32)
            for candidate in range(5)
        ])
        records.append({
            "step": step,
            "u_time": np.full(50, offset + 1, np.float32),
            "u_iter_time": np.full((4, 50), offset + 2, np.float32),
            "a_hats": a_hats,
        })
        pcp_steps.append({
            "step_idx": step,
            "s": 0.5,
            "z_hat": np.full((50, 7), offset + 3, np.float32),
        })

    class FakeRecorder:
        def new_episode(self):
            pass

        def current_chunks(self):
            return [{"steps": records}]

    fake_tap = SimpleNamespace(pcp_chunks=[{
        "obs_enc": np.arange(8, dtype=np.float32), "steps": pcp_steps,
    }])
    initial_action = torch.full((1, 50, 7), 42.0)
    model = SimpleNamespace(_pnp=SimpleNamespace(action_dim=7, strategy=None))
    policy = SimpleNamespace(
        model=model,
        predict_action_chunk=lambda _batch, noise=None: initial_action,
    )
    with patch("pnp.pnp.PnPRecorder", return_value=FakeRecorder()), patch(
            "pnp.tap.RolloutTap", return_value=fake_tap):
        action, score, details, features = measure_chunk_uncertainty(
            policy, {}, torch.zeros_like(initial_action), probe_steps=(3, 4),
            num_iterations=5, uncertainty_horizon=20, return_details=True,
            return_features=True)

    assert torch.equal(action, initial_action)
    assert score == details["u20"]
    assert features["z_hat"].shape == (2, 50, 7)
    assert np.all(features["first_a_hat"][0] == 0)
    assert np.all(features["last_a_hat"][0] == 4)
    assert np.all(features["first_a_hat"][1] == 10)
    assert np.all(features["last_a_hat"][1] == 14)



def test_encoding_cache_hashes_nested_camera_tensor_lists():
    cache = _EncodingCache(model_revision="revision")
    images = [torch.zeros((1, 3, 4, 4)), torch.ones((1, 3, 4, 4))]
    image_masks = [torch.ones((1,), dtype=torch.bool),
                   torch.ones((1,), dtype=torch.bool)]
    tokens = torch.tensor([[1, 2, 3]])
    masks = torch.ones_like(tokens, dtype=torch.bool)
    first = cache.key(images, image_masks, tokens, masks)
    assert first == cache.key(images, image_masks, tokens, masks)
    changed = [images[0], images[1].clone()]
    changed[1][0, 0, 0, 0] = 2
    assert first != cache.key(changed, image_masks, tokens, masks)

def test_four_worker_notebooks_are_fixed_shards_with_smoke_limit():
    paths = sorted((ROOT / "notebooks" / "workers").glob(
        "45_uncertainty_critic_candidates_worker_*.ipynb"))
    assert len(paths) == 4
    sources = [_notebook_source(path) for path in paths]
    for shard_index, source in enumerate(sources):
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "SHARD_COUNT = 4" in source
        assert "EPISODE_LIMIT = None" in source
        assert "run_worker(" in source
        assert "ordinary pre-refinement action chunk" in source
        assert "there is no refinement and no LIBERO-PRO data" in source
