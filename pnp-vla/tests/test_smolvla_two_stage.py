"""Data-split and ranking invariants for the local two-stage SmolVLA run."""
import hashlib

import numpy as np
import torch

from pnp.smolvla_two_stage import (
    CACHE_FORMAT, LEGACY_CACHE_FORMAT, _materialize,
    pairwise_root_loss, trajectory_spec,
)


def test_trajectory_spec_keeps_source_and_branch_time_distinct():
    group = {
        "candidate_group_id": "group-1", "suite": "libero_10",
        "chunk_idx": 7, "source_training_data_path": "source.json",
        "candidates": [
            {"candidate_kind": "stored_source", "training_data_path": "source.json",
             "success": True},
            {"candidate_kind": "fresh_seed_1", "training_data_path": "branch.npz",
             "success": False},
        ],
    }
    source = trajectory_spec(group, "stored_source")
    branch = trajectory_spec(group, "fresh_seed_1")
    assert (source["path"], source["start"], source["step_offset"]) == (
        "source.json", 0, 0)
    assert (branch["path"], branch["start"], branch["step_offset"]) == (
        "branch.npz", 1, 70)


def test_pairwise_loss_rewards_correct_within_state_ordering():
    labels = torch.tensor([[True, False, False], [False, False, False]])
    correct = torch.tensor([[.8, .2, .3], [.9, .9, .9]], requires_grad=True)
    reversed_scores = torch.tensor([[.2, .8, .7], [.9, .9, .9]])
    loss = pairwise_root_loss(correct, labels)
    assert loss < pairwise_root_loss(reversed_scores, labels)
    loss.backward()
    assert correct.grad[0, 0] < 0
    assert torch.count_nonzero(correct.grad[1]) == 0


def test_legacy_pretrain_cache_drops_outcome_dependent_partial_chunk(tmp_path):
    spec = {"group_id": "g", "kind": "stored_source", "success": True,
            "path": "unused", "start": 0, "step_offset": 0, "suite": "libero_10"}
    legacy = tmp_path / "old"
    destination = tmp_path / "new"
    legacy.mkdir()
    destination.mkdir()
    digest = hashlib.sha256(b"g|stored_source").hexdigest()[:24]
    filename = f"trajectory_{digest}.npz"
    action = np.ones((2, 10, 7), np.float32)
    action[1, 4:] = 0  # old cache used observed termination as padding
    validity = np.ones((2, 10), bool)
    validity[1, 4:] = False
    np.savez_compressed(
        legacy / filename, format=np.asarray(LEGACY_CACHE_FORMAT),
        group_id=np.asarray("g"), kind=np.asarray("stored_source"),
        prefix=np.ones((2, 3, 4), np.float16), pad=np.ones((2, 3), bool),
        robot=np.ones((2, 10), np.float32),
        proprio=np.ones((2, 8), np.float32),
        action=action, action_valid=validity,
        success=np.ones(2, bool),
    )
    result = _materialize(None, spec, destination, legacy)
    assert result["windows"] == 1
    with np.load(destination / filename, allow_pickle=False) as cache:
        assert str(cache["format"]) == CACHE_FORMAT
        assert cache["action_valid"].all()
        np.testing.assert_array_equal(cache["action"], action[:1])
