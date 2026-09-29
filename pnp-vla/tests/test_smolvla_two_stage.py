"""Data-split and ranking invariants for the local two-stage SmolVLA run."""
import torch

from pnp.smolvla_two_stage import pairwise_root_loss, trajectory_spec


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
