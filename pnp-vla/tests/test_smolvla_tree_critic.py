import numpy as np
import pytest
import torch

from pnp.smolvla_tree_critic import (
    TREE_Q10_KINDS,
    _candidate_order,
    _selection_batch_indices,
    _split_group_ids,
    discounted_fork_return,
    listwise_success_selection_loss,
    pairwise_success_ranking_loss,
    stock_relative_difference_loss,
)


def test_discounted_fork_return_uses_remaining_environment_actions():
    assert discounted_fork_return(False, n_steps=20, root_step=10) == 0.0
    assert discounted_fork_return(True, n_steps=11, root_step=10) == 1.0
    assert discounted_fork_return(True, n_steps=20, root_step=10) == pytest.approx(.99 ** 9)


def test_stock_relative_loss_uses_slot_zero_as_reference():
    target = torch.zeros(1, 9)
    target[0, 1] = .6
    perfect = target.clone()
    collapsed = torch.full_like(target, .3)
    assert stock_relative_difference_loss(perfect, target).item() == pytest.approx(0.0)
    assert stock_relative_difference_loss(collapsed, target).item() > 0


def test_candidate_order_places_exact_stock_first():
    rows = [{"candidate_kind": kind} for kind in reversed(TREE_Q10_KINDS)]
    assert [row["candidate_kind"] for row in _candidate_order(rows)] == list(TREE_Q10_KINDS)


def test_group_split_is_deterministic_and_disjoint():
    entries = [
        {"candidate_group_id": f"g{i}", "suite": f"suite{i % 2}",
         "stock_success": bool(i % 3)}
        for i in range(30)
    ]
    train_a, validation_a = _split_group_ids(entries)
    train_b, validation_b = _split_group_ids(list(reversed(entries)))
    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == {entry["candidate_group_id"] for entry in entries}


def test_direct_selection_losses_reward_success_over_failure():
    success = torch.tensor([[True, False, True, False]])
    correct = torch.tensor([[2.0, -1.0, 1.0, -2.0]], requires_grad=True)
    reversed_scores = -correct.detach()
    assert pairwise_success_ranking_loss(correct, success) < pairwise_success_ranking_loss(
        reversed_scores, success)
    assert listwise_success_selection_loss(correct, success) < listwise_success_selection_loss(
        reversed_scores, success)
    (pairwise_success_ranking_loss(correct, success)
     + listwise_success_selection_loss(correct, success)).backward()
    assert correct.grad is not None


def test_selection_sampler_has_exact_65_percent_mixed_slots_per_five_updates():
    mixed = np.arange(20)
    nonmixed = np.arange(20, 40)
    batches = [
        _selection_batch_indices(mixed=mixed, nonmixed=nonmixed, update=step, seed=42)
        for step in range(1, 6)
    ]
    assert sum(np.isin(batch, mixed).sum() for batch in batches) == 26
    assert all(len(set(batch.tolist())) == 8 for batch in batches)
