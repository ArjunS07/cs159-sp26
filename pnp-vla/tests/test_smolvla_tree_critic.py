import numpy as np
import pytest
import torch

from pnp.smolvla_tree_critic import (
    TREE_Q10_KINDS,
    _candidate_order,
    _split_group_ids,
    discounted_fork_return,
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
