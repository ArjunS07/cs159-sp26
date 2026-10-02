import pytest
import torch

from pnp.smolvla_critic_guidance_reproduction import (
    DEFAULT_ARMS,
    REFINED_PARENT_ARMS,
    STOCK_PARENT_ARMS,
    bounded_standardized_move,
    resolve_arms,
)


def test_resolve_default_arms_are_unique_and_include_controls():
    arms = resolve_arms(DEFAULT_ARMS)
    assert len({arm["method"] for arm in arms}) == len(arms)
    assert {arm["mode"] for arm in arms} >= {"none", "ascent", "descent", "random"}
    assert arms[0]["kind"] == "pnp" and arms[0]["radius"] == 0


def test_stock_and_refined_workers_have_disjoint_guided_methods():
    stock = resolve_arms(STOCK_PARENT_ARMS)
    refined = resolve_arms(REFINED_PARENT_ARMS)
    assert stock[0]["kind"] == "stock"
    assert stock[1]["kind"] == "stock_guided" and stock[1]["mode"] == "none"
    assert all(arm["kind"] == "stock_guided" for arm in stock[1:])
    assert refined[0]["kind"] == "pnp" and refined[0]["mode"] == "none"
    assert all(arm["kind"] == "pnp" for arm in refined[1:])
    assert {arm["method"] for arm in stock}.isdisjoint(
        {arm["method"] for arm in refined})


def test_bounded_move_hits_standardized_rms_radius_and_masks_tail():
    anchor = torch.zeros((2, 10, 7))
    gradient = torch.arange(1, anchor.numel() + 1, dtype=torch.float32).reshape_as(anchor)
    std = torch.linspace(0.1, 0.7, 7)
    valid = torch.tensor([
        [True] * 10,
        [True] * 4 + [False] * 6,
    ])
    radius = 0.02
    moved, usable = bounded_standardized_move(
        anchor, anchor, gradient, valid, std, radius)

    assert usable.tolist() == [True, True]
    standardized = (moved - anchor) / std
    denom = valid.sum(1) * 7
    rms = (standardized.square().sum((1, 2)) / denom).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, radius), atol=1e-6)
    assert torch.count_nonzero(moved[1, 4:]) == 0


def test_bounded_move_rejects_zero_direction_without_motion():
    anchor = torch.randn((1, 10, 7))
    valid = torch.ones((1, 10), dtype=torch.bool)
    moved, usable = bounded_standardized_move(
        anchor, anchor, torch.zeros_like(anchor), valid, torch.ones(7), 0.02)
    assert usable.tolist() == [False]
    assert torch.equal(moved, anchor)


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        resolve_arms(["not_an_arm"])
