import pytest
import torch

from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_scalar_returns import ScalarCritic
from pnp.smolvla_anchored_critic import AnchoredCritic


def fixture(family):
    torch.manual_seed(123)
    base = ScalarCritic(prefix_dim=8, robot_dim=4, proprio_dim=3,
                        config=QPlanningModelConfig(action_horizon=10, width=16, n_heads=4, n_layers=1,
                                                    ffn_width=32, dropout=0, prefix_pool_tokens=4))
    model = AnchoredCritic(base, family, width=16)
    prefix = torch.randn(2, 4, 8)
    pad = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    robot, proprio = torch.randn(2, 4), torch.randn(2, 3)
    reference = torch.randn(2, 10, 7)
    valid = torch.arange(10)[None] < torch.tensor([10, 5])[:, None]
    return model, (prefix, pad, robot, proprio), reference, valid


@pytest.mark.parametrize("family", ["transformer", "temporal_cnn"])
def test_initial_baseline_and_parameter_cancellation(family):
    model, context, ref, valid = fixture(family)
    action = ref + torch.randn_like(ref) * .02
    output = model(*context, action, valid, ref)
    torch.testing.assert_close(output, model.base(*context, ref, valid), rtol=0, atol=0)
    output.sum().backward()
    assert model.readout[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    model.zero_grad()
    # Even after training, stock residual cancels and has no parameter gradient.
    with torch.no_grad():
        model.readout[-1].weight.normal_()
    stock = model(*context, ref, valid, ref)
    torch.testing.assert_close(stock, model.base(*context, ref, valid), rtol=0, atol=1e-6)
    stock.sum().backward()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            torch.testing.assert_close(p.grad, torch.zeros_like(p.grad), atol=1e-5, rtol=0)


@pytest.mark.parametrize("family", ["transformer", "temporal_cnn"])
def test_mask_reference_and_candidate_gradients(family):
    model, context, ref, valid = fixture(family)
    with torch.no_grad():
        model.readout[-1].weight.normal_(std=.02)
    ref.requires_grad_(True)
    action = (ref.detach() + .01).requires_grad_(True)
    output = model(*context, action, valid, ref)
    gradient, = torch.autograd.grad(output.sum(), action)
    assert ref.grad is None
    assert gradient[valid].abs().sum() > 0
    assert torch.count_nonzero(gradient[~valid]) == 0
    changed = action.detach().clone()
    changed[~valid] += 1000
    torch.testing.assert_close(output.detach(), model(*context, changed, valid, ref), rtol=0, atol=0)
    epsilon = 1e-3
    plus, minus = action.detach().clone(), action.detach().clone()
    plus[0, 2, 1] += epsilon
    minus[0, 2, 1] -= epsilon
    numerical = (model(*context, plus, valid, ref)[0, 0] - model(*context, minus, valid, ref)[0, 0]) / (2 * epsilon)
    torch.testing.assert_close(gradient[0, 2, 1], numerical, atol=3e-3, rtol=.03)


def test_baseline_stays_frozen_in_training_mode():
    model, _, _, _ = fixture("temporal_cnn")
    model.train()
    assert model.training and not model.base.training
    assert not any(p.requires_grad for p in model.base.parameters())
