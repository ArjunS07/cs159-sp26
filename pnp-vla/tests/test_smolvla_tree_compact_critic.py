import torch

from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.qplanning_critic.model import QPlanningCritic
from pnp.smolvla_tree_compact_critic import (
    CompactTreeTrainConfig,
    nonmixed_score_consistency_loss,
)


def test_nonmixed_consistency_penalizes_unsupported_score_spread():
    success = torch.tensor([
        [True, True, True],
        [False, False, False],
        [True, False, True],
    ])
    equal = torch.tensor([[.3, .3, .3], [.6, .6, .6], [.1, .9, .2]])
    spread = torch.tensor([[.1, .3, .5], [.2, .6, 1.0], [.1, .9, .2]], requires_grad=True)
    assert nonmixed_score_consistency_loss(equal, success).item() == 0.0
    loss = nonmixed_score_consistency_loss(spread, success)
    assert loss.item() > 0
    loss.backward()
    assert spread.grad is not None
    assert torch.all(spread.grad[2] == 0)  # mixed trees are supervised by ranking, not this term


def test_declared_compact_architecture_is_under_four_million_parameters():
    config = CompactTreeTrainConfig()
    model = QPlanningCritic(
        prefix_dim=960, robot_dim=9, proprio_dim=8,
        config=QPlanningModelConfig(
            action_horizon=10, action_dim=7, width=config.width,
            n_layers=config.n_layers, n_heads=config.n_heads,
            ffn_width=config.ffn_width, dropout=config.dropout))
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 3_000_000 < count < 4_000_000
