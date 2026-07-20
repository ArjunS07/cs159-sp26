import torch

from pnp.verifier.model import CleanChunkVerifier


def test_candidate_scoring_shape_and_permutation_equivariance():
    torch.manual_seed(0)
    model = CleanChunkVerifier(obs_dim=16, dropout=0).eval()
    obs = torch.randn(2, 16)
    position = torch.tensor([0.0, 0.5])
    actions = torch.randn(2, 4, 50, 7)
    mask = torch.ones(2, 4, 50, dtype=torch.bool)
    context = model.encode_context(obs, position)
    score = model.score_candidates(context, actions, mask, 10)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model.score_candidates(context, actions[:, permutation], mask[:, permutation], 10)
    assert score.shape == (2, 4)
    assert torch.allclose(permuted, score[:, permutation], atol=1e-6)


def test_masked_padding_does_not_change_score():
    torch.manual_seed(1)
    model = CleanChunkVerifier(obs_dim=16, dropout=0).eval()
    obs, position = torch.randn(2, 16), torch.zeros(2)
    actions = torch.randn(2, 50, 7)
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[:, :10] = True
    changed = actions.clone()
    changed[:, 10:] = torch.randn_like(changed[:, 10:]) * 100
    a = model(obs, actions, mask, position, 10).joint_logit
    b = model(obs, changed, mask, position, 10).joint_logit
    assert torch.allclose(a, b, atol=1e-6)


def test_state_head_is_action_independent():
    model = CleanChunkVerifier(obs_dim=16, dropout=0).eval()
    obs, position = torch.randn(2, 16), torch.zeros(2)
    mask = torch.ones(2, 50, dtype=torch.bool)
    a = model(obs, torch.randn(2, 50, 7), mask, position).state_logit
    b = model(obs, torch.randn(2, 50, 7), mask, position).state_logit
    assert torch.allclose(a, b)
