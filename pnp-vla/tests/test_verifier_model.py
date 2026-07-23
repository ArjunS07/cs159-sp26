import torch

from pnp.verifier.model import CleanChunkVerifier, CompactAdvantageVerifier
from pnp.verifier.train import summarize_candidate_records


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


def test_compact_ranker_is_permutation_equivariant_and_prefix_only():
    torch.manual_seed(2)
    model = CompactAdvantageVerifier(obs_dim=16, dropout=0).eval()
    context = model.encode_context(torch.randn(2, 16), torch.tensor([0.0, 0.5]))
    actions = torch.randn(2, 4, 50, 7)
    mask = torch.ones(2, 4, 50, dtype=torch.bool)
    score = model.rank_candidates(context, actions, mask, 10)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model.rank_candidates(
        context, actions[:, permutation], mask[:, permutation], 10)
    changed = actions.clone()
    changed[:, :, 10:] = torch.randn_like(changed[:, :, 10:]) * 100
    future_changed = model.rank_candidates(context, changed, mask, 10)
    assert score.shape == (2, 4)
    assert torch.allclose(permuted, score[:, permutation], atol=1e-6)
    assert torch.allclose(future_changed, score, atol=1e-6)


def test_compact_freeze_value_pathway_leaves_only_advantage_trainable():
    model = CompactAdvantageVerifier(obs_dim=16)
    model.freeze_value_pathway()
    assert not any(parameter.requires_grad for module in (
        model.obs_encoder, model.position_encoder, model.state_head
    ) for parameter in module.parameters())
    assert all(parameter.requires_grad for module in (
        model.action_in, model.temporal, model.context_rank,
        model.action_rank, model.advantage_head
    ) for parameter in module.parameters())


def test_candidate_metrics_average_pairs_at_the_group_level():
    records = [
        {"group_id": "a", "benchmark": "libero", "uncertainty_stratum": "high",
         "pair_accuracy": 1.0, "margin": 2.0, "comparisons": 1,
         "top1": 1.0, "default": 0.0, "random": 0.5, "oracle": 1.0},
        {"group_id": "b", "benchmark": "libero", "uncertainty_stratum": "high",
         "pair_accuracy": 0.0, "margin": -1.0, "comparisons": 4,
         "top1": 0.0, "default": 1.0, "random": 0.5, "oracle": 1.0},
    ]
    metrics = summarize_candidate_records(records, n_bootstrap=100)
    assert metrics["group_macro_ranking_accuracy"] == 0.5
    assert metrics["n_pairwise_comparisons"] == 5
    assert metrics["top1_success"] == 0.5
