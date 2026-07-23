import numpy as np
import torch

from pnp.verifier.data import CleanChunkExample
from pnp.verifier.model import CompactAdvantageVerifier
from pnp.verifier.train import (
    AdvantageTrainConfig, _loader, dataset_hash, summarize_candidate_records,
)


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


def test_weighted_sampler_changes_draws_across_epochs_reproducibly():
    examples = [
        CleanChunkExample(
            rollout_id=f"r{i}", experiment="e", benchmark="libero", suite="s",
            task_idx=0, episode_idx=i, chunk_idx=0, chunk_position=0,
            obs_enc=np.full(4, i, np.float32),
            actions=np.zeros((50, 7), np.float32),
            action_mask=np.ones(50, bool), success=i % 2)
        for i in range(20)
    ]
    config = AdvantageTrainConfig(seed=11, batch_rollouts=20)
    draws = [
        next(iter(_loader(examples, config, train=True, seed_offset=epoch)))["rollout_id"]
        for epoch in (0, 1, 0)
    ]
    assert draws[0] != draws[1]
    assert draws[0] == draws[2]


def test_dataset_hash_detects_label_and_artifact_changes():
    example = CleanChunkExample(
        rollout_id="r", experiment="e", benchmark="libero", suite="s",
        task_idx=0, episode_idx=0, chunk_idx=0, chunk_position=0,
        obs_enc=np.zeros(4, np.float32), actions=np.zeros((50, 7), np.float32),
        action_mask=np.ones(50, bool), success=0)
    changed_label = CleanChunkExample(**{**example.__dict__, "success": 1})
    changed_actions = CleanChunkExample(**{
        **example.__dict__, "actions": np.ones((50, 7), np.float32)})
    assert dataset_hash([example]) != dataset_hash([changed_label])
    assert dataset_hash([example]) != dataset_hash([changed_actions])
