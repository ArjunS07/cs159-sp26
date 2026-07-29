import numpy as np
import torch

from pnp.verifier.data import CleanChunkExample
from pnp.verifier.model import CompactAdvantageVerifier
from pnp.verifier.critic import HybridChunkCritic
from pnp.verifier.train import (
    AdvantageTrainConfig, _loader, dataset_hash, paired_candidate_comparison,
    summarize_candidate_records, train_advantage, verifier_registration_eligibility,
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


def test_hybrid_critic_is_permutation_equivariant_and_prefix_only():
    torch.manual_seed(3)
    model = HybridChunkCritic(obs_dim=16, width=32, dropout=0).eval()
    context = model.encode_context(torch.randn(2, 16), torch.tensor([0.0, 0.5]))
    actions = torch.randn(2, 4, 50, 7)
    mask = torch.ones(2, 4, 50, dtype=torch.bool)
    score = model.rank_candidates(context, actions, mask, 10)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model.rank_candidates(context, actions[:, permutation],
                                     mask[:, permutation], 10)
    changed = actions.clone()
    changed[:, :, 10:] = torch.randn_like(changed[:, :, 10:]) * 100
    assert score.shape == (2, 4)
    assert torch.allclose(permuted, score[:, permutation], atol=1e-6)
    assert torch.allclose(model.rank_candidates(context, changed, mask, 10),
                          score, atol=1e-6)
    with np.testing.assert_raises_regex(ValueError, "prefix_length=10"):
        model.rank_candidates(context, actions, mask, 5)


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


def test_conditioned_architectures_are_candidate_equivariant_and_state_sensitive():
    torch.manual_seed(7)
    actions = torch.randn(2, 3, 50, 7)
    mask = torch.ones(2, 3, 50, dtype=torch.bool)
    observations = torch.stack([torch.zeros(16), torch.ones(16)])
    for architecture in ("film", "cross_attention"):
        model = CompactAdvantageVerifier(
            obs_dim=16, action_width=32, dropout=0,
            conditioning=architecture).eval()
        context = model.encode_context(observations, torch.zeros(2))
        scores = model.rank_candidates(context, actions, mask, 10)
        permutation = torch.tensor([2, 0, 1])
        permuted = model.rank_candidates(
            context, actions[:, permutation], mask[:, permutation], 10)
        assert torch.allclose(permuted, scores[:, permutation], atol=1e-6)
        assert not torch.allclose(scores[0], scores[1])


def test_action_only_architecture_is_invariant_to_context():
    torch.manual_seed(9)
    model = CompactAdvantageVerifier(
        obs_dim=16, conditioning="action_only", dropout=0).eval()
    actions = torch.randn(1, 4, 50, 7).expand(2, -1, -1, -1)
    mask = torch.ones(2, 4, 50, dtype=torch.bool)
    context = model.encode_context(
        torch.stack([torch.zeros(16), torch.ones(16)]), torch.tensor([0., 1.]))
    scores = model.rank_candidates(context, actions, mask, 10)
    assert torch.allclose(scores[0], scores[1], atol=1e-6)


def test_legacy_checkpoint_can_omit_v2_conditioning_modules():
    model = CompactAdvantageVerifier(obs_dim=16)
    legacy = {key: value for key, value in model.state_dict().items()
              if not key.startswith(("film_", "cross_"))}
    restored = CompactAdvantageVerifier(obs_dim=16)
    result = restored.load_state_dict(legacy)
    assert result.unexpected_keys == []
    assert result.missing_keys


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


def test_rank_training_restores_and_reports_the_best_epoch():
    examples = []
    for group in range(4):
        for candidate, success in enumerate((0, 1)):
            actions = np.zeros((50, 7), np.float32)
            actions[:10, 0] = success * 2 - 1
            examples.append(CleanChunkExample(
                rollout_id=f"g{group}-c{candidate}", experiment="synthetic",
                benchmark="libero", suite="s", task_idx=0, episode_idx=group,
                chunk_idx=0, chunk_position=0, obs_enc=np.full(16, group, np.float32),
                actions=actions, action_mask=np.ones(50, bool), success=success,
                candidate_group_id=f"g{group}",
                candidate_kind="default" if candidate == 0 else "fresh_noise_1"))
    model = CompactAdvantageVerifier(obs_dim=16, dropout=0, conditioning="film")
    config = AdvantageTrainConfig(
        rank_epochs=3, patience=2, rank_lr=1e-3, batch_rollouts=4)
    _, metadata = train_advantage(
        model, examples[:6], examples[6:], torch.device("cpu"), config=config)
    assert metadata["best_rank_epoch"] is not None
    assert 0 <= metadata["best_rank_epoch"] < metadata["rank_epochs_ran"]
    assert len(metadata["rank_history"]) == metadata["rank_epochs_ran"]


def test_paired_registration_gate_requires_all_four_signals():
    selected = [{"group_id": f"g{i}", "pair_accuracy": 1., "top1": 1.}
                for i in range(5)]
    control = [{"group_id": f"g{i}", "pair_accuracy": 0., "top1": 0.}
               for i in range(5)]
    comparison = paired_candidate_comparison(
        selected, control, n_bootstrap=100)
    metrics = {"ranking_accuracy_ci95": [.7, 1.],
               "top1_uplift_default_ci95": [.1, .5]}
    gate = verifier_registration_eligibility(metrics, comparison, comparison)
    assert gate["eligible"]
