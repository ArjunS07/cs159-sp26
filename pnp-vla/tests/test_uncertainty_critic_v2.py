import io
from types import SimpleNamespace

import numpy as np
import torch

from pnp.uncertainty_critic_v2 import (
    ResidualTrainConfig, StepAlignedGroupData, StepAlignedResidualCritic,
    _decode_step_aligned_row, direct_uncertainty_gradient_test,
    evaluate_residual_critic, residual_gradient_diagnostic,
    train_residual_critic, validate_step_aligned_groups,
)


def _groups(n=8, obs_dim=16):
    rng = np.random.default_rng(8)
    actions = rng.normal(size=(n, 6, 20, 7)).astype(np.float32)
    target = np.maximum(
        .015 + .002 * actions[..., 0].mean(-1), .001).astype(np.float32)
    return StepAlignedGroupData(
        obs=rng.normal(size=(n, obs_dim)).astype(np.float32),
        initial_actions=(actions * 1.05).astype(np.float32),
        z_hat_actions=actions,
        targets_u20=target,
        chunk_position=np.linspace(0, .7, n, dtype=np.float32),
        probe_step=np.asarray([3, 4] * (n // 2), dtype=np.int16),
        probe_s=np.asarray([.7, .6] * (n // 2), dtype=np.float32),
        episode_idx=np.asarray([20, 20, 21, 21, 36, 36, 37, 37], dtype=np.int16),
        task_idx=np.zeros(n, dtype=np.int16),
        suite=np.full(n, "libero_goal", dtype="U40"),
        rollout_id=np.asarray([f"r{i // 2}" for i in range(n)], dtype="U32"),
        chunk_idx=np.zeros(n, dtype=np.int16),
    )


def test_step_decoder_aligns_each_zhat_with_its_own_u20():
    groups, candidates, steps = 2, 6, 2
    z_hat = np.zeros((groups, candidates, steps, 50, 7), np.float32)
    z_hat[:, :, 0] = 3
    z_hat[:, :, 1] = 4
    u_time = np.zeros((groups, candidates, steps, 50), np.float32)
    u_time[:, :, 0] = 30
    u_time[:, :, 1] = 40
    artifact = {
        "group_chunk_idx": np.asarray([0, 2], np.int16),
        "group_chunk_pos": np.asarray([0., .2], np.float32),
        "candidate_initial_action_chunk": np.zeros((groups, candidates, 50, 7), np.float32),
        "candidate_z_hat": z_hat, "candidate_u_time": u_time,
        "obs_enc": np.zeros((groups, 16), np.float32),
        "step_indices": np.asarray([3, 4], np.int16),
    }
    buffer = io.BytesIO(); np.savez_compressed(buffer, **artifact)

    class Store:
        def _download(self, _):
            return buffer.getvalue()

    decoded = _decode_step_aligned_row(Store(), {
        "rollout_id": "r", "generated_chunks_path": "p", "episode_idx": 20,
        "task_idx": 0, "suite": "libero_goal"})
    assert decoded["probe_step"].tolist() == [3, 4, 3, 4]
    assert np.all(decoded["z_hat_actions"][0] == 3)
    assert np.all(decoded["z_hat_actions"][1] == 4)
    assert np.all(decoded["targets_u20"][0] == 30)
    assert np.all(decoded["targets_u20"][1] == 40)


def test_residual_critic_is_equivariant_and_action_differentiable():
    groups = validate_step_aligned_groups(_groups())
    model = StepAlignedResidualCritic(obs_dim=16, dropout=0)
    model.set_statistics(
        groups.z_hat_actions.reshape(-1, 7).mean(0),
        groups.z_hat_actions.reshape(-1, 7).std(0), groups.targets_u20)
    obs = torch.from_numpy(groups.obs[:2])
    actions = torch.from_numpy(groups.z_hat_actions[:2]).requires_grad_(True)
    position = torch.from_numpy(groups.chunk_position[:2])
    probe_s = torch.from_numpy(groups.probe_s[:2])
    score = model.action_score(obs, actions, position, probe_s)
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    permuted = model.action_score(obs, actions[:, permutation], position, probe_s)
    assert torch.allclose(permuted, score[:, permutation], atol=1e-6)
    gradient = torch.autograd.grad(score.mean(), actions)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0


def test_residual_training_smoke_and_metrics():
    groups = _groups()
    train, validation = groups.subset(groups.train_mask), groups.subset(groups.validation_mask)
    config = ResidualTrainConfig(
        epochs=2, patience=2, batch_groups=2, dropout=0)
    model, history, metadata = train_residual_critic(
        train, validation, "cpu", config=config, progress=False)
    metrics = evaluate_residual_critic(
        model, validation, "cpu", "z_hat_obs", config)
    diagnostic = residual_gradient_diagnostic(
        model, validation, "cpu", "z_hat_obs")
    assert len(history) == 2
    assert metadata["best_epoch"] in (0, 1)
    assert 0 <= metrics["within_group_ranking_accuracy"] <= 1
    assert diagnostic["all_finite"] and diagnostic["nonzero_fraction"] > 0


def test_direct_u20_gradient_uses_live_latent_and_common_randomness():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.15))
            self._pnp = SimpleNamespace(strategy=None, action_dim=7)

    class FakePolicy:
        def __init__(self):
            self.model = FakeModel()

        def predict_action_chunk(self, batch, noise=None):
            tap = self.model._pnp.strategy
            ctx = SimpleNamespace(step=3, num_steps=10)

            def vf(value):
                return self.model.weight * value.square()

            tap.step(noise, .7, vf, ctx)
            tap.finish(ctx)
            return noise

    policy = FakePolicy()
    noise = torch.linspace(-1, 1, 50 * 7).reshape(1, 50, 7)
    records = direct_uncertainty_gradient_test(
        policy, {}, noise, epsilons=(1e-4,), checkpoint_vfield=False
        ) if False else direct_uncertainty_gradient_test(
            policy, {}, noise, epsilons=(1e-4,))
    assert len(records) == 1
    assert records[0]["gradient_finite"]
    # At an infinitesimal common-randomness step, descent must beat ascent.
    assert records[0]["descent_u20"] <= records[0]["ascent_u20"]
