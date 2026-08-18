import ast
import io
import json
from pathlib import Path

import numpy as np
import torch

from pnp.uncertainty_critic import CANDIDATE_COUNT
from pnp.uncertainty_critic_train import (
    CandidateGroupData, CriticTrainConfig, UncertaintyGradientCritic,
    _decode_row, dataset_audit, evaluate_critic, gradient_diagnostic,
    load_checkpoint, save_checkpoint, validate_candidate_groups,
)


ROOT = Path(__file__).parents[1]


def _groups(n=6, obs_dim=16):
    rng = np.random.default_rng(4)
    initial = rng.normal(size=(n, CANDIDATE_COUNT, 20, 7)).astype(np.float32)
    target = (.02 + .003 * initial[..., 0].mean(-1)).astype(np.float32)
    target = np.maximum(target, .001)
    return CandidateGroupData(
        obs=rng.normal(size=(n, obs_dim)).astype(np.float32),
        initial_actions=initial,
        z_hat_actions=(initial * .9).astype(np.float32),
        targets_u10=(target * 1.1).astype(np.float32),
        targets_u20=target,
        targets_u50=(target * .8).astype(np.float32),
        chunk_position=np.linspace(0, 1, n, dtype=np.float32),
        episode_idx=np.asarray([20, 21, 22, 36, 37, 38][:n], dtype=np.int16),
        task_idx=np.zeros(n, dtype=np.int16),
        suite=np.full(n, "libero_goal", dtype="U40"),
        rollout_id=np.asarray([f"r{i}" for i in range(n)], dtype="U32"),
        chunk_idx=np.zeros(n, dtype=np.int16),
    )


def test_artifact_decoder_uses_initial_prefix_late_zhat_and_mean_u20():
    groups, candidates, steps = 2, CANDIDATE_COUNT, 2
    u_time = np.arange(groups * candidates * steps * 50, dtype=np.float32).reshape(
        groups, candidates, steps, 50)
    artifact = {
        "group_chunk_idx": np.asarray([0, 2], dtype=np.int16),
        "group_chunk_pos": np.asarray([0., .2], dtype=np.float32),
        "candidate_initial_action_chunk": np.zeros((groups, candidates, 50, 7), np.float32),
        "candidate_z_hat": np.stack([
            np.zeros((groups, candidates, 50, 7), np.float32),
            np.ones((groups, candidates, 50, 7), np.float32),
        ], axis=2),
        "candidate_u_time": u_time,
        "obs_enc": np.zeros((groups, 16), np.float32),
        "step_indices": np.asarray([3, 4], dtype=np.int16),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **artifact)

    class Store:
        def _download(self, _path):
            return buffer.getvalue()

    decoded = _decode_row(Store(), {
        "rollout_id": "r", "generated_chunks_path": "x",
        "episode_idx": 20, "task_idx": 1, "suite": "libero_goal",
    })
    assert decoded["initial_actions"].shape == (2, 6, 20, 7)
    assert np.all(decoded["z_hat_actions"] == 1)
    np.testing.assert_allclose(
        decoded["targets_u20"], u_time[..., :20].mean(axis=(-2, -1)))


def test_group_validation_split_and_audit():
    groups = validate_candidate_groups(_groups())
    assert groups.train_mask.sum() == 3
    assert groups.validation_mask.sum() == 3
    audit = dataset_audit(groups)
    assert audit["observation_groups"] == 6
    assert audit["candidate_examples"] == 36
    assert audit["suites"] == 1


def test_critic_is_candidate_equivariant_and_has_action_gradient():
    torch.manual_seed(1)
    groups = _groups()
    model = UncertaintyGradientCritic(obs_dim=16, dropout=0)
    model.set_statistics(
        groups.initial_actions.reshape(-1, 7).mean(0),
        groups.initial_actions.reshape(-1, 7).std(0),
        groups.targets_u20.reshape(-1))
    obs = torch.from_numpy(groups.obs[:2])
    actions = torch.from_numpy(groups.initial_actions[:2]).requires_grad_(True)
    position = torch.from_numpy(groups.chunk_position[:2])
    score = model(obs, actions, position)
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    permuted = model(obs, actions[:, permutation], position)
    assert score.shape == (2, 6)
    assert torch.allclose(permuted, score[:, permutation], atol=1e-6)
    gradient = torch.autograd.grad(score.mean(), actions)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


def test_checkpoint_round_trip_and_metrics(tmp_path):
    groups = _groups()
    model = UncertaintyGradientCritic(obs_dim=16, dropout=0)
    model.set_statistics(
        groups.initial_actions.reshape(-1, 7).mean(0),
        groups.initial_actions.reshape(-1, 7).std(0),
        groups.targets_u20.reshape(-1))
    config = CriticTrainConfig(epochs=1, dropout=0)
    metrics = evaluate_critic(model, groups, "cpu", "initial_obs", config)
    diagnostic = gradient_diagnostic(model, groups, "cpu", "initial_obs")
    assert metrics["groups"] == len(groups)
    assert diagnostic["all_finite"]
    path = tmp_path / "critic.pt"
    save_checkpoint(path, model, {}, metrics, groups, "initial_obs", config)
    restored, payload = load_checkpoint(path)
    assert payload["representation"] == "initial_obs"
    before = model(
        torch.from_numpy(groups.obs[:1]),
        torch.from_numpy(groups.initial_actions[:1]),
        torch.from_numpy(groups.chunk_position[:1]))
    after = restored(
        torch.from_numpy(groups.obs[:1]),
        torch.from_numpy(groups.initial_actions[:1]),
        torch.from_numpy(groups.chunk_position[:1]))
    assert torch.equal(before, after)


def test_training_notebook_is_colab_ready_and_explicit_about_scope():
    path = ROOT / "notebooks" / "46_train_uncertainty_gradient_critic.ipynb"
    document = json.loads(path.read_text(encoding="utf-8"))
    source = ""
    for cell in document["cells"]:
        text = "".join(cell.get("source", []))
        source += text
        if cell["cell_type"] == "code":
            ast.parse(text)
    assert "fetch_candidate_rows(store, require_complete=True)" in source
    assert "initial_obs" in source
    assert "initial_action_only" in source
    assert "z_hat_obs" in source
    assert "does **not** show that taking the gradient improves success rate" in source
    assert "u20_critic_initial_obs_v1.pt" in source
