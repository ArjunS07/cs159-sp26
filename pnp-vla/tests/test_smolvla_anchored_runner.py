"""Scientific accounting and full-state resumption for the anchored MC runner."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pnp.qplanning_critic.config import QPlanningModelConfig
from pnp.smolvla_scalar_returns import ScalarCritic
from pnp.smolvla_anchored_critic import AnchoredCritic
from scripts.train_smolvla_anchored_local import (
    DIGEST, gradient_diagnostics, root_positions, root_scores,
    restore_checkpoint, save_checkpoint, scoring_metrics,
)


def setup_model(family, device):
    torch.manual_seed(13)
    base = ScalarCritic(prefix_dim=6, robot_dim=3, proprio_dim=4,
                        config=QPlanningModelConfig(action_horizon=10, width=16,
                            n_layers=1, n_heads=2, ffn_width=32, dropout=0))
    base.set_action_statistics(torch.arange(7) * .1, torch.arange(7) * .2 + .5)
    model = AnchoredCritic(base, family, width=16).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    batch = {"prefix": torch.randn(2, 3, 6), "pad": torch.ones(2, 3, dtype=torch.bool),
             "robot": torch.randn(2, 3), "proprio": torch.randn(2, 4),
             "action": torch.randn(2, 9, 10, 7),
             "action_valid": torch.ones(2, 9, 10, dtype=torch.bool),
             "success": torch.tensor([[0,1,1,0,0,0,0,0,0], [1,0,0,1,1,1,1,1,1]], dtype=torch.float32)}
    return model, optimizer, {k: v.to(device) for k, v in batch.items()}


def update(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(root_scores(model, batch), batch["success"]).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
    optimizer.step()


@pytest.mark.parametrize("family", ["transformer", "temporal_cnn"])
def test_anchored_runner_resume_and_stock_invariance(tmp_path, family):
    device = torch.device("cpu")
    model, optimizer, batch = setup_model(family, device)
    initial_stock = root_scores(model.eval(), batch)[:, 0].detach().clone()
    update(model, optimizer, batch)
    config = {"family": family, "snapshot_digest": DIGEST}
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, 1, config, [], "test-run", device)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    restored, resumed_optimizer, _ = setup_model(family, device)
    restore_checkpoint(saved, restored, resumed_optimizer, config, device)
    update(model, optimizer, batch)
    update(restored, resumed_optimizer, batch)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(root_scores(model.eval(), batch)[:, 0], initial_stock, rtol=0, atol=0)
    with pytest.raises(ValueError, match="experiment contract"):
        restore_checkpoint(saved, restored, resumed_optimizer, {**config, "family": "wrong"}, device)


def test_metric_decomposition_counts_and_paired_root_sampling():
    logits = np.array([[0, 2, -2, 1, 0, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0, 0.]])
    labels = np.array([[0,1,0,0,0,0,0,0,0], [1,0,0,1,1,1,1,1,1]], dtype=bool)
    metrics = scoring_metrics(logits, labels)
    assert metrics["rescues"] == 1 and metrics["spoils"] == 1
    assert metrics["selected_successes"] == 1 and metrics["stock_successes"] == 1
    assert metrics["bce"] == pytest.approx(metrics["shared_stock_bce_component"] + 8/9 * metrics["nonstock_bce"])
    assert np.array_equal(root_positions(17, 1280), root_positions(17, 1280))
    assert len(set(root_positions(17, 1280))) == 4


def test_initial_gradient_diagnostic_reports_zero_and_all_candidate_counts():
    model, _, batch = setup_model("temporal_cnn", torch.device("cpu"))
    class Roots:
        entries = [{"candidate_group_id": "r0"}, {"candidate_group_id": "r1"}]
        def __len__(self):
            return 2
        def __getitem__(self, index):
            item = {k: v[index].numpy() for k, v in batch.items()}
            item["actions"] = item.pop("action")
            return item
    report = gradient_diagnostics(model, Roots(), torch.device("cpu"), model.action_std)
    summary = report["summary"]
    assert summary["near_zero_gradient_roots"] == 2
    assert summary["rescue_candidates"] == summary["rescue_zero"] == 2
    assert summary["spoil_candidates"] == summary["spoil_zero"] == 2
    assert summary["gradient_macro_pair_accuracy"] == .5
    assert all(row["gradient_projection"][0] == 0 for row in report["root_records"])
