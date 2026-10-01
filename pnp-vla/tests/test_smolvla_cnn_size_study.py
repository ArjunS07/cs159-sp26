"""Capacity study: cached differentiation and chronological stopping/recovery."""
import copy

import pytest
import torch
import torch.nn.functional as F

from scripts.run_smolvla_cnn_size_study import (
    advance_early, cached_scores, early_state, restore, save, selected_path,
)
from scripts.train_smolvla_anchored_local import root_scores
from tests.test_smolvla_anchored_runner import setup_model


def test_early_stopping_retains_anchor_and_cannot_look_past_stop():
    state = early_state(.4)
    for update in range(250, 2001, 250):
        assert not advance_early(state, update, .3995)
    assert state["would_stop_update"] == 2000
    assert state["selected_update"] == 0
    before = dict(state)
    assert not advance_early(state, 2250, .1)
    assert state == before


def test_early_improvement_resets_patience_and_tracks_selected_and_stop():
    state = early_state(.4)
    assert not advance_early(state, 250, .41)
    assert advance_early(state, 500, .398)
    assert state["selected_update"] == 500 and state["misses"] == 0
    for update in range(750, 2501, 250):
        advance_early(state, update, .398)
    assert state["would_stop_update"] == 2500
    assert state["selected_update"] == 500


def test_cached_features_preserve_scores_and_all_residual_parameter_gradients():
    model, optimizer, batch = setup_model("temporal_cnn", torch.device("cpu"))
    with torch.no_grad():
        model.readout[-1].weight.normal_(std=.01)
    clone = copy.deepcopy(model)
    features, baseline = model.reference_features(batch["prefix"], batch["pad"], batch["robot"],
        batch["proprio"], batch["action"][:, 0], batch["action_valid"][:, 0])
    cached = {"features": features, "baseline": baseline, "action": batch["action"],
              "valid": batch["action_valid"], "labels": batch["success"]}
    direct_values = root_scores(model, batch)
    cached_values = cached_scores(clone, cached)
    torch.testing.assert_close(direct_values, cached_values, rtol=0, atol=0)
    F.binary_cross_entropy_with_logits(direct_values, batch["success"]).backward()
    F.binary_cross_entropy_with_logits(cached_values, batch["success"]).backward()
    for (name, p), (other, q) in zip(model.named_parameters(), clone.named_parameters()):
        assert name == other
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        else:
            assert p.grad is None and q.grad is None


def test_size_study_resume_reproduces_update_and_enforces_full_contract(tmp_path):
    model, optimizer, batch = setup_model("temporal_cnn", torch.device("cpu"))
    def update(m, o):
        o.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(root_scores(m, batch), batch["success"]).backward()
        o.step()
    update(model, optimizer)
    config = {"width": 16, "seed": 13}
    state = early_state(.4)
    path = tmp_path / "latest.pt"
    save(path, model, optimizer, 1, config, [], state, "run", torch.device("cpu"))
    saved = torch.load(path, weights_only=False)
    resumed, resumed_optimizer, _ = setup_model("temporal_cnn", torch.device("cpu"))
    restore(saved, resumed, resumed_optimizer, config, torch.device("cpu"))
    update(model, optimizer); update(resumed, resumed_optimizer)
    for a, b in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="contract differs"):
        restore(saved, resumed, resumed_optimizer, {**config, "seed": 99}, torch.device("cpu"))
    assert selected_path(tmp_path, 250) != selected_path(tmp_path, 500)
