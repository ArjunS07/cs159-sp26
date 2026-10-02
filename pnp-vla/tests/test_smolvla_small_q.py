import numpy as np
import pytest
import torch

from pnp.smolvla_small_q import SmallSuccessQ, root_diagnostics, small_root_scores


def test_small_q_scores_nine_chunks_and_state_only_ignores_actions():
    torch.manual_seed(7)
    batch = {
        "prefix": torch.randn(2, 3, 4),
        "pad": torch.tensor([[True, True, False], [True, True, True]]),
        "robot": torch.randn(2, 5),
        "proprio": torch.randn(2, 6),
        "action": torch.randn(2, 9, 10, 7),
        "action_valid": torch.ones(2, 9, 10, dtype=torch.bool),
    }
    model = SmallSuccessQ(4, 5, 6, width=32, dropout=0).eval()
    scores = small_root_scores(model, batch)
    assert scores.shape == (2, 9)
    state_model = SmallSuccessQ(4, 5, 6, width=32, state_only=True, dropout=0).eval()
    state_scores = small_root_scores(state_model, batch)
    assert torch.allclose(state_scores, state_scores[:, :1].expand_as(state_scores))


def test_root_diagnostics_count_rescues_and_weight_mixed_roots_equally():
    outcomes = np.zeros((2, 9), bool)
    outcomes[0, 1] = True
    outcomes[1, :] = True
    scores = np.full((2, 9), .2)
    scores[0, 1] = .8
    scores[1, 0] = .9
    report = root_diagnostics(scores, outcomes)
    assert report["mixed_roots"] == 1
    assert report["oracle_successes"] == 2
    assert report["original_failures_with_successful_alternative"] == 1
    assert report["argmax"]["rescues"] == 1
    assert report["argmax"]["spoils"] == 0
    assert report["mean_root_pair_accuracy"] == pytest.approx(1.0)
