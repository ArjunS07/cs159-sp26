import numpy as np
import pytest

from pnp.smolvla_scalar_returns import restore_transitions


def fixture():
    actions = np.arange(140, dtype=np.float32).reshape(2, 10, 7)
    raw = {"bellman/action": actions, "boundary/step": np.array([0, 10, 13]),
           "terminated": np.array([False] * 12 + [True]), "truncated": np.zeros(13, bool),
           "step_success": np.array([False] * 12 + [True])}
    old = {"action": np.zeros_like(actions), "prefix": np.zeros((2, 4, 8)),
           "pad": np.ones((2, 4), bool), "robot": np.array([[0, 1], [0, 210 / 220]]),
           "proprio": np.zeros((2, 3))}
    spec = {"start": 0, "step_offset": 0, "suite": "libero_spatial", "success": True}
    return old, raw, spec


def test_terminal_action_proposal_is_not_truncated_by_success():
    old, raw, spec = fixture()
    result = restore_transitions(old, raw, spec)
    np.testing.assert_array_equal(result["action"], raw["bellman/action"])
    assert result["action_valid"].all()
    np.testing.assert_array_equal(result["reward"], [0, 1])
    np.testing.assert_array_equal(result["discount"], [1, 0])


def test_known_deadline_controls_mask_and_failed_terminal_target():
    old, raw, spec = fixture()
    spec.update(step_offset=205, success=False)
    raw["step_success"][:] = False
    raw["terminated"][:] = False
    raw["truncated"][-1] = True
    old["robot"][:, -1] = [15 / 220, 5 / 220]
    result = restore_transitions(old, raw, spec)
    assert result["action_valid"][0].sum() == 10
    assert result["action_valid"][1].sum() == 5
    assert result["reward"].sum() == 0
    assert result["discount"][-1] == 0


def test_branch_root_is_added_before_recorded_continuation():
    old, raw, spec = fixture()
    root = {key: old[key][0] for key in ("prefix", "pad", "robot", "proprio")}
    old = {key: value[1:] for key, value in old.items()}
    spec["start"] = 1
    result = restore_transitions(old, raw, spec, root)
    assert len(result["action"]) == 2
    assert result["robot"][0, -1] == 1


def test_bad_time_alignment_is_rejected():
    old, raw, spec = fixture()
    old["robot"][0, -1] = .5
    with pytest.raises(ValueError, match="time alignment"):
        restore_transitions(old, raw, spec)


def test_recovery_restores_optimizer_and_next_update(tmp_path, monkeypatch):
    import copy
    import torch
    from pnp.smolvla_scalar_returns import ScalarCritic
    from pnp.qplanning_critic.config import QPlanningModelConfig
    from scripts.train_smolvla_scalar_returns_local import save_recovery, load_recovery

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    torch.manual_seed(42)
    cfg = QPlanningModelConfig(action_horizon=10, action_dim=7, width=16,
                              n_layers=2, n_heads=2, ffn_width=32, dropout=0,
                              n_bins=2, prefix_pool_tokens=4)
    def make():
        model = ScalarCritic(prefix_dim=8, robot_dim=2, proprio_dim=3, config=cfg)
        return model, copy.deepcopy(model), torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, target, optimizer = make()
    inputs = (torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool),
              torch.randn(2, 2), torch.randn(2, 3), torch.randn(2, 10, 7),
              torch.ones(2, 10, dtype=torch.bool))
    def update(m, opt):
        opt.zero_grad(); loss = m(*inputs).square().mean(); loss.backward(); opt.step()
    update(model, optimizer)
    path = tmp_path / "latest.pt"
    config = {"updates": 2000, "batch_size": 32, "lr": 1e-3}
    save_recovery(path, model, target, optimizer, arm="td", step=1,
                  digest="test", config=config, history=[], run_id="local-test")
    update(model, optimizer)
    recovered, recovered_target, recovered_opt = make()
    state = load_recovery(path, recovered, recovered_target, recovered_opt,
                          arm="td", digest="test", config=config)
    assert state["update"] == 1
    update(recovered, recovered_opt)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, recovered.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="mismatch"):
        load_recovery(path, recovered, recovered_target, recovered_opt,
                      arm="mc", digest="test", config=config)


def test_root_scoring_uses_scalar_probability_not_categorical_softmax():
    import torch
    from pnp.smolvla_tree_bellman_finetune import _root_scores

    class Critic:
        def expected_value(self, prefix, pad, robot, proprio, action, action_valid):
            return action[:, 0, 0].sigmoid()

    actions = torch.zeros(1, 3, 10, 7)
    actions[0, :, 0, 0] = torch.tensor([-2., 0., 2.])
    batch = {"action": actions, "action_valid": torch.ones(1, 3, 10, dtype=torch.bool),
             "prefix": torch.zeros(1, 4, 8), "pad": torch.ones(1, 4, dtype=torch.bool),
             "robot": torch.zeros(1, 2), "proprio": torch.zeros(1, 3)}
    scores = _root_scores(Critic(), batch)
    torch.testing.assert_close(scores, torch.tensor([[-2., 0., 2.]]).sigmoid())
    assert scores.argmax().item() == 2


def test_nstep_sampler_stops_bootstrap_at_terminal(tmp_path):
    from pnp.smolvla_scalar_returns import ReturnSampler
    arrays = {"prefix": np.zeros((4, 2, 8), np.float32), "pad": np.ones((4, 2), bool),
              "robot": np.zeros((4, 2), np.float32), "proprio": np.zeros((4, 3), np.float32),
              "action": np.arange(280, dtype=np.float32).reshape(4, 10, 7),
              "action_valid": np.ones((4, 10), bool), "reward": np.array([0, 0, 0, 1], np.float32),
              "discount": np.array([1, 1, 1, 0], np.float32), "success": np.ones(4, np.float32)}
    np.savez(tmp_path / "test.npz", **arrays)
    index = {"cache_dir": str(tmp_path), "train_group_ids": ["g"],
             "entries": [{"group_id": "g", "file": "test.npz", "windows": 4}]}
    class RNG:
        def integers(self, *args): return 0
    sampler = ReturnSampler(index)
    one = sampler.batch(RNG(), 1, 1)
    five = sampler.batch(RNG(), 1, 5)
    assert one["reward"].item() == 0 and one["discount"].item() == 1
    assert five["reward"].item() == 1 and five["discount"].item() == 0
    np.testing.assert_array_equal(one["next_action"][0], arrays["action"][1])


def test_late_fusion_initially_preserves_base_function():
    import torch
    from pnp.smolvla_scalar_returns import ScalarCritic, LateFusionScalarCritic
    from pnp.qplanning_critic.config import QPlanningModelConfig
    cfg = QPlanningModelConfig(action_horizon=10, action_dim=7, width=16, n_layers=2,
                              n_heads=2, ffn_width=32, dropout=0, n_bins=2, prefix_pool_tokens=4)
    kwargs = dict(prefix_dim=8, robot_dim=2, proprio_dim=3, config=cfg)
    base = ScalarCritic(**kwargs).eval()
    late = LateFusionScalarCritic(**kwargs).eval()
    late.load_state_dict(base.state_dict(), strict=False)
    inputs = (torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool),
              torch.randn(2, 2), torch.randn(2, 3), torch.randn(2, 10, 7),
              torch.ones(2, 10, dtype=torch.bool))
    torch.testing.assert_close(base(*inputs), late(*inputs), rtol=0, atol=0)
