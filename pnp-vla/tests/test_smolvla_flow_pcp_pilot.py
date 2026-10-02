import sys
import types
import numpy as np
import pytest
import torch

from pnp.pnp import PnPRecorder, run_probe
from pnp.sampler import install_smolvla_patch, set_strategy
from pnp.smolvla_flow_pcp_pilot import FlowPCPTap, apply_flow_q_correction
from pnp.smolvla_tree_collection import _pnp_config
from pnp.tap import BatchedRolloutTap


def score(action):
    return action.mean((1, 2)).sigmoid()


def probe():
    x = torch.arange(400, dtype=torch.float32).reshape(1, 50, 8) / 400
    generator = torch.Generator().manual_seed(71)
    return run_probe(x, .7, lambda value: .2 * value + .1, k=1, adim=7, generators=[generator])


def test_full_width_clean_correction_preserves_latent_tail_and_shared_noise():
    pr = probe()
    valid = np.arange(10) < 6
    zero, meta = apply_flow_q_correction(pr, score, valid, mode="zero", action_rms=0, random_seed=5)
    assert zero is pr.x_acc
    for mode in ("q_plus", "q_minus", "random"):
        updated, meta = apply_flow_q_correction(pr, score, valid, mode=mode, action_rms=.06, random_seed=5)
        assert meta["realized_clean_action_rms"] == pytest.approx(.06, abs=1e-6)
        assert meta["realized_latent_action_rms"] == pytest.approx(.018, abs=1e-6)
        assert torch.equal(updated[:, 6:], pr.x_acc[:, 6:])
        assert torch.equal(updated[:, :, 7:], pr.x_acc[:, :, 7:])
        assert meta["injected_clean_tail_max_abs"] == 0
    plus, _ = apply_flow_q_correction(pr, score, valid, mode="q_plus", action_rms=.02, random_seed=5, correction_mode="lambda")
    z = pr.z_hat_full[:, :10, :7].clone().requires_grad_(True)
    gradient = torch.autograd.grad(score(z).sum(), z)[0] * torch.as_tensor(valid)[None, :, None]
    expected = pr.x_acc.clone(); expected[:, :10, :7] += .3 * .02 * gradient
    torch.testing.assert_close(plus, expected)


class FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(num_steps=10, chunk_size=50, max_action_dim=8, use_cache=True)
        self.vlm_with_expert = types.SimpleNamespace(forward=lambda **kwargs: (None, "cache"))
        self.seen = []
    def sample_actions(self, *args, noise=None, **kwargs):
        return noise
    def embed_prefix(self, *args, state=None):
        return torch.ones(1, 4, 8), torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)
    def denoise_step(self, *, x_t, timestep, **kwargs):
        self.seen.append(float(timestep[0]))
        return x_t * .2 + x_t.mean((1, 2), keepdim=True) * .03


@pytest.mark.parametrize("correction_step", [2, 3])
def test_actual_sampler_zero_is_full_chunk_bitwise_pnp_and_rng_matched(monkeypatch, correction_step):
    module = types.ModuleType("lerobot.policies.smolvla.modeling_smolvla")
    module.make_att_2d_masks = lambda pad, att: pad
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = FakeModel(); install_smolvla_patch(model)
    cfg = _pnp_config()
    valid = np.ones(10, bool)
    def tap(cls, **kwargs):
        rec = PnPRecorder(); rec.new_episode()
        return cls(cfg, [rec], [torch.Generator().manual_seed(9)], "cpu", 7, **kwargs)
    def run(strategy):
        set_strategy(model, strategy)
        return model.sample_actions([], [], torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 7),
            noise=torch.arange(400, dtype=torch.float32).reshape(1, 50, 8) / 400)
    baseline_tap = tap(BatchedRolloutTap)
    baseline = run(baseline_tap)
    zero_tap = tap(FlowPCPTap, score=score, valid=valid, mode="zero", action_rms=0,
                   random_seed=88, correction_step=correction_step)
    zero = run(zero_tap)
    assert torch.equal(zero, baseline)
    assert torch.equal(zero_tap.generators[0].get_state(), baseline_tap.generators[0].get_state())
    reference = zero_tap.correction_records[0]
    for mode in ("q_plus", "q_minus", "random"):
        changed_tap = tap(FlowPCPTap, score=score, valid=valid, mode=mode, action_rms=.02,
                         random_seed=88, correction_step=correction_step)
        changed = run(changed_tap)
        assert not torch.equal(changed, baseline)
        record = changed_tap.correction_records[0]
        assert record["s"] == pytest.approx(1 - correction_step / 10)
        assert record["shared_last_eps_sha256"] == reference["shared_last_eps_sha256"]
        assert record["clean_estimate_sha256"] == reference["clean_estimate_sha256"]
        assert torch.equal(changed_tap.generators[0].get_state(), baseline_tap.generators[0].get_state())
    assert model.seen[-1] == pytest.approx(.1)  # sampler continues after the hook


def test_nonfinite_strength_and_overflowing_gradient_norm_are_rejected():
    pr = probe()
    for strength in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="invalid correction"):
            apply_flow_q_correction(pr, score, np.ones(10, bool), mode="q_plus",
                action_rms=strength, random_seed=1)
    def huge_gradient(action):
        return ((action - action.detach()) * 1e30).sum((1, 2))
    with pytest.raises(ValueError, match="nonfinite scalar"):
        apply_flow_q_correction(pr, huge_gradient, np.ones(10, bool), mode="q_plus",
            action_rms=.02, random_seed=1)
