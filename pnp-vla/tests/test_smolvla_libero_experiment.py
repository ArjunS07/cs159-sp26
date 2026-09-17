import json
import sys
import types
from pathlib import Path

import torch

from pnp import Method
from pnp.config import RolloutConfig
from pnp.experiments import (
    LIBERO_ACTION_STEPS,
    SMOLVLA_LIBERO_K,
    SMOLVLA_LIBERO_STEPS,
    build_smolvla_libero_methods,
)
from pnp.pnp import PnPRecorder
from pnp.sampler import install_smolvla_patch, set_strategy
from pnp.tap import PrefixCaptureTap, RolloutTap


def test_smolvla_method_matrix_is_matched_and_uses_ten_actions():
    methods = build_smolvla_libero_methods()
    assert [name for name, _ in methods] == [Method.UNCERTAINTY, Method.REFINEMENT]
    stock, refine = (config for _, config in methods)
    assert not stock.refine
    assert refine.refine and not refine.refine_average
    for config in (stock, refine):
        assert config.n_action_steps == LIBERO_ACTION_STEPS == 10
        assert config.pnp_k == SMOLVLA_LIBERO_K == 5
        assert config.pnp_steps == SMOLVLA_LIBERO_STEPS == (3, 4)
        assert config.save_time_uncertainty
        assert config.save_trajectory
        assert config.video == "off"
        assert not config.save_observations


def test_two_smolvla_launchers_have_fixed_unique_indices():
    worker_dir = Path(__file__).parents[1] / "notebooks" / "workers"
    launchers = sorted(worker_dir.glob("83_smolvla_libero_a10_worker_*.ipynb"))
    assert len(launchers) == 2
    for index, path in enumerate(launchers):
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "SHARD_COUNT = 2" in source
        assert f"SHARD_INDEX = {index}" in source
        assert "run_smolvla_libero_worker" in source
        assert "ROLLOUT_BATCH_SIZE = 8" in source
        assert "'n_action_steps': 10" in source
        assert "'integration_steps': 10" in source
        assert "'video': 'off'" in source


class _FakeVLM:
    def forward(self, **kwargs):
        return None, "fake-cache"


class _FakeSmolModel:
    def __init__(self):
        self.config = types.SimpleNamespace(
            num_steps=2, chunk_size=3, max_action_dim=2, use_cache=True)
        self.vlm_with_expert = _FakeVLM()

    def sample_actions(
            self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        return noise + 7.0

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def embed_prefix(self, images, img_masks, tokens, masks, state=None):
        batch = state.shape[0]
        return (
            torch.ones((batch, 2, 4), device=state.device),
            torch.ones((batch, 2), dtype=torch.bool, device=state.device),
            torch.ones((batch, 2), dtype=torch.bool, device=state.device),
        )

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        return torch.ones_like(x_t)


def test_smolvla_measurement_tap_returns_exact_stock_chunk(monkeypatch):
    module = types.ModuleType("lerobot.policies.smolvla.modeling_smolvla")
    module.make_att_2d_masks = lambda pad, att: torch.ones(
        (pad.shape[0], pad.shape[1], pad.shape[1]),
        dtype=torch.bool, device=pad.device)
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.policies", types.ModuleType("lerobot.policies"))
    monkeypatch.setitem(
        sys.modules, "lerobot.policies.smolvla", types.ModuleType("lerobot.policies.smolvla"))
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla.modeling_smolvla", module)

    model = _FakeSmolModel()
    install_smolvla_patch(model)
    recorder = PnPRecorder()
    recorder.new_episode()
    config = RolloutConfig(pnp_steps=(0,), pnp_k=2, action_dim=2)
    set_strategy(model, RolloutTap(config, recorder, "cpu", adim=2))
    noise = torch.zeros((1, 3, 2))
    output = model.sample_actions(
        [], [], torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool), torch.zeros((1, 2)), noise=noise)
    assert torch.equal(output, noise + 7.0)
    assert len(recorder.current_chunks()) == 1
    assert recorder.current_chunks()[0]["steps"][0]["step"] == 0


def test_smolvla_sampler_exposes_training_prefix(monkeypatch):
    module = types.ModuleType("lerobot.policies.smolvla.modeling_smolvla")
    module.make_att_2d_masks = lambda pad, att: torch.ones(
        (pad.shape[0], pad.shape[1], pad.shape[1]),
        dtype=torch.bool, device=pad.device)
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.policies", types.ModuleType("lerobot.policies"))
    monkeypatch.setitem(
        sys.modules, "lerobot.policies.smolvla", types.ModuleType("lerobot.policies.smolvla"))
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla.modeling_smolvla", module)

    model = _FakeSmolModel()
    install_smolvla_patch(model)
    tap = PrefixCaptureTap()
    set_strategy(model, tap)
    noise = torch.zeros((1, 3, 2))
    model.sample_actions(
        [torch.zeros((1, 3, 2, 2))], [torch.ones((1,), dtype=torch.bool)],
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool), torch.zeros((1, 2)), noise=noise)

    assert len(tap.training_prefixes) == 1
    prefix = tap.training_prefixes[0]
    assert prefix["prefix_embeddings"].shape == (2, 4)
    assert prefix["token_ids"].shape == (2,)
    assert len(prefix["processed_images"]) == 1
