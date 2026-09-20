import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.pnp import PnPRecorder
from pnp.sampler import install_smolvla_patch, set_strategy
from pnp.smolvla_blend_ablation_experiment import (
    build_smolvla_blend_ablation_methods,
    build_smolvla_consensus_a1_method,
)
from pnp.tap import RolloutTap


def test_blend_ablation_configs_are_the_requested_two_arms():
    methods = build_smolvla_blend_ablation_methods()
    assert [name for name, _ in methods] == [
        Method.SMOLVLA_STOCK_REFINE_RAW_AVERAGE,
        Method.SMOLVLA_FOUR_CANDIDATE_PROJECT_K3,
    ]
    raw = methods[0][1]
    assert raw.refine and raw.consensus_average_only
    assert tuple(raw.pnp_steps) == (1, 2, 3)
    assert tuple(raw.pnp_k_by_step) == (3, 1, 1)
    assert raw.consensus_projection_k is None
    multi = methods[1][1]
    assert not multi.refine
    assert multi.consensus_candidate_count == 4
    assert multi.consensus_candidate_inference_steps == 5
    assert multi.consensus_projection_k == 3
    assert multi.consensus_projection_step == 5
    assert all(config.n_action_steps == 10 for _, config in methods)


def test_consensus_a1_is_same_k3_projection_with_one_executed_action():
    method, config = build_smolvla_consensus_a1_method()
    assert method == Method.SMOLVLA_CONSENSUS_PROJECT_K3_A1
    assert config.refine
    assert tuple(config.pnp_steps) == (1, 2, 3)
    assert tuple(config.pnp_k_by_step) == (3, 1, 1)
    assert config.consensus_projection_k == 3
    assert config.consensus_projection_step == 5
    assert config.n_action_steps == 1


def test_candidate_consensus_validation_and_hash_material():
    with pytest.raises(ValueError, match="both count and inference_steps"):
        RolloutConfig(consensus_candidate_count=4)
    with pytest.raises(ValueError, match="requires projection settings"):
        RolloutConfig(
            consensus_candidate_count=4,
            consensus_candidate_inference_steps=5)
    config = build_smolvla_blend_ablation_methods()[1][1]
    logical = config.logical_dict()
    assert logical["consensus_candidate_count"] == 4
    assert logical["consensus_candidate_inference_steps"] == 5


def test_candidate_average_preserves_first_gripper_only_on_disagreement():
    config = build_smolvla_blend_ablation_methods()[1][1]
    tap = RolloutTap(config, SimpleNamespace(), device="cpu", adim=7)
    candidates = torch.tensor([
        [[[0., 0., 0., 0., 0., 0., -1.]]],
        [[[2., 2., 2., 2., 2., 2., 1.]]],
        [[[4., 4., 4., 4., 4., 4., 1.]]],
        [[[6., 6., 6., 6., 6., 6., 1.]]],
    ])
    average = tap.average_candidate_actions(candidates)
    assert torch.allclose(average[..., :6], torch.full_like(average[..., :6], 3.0))
    assert average[..., 6].item() == -1.0


def test_worker_notebooks_call_the_expected_entrypoints():
    root = Path(__file__).parents[1] / "notebooks" / "workers"
    expected = {
        "100_smolvla_a10_blend_ablation_eval.ipynb":
            "run_smolvla_blend_ablation_eval_worker",
        "101_smolvla_a1_consensus_projection_k3_eval.ipynb":
            "run_smolvla_consensus_a1_eval_worker",
    }
    for filename, entrypoint in expected.items():
        notebook = json.loads((root / filename).read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert entrypoint in source
        assert "'identities': 400" in source
        assert "'video': 'off'" in source


def test_four_candidates_are_integrated_as_one_expanded_batch(monkeypatch):
    module = types.ModuleType("lerobot.policies.smolvla.modeling_smolvla")
    module.make_att_2d_masks = lambda pad, att: torch.ones(
        (pad.shape[0], pad.shape[1], pad.shape[1]),
        dtype=torch.bool, device=pad.device)
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.policies", types.ModuleType("lerobot.policies"))
    monkeypatch.setitem(
        sys.modules, "lerobot.policies.smolvla", types.ModuleType("lerobot.policies.smolvla"))
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla.modeling_smolvla", module)

    class FakeVLM:
        def forward(self, **kwargs):
            return None, "cache"

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(
                num_steps=10, chunk_size=3, max_action_dim=2, use_cache=True)
            self.vlm_with_expert = FakeVLM()
            self.seen_batches = []

        def sample_actions(self, images, img_masks, tokens, masks, state, noise=None, **kwargs):
            return noise

        def sample_noise(self, shape, device):
            return torch.zeros(shape, device=device)

        def embed_prefix(self, images, img_masks, tokens, masks, state=None):
            batch = state.shape[0]
            return (torch.ones((batch, 2, 4)), torch.ones((batch, 2), dtype=torch.bool),
                    torch.ones((batch, 2), dtype=torch.bool))

        def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
            self.seen_batches.append(x_t.shape[0])
            return torch.ones_like(x_t)

    model = FakeModel()
    install_smolvla_patch(model)
    config = build_smolvla_blend_ablation_methods()[1][1]
    recorder = PnPRecorder()
    recorder.new_episode()
    set_strategy(model, RolloutTap(config, recorder, "cpu", adim=2))
    model._pnp.num_steps = 10
    output = model.sample_actions(
        [torch.zeros((1, 3, 2, 2))], [torch.ones((1,), dtype=torch.bool)],
        torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2)), noise=torch.zeros((1, 3, 2)))
    assert output.shape == (1, 3, 2)
    assert model.seen_batches[:5] == [4] * 5
    assert all(batch == 1 for batch in model.seen_batches[5:])
