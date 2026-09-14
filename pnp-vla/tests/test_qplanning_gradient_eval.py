import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from pnp.config import ALL_METHODS, Method
from pnp.qplanning_critic.inference import direct_latent_q_update
from pnp.qplanning_gradient_eval_experiment import (
    QGUIDE_DENOISE_STEPS, QGUIDE_EULER_STEP, QGUIDE_METHODS,
    QGUIDE_UPDATE_RMS, build_qguide_method)


ROOT = Path(__file__).parents[1]


def test_qguide_methods_are_distinct_stock_decoder_interventions():
    assert set(QGUIDE_METHODS.values()).issubset(ALL_METHODS)
    scorer = SimpleNamespace(checkpoint_id="critic-id")
    for strategy, expected_method in QGUIDE_METHODS.items():
        method, config = build_qguide_method(strategy, "repo@revision", scorer)
        assert method == expected_method
        assert config.num_samples is None
        assert config.q_guidance_ckpt_id == "critic-id"
        assert config.q_guidance_step == QGUIDE_EULER_STEP
        assert config.q_guidance_step_size == QGUIDE_UPDATE_RMS
        assert config.num_inference_steps == QGUIDE_DENOISE_STEPS
        assert config.n_action_steps == 10
        assert config.video == "off" and not config.save_observations
        assert not config.save_generated_chunks
        logical = config.logical_dict()
        assert logical["q_guidance_ckpt_id"] == "critic-id"
        assert logical["q_guidance_step"] == QGUIDE_EULER_STEP


class _LinearScorer:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(action_dim=7))

    def score_with_grad(self, prefix, prefix_valid, robot, proprio, actions):
        return actions[:, :50, :7].mean(dim=(1, 2))

    @torch.no_grad()
    def score(self, prefix, prefix_valid, robot, proprio, actions, *, batch_size=1):
        return actions[:, :50, :7].mean(dim=(1, 2))


def test_direct_latent_q_update_ascends_q_and_has_exact_rms():
    x = torch.zeros((1, 50, 32), dtype=torch.float32)

    def vf(value):
        return value * 0.5

    updated, telemetry, before, after = direct_latent_q_update(
        x, 0.7, vf, scorer=_LinearScorer(),
        prefix=torch.zeros((1, 2, 4)),
        prefix_valid=torch.ones((1, 2), dtype=torch.bool),
        robot_state=torch.zeros(9), policy_proprio=torch.zeros(8),
        step_size=0.005, checkpoint_vfield=False)
    assert updated.shape == x.shape
    assert before.shape == after.shape == x.shape
    assert abs(telemetry["update_rms"] - 0.005) < 1e-6
    assert telemetry["post_q"] > telemetry["pre_q"]
    assert telemetry["delta_q"] > 0
    assert telemetry["first10_policy_action_rms"] > 0


def test_notebook_75_runs_three_guides_with_historical_stock():
    path = ROOT / "notebooks" / "75_eval_q50_latent_guidance_heldout160.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "single full PRO160 worker" in source
    assert "EPISODE_LIMIT = None" in source
    assert "checkpoint_step_008000.pt" in source
    assert source.count("checkpoint_step_006000.pt") == 2
    assert "LATENT_UPDATE_RMS = 0.005" in source
    assert "run_qplanning_latent_guidance_heldout160(" in source
    assert "reuse exact matched notebook-68 outcomes" in source
    assert "periodic_print_every_complete_identities': 10" in source
    assert "Videos, frames, and generated chunks are off" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"{path.name}:cell{index}")
