import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from pnp.config import Method, RolloutConfig
from pnp.smolvla_u20_gated_blend_experiment import (
    SMOLVLA_U20_Q75_BY_SUITE,
    build_smolvla_u20_gated_s03_method,
)
from pnp.store import SupabaseStore
from pnp.tap import BatchedRolloutTap


def _records(value):
    return [[{
        "step": step,
        "u_time": np.full(50, value, np.float32),
    } for step in (1, 2, 3)]]


def test_gated_s03_config_is_online_a10_contract():
    method, config = build_smolvla_u20_gated_s03_method("libero_10")
    assert method == Method.SMOLVLA_U20_GATED_CONSENSUS_S03_K3
    assert config.consensus_projection_gate_threshold == pytest.approx(
        SMOLVLA_U20_Q75_BY_SUITE["libero_10"])
    assert config.consensus_projection_gate_horizon == 20
    assert config.consensus_projection_step == 7
    assert config.consensus_projection_k == 3
    assert config.n_action_steps == 10
    assert tuple(config.pnp_steps) == (1, 2, 3)
    assert tuple(config.pnp_k_by_step) == (3, 1, 1)
    assert config.video == "off"


def test_consensus_gate_requires_complete_valid_configuration():
    with pytest.raises(ValueError, match="threshold and horizon"):
        RolloutConfig(
            pnp_steps=(1,), refine=True, num_inference_steps=10,
            consensus_projection_k=3, consensus_projection_step=7,
            consensus_projection_gate_threshold=0.1)
    with pytest.raises(ValueError, match="requires projection settings"):
        RolloutConfig(
            pnp_steps=(1,), refine=True,
            consensus_projection_gate_threshold=0.1,
            consensus_projection_gate_horizon=20)


def test_batch1_gate_returns_exact_stock_below_threshold_and_projection_above():
    _, config = build_smolvla_u20_gated_s03_method("libero_10")
    recorder = SimpleNamespace()
    tap = BatchedRolloutTap(config, [recorder], [123], "cpu", 7)
    stock = torch.zeros((1, 50, 8))
    refined = torch.ones_like(stock)
    projected = torch.full_like(stock, 2.0)
    tap.project_clean_consensus = lambda consensus, vfield, ctx: projected

    low_ctx = SimpleNamespace(records=_records(0.01))
    low = tap.project_consensus_action(stock, refined, None, low_ctx)
    assert torch.equal(low, stock)
    assert tap.consensus_gate_records[0][-1]["gate_fired"] is False

    high_ctx = SimpleNamespace(records=_records(0.20))
    high = tap.project_consensus_action(stock, refined, None, high_ctx)
    assert torch.equal(high, projected)
    assert tap.consensus_gate_records[0][-1]["gate_fired"] is True


def test_worker_notebook_is_explicitly_online_and_batch1():
    path = (Path(__file__).parents[1] / "notebooks" / "workers"
            / "108_smolvla_a10_online_u20_q75_gated_s03_blend_eval.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_smolvla_u20_gated_s03_eval_worker" in source
    assert "ROLLOUT_BATCH_SIZE = 1" in source
    assert "'online_per_chunk': True" in source
    assert "'gate_horizon': 20" in source
    assert "'projection_s': 0.3" in source
    assert "'n_action_steps': 10" in source
    assert "'print_every_identities': 10" in source
    assert "'video': 'off'" in source


def test_gate_detail_is_nested_in_json_not_forwarded_as_sql_column():
    method, config = build_smolvla_u20_gated_s03_method("libero_10")
    store = object.__new__(SupabaseStore)
    store.log_episode = Mock(return_value="rid")
    detail = {
        "records": [{"chunk_idx": 0, "score": 0.2, "gate_fired": True}],
        "n_considered": 1,
        "n_fired": 1,
        "threshold": 0.1,
        "horizon": 20,
    }
    result = {
        "chunk_noise_seeds": [], "n_chunks": 1, "episode_seed": 1,
        "success": True, "n_steps": 10, "elapsed_s": 1.0,
        "status": "completed", "error_msg": None, "nan_action_count": 0,
        "n_vf_evals": 1, "instability": {},
        "refinement_gate_telemetry": {
            "n_corrections_applied": 1, "gate_fire_rate": 1.0,
            "consensus_projection_gate": detail,
        },
    }
    episode = {
        "benchmark": "libero", "suite": "libero_10", "task_idx": 0,
        "episode_idx": 0, "init_state_hash": "state", "max_steps": 300,
    }

    store.log_result("rid", episode, method, config, result)
    row = store.log_episode.call_args.args[0]
    assert "consensus_projection_gate" not in row
    assert row["n_corrections_applied"] == 1
    assert row["gate_fire_rate"] == pytest.approx(1.0)
    assert row["ms_candidate_u"]["consensus_projection_gate"] == detail
