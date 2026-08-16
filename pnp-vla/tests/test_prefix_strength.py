import json
from pathlib import Path

import pytest
import torch

from pnp.config import ALL_METHODS, Method, RolloutConfig
from pnp.diversity import build_source_prefix_strength_methods
from pnp.pnp import _pnp_seed_perturb, run_probe, temporal_prefix_weights
from pnp.store import SupabaseStore


ROOT = Path(__file__).parents[1]
ARMS = [Method.PREFIX_REFINEMENT, Method.REDUCED_STRENGTH_REFINEMENT]


def test_prefix_mask_updates_only_executed_positions():
    weights = temporal_prefix_weights(
        50, 10, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.equal(weights[:10], torch.ones(10))
    assert torch.equal(weights[10:], torch.zeros(40))
    with pytest.raises(ValueError, match="prefix"):
        temporal_prefix_weights(
            50, 50, device=torch.device("cpu"), dtype=torch.float32)


def test_reduced_inner_strength_changes_each_probe_excursion():
    x_t = torch.zeros((1, 50, 7))
    vfield = lambda x: torch.ones_like(x)
    _pnp_seed_perturb(123)
    full = run_probe(x_t, 0.5, vfield, k=1)
    _pnp_seed_perturb(123)
    half = run_probe(
        x_t, 0.5, vfield, k=1,
        temporal_update_weights=torch.full((50,), 0.5))

    assert torch.allclose(half.x_acc, 0.5 * full.x_acc)


def test_prefix_and_strength_options_are_validated_and_hashed():
    ordinary = RolloutConfig(pnp_steps=(3, 4), pnp_k=5, refine=True)
    prefix = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True,
        n_action_steps=10, refine_prefix_only=True)
    reduced = RolloutConfig(
        pnp_steps=(3, 4), pnp_k=5, refine=True,
        n_action_steps=10, refine_inner_strength=0.5)

    assert "refine_prefix_only" not in ordinary.logical_dict()
    assert "refine_inner_strength" not in ordinary.logical_dict()
    assert prefix.logical_dict()["refine_prefix_only"] is True
    assert reduced.logical_dict()["refine_inner_strength"] == 0.5
    hashes = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(method, config))
        for method, config in zip(ARMS, (prefix, reduced))}
    assert len(hashes) == 2
    with pytest.raises(ValueError, match="explicit n_action_steps"):
        RolloutConfig(pnp_steps=(3,), refine=True, refine_prefix_only=True)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        RolloutConfig(pnp_steps=(3,), refine=True, refine_inner_strength=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        RolloutConfig(
            pnp_steps=(3,), refine=True, n_action_steps=10,
            refine_prefix_only=True, refine_tail_decay_end=20)


def test_prefix_strength_pilot_has_two_unique_paired_configs():
    methods = build_source_prefix_strength_methods(inner_strength=0.5)

    assert [method for method, _ in methods] == ARMS
    assert all(method in ALL_METHODS for method in ARMS)
    assert [config.n_action_steps for _, config in methods] == [10, 10]
    assert [config.pnp_steps for _, config in methods] == [(3, 4), (3, 4)]
    assert [config.pnp_k for _, config in methods] == [5, 5]
    assert methods[0][1].refine_prefix_only is True
    assert methods[1][1].refine_inner_strength == 0.5
    assert [config.save_time_uncertainty for _, config in methods] == [True, True]


def test_prefix_strength_workers_are_two_full_cohort_shards():
    for shard_index in range(2):
        path = ROOT / "notebooks" / "workers" / (
            f"38_source_prefix_strength_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_prefix_strength_worker(" in source
        assert "EPISODES_PER_TASK = 10" in source
        assert "INNER_STRENGTH = 0.5" in source
        assert "SHARD_COUNT = 2" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "full_cohort_identities': 1300" in source
        assert "rollouts_in_this_shard': 1300" in source
        assert "episodes_per_task=EPISODES_PER_TASK" in source
        assert "source_model_revision=SOURCE_MODEL_REVISION" in source
