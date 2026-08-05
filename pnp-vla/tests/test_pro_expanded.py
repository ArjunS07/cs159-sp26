"""Guards for the expanded 16-suite LIBERO-PRO run (K=5, refine-last (3,4), u_iter telemetry).

Everything here is pure logic -- no GPU, no LIBERO, no Supabase.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pnp import Method
from pnp.config import (
    LIBERO_PRO_MAX_STEPS,
    MAX_STEPS_MAP,
    LOGICAL_FIELDS,
    resolve_max_steps,
)
from pnp.experiments import (
    PRO_EXPANDED_EXPERIMENT,
    PRO_EXPANDED_K,
    PRO_EXPANDED_STEPS,
    PRO_EXPERIMENT,
    build_pro_expanded_methods,
    build_pro_methods,
)
from pnp.libero_pro import (
    EXPANDED_PRO_SUITES,
    HF_PRO_SUITES,
    OBJECT_ALIASES,
    register_object_aliases,
)
from pnp.store import SupabaseStore

WORKERS = Path(__file__).parents[1] / "notebooks" / "workers"


# ─────────────────────────────────────────────────────────────────────────────
# max_steps: LIBERO-PRO gives every perturbed suite its BASE suite's canonical limit.
# Pinned verbatim from the upstream TASK_MAX_STEPS table so a regression to the flat
# 280 fallback (which truncated libero_10_swap by 46%) cannot pass silently.
# ─────────────────────────────────────────────────────────────────────────────
UPSTREAM_PRO_MAX_STEPS = {
    "libero_object_temp_x0.1": 280, "libero_object_temp_x0.2": 280,
    "libero_object_temp_x0.3": 280, "libero_object_temp_y0.1": 280,
    "libero_object_temp_y0.2": 280, "libero_object_temp_y0.3": 280,
    "libero_goal_swap": 300, "libero_object_swap": 280,
    "libero_spatial_swap": 220, "libero_10_swap": 520,
    "libero_goal_task": 300, "libero_object_task": 280,
    "libero_goal_with_milk": 300, "libero_spatial_with_milk": 220,
    "libero_object_with_mug": 280, "libero_goal_with_yellow_book": 300,
}


@pytest.mark.parametrize("suite,expected", sorted(UPSTREAM_PRO_MAX_STEPS.items()))
def test_pro_max_steps_matches_upstream_table(suite, expected):
    assert resolve_max_steps(suite) == expected


def test_max_steps_table_covers_every_expanded_suite():
    assert set(UPSTREAM_PRO_MAX_STEPS) == set(EXPANDED_PRO_SUITES)


def test_stock_suites_still_resolve_to_their_own_entry():
    for suite, expected in MAX_STEPS_MAP.items():
        assert resolve_max_steps(suite) == expected


def test_longest_base_prefix_wins():
    # libero_10* must not be shadowed by a shorter match, and an unknown base falls back.
    assert resolve_max_steps("libero_10_task") == MAX_STEPS_MAP["libero_10"]
    assert resolve_max_steps("libero_mine_something") == LIBERO_PRO_MAX_STEPS


# ─────────────────────────────────────────────────────────────────────────────
# Method matrix
# ─────────────────────────────────────────────────────────────────────────────
def test_expanded_methods_are_three_unique_arms_at_k5():
    methods = build_pro_expanded_methods(include_control=True)
    assert [name for name, _ in methods] == [
        Method.UNCERTAINTY, Method.EXTRA_STEPS, Method.REFINEMENT,
    ]
    hashes = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(name, config))
        for name, config in methods
    }
    assert len(hashes) == 3

    observed, control, refine = (config for _, config in methods)
    assert observed.pnp_steps == (3, 4) and observed.pnp_k == 5
    assert observed.save_pcp_features and not observed.refine
    # The control must match the probed arms' budget at THIS k: 10 + 5*2.
    assert control.num_inference_steps == 20
    assert not control.has_probe
    assert refine.pnp_steps == (3, 4) and refine.pnp_k == 5
    assert refine.refine and not refine.refine_average


def test_control_is_deferred_by_default():
    names = [name for name, _ in build_pro_expanded_methods()]
    assert names == [Method.UNCERTAINTY, Method.REFINEMENT]


def test_deferring_the_control_preserves_the_other_arms_identities():
    """The whole reason the control can be back-filled later: dropping it must not perturb the
    telemetry arms' rollout ids, or a resumed worker would recollect everything."""
    episode = {"benchmark": "libero_pro", "suite": "libero_goal_swap", "task_idx": 0,
               "ep_idx": 0, "init_state_hash": "abc123abc123"}
    store = object.__new__(SupabaseStore)

    def ids(methods):
        return {name: store.rollout_id(PRO_EXPANDED_EXPERIMENT, episode, name, config)
                for name, config in methods}

    without = ids(build_pro_expanded_methods(include_control=False))
    with_control = ids(build_pro_expanded_methods(include_control=True))
    assert set(without) == {Method.UNCERTAINTY, Method.REFINEMENT}
    for name, rid in without.items():
        assert with_control[name] == rid
    # ...and the control itself is a distinct id, so back-filling collects only what is missing.
    assert with_control[Method.EXTRA_STEPS] not in without.values()


def test_expanded_arms_never_write_ahats_blobs():
    # The decay signal is u_iter (112 B/step), not a_hats stacks (7 KB/step).
    for control in (False, True):
        assert not any(config.save_ahats for _, config
                       in build_pro_expanded_methods(include_control=control))


def test_expanded_experiment_is_separate_from_the_canonical_one():
    assert PRO_EXPANDED_EXPERIMENT != PRO_EXPERIMENT
    canonical = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(name, config))
        for name, config in build_pro_methods()
    }
    expanded = {
        SupabaseStore.config_hash(SupabaseStore._logical_key(name, config))
        for name, config in build_pro_expanded_methods(include_control=True)
    }
    # Different K and schedule => no config collides, so neither run can absorb the other's rows.
    assert not (canonical & expanded)


def test_sinks_stay_out_of_the_rollout_identity():
    """save_* flags are persistence-only; they must never enter the identity hash."""
    for sink in ("save_ahats", "save_pcp_features", "save_observations", "save_trajectory",
                 "save_generated_chunks", "save_uncertainty", "video", "compute_multimodal"):
        assert sink not in LOGICAL_FIELDS


# ─────────────────────────────────────────────────────────────────────────────
# Asset sourcing
# ─────────────────────────────────────────────────────────────────────────────
def test_hf_suites_are_the_official_sixteen():
    assert len(HF_PRO_SUITES) == 16
    assert HF_PRO_SUITES == {
        f"libero_{base}_{kind}"
        for base in ("10", "goal", "object", "spatial")
        for kind in ("lan", "object", "swap", "task")
    }


def test_expanded_suites_route_to_the_right_source():
    """_swap/_task -> HuggingFace; position-perturb and distractor -> the pinned git clone."""
    from_hf = [s for s in EXPANDED_PRO_SUITES if s in HF_PRO_SUITES]
    from_clone = [s for s in EXPANDED_PRO_SUITES if s not in HF_PRO_SUITES]
    assert sorted(from_hf) == sorted([
        "libero_goal_swap", "libero_object_swap", "libero_spatial_swap", "libero_10_swap",
        "libero_goal_task", "libero_object_task",
    ])
    # Position-perturb suites must NEVER be HF-sourced: the clone's per-suite init files are
    # what carry the perturbed object poses.
    assert all("_temp_" in s or "_with_" in s for s in from_clone)


def test_object_aliases_are_registered_into_the_live_dict():
    fake = {base: object() for base in set(OBJECT_ALIASES.values())}
    import pnp.libero_pro as libero_pro_module
    import sys
    import types

    module = types.ModuleType("libero.libero.envs.objects")
    module.OBJECTS_DICT = fake
    saved = {k: sys.modules.get(k) for k in
             ("libero", "libero.libero", "libero.libero.envs", "libero.libero.envs.objects")}
    try:
        for name in ("libero", "libero.libero", "libero.libero.envs"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["libero.libero.envs.objects"] = module
        added = libero_pro_module.register_object_aliases()
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    assert sorted(added) == sorted(OBJECT_ALIASES)
    for alias, base in OBJECT_ALIASES.items():
        assert fake[alias] is fake[base]


# ─────────────────────────────────────────────────────────────────────────────
# Per-iteration uncertainty
# ─────────────────────────────────────────────────────────────────────────────
def _probe(s, k=PRO_EXPANDED_K, chunk=4, adim=7):
    import torch
    from pnp.pnp import run_probe
    scales = iter([1.0, 0.5, 0.25, 0.125, 0.0625][:k])

    def vfield(inp):
        return torch.full_like(inp, next(scales))

    return run_probe(torch.zeros(1, chunk, adim), s=s, vfield=vfield, k=k, adim=adim)


def test_u_iter_is_computed_from_consecutive_a_hats():
    """At s=0 the perturbation term vanishes, so every a_hat is identical and every
    consecutive difference must be exactly zero. Deterministic, unlike s>0."""
    pytest.importorskip("torch")
    rec = _probe(s=0.0).rec
    assert rec["u_iter"].shape == (PRO_EXPANDED_K - 1,)
    assert rec["u_iter_vec"].shape == (PRO_EXPANDED_K - 1, 7)
    assert np.allclose(rec["u_iter"], 0.0)
    assert np.allclose(rec["u_iter_vec"], 0.0)
    assert rec["u_mean"] == pytest.approx(0.0)


def test_u_iter_keeps_the_iteration_axis_that_u_mean_averages_away():
    pytest.importorskip("torch")
    rec = _probe(s=0.5).rec
    u_iter, u_iter_vec = rec["u_iter"], rec["u_iter_vec"]

    assert u_iter.shape == (PRO_EXPANDED_K - 1,)
    assert u_iter_vec.shape == (PRO_EXPANDED_K - 1, 7)
    # Reducing over the action dims must reproduce the scalar sequence.
    assert np.allclose(u_iter_vec.mean(axis=1), u_iter, atol=1e-6)
    # u_mean is exactly the average of the per-iteration values it collapses.
    assert rec["u_mean"] == pytest.approx(float(u_iter.mean()), rel=1e-5)
    # The axis carries real per-iteration information rather than u_mean broadcast K-1 times.
    # Whether it DECAYS is the empirical question this run exists to answer -- not asserted here.
    assert len(set(u_iter.tolist())) > 1


def test_u_iter_survives_the_recorder_to_row_mapping():
    store = object.__new__(SupabaseStore)
    rec_ep = {"chunks": [{"chunk_idx": 0, "steps": [{
        "step": 3, "s": 0.7, "u_mean": 0.05, "u_max": 0.1, "a_std_mean": 0.02,
        "u_vec": np.zeros(7), "a_std_vec": np.zeros(7), "a_mean_vec": np.zeros(7),
        "u_iter": np.array([0.4, 0.3, 0.2, 0.1]),
        "u_iter_vec": np.ones((4, 7)) * 0.25,
    }]}]}
    _, vectors, _ = store._recorder_to_rows(rec_ep, [11])
    assert len(vectors) == 1
    assert vectors[0]["u_iter"] == pytest.approx([0.4, 0.3, 0.2, 0.1])
    assert np.array(vectors[0]["u_iter_vec"]).shape == (4, 7)
    # Must be JSON-serializable for the JSONB columns.
    json.dumps(vectors[0])


# ─────────────────────────────────────────────────────────────────────────────
# Launchers
# ─────────────────────────────────────────────────────────────────────────────
def test_expanded_workers_are_six_fixed_disjoint_shards():
    indices = []
    for shard_index in range(6):
        path = WORKERS / f"libero_pro16_worker_{shard_index}.ipynb"
        source = "".join(
            "".join(cell["source"])
            for cell in json.loads(path.read_text())["cells"]
        )
        assert "run_libero_pro_expanded_worker" in source
        assert "SHARD_COUNT = 6" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        indices.append(shard_index)
    assert indices == list(range(6))


def test_canonical_workers_still_call_the_canonical_driver():
    """Regenerating the launchers must not repoint the canonical run and break its resumption."""
    for shard_index in range(6):
        source = "".join(
            "".join(cell["source"])
            for cell in json.loads(
                (WORKERS / f"libero_pro_worker_{shard_index}.ipynb").read_text())["cells"]
        )
        assert "run_libero_pro_worker(" in source
        assert "run_libero_pro_expanded_worker" not in source
