"""Contracts for the deliberately small action-contrast collection pilot."""
from __future__ import annotations

import numpy as np

from pnp import smolvla_action_contrast_pilot as pilot


def test_pilot_shards_are_disjoint_and_cover_eighteen_roots(monkeypatch):
    items = [{"suite": f"suite_{suite}", "source_success": bool(index % 2),
              "source_rollout_id": f"{suite}-{index}", "ordinal": suite * 200 + index}
             for suite in range(4) for index in range(200)]
    monkeypatch.setattr(pilot.tree, "load_or_build_smolvla_tree_manifest",
                        lambda store: {"manifest_hash": "old", "payload": {"items": items}})
    chosen, digest = pilot._pilot_items(object())
    assert len(chosen) == 18
    assert [item["ordinal"] for item in chosen] == list(range(18))
    assert len({item["source_rollout_id"] for item in chosen}) == 18
    assert all(sum(item["suite"] == f"suite_{suite}" and
                   item["source_success"] == success for item in chosen) >= 2
               for suite in range(4) for success in (False, True))
    assert all(sum(item["ordinal"] % 3 == shard for item in chosen) == 6
               for shard in range(3))
    assert pilot._pilot_items(object())[1] == digest


def test_controlled_chunks_change_only_requested_translation_and_first_ten_steps():
    source = np.zeros((50, 7), np.float32)
    extras = pilot._controlled_chunks(source)
    assert set(extras) | {"stored_source", "fresh_seed_1"} == pilot.KINDS
    for chunk, metadata in extras.values():
        axis = metadata["action_axis"]
        assert np.array_equal(chunk[10:], source[10:])
        assert np.array_equal(chunk[:10, np.arange(7) != axis],
                              source[:10, np.arange(7) != axis])
        assert np.all(np.sign(chunk[:10, axis]) == np.sign(metadata["requested_scale"]))
        assert 0 < metadata["realized_max_abs_delta"] <= abs(metadata["requested_scale"])
