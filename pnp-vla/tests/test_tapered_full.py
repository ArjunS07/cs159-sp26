import json
from pathlib import Path

from pnp.config import Method
from pnp.diversity import build_source_tapered_full_config
from pnp.store import SupabaseStore


ROOT = Path(__file__).parents[1]


def test_full_tapered_config_is_fixed_and_distinct():
    config = build_source_tapered_full_config()

    assert config.n_action_steps == 10
    assert config.pnp_k == 5
    assert config.pnp_steps == (3, 4)
    assert config.refine is True
    assert config.refine_average is False
    assert config.refine_tail_decay_end == 20
    assert config.skip_unused_renders is True
    assert config.render_lead == 2
    assert config.logical_dict()["refine_tail_decay_end"] == 20
    assert SupabaseStore.config_hash(SupabaseStore._logical_key(
        Method.TAPERED_REFINEMENT, config))


def test_full_tapered_workers_are_two_fixed_650_identity_shards():
    for shard_index in range(2):
        path = ROOT / "notebooks" / "workers" / (
            f"37_source_tapered_full_worker_{shard_index}.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "run_source_tapered_full_worker(" in source
        assert "EPISODES_PER_TASK = 10" in source
        assert "SHARD_COUNT = 2" in source
        assert f"SHARD_INDEX = {shard_index}" in source
        assert "source_model_revision=SOURCE_MODEL_REVISION" in source
        assert "historical full PnP" in source
        assert "historical unrefined" in source
