import numpy as np

from pnp.smolvla_reconstruction_calibration import _calibration_items, _error_row


def test_error_row_separates_active_and_padded_dimensions():
    target = np.zeros((50, 10), np.float32)
    prediction = target.copy()
    prediction[:10, :7] = 0.01
    prediction[:10, 7:] = 1.0
    item = {
        "source_rollout_id": "r", "suite": "s", "task_idx": 1,
        "episode_idx": 2, "chunk_idx": 3, "source_success": False,
        "selection_strategy": "uniform",
    }
    row = _error_row(
        prediction, target, active_dim=7, item=item, mode="batch_shape",
        batch_size=9, draw_offset=0, position_scheme="generated_chunk_width")
    assert np.isclose(row["first10_active_rms"], 0.01)
    assert np.isclose(row["first10_padded_rms"], 1.0)
    assert row["first10_all_rms"] > row["first10_active_rms"]


def test_calibration_sample_round_robins_groups_deterministically():
    items = [{
        "source_rollout_id": f"r{i}", "suite": f"s{i % 2}",
        "source_success": bool(i % 2),
        "selection_strategy": "u10_priority" if i % 3 else "uniform",
        "selection_tiebreak": f"{i:03d}",
    } for i in range(20)]
    first = _calibration_items(items, 12)
    second = _calibration_items(list(reversed(items)), 12)
    assert [row["source_rollout_id"] for row in first] == [
        row["source_rollout_id"] for row in second]
    assert len(first) == 12
