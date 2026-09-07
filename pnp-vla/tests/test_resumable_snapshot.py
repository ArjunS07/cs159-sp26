import io
import json

import httpx
import numpy as np

from pnp.pcp_critic.resumable_snapshot import (
    _resumable_action_statistics, load_training_fields_with_retry)
from pnp.store import TRAINING_DATA_MULTIPART_FORMAT


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _objects(value):
    actions = np.full((2, 50, 7), value, np.float32)
    manifest = {
        "format": TRAINING_DATA_MULTIPART_FORMAT,
        "arrays": {
            "bellman/action": {
                "shape": list(actions.shape), "dtype": actions.dtype.str,
                "parts": [{"path": f"part-{value}", "start": None, "stop": None}],
            },
            "unused/raw_images": {
                "shape": [100, 100], "dtype": np.dtype(np.uint8).str,
                "parts": [{"path": f"unused-{value}", "start": None, "stop": None}],
            },
        },
    }
    return {f"manifest-{value}": json.dumps(manifest).encode(),
            f"part-{value}": _npz(**{"bellman/action": actions}),
            f"unused-{value}": _npz(**{"unused/raw_images": np.zeros((100, 100), np.uint8)})}


class FakeStore:
    def __init__(self, objects, fail_once=None):
        self.objects = objects
        self.fail_once = fail_once
        self.downloaded = []

    def _download(self, path):
        self.downloaded.append(path)
        if path == self.fail_once:
            self.fail_once = None
            raise httpx.ReadError("connection reset")
        return self.objects[path]


def test_selected_field_loader_retries_and_skips_unused_parts(monkeypatch):
    store = FakeStore(_objects(1), fail_once="part-1")
    monkeypatch.setattr("pnp.pcp_critic.resumable_snapshot.time.sleep", lambda _: None)
    loaded = load_training_fields_with_retry(
        store, "manifest-1", ("bellman/action",), attempts=2)
    assert loaded["bellman/action"].shape == (2, 50, 7)
    assert store.downloaded == ["manifest-1", "part-1", "part-1"]
    assert "unused-1" not in store.downloaded


def test_action_statistics_resume_without_redownloading_completed_rows(tmp_path):
    objects = {**_objects(1), **_objects(2)}
    rows = [
        {"rollout_id": "r1", "training_data_path": "manifest-1",
         "training_data_schema_version": 1, "run_id": "run"},
        {"rollout_id": "r2", "training_data_path": "manifest-2",
         "training_data_schema_version": 1, "run_id": "run"},
    ]
    progress = tmp_path / "progress.json"
    first = FakeStore(objects)
    mean, _, transitions = _resumable_action_statistics(
        first, rows, progress_path=progress, checkpoint_interval=1)
    assert transitions == 4
    assert np.allclose(mean, 1.5)
    second = FakeStore({})
    resumed_mean, _, resumed_transitions = _resumable_action_statistics(
        second, rows, progress_path=progress, checkpoint_interval=1)
    assert resumed_transitions == transitions
    assert np.allclose(resumed_mean, mean)
    assert second.downloaded == []
