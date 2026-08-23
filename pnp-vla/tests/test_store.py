import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np

from pnp.store import (SupabaseStore, TRAINING_DATA_MULTIPART_FORMAT,
                       _training_data_payloads)


class StoreSerializationTests(unittest.TestCase):
    def test_dim_cols_accepts_numpy_array(self):
        cols = SupabaseStore._dim_cols(np.array([1.25, 2.5]), "u_d")

        self.assertEqual(cols["u_d0"], 1.25)
        self.assertEqual(cols["u_d1"], 2.5)
        self.assertIsNone(cols["u_d2"])

    def test_dim_cols_accepts_none(self):
        cols = SupabaseStore._dim_cols(None, "a_std_d")

        self.assertTrue(all(value is None for value in cols.values()))

    @patch("pnp.store.time.sleep")
    def test_upload_retries_transport_timeout(self, sleep):
        class Files:
            calls = 0

            def upload(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise httpx.ReadTimeout("temporary")

        files = Files()
        storage = SimpleNamespace(from_=lambda _bucket: files)
        store = SupabaseStore.__new__(SupabaseStore)
        store.client = SimpleNamespace(storage=storage)
        store.bucket = "artifacts"
        store._bytes_written = 0

        store._upload("x", b"abc", attempts=2)

        self.assertEqual(files.calls, 2)
        self.assertEqual(store._bytes_written, 3)
        sleep.assert_called_once_with(1)

    def test_large_training_artifact_round_trips_through_multipart_storage(self):
        rng = np.random.default_rng(7)
        arrays = {
            "prefix/prefix_embeddings": rng.integers(
                0, 256, size=(9, 4096), dtype=np.uint8),
            "boundary/raw_agentview": rng.integers(
                0, 256, size=(9, 2048), dtype=np.uint8),
            "schema_version": np.asarray(1, dtype=np.int16),
        }
        path, uploads = _training_data_payloads("rollout", arrays, max_part_bytes=10_000)
        self.assertTrue(path.endswith("/manifest.json"))
        self.assertGreater(len(uploads), 2)
        payloads = dict(uploads)
        manifest = __import__("json").loads(payloads[path])
        self.assertEqual(manifest["format"], TRAINING_DATA_MULTIPART_FORMAT)
        self.assertTrue(all(len(data) <= 10_000 for key, data in uploads if key != path))

        store = object.__new__(SupabaseStore)
        store._download = payloads.__getitem__
        loaded = store.load_training_data(path)
        self.assertEqual(set(loaded), set(arrays))
        for key, value in arrays.items():
            np.testing.assert_array_equal(loaded[key], value)


if __name__ == "__main__":
    unittest.main()
