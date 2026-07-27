import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np

from pnp.store import SupabaseStore


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


if __name__ == "__main__":
    unittest.main()
