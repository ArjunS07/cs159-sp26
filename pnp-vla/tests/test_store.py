import unittest

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


if __name__ == "__main__":
    unittest.main()
