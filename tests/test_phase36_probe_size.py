from __future__ import annotations

import unittest

from scripts.run_phase36_probe_size import PROTOCOLS, nested_probe_indices


class Phase36ProbeSizeTests(unittest.TestCase):
    def test_nested_indices_are_deterministic_disjoint_and_nested(self):
        local = tuple(range(200))
        support, query = nested_probe_indices(local, probe_seed=12345)
        repeated = nested_probe_indices(local, probe_seed=12345)
        self.assertEqual((support, query), repeated)
        self.assertEqual(len(support), 64)
        self.assertEqual(len(query), 64)
        self.assertFalse(set(support).intersection(query))
        selections = {
            name: (support[:support_size], query[:query_size])
            for name, (support_size, query_size) in PROTOCOLS.items()
        }
        self.assertEqual(selections["P0_32_32"][0], selections["P2_64_32"][0][:32])
        self.assertEqual(selections["P0_32_32"][1], selections["P1_32_64"][1][:32])
        self.assertEqual(selections["P1_32_64"][1], selections["P3_64_64"][1])
        self.assertEqual(selections["P2_64_32"][0], selections["P3_64_64"][0])

    def test_nested_indices_require_128_samples(self):
        with self.assertRaises(ValueError):
            nested_probe_indices(tuple(range(127)), probe_seed=1)


if __name__ == "__main__":
    unittest.main()
