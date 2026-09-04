from __future__ import annotations

import unittest

import numpy as np

from fedrad.data import (
    generate_dirichlet_partition,
    partition_fingerprint,
    validate_partition,
)


class DataTests(unittest.TestCase):
    def test_dirichlet_partition_replay_and_validation(self):
        labels = np.repeat(np.arange(10), 20)
        first = generate_dirichlet_partition(
            labels,
            num_clients=4,
            beta=0.3,
            min_client_samples=2,
            rng=np.random.default_rng(123),
        )
        second = generate_dirichlet_partition(
            labels,
            num_clients=4,
            beta=0.3,
            min_client_samples=2,
            rng=np.random.default_rng(123),
        )
        validated = validate_partition(
            first,
            num_clients=4,
            dataset_size=len(labels),
            min_client_samples=2,
        )
        self.assertEqual(partition_fingerprint(first), partition_fingerprint(second))
        self.assertEqual(validated, first)

    def test_partition_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_partition(
                {0: (0, 1), 1: (1, 2)},
                num_clients=2,
                dataset_size=3,
                min_client_samples=1,
            )


if __name__ == "__main__":
    unittest.main()

