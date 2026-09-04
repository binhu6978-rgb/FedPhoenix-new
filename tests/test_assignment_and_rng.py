from __future__ import annotations

import unittest

import numpy as np

from fedrad.rng import RNGStreams
from fedrad.task_bank import build_task_bank
from fedrad.trainer import baseline_assignments
from fedrad.types import clone_state_dict
from tests.helpers import initialized_tiny_model, tiny_config


class AssignmentAndRNGTests(unittest.TestCase):
    def test_assignment_order_baseline(self):
        config = tiny_config()
        model = initialized_tiny_model(config.seed)
        bank = build_task_bank(
            config=config,
            model_template=model,
            global_state=clone_state_dict(model.state_dict()),
            round_idx=0,
            task_count=2,
            rngs=RNGStreams(config.seed),
        )
        selected = (3, 1)
        self.assertEqual(baseline_assignments(selected, bank), ((3, 0), (1, 1)))

    def test_local_seed_identity(self):
        rngs = RNGStreams(7)
        expected = rngs.local_seed(round_idx=5, client_id=13)
        assignments = ((13, 0), (13, 9))
        seeds = [rngs.local_seed(5, client_id) for client_id, _ in assignments]
        self.assertEqual(seeds, [expected, expected])
        self.assertNotEqual(expected, rngs.probe_seed(5, 13))

    def test_client_sampling_matches_isolated_legacy_stream(self):
        seed = 11
        expected_rng = np.random.RandomState(seed)
        expected_first = tuple(expected_rng.choice(10, 3, replace=False).tolist())
        expected_second = tuple(expected_rng.choice(10, 3, replace=False).tolist())
        actual_rng = RNGStreams(seed)
        self.assertEqual(actual_rng.sample_clients(10, 3), expected_first)
        self.assertEqual(actual_rng.sample_clients(10, 3), expected_second)


if __name__ == "__main__":
    unittest.main()

