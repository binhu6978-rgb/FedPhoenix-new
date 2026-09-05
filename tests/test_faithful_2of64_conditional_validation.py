from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_faithful_2of64_conditional_validation import (
    assignment_value,
    hungarian,
    interaction_null,
    overlap_null,
    two_way,
)
from scripts.run_faithful_2of64_conditional_validation import full_local_seed


class FaithfulConditionalValidationTests(unittest.TestCase):
    def test_replicate_seeds_are_independent_of_task_position(self):
        self.assertEqual(full_local_seed(1, 20, 9, 1), full_local_seed(1, 20, 9, 1))
        self.assertNotEqual(full_local_seed(1, 20, 9, 1), full_local_seed(1, 20, 9, 2))
        self.assertNotEqual(full_local_seed(1, 20, 9, 1), full_local_seed(1, 20, 17, 1))

    def test_two_way_decomposition_and_assignment(self):
        client = np.array([[1.0], [2.0], [3.0]])
        state = np.array([[10.0, 20.0, 30.0]])
        interaction = np.array([[3.0, -1.0, -2.0], [-2.0, 3.0, -1.0], [-1.0, -2.0, 3.0]])
        values = client + state + interaction
        observed, metrics = two_way(values)
        np.testing.assert_allclose(observed, interaction)
        self.assertAlmostEqual(metrics["total_variance"], metrics["client_main_effect_variance"] + metrics["state_main_effect_variance"] + metrics["interaction_variance"])
        assignment, objective = hungarian(values)
        self.assertEqual(assignment, (0, 1, 2))
        self.assertAlmostEqual(objective, assignment_value(values, assignment))

    def test_structure_preserving_interaction_null_and_overlap_null(self):
        interaction = np.array([[2.0, -2.0], [-2.0, 2.0]])
        row, pearsons, spearmans = interaction_null(interaction, interaction, np.random.default_rng(7), count=100)
        self.assertAlmostEqual(row["interaction_pearson"], 1.0)
        self.assertEqual(len(pearsons), 100); self.assertEqual(len(spearmans), 100)
        overlap, values = overlap_null((0, 1, 2), (0, 1, 2), np.random.default_rng(9), count=100)
        self.assertEqual(overlap["observed_pair_overlap"], 1.0)
        self.assertEqual(len(values), 100)


if __name__ == "__main__":
    unittest.main()
