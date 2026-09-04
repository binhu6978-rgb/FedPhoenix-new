from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_phase41_interaction import (
    interaction_noise_variance,
    stable_reversal_summary,
    two_way_decomposition,
)


class Phase41InteractionTests(unittest.TestCase):
    def test_balanced_decomposition_is_exact(self):
        client = np.array([[-2.0], [2.0]])
        state = np.array([[-1.0, 1.0]])
        interaction = np.array([[1.0, -1.0], [-1.0, 1.0]])
        matrix = 5.0 + client + state + interaction
        result = two_way_decomposition(matrix)
        self.assertAlmostEqual(result["client_main_effect_variance"], 4.0)
        self.assertAlmostEqual(result["state_main_effect_variance"], 1.0)
        self.assertAlmostEqual(result["interaction_variance"], 1.0)
        self.assertAlmostEqual(result["total_variance"], 6.0)
        self.assertAlmostEqual(result["variance_reconstruction_error"], 0.0)

    def test_stable_reversal_requires_replicable_orders(self):
        # Client 0 beats client 1 in state 0 and loses in state 1 in every
        # replicate; the differences are large enough for the df=2 t interval.
        stack = np.array(
            [
                [[1.0, -1.0], [0.0, 0.0]],
                [[1.1, -1.1], [0.0, 0.0]],
                [[0.9, -0.9], [0.0, 0.0]],
            ]
        )
        result = stable_reversal_summary(stack)
        self.assertEqual(result["candidate_client_pair_state_pair_count"], 1)
        self.assertEqual(result["stable_comparable_count"], 1)
        self.assertEqual(result["stable_reversal_count"], 1)

    def test_interaction_noise_removes_replicate_row_main_effects(self):
        # Probe variation that is only client-main-effect variation is real raw
        # cell noise, but it must not be charged to the interaction subspace.
        stack = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[5.0, 5.0], [6.0, 6.0]],
            ]
        )
        result = interaction_noise_variance(stack)
        self.assertGreater(result["raw_cell_measurement_noise_variance"], 0.0)
        self.assertAlmostEqual(
            result["interaction_measurement_noise_variance_single_replicate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
