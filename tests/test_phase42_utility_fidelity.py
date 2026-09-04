from __future__ import annotations

import unittest

import numpy as np

from fedrad.config import FedRADConfig
from scripts.run_phase42_utility_fidelity import (
    RATIO_EPS,
    _analysis_rows,
    _double_center,
    candidate_scores,
)


class Phase42UtilityFidelityTests(unittest.TestCase):
    def test_ratio_candidate_has_fixed_damage_normalization(self):
        raw = {
            "G": np.array([[2.0, 1.0], [3.0, 4.0]]),
            "A": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "D": np.array([[0.0, 1.0], [0.0, 1.0]]),
            "C": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "global_loss": np.array([[3.0, 5.0], [7.0, 11.0]]),
            "reset_loss": np.array([[4.0, 7.0], [11.0, 16.0]]),
            "adapted_loss": np.array([[2.0, 6.0], [8.0, 12.0]]),
        }
        scores = candidate_scores(raw, FedRADConfig())
        expected = raw["G"] / (np.abs(raw["reset_loss"] - raw["global_loss"]) + RATIO_EPS)
        np.testing.assert_allclose(scores["R"], expected)
        self.assertTrue(np.isfinite(scores["Legacy"]).all())
        self.assertTrue(np.isfinite(scores["GAD"]).all())

    def test_double_center_removes_assignment_irrelevant_main_effects(self):
        client = np.array([[1.0], [2.0], [3.0]])
        state = np.array([[10.0, 20.0, 30.0]])
        interaction = np.array([[1.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
        centered = _double_center(client + state + interaction)
        np.testing.assert_allclose(centered, interaction)

    def test_assignment_oracle_is_evaluated_on_v_not_score(self):
        value = np.array([[5.0, 0.0], [0.0, 4.0]])
        # This deliberately chooses the off-diagonal assignment, which is bad on V.
        scores = {"bad": np.array([[0.0, 1.0], [1.0, 0.0]])}
        _, assignments, random_values = _analysis_rows(
            round_number=1, scores=scores, value=value, client_order=(0, 1),
            task_order=(0, 1), random_seed=7,
        )
        bad = next(row for row in assignments if row["candidate"] == "bad")
        self.assertEqual(bad["value_on_V_full"], 0.0)
        self.assertGreater(len(random_values), 0)


if __name__ == "__main__":
    unittest.main()
