from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_phase36_averaging import averaged_candidate_matrix


class Phase36AveragingTests(unittest.TestCase):
    def test_average_is_applied_before_component_zscore(self):
        left = {
            "G": np.asarray([[0.0, 2.0], [4.0, 6.0]]),
            "A": np.asarray([[1.0, 4.0], [2.0, 8.0]]),
            "D": np.asarray([[0.0, 0.0], [2.0, 4.0]]),
            "C": np.asarray([[3.0, 1.0], [0.0, 2.0]]),
        }
        right = {
            "G": np.asarray([[6.0, 2.0], [1.0, 0.0]]),
            "A": np.asarray([[0.0, 3.0], [7.0, 5.0]]),
            "D": np.asarray([[1.0, 5.0], [0.0, 1.0]]),
            "C": np.asarray([[0.0, 2.0], [4.0, 8.0]]),
        }
        options = {"z_eps": 1e-12, "std_atol": 1e-12, "std_rtol": 1e-7}
        actual = averaged_candidate_matrix(
            left, right, candidate="Full", **options
        )

        def z(name: str) -> np.ndarray:
            raw = 0.5 * (left[name] + right[name])
            return (raw - raw.mean()) / (raw.std(ddof=0) + options["z_eps"])

        expected = z("G") + z("A") - z("D") + 0.25 * z("C")
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_g_only_ignores_other_components(self):
        left = {"G": np.asarray([[0.0, 1.0], [2.0, 4.0]])}
        right = {"G": np.asarray([[4.0, 2.0], [1.0, 0.0]])}
        actual = averaged_candidate_matrix(
            left,
            right,
            candidate="G",
            z_eps=1e-12,
            std_atol=1e-12,
            std_rtol=1e-7,
        )
        self.assertAlmostEqual(float(actual.mean()), 0.0)
        self.assertAlmostEqual(float(actual.std(ddof=0)), 1.0)


if __name__ == "__main__":
    unittest.main()
