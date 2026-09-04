from __future__ import annotations

import unittest

import numpy as np

from fedrad.assignment import assign_with_gate
from fedrad.scoring import ScoreMatrices, build_score_matrices, matrix_zscore
from fedrad.types import ProbeResult
from tests.helpers import tiny_config


def _result(client: int, task: int, G: float, A: float, D: float, C: float):
    return ProbeResult(
        round_idx=0, client_id=client, task_id=task, task_state_hash=f"t{task}",
        support_hash=f"s{client}", query_hash=f"q{client}", global_loss=2.0,
        reset_loss=2.0, adapted_loss=1.0, G=G, A=A, D=D, C=C,
        gradient_norm=1.0, delta_norm=1.0, alignment_valid=True,
        probe_seed=100 + client,
    )


def _score_bundle(Q: np.ndarray) -> ScoreMatrices:
    Q = np.asarray(Q, dtype=np.float64)
    zeros = np.zeros_like(Q)
    size = Q.shape[0]
    return ScoreMatrices(
        client_order=tuple(range(10, 10 + size)),
        task_order=tuple(range(size)),
        G=zeros, A=zeros, D=zeros, C=zeros,
        ZG=zeros, ZA=zeros, ZD=zeros, ZC=zeros, Q=Q,
        component_degenerate=(True, True, True, True),
    )


class Phase2ScoringAssignmentTests(unittest.TestCase):
    def test_score_formula(self):
        config = tiny_config(
            lambda_G=1.0, lambda_A=2.0, lambda_D=3.0, lambda_C=0.5
        )
        values = [
            _result(7, 0, 0, 4, 1, -1), _result(7, 1, 1, 3, 0, 0),
            _result(3, 0, 2, 2, 0, 1), _result(3, 1, 5, 1, 2, 2),
        ]
        scores = build_score_matrices(
            selected_clients=(7, 3), task_ids=(0, 1), results=values, config=config
        )
        expected = (
            config.lambda_G * scores.ZG + config.lambda_A * scores.ZA
            - config.lambda_D * scores.ZD + config.lambda_C * scores.ZC
        )
        np.testing.assert_allclose(scores.Q, expected, rtol=0, atol=0)

    def test_g_only_score_is_exactly_zg(self):
        config = tiny_config(
            score_mode="g_only",
            lambda_G=7.0,
            lambda_A=11.0,
            lambda_D=13.0,
            lambda_C=17.0,
        )
        values = [
            _result(7, 0, 0, 40, 10, -10), _result(7, 1, 1, 30, 0, 0),
            _result(3, 0, 2, 20, 0, 10), _result(3, 1, 5, 10, 20, 20),
        ]
        scores = build_score_matrices(
            selected_clients=(7, 3), task_ids=(0, 1), results=values, config=config
        )
        np.testing.assert_allclose(scores.Q, scores.ZG, rtol=0, atol=0)

    def test_matrix_level_normalization(self):
        matrix = np.asarray([[0.0, 1.0], [2.0, 7.0]])
        normalized, degenerate = matrix_zscore(
            matrix, z_eps=1e-12, std_atol=1e-12, std_rtol=1e-7
        )
        self.assertFalse(degenerate)
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(normalized.std(ddof=0)), 1.0, places=10)
        self.assertFalse(np.allclose(normalized.mean(axis=1), 0.0))
        self.assertFalse(np.allclose(normalized.mean(axis=0), 0.0))

    def test_degenerate_component(self):
        normalized, degenerate = matrix_zscore(
            np.full((3, 3), 4.0), z_eps=1e-12, std_atol=1e-12, std_rtol=1e-7
        )
        self.assertTrue(degenerate)
        np.testing.assert_array_equal(normalized, np.zeros((3, 3)))
        self.assertTrue(np.isfinite(normalized).all())

    def test_hungarian_bijection(self):
        decision = assign_with_gate(_score_bundle([[1, 9, 2], [8, 2, 3], [4, 5, 7]]), gate_tau=0.1)
        self.assertEqual(len({p.client_id for p in decision.hungarian_pairs}), 3)
        self.assertEqual(len({p.task_id for p in decision.hungarian_pairs}), 3)
        self.assertEqual([p.client_row for p in decision.hungarian_pairs], [0, 1, 2])

    def test_hungarian_optimality(self):
        decision = assign_with_gate(_score_bundle([[1, 9, 2], [8, 2, 3], [4, 5, 7]]), gate_tau=0.1)
        self.assertEqual([p.task_id for p in decision.hungarian_pairs], [1, 0, 2])
        self.assertAlmostEqual(decision.hungarian_score, 24.0)

    def test_gate_formula(self):
        decision = assign_with_gate(_score_bundle([[-100, 5], [5, 0]]), gate_tau=4.0)
        self.assertAlmostEqual(decision.hungarian_score, 10.0)
        self.assertAlmostEqual(decision.baseline_score, -100.0)
        self.assertAlmostEqual(decision.gamma, 5.0)
        self.assertTrue(decision.gate_passed)

    def test_gate_fallback(self):
        decision = assign_with_gate(_score_bundle([[-100, 5], [5, 0]]), gate_tau=6.0)
        self.assertFalse(decision.gate_passed)
        self.assertEqual(decision.final_pairs, decision.baseline_pairs)
        self.assertEqual(decision.fallback_reason, "gamma_below_tau")


if __name__ == "__main__":
    unittest.main()
