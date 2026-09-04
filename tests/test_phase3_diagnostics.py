from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import unittest

import numpy as np
import torch

from fedrad.diagnostics import (
    assignment_overlap,
    assignment_score,
    component_contribution_assignments,
    double_center,
    interaction_metrics,
    maximum_assignment,
    null_hungarian_gammas,
    random_assignment_scores,
    second_best_assignment,
)
from fedrad.rng import RNGStreams
from fedrad.trainer import FedRADTrainer
from tests.helpers import initialized_tiny_model, tiny_config, tiny_data


class Phase3DiagnosticsTests(unittest.TestCase):
    def test_second_best_assignment_matches_exhaustive_search(self):
        matrix = np.asarray(
            [[7.0, 2.0, 4.0], [3.0, 8.0, 5.0], [6.0, 1.0, 9.0]]
        )
        ranked = sorted(
            (
                (assignment_score(matrix, assignment), assignment)
                for assignment in permutations(range(3))
            ),
            reverse=True,
        )
        best, best_score, second, second_score = second_best_assignment(matrix)
        self.assertEqual(best, ranked[0][1])
        self.assertEqual(second, ranked[1][1])
        self.assertEqual(best_score, ranked[0][0])
        self.assertEqual(second_score, ranked[1][0])

    def test_square_assignment_additive_invariance(self):
        rng = np.random.default_rng(881)
        for _ in range(20):
            Q = rng.normal(size=(7, 7))
            row_offsets = rng.normal(size=7)
            column_offsets = rng.normal(size=7)
            shifted = Q + row_offsets[:, None] + column_offsets[None, :]
            original_assignment, original_score = maximum_assignment(Q)
            shifted_assignment, shifted_score = maximum_assignment(shifted)
            self.assertEqual(original_assignment, shifted_assignment)
            self.assertNotEqual(original_score, shifted_score)

    def test_double_center_removes_additive_effects(self):
        interaction = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
        row = np.asarray([10.0, -4.0])[:, None]
        column = np.asarray([7.0, -3.0])[None, :]
        centered = double_center(interaction + row + column)
        np.testing.assert_allclose(centered, interaction, atol=1e-12, rtol=0)
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-12)

    def test_interaction_metrics_separate_main_effects(self):
        values = np.asarray([[10.0, 11.0], [-5.0, -4.0]])
        metrics = interaction_metrics(values)
        self.assertGreater(metrics["overall_std"], 0.0)
        self.assertAlmostEqual(metrics["interaction_std"], 0.0)
        self.assertAlmostEqual(metrics["interaction_overall_ratio"], 0.0)

    def test_assignment_sampling_diagnostics_are_deterministic(self):
        Q = np.asarray([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [1.0, 2.0, 4.0]])
        random_left = random_assignment_scores(Q, samples=50, seed=91)
        random_right = random_assignment_scores(Q, samples=50, seed=91)
        null_left = null_hungarian_gammas(Q, samples=50, seed=92)
        null_right = null_hungarian_gammas(Q, samples=50, seed=92)
        np.testing.assert_array_equal(random_left, random_right)
        np.testing.assert_array_equal(null_left, null_right)
        self.assertTrue(np.isfinite(random_left).all())
        self.assertTrue(np.isfinite(null_left).all())

    def test_component_contribution_overlap(self):
        zeros = np.zeros((2, 2), dtype=np.float64)
        ZG = np.asarray([[2.0, -2.0], [-2.0, 2.0]])
        Q = ZG.copy()
        report = component_contribution_assignments(
            Q=Q,
            normalized={"G": ZG, "A": zeros, "D": zeros, "C": zeros},
            weights={"G": 1.0, "A": 1.0, "D": 1.0, "C": 0.25},
        )
        self.assertEqual(report["without_G"]["overlap_with_full"], 1.0)
        full, _ = maximum_assignment(Q)
        self.assertEqual(assignment_overlap(full, full), 1.0)

    def test_probe_replicates_do_not_change_formal_trajectory(self):
        base = tiny_config(
            rounds=2,
            probe_support_size=3,
            probe_query_size=3,
            development_force_hungarian=True,
            verify_task_replay=False,
        )
        diagnostic = replace(
            base,
            diagnostic_probe_rounds=(1, 2),
            diagnostic_probe_replicates=3,
        )
        data = tiny_data()
        template = initialized_tiny_model(base.seed)

        def run(config):
            return FedRADTrainer(
                config=config,
                data=data,
                model_template=template,
                rngs=RNGStreams(config.seed),
                device=torch.device("cpu"),
            ).run()

        ordinary = run(base)
        with_replicates = run(diagnostic)
        self.assertEqual(ordinary.final_state_hash, with_replicates.final_state_hash)
        fields = (
            "selected_clients", "assignments", "task_seeds", "task_hashes",
            "local_seeds", "global_state_hash", "G", "A", "D", "C", "Q",
            "hungarian_assignments", "gamma",
        )
        for left, right in zip(ordinary.traces, with_replicates.traces):
            for field in fields:
                self.assertEqual(getattr(left, field), getattr(right, field))

    def test_probe_replicate_seed_is_task_independent(self):
        rngs = RNGStreams(1)
        seeds = [rngs.probe_replicate_seed(7, 12, replicate) for replicate in range(3)]
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(seeds[0], rngs.probe_seed(7, 12))


if __name__ == "__main__":
    unittest.main()
