from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import unittest

import numpy as np
import torch

from fedrad.assignment import (
    assign_functional_recovery,
    best_vs_second_assignment_margin,
)
from fedrad.logger import NullLogger
from fedrad.rng import RNGStreams
from fedrad.scoring import build_score_matrices, mean_score_matrices
from fedrad.trainer import FedRADTrainer
from fedrad.types import ProbeResult
from tests.helpers import initialized_tiny_model, tiny_config, tiny_data


class _CaptureLogger(NullLogger):
    def __init__(self) -> None:
        self.probe_results: list[ProbeResult] = []

    def log_probe_results(self, results) -> None:
        self.probe_results.extend(results)


class MultiProbeFunctionalRecoveryTests(unittest.TestCase):
    def _run(self, replicates: int, reliability_mode: str = "mean"):
        config = tiny_config(
            rounds=1,
            probe_support_size=3,
            probe_query_size=3,
            score_mode="functional",
            functional_probe_replicates=replicates,
            functional_reliability_mode=reliability_mode,
            verify_task_replay=False,
        )
        logger = _CaptureLogger()
        result = FedRADTrainer(
            config=config,
            data=tiny_data(),
            model_template=initialized_tiny_model(config.seed),
            rngs=RNGStreams(config.seed),
            device=torch.device("cpu"),
            logger=logger,
        ).run()
        return config, result.traces[0], logger.probe_results

    @staticmethod
    def _replicate_scores(config, trace, results):
        grouped = defaultdict(list)
        for result in results:
            grouped[result.probe_replicate].append(result)
        return [
            build_score_matrices(
                selected_clients=trace.selected_clients,
                task_ids=tuple(range(config.clients_per_round)),
                results=grouped[replicate],
                config=config,
            )
            for replicate in sorted(grouped)
        ]

    def test_m1_exactly_uses_the_single_raw_functional_matrix(self):
        config, trace, results = self._run(1)
        scores = self._replicate_scores(config, trace, results)[0]
        expected = assign_functional_recovery(
            client_order=scores.client_order,
            task_order=scores.task_order,
            Q=scores.G,
        )
        np.testing.assert_array_equal(np.asarray(trace.Q), scores.G)
        self.assertEqual(
            trace.assignments,
            tuple((pair.client_id, pair.task_id) for pair in expected.final_pairs),
        )
        self.assertEqual(len(results), config.clients_per_round**2)

    def test_replicates_are_independent_but_each_batch_is_shared_across_states(self):
        config, trace, results = self._run(3)
        by_client_replicate = defaultdict(list)
        for result in results:
            by_client_replicate[(result.client_id, result.probe_replicate)].append(
                result
            )
        self.assertEqual(
            len(results),
            config.functional_probe_replicates * config.clients_per_round**2,
        )
        for values in by_client_replicate.values():
            self.assertEqual(len({value.support_hash for value in values}), 1)
            self.assertEqual(len({value.query_hash for value in values}), 1)
            self.assertEqual(len({value.probe_seed for value in values}), 1)
        for client_id in trace.selected_clients:
            batches = [
                (
                    by_client_replicate[(client_id, replicate)][0].support_hash,
                    by_client_replicate[(client_id, replicate)][0].query_hash,
                )
                for replicate in range(config.functional_probe_replicates)
            ]
            self.assertEqual(len(set(batches)), config.functional_probe_replicates)

    def test_mean_aggregation_drives_the_hungarian_assignment(self):
        config, trace, results = self._run(3)
        replicate_scores = self._replicate_scores(config, trace, results)
        mean_scores = mean_score_matrices(replicate_scores, config=config)
        expected_g = np.mean(
            np.stack([scores.G for scores in replicate_scores], axis=0), axis=0
        )
        np.testing.assert_allclose(mean_scores.G, expected_g, rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(trace.Q), expected_g, rtol=0, atol=0)
        decision = assign_functional_recovery(
            client_order=mean_scores.client_order,
            task_order=mean_scores.task_order,
            Q=mean_scores.G,
        )
        self.assertEqual(
            trace.assignments,
            tuple((pair.client_id, pair.task_id) for pair in decision.final_pairs),
        )
        self.assertEqual(len({task for _, task in trace.assignments}), 2)

    def test_two_probe_reliability_lcbs_have_closed_form(self):
        base = tiny_config(
            score_mode="functional",
            functional_probe_replicates=2,
        )
        first = self._synthetic_scores(base, np.asarray([[4.0, 1.0], [2.0, 5.0]]))
        second = self._synthetic_scores(base, np.asarray([[2.0, 3.0], [4.0, 1.0]]))

        half = mean_score_matrices(
            [first, second],
            config=replace(base, functional_reliability_mode="half_se_lcb"),
        )
        one = mean_score_matrices(
            [first, second],
            config=replace(base, functional_reliability_mode="one_se_lcb"),
        )
        mean = (first.G + second.G) / 2.0
        sample_se = np.abs(first.G - second.G) / 2.0
        np.testing.assert_allclose(half.Q, mean - 0.5 * sample_se)
        np.testing.assert_allclose(one.Q, np.minimum(first.G, second.G))

    def test_trainer_dispatches_with_reliability_adjusted_q(self):
        config, trace, results = self._run(2, "one_se_lcb")
        replicate_scores = self._replicate_scores(config, trace, results)
        expected_q = np.minimum(replicate_scores[0].G, replicate_scores[1].G)
        np.testing.assert_allclose(np.asarray(trace.Q), expected_q)
        decision = assign_functional_recovery(
            client_order=replicate_scores[0].client_order,
            task_order=replicate_scores[0].task_order,
            Q=expected_q,
        )
        self.assertEqual(
            trace.assignments,
            tuple((pair.client_id, pair.task_id) for pair in decision.final_pairs),
        )

    @staticmethod
    def _synthetic_scores(config, values):
        results = []
        for client_id in range(2):
            for task_id in range(2):
                value = float(values[client_id, task_id])
                results.append(
                    ProbeResult(
                        round_idx=0,
                        client_id=client_id,
                        task_id=task_id,
                        global_loss=0.0,
                        reset_loss=value,
                        adapted_loss=0.0,
                        G=value,
                        A=0.0,
                        D=0.0,
                        C=0.0,
                        gradient_norm=0.0,
                        delta_norm=0.0,
                        alignment_valid=True,
                        probe_seed=0,
                        support_hash="support",
                        query_hash="query",
                        task_state_hash="task",
                        probe_replicate=0,
                    )
                )
        return build_score_matrices(
            selected_clients=(0, 1),
            task_ids=(0, 1),
            results=results,
            config=config,
        )

    def test_extra_probes_do_not_change_client_or_formal_seed_streams(self):
        _, single, _ = self._run(1)
        _, multi, _ = self._run(3)
        self.assertEqual(single.selected_clients, multi.selected_clients)
        self.assertEqual(single.task_seeds, multi.task_seeds)
        self.assertEqual(single.local_seeds, multi.local_seeds)
        for (_, task_id), initial_hash in zip(
            multi.assignments, multi.formal_initial_hashes
        ):
            self.assertEqual(initial_hash, multi.task_hashes[task_id])

    def test_replicate_seed_namespace_and_configuration_validation(self):
        rngs = RNGStreams(1)
        seeds = [rngs.probe_replicate_seed(7, 13, replicate) for replicate in range(5)]
        self.assertEqual(len(set(seeds)), 5)
        self.assertNotIn(rngs.local_seed(7, 13), seeds)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            replace(tiny_config(), functional_probe_replicates=0).validate()
        with self.assertRaisesRegex(ValueError, "score_mode=functional"):
            replace(tiny_config(), functional_probe_replicates=2).validate()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            replace(
                tiny_config(score_mode="functional"),
                functional_probe_replicates=3,
                functional_reliability_mode="one_se_lcb",
            ).validate()

    def test_best_vs_second_assignment_margin(self):
        matrix = np.asarray([[10.0, 0.0], [0.0, 9.0]])
        self.assertAlmostEqual(best_vs_second_assignment_margin(matrix), 19.0)


if __name__ == "__main__":
    unittest.main()
