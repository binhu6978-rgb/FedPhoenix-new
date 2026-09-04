from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from fedrad.rng import RNGStreams
from fedrad.trainer import CleanFedPhoenixTrainer, FedRADTrainer
from tests.helpers import initialized_tiny_model, tiny_config, tiny_data


class Phase2TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tiny_config(
            rounds=2,
            probe_support_size=3,
            probe_query_size=3,
            development_force_hungarian=True,
            verify_task_replay=False,
        )
        data = tiny_data()
        template = initialized_tiny_model(cls.config.seed)

        def run_once():
            return FedRADTrainer(
                config=cls.config,
                data=data,
                model_template=template,
                rngs=RNGStreams(cls.config.seed),
                device=torch.device("cpu"),
            ).run()

        cls.first = run_once()
        cls.second = run_once()

    def test_probe_state_never_enters_local_training(self):
        for trace in self.first.traces:
            task_hash_by_id = dict(enumerate(trace.task_hashes))
            for (_, task_id), initial_hash in zip(
                trace.assignments, trace.formal_initial_hashes
            ):
                self.assertEqual(initial_hash, task_hash_by_id[task_id])

    def test_fedrad_replay(self):
        self.assertEqual(self.first.initial_state_hash, self.second.initial_state_hash)
        self.assertEqual(self.first.final_state_hash, self.second.final_state_hash)
        attributes = (
            "selected_clients", "task_seeds", "task_hashes",
            "probe_support_hashes", "probe_query_hashes", "G", "A", "D", "C",
            "Q", "hungarian_assignments", "gamma", "assignments", "local_seeds",
            "global_state_hash",
        )
        for left, right in zip(self.first.traces, self.second.traces):
            for attribute in attributes:
                self.assertEqual(getattr(left, attribute), getattr(right, attribute))

    def test_probe_pipeline_preserves_baseline_when_gate_falls_back(self):
        config = replace(
            self.config,
            gate_tau=1_000_000.0,
            development_force_hungarian=False,
        )
        data = tiny_data()
        template = initialized_tiny_model(config.seed)
        clean = CleanFedPhoenixTrainer(
            config=config,
            data=data,
            model_template=template,
            rngs=RNGStreams(config.seed),
            device=torch.device("cpu"),
        ).run()
        probed = FedRADTrainer(
            config=config,
            data=data,
            model_template=template,
            rngs=RNGStreams(config.seed),
            device=torch.device("cpu"),
        ).run()
        self.assertEqual(clean.final_state_hash, probed.final_state_hash)
        for clean_trace, probed_trace in zip(clean.traces, probed.traces):
            for attribute in (
                "selected_clients", "assignments", "task_seeds", "task_hashes",
                "local_seeds", "global_state_hash",
            ):
                self.assertEqual(
                    getattr(clean_trace, attribute), getattr(probed_trace, attribute)
                )
            self.assertEqual(probed_trace.fallback_reason, "gamma_below_tau")


if __name__ == "__main__":
    unittest.main()
