from __future__ import annotations

import unittest

import torch

from fedrad.rng import RNGStreams
from fedrad.trainer import CleanFedPhoenixTrainer
from tests.helpers import initialized_tiny_model, tiny_config, tiny_data


class SeedReplayTests(unittest.TestCase):
    def test_seed_replay(self):
        config = tiny_config(rounds=2)
        data = tiny_data()
        template = initialized_tiny_model(config.seed)

        def run_once():
            trainer = CleanFedPhoenixTrainer(
                config=config,
                data=data,
                model_template=template,
                rngs=RNGStreams(config.seed),
                device=torch.device("cpu"),
            )
            return trainer.run()

        first = run_once()
        second = run_once()
        self.assertEqual(first.initial_state_hash, second.initial_state_hash)
        self.assertEqual(first.final_state_hash, second.final_state_hash)
        for left, right in zip(first.traces, second.traces):
            self.assertEqual(left.selected_clients, right.selected_clients)
            self.assertEqual(left.task_seeds, right.task_seeds)
            self.assertEqual(left.task_hashes, right.task_hashes)
            self.assertEqual(left.local_seeds, right.local_seeds)
            self.assertEqual(left.global_state_hash, right.global_state_hash)


if __name__ == "__main__":
    unittest.main()

