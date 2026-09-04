from __future__ import annotations

import unittest

import torch

from fedrad.evaluation import Evaluator
from fedrad.local_trainer import LocalTrainer
from fedrad.models import build_model, extract_logits
from fedrad.rng import RNGStreams
from fedrad.task_bank import build_task_bank
from fedrad.types import clone_state_dict
from tests.helpers import initialized_tiny_model, tiny_config, tiny_data


class ModelAndLocalTests(unittest.TestCase):
    def test_project_resnet_forward_contract(self):
        config = tiny_config()
        model = build_model(config, initialization_seed=config.seed)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(extract_logits(output).shape), (2, 10))

    def test_local_training_starts_from_original_task(self):
        config = tiny_config(rounds=1)
        data = tiny_data()
        model = initialized_tiny_model(config.seed)
        rngs = RNGStreams(config.seed)
        bank = build_task_bank(
            config=config,
            model_template=model,
            global_state=clone_state_dict(model.state_dict()),
            round_idx=0,
            task_count=2,
            rngs=rngs,
        )
        trainer = LocalTrainer(
            config=config,
            model_template=model,
            train_dataset=data.train_dataset,
            device=torch.device("cpu"),
        )
        task = bank.tasks[1]
        seed = rngs.local_seed(0, 2)
        update = trainer.train_client(
            initial_state=task.clone_reset_state(),
            client_id=2,
            task_id=1,
            client_indices=data.partitions[2],
            round_idx=0,
            local_seed=seed,
        )
        self.assertEqual(update.initial_state_hash, task.state_hash)
        self.assertEqual(update.local_seed, seed)
        task.verify_hash()

    def test_evaluation_changes_neither_input_state_nor_training_rng(self):
        config = tiny_config(rounds=1)
        data = tiny_data()
        model = initialized_tiny_model(config.seed)
        state = clone_state_dict(model.state_dict())
        state_before = {key: value.clone() for key, value in state.items()}
        torch.manual_seed(991)
        rng_before = torch.random.get_rng_state().clone()
        evaluator = Evaluator(
            config=config,
            model_template=model,
            dataset=data.test_dataset,
            device=torch.device("cpu"),
            evaluation_seed=12345,
        )
        result = evaluator.evaluate(state)
        self.assertEqual(result.num_examples, len(data.test_dataset))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for key in state:
            self.assertTrue(torch.equal(state[key], state_before[key]), key)


if __name__ == "__main__":
    unittest.main()
