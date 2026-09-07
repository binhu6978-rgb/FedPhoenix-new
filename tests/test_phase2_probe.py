from __future__ import annotations

from dataclasses import replace
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fedrad.local_trainer import LocalTrainer
from fedrad.probe import (
    ProbeRunner,
    _alignment_from_first_backward,
    materialize_probe_batch,
)
from fedrad.rng import RNGStreams
from fedrad.task_bank import build_task_bank
from fedrad.types import ResetSlice, TaskSpec, clone_state_dict, state_dict_hash
from tests.helpers import TinyCIFAR, initialized_tiny_model, tiny_config, tiny_data


class RecordingDataset(Dataset):
    def __init__(self, source: Dataset):
        self.source = source
        self.accesses: list[int] = []

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index: int):
        self.accesses.append(int(index))
        return self.source[index]


class Phase2ProbeTests(unittest.TestCase):
    def test_probe_protocol_64_32(self):
        config = tiny_config(rounds=1, verify_task_replay=False)
        self.assertEqual(config.probe_protocol, "support64_query32_steps1")
        dataset = TinyCIFAR(size=128, seed=901)
        model = initialized_tiny_model(config.seed)
        rngs = RNGStreams(config.seed)
        global_state = clone_state_dict(model.state_dict())
        bank = build_task_bank(
            config=config,
            model_template=model,
            global_state=global_state,
            round_idx=0,
            task_count=2,
            rngs=rngs,
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        batch = materialize_probe_batch(
            dataset=dataset,
            client_indices=tuple(range(128)),
            client_id=0,
            round_idx=0,
            probe_seed=rngs.probe_seed(0, 0),
            support_limit=config.probe_support_size,
            query_limit=config.probe_query_size,
        )
        self.assertEqual(len(batch.support_indices), 64)
        self.assertEqual(len(batch.query_indices), 32)
        self.assertFalse(set(batch.support_indices) & set(batch.query_indices))
        support_before = batch.support_images.clone()
        query_before = batch.query_images.clone()
        runner = ProbeRunner(
            config=config, model_template=model, device=torch.device("cpu")
        )
        global_loss = runner.global_reference(global_state=global_state, batch=batch)
        results = [
            runner.run_pair(task=task, batch=batch, global_loss=global_loss)
            for task in bank.tasks
        ]
        self.assertEqual({result.support_hash for result in results}, {batch.support_hash})
        self.assertEqual({result.query_hash for result in results}, {batch.query_hash})
        torch.testing.assert_close(batch.support_images, support_before)
        torch.testing.assert_close(batch.query_images, query_before)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def setUp(self):
        self.config = tiny_config(
            rounds=1,
            probe_support_size=3,
            probe_query_size=3,
            verify_task_replay=False,
        )
        self.data = tiny_data()
        self.model = initialized_tiny_model(self.config.seed)
        self.rngs = RNGStreams(self.config.seed)
        self.global_state = clone_state_dict(self.model.state_dict())
        self.bank = build_task_bank(
            config=self.config,
            model_template=self.model,
            global_state=self.global_state,
            round_idx=0,
            task_count=2,
            rngs=self.rngs,
        )
        self.batch = materialize_probe_batch(
            dataset=self.data.train_dataset,
            client_indices=self.data.partitions[0],
            client_id=0,
            round_idx=0,
            probe_seed=self.rngs.probe_seed(0, 0),
            support_limit=self.config.probe_support_size,
            query_limit=self.config.probe_query_size,
        )
        self.runner = ProbeRunner(
            config=self.config,
            model_template=self.model,
            device=torch.device("cpu"),
        )

    def _run_all_tasks(self):
        global_loss = self.runner.global_reference(
            global_state=self.global_state, batch=self.batch
        )
        return [
            self.runner.run_pair(task=task, batch=self.batch, global_loss=global_loss)
            for task in self.bank.tasks
        ]

    def test_probe_batch_reuse(self):
        results = self._run_all_tasks()
        self.assertEqual({result.support_hash for result in results}, {self.batch.support_hash})
        self.assertEqual({result.query_hash for result in results}, {self.batch.query_hash})
        self.assertEqual({result.probe_seed for result in results}, {self.batch.probe_seed})

    def test_probe_depth_only_repeats_adaptation_on_the_fixed_batch(self):
        task = self.bank.task(0)
        global_loss = self.runner.global_reference(
            global_state=self.global_state, batch=self.batch
        )
        task_hash = task.state_hash
        support_before = self.batch.support_images.clone()
        query_before = self.batch.query_images.clone()
        one_step = self.runner.run_pair(
            task=task, batch=self.batch, global_loss=global_loss
        )
        deeper = ProbeRunner(
            config=replace(self.config, probe_steps=3),
            model_template=self.model,
            device=torch.device("cpu"),
        ).run_pair(task=task, batch=self.batch, global_loss=global_loss)

        self.assertEqual(one_step.support_hash, deeper.support_hash)
        self.assertEqual(one_step.query_hash, deeper.query_hash)
        self.assertEqual(one_step.reset_loss, deeper.reset_loss)
        self.assertNotEqual(one_step.adapted_loss, deeper.adapted_loss)
        self.assertEqual(task.state_hash, task_hash)
        torch.testing.assert_close(self.batch.support_images, support_before)
        torch.testing.assert_close(self.batch.query_images, query_before)

    def test_probe_support_query_disjoint(self):
        self.assertFalse(set(self.batch.support_indices) & set(self.batch.query_indices))
        self.assertGreaterEqual(len(self.batch.support_indices), 1)
        self.assertGreaterEqual(len(self.batch.query_indices), 1)

    def test_probe_task_state_isolation(self):
        global_hash = state_dict_hash(self.global_state)
        task_hashes = tuple(task.state_hash for task in self.bank.tasks)
        self._run_all_tasks()
        self.assertEqual(state_dict_hash(self.global_state), global_hash)
        self.assertEqual(tuple(task.state_hash for task in self.bank.tasks), task_hashes)
        for task in self.bank.tasks:
            task.verify_hash()

    def test_probe_starts_from_original_task(self):
        results = self._run_all_tasks()
        self.assertEqual(
            [result.task_state_hash for result in results],
            [task.state_hash for task in self.bank.tasks],
        )

    def test_probe_no_formal_rng_pollution(self):
        task = self.bank.task(0)
        local_seed = self.rngs.local_seed(0, 0)
        baseline_dataset = RecordingDataset(self.data.train_dataset)
        baseline_trainer = LocalTrainer(
            config=self.config,
            model_template=self.model,
            train_dataset=baseline_dataset,
            device=torch.device("cpu"),
        )
        baseline_update = baseline_trainer.train_client(
            initial_state=task.clone_reset_state(),
            client_id=0,
            task_id=0,
            client_indices=self.data.partitions[0],
            round_idx=0,
            local_seed=local_seed,
        )
        baseline_order = tuple(baseline_dataset.accesses)

        probed_dataset = RecordingDataset(self.data.train_dataset)
        probe_batch = materialize_probe_batch(
            dataset=probed_dataset,
            client_indices=self.data.partitions[0],
            client_id=0,
            round_idx=0,
            probe_seed=self.rngs.probe_seed(0, 0),
            support_limit=3,
            query_limit=3,
        )
        loss = self.runner.global_reference(
            global_state=self.global_state, batch=probe_batch
        )
        self.runner.run_pair(task=task, batch=probe_batch, global_loss=loss)
        probed_dataset.accesses.clear()
        probed_trainer = LocalTrainer(
            config=self.config,
            model_template=self.model,
            train_dataset=probed_dataset,
            device=torch.device("cpu"),
        )
        probed_update = probed_trainer.train_client(
            initial_state=task.clone_reset_state(),
            client_id=0,
            task_id=0,
            client_indices=self.data.partitions[0],
            round_idx=0,
            local_seed=local_seed,
        )
        self.assertEqual(tuple(probed_dataset.accesses), baseline_order)
        self.assertEqual(probed_update.state_hash, baseline_update.state_hash)

    def test_alignment_direction(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        model.weight.grad = torch.tensor([[1.0, 0.0]])
        reset_state = clone_state_dict(model.state_dict())
        task = TaskSpec(
            round_idx=0,
            task_id=0,
            reset_seed=1,
            parent_state_hash="parent",
            state_hash=state_dict_hash(reset_state),
            omega=(ResetSlice("weight", (0,), (1, 2)),),
            delta_norm=1.0,
            reset_trace={},
            active=True,
            _reset_state=reset_state,
            _delta={"weight": torch.tensor([[-1.0, 0.0]])},
        )
        cosine, gradient_norm, delta_norm, valid = _alignment_from_first_backward(
            model, task, norm_eps=1e-12, device=torch.device("cpu")
        )
        self.assertTrue(valid)
        self.assertEqual(gradient_norm, 1.0)
        self.assertEqual(delta_norm, 1.0)
        self.assertAlmostEqual(cosine, 1.0)


if __name__ == "__main__":
    unittest.main()
