from __future__ import annotations

import copy
import unittest

import torch

from Algorithm.Phoenix_util import reset_kernels_for_task
from fedrad.rng import RNGStreams
from fedrad.task_bank import build_task_bank
from fedrad.types import clone_state_dict, state_dict_hash
from tests.helpers import initialized_tiny_model, tiny_config


class TaskBankTests(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.model = initialized_tiny_model(self.config.seed)
        self.global_state = clone_state_dict(self.model.state_dict())

    def _build(self):
        return build_task_bank(
            config=self.config,
            model_template=self.model,
            global_state=self.global_state,
            round_idx=3,
            task_count=2,
            rngs=RNGStreams(self.config.seed),
        )

    def test_same_task_replay(self):
        first = self._build()
        second = self._build()
        self.assertEqual(
            [task.state_hash for task in first.tasks],
            [task.state_hash for task in second.tasks],
        )
        for left, right in zip(first.tasks, second.tasks):
            for key in left.clone_reset_state():
                self.assertTrue(
                    torch.equal(
                        left.clone_reset_state()[key], right.clone_reset_state()[key]
                    )
                )

    def test_task_delta_support(self):
        task = self._build().tasks[0]
        reset_state = task.clone_reset_state()
        delta = task.clone_delta()
        omega_by_name = {item.parameter_name: item for item in task.omega}
        for key, global_tensor in self.global_state.items():
            if key not in omega_by_name:
                self.assertTrue(torch.equal(global_tensor, reset_state[key]), key)
                continue
            indices = list(omega_by_name[key].output_indices)
            expected = global_tensor[indices] - reset_state[key][indices]
            self.assertTrue(torch.equal(delta[key], expected), key)
            mask = torch.ones(global_tensor.shape[0], dtype=torch.bool)
            mask[indices] = False
            self.assertTrue(torch.equal(global_tensor[mask], reset_state[key][mask]))

    def test_global_unchanged_after_task_build(self):
        before = state_dict_hash(self.global_state)
        bank = self._build()
        self.assertEqual(before, state_dict_hash(self.global_state))
        self.assertEqual(before, bank.parent_state_hash)

        exposed_copy = bank.tasks[0].clone_reset_state()
        first_key = next(iter(exposed_copy))
        exposed_copy[first_key].zero_()
        bank.tasks[0].verify_hash()

    def test_reset_primitive_parity(self):
        bank = self._build()
        task = bank.tasks[0]
        direct_model = copy.deepcopy(self.model)
        direct_model.load_state_dict(self.global_state)
        direct_trace = reset_kernels_for_task(
            direct_model,
            reset_ratio=self.config.reset_ratio,
            seed=task.reset_seed,
            layer_scope="all",
            init_method=self.config.reset_method,
            at_least_one=False,
            current_iter=3,
            conv_transition_period=self.config.fp_conv_rounds,
        )
        self.assertEqual(task.state_hash, state_dict_hash(direct_model.state_dict()))
        self.assertEqual(dict(task.reset_trace), direct_trace)


if __name__ == "__main__":
    unittest.main()
