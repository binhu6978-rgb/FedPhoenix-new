from __future__ import annotations

import unittest

import torch

from fedrad.aggregation import weighted_fedavg
from fedrad.types import ClientUpdate, clone_state_dict, state_dict_hash


def make_update(client_id, count, value, running_mean, tracked, flag=4):
    state = {
        "weight": torch.tensor([value], dtype=torch.float32),
        "bn.running_mean": torch.tensor([running_mean], dtype=torch.float32),
        "bn.running_var": torch.tensor([running_mean + 1], dtype=torch.float32),
        "bn.num_batches_tracked": torch.tensor(tracked, dtype=torch.int64),
        "fixed_integer_buffer": torch.tensor(flag, dtype=torch.int32),
    }
    return ClientUpdate(
        client_id=client_id,
        task_id=client_id,
        initial_state_hash="initial",
        num_examples=count,
        mean_train_loss=0.0,
        local_seed=client_id,
        state_hash=state_dict_hash(state),
        _state=clone_state_dict(state),
    )


class AggregationTests(unittest.TestCase):
    def test_weighted_aggregation(self):
        result = weighted_fedavg(
            [
                make_update(0, 1, 1.0, 1.0, 7),
                make_update(1, 3, 3.0, 5.0, 2),
            ]
        )
        self.assertTrue(torch.allclose(result["weight"], torch.tensor([2.5])))
        self.assertTrue(
            torch.allclose(result["bn.running_mean"], torch.tensor([4.0]))
        )
        self.assertTrue(
            torch.allclose(result["bn.running_var"], torch.tensor([5.0]))
        )
        self.assertEqual(result["bn.num_batches_tracked"].item(), 7)
        self.assertEqual(result["fixed_integer_buffer"].item(), 4)

    def test_differing_unknown_integer_buffer_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported non-floating buffer"):
            weighted_fedavg(
                [
                    make_update(0, 1, 1.0, 1.0, 1, flag=4),
                    make_update(1, 1, 2.0, 2.0, 2, flag=5),
                ]
            )


if __name__ == "__main__":
    unittest.main()

