import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn

from fedrad import probe
from fedrad.types import state_dict_hash
from tests import test_phase2_probe


class QueryModeTests(unittest.TestCase):
    setUp = test_phase2_probe.Phase2ProbeTests.setUp
    def test_private_query_preserves_bn_rng_and_mode(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(4)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                return {"output": self.dropout(self.bn(x))}

        model = Model().eval()
        images = torch.randn(8, 4)
        labels = torch.arange(8) % 4
        before = state_dict_hash(model.state_dict())
        rng = torch.random.get_rng_state().clone()
        for training in (True, False):
            actual = probe._isolated_query_loss(
                model, images, labels, training=training, device=torch.device("cpu")
            )
            with torch.random.fork_rng():
                expected = probe._query_loss(copy.deepcopy(model).train(training), images, labels)
            self.assertEqual(actual, expected)
            self.assertEqual(before, state_dict_hash(model.state_dict()))
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertFalse(model.training)

    def test_modes_do_not_change_support_adaptation(self):
        snapshots = {}
        for mode in ("legacy", "eval_eval", "train_train"):
            seen = []
            query_loss = probe._query_loss

            def record(model, images, labels):
                seen.append((model.training, state_dict_hash(model.state_dict())))
                return query_loss(model, images, labels)

            runner = probe.ProbeRunner(
                config=replace(self.config, probe_steps=5, score_mode="functional", probe_query_mode=mode),
                model_template=self.model, device=torch.device("cpu"),
            )
            with patch.object(probe, "_query_loss", side_effect=record):
                result = runner.run_pair(task=self.bank.task(0), batch=self.batch, global_loss=1.0)
            self.assertEqual(result.G, result.reset_loss - result.adapted_loss)
            snapshots[mode] = seen
        self.assertEqual([item[0] for item in snapshots["legacy"]], [False, True])
        self.assertEqual([item[0] for item in snapshots["eval_eval"]], [False, False])
        self.assertEqual([item[0] for item in snapshots["train_train"]], [True, True])
        for mode in ("eval_eval", "train_train"):
            self.assertEqual([item[1] for item in snapshots[mode]], [item[1] for item in snapshots["legacy"]])
