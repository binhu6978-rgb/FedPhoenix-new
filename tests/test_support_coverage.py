from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fedrad.probe import ProbeRunner, materialize_probe_batch
from tests import test_phase2_probe
from tests.helpers import TinyCIFAR


class SupportCoverageTests(unittest.TestCase):
    setUp = test_phase2_probe.Phase2ProbeTests.setUp

    def make_batch(self, count, mode, seed=42):
        return materialize_probe_batch(
            dataset=TinyCIFAR(size=count), client_indices=range(count), client_id=0,
            round_idx=0, probe_seed=seed, support_limit=64, query_limit=32,
            support_coverage=mode,
        )

    def test_coverage_preserves_query_first_batch_and_rng(self):
        for count in (2, 10, 96, 200, 600):
            fixed = self.make_batch(count, "fixed")
            for mode in ("fresh", "refresh_once"):
                py_rng = random.getstate()
                np_rng = np.random.get_state()
                torch_rng = torch.random.get_rng_state().clone()
                batch = self.make_batch(count, mode)
                repeat = self.make_batch(count, mode)
                self.assertEqual(py_rng, random.getstate())
                np.testing.assert_equal(np_rng, np.random.get_state())
                self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
                self.assertEqual(batch.query_hash, fixed.query_hash)
                self.assertEqual(batch.support_batch_hashes[0], fixed.support_hash)
                self.assertEqual(batch.support_hash, repeat.support_hash)
                for indices in batch.support_batch_indices:
                    self.assertEqual(len(indices), len(fixed.support_indices))
                    self.assertEqual(len(set(indices)), len(indices))
                    self.assertFalse(set(indices) & set(batch.query_indices))
                unique = set(i for indices in batch.support_batch_indices for i in indices)
                expected = min(len(batch.support_batch_indices) * len(fixed.support_indices), count-len(batch.query_indices))
                self.assertEqual(len(unique), expected)
        self.assertNotEqual(self.make_batch(600, "fresh", 42).support_hash, self.make_batch(600, "fresh", 43).support_hash)

    def test_runner_consumes_expected_schedule_and_preserves_tasks(self):
        for mode, expected in (("fresh", (0,1,2,3,4)), ("refresh_once", (0,0,0,1,1))):
            batch = self.make_batch(600, mode)
            config = replace(self.config, probe_steps=5, probe_support_coverage=mode,
                             score_mode="functional", functional_probe_replicates=2)
            config.validate()
            runner = ProbeRunner(config=config, model_template=self.model, device=torch.device("cpu"))
            task = self.bank.task(0)
            seen = []
            forward = type(self.model).forward

            def record(model, images):
                if model.training and len(images) == 64:
                    seen.append(images.detach().clone())
                return forward(model, images)

            with patch.object(type(self.model), "forward", record):
                result = runner.run_pair(task=task, batch=batch, global_loss=1.0)
            self.assertEqual(batch.support_step_batch_ids, expected)
            self.assertEqual(len(seen), 5)
            for image, index in zip(seen, expected):
                torch.testing.assert_close(image, batch.support_batch_images[index], rtol=0, atol=0)
            task.verify_hash()
            self.assertEqual(result.G, result.reset_loss-result.adapted_loss)
