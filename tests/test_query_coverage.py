from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fedrad import probe
from fedrad.types import state_dict_hash
from tests import test_phase2_probe
from tests.helpers import TinyCIFAR


class QueryCoverageTests(unittest.TestCase):
    setUp = test_phase2_probe.Phase2ProbeTests.setUp

    def make_batch(self, count, batches):
        return probe.materialize_probe_batch(dataset=TinyCIFAR(size=count),
            client_indices=range(count), client_id=0, round_idx=0, probe_seed=42,
            support_limit=64, query_limit=32, query_batches=batches)

    def test_materialization_and_small_clients(self):
        for count in (2, 10, 96, 150, 600):
            original = self.make_batch(count, 1)
            for batches in (2, 4):
                batch = self.make_batch(count, batches)
                self.assertEqual(original.support_hash, batch.support_hash)
                self.assertEqual(original.query_hash, batch.query_batch_hashes[0])
                self.assertEqual(batch.query_hash, self.make_batch(count, batches).query_hash)
                for indices in batch.query_batch_indices:
                    self.assertEqual(len(indices), len(original.query_indices))
                    self.assertEqual(len(indices), len(set(indices)))
                    self.assertFalse(set(indices) & set(batch.support_indices))
                unique = set(i for group in batch.query_batch_indices for i in group)
                self.assertEqual(len(unique), min(count-len(batch.support_indices), batches*len(original.query_indices)))

    def test_average_matches_separate_queries_and_adaptation_is_identical(self):
        batch = self.make_batch(600, 4)
        config = replace(self.config, probe_steps=5, score_mode="functional", functional_probe_replicates=2)
        snapshots = []
        original_query = probe._query_loss

        def record(model, images, labels):
            snapshots.append((model.training, state_dict_hash(model.state_dict())))
            return original_query(model, images, labels)

        def run(config, batch):
            runner = probe.ProbeRunner(config=config, model_template=self.model, device=torch.device("cpu"))
            return runner.run_pair(task=self.bank.task(0), batch=batch, global_loss=1.0)

        with patch.object(probe, "_query_loss", side_effect=record):
            multi = run(replace(config, probe_query_batches=4), batch)
        multi_states = snapshots.copy()
        snapshots.clear()
        results = []
        with patch.object(probe, "_query_loss", side_effect=record):
            for indices, images, labels in zip(batch.query_batch_indices, batch.query_batch_images, batch.query_batch_labels):
                results.append(run(config, replace(batch, query_indices=indices, query_images=images, query_labels=labels)))
        self.assertEqual(len(multi_states), 8)
        self.assertEqual(multi_states[:4], [snapshots[0]]*4)
        self.assertEqual(multi_states[4:], [snapshots[1]]*4)
        for index in range(0, 8, 2):
            self.assertEqual(snapshots[index:index+2], snapshots[:2])
        self.assertAlmostEqual(multi.G, float(np.mean([r.G for r in results])), places=7)
        self.bank.task(0).verify_hash()
