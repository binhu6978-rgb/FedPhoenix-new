from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from fedrad.rng import RNGStreams
from fedrad.trainer import CleanFedPhoenixTrainer, FedRADTrainer
from tests.helpers import initialized_tiny_model, tiny_config, tiny_data


class Phase39WarmupTests(unittest.TestCase):
    def test_warmup_is_exact_clean_replay_and_skips_probe(self):
        base = tiny_config(
            rounds=3,
            probe_support_size=3,
            probe_query_size=3,
            development_force_hungarian=True,
            verify_task_replay=False,
        )
        data = tiny_data()
        template = initialized_tiny_model(base.seed)
        clean = CleanFedPhoenixTrainer(
            config=base,
            data=data,
            model_template=template,
            rngs=RNGStreams(base.seed),
            device=torch.device("cpu"),
        ).run()

        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "rounds.jsonl"
            with reference_path.open("w", encoding="utf-8") as handle:
                for trace in clean.traces:
                    handle.write(
                        json.dumps(
                            {
                                "round_number": trace.round_idx + 1,
                                "selected_clients": list(trace.selected_clients),
                                "assignments": [list(pair) for pair in trace.assignments],
                                "task_seeds": list(trace.task_seeds),
                                "task_hashes": list(trace.task_hashes),
                                "local_seeds": list(trace.local_seeds),
                                "global_state_hash": trace.global_state_hash,
                            }
                        )
                        + "\n"
                    )
            delayed = replace(
                base,
                matching_start_round=3,
                warmup_reference_rounds_path=reference_path,
            )
            delayed.validate()
            result = FedRADTrainer(
                config=delayed,
                data=data,
                model_template=template,
                rngs=RNGStreams(delayed.seed),
                device=torch.device("cpu"),
            ).run()

        for clean_trace, delayed_trace in zip(clean.traces[:2], result.traces[:2]):
            for attribute in (
                "selected_clients",
                "assignments",
                "task_seeds",
                "task_hashes",
                "local_seeds",
                "global_state_hash",
            ):
                self.assertEqual(
                    getattr(clean_trace, attribute), getattr(delayed_trace, attribute)
                )
            self.assertEqual(delayed_trace.probe_support_hashes, ())
            self.assertEqual(delayed_trace.probe_query_hashes, ())
            self.assertEqual(delayed_trace.probe_seconds, 0.0)
            self.assertEqual(delayed_trace.fallback_reason, "matching_inactive")

        self.assertTrue(result.traces[2].probe_support_hashes)
        self.assertEqual(
            result.traces[2].fallback_reason, "development_force_hungarian"
        )

    def test_delayed_matching_requires_reference(self):
        config = tiny_config(rounds=3, matching_start_round=2)
        with self.assertRaisesRegex(ValueError, "warmup_reference_rounds_path"):
            config.validate()

    def test_delayed_functional_recovery_activates_only_at_configured_round(self):
        base = tiny_config(
            rounds=2,
            probe_support_size=3,
            probe_query_size=3,
            verify_task_replay=False,
        )
        data = tiny_data()
        template = initialized_tiny_model(base.seed)
        clean = CleanFedPhoenixTrainer(
            config=base,
            data=data,
            model_template=template,
            rngs=RNGStreams(base.seed),
            device=torch.device("cpu"),
        ).run()

        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "rounds.jsonl"
            with reference_path.open("w", encoding="utf-8") as handle:
                for trace in clean.traces:
                    handle.write(
                        json.dumps(
                            {
                                "round_number": trace.round_idx + 1,
                                "selected_clients": list(trace.selected_clients),
                                "assignments": [list(pair) for pair in trace.assignments],
                                "task_seeds": list(trace.task_seeds),
                                "task_hashes": list(trace.task_hashes),
                                "local_seeds": list(trace.local_seeds),
                                "global_state_hash": trace.global_state_hash,
                            }
                        )
                        + "\n"
                    )
            delayed = replace(
                base,
                matching_start_round=2,
                score_mode="functional",
                warmup_reference_rounds_path=reference_path,
            )
            delayed.validate()
            result = FedRADTrainer(
                config=delayed,
                data=data,
                model_template=template,
                rngs=RNGStreams(delayed.seed),
                device=torch.device("cpu"),
            ).run()

        self.assertEqual(result.traces[0].global_state_hash, clean.traces[0].global_state_hash)
        self.assertEqual(result.traces[0].fallback_reason, "matching_inactive")
        self.assertEqual(result.traces[0].probe_seconds, 0.0)
        self.assertEqual(
            result.traces[1].fallback_reason,
            "functional_recovery_hungarian",
        )
        self.assertTrue(result.traces[1].probe_support_hashes)


if __name__ == "__main__":
    unittest.main()
