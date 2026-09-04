from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from fedrad.config import FedRADConfig
from fedrad.scoring import ScoreMatrices
from fedrad.task_bank import TaskBank
from fedrad.types import (
    AssignmentDecision,
    ProbeResult,
    RoundTrace,
    clone_state_dict,
)


class NullLogger:
    run_dir: Path | None = None

    def log_config(self, _: Mapping[str, Any]) -> None:
        pass

    def log_task_bank(self, _: TaskBank) -> None:
        pass

    def log_probe_results(self, _: Sequence[ProbeResult]) -> None:
        pass

    def log_scores(self, _: int, __: ScoreMatrices) -> None:
        pass

    def log_assignment(self, _: int, __: AssignmentDecision) -> None:
        pass

    def log_probe_reliability_results(
        self, _: int, __: Sequence[ProbeResult]
    ) -> None:
        pass

    def log_reliability_scores(
        self, _: int, __: int, ___: ScoreMatrices
    ) -> None:
        pass

    def log_reliability_assignment(
        self, _: int, __: int, ___: AssignmentDecision
    ) -> None:
        pass

    def save_diagnostic_checkpoint(
        self, _: int, __: Mapping[str, torch.Tensor]
    ) -> None:
        pass

    def log_round(self, _: RoundTrace, **__: Any) -> None:
        pass

    def write_summary(self, _: Mapping[str, Any]) -> None:
        pass


class RunLogger:
    _PROBE_FIELDS = (
        "round", "client_id", "task_id", "global_loss", "reset_loss",
        "adapted_loss", "G", "A", "D", "C", "gradient_norm",
        "delta_norm", "alignment_valid", "probe_seed", "support_hash",
        "query_hash", "task_hash", "probe_replicate",
    )

    def __init__(
        self,
        config: FedRADConfig,
        run_name: str | None = None,
        *,
        algorithm: str = "clean_fedphoenix",
    ):
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = "" if not run_name else "_" + "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in run_name
        )
        safe_algorithm = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in algorithm.lower()
        )
        self.run_dir = Path(config.output_root).resolve() / (
            f"{safe_algorithm}_seed{config.seed}_{timestamp}{safe_name}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.rounds_path = self.run_dir / "rounds.jsonl"
        self.tasks_path = self.run_dir / "tasks.jsonl"
        self.probe_pairs_path = self.run_dir / "probe_pairs.csv"
        self.assignments_path = self.run_dir / "assignments.jsonl"
        self.scores_dir = self.run_dir / "scores"
        self.reliability_pairs_path = self.run_dir / "probe_reliability_pairs.csv"
        self.reliability_assignments_path = (
            self.run_dir / "probe_reliability_assignments.jsonl"
        )
        self.reliability_scores_dir = self.run_dir / "reliability_scores"
        self.checkpoints_dir = self.run_dir / "diagnostic_checkpoints"

    @staticmethod
    def _append(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()

    def log_config(self, payload: Mapping[str, Any]) -> None:
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)

    def log_task_bank(self, bank: TaskBank) -> None:
        for task in bank.tasks:
            self._append(
                self.tasks_path,
                {
                    "round_idx": task.round_idx,
                    "task_id": task.task_id,
                    "reset_seed": task.reset_seed,
                    "parent_state_hash": task.parent_state_hash,
                    "state_hash": task.state_hash,
                    "active": task.active,
                    "delta_norm": task.delta_norm,
                    "omega": [
                        {
                            "parameter_name": item.parameter_name,
                            "output_indices": list(item.output_indices),
                            "parameter_shape": list(item.parameter_shape),
                        }
                        for item in task.omega
                    ],
                    "reset_trace": task.reset_trace,
                },
            )

    def _write_probe_results(
        self, path: Path, results: Sequence[ProbeResult]
    ) -> None:
        if not results:
            return
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._PROBE_FIELDS)
            if write_header:
                writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "round": result.round_idx + 1,
                        "client_id": result.client_id,
                        "task_id": result.task_id,
                        "global_loss": result.global_loss,
                        "reset_loss": result.reset_loss,
                        "adapted_loss": result.adapted_loss,
                        "G": result.G,
                        "A": result.A,
                        "D": result.D,
                        "C": result.C,
                        "gradient_norm": result.gradient_norm,
                        "delta_norm": result.delta_norm,
                        "alignment_valid": result.alignment_valid,
                        "probe_seed": result.probe_seed,
                        "support_hash": result.support_hash,
                        "query_hash": result.query_hash,
                        "task_hash": result.task_state_hash,
                        "probe_replicate": result.probe_replicate,
                    }
                )
            handle.flush()

    def log_probe_results(self, results: Sequence[ProbeResult]) -> None:
        self._write_probe_results(self.probe_pairs_path, results)

    def log_probe_reliability_results(
        self, replicate: int, results: Sequence[ProbeResult]
    ) -> None:
        if any(result.probe_replicate != replicate for result in results):
            raise ValueError("Reliability probe replicate metadata mismatch")
        self._write_probe_results(self.reliability_pairs_path, results)

    def log_scores(self, round_idx: int, scores: ScoreMatrices) -> None:
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.scores_dir / f"round_{round_idx + 1:04d}.npz",
            client_order=np.asarray(scores.client_order, dtype=np.int64),
            task_order=np.asarray(scores.task_order, dtype=np.int64),
            G=scores.G, A=scores.A, D=scores.D, C=scores.C,
            ZG=scores.ZG, ZA=scores.ZA, ZD=scores.ZD, ZC=scores.ZC, Q=scores.Q,
            component_names=np.asarray(("G", "A", "D", "C")),
            component_degenerate=np.asarray(scores.component_degenerate, dtype=np.bool_),
        )

    def log_reliability_scores(
        self, round_idx: int, replicate: int, scores: ScoreMatrices
    ) -> None:
        self.reliability_scores_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.reliability_scores_dir
            / f"round_{round_idx + 1:04d}_rep_{replicate}.npz",
            client_order=np.asarray(scores.client_order, dtype=np.int64),
            task_order=np.asarray(scores.task_order, dtype=np.int64),
            G=scores.G, A=scores.A, D=scores.D, C=scores.C,
            ZG=scores.ZG, ZA=scores.ZA, ZD=scores.ZD, ZC=scores.ZC, Q=scores.Q,
            component_names=np.asarray(("G", "A", "D", "C")),
            component_degenerate=np.asarray(scores.component_degenerate, dtype=np.bool_),
        )

    @staticmethod
    def _pairs(pairs: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {
                "client_id": pair.client_id,
                "client_row": pair.client_row,
                "task_id": pair.task_id,
                "score": pair.score,
            }
            for pair in pairs
        ]

    def log_assignment(self, round_idx: int, decision: AssignmentDecision) -> None:
        self._append(
            self.assignments_path,
            {
                "round_idx": round_idx,
                "round_number": round_idx + 1,
                "baseline_pairs": self._pairs(decision.baseline_pairs),
                "hungarian_pairs": self._pairs(decision.hungarian_pairs),
                "final_pairs": self._pairs(decision.final_pairs),
                "hungarian_score": decision.hungarian_score,
                "baseline_score": decision.baseline_score,
                "Gamma": decision.gamma,
                "gate_tau": decision.gate_tau,
                "gate_passed": decision.gate_passed,
                "development_force_hungarian": decision.development_force_hungarian,
                "fallback_reason": decision.fallback_reason,
            },
        )

    def log_reliability_assignment(
        self,
        round_idx: int,
        replicate: int,
        decision: AssignmentDecision,
    ) -> None:
        self._append(
            self.reliability_assignments_path,
            {
                "round_idx": round_idx,
                "round_number": round_idx + 1,
                "probe_replicate": replicate,
                "hungarian_pairs": self._pairs(decision.hungarian_pairs),
                "hungarian_score": decision.hungarian_score,
                "Gamma": decision.gamma,
                "component_gate_tau_unused_for_reliability": decision.gate_tau,
            },
        )

    def save_diagnostic_checkpoint(
        self, round_idx: int, state: Mapping[str, torch.Tensor]
    ) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            clone_state_dict(state),
            self.checkpoints_dir / f"pre_round_{round_idx + 1:04d}_global_state.pt",
        )

    def log_round(self, trace: RoundTrace, **diagnostics: Any) -> None:
        self._append(
            self.rounds_path,
            {
                "round_idx": trace.round_idx,
                "round_number": trace.round_idx + 1,
                "selected_clients": list(trace.selected_clients),
                "assignments": [list(pair) for pair in trace.assignments],
                "task_seeds": list(trace.task_seeds),
                "task_hashes": list(trace.task_hashes),
                "local_seeds": list(trace.local_seeds),
                "global_state_hash": trace.global_state_hash,
                "diagnostic_accuracy": trace.diagnostic_accuracy,
                "diagnostic_loss": trace.diagnostic_loss,
                **diagnostics,
            },
        )

    def write_summary(self, payload: Mapping[str, Any]) -> None:
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
