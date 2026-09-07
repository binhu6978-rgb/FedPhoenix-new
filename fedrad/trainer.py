from __future__ import annotations

import copy
import json
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn

from fedrad.aggregation import weighted_fedavg
from fedrad.assignment import (
    assign_consensus_functional_recovery,
    assign_functional_recovery,
    assign_with_gate,
    best_vs_second_assignment_margin,
)
from fedrad.config import FedRADConfig
from fedrad.data import FederatedData
from fedrad.evaluation import Evaluator
from fedrad.local_trainer import LocalTrainer
from fedrad.logger import NullLogger
from fedrad.probe import ProbeRunner, materialize_probe_batch
from fedrad.rng import RNGStreams
from fedrad.task_bank import TaskBank, build_task_bank
from fedrad.types import (
    AssignmentDecision,
    FedRADRoundTrace,
    ProbeBatch,
    ProbeResult,
    RoundTrace,
    TrainingResult,
    clone_state_dict,
    state_dict_hash,
)
from fedrad.scoring import ScoreMatrices, build_score_matrices, mean_score_matrices


def _functional_replicate_diagnostics(
    replicate_scores: Sequence[ScoreMatrices],
    *,
    decision: AssignmentDecision,
    assignment_Q: np.ndarray,
    z_eps: float,
) -> dict[str, object]:
    raw = np.stack([score.G for score in replicate_scores], axis=0)
    means = np.mean(raw, axis=0)
    variances = np.var(raw, axis=0, ddof=0)
    standard_deviations = np.sqrt(variances)
    valid_cv = np.abs(means) > float(z_eps)
    replicate_assignments = [
        assign_functional_recovery(
            client_order=score.client_order,
            task_order=score.task_order,
            Q=score.G,
        )
        for score in replicate_scores
    ]
    final_edges = {
        (pair.client_id, pair.task_id) for pair in decision.final_pairs
    }
    overlaps = [
        len(
            final_edges.intersection(
                (pair.client_id, pair.task_id)
                for pair in replicate.final_pairs
            )
        )
        / len(final_edges)
        for replicate in replicate_assignments
    ]
    baseline_by_client = {
        pair.client_id: pair.task_id for pair in decision.baseline_pairs
    }
    column_by_task = {
        task_id: column
        for column, task_id in enumerate(replicate_scores[0].task_order)
    }
    changed = sum(
        baseline_by_client[pair.client_id] != pair.task_id
        for pair in decision.final_pairs
    )
    replicate_edge_sets = [
        {(pair.client_id, pair.task_id) for pair in item.final_pairs}
        for item in replicate_assignments
    ]
    selected_edges = {
        (pair.client_id, pair.task_id) for pair in decision.final_pairs
    }
    consensus_edges = replicate_edge_sets[0].intersection(
        *replicate_edge_sets[1:]
    )
    union_edges = replicate_edge_sets[0].union(*replicate_edge_sets[1:])
    reassigned_pairs = [
        pair
        for pair in decision.final_pairs
        if baseline_by_client[pair.client_id] != pair.task_id
    ]
    bilateral_gain_count = sum(
        all(
            score.G[
                pair.client_row, column_by_task[pair.task_id]
            ]
            > score.G[pair.client_row, pair.client_row]
            for score in replicate_scores
        )
        for pair in reassigned_pairs
    )
    return {
        "functional_probe_replicates": len(replicate_scores),
        "mean_score_variance_across_replicates": float(np.mean(variances)),
        "mean_pairwise_score_std": float(np.mean(standard_deviations)),
        "max_pairwise_score_std": float(np.max(standard_deviations)),
        "mean_pairwise_score_standard_error": float(
            np.mean(standard_deviations) / np.sqrt(len(replicate_scores))
        ),
        "mean_coefficient_of_variation": (
            float(np.mean(standard_deviations[valid_cv] / np.abs(means[valid_cv])))
            if np.any(valid_cv)
            else None
        ),
        "coefficient_of_variation_valid_pairs": int(np.sum(valid_cv)),
        "mean_replicate_assignment_overlap": float(np.mean(overlaps)),
        "replicate_hungarian_assignments": [
            [[pair.client_id, pair.task_id] for pair in replicate.final_pairs]
            for replicate in replicate_assignments
        ],
        "assignment_change_rate": float(changed / len(final_edges)),
        "assignment_objective": float(decision.hungarian_score),
        "assignment_margin": best_vs_second_assignment_margin(assignment_Q),
        "mean_reliability_penalty": float(np.mean(means - assignment_Q)),
        "mean_selected_reliability_penalty": float(
            np.mean(
                [
                    means[pair.client_row, column_by_task[pair.task_id]]
                    - assignment_Q[
                        pair.client_row, column_by_task[pair.task_id]
                    ]
                    for pair in decision.final_pairs
                ]
            )
        ),
        "replicate_consensus_pair_count": len(consensus_edges),
        "final_pair_supported_by_both_rate": float(
            len(selected_edges.intersection(consensus_edges)) / len(selected_edges)
        ),
        "final_pair_supported_by_either_rate": float(
            len(selected_edges.intersection(union_edges)) / len(selected_edges)
        ),
        "reassigned_pair_bilateral_gain_rate": (
            float(bilateral_gain_count / len(reassigned_pairs))
            if reassigned_pairs
            else 1.0
        ),
    }


def baseline_assignments(
    selected_clients: Sequence[int], bank: TaskBank
) -> tuple[tuple[int, int], ...]:
    if len(selected_clients) != len(bank.tasks):
        raise ValueError("Baseline requires one task per selected client")
    # Deliberately preserve sampling order: selected_clients[j] -> task[j].
    return tuple(
        (int(client_id), task_id)
        for task_id, client_id in enumerate(selected_clients)
    )


def _task_damage_diagnostics(bank: TaskBank) -> dict[str, float | int]:
    reset_counts = [
        int(task.reset_trace.get("num_reset_kernels", 0)) for task in bank.tasks
    ]
    return {
        "mean_reset_delta_norm": float(
            np.mean([task.delta_norm for task in bank.tasks])
        ),
        "mean_active_reset_kernels": float(np.mean(reset_counts)),
        "total_active_reset_kernels": int(sum(reset_counts)),
    }


class CleanFedPhoenixTrainer:
    """Phase-1 FedPhoenix only: reset, local SGD, and weighted FedAvg."""

    def __init__(
        self,
        *,
        config: FedRADConfig,
        data: FederatedData,
        model_template: nn.Module,
        rngs: RNGStreams,
        device: torch.device,
        logger: object | None = None,
    ):
        self.config = config
        self.data = data
        self.model_template = copy.deepcopy(model_template).to("cpu")
        self.rngs = rngs
        self.device = device
        self.logger = logger if logger is not None else NullLogger()
        self.local_trainer = LocalTrainer(
            config=config,
            model_template=self.model_template,
            train_dataset=data.train_dataset,
            device=device,
        )
        self.evaluator = Evaluator(
            config=config,
            model_template=self.model_template,
            dataset=data.test_dataset,
            device=device,
            evaluation_seed=rngs.evaluation_seed,
        )

    def run(self) -> TrainingResult:
        global_state = clone_state_dict(self.model_template.state_dict())
        initial_hash = state_dict_hash(global_state)
        traces: list[RoundTrace] = []
        self.logger.log_config(
            {
                **self.config.as_serializable_dict(),
                "algorithm": "CleanFedPhoenix",
                "partition_fingerprint": self.data.partition_fingerprint,
                "partition_path": str(self.data.partition_path),
                "initial_state_hash": initial_hash,
                "primary_evaluation_rule": (
                    "pre-specified final round; per-round test metrics are diagnostic only"
                ),
            }
        )

        for round_idx in range(self.config.rounds):
            round_start = time.perf_counter()
            selected_clients = self.rngs.sample_clients(
                self.config.num_users, self.config.clients_per_round
            )
            bank = build_task_bank(
                config=self.config,
                model_template=self.model_template,
                global_state=global_state,
                round_idx=round_idx,
                task_count=self.config.clients_per_round,
                rngs=self.rngs,
            )
            assignments = baseline_assignments(selected_clients, bank)
            self.logger.log_task_bank(bank)
            if (
                self.config.diagnostic_save_checkpoints
                and round_idx + 1 in self.config.diagnostic_probe_rounds
            ):
                self.logger.save_diagnostic_checkpoint(round_idx, global_state)

            updates = []
            local_seeds: list[int] = []
            for client_id, task_id in assignments:
                task = bank.task(task_id)
                local_seed = self.rngs.local_seed(round_idx, client_id)
                update = self.local_trainer.train_client(
                    initial_state=task.clone_reset_state(),
                    client_id=client_id,
                    task_id=task_id,
                    client_indices=self.data.partitions[client_id],
                    round_idx=round_idx,
                    local_seed=local_seed,
                )
                if update.initial_state_hash != task.state_hash:
                    raise RuntimeError(
                        f"Client {client_id} did not start from original task {task_id}"
                    )
                task.verify_hash()
                updates.append(update)
                local_seeds.append(local_seed)

            global_state = weighted_fedavg(updates)
            global_hash = state_dict_hash(global_state)
            evaluation = None
            if (round_idx + 1) % self.config.eval_every == 0:
                evaluation = self.evaluator.evaluate(global_state)

            trace = RoundTrace(
                round_idx=round_idx,
                selected_clients=selected_clients,
                assignments=assignments,
                task_seeds=tuple(task.reset_seed for task in bank.tasks),
                task_hashes=tuple(task.state_hash for task in bank.tasks),
                local_seeds=tuple(local_seeds),
                global_state_hash=global_hash,
                diagnostic_accuracy=(evaluation.accuracy if evaluation else None),
                diagnostic_loss=(evaluation.loss if evaluation else None),
            )
            traces.append(trace)
            total_examples = sum(update.num_examples for update in updates)
            mean_train_loss = sum(
                update.mean_train_loss * update.num_examples for update in updates
            ) / total_examples
            round_seconds = time.perf_counter() - round_start
            self.logger.log_round(
                trace,
                mean_client_train_loss=float(mean_train_loss),
                round_seconds=float(round_seconds),
                **_task_damage_diagnostics(bank),
            )
            accuracy_text = (
                "not evaluated"
                if evaluation is None
                else f"diagnostic_accuracy={evaluation.accuracy:.4f}%"
            )
            print(
                f"ROUND clean_fedphoenix round={round_idx + 1} "
                f"clients={list(selected_clients)} {accuracy_text} "
                f"global_hash={global_hash}",
                flush=True,
            )

        final_hash = state_dict_hash(global_state)
        self.logger.write_summary(
            {
                "status": "complete",
                "rounds": self.config.rounds,
                "initial_state_hash": initial_hash,
                "final_state_hash": final_hash,
                "partition_fingerprint": self.data.partition_fingerprint,
                "final_diagnostic_accuracy": traces[-1].diagnostic_accuracy,
                "final_diagnostic_loss": traces[-1].diagnostic_loss,
                "no_probe_score_matching_or_gate": True,
            }
        )
        return TrainingResult(
            initial_state_hash=initial_hash,
            final_state_hash=final_hash,
            traces=tuple(traces),
            _final_state=clone_state_dict(global_state),
        )


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class FedRADTrainer:
    """Phase-2 recovery-aware assignment with untouched Phase-1 local/FedAvg."""

    def __init__(
        self,
        *,
        config: FedRADConfig,
        data: FederatedData,
        model_template: nn.Module,
        rngs: RNGStreams,
        device: torch.device,
        logger: object | None = None,
    ):
        self.config = config
        self.data = data
        self.model_template = copy.deepcopy(model_template).to("cpu")
        self.rngs = rngs
        self.device = device
        self.logger = logger if logger is not None else NullLogger()
        self.local_trainer = LocalTrainer(
            config=config,
            model_template=self.model_template,
            train_dataset=data.train_dataset,
            device=device,
        )
        self.probe_runner = ProbeRunner(
            config=config,
            model_template=self.model_template,
            device=device,
        )
        self.evaluator = Evaluator(
            config=config,
            model_template=self.model_template,
            dataset=data.test_dataset,
            device=device,
            evaluation_seed=rngs.evaluation_seed,
        )
        self._warmup_reference = self._load_warmup_reference()

    def _load_warmup_reference(self) -> dict[int, dict[str, object]]:
        path = self.config.warmup_reference_rounds_path
        if self.config.matching_start_round == 1:
            return {}
        if path is None:
            raise ValueError("delayed matching requires a Clean warm-up reference")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_round = {int(row["round_number"]): row for row in rows}
        required = set(range(1, self.config.matching_start_round))
        if not required.issubset(by_round):
            missing = sorted(required.difference(by_round))
            raise ValueError(f"Clean warm-up reference is missing rounds: {missing}")
        return by_round

    def _assert_warmup_replay(self, trace: FedRADRoundTrace) -> None:
        round_number = trace.round_idx + 1
        reference = self._warmup_reference[round_number]
        actual = {
            "selected_clients": list(trace.selected_clients),
            "assignments": [list(pair) for pair in trace.assignments],
            "task_seeds": list(trace.task_seeds),
            "task_hashes": list(trace.task_hashes),
            "local_seeds": list(trace.local_seeds),
            "global_state_hash": trace.global_state_hash,
        }
        for key, value in actual.items():
            if reference[key] != value:
                raise RuntimeError(
                    f"warm-up replay diverged from Clean at round {round_number}: {key}"
                )

    def _run_warmup_round(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        bank: TaskBank,
        selected_clients: Sequence[int],
        assignments: tuple[tuple[int, int], ...],
        round_idx: int,
    ) -> tuple[dict[str, torch.Tensor], FedRADRoundTrace]:
        _synchronize(self.device)
        formal_start = time.perf_counter()
        updates = []
        local_seeds: list[int] = []
        formal_initial_hashes: list[str] = []
        for client_id, task_id in assignments:
            task = bank.task(task_id)
            local_seed = self.rngs.local_seed(round_idx, client_id)
            update = self.local_trainer.train_client(
                initial_state=task.clone_reset_state(),
                client_id=client_id,
                task_id=task_id,
                client_indices=self.data.partitions[client_id],
                round_idx=round_idx,
                local_seed=local_seed,
            )
            if update.initial_state_hash != task.state_hash:
                raise RuntimeError("Warm-up local training did not start from TaskSpec")
            task.verify_hash()
            updates.append(update)
            local_seeds.append(local_seed)
            formal_initial_hashes.append(update.initial_state_hash)
        next_state = weighted_fedavg(updates)
        _synchronize(self.device)
        formal_training_seconds = time.perf_counter() - formal_start
        global_hash = state_dict_hash(next_state)
        evaluation = None
        if (round_idx + 1) % self.config.eval_every == 0:
            evaluation = self.evaluator.evaluate(next_state)
        peak_gpu_memory_bytes = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        trace = FedRADRoundTrace(
            round_idx=round_idx,
            selected_clients=tuple(selected_clients),
            assignments=assignments,
            task_seeds=tuple(task.reset_seed for task in bank.tasks),
            task_hashes=tuple(task.state_hash for task in bank.tasks),
            local_seeds=tuple(local_seeds),
            global_state_hash=global_hash,
            diagnostic_accuracy=(evaluation.accuracy if evaluation else None),
            diagnostic_loss=(evaluation.loss if evaluation else None),
            probe_support_hashes=(),
            probe_query_hashes=(),
            G=(), A=(), D=(), C=(), Q=(),
            baseline_assignments=assignments,
            hungarian_assignments=assignments,
            formal_initial_hashes=tuple(formal_initial_hashes),
            gamma=0.0,
            baseline_score=0.0,
            hungarian_score=0.0,
            gate_passed=False,
            fallback_reason="matching_inactive",
            component_degenerate=(True, True, True, True),
            probe_seconds=0.0,
            reliability_probe_seconds=0.0,
            matching_seconds=0.0,
            formal_training_seconds=formal_training_seconds,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        )
        self._assert_warmup_replay(trace)
        total_examples = sum(update.num_examples for update in updates)
        mean_train_loss = sum(
            update.mean_train_loss * update.num_examples for update in updates
        ) / total_examples
        self.logger.log_round(
            trace,
            mean_client_train_loss=float(mean_train_loss),
            baseline_assignments=[list(pair) for pair in assignments],
            hungarian_assignments=[list(pair) for pair in assignments],
            formal_initial_hashes=list(trace.formal_initial_hashes),
            Gamma=0.0,
            baseline_score=0.0,
            hungarian_score=0.0,
            gate_tau=self.config.gate_tau,
            gate_passed=False,
            fallback_reason="matching_inactive",
            component_degenerate=[True, True, True, True],
            probe_seconds=0.0,
            reliability_probe_seconds=0.0,
            matching_seconds=0.0,
            formal_training_seconds=formal_training_seconds,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
            matching_active=False,
            score_mode=self.config.score_mode,
            **_task_damage_diagnostics(bank),
        )
        accuracy_text = (
            "not evaluated"
            if evaluation is None
            else f"diagnostic_accuracy={evaluation.accuracy:.4f}%"
        )
        print(
            f"ROUND fedrad-warmup round={round_idx + 1} "
            f"clients={list(selected_clients)} {accuracy_text} "
            f"global_hash={global_hash}",
            flush=True,
        )
        return next_state, trace

    def _run_probe_grid(
        self,
        *,
        global_state: dict[str, torch.Tensor],
        bank: TaskBank,
        selected_clients: Sequence[int],
        round_idx: int,
        replicate: int,
    ) -> tuple[list[ProbeBatch], list[ProbeResult]]:
        batches: list[ProbeBatch] = []
        results: list[ProbeResult] = []
        for client_id in selected_clients:
            probe_seed = self.rngs.probe_replicate_seed(
                round_idx, client_id, replicate
            )
            batch = materialize_probe_batch(
                dataset=self.data.train_dataset,
                client_indices=self.data.partitions[client_id],
                client_id=client_id,
                round_idx=round_idx,
                probe_seed=probe_seed,
                support_limit=self.config.probe_support_size,
                query_limit=self.config.probe_query_size,
                probe_replicate=replicate,
            )
            batches.append(batch)
            global_loss = self.probe_runner.global_reference(
                global_state=global_state, batch=batch
            )
            for task in bank.tasks:
                results.append(
                    self.probe_runner.run_pair(
                        task=task,
                        batch=batch,
                        global_loss=global_loss,
                    )
                )
        return batches, results

    def run(self) -> TrainingResult:
        global_state = clone_state_dict(self.model_template.state_dict())
        initial_hash = state_dict_hash(global_state)
        traces: list[FedRADRoundTrace] = []
        functional_round_diagnostics: list[dict[str, object]] = []
        self.logger.log_config(
            {
                **self.config.as_serializable_dict(),
                "algorithm": "FedRAD",
                "phase": 2,
                "diagnostic_phase": (
                    3 if self.config.diagnostic_probe_rounds else None
                ),
                "partition_fingerprint": self.data.partition_fingerprint,
                "partition_path": str(self.data.partition_path),
                "initial_state_hash": initial_hash,
                "scoring_weights_notice": self.config.scoring_weights_status,
                "gate_tau_notice": self.config.gate_tau_status,
                "development_force_hungarian_notice": (
                    "debug/smoke only; not part of the paper method"
                ),
                "primary_evaluation_rule": (
                    "pre-specified final round; per-round test metrics are diagnostic only"
                ),
            }
        )

        for round_idx in range(self.config.rounds):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            selected_clients = self.rngs.sample_clients(
                self.config.num_users, self.config.clients_per_round
            )
            bank = build_task_bank(
                config=self.config,
                model_template=self.model_template,
                global_state=global_state,
                round_idx=round_idx,
                task_count=self.config.clients_per_round,
                rngs=self.rngs,
            )
            baseline = baseline_assignments(selected_clients, bank)
            self.logger.log_task_bank(bank)
            if round_idx + 1 < self.config.matching_start_round:
                global_state, warmup_trace = self._run_warmup_round(
                    global_state=global_state,
                    bank=bank,
                    selected_clients=selected_clients,
                    assignments=baseline,
                    round_idx=round_idx,
                )
                traces.append(warmup_trace)
                continue
            if (
                self.config.diagnostic_save_checkpoints
                and round_idx + 1 in self.config.diagnostic_probe_rounds
            ):
                self.logger.save_diagnostic_checkpoint(round_idx, global_state)
            global_hash_before_probe = state_dict_hash(global_state)
            task_hashes_before_probe = tuple(task.state_hash for task in bank.tasks)

            _synchronize(self.device)
            probe_start = time.perf_counter()
            batches: list[ProbeBatch] = []
            probe_result_sets: list[list[ProbeResult]] = []
            replicate_count = (
                self.config.functional_probe_replicates
                if self.config.score_mode == "functional"
                else 1
            )
            for replicate in range(replicate_count):
                replicate_batches, replicate_results = self._run_probe_grid(
                    global_state=global_state,
                    bank=bank,
                    selected_clients=selected_clients,
                    round_idx=round_idx,
                    replicate=replicate,
                )
                batches.extend(replicate_batches)
                probe_result_sets.append(replicate_results)
            _synchronize(self.device)
            probe_seconds = time.perf_counter() - probe_start
            if state_dict_hash(global_state) != global_hash_before_probe:
                raise RuntimeError("FedRAD probe mutated the global model state")
            for task, expected_hash in zip(bank.tasks, task_hashes_before_probe):
                task.verify_hash()
                if task.state_hash != expected_hash:
                    raise RuntimeError("FedRAD probe mutated a TaskSpec")
            for replicate_results in probe_result_sets:
                self.logger.log_probe_results(replicate_results)

            matching_start = time.perf_counter()
            replicate_scores = [
                build_score_matrices(
                    selected_clients=selected_clients,
                    task_ids=tuple(task.task_id for task in bank.tasks),
                    results=replicate_results,
                    config=self.config,
                )
                for replicate_results in probe_result_sets
            ]
            if self.config.score_mode == "functional":
                scores = mean_score_matrices(replicate_scores, config=self.config)
                # Formal Ours uses the configured reliability-adjusted raw-G
                # estimate. The default mean mode exactly preserves M2.
                if self.config.functional_assignment_mode == "hungarian":
                    decision = assign_functional_recovery(
                        client_order=tuple(selected_clients),
                        task_order=tuple(task.task_id for task in bank.tasks),
                        Q=scores.Q,
                    )
                else:
                    decision = assign_consensus_functional_recovery(
                        client_order=tuple(selected_clients),
                        task_order=tuple(task.task_id for task in bank.tasks),
                        mean_Q=scores.Q,
                        replicate_Q=[score.G for score in replicate_scores],
                        mode=self.config.functional_assignment_mode,
                    )
                round_functional_diagnostics = _functional_replicate_diagnostics(
                    replicate_scores,
                    decision=decision,
                    assignment_Q=scores.Q,
                    z_eps=self.config.z_eps,
                )
                functional_round_diagnostics.append(round_functional_diagnostics)
            else:
                scores = replicate_scores[0]
                decision = assign_with_gate(
                    scores,
                    gate_tau=self.config.gate_tau,
                    development_force_hungarian=(
                        self.config.development_force_hungarian
                    ),
                )
                round_functional_diagnostics = {}
            matching_seconds = time.perf_counter() - matching_start
            self.logger.log_scores(round_idx, scores)
            self.logger.log_assignment(round_idx, decision)

            reliability_probe_seconds = 0.0
            if round_idx + 1 in self.config.diagnostic_probe_rounds:
                probe_results = probe_result_sets[0]
                self.logger.log_probe_reliability_results(0, probe_results)
                self.logger.log_reliability_scores(round_idx, 0, scores)
                self.logger.log_reliability_assignment(round_idx, 0, decision)
                _synchronize(self.device)
                reliability_start = time.perf_counter()
                for replicate in range(1, self.config.diagnostic_probe_replicates):
                    _, replicate_results = self._run_probe_grid(
                        global_state=global_state,
                        bank=bank,
                        selected_clients=selected_clients,
                        round_idx=round_idx,
                        replicate=replicate,
                    )
                    if state_dict_hash(global_state) != global_hash_before_probe:
                        raise RuntimeError("Reliability probe mutated global state")
                    for task, expected_hash in zip(
                        bank.tasks, task_hashes_before_probe
                    ):
                        task.verify_hash()
                        if task.state_hash != expected_hash:
                            raise RuntimeError("Reliability probe mutated a TaskSpec")
                    reliability_scores = build_score_matrices(
                        selected_clients=selected_clients,
                        task_ids=tuple(task.task_id for task in bank.tasks),
                        results=replicate_results,
                        config=self.config,
                    )
                    replicate_decision = assign_with_gate(
                        reliability_scores,
                        gate_tau=self.config.gate_tau,
                        development_force_hungarian=False,
                    )
                    self.logger.log_probe_reliability_results(
                        replicate, replicate_results
                    )
                    self.logger.log_reliability_scores(
                        round_idx, replicate, reliability_scores
                    )
                    self.logger.log_reliability_assignment(
                        round_idx, replicate, replicate_decision
                    )
                _synchronize(self.device)
                reliability_probe_seconds = (
                    time.perf_counter() - reliability_start
                )
            assignments = tuple(
                (pair.client_id, pair.task_id) for pair in decision.final_pairs
            )
            if baseline != tuple(
                (pair.client_id, pair.task_id) for pair in decision.baseline_pairs
            ):
                raise RuntimeError("Scoring baseline mapping changed Phase-1 order")

            _synchronize(self.device)
            formal_start = time.perf_counter()
            updates = []
            local_seeds: list[int] = []
            formal_initial_hashes: list[str] = []
            for client_id, task_id in assignments:
                task = bank.task(task_id)
                local_seed = self.rngs.local_seed(round_idx, client_id)
                update = self.local_trainer.train_client(
                    initial_state=task.clone_reset_state(),
                    client_id=client_id,
                    task_id=task_id,
                    client_indices=self.data.partitions[client_id],
                    round_idx=round_idx,
                    local_seed=local_seed,
                )
                if update.initial_state_hash != task.state_hash:
                    raise RuntimeError(
                        "Probe-adapted state entered formal local training: "
                        f"client={client_id}, task={task_id}"
                    )
                task.verify_hash()
                updates.append(update)
                local_seeds.append(local_seed)
                formal_initial_hashes.append(update.initial_state_hash)
            global_state = weighted_fedavg(updates)
            _synchronize(self.device)
            formal_training_seconds = time.perf_counter() - formal_start
            global_hash = state_dict_hash(global_state)
            evaluation = None
            if (round_idx + 1) % self.config.eval_every == 0:
                evaluation = self.evaluator.evaluate(global_state)
            peak_gpu_memory_bytes = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )

            trace = FedRADRoundTrace(
                round_idx=round_idx,
                selected_clients=selected_clients,
                assignments=assignments,
                task_seeds=tuple(task.reset_seed for task in bank.tasks),
                task_hashes=tuple(task.state_hash for task in bank.tasks),
                local_seeds=tuple(local_seeds),
                global_state_hash=global_hash,
                diagnostic_accuracy=(evaluation.accuracy if evaluation else None),
                diagnostic_loss=(evaluation.loss if evaluation else None),
                probe_support_hashes=tuple(batch.support_hash for batch in batches),
                probe_query_hashes=tuple(batch.query_hash for batch in batches),
                G=_matrix_tuple(scores.G),
                A=_matrix_tuple(scores.A),
                D=_matrix_tuple(scores.D),
                C=_matrix_tuple(scores.C),
                Q=_matrix_tuple(scores.Q),
                baseline_assignments=baseline,
                hungarian_assignments=tuple(
                    (pair.client_id, pair.task_id)
                    for pair in decision.hungarian_pairs
                ),
                formal_initial_hashes=tuple(formal_initial_hashes),
                gamma=decision.gamma,
                baseline_score=decision.baseline_score,
                hungarian_score=decision.hungarian_score,
                gate_passed=decision.gate_passed,
                fallback_reason=decision.fallback_reason,
                component_degenerate=scores.component_degenerate,
                probe_seconds=probe_seconds,
                reliability_probe_seconds=reliability_probe_seconds,
                matching_seconds=matching_seconds,
                formal_training_seconds=formal_training_seconds,
                peak_gpu_memory_bytes=peak_gpu_memory_bytes,
            )
            traces.append(trace)
            total_examples = sum(update.num_examples for update in updates)
            mean_train_loss = sum(
                update.mean_train_loss * update.num_examples for update in updates
            ) / total_examples
            self.logger.log_round(
                trace,
                mean_client_train_loss=float(mean_train_loss),
                baseline_assignments=[list(pair) for pair in baseline],
                hungarian_assignments=[list(pair) for pair in trace.hungarian_assignments],
                formal_initial_hashes=list(trace.formal_initial_hashes),
                Gamma=decision.gamma,
                baseline_score=decision.baseline_score,
                hungarian_score=decision.hungarian_score,
                gate_tau=decision.gate_tau,
                gate_passed=decision.gate_passed,
                fallback_reason=decision.fallback_reason,
                component_degenerate=list(scores.component_degenerate),
                probe_seconds=probe_seconds,
                reliability_probe_seconds=reliability_probe_seconds,
                matching_seconds=matching_seconds,
                formal_training_seconds=formal_training_seconds,
                peak_gpu_memory_bytes=peak_gpu_memory_bytes,
                matching_active=True,
                score_mode=self.config.score_mode,
                **round_functional_diagnostics,
                **_task_damage_diagnostics(bank),
            )
            accuracy_text = (
                "not evaluated"
                if evaluation is None
                else f"diagnostic_accuracy={evaluation.accuracy:.4f}%"
            )
            print(
                f"ROUND fedrad round={round_idx + 1} clients={list(selected_clients)} "
                f"{accuracy_text} Gamma={decision.gamma:.6f} "
                f"gate_passed={decision.gate_passed} "
                f"fallback={decision.fallback_reason} global_hash={global_hash}",
                flush=True,
            )

        final_hash = state_dict_hash(global_state)
        self.logger.write_summary(
            {
                "status": "complete",
                "rounds": self.config.rounds,
                "initial_state_hash": initial_hash,
                "final_state_hash": final_hash,
                "partition_fingerprint": self.data.partition_fingerprint,
                "final_diagnostic_accuracy": traces[-1].diagnostic_accuracy,
                "final_diagnostic_loss": traces[-1].diagnostic_loss,
                "mean_probe_seconds": float(np.mean([t.probe_seconds for t in traces])),
                "mean_matching_seconds": float(
                    np.mean([t.matching_seconds for t in traces])
                ),
                "total_reliability_probe_seconds": float(
                    sum(t.reliability_probe_seconds for t in traces)
                ),
                "mean_formal_training_seconds": float(
                    np.mean([t.formal_training_seconds for t in traces])
                ),
                "peak_gpu_memory_bytes": max(
                    trace.peak_gpu_memory_bytes for trace in traces
                ),
                "development_force_hungarian": (
                    self.config.development_force_hungarian
                ),
                "matching_start_round": self.config.matching_start_round,
                "score_mode": self.config.score_mode,
                "functional_probe_replicates": (
                    self.config.functional_probe_replicates
                ),
                "functional_reliability_mode": (
                    self.config.functional_reliability_mode
                ),
                "functional_assignment_mode": (
                    self.config.functional_assignment_mode
                ),
                "mean_score_variance_across_replicates": (
                    float(
                        np.mean(
                            [
                                row["mean_score_variance_across_replicates"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_pairwise_score_std": (
                    float(
                        np.mean(
                            [
                                row["mean_pairwise_score_std"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_pairwise_score_standard_error": (
                    float(
                        np.mean(
                            [
                                row["mean_pairwise_score_standard_error"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_replicate_assignment_overlap": (
                    float(
                        np.mean(
                            [
                                row["mean_replicate_assignment_overlap"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_assignment_change_rate": (
                    float(
                        np.mean(
                            [
                                row["assignment_change_rate"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_assignment_margin": (
                    float(
                        np.mean(
                            [
                                row["assignment_margin"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_reliability_penalty": (
                    float(
                        np.mean(
                            [
                                row["mean_reliability_penalty"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_selected_reliability_penalty": (
                    float(
                        np.mean(
                            [
                                row["mean_selected_reliability_penalty"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_replicate_consensus_pair_count": (
                    float(
                        np.mean(
                            [
                                row["replicate_consensus_pair_count"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_final_pair_supported_by_both_rate": (
                    float(
                        np.mean(
                            [
                                row["final_pair_supported_by_both_rate"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_final_pair_supported_by_either_rate": (
                    float(
                        np.mean(
                            [
                                row["final_pair_supported_by_either_rate"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "mean_reassigned_pair_bilateral_gain_rate": (
                    float(
                        np.mean(
                            [
                                row["reassigned_pair_bilateral_gain_rate"]
                                for row in functional_round_diagnostics
                            ]
                        )
                    )
                    if functional_round_diagnostics
                    else None
                ),
                "warmup_rounds_asserted": self.config.matching_start_round - 1,
                "scoring_weights_notice": self.config.scoring_weights_status,
                "gate_tau_notice": self.config.gate_tau_status,
            }
        )
        return TrainingResult(
            initial_state_hash=initial_hash,
            final_state_hash=final_hash,
            traces=tuple(traces),
            _final_state=clone_state_dict(global_state),
        )
