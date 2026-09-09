from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

import torch


StateDict: TypeAlias = dict[str, torch.Tensor]


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> StateDict:
    return {
        key: value.detach().to("cpu").clone()
        for key, value in state.items()
    }


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, tensor metadata, and exact CPU bytes in stable key order."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().to("cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def assert_finite_state(state: Mapping[str, torch.Tensor], label: str) -> None:
    for key, tensor in state.items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"{label}.{key} contains NaN or Inf")


@dataclass(frozen=True)
class ResetSlice:
    parameter_name: str
    output_indices: tuple[int, ...]
    parameter_shape: tuple[int, ...]


@dataclass(frozen=True)
class TaskSpec:
    round_idx: int
    task_id: int
    reset_seed: int
    parent_state_hash: str
    state_hash: str
    omega: tuple[ResetSlice, ...]
    delta_norm: float
    reset_trace: Mapping[str, object]
    active: bool
    _reset_state: Mapping[str, torch.Tensor] = field(repr=False)
    _delta: Mapping[str, torch.Tensor] = field(repr=False)

    def clone_reset_state(self) -> StateDict:
        return clone_state_dict(self._reset_state)

    def clone_delta(self) -> StateDict:
        return clone_state_dict(self._delta)

    def verify_hash(self) -> None:
        actual = state_dict_hash(self._reset_state)
        if actual != self.state_hash:
            raise RuntimeError(
                f"Task {self.task_id} state mutated: {actual} != {self.state_hash}"
            )


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    task_id: int
    initial_state_hash: str
    num_examples: int
    mean_train_loss: float
    local_seed: int
    state_hash: str
    _state: Mapping[str, torch.Tensor] = field(repr=False)

    def clone_state(self) -> StateDict:
        return clone_state_dict(self._state)


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    num_examples: int


@dataclass(frozen=True)
class ProbeBatch:
    round_idx: int
    client_id: int
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    support_images: torch.Tensor = field(repr=False)
    support_labels: torch.Tensor = field(repr=False)
    query_images: torch.Tensor = field(repr=False)
    query_labels: torch.Tensor = field(repr=False)
    support_hash: str
    query_hash: str
    probe_seed: int
    probe_replicate: int = 0
    support_batch_indices: tuple[tuple[int, ...], ...] = ()
    support_batch_images: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    support_batch_labels: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    support_batch_hashes: tuple[str, ...] = ()
    support_step_batch_ids: tuple[int, ...] = ()
    query_batch_indices: tuple[tuple[int, ...], ...] = ()
    query_batch_images: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    query_batch_labels: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    query_batch_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeResult:
    round_idx: int
    client_id: int
    task_id: int
    task_state_hash: str
    support_hash: str
    query_hash: str
    global_loss: float
    reset_loss: float
    adapted_loss: float
    G: float
    A: float
    D: float
    C: float
    gradient_norm: float
    delta_norm: float
    alignment_valid: bool
    probe_seed: int
    probe_replicate: int = 0


@dataclass(frozen=True)
class AssignmentPair:
    client_id: int
    client_row: int
    task_id: int
    score: float


@dataclass(frozen=True)
class AssignmentDecision:
    baseline_pairs: tuple[AssignmentPair, ...]
    hungarian_pairs: tuple[AssignmentPair, ...]
    final_pairs: tuple[AssignmentPair, ...]
    hungarian_score: float
    baseline_score: float
    gamma: float
    gate_tau: float
    gate_passed: bool
    development_force_hungarian: bool
    fallback_reason: str


@dataclass(frozen=True)
class RoundTrace:
    round_idx: int
    selected_clients: tuple[int, ...]
    assignments: tuple[tuple[int, int], ...]
    task_seeds: tuple[int, ...]
    task_hashes: tuple[str, ...]
    local_seeds: tuple[int, ...]
    global_state_hash: str
    diagnostic_accuracy: float | None
    diagnostic_loss: float | None


@dataclass(frozen=True)
class FedRADRoundTrace(RoundTrace):
    probe_support_hashes: tuple[str, ...]
    probe_query_hashes: tuple[str, ...]
    G: tuple[tuple[float, ...], ...]
    A: tuple[tuple[float, ...], ...]
    D: tuple[tuple[float, ...], ...]
    C: tuple[tuple[float, ...], ...]
    Q: tuple[tuple[float, ...], ...]
    baseline_assignments: tuple[tuple[int, int], ...]
    hungarian_assignments: tuple[tuple[int, int], ...]
    formal_initial_hashes: tuple[str, ...]
    gamma: float
    baseline_score: float
    hungarian_score: float
    gate_passed: bool
    fallback_reason: str
    component_degenerate: tuple[bool, bool, bool, bool]
    probe_seconds: float
    reliability_probe_seconds: float
    matching_seconds: float
    formal_training_seconds: float
    peak_gpu_memory_bytes: int


@dataclass(frozen=True)
class TrainingResult:
    initial_state_hash: str
    final_state_hash: str
    traces: tuple[RoundTrace, ...]
    _final_state: Mapping[str, torch.Tensor] = field(repr=False)

    def clone_final_state(self) -> StateDict:
        return clone_state_dict(self._final_state)
