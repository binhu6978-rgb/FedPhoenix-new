from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from fedrad.scoring import ScoreMatrices
from fedrad.types import AssignmentDecision, AssignmentPair


def assign_functional_recovery(
    *, client_order: tuple[int, ...], task_order: tuple[int, ...], Q: np.ndarray
) -> AssignmentDecision:
    """Direct maximum-weight assignment for the formal functional-recovery path.

    Unlike the historical FedRAD helper, this has no gate or fallback: a
    finite utility matrix always dispatches the Hungarian permutation.
    """
    value = np.asarray(Q, dtype=np.float64)
    size = len(client_order)
    if value.shape != (size, size) or len(task_order) != size:
        raise ValueError("functional recovery requires a square aligned Q")
    if not np.isfinite(value).all():
        raise ValueError("functional recovery Q contains NaN or Inf")
    rows, cols = linear_sum_assignment(-value)
    chosen = tuple(AssignmentPair(client_order[r], r, task_order[c], float(value[r, c])) for r, c in sorted(zip(rows.tolist(), cols.tolist())))
    baseline = tuple(AssignmentPair(client_id, r, task_order[r], float(value[r, r])) for r, client_id in enumerate(client_order))
    _validate_bijection(chosen, expected_size=size)
    return AssignmentDecision(baseline, chosen, chosen, float(sum(x.score for x in chosen)), float(sum(x.score for x in baseline)), 0.0, 0.0, True, False, "functional_recovery_hungarian")


def _validate_bijection(
    pairs: tuple[AssignmentPair, ...], *, expected_size: int
) -> None:
    if len(pairs) != expected_size:
        raise RuntimeError("Assignment does not contain exactly K pairs")
    rows = {pair.client_row for pair in pairs}
    clients = {pair.client_id for pair in pairs}
    tasks = {pair.task_id for pair in pairs}
    if len(rows) != expected_size or len(clients) != expected_size:
        raise RuntimeError("Assignment repeats a client")
    if len(tasks) != expected_size:
        raise RuntimeError("Assignment repeats a task")


def assign_with_gate(
    scores: ScoreMatrices,
    *,
    gate_tau: float,
    development_force_hungarian: bool = False,
) -> AssignmentDecision:
    Q = np.asarray(scores.Q, dtype=np.float64)
    size = len(scores.client_order)
    if Q.shape != (size, size) or size != len(scores.task_order):
        raise ValueError("Hungarian assignment requires square Q aligned to orders")
    if not np.isfinite(Q).all():
        raise ValueError("Hungarian assignment received NaN or Inf")
    if gate_tau <= 0:
        raise ValueError("gate_tau must be positive")

    row_indices, column_indices = linear_sum_assignment(-Q)
    by_row = sorted(
        zip(row_indices.tolist(), column_indices.tolist()), key=lambda item: item[0]
    )
    hungarian_pairs = tuple(
        AssignmentPair(
            client_id=scores.client_order[row],
            client_row=row,
            task_id=scores.task_order[column],
            score=float(Q[row, column]),
        )
        for row, column in by_row
    )
    baseline_pairs = tuple(
        AssignmentPair(
            client_id=client_id,
            client_row=row,
            task_id=scores.task_order[row],
            score=float(Q[row, row]),
        )
        for row, client_id in enumerate(scores.client_order)
    )
    _validate_bijection(hungarian_pairs, expected_size=size)
    _validate_bijection(baseline_pairs, expected_size=size)

    hungarian_score = float(sum(pair.score for pair in hungarian_pairs))
    baseline_score = float(sum(pair.score for pair in baseline_pairs))
    gamma = hungarian_score / size
    if not all(math.isfinite(value) for value in (hungarian_score, baseline_score, gamma)):
        raise ValueError("Assignment scores contain NaN or Inf")
    gate_passed = gamma >= float(gate_tau)
    if development_force_hungarian:
        final_pairs = hungarian_pairs
        fallback_reason = "development_force_hungarian"
    elif gate_passed:
        final_pairs = hungarian_pairs
        fallback_reason = "gate_passed"
    else:
        final_pairs = baseline_pairs
        fallback_reason = "gamma_below_tau"
    _validate_bijection(final_pairs, expected_size=size)
    return AssignmentDecision(
        baseline_pairs=baseline_pairs,
        hungarian_pairs=hungarian_pairs,
        final_pairs=final_pairs,
        hungarian_score=hungarian_score,
        baseline_score=baseline_score,
        gamma=gamma,
        gate_tau=float(gate_tau),
        gate_passed=gate_passed,
        development_force_hungarian=bool(development_force_hungarian),
        fallback_reason=fallback_reason,
    )
