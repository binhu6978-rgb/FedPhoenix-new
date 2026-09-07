from __future__ import annotations

import math
from typing import Sequence

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


def assign_consensus_functional_recovery(
    *,
    client_order: tuple[int, ...],
    task_order: tuple[int, ...],
    mean_Q: np.ndarray,
    replicate_Q: Sequence[np.ndarray],
    mode: str,
) -> AssignmentDecision:
    """Use two Probe-optimal permutations as structural assignment evidence."""
    if mode not in {"consensus_lock", "union_restrict", "bilateral_gain"}:
        raise ValueError("Unknown consensus Functional Recovery assignment mode")
    value = np.asarray(mean_Q, dtype=np.float64)
    size = len(client_order)
    if value.shape != (size, size) or len(task_order) != size:
        raise ValueError("consensus assignment requires a square aligned mean Q")
    if len(replicate_Q) != 2:
        raise ValueError("consensus assignment requires exactly two Probe matrices")
    probes = tuple(np.asarray(matrix, dtype=np.float64) for matrix in replicate_Q)
    if any(matrix.shape != value.shape for matrix in probes):
        raise ValueError("consensus Probe matrices have different shapes")
    if not np.isfinite(value).all() or not all(
        np.isfinite(matrix).all() for matrix in probes
    ):
        raise ValueError("consensus assignment received NaN or Inf")

    replicate_columns: list[np.ndarray] = []
    for matrix in probes:
        rows, columns = linear_sum_assignment(-matrix)
        by_row = np.empty(size, dtype=np.int64)
        by_row[rows] = columns
        replicate_columns.append(by_row)

    if mode == "consensus_lock":
        agreed_rows = np.flatnonzero(replicate_columns[0] == replicate_columns[1])
        columns = np.full(size, -1, dtype=np.int64)
        columns[agreed_rows] = replicate_columns[0][agreed_rows]
        used_columns = set(columns[agreed_rows].tolist())
        remaining_rows = [row for row in range(size) if columns[row] < 0]
        remaining_columns = [
            column for column in range(size) if column not in used_columns
        ]
        if remaining_rows:
            sub_rows, sub_columns = linear_sum_assignment(
                -value[np.ix_(remaining_rows, remaining_columns)]
            )
            for sub_row, sub_column in zip(sub_rows.tolist(), sub_columns.tolist()):
                columns[remaining_rows[sub_row]] = remaining_columns[sub_column]
    else:
        allowed = np.eye(size, dtype=np.bool_)
        row_indices = np.arange(size)
        if mode == "union_restrict":
            allowed[row_indices, replicate_columns[0]] = True
            allowed[row_indices, replicate_columns[1]] = True
        else:
            first_gain = probes[0] > np.diag(probes[0])[:, None]
            second_gain = probes[1] > np.diag(probes[1])[:, None]
            allowed |= first_gain & second_gain
        cost = -value.copy()
        cost[~allowed] = np.inf
        rows, selected_columns = linear_sum_assignment(cost)
        columns = np.empty(size, dtype=np.int64)
        columns[rows] = selected_columns

    chosen = tuple(
        AssignmentPair(
            client_order[row],
            row,
            task_order[int(columns[row])],
            float(value[row, columns[row]]),
        )
        for row in range(size)
    )
    baseline = tuple(
        AssignmentPair(
            client_id,
            row,
            task_order[row],
            float(value[row, row]),
        )
        for row, client_id in enumerate(client_order)
    )
    _validate_bijection(chosen, expected_size=size)
    return AssignmentDecision(
        baseline,
        chosen,
        chosen,
        float(sum(pair.score for pair in chosen)),
        float(sum(pair.score for pair in baseline)),
        0.0,
        0.0,
        True,
        False,
        f"functional_recovery_{mode}",
    )


def best_vs_second_assignment_margin(Q: np.ndarray) -> float:
    """Return the objective gap between the best and second-best bijections."""
    value = np.asarray(Q, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not value.size:
        raise ValueError("assignment margin requires a non-empty square matrix")
    if not np.isfinite(value).all():
        raise ValueError("assignment margin received NaN or Inf")
    if value.shape[0] == 1:
        return 0.0
    rows, columns = linear_sum_assignment(-value)
    best = float(value[rows, columns].sum())
    alternatives: list[float] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        cost = -value.copy()
        cost[row, column] = np.inf
        alt_rows, alt_columns = linear_sum_assignment(cost)
        alternatives.append(float(value[alt_rows, alt_columns].sum()))
    margin = best - max(alternatives)
    if not math.isfinite(margin) or margin < -1e-10:
        raise RuntimeError("invalid best-vs-second assignment margin")
    return max(0.0, float(margin))


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
