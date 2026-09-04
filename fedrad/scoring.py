from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fedrad.config import FedRADConfig
from fedrad.types import ProbeResult


@dataclass(frozen=True)
class ScoreMatrices:
    client_order: tuple[int, ...]
    task_order: tuple[int, ...]
    G: np.ndarray
    A: np.ndarray
    D: np.ndarray
    C: np.ndarray
    ZG: np.ndarray
    ZA: np.ndarray
    ZD: np.ndarray
    ZC: np.ndarray
    Q: np.ndarray
    component_degenerate: tuple[bool, bool, bool, bool]


def matrix_zscore(
    matrix: np.ndarray,
    *,
    z_eps: float,
    std_atol: float,
    std_rtol: float,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Score components must be non-empty matrices")
    if not np.isfinite(values).all():
        raise ValueError("Score component contains NaN or Inf")
    standard_deviation = float(np.std(values, ddof=0))
    max_abs = float(np.max(np.abs(values)))
    threshold = float(std_atol) + float(std_rtol) * max(max_abs, 1.0)
    if standard_deviation <= threshold:
        return np.zeros_like(values), True
    normalized = (values - float(np.mean(values))) / (
        standard_deviation + float(z_eps)
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized score component contains NaN or Inf")
    return normalized, False


def build_score_matrices(
    *,
    selected_clients: Sequence[int],
    task_ids: Sequence[int],
    results: Sequence[ProbeResult],
    config: FedRADConfig,
) -> ScoreMatrices:
    client_order = tuple(int(client_id) for client_id in selected_clients)
    task_order = tuple(int(task_id) for task_id in task_ids)
    if len(client_order) != len(set(client_order)):
        raise ValueError("Selected client IDs must be unique")
    if len(task_order) != len(set(task_order)):
        raise ValueError("Task IDs must be unique")
    if not client_order or len(client_order) != len(task_order):
        raise ValueError("FedRAD scoring requires a non-empty square KxK grid")

    row_by_client = {client_id: row for row, client_id in enumerate(client_order)}
    column_by_task = {task_id: column for column, task_id in enumerate(task_order)}
    shape = (len(client_order), len(task_order))
    components = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("G", "A", "D", "C")
    }
    seen: set[tuple[int, int]] = set()
    for result in results:
        pair = (int(result.client_id), int(result.task_id))
        if pair in seen:
            raise ValueError(f"Duplicate probe result for pair {pair}")
        if pair[0] not in row_by_client or pair[1] not in column_by_task:
            raise ValueError(f"Probe result has unexpected pair {pair}")
        seen.add(pair)
        row = row_by_client[pair[0]]
        column = column_by_task[pair[1]]
        for name in components:
            components[name][row, column] = float(getattr(result, name))

    expected = len(client_order) * len(task_order)
    if len(seen) != expected:
        raise ValueError(f"Expected {expected} probe pairs, received {len(seen)}")
    if not all(np.isfinite(matrix).all() for matrix in components.values()):
        raise ValueError("Probe grid is incomplete or non-finite")

    normalized: dict[str, np.ndarray] = {}
    degenerate: list[bool] = []
    for name in ("G", "A", "D", "C"):
        z, flag = matrix_zscore(
            components[name],
            z_eps=config.z_eps,
            std_atol=config.std_atol,
            std_rtol=config.std_rtol,
        )
        normalized[name] = z
        degenerate.append(flag)

    if config.score_mode == "g_only":
        Q = normalized["G"].copy()
    else:
        Q = (
            config.lambda_G * normalized["G"]
            + config.lambda_A * normalized["A"]
            - config.lambda_D * normalized["D"]
            + config.lambda_C * normalized["C"]
        )
    if not np.isfinite(Q).all():
        raise ValueError("Q contains NaN or Inf")
    return ScoreMatrices(
        client_order=client_order,
        task_order=task_order,
        G=components["G"],
        A=components["A"],
        D=components["D"],
        C=components["C"],
        ZG=normalized["G"],
        ZA=normalized["A"],
        ZD=normalized["D"],
        ZC=normalized["C"],
        Q=Q,
        component_degenerate=tuple(degenerate),
    )
