from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def double_center(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("double_center expects a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError("diagnostic matrix contains NaN or Inf")
    return (
        values
        - values.mean(axis=1, keepdims=True)
        - values.mean(axis=0, keepdims=True)
        + values.mean()
    )


def interaction_metrics(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    interaction = double_center(values)
    overall_std = float(values.std(ddof=0))
    interaction_std = float(interaction.std(ddof=0))
    ratio = interaction_std / overall_std if overall_std > 0.0 else 0.0
    return {
        "overall_std": overall_std,
        "mean_row_std": float(values.std(axis=1, ddof=0).mean()),
        "mean_col_std": float(values.std(axis=0, ddof=0).mean()),
        "interaction_std": interaction_std,
        "interaction_overall_ratio": ratio,
    }


def pearson_flat(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("correlation arrays must have equal non-zero size")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(value):
        raise ValueError("Pearson correlation is NaN or Inf")
    return value


def spearman_flat(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("correlation arrays must have equal non-zero size")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    value = float(spearmanr(x, y).statistic)
    if not math.isfinite(value):
        raise ValueError("Spearman correlation is NaN or Inf")
    return value


def within_row_spearman(
    left: np.ndarray, right: np.ndarray
) -> tuple[float, float, tuple[float, ...]]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("within-row correlation expects equal matrices")
    values = tuple(spearman_flat(x[row], y[row]) for row in range(x.shape[0]))
    return float(np.mean(values)), float(np.std(values, ddof=0)), values


def maximum_assignment(matrix: np.ndarray) -> tuple[tuple[int, ...], float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("assignment diagnostic requires a square matrix")
    rows, columns = linear_sum_assignment(-values)
    by_row = sorted(zip(rows.tolist(), columns.tolist()))
    assignment = tuple(column for _, column in by_row)
    score = float(sum(values[row, column] for row, column in by_row))
    return assignment, score


def assignment_score(matrix: np.ndarray, assignment: Sequence[int]) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    columns = tuple(int(column) for column in assignment)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or len(columns) != values.shape[0]
        or set(columns) != set(range(values.shape[1]))
    ):
        raise ValueError("assignment must be a permutation for a square matrix")
    return float(values[np.arange(values.shape[0]), np.asarray(columns)].sum())


def second_best_assignment(
    matrix: np.ndarray,
) -> tuple[tuple[int, ...], float, tuple[int, ...], float]:
    """Return the best and exact second-best feasible one-to-one assignments.

    Every assignment distinct from the optimum excludes at least one selected
    edge. Re-solving once per selected edge therefore covers the complete set
    of alternatives without introducing an approximate random search.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.shape[0] < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("second-best diagnostic requires a finite square matrix")
    best, best_score = maximum_assignment(values)
    alternatives: list[tuple[float, tuple[int, ...]]] = []
    for row, selected_column in enumerate(best):
        constrained = values.copy()
        constrained[row, selected_column] = -np.inf
        rows, columns = linear_sum_assignment(-constrained)
        by_row = sorted(zip(rows.tolist(), columns.tolist()))
        alternative = tuple(column for _, column in by_row)
        score = float(sum(values[r, column] for r, column in by_row))
        alternatives.append((score, alternative))
    second_score, second = max(alternatives, key=lambda item: item[0])
    return best, best_score, second, second_score


def assignment_overlap(
    left: Sequence[int], right: Sequence[int]
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("assignments must have equal non-zero length")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def random_assignment_scores(
    matrix: np.ndarray, *, samples: int, seed: int
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if samples < 1 or values.shape[0] != values.shape[1]:
        raise ValueError("invalid random assignment request")
    rng = np.random.default_rng(int(seed))
    rows = np.arange(values.shape[0])
    output = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        output[sample] = float(values[rows, rng.permutation(values.shape[1])].sum())
    return output


def null_hungarian_gammas(
    matrix: np.ndarray, *, samples: int, seed: int
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if samples < 1 or values.shape[0] != values.shape[1]:
        raise ValueError("invalid null-calibration request")
    rng = np.random.default_rng(int(seed))
    output = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        null = np.stack(
            [rng.permutation(values[row]) for row in range(values.shape[0])]
        )
        _, score = maximum_assignment(null)
        output[sample] = score / values.shape[0]
    return output


def component_contribution_assignments(
    *,
    Q: np.ndarray,
    normalized: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> dict[str, dict[str, object]]:
    full_assignment, _ = maximum_assignment(Q)
    signs = {"G": 1.0, "A": 1.0, "D": -1.0, "C": 1.0}
    output: dict[str, dict[str, object]] = {}
    for name in ("G", "A", "D", "C"):
        without = np.asarray(Q) - signs[name] * weights[name] * normalized[name]
        assignment, score = maximum_assignment(without)
        output[f"without_{name}"] = {
            "assignment": list(assignment),
            "overlap_with_full": assignment_overlap(full_assignment, assignment),
            "score_under_reduced_Q": score,
        }
    return output
