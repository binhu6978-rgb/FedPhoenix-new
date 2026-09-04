from __future__ import annotations

import argparse
import csv
import importlib.util
from itertools import combinations, permutations
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

_FALLBACK_SITE = os.environ.get("FEDRAD_FALLBACK_SITE_PACKAGES")
_FALLBACK_DLLS = os.environ.get("FEDRAD_FALLBACK_DLL_DIRS", "")
if _FALLBACK_SITE:
    sys.path.insert(0, _FALLBACK_SITE)
if os.name == "nt":
    for _directory in filter(None, _FALLBACK_DLLS.split(os.pathsep)):
        os.add_dll_directory(_directory)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load the NumPy/SciPy-only diagnostic module directly.  Importing the package
# root also imports the training stack (and therefore PyTorch), which is neither
# required nor desirable for an offline analysis script.
_DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "fedrad_offline_diagnostics", PROJECT_ROOT / "fedrad" / "diagnostics.py"
)
if _DIAGNOSTICS_SPEC is None or _DIAGNOSTICS_SPEC.loader is None:
    raise RuntimeError("could not load fedrad diagnostics")
_DIAGNOSTICS = importlib.util.module_from_spec(_DIAGNOSTICS_SPEC)
_DIAGNOSTICS_SPEC.loader.exec_module(_DIAGNOSTICS)
assignment_overlap = _DIAGNOSTICS.assignment_overlap
assignment_score = _DIAGNOSTICS.assignment_score
double_center = _DIAGNOSTICS.double_center
pearson_flat = _DIAGNOSTICS.pearson_flat
second_best_assignment = _DIAGNOSTICS.second_best_assignment
spearman_flat = _DIAGNOSTICS.spearman_flat


CANDIDATE_NAMES = ("Full", "G", "A", "C", "G+C", "A+C", "G+A")
FUNCTIONAL_METRICS = ("G", "A", "adapted_loss")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary requires finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values))


def _midrank_percentile(
    value: float, reference: np.ndarray, *, higher_is_better: bool = True
) -> float:
    samples = np.asarray(reference, dtype=np.float64)
    if higher_is_better:
        count = np.sum(samples < value) + 0.5 * np.sum(samples == value)
    else:
        count = np.sum(samples > value) + 0.5 * np.sum(samples == value)
    return float(100.0 * count / samples.size)


def _random_permutations(size: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.stack([rng.permutation(size) for _ in range(samples)])


def _scores_for_permutations(
    matrix: np.ndarray, assignments: np.ndarray, *, average: bool = False
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    rows = np.arange(values.shape[0])[None, :]
    scores = values[rows, assignments].sum(axis=1)
    return scores / values.shape[0] if average else scores


def _candidate_matrices(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    zg, za, zd, zc = (data[name] for name in ("ZG", "ZA", "ZD", "ZC"))
    candidates = {
        "Full": data["Q"],
        "G": zg,
        "A": za,
        "C": zc,
        "G+C": zg + 0.25 * zc,
        "A+C": za + 0.25 * zc,
        "G+A": zg + za,
    }
    if not np.allclose(
        candidates["Full"], zg + za - zd + 0.25 * zc, atol=1e-12, rtol=0
    ):
        raise ValueError("saved Full Q does not match the formal score")
    return candidates


def _load_replicates(run_dir: Path) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    score_paths = sorted((run_dir / "reliability_scores").glob("*.npz"))
    replicas: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for path in score_paths:
        stem = path.stem.split("_")
        round_number, replicate = int(stem[1]), int(stem[3])
        with np.load(path, allow_pickle=False) as saved:
            replicas.setdefault(round_number, {})[replicate] = {
                name: np.asarray(saved[name], dtype=np.float64)
                for name in ("G", "A", "D", "C", "ZG", "ZA", "ZD", "ZC", "Q")
            }
            replicas[round_number][replicate]["client_order"] = np.asarray(
                saved["client_order"], dtype=np.int64
            )
            replicas[round_number][replicate]["task_order"] = np.asarray(
                saved["task_order"], dtype=np.int64
            )

    raw_rows: dict[tuple[int, int], list[dict[str, str]]] = {}
    with (run_dir / "probe_reliability_pairs.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            key = (int(row["round"]), int(row["probe_replicate"]))
            raw_rows.setdefault(key, []).append(row)

    for round_number, round_replicas in replicas.items():
        if set(round_replicas) != {0, 1, 2}:
            raise ValueError(f"round {round_number} does not contain three replicates")
        for replicate, data in round_replicas.items():
            clients = tuple(int(value) for value in data["client_order"])
            tasks = tuple(int(value) for value in data["task_order"])
            by_client = {value: index for index, value in enumerate(clients)}
            by_task = {value: index for index, value in enumerate(tasks)}
            rows = raw_rows[(round_number, replicate)]
            if len(rows) != len(clients) * len(tasks):
                raise ValueError("raw reliability grid is incomplete")
            for name in ("global_loss", "reset_loss", "adapted_loss"):
                matrix = np.full((len(clients), len(tasks)), np.nan, dtype=np.float64)
                for row in rows:
                    matrix[
                        by_client[int(row["client_id"])], by_task[int(row["task_id"])]
                    ] = float(row[name])
                if not np.isfinite(matrix).all():
                    raise ValueError(f"incomplete {name} matrix")
                data[name] = matrix
            for name in ("G", "A", "D", "C"):
                matrix = np.full((len(clients), len(tasks)), np.nan, dtype=np.float64)
                for row in rows:
                    matrix[
                        by_client[int(row["client_id"])], by_task[int(row["task_id"])]
                    ] = float(row[name])
                if not np.allclose(matrix, data[name], atol=1e-12, rtol=0):
                    raise ValueError(f"CSV and NPZ disagree for {name}")
    return replicas


def _baseline_assignments(
    run_dir: Path, replicas: dict[int, dict[int, dict[str, np.ndarray]]]
) -> dict[int, tuple[int, ...]]:
    output: dict[int, tuple[int, ...]] = {}
    for row in _jsonl(run_dir / "assignments.jsonl"):
        round_number = int(row["round_number"])
        if round_number not in replicas:
            continue
        reference = replicas[round_number][0]
        tasks = tuple(int(value) for value in reference["task_order"])
        task_column = {task_id: column for column, task_id in enumerate(tasks)}
        ordered = sorted(row["baseline_pairs"], key=lambda pair: int(pair["client_row"]))
        assignment = tuple(task_column[int(pair["task_id"])] for pair in ordered)
        if set(assignment) != set(range(len(tasks))):
            raise ValueError("baseline assignment is not one-to-one")
        output[round_number] = assignment
    if set(output) != set(replicas):
        raise ValueError("missing baseline assignment for representative round")
    return output


def _functional_report(
    heldout: dict[str, np.ndarray],
    assignment: tuple[int, ...],
    baseline: tuple[int, ...],
    random_assignments: np.ndarray,
) -> dict[str, float]:
    report: dict[str, float] = {}
    for metric in FUNCTIONAL_METRICS:
        matrix = heldout[metric]
        cross = assignment_score(matrix, assignment) / matrix.shape[0]
        base = assignment_score(matrix, baseline) / matrix.shape[0]
        random = _scores_for_permutations(
            matrix, random_assignments, average=True
        )
        higher = metric != "adapted_loss"
        report.update(
            {
                f"heldout_{metric}": cross,
                f"baseline_{metric}": base,
                f"{metric}_vs_baseline": cross - base,
                f"random_{metric}_mean": float(random.mean()),
                f"random_{metric}_std": float(random.std(ddof=0)),
                f"random_{metric}_p90": float(np.percentile(random, 90)),
                f"random_{metric}_p95": float(np.percentile(random, 95)),
                f"{metric}_quality_percentile": _midrank_percentile(
                    cross, random, higher_is_better=higher
                ),
            }
        )
    return report


def _rank_candidates(aggregates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    criteria = (
        "cross_random_percentile",
        "cross_vs_baseline_random_std",
        "functional_quality_percentile",
        "reliability_index",
        "pair_overlap",
        "margin_relative_to_q_std",
    )
    criterion_ranks: dict[str, np.ndarray] = {}
    for criterion in criteria:
        values = np.asarray(
            [aggregates[name][criterion] for name in CANDIDATE_NAMES],
            dtype=np.float64,
        )
        criterion_ranks[criterion] = rankdata(-values, method="average")
    score_ranks = rankdata(
        -np.asarray(
            [aggregates[name]["cross_random_percentile"] for name in CANDIDATE_NAMES]
        ),
        method="average",
    )
    functional_ranks = rankdata(
        -np.asarray(
            [
                aggregates[name]["functional_quality_percentile"]
                for name in CANDIDATE_NAMES
            ]
        ),
        method="average",
    )
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(CANDIDATE_NAMES):
        aggregate = aggregates[name]
        row = {
            "candidate": name,
            "Q_int_pearson": aggregate["Q_int_pearson"],
            "Q_int_spearman": aggregate["Q_int_spearman"],
            "pair_overlap": aggregate["pair_overlap"],
            "exact_assignment_fraction": aggregate["exact_assignment_fraction"],
            "cross_vs_baseline": aggregate["cross_vs_baseline"],
            "cross_vs_baseline_random_std": aggregate[
                "cross_vs_baseline_random_std"
            ],
            "cross_random_percentile": aggregate["cross_random_percentile"],
            "cross_above_baseline_fraction": aggregate[
                "cross_above_baseline_fraction"
            ],
            "cross_above_random_p90_fraction": aggregate[
                "cross_above_random_p90_fraction"
            ],
            "heldout_G_vs_baseline": aggregate["heldout_G_vs_baseline"],
            "heldout_A_vs_baseline": aggregate["heldout_A_vs_baseline"],
            "adapted_loss_vs_baseline": aggregate[
                "adapted_loss_vs_baseline"
            ],
            "heldout_G_quality_percentile": aggregate[
                "heldout_G_quality_percentile"
            ],
            "heldout_A_quality_percentile": aggregate[
                "heldout_A_quality_percentile"
            ],
            "adapted_loss_quality_percentile": aggregate[
                "adapted_loss_quality_percentile"
            ],
            "functional_quality_percentile": aggregate[
                "functional_quality_percentile"
            ],
            "margin": aggregate["margin"],
            "margin_per_K": aggregate["margin_per_K"],
            "margin_relative_to_q_std": aggregate[
                "margin_relative_to_q_std"
            ],
        }
        rank_sum = 0.0
        for criterion in criteria:
            value = float(criterion_ranks[criterion][index])
            row[f"rank_{criterion}"] = value
            rank_sum += value
        row["rank_sum"] = rank_sum
        row["score_generalization_rank"] = float(score_ranks[index])
        row["functional_recovery_rank"] = float(functional_ranks[index])
        rows.append(row)
    diagnostic_order = sorted(
        rows, key=lambda row: (row["rank_sum"], row["candidate"])
    )
    for rank, row in enumerate(diagnostic_order, start=1):
        row["diagnostic_consensus_rank"] = rank
    rows.sort(
        key=lambda row: (
            row["functional_recovery_rank"],
            row["score_generalization_rank"],
            row["candidate"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["method_rank"] = rank
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, ranking: list[dict[str, Any]]) -> None:
    headers = (
        "Rank", "Candidate", "Q_int P/S", "Overlap", "Cross-base",
        "Random pct", "Held-out G Δ", "Held-out A Δ", "Adapt loss Δ", "Margin/K",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]
    for row in ranking:
        values = (
            str(row["method_rank"]),
            row["candidate"],
            f'{row["Q_int_pearson"]:.3f}/{row["Q_int_spearman"]:.3f}',
            f'{row["pair_overlap"]:.3f}',
            f'{row["cross_vs_baseline"]:.3f}',
            f'{row["cross_random_percentile"]:.1f}',
            f'{row["heldout_G_vs_baseline"]:.4f}',
            f'{row["heldout_A_vs_baseline"]:.4f}',
            f'{row["adapted_loss_vs_baseline"]:.4f}',
            f'{row["margin_per_K"]:.4f}',
        )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(path: Path, ranking: list[dict[str, Any]]) -> None:
    ordered = sorted(ranking, key=lambda row: CANDIDATE_NAMES.index(row["candidate"]))
    names = [row["candidate"] for row in ordered]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("cross_random_percentile", "Cross-probe random percentile", 50.0),
        ("functional_quality_percentile", "Functional quality percentile", 50.0),
        ("pair_overlap", "Hungarian pair overlap", 0.1),
        ("Q_int_pearson", "Interaction reliability (Pearson)", 0.0),
    )
    for axis, (key, title, reference) in zip(axes.flat, panels):
        axis.bar(names, [row[key] for row in ordered])
        axis.axhline(reference, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze(run_dir: Path, *, random_samples: int, seed: int) -> dict[str, Any]:
    replicas = _load_replicates(run_dir)
    baselines = _baseline_assignments(run_dir, replicas)
    rounds = sorted(replicas)
    candidate_data: dict[str, dict[str, list[Any]]] = {
        name: {
            "pearson": [], "spearman": [], "overlap": [], "exact": [],
            "cross": [], "margins": [], "margin_pair": [], "margin_overlap": [],
        }
        for name in CANDIDATE_NAMES
    }
    ordered_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    d_rows: list[dict[str, Any]] = []
    d_cross_rows: list[dict[str, Any]] = []

    for round_number in rounds:
        reps = replicas[round_number]
        candidates = {
            replicate: _candidate_matrices(data)
            for replicate, data in reps.items()
        }
        assignments: dict[str, dict[int, tuple[int, ...]]] = {
            name: {} for name in CANDIDATE_NAMES
        }
        margin_by_candidate: dict[str, dict[int, float]] = {
            name: {} for name in CANDIDATE_NAMES
        }
        for replicate, data in reps.items():
            a_minus_d_corr = pearson_flat(
                double_center(data["A"]), -double_center(data["D"])
            )
            without_d = data["ZG"] + data["ZA"] + 0.25 * data["ZC"]
            full_assignment, _ = second_best_assignment(data["Q"])[:2]
            without_d_assignment, _ = second_best_assignment(without_d)[:2]
            d_rows.append(
                {
                    "round": round_number,
                    "replicate": replicate,
                    "D_positive_ratio": float(np.mean(data["D"] > 0.0)),
                    "corr_A_int_minus_D_int": a_minus_d_corr,
                    "full_without_D_overlap": assignment_overlap(
                        full_assignment, without_d_assignment
                    ),
                }
            )
            for name, matrix in candidates[replicate].items():
                best, best_score, second, second_score = second_best_assignment(matrix)
                margin = best_score - second_score
                q_std = float(matrix.std(ddof=0))
                assignments[name][replicate] = best
                margin_by_candidate[name][replicate] = margin
                candidate_data[name]["margins"].append(margin)
                margin_rows.append(
                    {
                        "candidate": name,
                        "round": round_number,
                        "replicate": replicate,
                        "best_score": best_score,
                        "second_best_score": second_score,
                        "absolute_margin": margin,
                        "margin_per_K": margin / matrix.shape[0],
                        "Q_std": q_std,
                        "margin_relative_to_Q_std": margin / q_std if q_std else 0.0,
                        "best_assignment": json.dumps(best),
                        "second_assignment": json.dumps(second),
                    }
                )

        for name in CANDIDATE_NAMES:
            for left, right in combinations(sorted(reps), 2):
                left_int = double_center(candidates[left][name])
                right_int = double_center(candidates[right][name])
                overlap = assignment_overlap(
                    assignments[name][left], assignments[name][right]
                )
                candidate_data[name]["pearson"].append(
                    pearson_flat(left_int, right_int)
                )
                candidate_data[name]["spearman"].append(
                    spearman_flat(left_int, right_int)
                )
                candidate_data[name]["overlap"].append(overlap)
                candidate_data[name]["exact"].append(
                    assignments[name][left] == assignments[name][right]
                )
                candidate_data[name]["margin_pair"].append(
                    0.5
                    * (
                        margin_by_candidate[name][left]
                        + margin_by_candidate[name][right]
                    )
                )
                candidate_data[name]["margin_overlap"].append(overlap)

        random_by_eval = {
            replicate: _random_permutations(
                len(data["task_order"]),
                random_samples,
                seed + round_number * 100 + replicate,
            )
            for replicate, data in reps.items()
        }
        for source, heldout_replicate in permutations(sorted(reps), 2):
            heldout = reps[heldout_replicate]
            random_assignments = random_by_eval[heldout_replicate]
            baseline = baselines[round_number]
            full_assignment = assignments["Full"][source]
            without_d_matrix = (
                reps[source]["ZG"]
                + reps[source]["ZA"]
                + 0.25 * reps[source]["ZC"]
            )
            without_d_assignment, _ = second_best_assignment(without_d_matrix)[:2]
            full_functional = _functional_report(
                heldout, full_assignment, baseline, random_assignments
            )
            without_d_functional = _functional_report(
                heldout, without_d_assignment, baseline, random_assignments
            )
            d_cross_rows.append(
                {
                    "round": round_number,
                    "source_replicate": source,
                    "heldout_replicate": heldout_replicate,
                    "without_D_vs_full_heldout_G": (
                        without_d_functional["heldout_G"]
                        - full_functional["heldout_G"]
                    ),
                    "without_D_vs_full_heldout_A": (
                        without_d_functional["heldout_A"]
                        - full_functional["heldout_A"]
                    ),
                    "without_D_vs_full_adapted_loss": (
                        without_d_functional["heldout_adapted_loss"]
                        - full_functional["heldout_adapted_loss"]
                    ),
                    "full_G_vs_baseline": full_functional["G_vs_baseline"],
                    "without_D_G_vs_baseline": without_d_functional[
                        "G_vs_baseline"
                    ],
                    "full_A_vs_baseline": full_functional["A_vs_baseline"],
                    "without_D_A_vs_baseline": without_d_functional[
                        "A_vs_baseline"
                    ],
                    "full_adapted_loss_vs_baseline": full_functional[
                        "adapted_loss_vs_baseline"
                    ],
                    "without_D_adapted_loss_vs_baseline": without_d_functional[
                        "adapted_loss_vs_baseline"
                    ],
                }
            )
            for name in CANDIDATE_NAMES:
                matrix = candidates[heldout_replicate][name]
                assignment = assignments[name][source]
                cross_score = assignment_score(matrix, assignment)
                baseline_score = assignment_score(matrix, baseline)
                random_scores = _scores_for_permutations(
                    matrix, random_assignments
                )
                row: dict[str, Any] = {
                    "candidate": name,
                    "round": round_number,
                    "source_replicate": source,
                    "heldout_replicate": heldout_replicate,
                    "cross_score": cross_score,
                    "baseline_score": baseline_score,
                    "cross_minus_baseline": cross_score - baseline_score,
                    "random_mean": float(random_scores.mean()),
                    "random_std": float(random_scores.std(ddof=0)),
                    "random_p90": float(np.percentile(random_scores, 90)),
                    "random_p95": float(np.percentile(random_scores, 95)),
                    "cross_percentile": _midrank_percentile(
                        cross_score, random_scores
                    ),
                    "cross_minus_baseline_random_std": (
                        (cross_score - baseline_score) / random_scores.std(ddof=0)
                        if random_scores.std(ddof=0) > 0.0
                        else 0.0
                    ),
                    "cross_above_baseline": bool(cross_score > baseline_score),
                    "cross_above_random_p90": bool(
                        cross_score > np.percentile(random_scores, 90)
                    ),
                }
                row.update(
                    _functional_report(
                        heldout, assignment, baseline, random_assignments
                    )
                )
                candidate_data[name]["cross"].append(row)
                ordered_rows.append(row)

    aggregates: dict[str, dict[str, Any]] = {}
    for name, data in candidate_data.items():
        cross_rows = data["cross"]
        pearson = _summary(data["pearson"])
        spearman = _summary(data["spearman"])
        margins = [row for row in margin_rows if row["candidate"] == name]
        functional_quality = [
            np.mean(
                [
                    row["G_quality_percentile"],
                    row["A_quality_percentile"],
                    row["adapted_loss_quality_percentile"],
                ]
            )
            for row in cross_rows
        ]
        aggregates[name] = {
            "Q_int_pearson": pearson["mean"],
            "Q_int_spearman": spearman["mean"],
            "Q_int_pearson_summary": pearson,
            "Q_int_spearman_summary": spearman,
            "reliability_index": 0.5 * (pearson["mean"] + spearman["mean"]),
            "pair_overlap": float(np.mean(data["overlap"])),
            "pair_overlap_summary": _summary(data["overlap"]),
            "exact_assignment_fraction": float(np.mean(data["exact"])),
            "cross_score": _mean(cross_rows, "cross_score"),
            "baseline_score": _mean(cross_rows, "baseline_score"),
            "cross_vs_baseline": _mean(cross_rows, "cross_minus_baseline"),
            "cross_vs_baseline_random_std": _mean(
                cross_rows, "cross_minus_baseline_random_std"
            ),
            "cross_random_percentile": _mean(cross_rows, "cross_percentile"),
            "cross_random_percentile_summary": _summary(
                [row["cross_percentile"] for row in cross_rows]
            ),
            "random_score_mean": _mean(cross_rows, "random_mean"),
            "random_score_std": _mean(cross_rows, "random_std"),
            "random_score_p90": _mean(cross_rows, "random_p90"),
            "random_score_p95": _mean(cross_rows, "random_p95"),
            "cross_above_baseline_fraction": float(
                np.mean([row["cross_above_baseline"] for row in cross_rows])
            ),
            "cross_above_random_p90_fraction": float(
                np.mean([row["cross_above_random_p90"] for row in cross_rows])
            ),
            "heldout_G": _mean(cross_rows, "heldout_G"),
            "heldout_A": _mean(cross_rows, "heldout_A"),
            "heldout_adapted_loss": _mean(cross_rows, "heldout_adapted_loss"),
            "heldout_G_vs_baseline": _mean(cross_rows, "G_vs_baseline"),
            "heldout_A_vs_baseline": _mean(cross_rows, "A_vs_baseline"),
            "adapted_loss_vs_baseline": _mean(
                cross_rows, "adapted_loss_vs_baseline"
            ),
            "heldout_G_quality_percentile": _mean(
                cross_rows, "G_quality_percentile"
            ),
            "heldout_A_quality_percentile": _mean(
                cross_rows, "A_quality_percentile"
            ),
            "adapted_loss_quality_percentile": _mean(
                cross_rows, "adapted_loss_quality_percentile"
            ),
            "functional_quality_percentile": float(np.mean(functional_quality)),
            "margin": _mean(margins, "absolute_margin"),
            "margin_per_K": _mean(margins, "margin_per_K"),
            "margin_relative_to_q_std": _mean(
                margins, "margin_relative_to_Q_std"
            ),
            "margin_overlap_pearson": pearson_flat(
                np.asarray(data["margin_pair"]),
                np.asarray(data["margin_overlap"]),
            ),
            "margin_overlap_spearman": spearman_flat(
                np.asarray(data["margin_pair"]),
                np.asarray(data["margin_overlap"]),
            ),
        }

    ranking = _rank_candidates(aggregates)
    per_round_candidate: list[dict[str, Any]] = []
    for name in CANDIDATE_NAMES:
        for round_number in rounds:
            selected = [
                row
                for row in ordered_rows
                if row["candidate"] == name and row["round"] == round_number
            ]
            per_round_candidate.append(
                {
                    "candidate": name,
                    "round": round_number,
                    "cross_vs_baseline": _mean(selected, "cross_minus_baseline"),
                    "cross_random_percentile": _mean(selected, "cross_percentile"),
                    "heldout_G_vs_baseline": _mean(selected, "G_vs_baseline"),
                    "heldout_A_vs_baseline": _mean(selected, "A_vs_baseline"),
                    "adapted_loss_vs_baseline": _mean(
                        selected, "adapted_loss_vs_baseline"
                    ),
                    "heldout_G_quality_percentile": _mean(
                        selected, "G_quality_percentile"
                    ),
                    "heldout_A_quality_percentile": _mean(
                        selected, "A_quality_percentile"
                    ),
                    "adapted_loss_quality_percentile": _mean(
                        selected, "adapted_loss_quality_percentile"
                    ),
                    "functional_quality_percentile": float(
                        np.mean(
                            [
                                _mean(selected, "G_quality_percentile"),
                                _mean(selected, "A_quality_percentile"),
                                _mean(selected, "adapted_loss_quality_percentile"),
                            ]
                        )
                    ),
                }
            )
    d_diagnostic = {
        "D_positive_ratio": _summary(
            [row["D_positive_ratio"] for row in d_rows]
        ),
        "corr_A_int_minus_D_int": _summary(
            [row["corr_A_int_minus_D_int"] for row in d_rows]
        ),
        "full_without_D_overlap": _summary(
            [row["full_without_D_overlap"] for row in d_rows]
        ),
        "without_D_vs_full_heldout_G": _summary(
            [row["without_D_vs_full_heldout_G"] for row in d_cross_rows]
        ),
        "without_D_vs_full_heldout_A": _summary(
            [row["without_D_vs_full_heldout_A"] for row in d_cross_rows]
        ),
        "without_D_vs_full_adapted_loss": _summary(
            [row["without_D_vs_full_adapted_loss"] for row in d_cross_rows]
        ),
        "without_D_G_vs_baseline": _summary(
            [row["without_D_G_vs_baseline"] for row in d_cross_rows]
        ),
        "without_D_A_vs_baseline": _summary(
            [row["without_D_A_vs_baseline"] for row in d_cross_rows]
        ),
        "without_D_adapted_loss_vs_baseline": _summary(
            [row["without_D_adapted_loss_vs_baseline"] for row in d_cross_rows]
        ),
    }
    return {
        "protocol": {
            "representative_rounds": rounds,
            "probe_replicates_per_round": 3,
            "ordered_cross_probe_pairs_per_round": 6,
            "random_assignments_per_heldout_matrix": random_samples,
            "candidate_formulas": {
                "Full": "ZG + ZA - ZD + 0.25 ZC",
                "G": "ZG",
                "A": "ZA",
                "C": "ZC",
                "G+C": "ZG + 0.25 ZC",
                "A+C": "ZA + 0.25 ZC",
                "G+A": "ZG + ZA",
            },
            "uses_saved_raw_matrices_only": True,
            "new_probe_forward_passes": 0,
            "uses_test_accuracy": False,
            "changes_formal_score_or_gate": False,
            "ranking_rule": (
                "method_rank prioritizes held-out functional recovery percentile, "
                "then cross-score random percentile; score-generalization and an "
                "equal-weight six-diagnostic consensus rank are also reported separately"
            ),
        },
        "candidate_aggregates": aggregates,
        "candidate_ranking": ranking,
        "per_round_candidate": per_round_candidate,
        "D_saturation_diagnostic": d_diagnostic,
        "ordered_cross_probe": ordered_rows,
        "margins": margin_rows,
        "D_detail": d_rows,
        "D_cross_probe_detail": d_cross_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedrad-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=935_001)
    args = parser.parse_args()
    if args.random_samples < 1:
        raise ValueError("random-samples must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(
        args.fedrad_run.resolve(),
        random_samples=args.random_samples,
        seed=args.seed,
    )
    (output_dir / "phase35_cross_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "candidate_ranking.csv", report["candidate_ranking"])
    _write_markdown(
        output_dir / "candidate_ranking.md", report["candidate_ranking"]
    )
    _write_csv(
        output_dir / "ordered_cross_probe.csv", report["ordered_cross_probe"]
    )
    _write_csv(
        output_dir / "candidate_per_round.csv", report["per_round_candidate"]
    )
    _write_csv(output_dir / "assignment_margins.csv", report["margins"])
    _write_csv(output_dir / "D_diagnostic.csv", report["D_detail"])
    _write_csv(
        output_dir / "D_cross_probe.csv", report["D_cross_probe_detail"]
    )
    _plot_summary(
        output_dir / "candidate_cross_probe_summary.png",
        report["candidate_ranking"],
    )
    print(json.dumps(
        {
            "protocol": report["protocol"],
            "candidate_ranking": report["candidate_ranking"],
            "D_saturation_diagnostic": report["D_saturation_diagnostic"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
