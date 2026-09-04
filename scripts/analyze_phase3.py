from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedrad.diagnostics import (
    assignment_overlap,
    component_contribution_assignments,
    double_center,
    interaction_metrics,
    maximum_assignment,
    null_hungarian_gammas,
    pearson_flat,
    random_assignment_scores,
    spearman_flat,
    within_row_spearman,
)


COMPONENTS = ("G", "A", "D", "C", "Q")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _assignment_from_log(row: dict[str, Any], key: str) -> tuple[int, ...]:
    return tuple(int(pair["task_id"]) for pair in row[key])


def _probe_validity(run_dir: Path) -> dict[int, float]:
    with (run_dir / "probe_pairs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[bool]] = {}
    for row in rows:
        grouped.setdefault(int(row["round"]), []).append(
            row["alignment_valid"].lower() == "true"
        )
    return {
        round_number: sum(values) / len(values)
        for round_number, values in grouped.items()
    }


def _analyze_main_trajectory(
    run_dir: Path, *, random_samples: int, null_samples: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    rounds = _jsonl(run_dir / "rounds.jsonl")
    assignments = _jsonl(run_dir / "assignments.jsonl")
    validity = _probe_validity(run_dir)
    dynamics: list[dict[str, Any]] = []
    contributions: dict[str, list[float]] = {
        f"without_{name}": [] for name in ("G", "A", "D", "C")
    }
    null_all: list[np.ndarray] = []
    random_all: list[np.ndarray] = []
    interaction_assignment_equal: list[bool] = []
    raw_ga: list[float] = []
    interaction_ga: list[float] = []
    row_ga_means: list[float] = []
    row_ga_stds: list[float] = []

    for index, (round_row, assignment_row) in enumerate(zip(rounds, assignments)):
        round_number = int(round_row["round_number"])
        score_file = np.load(
            run_dir / "scores" / f"round_{round_number:04d}.npz",
            allow_pickle=False,
        )
        matrices = {
            name: np.asarray(score_file[name], dtype=np.float64)
            for name in COMPONENTS
        }
        metrics = {name: interaction_metrics(matrix) for name, matrix in matrices.items()}
        q_assignment, q_score = maximum_assignment(matrices["Q"])
        q_int_assignment, _ = maximum_assignment(double_center(matrices["Q"]))
        interaction_assignment_equal.append(q_assignment == q_int_assignment)

        ga_raw = pearson_flat(matrices["G"], matrices["A"])
        ga_interaction = pearson_flat(
            double_center(matrices["G"]), double_center(matrices["A"])
        )
        ga_row_mean, ga_row_std, _ = within_row_spearman(
            matrices["G"], matrices["A"]
        )
        raw_ga.append(ga_raw)
        interaction_ga.append(ga_interaction)
        row_ga_means.append(ga_row_mean)
        row_ga_stds.append(ga_row_std)

        weights = {"G": 1.0, "A": 1.0, "D": 1.0, "C": 0.25}
        normalized = {
            name: np.asarray(score_file[f"Z{name}"], dtype=np.float64)
            for name in ("G", "A", "D", "C")
        }
        contribution = component_contribution_assignments(
            Q=matrices["Q"], normalized=normalized, weights=weights
        )
        for name, report in contribution.items():
            contributions[name].append(float(report["overlap_with_full"]))

        random_scores = random_assignment_scores(
            matrices["Q"], samples=random_samples, seed=seed + round_number
        )
        random_all.append(random_scores)
        null_gamma = null_hungarian_gammas(
            matrices["Q"],
            samples=null_samples,
            seed=seed + 100_000 + round_number,
        )
        null_all.append(null_gamma)
        baseline_tasks = _assignment_from_log(assignment_row, "baseline_pairs")
        hungarian_tasks = _assignment_from_log(assignment_row, "hungarian_pairs")
        positive_d = matrices["D"][matrices["D"] > 0.0]
        row: dict[str, Any] = {
            "round": round_number,
            "accuracy": round_row["diagnostic_accuracy"],
            "G_mean": float(matrices["G"].mean()),
            "A_mean": float(matrices["A"].mean()),
            "D_positive_ratio": float(np.mean(matrices["D"] > 0.0)),
            "D_mean_positive": (
                float(positive_d.mean()) if positive_d.size else 0.0
            ),
            "C_mean": float(matrices["C"].mean()),
            "alignment_valid_ratio": validity[round_number],
            "Gamma": assignment_row["Gamma"],
            "hungarian_score": assignment_row["hungarian_score"],
            "baseline_score": assignment_row["baseline_score"],
            "changed_pairs": int(
                sum(a != b for a, b in zip(baseline_tasks, hungarian_tasks))
            ),
            "Q_vs_Qint_assignment_equal": q_assignment == q_int_assignment,
            "G_A_raw_pearson": ga_raw,
            "G_A_interaction_pearson": ga_interaction,
            "G_A_within_row_spearman_mean": ga_row_mean,
            "G_A_within_row_spearman_std": ga_row_std,
            "random_score_mean": float(random_scores.mean()),
            "random_score_std": float(random_scores.std(ddof=0)),
            "random_score_p90": float(np.percentile(random_scores, 90)),
            "random_score_p95": float(np.percentile(random_scores, 95)),
            "random_score_p99": float(np.percentile(random_scores, 99)),
            "null_gamma_mean": float(null_gamma.mean()),
            "null_gamma_std": float(null_gamma.std(ddof=0)),
            "null_gamma_p90": float(np.percentile(null_gamma, 90)),
            "null_gamma_p95": float(np.percentile(null_gamma, 95)),
            "null_gamma_p99": float(np.percentile(null_gamma, 99)),
            "mean_reset_delta_norm": round_row["mean_reset_delta_norm"],
            "mean_active_reset_kernels": round_row["mean_active_reset_kernels"],
            "probe_seconds": round_row["probe_seconds"],
            "reliability_probe_seconds": round_row["reliability_probe_seconds"],
            "formal_training_seconds": round_row["formal_training_seconds"],
            "matching_seconds": round_row["matching_seconds"],
            "peak_gpu_memory_bytes": round_row["peak_gpu_memory_bytes"],
        }
        for component, component_metrics in metrics.items():
            for metric, value in component_metrics.items():
                row[f"{component}_{metric}"] = value
        for name, report in contribution.items():
            row[f"{name}_overlap"] = report["overlap_with_full"]
        dynamics.append(row)
        score_file.close()

    null_pooled = np.concatenate(null_all)
    random_pooled = np.concatenate(random_all)
    component_interaction = {
        component: {
            metric: _summary([row[f"{component}_{metric}"] for row in dynamics])
            for metric in (
                "overall_std",
                "mean_row_std",
                "mean_col_std",
                "interaction_std",
                "interaction_overall_ratio",
            )
        }
        for component in COMPONENTS
    }
    candidate_tau = float(np.percentile(null_pooled, 95))
    aggregate = {
        "round_count": len(dynamics),
        "Q_vs_Qint_assignment_equal_all_rounds": all(interaction_assignment_equal),
        "component_interaction_metrics": component_interaction,
        "G_A_redundancy": {
            "raw_pearson_across_rounds": _summary(raw_ga),
            "interaction_pearson_across_rounds": _summary(interaction_ga),
            "within_client_spearman_round_mean": _summary(row_ga_means),
            "within_client_spearman_round_std": _summary(row_ga_stds),
        },
        "component_contribution_overlap": {
            name: _summary(values) for name, values in contributions.items()
        },
        "observed_Gamma": _summary([row["Gamma"] for row in dynamics]),
        "random_assignment_score": _summary(random_pooled),
        "null_Gamma": _summary(null_pooled),
        "candidate_tau_p95": candidate_tau,
        "candidate_tau_status": "offline null-calibration candidate; not formal",
        "observed_round_fraction_above_candidate_tau": float(
            np.mean([row["Gamma"] >= candidate_tau for row in dynamics])
        ),
        "recovery_dynamics": {
            "D_positive_ratio": _summary(
                [row["D_positive_ratio"] for row in dynamics]
            ),
            "D_mean_positive": _summary(
                [row["D_mean_positive"] for row in dynamics]
            ),
            "alignment_valid_ratio": _summary(
                [row["alignment_valid_ratio"] for row in dynamics]
            ),
            "changed_pairs": _summary([row["changed_pairs"] for row in dynamics]),
            "mean_reset_delta_norm": _summary(
                [row["mean_reset_delta_norm"] for row in dynamics]
            ),
            "mean_active_reset_kernels": _summary(
                [row["mean_active_reset_kernels"] for row in dynamics]
            ),
            "selected_round_snapshots": {
                str(row["round"]): row
                for row in dynamics
                if row["round"] in {1, 5, 10, 20, 40}
            },
        },
        "cost": {
            "probe_seconds": _summary([row["probe_seconds"] for row in dynamics]),
            "formal_training_seconds": _summary(
                [row["formal_training_seconds"] for row in dynamics]
            ),
            "matching_seconds": _summary(
                [row["matching_seconds"] for row in dynamics]
            ),
            "total_reliability_probe_seconds": float(
                sum(row["reliability_probe_seconds"] for row in dynamics)
            ),
            "peak_gpu_memory_bytes": int(
                max(row["peak_gpu_memory_bytes"] for row in dynamics)
            ),
        },
    }
    return dynamics, aggregate, null_pooled


def _analyze_reliability(run_dir: Path) -> dict[str, Any]:
    assignment_rows = _jsonl(run_dir / "probe_reliability_assignments.jsonl")
    grouped_assignments: dict[int, dict[int, dict[str, Any]]] = {}
    for row in assignment_rows:
        grouped_assignments.setdefault(int(row["round_number"]), {})[
            int(row["probe_replicate"])
        ] = row

    per_round: list[dict[str, Any]] = []
    correlation_accumulator: dict[str, list[float]] = {}
    overlaps: list[float] = []
    exact_matches: list[bool] = []
    all_gammas: list[float] = []
    for round_number in sorted(grouped_assignments):
        replicas: dict[int, dict[str, np.ndarray]] = {}
        for replicate in sorted(grouped_assignments[round_number]):
            score_file = np.load(
                run_dir
                / "reliability_scores"
                / f"round_{round_number:04d}_rep_{replicate}.npz",
                allow_pickle=False,
            )
            replicas[replicate] = {
                name: np.asarray(score_file[name], dtype=np.float64)
                for name in COMPONENTS
            }
            score_file.close()
        pair_reports: list[dict[str, Any]] = []
        for left, right in combinations(sorted(replicas), 2):
            report: dict[str, Any] = {"replicates": [left, right]}
            for component in COMPONENTS:
                left_int = double_center(replicas[left][component])
                right_int = double_center(replicas[right][component])
                for correlation_name, value in (
                    ("pearson", pearson_flat(left_int, right_int)),
                    ("spearman", spearman_flat(left_int, right_int)),
                ):
                    key = f"{component}_int_{correlation_name}"
                    report[key] = value
                    correlation_accumulator.setdefault(key, []).append(value)
            left_assignment = _assignment_from_log(
                grouped_assignments[round_number][left], "hungarian_pairs"
            )
            right_assignment = _assignment_from_log(
                grouped_assignments[round_number][right], "hungarian_pairs"
            )
            overlap = assignment_overlap(left_assignment, right_assignment)
            exact = left_assignment == right_assignment
            report["hungarian_overlap"] = overlap
            report["exact_same_assignment"] = exact
            overlaps.append(overlap)
            exact_matches.append(exact)
            pair_reports.append(report)
        gammas = [
            float(grouped_assignments[round_number][replicate]["Gamma"])
            for replicate in sorted(grouped_assignments[round_number])
        ]
        all_gammas.extend(gammas)
        per_round.append(
            {
                "round": round_number,
                "pairwise": pair_reports,
                "Gamma": _summary(gammas),
            }
        )
    return {
        "per_round": per_round,
        "aggregate_interaction_correlations": {
            key: _summary(values) for key, values in correlation_accumulator.items()
        },
        "hungarian_overlap": _summary(overlaps),
        "exact_assignment_fraction": float(np.mean(exact_matches)),
        "Gamma_all_replicates": _summary(all_gammas),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_dynamics(rows: list[dict[str, Any]], path: Path) -> None:
    rounds = [row["round"] for row in rows]
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for component, axis in zip(("G", "A", "C"), axes.flat[:3]):
        axis.plot(rounds, [row[f"{component}_overall_std"] for row in rows], label="overall std")
        axis.plot(rounds, [row[f"{component}_interaction_std"] for row in rows], label="interaction std")
        axis.set_title(component)
        axis.legend()
    axes[1, 1].plot(rounds, [row["D_positive_ratio"] for row in rows])
    axes[1, 1].set_title("D positive ratio")
    axes[2, 0].plot(rounds, [row["Q_overall_std"] for row in rows], label="Q std")
    axes[2, 0].plot(rounds, [row["Q_interaction_std"] for row in rows], label="Q interaction std")
    axes[2, 0].legend()
    axes[2, 0].set_title("Q")
    axes[2, 1].plot(rounds, [row["Gamma"] for row in rows], label="Gamma")
    axes[2, 1].plot(rounds, [row["null_gamma_p95"] for row in rows], label="per-round null p95")
    axes[2, 1].legend()
    axes[2, 1].set_title("Observed vs null")
    for axis in axes.flat:
        axis.set_xlabel("round")
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_accuracy(
    clean_rows: list[dict[str, Any]], fedrad_rows: list[dict[str, Any]], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(
        [row["round_number"] for row in clean_rows],
        [row["diagnostic_accuracy"] for row in clean_rows],
        label="clean FedPhoenix",
    )
    axis.plot(
        [row["round_number"] for row in fedrad_rows],
        [row["diagnostic_accuracy"] for row in fedrad_rows],
        label="forced-Hungarian FedRAD",
    )
    axis.set_xlabel("round")
    axis.set_ylabel("diagnostic full accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--fedrad-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=910_003)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics, aggregate, _ = _analyze_main_trajectory(
        args.fedrad_run,
        random_samples=args.random_samples,
        null_samples=args.null_samples,
        seed=args.seed,
    )
    reliability = _analyze_reliability(args.fedrad_run)
    clean_rounds = _jsonl(args.clean_run / "rounds.jsonl")
    fedrad_rounds = _jsonl(args.fedrad_run / "rounds.jsonl")
    accuracy_rows = [
        {
            "round": clean["round_number"],
            "clean_accuracy": clean["diagnostic_accuracy"],
            "fedrad_accuracy": fedrad["diagnostic_accuracy"],
        }
        for clean, fedrad in zip(clean_rounds, fedrad_rounds)
    ]
    report = {
        "clean_run": str(args.clean_run.resolve()),
        "fedrad_run": str(args.fedrad_run.resolve()),
        "trajectory": aggregate,
        "reliability": reliability,
        "short_accuracy": {
            "clean": _summary([row["diagnostic_accuracy"] for row in clean_rounds]),
            "fedrad": _summary([row["diagnostic_accuracy"] for row in fedrad_rounds]),
            "clean_final": clean_rounds[-1]["diagnostic_accuracy"],
            "fedrad_final": fedrad_rounds[-1]["diagnostic_accuracy"],
            "clean_last10_mean": float(
                np.mean([row["diagnostic_accuracy"] for row in clean_rounds[-10:]])
            ),
            "fedrad_last10_mean": float(
                np.mean([row["diagnostic_accuracy"] for row in fedrad_rounds[-10:]])
            ),
        },
        "calibration_protocol": {
            "random_assignments_per_round": args.random_samples,
            "independent_within_row_null_permutations_per_round": args.null_samples,
            "uses_test_accuracy": False,
        },
    }
    _write_csv(output_dir / "dynamics.csv", dynamics)
    _write_csv(output_dir / "accuracy_curves.csv", accuracy_rows)
    (output_dir / "phase3_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_dynamics(dynamics, output_dir / "recovery_dynamics.png")
    _plot_accuracy(clean_rounds, fedrad_rounds, output_dir / "accuracy_curves.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
