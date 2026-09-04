from __future__ import annotations

import argparse
import csv
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
    double_center,
    pearson_flat,
    second_best_assignment,
    spearman_flat,
)
from fedrad.scoring import matrix_zscore
from scripts.analyze_phase35_cross_probe import (
    _baseline_assignments,
    _candidate_matrices,
    _functional_report,
    _load_replicates,
    _random_permutations,
)


CANDIDATES = ("Full", "G")


def averaged_candidate_matrix(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    candidate: str,
    z_eps: float,
    std_atol: float,
    std_rtol: float,
) -> np.ndarray:
    normalized: dict[str, np.ndarray] = {}
    names = ("G",) if candidate == "G" else ("G", "A", "D", "C")
    if candidate not in CANDIDATES:
        raise ValueError(f"unsupported candidate: {candidate}")
    for name in names:
        raw_average = 0.5 * (left[name] + right[name])
        normalized[name], _ = matrix_zscore(
            raw_average,
            z_eps=z_eps,
            std_atol=std_atol,
            std_rtol=std_rtol,
        )
    if candidate == "G":
        return normalized["G"]
    return (
        normalized["G"]
        + normalized["A"]
        - normalized["D"]
        + 0.25 * normalized["C"]
    )


def _functional_percentile(report: dict[str, float]) -> float:
    return float(
        np.mean(
            [
                report["G_quality_percentile"],
                report["A_quality_percentile"],
                report["adapted_loss_quality_percentile"],
            ]
        )
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
    }


def _block_bootstrap_ci(
    rows: list[dict[str, Any]], key: str, *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    rounds = sorted({int(row["round"]) for row in rows})
    block_means = np.asarray(
        [
            np.mean([float(row[key]) for row in rows if int(row["round"]) == value])
            for value in rounds
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = block_means[
        rng.integers(0, len(block_means), size=(samples, len(block_means)))
    ].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir: Path, *, random_samples: int, seed: int) -> dict[str, Any]:
    replicas = _load_replicates(run_dir)
    baselines = _baseline_assignments(run_dir, replicas)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    z_options = {
        "z_eps": float(config["z_eps"]),
        "std_atol": float(config["std_atol"]),
        "std_rtol": float(config["std_rtol"]),
    }
    rows: list[dict[str, Any]] = []

    for round_number in sorted(replicas):
        round_replicas = replicas[round_number]
        single_candidates = {
            replicate: _candidate_matrices(data)
            for replicate, data in round_replicas.items()
        }
        for heldout_replicate in sorted(round_replicas):
            source_replicates = tuple(
                value
                for value in sorted(round_replicas)
                if value != heldout_replicate
            )
            random_assignments = _random_permutations(
                len(round_replicas[heldout_replicate]["task_order"]),
                random_samples,
                seed + 100 * round_number + heldout_replicate,
            )
            for candidate in CANDIDATES:
                heldout_matrix = single_candidates[heldout_replicate][candidate]
                heldout_optimal, _, _, _ = second_best_assignment(heldout_matrix)
                single_reports: list[dict[str, float]] = []
                single_q_pearson: list[float] = []
                single_q_spearman: list[float] = []
                single_heldout_overlap: list[float] = []
                single_exact: list[bool] = []
                single_margin_per_k: list[float] = []
                single_margin_relative: list[float] = []
                single_assignments: list[tuple[int, ...]] = []
                for source in source_replicates:
                    source_matrix = single_candidates[source][candidate]
                    assignment, best_score, _, second_score = second_best_assignment(
                        source_matrix
                    )
                    single_assignments.append(assignment)
                    single_reports.append(
                        _functional_report(
                            round_replicas[heldout_replicate],
                            assignment,
                            baselines[round_number],
                            random_assignments,
                        )
                    )
                    single_q_pearson.append(
                        pearson_flat(
                            double_center(source_matrix),
                            double_center(heldout_matrix),
                        )
                    )
                    single_q_spearman.append(
                        spearman_flat(
                            double_center(source_matrix),
                            double_center(heldout_matrix),
                        )
                    )
                    overlap = assignment_overlap(assignment, heldout_optimal)
                    single_heldout_overlap.append(overlap)
                    single_exact.append(assignment == heldout_optimal)
                    margin = best_score - second_score
                    single_margin_per_k.append(margin / source_matrix.shape[0])
                    single_margin_relative.append(
                        margin / float(source_matrix.std(ddof=0))
                    )

                averaged_matrix = averaged_candidate_matrix(
                    round_replicas[source_replicates[0]],
                    round_replicas[source_replicates[1]],
                    candidate=candidate,
                    **z_options,
                )
                avg_assignment, avg_best, _, avg_second = second_best_assignment(
                    averaged_matrix
                )
                avg_report = _functional_report(
                    round_replicas[heldout_replicate],
                    avg_assignment,
                    baselines[round_number],
                    random_assignments,
                )
                avg_margin = avg_best - avg_second
                single_functional = float(
                    np.mean([_functional_percentile(item) for item in single_reports])
                )
                averaged_functional = _functional_percentile(avg_report)
                single_overlap = float(np.mean(single_heldout_overlap))
                averaged_overlap = assignment_overlap(avg_assignment, heldout_optimal)
                single_q_p = float(np.mean(single_q_pearson))
                single_q_s = float(np.mean(single_q_spearman))
                avg_q_p = pearson_flat(
                    double_center(averaged_matrix), double_center(heldout_matrix)
                )
                avg_q_s = spearman_flat(
                    double_center(averaged_matrix), double_center(heldout_matrix)
                )
                row: dict[str, Any] = {
                    "candidate": candidate,
                    "round": round_number,
                    "source_replicates": json.dumps(source_replicates),
                    "heldout_replicate": heldout_replicate,
                    "single_Q_int_pearson": single_q_p,
                    "averaged_Q_int_pearson": avg_q_p,
                    "delta_Q_int_pearson": avg_q_p - single_q_p,
                    "single_Q_int_spearman": single_q_s,
                    "averaged_Q_int_spearman": avg_q_s,
                    "delta_Q_int_spearman": avg_q_s - single_q_s,
                    "single_functional_percentile": single_functional,
                    "averaged_functional_percentile": averaged_functional,
                    "delta_functional_percentile": (
                        averaged_functional - single_functional
                    ),
                    "single_heldout_optimal_overlap": single_overlap,
                    "averaged_heldout_optimal_overlap": averaged_overlap,
                    "delta_heldout_optimal_overlap": averaged_overlap - single_overlap,
                    "single_exact_heldout_fraction": float(np.mean(single_exact)),
                    "averaged_exact_heldout": bool(avg_assignment == heldout_optimal),
                    "source_single_pair_overlap": assignment_overlap(
                        single_assignments[0], single_assignments[1]
                    ),
                    "averaged_to_source_overlap": float(
                        np.mean(
                            [
                                assignment_overlap(avg_assignment, assignment)
                                for assignment in single_assignments
                            ]
                        )
                    ),
                    "single_margin_per_K": float(np.mean(single_margin_per_k)),
                    "averaged_margin_per_K": avg_margin / averaged_matrix.shape[0],
                    "delta_margin_per_K": (
                        avg_margin / averaged_matrix.shape[0]
                        - float(np.mean(single_margin_per_k))
                    ),
                    "single_margin_relative_Q_std": float(
                        np.mean(single_margin_relative)
                    ),
                    "averaged_margin_relative_Q_std": (
                        avg_margin / float(averaged_matrix.std(ddof=0))
                    ),
                    "delta_margin_relative_Q_std": (
                        avg_margin / float(averaged_matrix.std(ddof=0))
                        - float(np.mean(single_margin_relative))
                    ),
                }
                for metric in ("G", "A", "adapted_loss"):
                    row[f"single_{metric}_vs_baseline"] = float(
                        np.mean(
                            [item[f"{metric}_vs_baseline"] for item in single_reports]
                        )
                    )
                    row[f"averaged_{metric}_vs_baseline"] = avg_report[
                        f"{metric}_vs_baseline"
                    ]
                    row[f"single_{metric}_quality_percentile"] = float(
                        np.mean(
                            [
                                item[f"{metric}_quality_percentile"]
                                for item in single_reports
                            ]
                        )
                    )
                    row[f"averaged_{metric}_quality_percentile"] = avg_report[
                        f"{metric}_quality_percentile"
                    ]
                rows.append(row)

    aggregates: dict[str, dict[str, Any]] = {}
    delta_keys = (
        "delta_Q_int_pearson",
        "delta_Q_int_spearman",
        "delta_functional_percentile",
        "delta_heldout_optimal_overlap",
        "delta_margin_per_K",
        "delta_margin_relative_Q_std",
    )
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        aggregate: dict[str, Any] = {
            "cases": len(selected),
            "single": {
                "Q_int_pearson": float(
                    np.mean([row["single_Q_int_pearson"] for row in selected])
                ),
                "Q_int_spearman": float(
                    np.mean([row["single_Q_int_spearman"] for row in selected])
                ),
                "functional_percentile": float(
                    np.mean(
                        [row["single_functional_percentile"] for row in selected]
                    )
                ),
                "heldout_optimal_overlap": float(
                    np.mean(
                        [row["single_heldout_optimal_overlap"] for row in selected]
                    )
                ),
                "exact_heldout_fraction": float(
                    np.mean(
                        [row["single_exact_heldout_fraction"] for row in selected]
                    )
                ),
                "margin_per_K": float(
                    np.mean([row["single_margin_per_K"] for row in selected])
                ),
                "margin_relative_Q_std": float(
                    np.mean(
                        [row["single_margin_relative_Q_std"] for row in selected]
                    )
                ),
            },
            "averaged": {
                "Q_int_pearson": float(
                    np.mean([row["averaged_Q_int_pearson"] for row in selected])
                ),
                "Q_int_spearman": float(
                    np.mean([row["averaged_Q_int_spearman"] for row in selected])
                ),
                "functional_percentile": float(
                    np.mean(
                        [row["averaged_functional_percentile"] for row in selected]
                    )
                ),
                "heldout_optimal_overlap": float(
                    np.mean(
                        [row["averaged_heldout_optimal_overlap"] for row in selected]
                    )
                ),
                "exact_heldout_fraction": float(
                    np.mean([row["averaged_exact_heldout"] for row in selected])
                ),
                "margin_per_K": float(
                    np.mean([row["averaged_margin_per_K"] for row in selected])
                ),
                "margin_relative_Q_std": float(
                    np.mean(
                        [row["averaged_margin_relative_Q_std"] for row in selected]
                    )
                ),
                "to_source_overlap": float(
                    np.mean(
                        [row["averaged_to_source_overlap"] for row in selected]
                    )
                ),
            },
            "source_single_pair_overlap": float(
                np.mean([row["source_single_pair_overlap"] for row in selected])
            ),
            "deltas": {},
        }
        for index, key in enumerate(delta_keys):
            low, high = _block_bootstrap_ci(
                selected, key, seed=seed + 10_000 + index
            )
            aggregate["deltas"][key] = {
                **_summary([row[key] for row in selected]),
                "round_block_bootstrap_ci95": [low, high],
                "positive_fraction": float(
                    np.mean([float(row[key]) > 0.0 for row in selected])
                ),
            }
        for metric in ("G", "A", "adapted_loss"):
            aggregate["single"][f"{metric}_quality_percentile"] = float(
                np.mean(
                    [row[f"single_{metric}_quality_percentile"] for row in selected]
                )
            )
            aggregate["averaged"][f"{metric}_quality_percentile"] = float(
                np.mean(
                    [
                        row[f"averaged_{metric}_quality_percentile"]
                        for row in selected
                    ]
                )
            )
        directional = {
            "interaction_reliability_improved": (
                aggregate["deltas"]["delta_Q_int_pearson"]["mean"] > 0.0
                and aggregate["deltas"]["delta_Q_int_spearman"]["mean"] > 0.0
            ),
            "functional_recovery_non_decreasing": (
                aggregate["deltas"]["delta_functional_percentile"]["mean"]
                >= 0.0
            ),
            "assignment_overlap_improved": (
                aggregate["deltas"]["delta_heldout_optimal_overlap"]["mean"]
                > 0.0
            ),
        }
        aggregate["directional_checks"] = directional
        aggregate["all_three_directional_checks_pass"] = bool(
            all(directional.values())
        )
        aggregates[candidate] = aggregate

    part_b_trigger = bool(
        all(
            aggregates[candidate]["all_three_directional_checks_pass"]
            for candidate in CANDIDATES
        )
    )
    return {
        "protocol": {
            "rounds": sorted(replicas),
            "replicates": 3,
            "averaged_cases_per_candidate": 15,
            "fair_single_evaluations_per_candidate": 30,
            "random_assignments_per_heldout": random_samples,
            "averages_raw_components_then_reapplies_matrix_zscore": True,
            "new_probe_forward_passes": 0,
            "uses_test_accuracy": False,
            "changes_formal_score_or_gate": False,
            "part_b_trigger_rule": (
                "Both Full and G must improve mean Pearson and Spearman, not "
                "decrease mean held-out functional percentile, and improve mean "
                "overlap with held-out optimal assignment."
            ),
        },
        "aggregates": aggregates,
        "part_b_trigger": part_b_trigger,
        "cases": rows,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3.6 Part A — Two-Probe Averaging",
        "",
        "| Candidate | Estimator | Functional pct | Q-int P/S | Held-out overlap | Exact | Margin/K | Margin/Q std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES:
        aggregate = report["aggregates"][candidate]
        for key, label in (("single", "Single 32/32"), ("averaged", "Average 2x32/32")):
            row = aggregate[key]
            lines.append(
                f'| {candidate} | {label} | {row["functional_percentile"]:.2f} | '
                f'{row["Q_int_pearson"]:.3f}/{row["Q_int_spearman"]:.3f} | '
                f'{row["heldout_optimal_overlap"]:.3f} | '
                f'{row["exact_heldout_fraction"]:.3f} | {row["margin_per_K"]:.4f} | '
                f'{row["margin_relative_Q_std"]:.4f} |'
            )
    lines.extend(
        [
            "",
            f'Part B trigger: **{report["part_b_trigger"]}**',
            "",
            report["protocol"]["part_b_trigger_rule"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, report: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = (
        ("functional_percentile", "Held-out functional percentile"),
        ("heldout_optimal_overlap", "Overlap with held-out optimum"),
        ("Q_int_pearson", "Q-int Pearson"),
    )
    x = np.arange(len(CANDIDATES))
    width = 0.34
    for axis, (metric, title) in zip(axes, panels):
        single = [report["aggregates"][name]["single"][metric] for name in CANDIDATES]
        averaged = [report["aggregates"][name]["averaged"][metric] for name in CANDIDATES]
        axis.bar(x - width / 2, single, width, label="single")
        axis.bar(x + width / 2, averaged, width, label="average 2 probes")
        axis.set_xticks(x, CANDIDATES)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedrad-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=936_001)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(
        args.fedrad_run.resolve(),
        random_samples=args.random_samples,
        seed=args.seed,
    )
    (output_dir / "part_a_two_probe_averaging.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "part_a_cases.csv", report["cases"])
    _write_markdown(output_dir / "part_a_summary.md", report)
    _plot(output_dir / "part_a_comparison.png", report)
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "aggregates": report["aggregates"],
                "part_b_trigger": report["part_b_trigger"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
