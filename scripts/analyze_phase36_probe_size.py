from __future__ import annotations

import argparse
from itertools import combinations, permutations
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
from scripts.analyze_phase35_cross_probe import (
    _baseline_assignments,
    _functional_report,
    _random_permutations,
)
from scripts.analyze_phase36_averaging import averaged_candidate_matrix
from scripts.run_phase36_probe_size import PROTOCOLS


CANDIDATES = ("Full", "G")
COMPONENTS = ("G", "A", "D", "C")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
    }


def _candidate(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "Full":
        return data["Q"]
    if name == "G":
        return data["ZG"]
    raise ValueError(name)


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


def _load_raw(
    diagnostic_dir: Path,
) -> dict[str, dict[int, dict[int, dict[str, np.ndarray]]]]:
    output: dict[str, dict[int, dict[int, dict[str, np.ndarray]]]] = {}
    for path in sorted((diagnostic_dir / "raw").glob("*.npz")):
        parts = path.stem.split("_")
        protocol = "_".join(parts[:3])
        round_number = int(parts[4])
        replicate = int(parts[6])
        with np.load(path, allow_pickle=False) as saved:
            data = {
                name: np.asarray(saved[name], dtype=np.float64)
                for name in (
                    "G", "A", "D", "C", "ZG", "ZA", "ZD", "ZC", "Q",
                    "global_loss", "reset_loss", "adapted_loss",
                )
            }
            data["client_order"] = np.asarray(saved["client_order"], dtype=np.int64)
            data["task_order"] = np.asarray(saved["task_order"], dtype=np.int64)
        output.setdefault(protocol, {}).setdefault(round_number, {})[replicate] = data
    if set(output) != set(PROTOCOLS):
        raise ValueError(f"missing protocols: {set(PROTOCOLS) - set(output)}")
    for protocol in output:
        if any(set(reps) != {0, 1, 2} for reps in output[protocol].values()):
            raise ValueError(f"incomplete replicates for {protocol}")
    return output


def _protocol_candidate(
    protocol: str,
    candidate: str,
    rounds: dict[int, dict[int, dict[str, np.ndarray]]],
    baselines: dict[int, tuple[int, ...]],
    *,
    random_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    component_correlations: dict[str, dict[str, list[float]]] = {
        name: {"pearson": [], "spearman": []} for name in COMPONENTS
    }
    q_pearson: list[float] = []
    q_spearman: list[float] = []
    overlaps: list[float] = []
    exact: list[bool] = []
    margin_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    for round_number, replicas in sorted(rounds.items()):
        assignments: dict[int, tuple[int, ...]] = {}
        for replicate, data in sorted(replicas.items()):
            matrix = _candidate(data, candidate)
            assignment, best, _, second = second_best_assignment(matrix)
            assignments[replicate] = assignment
            margin = best - second
            margin_rows.append(
                {
                    "protocol": protocol,
                    "candidate": candidate,
                    "round": round_number,
                    "replicate": replicate,
                    "margin_per_K": margin / matrix.shape[0],
                    "margin_relative_Q_std": margin / float(matrix.std(ddof=0)),
                }
            )
        for left, right in combinations(sorted(replicas), 2):
            left_q = _candidate(replicas[left], candidate)
            right_q = _candidate(replicas[right], candidate)
            q_pearson.append(
                pearson_flat(double_center(left_q), double_center(right_q))
            )
            q_spearman.append(
                spearman_flat(double_center(left_q), double_center(right_q))
            )
            overlaps.append(assignment_overlap(assignments[left], assignments[right]))
            exact.append(assignments[left] == assignments[right])
            if candidate == "Full":
                for component in COMPONENTS:
                    left_int = double_center(replicas[left][component])
                    right_int = double_center(replicas[right][component])
                    component_correlations[component]["pearson"].append(
                        pearson_flat(left_int, right_int)
                    )
                    component_correlations[component]["spearman"].append(
                        spearman_flat(left_int, right_int)
                    )
        random_by_heldout = {
            replicate: _random_permutations(
                len(data["task_order"]),
                random_samples,
                seed + round_number * 100 + replicate,
            )
            for replicate, data in replicas.items()
        }
        for source, heldout in permutations(sorted(replicas), 2):
            functional = _functional_report(
                replicas[heldout],
                assignments[source],
                baselines[round_number],
                random_by_heldout[heldout],
            )
            cross_rows.append(
                {
                    "protocol": protocol,
                    "candidate": candidate,
                    "round": round_number,
                    "source_replicate": source,
                    "heldout_replicate": heldout,
                    "functional_percentile": _functional_percentile(functional),
                    **functional,
                }
            )

    aggregate = {
        "Q_int_pearson": _summary(q_pearson),
        "Q_int_spearman": _summary(q_spearman),
        "pair_overlap": _summary(overlaps),
        "exact_assignment_fraction": float(np.mean(exact)),
        "functional_percentile": _summary(
            [row["functional_percentile"] for row in cross_rows]
        ),
        "heldout_G_quality_percentile": _summary(
            [row["G_quality_percentile"] for row in cross_rows]
        ),
        "heldout_A_quality_percentile": _summary(
            [row["A_quality_percentile"] for row in cross_rows]
        ),
        "adapted_loss_quality_percentile": _summary(
            [row["adapted_loss_quality_percentile"] for row in cross_rows]
        ),
        "heldout_G_vs_baseline": _summary(
            [row["G_vs_baseline"] for row in cross_rows]
        ),
        "heldout_A_vs_baseline": _summary(
            [row["A_vs_baseline"] for row in cross_rows]
        ),
        "adapted_loss_vs_baseline": _summary(
            [row["adapted_loss_vs_baseline"] for row in cross_rows]
        ),
        "margin_per_K": _summary([row["margin_per_K"] for row in margin_rows]),
        "margin_relative_Q_std": _summary(
            [row["margin_relative_Q_std"] for row in margin_rows]
        ),
    }
    if candidate == "Full":
        aggregate["component_reliability"] = {
            component: {
                metric: _summary(values)
                for metric, values in correlations.items()
            }
            for component, correlations in component_correlations.items()
        }
    return aggregate, cross_rows, margin_rows


def _average_p0(
    candidate: str,
    rounds: dict[int, dict[int, dict[str, np.ndarray]]],
    baselines: dict[int, tuple[int, ...]],
    *,
    random_samples: int,
    seed: int,
    z_options: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    q_pearson: list[float] = []
    q_spearman: list[float] = []
    overlap: list[float] = []
    exact: list[bool] = []
    margins_k: list[float] = []
    margins_relative: list[float] = []
    cross_rows: list[dict[str, Any]] = []
    for round_number, replicas in sorted(rounds.items()):
        single_assignments = {
            replicate: second_best_assignment(_candidate(data, candidate))[0]
            for replicate, data in replicas.items()
        }
        random_by_heldout = {
            replicate: _random_permutations(
                len(data["task_order"]),
                random_samples,
                seed + round_number * 100 + replicate,
            )
            for replicate, data in replicas.items()
        }
        for heldout in sorted(replicas):
            sources = tuple(value for value in sorted(replicas) if value != heldout)
            averaged_q = averaged_candidate_matrix(
                replicas[sources[0]],
                replicas[sources[1]],
                candidate=candidate,
                **z_options,
            )
            heldout_q = _candidate(replicas[heldout], candidate)
            assignment, best, _, second = second_best_assignment(averaged_q)
            heldout_optimal = single_assignments[heldout]
            q_pearson.append(
                pearson_flat(double_center(averaged_q), double_center(heldout_q))
            )
            q_spearman.append(
                spearman_flat(double_center(averaged_q), double_center(heldout_q))
            )
            overlap.append(assignment_overlap(assignment, heldout_optimal))
            exact.append(assignment == heldout_optimal)
            margin = best - second
            margins_k.append(margin / averaged_q.shape[0])
            margins_relative.append(margin / float(averaged_q.std(ddof=0)))
            functional = _functional_report(
                replicas[heldout],
                assignment,
                baselines[round_number],
                random_by_heldout[heldout],
            )
            cross_rows.append(
                {
                    "protocol": "AVG2_P0_32_32",
                    "candidate": candidate,
                    "round": round_number,
                    "source_replicates": json.dumps(sources),
                    "heldout_replicate": heldout,
                    "functional_percentile": _functional_percentile(functional),
                    **functional,
                }
            )
    return {
        "Q_int_pearson": _summary(q_pearson),
        "Q_int_spearman": _summary(q_spearman),
        "pair_overlap": _summary(overlap),
        "exact_assignment_fraction": float(np.mean(exact)),
        "functional_percentile": _summary(
            [row["functional_percentile"] for row in cross_rows]
        ),
        "heldout_G_quality_percentile": _summary(
            [row["G_quality_percentile"] for row in cross_rows]
        ),
        "heldout_A_quality_percentile": _summary(
            [row["A_quality_percentile"] for row in cross_rows]
        ),
        "adapted_loss_quality_percentile": _summary(
            [row["adapted_loss_quality_percentile"] for row in cross_rows]
        ),
        "heldout_G_vs_baseline": _summary(
            [row["G_vs_baseline"] for row in cross_rows]
        ),
        "heldout_A_vs_baseline": _summary(
            [row["A_vs_baseline"] for row in cross_rows]
        ),
        "adapted_loss_vs_baseline": _summary(
            [row["adapted_loss_vs_baseline"] for row in cross_rows]
        ),
        "margin_per_K": _summary(margins_k),
        "margin_relative_Q_std": _summary(margins_relative),
    }, cross_rows


def analyze(
    source_run: Path,
    diagnostic_dir: Path,
    *,
    random_samples: int,
    seed: int,
) -> dict[str, Any]:
    raw = _load_raw(diagnostic_dir)
    reference = raw["P0_32_32"]
    baselines = _baseline_assignments(source_run, reference)
    source_config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    z_options = {
        "z_eps": float(source_config["z_eps"]),
        "std_atol": float(source_config["std_atol"]),
        "std_rtol": float(source_config["std_rtol"]),
    }
    metadata = _jsonl(diagnostic_dir / "probe_grids.jsonl")
    compute: dict[str, dict[str, float]] = {}
    p0_time = float(
        np.mean([row["probe_seconds"] for row in metadata if row["protocol"] == "P0_32_32"])
    )
    for protocol in PROTOCOLS:
        selected = [row for row in metadata if row["protocol"] == protocol]
        mean_seconds = float(np.mean([row["probe_seconds"] for row in selected]))
        compute[protocol] = {
            "mean_probe_seconds": mean_seconds,
            "std_probe_seconds": float(
                np.std([row["probe_seconds"] for row in selected], ddof=0)
            ),
            "relative_time_vs_P0": mean_seconds / p0_time,
            "peak_gpu_memory_bytes": int(
                max(row["peak_gpu_memory_bytes"] for row in selected)
            ),
        }
    compute["AVG2_P0_32_32"] = {
        "mean_probe_seconds": 2.0 * p0_time,
        "std_probe_seconds": 2.0
        * compute["P0_32_32"]["std_probe_seconds"],
        "relative_time_vs_P0": 2.0,
        "peak_gpu_memory_bytes": compute["P0_32_32"]["peak_gpu_memory_bytes"],
    }

    aggregates: dict[str, dict[str, Any]] = {}
    all_cross: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    for protocol, rounds in raw.items():
        aggregates[protocol] = {}
        for candidate in CANDIDATES:
            aggregate, cross_rows, margin_rows = _protocol_candidate(
                protocol,
                candidate,
                rounds,
                baselines,
                random_samples=random_samples,
                seed=seed,
            )
            aggregates[protocol][candidate] = aggregate
            all_cross.extend(cross_rows)
            all_margins.extend(margin_rows)
    aggregates["AVG2_P0_32_32"] = {}
    for candidate in CANDIDATES:
        aggregate, cross_rows = _average_p0(
            candidate,
            reference,
            baselines,
            random_samples=random_samples,
            seed=seed,
            z_options=z_options,
        )
        aggregates["AVG2_P0_32_32"][candidate] = aggregate
        all_cross.extend(cross_rows)

    comparison_rows: list[dict[str, Any]] = []
    for protocol in (*PROTOCOLS, "AVG2_P0_32_32"):
        for candidate in CANDIDATES:
            item = aggregates[protocol][candidate]
            comparison_rows.append(
                {
                    "protocol": protocol,
                    "candidate": candidate,
                    "functional_percentile": item["functional_percentile"]["mean"],
                    "G_quality_percentile": item[
                        "heldout_G_quality_percentile"
                    ]["mean"],
                    "A_quality_percentile": item[
                        "heldout_A_quality_percentile"
                    ]["mean"],
                    "adapted_loss_quality_percentile": item[
                        "adapted_loss_quality_percentile"
                    ]["mean"],
                    "Q_int_pearson": item["Q_int_pearson"]["mean"],
                    "Q_int_spearman": item["Q_int_spearman"]["mean"],
                    "pair_overlap": item["pair_overlap"]["mean"],
                    "exact_assignment_fraction": item[
                        "exact_assignment_fraction"
                    ],
                    "margin_per_K": item["margin_per_K"]["mean"],
                    "margin_relative_Q_std": item[
                        "margin_relative_Q_std"
                    ]["mean"],
                    **compute[protocol],
                }
            )

    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for candidate in CANDIDATES:
        base = aggregates["P0_32_32"][candidate]
        deltas[candidate] = {}
        for protocol in ("P1_32_64", "P2_64_32", "P3_64_64", "AVG2_P0_32_32"):
            item = aggregates[protocol][candidate]
            deltas[candidate][protocol] = {
                "functional_percentile": (
                    item["functional_percentile"]["mean"]
                    - base["functional_percentile"]["mean"]
                ),
                "Q_int_pearson": (
                    item["Q_int_pearson"]["mean"]
                    - base["Q_int_pearson"]["mean"]
                ),
                "Q_int_spearman": (
                    item["Q_int_spearman"]["mean"]
                    - base["Q_int_spearman"]["mean"]
                ),
                "pair_overlap": (
                    item["pair_overlap"]["mean"]
                    - base["pair_overlap"]["mean"]
                ),
                "margin_per_K": (
                    item["margin_per_K"]["mean"]
                    - base["margin_per_K"]["mean"]
                ),
                "relative_time_vs_P0": compute[protocol]["relative_time_vs_P0"],
            }
    return {
        "protocol": {
            "rounds": sorted(reference),
            "replicates": 3,
            "random_assignments_per_heldout": random_samples,
            "nested_samples_verified": True,
            "new_training_trajectories": 0,
            "uses_test_accuracy": False,
            "changes_formal_score_or_gate": False,
            "timing_scope": (
                "model probe computation; shared one-time CPU tensor materialization excluded"
            ),
        },
        "aggregates": aggregates,
        "comparison_rows": comparison_rows,
        "deltas_vs_P0": deltas,
        "compute": compute,
        "cross_probe_detail": all_cross,
        "margin_detail": all_margins,
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 3.6 Probe-size comparison",
        "",
        "| Protocol | Score | Functional pct | Q-int P/S | Pair overlap | Exact | Margin/K | Seconds | Time/P0 | Peak MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f'| {row["protocol"]} | {row["candidate"]} | '
            f'{row["functional_percentile"]:.2f} | '
            f'{row["Q_int_pearson"]:.3f}/{row["Q_int_spearman"]:.3f} | '
            f'{row["pair_overlap"]:.3f} | {row["exact_assignment_fraction"]:.3f} | '
            f'{row["margin_per_K"]:.4f} | {row["mean_probe_seconds"]:.2f} | '
            f'{row["relative_time_vs_P0"]:.2f} | '
            f'{row["peak_gpu_memory_bytes"] / 1048576:.1f} |'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    protocols = (*PROTOCOLS, "AVG2_P0_32_32")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    metrics = (
        ("functional_percentile", "Functional percentile"),
        ("Q_int_pearson", "Q-int Pearson"),
        ("pair_overlap", "Pair overlap"),
    )
    x = np.arange(len(protocols))
    width = 0.36
    for axis, (metric, title) in zip(axes, metrics):
        for index, candidate in enumerate(CANDIDATES):
            values = [
                next(
                    row[metric]
                    for row in rows
                    if row["protocol"] == protocol
                    and row["candidate"] == candidate
                )
                for protocol in protocols
            ]
            axis.bar(x + (index - 0.5) * width, values, width, label=candidate)
        axis.set_xticks(x, protocols, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=936_101)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(
        args.source_run.resolve(),
        args.diagnostic_dir.resolve(),
        random_samples=args.random_samples,
        seed=args.seed,
    )
    (output_dir / "probe_size_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "probe_size_comparison.csv", report["comparison_rows"])
    _write_csv(output_dir / "probe_size_cross_probe.csv", report["cross_probe_detail"])
    _write_csv(output_dir / "probe_size_margins.csv", report["margin_detail"])
    _write_markdown(output_dir / "probe_size_comparison.md", report["comparison_rows"])
    _plot(output_dir / "probe_size_comparison.png", report["comparison_rows"])
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "comparison_rows": report["comparison_rows"],
                "deltas_vs_P0": report["deltas_vs_P0"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
