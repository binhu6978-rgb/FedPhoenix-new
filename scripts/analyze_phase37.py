from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CROSS_SPEC = importlib.util.spec_from_file_location(
    "phase35_offline", PROJECT_ROOT / "scripts" / "analyze_phase35_cross_probe.py"
)
if _CROSS_SPEC is None or _CROSS_SPEC.loader is None:
    raise RuntimeError("could not load cross-probe analysis helpers")
_CROSS = importlib.util.module_from_spec(_CROSS_SPEC)
_CROSS_SPEC.loader.exec_module(_CROSS)
_DIAG = _CROSS._DIAGNOSTICS

COMPONENTS = ("G", "A", "D", "C")
CORE_CONFIG_KEYS = (
    "dataset",
    "model",
    "num_classes",
    "num_users",
    "clients_per_round",
    "rounds",
    "dirichlet_beta",
    "min_client_samples",
    "seed",
    "local_epochs",
    "local_batch_size",
    "learning_rate",
    "momentum",
    "weight_decay",
    "num_workers",
    "reset_ratio",
    "fp_conv_rounds",
    "reset_method",
    "eval_batch_size",
    "eval_every",
    "deterministic",
    "data_root",
    "partition_path",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assignment_from_pairs(
    row: dict[str, Any], key: str, task_order: np.ndarray | None = None
) -> tuple[int, ...]:
    ordered = sorted(row[key], key=lambda pair: int(pair["client_row"]))
    task_ids = tuple(int(pair["task_id"]) for pair in ordered)
    if task_order is None:
        return task_ids
    columns = {int(task): index for index, task in enumerate(task_order)}
    return tuple(columns[task] for task in task_ids)


def _accuracy(clean_dir: Path, fedrad_dir: Path) -> dict[str, Any]:
    clean_rows = _jsonl(clean_dir / "rounds.jsonl")
    fedrad_rows = _jsonl(fedrad_dir / "rounds.jsonl")
    if len(clean_rows) != 40 or len(fedrad_rows) != 40:
        raise ValueError("Phase 3.7 accuracy comparison requires two complete 40-round runs")
    clean = np.asarray([row["diagnostic_accuracy"] for row in clean_rows])
    fedrad = np.asarray([row["diagnostic_accuracy"] for row in fedrad_rows])

    def metrics(values: np.ndarray) -> dict[str, Any]:
        peak_index = int(np.argmax(values))
        return {
            "peak": float(values[peak_index]),
            "peak_round": peak_index + 1,
            "top5_rounds_mean": float(np.sort(values)[-5:].mean()),
            "final": float(values[-1]),
            "last5_mean": float(values[-5:].mean()),
            "last10_mean": float(values[-10:].mean()),
        }

    clean_metrics = metrics(clean)
    fedrad_metrics = metrics(fedrad)
    return {
        "seed": 1,
        "clean": clean_metrics,
        "fedrad": fedrad_metrics,
        "delta": {
            key: fedrad_metrics[key] - clean_metrics[key]
            for key in ("peak", "top5_rounds_mean", "final", "last5_mean", "last10_mean")
        },
        "single_seed_only": True,
        "aggregate_mean_std": "not applicable after user limited validation to seed 1",
        "per_round": [
            {
                "round": index + 1,
                "clean_accuracy": float(clean[index]),
                "fedrad_accuracy": float(fedrad[index]),
                "delta": float(fedrad[index] - clean[index]),
            }
            for index in range(40)
        ],
    }


def _dynamics(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rounds = _jsonl(run_dir / "rounds.jsonl")
    assignments = _jsonl(run_dir / "assignments.jsonl")
    output: list[dict[str, Any]] = []
    for round_row, assignment_row in zip(rounds, assignments):
        round_number = int(round_row["round_number"])
        with np.load(
            run_dir / "scores" / f"round_{round_number:04d}.npz",
            allow_pickle=False,
        ) as saved:
            matrices = {
                name: np.asarray(saved[name], dtype=np.float64)
                for name in (*COMPONENTS, "Q")
            }
        best, best_score, _, second_score = _DIAG.second_best_assignment(matrices["Q"])
        baseline = _assignment_from_pairs(assignment_row, "baseline_pairs")
        hungarian = _assignment_from_pairs(assignment_row, "hungarian_pairs")
        if best != hungarian or not np.isclose(best_score, assignment_row["hungarian_score"]):
            raise ValueError(f"saved assignment mismatch in round {round_number}")
        q_std = float(matrices["Q"].std(ddof=0))
        row: dict[str, Any] = {
            "round": round_number,
            "accuracy": float(round_row["diagnostic_accuracy"]),
            "Gamma": float(assignment_row["Gamma"]),
            "hungarian_score": float(assignment_row["hungarian_score"]),
            "baseline_score": float(assignment_row["baseline_score"]),
            "changed_pairs": sum(a != b for a, b in zip(baseline, hungarian)),
            "changed_fraction": sum(a != b for a, b in zip(baseline, hungarian)) / len(best),
            "assignment_margin": best_score - second_score,
            "margin_per_K": (best_score - second_score) / len(best),
            "margin_per_Q_std": (best_score - second_score) / q_std if q_std else 0.0,
            "Q_std": q_std,
            "Q_interaction_std": float(_DIAG.double_center(matrices["Q"]).std(ddof=0)),
            "probe_seconds": float(round_row["probe_seconds"]),
            "reliability_probe_seconds": float(round_row["reliability_probe_seconds"]),
            "formal_training_seconds": float(round_row["formal_training_seconds"]),
            "matching_seconds": float(round_row["matching_seconds"]),
            "peak_gpu_memory_bytes": int(round_row["peak_gpu_memory_bytes"]),
        }
        for name in COMPONENTS:
            matrix = matrices[name]
            row[f"{name}_mean"] = float(matrix.mean())
            row[f"{name}_std"] = float(matrix.std(ddof=0))
            row[f"{name}_interaction_std"] = float(
                _DIAG.double_center(matrix).std(ddof=0)
            )
        row["D_positive_ratio"] = float(np.mean(matrices["D"] > 0.0))
        output.append(row)

    aggregate_keys = (
        "Gamma",
        "hungarian_score",
        "baseline_score",
        "changed_pairs",
        "changed_fraction",
        "assignment_margin",
        "margin_per_K",
        "margin_per_Q_std",
        "Q_std",
        "Q_interaction_std",
        "D_positive_ratio",
        *(f"{name}_mean" for name in COMPONENTS),
        *(f"{name}_std" for name in COMPONENTS),
        *(f"{name}_interaction_std" for name in COMPONENTS),
    )
    aggregate = {key: _summary([row[key] for row in output]) for key in aggregate_keys}
    return output, aggregate


def _correlations(rows: list[dict[str, Any]], predictor: str) -> dict[str, Any]:
    outcomes = {
        "delta_G": "G_vs_baseline",
        "delta_A": "A_vs_baseline",
        "adapted_loss_improvement": "adapted_loss_improvement",
        "functional_percentile": "functional_percentile",
    }
    output: dict[str, Any] = {}
    x = np.asarray([row[predictor] for row in rows], dtype=np.float64)
    for label, key in outcomes.items():
        y = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[label] = {
            "pearson": _DIAG.pearson_flat(x, y),
            "spearman": _DIAG.spearman_flat(x, y),
        }
    return output


def _round_mean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for round_number in sorted({int(row["round"]) for row in rows}):
        selected = [row for row in rows if int(row["round"]) == round_number]
        output.append(
            {
                "round": round_number,
                "Gamma": selected[0]["Gamma"],
                "margin_per_K": selected[0]["margin_per_K"],
                "margin_per_Q_std": selected[0]["margin_per_Q_std"],
                "G_vs_baseline": _mean(selected, "G_vs_baseline"),
                "A_vs_baseline": _mean(selected, "A_vs_baseline"),
                "adapted_loss_improvement": _mean(selected, "adapted_loss_improvement"),
                "functional_percentile": _mean(selected, "functional_percentile"),
            }
        )
    return output


def _terciles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    round_rows = _round_mean_rows(rows)
    ordered = sorted(round_rows, key=lambda row: row["Gamma"])
    # Five states cannot form equal thirds. Keep both tails symmetric: 2/1/2 states.
    groups = (ordered[:2], ordered[2:3], ordered[3:])
    labels = ("lower", "middle", "upper")
    output: list[dict[str, Any]] = []
    for label, states in zip(labels, groups):
        round_ids = {int(row["round"]) for row in states}
        observations = [row for row in rows if int(row["round"]) in round_ids]
        output.append(
            {
                "tercile": label,
                "rounds": sorted(round_ids),
                "round_states": len(states),
                "heldout_observations": len(observations),
                "Gamma_min": min(row["Gamma"] for row in states),
                "Gamma_max": max(row["Gamma"] for row in states),
                "functional_percentile_mean": _mean(observations, "functional_percentile"),
                "P_delta_G_positive": float(np.mean([row["G_vs_baseline"] > 0 for row in observations])),
                "P_delta_A_positive": float(np.mean([row["A_vs_baseline"] > 0 for row in observations])),
                "P_adapted_loss_improves": float(np.mean([row["adapted_loss_improvement"] > 0 for row in observations])),
            }
        )
    return output


def _candidate_report(
    replicas: dict[int, dict[int, dict[str, np.ndarray]]],
    baselines: dict[int, tuple[int, ...]],
    *,
    name: str,
    random_samples: int,
    seed: int,
) -> dict[str, Any]:
    cross_rows: list[dict[str, Any]] = []
    reliabilities: list[float] = []
    overlaps: list[float] = []
    margins: list[float] = []
    for round_number in sorted(replicas):
        reps = replicas[round_number]
        matrices = {
            replicate: _CROSS._candidate_matrices(data)[name]
            for replicate, data in reps.items()
        }
        primary_assignment, primary_score, _, second_score = _DIAG.second_best_assignment(
            matrices[0]
        )
        margins.append(primary_score - second_score)
        for heldout_replicate in (1, 2):
            heldout_matrix = matrices[heldout_replicate]
            heldout_assignment, _ = _DIAG.maximum_assignment(heldout_matrix)
            reliabilities.append(
                _DIAG.pearson_flat(
                    _DIAG.double_center(matrices[0]),
                    _DIAG.double_center(heldout_matrix),
                )
            )
            overlaps.append(_DIAG.assignment_overlap(primary_assignment, heldout_assignment))
            random_assignments = _CROSS._random_permutations(
                len(primary_assignment),
                random_samples,
                seed + round_number * 100 + heldout_replicate,
            )
            functional = _CROSS._functional_report(
                reps[heldout_replicate],
                primary_assignment,
                baselines[round_number],
                random_assignments,
            )
            cross_rows.append(functional)
    functional_percentiles = [
        np.mean(
            [
                row["G_quality_percentile"],
                row["A_quality_percentile"],
                row["adapted_loss_quality_percentile"],
            ]
        )
        for row in cross_rows
    ]
    return {
        "candidate": name,
        "primary_to_heldout_interaction_pearson": _summary(reliabilities),
        "primary_to_heldout_pair_overlap": _summary(overlaps),
        "primary_margin": _summary(margins),
        "primary_margin_per_K": _summary(np.asarray(margins) / 10.0),
        "functional_percentile": _summary(functional_percentiles),
        "delta_G": _summary([row["G_vs_baseline"] for row in cross_rows]),
        "delta_A": _summary([row["A_vs_baseline"] for row in cross_rows]),
        "adapted_loss_improvement": _summary(
            [-row["adapted_loss_vs_baseline"] for row in cross_rows]
        ),
    }


def _heldout(
    run_dir: Path, *, random_samples: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    replicas = _CROSS._load_replicates(run_dir)
    baselines = _CROSS._baseline_assignments(run_dir, replicas)
    rows: list[dict[str, Any]] = []
    for round_number in sorted(replicas):
        reps = replicas[round_number]
        primary = reps[0]
        primary_q = primary["Q"]
        assignment, score, _, second_score = _DIAG.second_best_assignment(primary_q)
        q_std = float(primary_q.std(ddof=0))
        for heldout_replicate in (1, 2):
            random_assignments = _CROSS._random_permutations(
                len(assignment),
                random_samples,
                seed + round_number * 100 + heldout_replicate,
            )
            functional = _CROSS._functional_report(
                reps[heldout_replicate],
                assignment,
                baselines[round_number],
                random_assignments,
            )
            row: dict[str, Any] = {
                "round": round_number,
                "heldout_replicate": heldout_replicate,
                "Gamma": score / len(assignment),
                "primary_score": score,
                "primary_margin": score - second_score,
                "margin_per_K": (score - second_score) / len(assignment),
                "margin_per_Q_std": (score - second_score) / q_std if q_std else 0.0,
                **functional,
            }
            row["adapted_loss_improvement"] = -row["adapted_loss_vs_baseline"]
            row["functional_percentile"] = float(
                np.mean(
                    [
                        row["G_quality_percentile"],
                        row["A_quality_percentile"],
                        row["adapted_loss_quality_percentile"],
                    ]
                )
            )
            rows.append(row)

    round_means = _round_mean_rows(rows)
    correlations = {
        predictor: {
            "observation_level_n10": _correlations(rows, predictor),
            "round_mean_n5": _correlations(round_means, predictor),
        }
        for predictor in ("Gamma", "margin_per_K", "margin_per_Q_std")
    }
    aggregate = {
        "round_states": 5,
        "heldout_observations": len(rows),
        "delta_G": _summary([row["G_vs_baseline"] for row in rows]),
        "delta_A": _summary([row["A_vs_baseline"] for row in rows]),
        "delta_adapted_loss": _summary(
            [row["adapted_loss_vs_baseline"] for row in rows]
        ),
        "adapted_loss_improvement": _summary(
            [row["adapted_loss_improvement"] for row in rows]
        ),
        "functional_percentile": _summary(
            [row["functional_percentile"] for row in rows]
        ),
        "P_delta_G_positive": float(np.mean([row["G_vs_baseline"] > 0 for row in rows])),
        "P_delta_A_positive": float(np.mean([row["A_vs_baseline"] > 0 for row in rows])),
        "P_adapted_loss_improves": float(
            np.mean([row["adapted_loss_improvement"] > 0 for row in rows])
        ),
        "Gamma_terciles": _terciles(rows),
        "correlations": correlations,
        "cluster_caution": (
            "two held-out observations share each primary predictor; round-mean n=5 "
            "is the non-duplicated sensitivity analysis"
        ),
    }
    candidates = {
        name: _candidate_report(
            replicas,
            baselines,
            name=name,
            random_samples=random_samples,
            seed=seed + (0 if name == "Full" else 50_000),
        )
        for name in ("Full", "G")
    }
    return rows, aggregate, candidates


def _gate_decision(heldout: dict[str, Any]) -> dict[str, Any]:
    gamma = heldout["correlations"]["Gamma"]
    obs = gamma["observation_level_n10"]["functional_percentile"]
    rounds = gamma["round_mean_n5"]["functional_percentile"]
    terciles = heldout["Gamma_terciles"]
    tail_gain = (
        terciles[-1]["functional_percentile_mean"]
        - terciles[0]["functional_percentile_mean"]
    )
    positive_tail_votes = sum(
        terciles[-1][key] > terciles[0][key]
        for key in (
            "P_delta_G_positive",
            "P_delta_A_positive",
            "P_adapted_loss_improves",
        )
    )
    clear = (
        obs["pearson"] >= 0.5
        and obs["spearman"] >= 0.5
        and rounds["pearson"] >= 0.5
        and rounds["spearman"] >= 0.5
        and tail_gain >= 10.0
        and positive_tail_votes >= 2
    )
    weak = (
        obs["pearson"] >= 0.3
        and obs["spearman"] >= 0.3
        and tail_gain > 0.0
    )
    verdict = "Gamma predictive" if clear else ("Gamma weakly predictive" if weak else "Gamma not predictive")
    return {
        "verdict": verdict,
        "predeclared_diagnostic_rule": {
            "predictive": (
                "observation and round-mean Pearson/Spearman >= 0.5, upper-minus-lower "
                "functional percentile >= 10, and at least two positive-rate improvements"
            ),
            "weakly_predictive": (
                "observation Pearson/Spearman >= 0.3 and positive upper-minus-lower percentile"
            ),
        },
        "upper_minus_lower_functional_percentile": tail_gain,
        "positive_rate_tail_votes": positive_tail_votes,
        "candidate_tau": (
            "not estimated: current gate not calibratable from available evidence"
            if not clear
            else "eligible for offline quantile analysis; not hard-coded"
        ),
    }


def _cross_probe_comparison(
    fedrad_dir: Path,
    historical_dir: Path,
    *,
    random_samples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, run_dir, analysis_seed in (
        ("development_64_32", fedrad_dir, seed),
        ("historical_32_32", historical_dir, seed + 10_000),
    ):
        analysis = _CROSS.analyze(
            run_dir, random_samples=random_samples, seed=analysis_seed
        )
        full = analysis["candidate_aggregates"]["Full"]
        output[label] = {
            "representative_rounds": analysis["protocol"]["representative_rounds"],
            "ordered_cross_probe_observations": len(
                [
                    row
                    for row in analysis["ordered_cross_probe"]
                    if row["candidate"] == "Full"
                ]
            ),
            "Full_Q_interaction_pearson": full["Q_int_pearson"],
            "Full_Q_interaction_spearman": full["Q_int_spearman"],
            "Full_pair_overlap": full["pair_overlap"],
            "Full_exact_assignment_fraction": full["exact_assignment_fraction"],
            "Full_functional_percentile": full["functional_quality_percentile"],
            "Full_margin_per_K": full["margin_per_K"],
        }
    return output


def _cost(clean_dir: Path, fedrad_dir: Path, dynamics: list[dict[str, Any]]) -> dict[str, Any]:
    clean_rounds = _jsonl(clean_dir / "rounds.jsonl")
    clean_total = float(sum(row["round_seconds"] for row in clean_rounds))
    primary_probe = float(sum(row["probe_seconds"] for row in dynamics))
    formal = float(sum(row["formal_training_seconds"] for row in dynamics))
    matching = float(sum(row["matching_seconds"] for row in dynamics))
    reliability = float(sum(row["reliability_probe_seconds"] for row in dynamics))
    fed_accounted_core = primary_probe + formal + matching
    fed_accounted_diagnostic = fed_accounted_core + reliability

    def wall_seconds(run_dir: Path) -> float:
        return float(
            (run_dir / "summary.json").stat().st_mtime
            - (run_dir / "config.json").stat().st_mtime
        )

    clean_wall = wall_seconds(clean_dir)
    fedrad_wall = wall_seconds(fedrad_dir)
    return {
        "primary_probe_seconds": primary_probe,
        "formal_training_seconds": formal,
        "matching_seconds": matching,
        "diagnostic_heldout_probe_seconds": reliability,
        "probe_over_formal_training": primary_probe / formal,
        "probe_including_heldout_over_formal_training": (primary_probe + reliability) / formal,
        "clean_logged_round_seconds": clean_total,
        "fedrad_accounted_core_seconds": fed_accounted_core,
        "fedrad_accounted_with_diagnostics_seconds": fed_accounted_diagnostic,
        "clean_wall_seconds_config_to_summary": clean_wall,
        "fedrad_wall_seconds_config_to_summary": fedrad_wall,
        "wall_overhead_vs_clean_seconds": fedrad_wall - clean_wall,
        "wall_overhead_vs_clean_ratio": fedrad_wall / clean_wall,
        "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in dynamics),
        "caution": (
            "FedRAD logs component timings but not whole-round seconds; filesystem wall time "
            "is reported for the direct end-to-end comparison"
        ),
    }


def _plot_accuracy(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot([row["round"] for row in rows], [row["clean_accuracy"] for row in rows], label="Clean")
    axis.plot([row["round"] for row in rows], [row["fedrad_accuracy"] for row in rows], label="FedRAD Full 64/32")
    axis.set(xlabel="Round", ylabel="Full accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    accuracy = report["accuracy"]
    heldout = report["heldout"]
    gate = report["gate_identifiability"]
    margin_corr = heldout["correlations"]["margin_per_K"]["observation_level_n10"]
    gamma_corr = heldout["correlations"]["Gamma"]["observation_level_n10"]
    full = report["full_vs_g_only"]["Full"]
    g_only = report["full_vs_g_only"]["G"]
    cost = report["cost"]
    lines = [
        "# Phase 3.7 — Measurement Fix Validation & Gate Identifiability",
        "",
        "> Scope override: per the latest user instruction, this report uses **seed 1 only**. "
        "The originally requested 3-seed mean/std and 30 held-out observations are therefore not applicable.",
        "",
        "## A. Seed-1 40-round training",
        "",
        "| Metric | Clean | FedRAD Full 64/32 | Delta |",
        "|---|---:|---:|---:|",
        f"| **Peak (primary)** | **{_fmt(accuracy['clean']['peak'])} (r{accuracy['clean']['peak_round']})** | **{_fmt(accuracy['fedrad']['peak'])} (r{accuracy['fedrad']['peak_round']})** | **{_fmt(accuracy['delta']['peak'])}** |",
        f"| Top-5 rounds mean | {_fmt(accuracy['clean']['top5_rounds_mean'])} | {_fmt(accuracy['fedrad']['top5_rounds_mean'])} | {_fmt(accuracy['delta']['top5_rounds_mean'])} |",
        f"| Final | {_fmt(accuracy['clean']['final'])} | {_fmt(accuracy['fedrad']['final'])} | {_fmt(accuracy['delta']['final'])} |",
        f"| Last-5 mean | {_fmt(accuracy['clean']['last5_mean'])} | {_fmt(accuracy['fedrad']['last5_mean'])} | {_fmt(accuracy['delta']['last5_mean'])} |",
        f"| Last-10 mean | {_fmt(accuracy['clean']['last10_mean'])} | {_fmt(accuracy['fedrad']['last10_mean'])} | {_fmt(accuracy['delta']['last10_mean'])} |",
        "",
        "This is a diagnostic single-seed comparison, not a paper-level result.",
        "",
        "## B. Recovery dynamics (40 rounds)",
        "",
        "| Metric | 64/32 mean | 32/32 historical mean |",
        "|---|---:|---:|",
    ]
    for key in ("Gamma", "changed_pairs", "margin_per_K", "margin_per_Q_std", "Q_interaction_std"):
        lines.append(
            f"| {key} | {_fmt(report['dynamics_64_32'][key]['mean'])} | "
            f"{_fmt(report['historical_32_32'][key]['mean'])} |"
        )
    lines.extend(
        [
            "",
            "Cross-probe Full reliability (ordered replicate pairs):",
            "",
            "| Protocol | Q-int Pearson | Q-int Spearman | Pair overlap | Functional pct |",
            "|---|---:|---:|---:|---:|",
            f"| 64/32 | {report['cross_probe_comparison']['development_64_32']['Full_Q_interaction_pearson']:.3f} | "
            f"{report['cross_probe_comparison']['development_64_32']['Full_Q_interaction_spearman']:.3f} | "
            f"{report['cross_probe_comparison']['development_64_32']['Full_pair_overlap']:.3f} | "
            f"{report['cross_probe_comparison']['development_64_32']['Full_functional_percentile']:.2f} |",
            f"| historical 32/32 | {report['cross_probe_comparison']['historical_32_32']['Full_Q_interaction_pearson']:.3f} | "
            f"{report['cross_probe_comparison']['historical_32_32']['Full_Q_interaction_spearman']:.3f} | "
            f"{report['cross_probe_comparison']['historical_32_32']['Full_pair_overlap']:.3f} | "
            f"{report['cross_probe_comparison']['historical_32_32']['Full_functional_percentile']:.2f} |",
            "",
            "The representative round sets differ (64/32 uses round 30; historical 32/32 uses round 5), so this is diagnostic rather than paired inference. Detailed per-round G/A/D/C/Q statistics are in `dynamics_64_32.csv`.",
            "",
            "## C. Primary → held-out validity",
            "",
            f"Five primary round states × two independent held-out probes = {heldout['heldout_observations']} observations.",
            "",
            "| Metric | Mean | Positive/improvement rate |",
            "|---|---:|---:|",
            f"| Delta G | {_fmt(heldout['delta_G']['mean'])} | {heldout['P_delta_G_positive']:.1%} |",
            f"| Delta A | {_fmt(heldout['delta_A']['mean'])} | {heldout['P_delta_A_positive']:.1%} |",
            f"| Delta adapted loss | {_fmt(heldout['delta_adapted_loss']['mean'])} | {heldout['P_adapted_loss_improves']:.1%} |",
            f"| Functional percentile | {heldout['functional_percentile']['mean']:.2f} | — |",
            "",
            "## D. Gamma validity",
            "",
            f"Verdict: **{gate['verdict']}**.",
            "",
            "| Outcome | Pearson | Spearman |",
            "|---|---:|---:|",
        ]
    )
    for label, values in gamma_corr.items():
        lines.append(f"| {label} | {_fmt(values['pearson'])} | {_fmt(values['spearman'])} |")
    lines.extend(["", "Gamma tail analysis (2/1/2 primary states because n=5):", ""])
    lines.extend(
        [
            "| Group | Rounds | Functional pct | P(ΔG>0) | P(ΔA>0) | P(loss improves) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in heldout["Gamma_terciles"]:
        lines.append(
            f"| {row['tercile']} | {row['rounds']} | {row['functional_percentile_mean']:.2f} | "
            f"{row['P_delta_G_positive']:.1%} | {row['P_delta_A_positive']:.1%} | "
            f"{row['P_adapted_loss_improves']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## E. Margin validity",
            "",
            "| Outcome | Pearson | Spearman |",
            "|---|---:|---:|",
        ]
    )
    for label, values in margin_corr.items():
        lines.append(f"| {label} | {_fmt(values['pearson'])} | {_fmt(values['spearman'])} |")
    lines.extend(
        [
            "",
            "Both observation-level n=10 and non-duplicated round-mean n=5 correlations are preserved in JSON.",
            "",
            "## F. Candidate tau",
            "",
            str(gate["candidate_tau"]),
            "",
            "## G. Full vs G-only (primary → held-out)",
            "",
            "| Candidate | Interaction r | Pair overlap | Functional pct | Margin/K |",
            "|---|---:|---:|---:|---:|",
            f"| Full | {full['primary_to_heldout_interaction_pearson']['mean']:.3f} | "
            f"{full['primary_to_heldout_pair_overlap']['mean']:.3f} | "
            f"{full['functional_percentile']['mean']:.2f} | {full['primary_margin_per_K']['mean']:.5f} |",
            f"| G-only | {g_only['primary_to_heldout_interaction_pearson']['mean']:.3f} | "
            f"{g_only['primary_to_heldout_pair_overlap']['mean']:.3f} | "
            f"{g_only['functional_percentile']['mean']:.2f} | {g_only['primary_margin_per_K']['mean']:.5f} |",
            "",
            "This remains an offline diagnostic; the formal Full score was not changed.",
            "",
            "## H. Cost",
            "",
            f"- Primary probe: {cost['primary_probe_seconds']:.1f}s; formal training: {cost['formal_training_seconds']:.1f}s; "
            f"probe/formal = {cost['probe_over_formal_training']:.3f}×.",
            f"- Two held-out diagnostics at five rounds added {cost['diagnostic_heldout_probe_seconds']:.1f}s.",
            f"- End-to-end wall ratio vs Clean: {cost['wall_overhead_vs_clean_ratio']:.3f}×; "
            f"peak GPU memory: {cost['peak_gpu_memory_bytes'] / 2**20:.1f} MiB.",
            "",
            "## I. Final conclusion",
            "",
            f"**{report['final_conclusion']}**",
            "",
            report["conclusion_reason"],
            "",
            "Phase 4 was not started.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--fedrad-run", type=Path, required=True)
    parser.add_argument("--historical-fedrad-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=937_001)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = args.clean_run.resolve()
    fedrad_dir = args.fedrad_run.resolve()
    historical_dir = args.historical_fedrad_run.resolve()
    clean_config = _json(clean_dir / "config.json")
    fedrad_config = _json(fedrad_dir / "config.json")
    clean_core = {key: clean_config[key] for key in CORE_CONFIG_KEYS}
    fedrad_core = {key: fedrad_config[key] for key in CORE_CONFIG_KEYS}
    clean_summary = _json(clean_dir / "summary.json")
    fedrad_summary = _json(fedrad_dir / "summary.json")
    provenance = {
        "clean_run": str(clean_dir),
        "fedrad_run": str(fedrad_dir),
        "historical_32_32_run": str(historical_dir),
        "core_training_config_equal": clean_core == fedrad_core,
        "core_training_config_fingerprint_clean": _fingerprint(clean_core),
        "core_training_config_fingerprint_fedrad": _fingerprint(fedrad_core),
        "partition_fingerprint_equal": clean_summary["partition_fingerprint"] == fedrad_summary["partition_fingerprint"],
        "partition_fingerprint": fedrad_summary["partition_fingerprint"],
        "initial_state_hash_equal": clean_summary["initial_state_hash"] == fedrad_summary["initial_state_hash"],
        "initial_state_hash": fedrad_summary["initial_state_hash"],
        "clean_reuse_valid": (
            clean_core == fedrad_core
            and clean_summary["partition_fingerprint"] == fedrad_summary["partition_fingerprint"]
            and clean_summary["initial_state_hash"] == fedrad_summary["initial_state_hash"]
            and bool(clean_summary.get("no_probe_score_matching_or_gate"))
        ),
        "probe_protocol": fedrad_config["probe_protocol"],
        "historical_replay_retained": True,
        "scope_override": "seed 1 only per latest user instruction; seeds 2/3 excluded",
        "excluded_runs": [
            "clean seed 2 complete but excluded",
            "clean seed 3 complete but excluded",
            "FedRAD seed 2 interrupted and incomplete",
        ],
    }
    if not provenance["clean_reuse_valid"]:
        raise ValueError("seed-1 Clean trajectory is not valid for reuse")
    if fedrad_config["probe_support_size"] != 64 or fedrad_config["probe_query_size"] != 32:
        raise ValueError("FedRAD run is not the Phase 3.7 64/32 protocol")

    accuracy = _accuracy(clean_dir, fedrad_dir)
    dynamics, dynamics_aggregate = _dynamics(fedrad_dir)
    historical_dynamics, historical_aggregate = _dynamics(historical_dir)
    heldout_rows, heldout, candidates = _heldout(
        fedrad_dir, random_samples=args.random_samples, seed=args.seed
    )
    cross_probe_comparison = _cross_probe_comparison(
        fedrad_dir,
        historical_dir,
        random_samples=args.random_samples,
        seed=args.seed + 100_000,
    )
    gate = _gate_decision(heldout)
    cost = _cost(clean_dir, fedrad_dir, dynamics)

    if gate["verdict"] == "Gamma not predictive" and heldout["functional_percentile"]["mean"] >= 75.0:
        conclusion = "matching useful but gate invalid"
        reason = (
            "64/32 primary matching is strongly above random on held-out functional metrics, "
            "but Gamma does not identify which rounds are more reliable. Peak Full accuracy "
            "improves only slightly, and one seed cannot establish robustness."
        )
    elif heldout["functional_percentile"]["mean"] >= 75.0:
        conclusion = "measurement fixed but training benefit unclear"
        reason = (
            "Held-out recovery is credible, while a single 40-round seed is insufficient to "
            "establish a stable optimization benefit."
        )
    else:
        conclusion = "core mechanism still insufficient"
        reason = "Held-out functional recovery is not strong enough to validate the matching mechanism."

    report = {
        "protocol": {
            "probe_protocol": "support64_query32_steps1",
            "formal_score": "Z(G) + Z(A) - Z(D) + 0.25 Z(C)",
            "development_force_hungarian": True,
            "representative_rounds": [1, 10, 20, 30, 40],
            "heldout_replicates_per_round": 2,
            "random_assignments_per_heldout": args.random_samples,
            "uses_test_accuracy_for_tuning": False,
            "changes_score_gate_tau_or_training": False,
        },
        "provenance": provenance,
        "accuracy": {key: value for key, value in accuracy.items() if key != "per_round"},
        "dynamics_64_32": dynamics_aggregate,
        "historical_32_32": historical_aggregate,
        "heldout": heldout,
        "cross_probe_comparison": cross_probe_comparison,
        "gate_identifiability": gate,
        "full_vs_g_only": candidates,
        "cost": cost,
        "final_conclusion": conclusion,
        "conclusion_reason": reason,
        "phase4_started": False,
    }
    _write_csv(output_dir / "accuracy_curves.csv", accuracy["per_round"])
    _write_csv(output_dir / "dynamics_64_32.csv", dynamics)
    _write_csv(output_dir / "dynamics_historical_32_32.csv", historical_dynamics)
    _write_csv(output_dir / "primary_to_heldout.csv", heldout_rows)
    (output_dir / "phase37_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_accuracy(output_dir / "accuracy_curves.png", accuracy["per_round"])
    _write_markdown(output_dir / "phase37_report.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
