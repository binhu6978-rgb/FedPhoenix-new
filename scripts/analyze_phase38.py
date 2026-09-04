from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE37_PATH = PROJECT_ROOT / "scripts" / "analyze_phase37.py"
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
MILESTONES = (40, 80, 120, 160, 200)


def _load_phase37() -> Any:
    spec = importlib.util.spec_from_file_location("phase37_helpers", PHASE37_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 3.7 analysis helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_pairing_control(
    clean_dir: Path,
    fedrad_dir: Path,
    clean_rows: list[dict[str, Any]],
    fedrad_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(clean_rows) != 200 or len(fedrad_rows) != 200:
        raise ValueError("Phase 3.8 requires two complete 200-round runs")
    clean_config = _json(clean_dir / "config.json")
    fedrad_config = _json(fedrad_dir / "config.json")
    mismatches = {
        key: [clean_config.get(key), fedrad_config.get(key)]
        for key in CORE_CONFIG_KEYS
        if clean_config.get(key) != fedrad_config.get(key)
    }
    if mismatches:
        raise ValueError(f"controlled config mismatch: {mismatches}")
    clean_summary = _json(clean_dir / "summary.json")
    fedrad_summary = _json(fedrad_dir / "summary.json")
    if clean_summary["initial_state_hash"] != fedrad_summary["initial_state_hash"]:
        raise ValueError("initial state hashes differ")
    if clean_summary["partition_fingerprint"] != fedrad_summary["partition_fingerprint"]:
        raise ValueError("partition fingerprints differ")

    # Task state hashes legitimately diverge after round 1 because each new bank is
    # reset from the method-specific global state. Seeds/specs, sampling, and the
    # baseline pairing must remain identical across the complete replay.
    replay_keys = ("selected_clients", "task_seeds", "local_seeds")
    for clean_row, fedrad_row in zip(clean_rows, fedrad_rows):
        if clean_row["round_number"] != fedrad_row["round_number"]:
            raise ValueError("round-number mismatch")
        for key in replay_keys:
            if clean_row[key] != fedrad_row[key]:
                raise ValueError(
                    f"controlled replay mismatch at round {clean_row['round_number']}: {key}"
                )
        if clean_row["assignments"] != fedrad_row["baseline_assignments"]:
            raise ValueError(
                f"baseline-pairing mismatch at round {clean_row['round_number']}"
            )
    if clean_rows[0]["task_hashes"] != fedrad_rows[0]["task_hashes"]:
        raise ValueError("round-1 task hashes differ despite identical initial state")
    return {
        "validated": True,
        "rounds": 200,
        "seed": clean_config["seed"],
        "initial_state_hash": clean_summary["initial_state_hash"],
        "partition_fingerprint": clean_summary["partition_fingerprint"],
        "identical_replay_fields": list(replay_keys),
        "round_1_task_hashes_identical": True,
        "later_task_hashes_note": "expected to diverge after the pairing-dependent global trajectories separate",
        "only_method_difference": "client-to-task pairing",
    }


def _accuracy(clean_rows: list[dict[str, Any]], fedrad_rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = np.asarray([row["diagnostic_accuracy"] for row in clean_rows], dtype=np.float64)
    fedrad = np.asarray([row["diagnostic_accuracy"] for row in fedrad_rows], dtype=np.float64)
    delta = fedrad - clean

    def method_metrics(values: np.ndarray) -> dict[str, Any]:
        peak = int(np.argmax(values))
        return {
            "round_40": float(values[39]),
            "round_80": float(values[79]),
            "round_120": float(values[119]),
            "round_160": float(values[159]),
            "round_200": float(values[199]),
            "last_20_mean": float(values[-20:].mean()),
            "last_50_mean": float(values[-50:].mean()),
            "trajectory_mean": float(values.mean()),
            "trapezoidal_auc_accuracy_rounds": float(np.trapezoid(values, dx=1.0)),
            "peak": float(values[peak]),
            "peak_round": peak + 1,
        }

    clean_metrics = method_metrics(clean)
    fedrad_metrics = method_metrics(fedrad)
    scalar_keys = tuple(key for key in clean_metrics if key != "peak_round")
    blocks = ((1, 40), (41, 80), (81, 120), (121, 160), (161, 200), (101, 200))
    ratios = {}
    for start, end in blocks:
        segment = delta[start - 1 : end]
        ratios[f"rounds_{start}_{end}"] = {
            "fedrad_greater_ratio": float(np.mean(segment > 0.0)),
            "fedrad_greater_rounds": int(np.sum(segment > 0.0)),
            "ties": int(np.sum(segment == 0.0)),
            "mean_delta": float(segment.mean()),
        }

    trailing = np.full(delta.shape, np.nan, dtype=np.float64)
    trailing[9:] = np.convolve(delta, np.ones(10) / 10.0, mode="valid")
    trajectory = [
        {
            "round": index + 1,
            "clean_accuracy": float(clean[index]),
            "fedrad_accuracy": float(fedrad[index]),
            "delta_fedrad_minus_clean": float(delta[index]),
            "delta_trailing_10_mean": (
                "" if not np.isfinite(trailing[index]) else float(trailing[index])
            ),
        }
        for index in range(200)
    ]
    return {
        "clean": clean_metrics,
        "fedrad": fedrad_metrics,
        "delta": {key: fedrad_metrics[key] - clean_metrics[key] for key in scalar_keys},
        "win_ratios": ratios,
        "delta_summary": _summary(delta),
        "trajectory": trajectory,
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _matching(
    fedrad_dir: Path,
    fedrad_rows: list[dict[str, Any]],
    delta: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase37 = _load_phase37()
    dynamics, aggregate = phase37._dynamics(fedrad_dir)
    if len(dynamics) != 200:
        raise ValueError("expected 200 matching-dynamics rows")
    for dynamic, round_row, round_delta in zip(dynamics, fedrad_rows, delta):
        dynamic["delta_fedrad_minus_clean"] = float(round_delta)
        dynamic["mean_reset_delta_norm"] = float(round_row["mean_reset_delta_norm"])
        dynamic["mean_active_reset_kernels"] = float(round_row["mean_active_reset_kernels"])
        dynamic["total_active_reset_kernels"] = int(round_row["total_active_reset_kernels"])

    keys = (
        "changed_pairs",
        "changed_fraction",
        "hungarian_score",
        "baseline_score",
        "Q_interaction_std",
        "assignment_margin",
        "margin_per_K",
        "margin_per_Q_std",
        "Gamma",
        "mean_reset_delta_norm",
        "mean_active_reset_kernels",
        "total_active_reset_kernels",
    )

    def segment(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {key: _summary([float(row[key]) for row in rows]) for key in keys}

    predictors = (
        "hungarian_score",
        "Q_interaction_std",
        "assignment_margin",
        "changed_pairs",
    )
    lag_correlations: dict[str, Any] = {}
    for predictor in predictors:
        x_all = np.asarray([row[predictor] for row in dynamics], dtype=np.float64)
        lag_correlations[predictor] = {}
        for lag in (1, 5, 10):
            lag_correlations[predictor][f"lag_{lag}"] = _correlation(
                x_all[:-lag], delta[lag:]
            )

    return dynamics, {
        "all_rounds": segment(dynamics),
        "rounds_1_100": segment(dynamics[:100]),
        "rounds_101_200": segment(dynamics[100:]),
        "phase37_aggregate": aggregate,
        "lag_correlations_with_future_accuracy_delta": lag_correlations,
    }


def _cost(clean_dir: Path, fedrad_dir: Path, clean_rows: list[dict[str, Any]], fedrad_rows: list[dict[str, Any]], accuracy: dict[str, Any]) -> dict[str, Any]:
    clean_wall = (clean_dir / "summary.json").stat().st_mtime - (clean_dir / "config.json").stat().st_mtime
    fedrad_wall = (fedrad_dir / "summary.json").stat().st_mtime - (fedrad_dir / "config.json").stat().st_mtime
    clean_round_loop = float(sum(float(row["round_seconds"]) for row in clean_rows))
    probe = float(sum(float(row["probe_seconds"]) for row in fedrad_rows))
    formal = float(sum(float(row["formal_training_seconds"]) for row in fedrad_rows))
    matching = float(sum(float(row["matching_seconds"]) for row in fedrad_rows))
    additional = fedrad_wall - clean_wall
    return {
        "clean_wall_seconds": float(clean_wall),
        "clean_round_loop_seconds": clean_round_loop,
        "clean_formal_training_seconds": None,
        "clean_formal_training_note": "not separately instrumented; round_seconds covers task construction, formal local training, aggregation, evaluation, and logging",
        "fedrad_wall_seconds": float(fedrad_wall),
        "fedrad_probe_seconds": probe,
        "fedrad_formal_training_seconds": formal,
        "fedrad_matching_seconds": matching,
        "fedrad_other_seconds_approx": float(fedrad_wall - probe - formal - matching),
        "fedrad_peak_gpu_memory_bytes": int(max(row["peak_gpu_memory_bytes"] for row in fedrad_rows)),
        "runtime_ratio_fedrad_over_clean": float(fedrad_wall / clean_wall),
        "additional_wall_seconds": float(additional),
        "final_accuracy_pp_per_additional_hour": float(accuracy["delta"]["round_200"] / (additional / 3600.0)),
        "trajectory_mean_pp_per_additional_hour": float(accuracy["delta"]["trajectory_mean"] / (additional / 3600.0)),
    }


def _verdict(accuracy: dict[str, Any]) -> tuple[str, str]:
    delta = accuracy["delta"]
    late_ratio = accuracy["win_ratios"]["rounds_101_200"]["fedrad_greater_ratio"]
    supported = (
        delta["round_200"] > 0.0
        and delta["last_20_mean"] > 0.0
        and delta["last_50_mean"] > 0.0
        and delta["trajectory_mean"] > 0.0
        and late_ratio > 0.5
    )
    positive = sum(
        value > 0.0
        for value in (
            delta["round_200"],
            delta["last_20_mean"],
            delta["last_50_mean"],
            delta["trajectory_mean"],
        )
    )
    if supported:
        return (
            "global utility supported in current run",
            "final, last-20, last-50, and trajectory mean are positive, and FedRAD wins a majority of rounds 101-200",
        )
    if positive >= 2 or delta["peak"] > 0.0:
        return (
            "global utility weak",
            "some utility indicators are positive, but the complete late-trajectory criteria are not all met",
        )
    return (
        "global utility unsupported",
        "the controlled trajectory does not show a consistent global-optimization advantage",
    )


def _plot(path: Path, trajectory: list[dict[str, Any]]) -> None:
    rounds = np.asarray([row["round"] for row in trajectory])
    delta = np.asarray([row["delta_fedrad_minus_clean"] for row in trajectory])
    moving = np.asarray(
        [np.nan if row["delta_trailing_10_mean"] == "" else row["delta_trailing_10_mean"] for row in trajectory],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axis.plot(rounds, delta, color="#8fb9df", linewidth=0.8, alpha=0.7, label="per-round delta")
    axis.plot(rounds, moving, color="#145a8d", linewidth=2.0, label="trailing 10-round mean")
    axis.set(xlabel="Communication round", ylabel="Accuracy delta (percentage points)", title="FedRAD - Clean accuracy")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _report(report: dict[str, Any]) -> str:
    accuracy = report["accuracy"]
    clean = accuracy["clean"]
    fedrad = accuracy["fedrad"]
    delta = accuracy["delta"]
    cost = report["cost"]
    dynamics = report["matching_dynamics"]["all_rounds"]
    late = report["matching_dynamics"]["rounds_101_200"]
    rows = [
        ("Round 40", "round_40"),
        ("Round 80", "round_80"),
        ("Round 120", "round_120"),
        ("Round 160", "round_160"),
        ("Round 200", "round_200"),
        ("Last-20", "last_20_mean"),
        ("Last-50", "last_50_mean"),
        ("1-200 AUC/Mean", "trajectory_mean"),
        ("Peak (diagnostic only)", "peak"),
    ]
    lines = [
        "# Phase 3.8: Single-Seed Global Utility Validation",
        "",
        "## 1. Accuracy",
        "",
        "| Metric | Clean | FedRAD | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {label} | {clean[key]:.3f}% | {fedrad[key]:.3f}% | {delta[key]:+.3f}pp |"
        for label, key in rows
    )
    lines.extend(
        [
            "",
            f"Peak rounds: Clean {clean['peak_round']}; FedRAD {fedrad['peak_round']}.",
            "",
            "## 2. Difference trajectory",
            "",
            "| Segment | FedRAD > Clean | Mean delta |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, value in accuracy["win_ratios"].items():
        label = key.removeprefix("rounds_").replace("_", "-")
        lines.append(
            f"| Rounds {label} | {value['fedrad_greater_rounds']}/{int(label.split('-')[1]) - int(label.split('-')[0]) + 1} ({100 * value['fedrad_greater_ratio']:.1f}%) | {value['mean_delta']:+.3f}pp |"
        )
    lines.extend(
        [
            "",
            "The per-round delta and trailing 10-round mean are saved in `accuracy_delta.csv` and `accuracy_delta.png`.",
            "",
            "## 3. Matching dynamics",
            "",
            f"- Changed pairs: {dynamics['changed_pairs']['mean']:.2f}/10 on average; rounds 101-200: {late['changed_pairs']['mean']:.2f}/10.",
            f"- Q interaction std: {dynamics['Q_interaction_std']['mean']:.4f} overall; late: {late['Q_interaction_std']['mean']:.4f}.",
            f"- Assignment margin: {dynamics['assignment_margin']['mean']:.4f} overall; late: {late['assignment_margin']['mean']:.4f}.",
            f"- Gamma: {dynamics['Gamma']['mean']:.4f} overall; late: {late['Gamma']['mean']:.4f}.",
            f"- Reset delta norm: {dynamics['mean_reset_delta_norm']['mean']:.4f} overall; late: {late['mean_reset_delta_norm']['mean']:.4f}. Active reset kernels remained {dynamics['total_active_reset_kernels']['mean']:.1f} per round on average.",
            "- Fixed lag-1/5/10 Pearson and Spearman results are saved in `phase38_report.json`; these are descriptive only.",
            "",
            "## 4. Cost",
            "",
            f"- Clean wall runtime: {cost['clean_wall_seconds'] / 60:.1f} min; measured round-loop runtime: {cost['clean_round_loop_seconds'] / 60:.1f} min.",
            "- Clean formal-training time was not separately instrumented in this completed run; its round timer also includes task construction, aggregation, evaluation, and logging.",
            f"- FedRAD wall runtime: {cost['fedrad_wall_seconds'] / 60:.1f} min; probe {cost['fedrad_probe_seconds'] / 60:.1f} min; formal training {cost['fedrad_formal_training_seconds'] / 60:.1f} min; matching {cost['fedrad_matching_seconds']:.3f}s.",
            f"- FedRAD/Clean wall-runtime ratio: {cost['runtime_ratio_fedrad_over_clean']:.3f}x; peak allocated GPU memory: {cost['fedrad_peak_gpu_memory_bytes'] / 2**20:.1f} MiB.",
            f"- Engineering diagnostic: {cost['final_accuracy_pp_per_additional_hour']:+.3f} final-accuracy pp and {cost['trajectory_mean_pp_per_additional_hour']:+.3f} trajectory-mean pp per additional GPU-hour.",
            "",
            "## 5. Final conclusion",
            "",
            f"`{report['verdict']}`",
            "",
            report["verdict_reason"] + ". This is a single-seed controlled mechanism result, not a cross-seed robustness claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_dir", type=Path)
    parser.add_argument("fedrad_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    clean_dir = args.clean_dir.resolve()
    fedrad_dir = args.fedrad_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = _jsonl(clean_dir / "rounds.jsonl")
    fedrad_rows = _jsonl(fedrad_dir / "rounds.jsonl")
    control = _validate_pairing_control(clean_dir, fedrad_dir, clean_rows, fedrad_rows)
    accuracy = _accuracy(clean_rows, fedrad_rows)
    delta = np.asarray(
        [row["delta_fedrad_minus_clean"] for row in accuracy["trajectory"]],
        dtype=np.float64,
    )
    dynamics_rows, dynamics = _matching(fedrad_dir, fedrad_rows, delta)
    cost = _cost(clean_dir, fedrad_dir, clean_rows, fedrad_rows, accuracy)
    verdict, reason = _verdict(accuracy)
    report = {
        "protocol": {
            "phase": "3.8",
            "seed": 1,
            "rounds": 200,
            "probe_support": 64,
            "probe_query": 32,
            "probe_steps": 1,
            "score": "Z(G) + Z(A) - Z(D) + 0.25 Z(C)",
            "assignment": "Hungarian",
            "development_force_hungarian": True,
            "heldout_probe_added": False,
        },
        "controlled_replay": control,
        "accuracy": {key: value for key, value in accuracy.items() if key != "trajectory"},
        "matching_dynamics": dynamics,
        "cost": cost,
        "verdict": verdict,
        "verdict_reason": reason,
    }
    _write_csv(output_dir / "accuracy_delta.csv", accuracy["trajectory"])
    _write_csv(output_dir / "matching_dynamics.csv", dynamics_rows)
    _plot(output_dir / "accuracy_delta.png", accuracy["trajectory"])
    (output_dir / "phase38_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "phase38_report.md").write_text(_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
