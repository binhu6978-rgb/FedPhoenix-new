from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(root: Path, marker: str) -> Path:
    matches = sorted(
        (path for path in root.iterdir() if path.is_dir() and marker in path.name),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"no run directory matching {marker!r}")
    return matches[-1]


def _metrics(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _jsonl(run_dir / "rounds.jsonl")
    if len(rows) != 200:
        raise ValueError(f"{run_dir.name} has {len(rows)} rounds, expected 200")
    values = np.asarray([row["diagnostic_accuracy"] for row in rows], dtype=np.float64)
    peak_index = int(np.argmax(values))
    wall_seconds = (run_dir / "summary.json").stat().st_mtime - (
        run_dir / "config.json"
    ).stat().st_mtime
    return rows, {
        "peak": float(values[peak_index]),
        "peak_round": peak_index + 1,
        "final": float(values[-1]),
        "last20": float(values[-20:].mean()),
        "last50": float(values[-50:].mean()),
        "runtime_seconds": float(wall_seconds),
    }


def _validate_warmup(
    name: str,
    run_dir: Path,
    rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    warmup_rounds: int,
    score_mode: str,
) -> dict[str, Any]:
    config = _json(run_dir / "config.json")
    expected_start = warmup_rounds + 1
    if config["matching_start_round"] != expected_start:
        raise ValueError(f"{name}: matching_start_round mismatch")
    if config["score_mode"] != score_mode:
        raise ValueError(f"{name}: score_mode mismatch")
    if not config["development_force_hungarian"]:
        raise ValueError(f"{name}: forced Hungarian is not enabled")

    replay_keys = (
        "selected_clients",
        "assignments",
        "task_seeds",
        "task_hashes",
        "local_seeds",
        "global_state_hash",
        "diagnostic_accuracy",
    )
    for index in range(warmup_rounds):
        for key in replay_keys:
            if rows[index][key] != clean_rows[index][key]:
                raise ValueError(f"{name}: warm-up mismatch round {index + 1}: {key}")
        if rows[index]["fallback_reason"] != "matching_inactive":
            raise ValueError(f"{name}: matching active during warm-up")
        if rows[index]["matching_active"] is not False:
            raise ValueError(f"{name}: matching_active flag is not false")
        if any(float(rows[index][key]) != 0.0 for key in ("probe_seconds", "matching_seconds")):
            raise ValueError(f"{name}: Probe/matching cost occurred during warm-up")
    for row in rows[warmup_rounds:]:
        if row["matching_active"] is not True:
            raise ValueError(f"{name}: matching inactive after start")

    assignments = _jsonl(run_dir / "assignments.jsonl")
    if len(assignments) != 200 - warmup_rounds:
        raise ValueError(f"{name}: unexpected assignment-log length")
    if int(assignments[0]["round_number"]) != expected_start:
        raise ValueError(f"{name}: assignment log starts at wrong round")
    score_files = sorted((run_dir / "scores").glob("round_*.npz"))
    if len(score_files) != 200 - warmup_rounds:
        raise ValueError(f"{name}: unexpected score-file count")
    if score_files[0].stem != f"round_{expected_start:04d}":
        raise ValueError(f"{name}: score files start at wrong round")

    delta = np.asarray(
        [row["diagnostic_accuracy"] for row in rows[warmup_rounds:]], dtype=np.float64
    ) - np.asarray(
        [row["diagnostic_accuracy"] for row in clean_rows[warmup_rounds:]],
        dtype=np.float64,
    )
    return {
        "validated": True,
        "warmup_rounds": warmup_rounds,
        "global_hashes_equal_to_clean": warmup_rounds,
        "probe_skipped_during_warmup": True,
        "matching_rounds": 200 - warmup_rounds,
        "post_start_fedrad_greater_ratio": float(np.mean(delta > 0.0)),
        "post_start_mean_delta": float(delta.mean()),
        "post_start_delta_last20": float(delta[-20:].mean()),
    }


def _trajectory_rows(
    variants: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]
) -> list[dict[str, Any]]:
    output = []
    for index in range(200):
        row: dict[str, Any] = {"round": index + 1}
        for name, (rounds, _) in variants.items():
            row[name] = rounds[index]["diagnostic_accuracy"]
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    order = ("Clean", "Full-All", "Full-Warm40", "Full-Warm80", "G-Warm40")
    lines = [
        "# Phase 3.9: Matching Timing / Minimal Score Validation",
        "",
        "| Variant | Peak | Peak Round | Delta vs Clean | Final | Last-20 | Last-50 | Runtime |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    clean_peak = metrics["Clean"]["peak"]
    for name in order:
        item = metrics[name]
        lines.append(
            f"| {name} | {item['peak']:.3f}% | {item['peak_round']} | "
            f"{item['peak'] - clean_peak:+.3f}pp | {item['final']:.3f}% | "
            f"{item['last20']:.3f}% | {item['last50']:.3f}% | "
            f"{item['runtime_seconds'] / 60:.1f} min |"
        )
    lines.extend(
        [
            "",
            "## Warm-up validation",
            "",
        ]
    )
    for name, item in report["warmup_validation"].items():
        lines.append(
            f"- {name}: all {item['warmup_rounds']} warm-up global hashes and replay fields equal Clean; "
            f"post-start win ratio {100 * item['post_start_fedrad_greater_ratio']:.1f}%, "
            f"mean delta {item['post_start_mean_delta']:+.3f}pp."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best variant: **{report['best_variant']}**.",
            f"- Peak improvement over Clean: **{report['peak_improvement_over_clean']:+.3f}pp**.",
            f"- Peak improvement over current Full-All: **{report['peak_improvement_over_full_all']:+.3f}pp**.",
            f"- {report['g_only_decision']}",
            "- Runtime for V1/V2/V3 is wall-clock time under concurrent GPU execution and is not a clean throughput benchmark.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_dir", type=Path)
    parser.add_argument("full_all_dir", type=Path)
    parser.add_argument("phase39_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    clean_dir = args.clean_dir.resolve()
    full_all_dir = args.full_all_dir.resolve()
    phase39_root = args.phase39_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "Clean": clean_dir,
        "Full-All": full_all_dir,
        "Full-Warm40": _run(phase39_root, "full_warm40_200r_controlled"),
        "Full-Warm80": _run(phase39_root, "full_warm80_200r_controlled"),
        "G-Warm40": _run(phase39_root, "g_warm40_200r_controlled"),
    }
    variants = {name: _metrics(path) for name, path in run_dirs.items()}
    clean_rows = variants["Clean"][0]
    validations = {
        "Full-Warm40": _validate_warmup(
            "Full-Warm40", run_dirs["Full-Warm40"], variants["Full-Warm40"][0], clean_rows, 40, "full"
        ),
        "Full-Warm80": _validate_warmup(
            "Full-Warm80", run_dirs["Full-Warm80"], variants["Full-Warm80"][0], clean_rows, 80, "full"
        ),
        "G-Warm40": _validate_warmup(
            "G-Warm40", run_dirs["G-Warm40"], variants["G-Warm40"][0], clean_rows, 40, "g_only"
        ),
    }
    metrics = {name: value[1] for name, value in variants.items()}
    best = max(metrics, key=lambda name: metrics[name]["peak"])
    g_only_decision = (
        "G-only direction stops: G-Warm40 did not exceed the best Full timing variant."
        if metrics["G-Warm40"]["peak"] <= metrics["Full-Warm40"]["peak"]
        else "G-only remains the best candidate."
    )
    report = {
        "protocol": {
            "phase": "3.9",
            "seed": 1,
            "rounds": 200,
            "primary_metric": "peak full accuracy",
            "runs_concurrent": True,
        },
        "run_directories": {name: str(path) for name, path in run_dirs.items()},
        "metrics": metrics,
        "warmup_validation": validations,
        "best_variant": best,
        "peak_improvement_over_clean": metrics[best]["peak"] - metrics["Clean"]["peak"],
        "peak_improvement_over_full_all": metrics[best]["peak"] - metrics["Full-All"]["peak"],
        "g_only_decision": g_only_decision,
    }
    _write_csv(output_dir / "phase39_trajectories.csv", _trajectory_rows(variants))
    (output_dir / "phase39_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "phase39_report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
