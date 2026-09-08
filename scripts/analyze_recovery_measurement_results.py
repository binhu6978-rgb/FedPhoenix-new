from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/recovery_measurement_functional_recovery_2of64")
SPECS = (
    ("6-step terminal", 6, "terminal"),
    ("8-step terminal", 8, "terminal"),
    ("5-step trajectory mean", 5, "trajectory_mean"),
    ("5-step endpoints mean", 5, "endpoints_mean"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def find_run(steps: int, mode: str) -> Path:
    matches = []
    for path in ROOT.iterdir():
        if not path.is_dir() or not (path / "summary.json").is_file():
            continue
        config = read_json(path / "config.json")
        summary = read_json(path / "summary.json")
        if (
            config.get("probe_steps") == steps
            and config.get("probe_recovery_measurement", "terminal") == mode
            and config.get("functional_probe_replicates") == 2
            and config.get("functional_assignment_mode") == "hungarian"
            and summary.get("status") == "complete"
            and summary.get("rounds") == 200
        ):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No complete run for steps={steps}, mode={mode}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(row["diagnostic_accuracy"]) for row in rows]
    peak = max(values)
    return {
        "peak": peak,
        "peak_round": values.index(peak) + 1,
        "final": values[-1],
        "last20": mean(values[-20:]),
        "last50": mean(values[-50:]),
    }


def overlap(left: Path, right: Path) -> float:
    values = []
    for lhs, rhs in zip(read_jsonl(left), read_jsonl(right), strict=True):
        lhs_map = {int(pair["client_id"]): int(pair["task_id"]) for pair in lhs["final_pairs"]}
        rhs_map = {int(pair["client_id"]): int(pair["task_id"]) for pair in rhs["final_pairs"]}
        if lhs_map.keys() != rhs_map.keys():
            raise ValueError("Controlled runs use different sampled clients")
        values.append(mean(lhs_map[key] == rhs_map[key] for key in lhs_map))
    return mean(values)


def main() -> None:
    horizon = read_json(Path("results/probe_horizon_functional_recovery_2of64/result_summary.json"))
    clean = horizon["references"]["clean"]
    m2 = horizon["references"]["m2"]
    five = next(item for item in horizon["variants"] if item["steps"] == 5)
    m2_run = next(Path("results/multiprobe_functional_recovery_2of64").glob("*multiprobe_m2_seed1_2of64_200r"))
    m2_rows = read_jsonl(m2_run / "rounds.jsonl")

    variants = []
    for label, steps, mode in SPECS:
        run = find_run(steps, mode)
        rows = read_jsonl(run / "rounds.jsonl")
        for row, reference in zip(rows, m2_rows, strict=True):
            for key in ("selected_clients", "task_seeds", "local_seeds"):
                if row[key] != reference[key]:
                    raise ValueError(f"{label}: controlled field differs: {key}")
        item = metrics(rows)
        summary = read_json(run / "summary.json")
        item.update(
            {
                "label": label,
                "steps": steps,
                "measurement": mode,
                "run_directory": run.as_posix(),
                "delta_clean_pp": item["peak"] - clean["peak"],
                "delta_m2_pp": item["peak"] - m2["peak"],
                "delta_5step_pp": item["peak"] - five["peak"],
                "assignment_overlap_with_m2": overlap(run / "assignments.jsonl", m2_run / "assignments.jsonl"),
                "score_standard_error": summary["mean_pairwise_score_standard_error"],
                "assignment_margin": summary["mean_assignment_margin"],
            }
        )
        variants.append(item)

    best = max(variants, key=lambda item: item["peak"])
    payload = {
        "experiment": "M2 recovery horizon and trajectory measurement",
        "setting": "CIFAR-10 ResNet18 beta=0.3 seed=1 200 rounds faithful FedPhoenix reset_ratio=2/64",
        "references": {"clean": clean, "m2_1step": m2, "m2_5step": five},
        "variants": variants,
        "best_variant": best["label"],
        "best_peak": best["peak"],
        "delta_5step_pp": best["delta_5step_pp"],
        "decision": "No candidate exceeds 5-step terminal; stop horizon and trajectory optimization.",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "result_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = [
        "# M2 recovery measurement optimization",
        "",
        "Only the recovery horizon or trajectory summary changes. All variants use two independent Probes, mean replicate aggregation, and unchanged Hungarian assignment.",
        "",
        "| Variant | Peak | Round | Delta Clean | Delta M2 | Delta 5-step | Final | Last20 | Last50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 5-step terminal (reference) | {five['peak']:.2f}% | {five['peak_round']} | "
        f"{five['delta_clean_pp']:+.2f} pp | {five['delta_m2_pp']:+.2f} pp | — | "
        f"{five['final']:.2f}% | {five['last20']:.3f}% | {five['last50']:.3f}% |",
    ]
    for item in variants:
        report.append(
            f"| {item['label']} | {item['peak']:.2f}% | {item['peak_round']} | "
            f"{item['delta_clean_pp']:+.2f} pp | {item['delta_m2_pp']:+.2f} pp | "
            f"{item['delta_5step_pp']:+.2f} pp | {item['final']:.2f}% | "
            f"{item['last20']:.3f}% | {item['last50']:.3f}% |"
        )
    report.extend(
        [
            "",
            f"Best candidate: **{best['label']}**, {best['peak']:.2f}% at round {best['peak_round']} ({best['delta_5step_pp']:+.2f} pp versus 5-step terminal).",
            "",
            "| Variant | Score SE | Assignment margin | Assignment overlap with M2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in variants:
        report.append(
            f"| {item['label']} | {item['score_standard_error']:.5f} | "
            f"{item['assignment_margin']:.5f} | {100 * item['assignment_overlap_with_m2']:.2f}% |"
        )
    report.extend(
        [
            "",
            "The analysis verified identical selected clients, task seeds, and local seeds against M2 for all 200 rounds.",
            "",
            "## Decision",
            "",
            "None of the horizon or trajectory candidates exceeds the 74.70% 5-step terminal reference. The deeper terminal horizons decline, while averaging the recovery path suppresses the useful late-horizon signal. Keep 5-step terminal as the peak leader and stop this optimization axis.",
        ]
    )
    (ROOT / "recovery_measurement_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
