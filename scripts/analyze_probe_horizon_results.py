from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/probe_horizon_functional_recovery_2of64")
DEPTHS = (2, 3, 5)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def complete_run(depth: int) -> Path:
    runs: list[Path] = []
    for run in ROOT.iterdir():
        if not run.is_dir() or not (run / "summary.json").is_file():
            continue
        config = read_json(run / "config.json")
        summary = read_json(run / "summary.json")
        if (
            config.get("probe_steps") == depth
            and config.get("functional_probe_replicates") == 2
            and config.get("functional_assignment_mode") == "hungarian"
            and summary.get("status") == "complete"
            and summary.get("rounds") == 200
        ):
            runs.append(run)
    if not runs:
        raise FileNotFoundError(f"No complete {depth}-step run")
    return max(runs, key=lambda path: path.stat().st_mtime_ns)


def accuracy_metrics(rounds: list[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(row["diagnostic_accuracy"]) for row in rounds]
    peak = max(values)
    return {
        "peak": peak,
        "peak_round": values.index(peak) + 1,
        "final": values[-1],
        "last20": mean(values[-20:]),
        "last50": mean(values[-50:]),
    }


def assignment_overlap(left: Path, right: Path) -> float:
    left_rows = read_jsonl(left)
    right_rows = read_jsonl(right)
    values: list[float] = []
    for left_round, right_round in zip(left_rows, right_rows):
        left_map = {
            int(pair["client_id"]): int(pair["task_id"])
            for pair in left_round["final_pairs"]
        }
        right_map = {
            int(pair["client_id"]): int(pair["task_id"])
            for pair in right_round["final_pairs"]
        }
        if left_map.keys() != right_map.keys():
            raise ValueError("Client sampling differs between controlled runs")
        values.append(
            sum(left_map[client] == right_map[client] for client in left_map)
            / len(left_map)
        )
    if len(values) != 200:
        raise ValueError("Expected 200 controlled assignment rounds")
    return mean(values)


def stage_deltas(
    rounds: list[dict[str, Any]],
    clean_rounds: list[dict[str, Any]],
    m2_rounds: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    values = [float(row["diagnostic_accuracy"]) for row in rounds]
    clean = [float(row["diagnostic_accuracy"]) for row in clean_rounds]
    m2 = [float(row["diagnostic_accuracy"]) for row in m2_rounds]
    output: list[dict[str, float | str]] = []
    for start, end in ((1, 40), (41, 80), (81, 120), (121, 160), (161, 200)):
        indices = range(start - 1, end)
        output.append(
            {
                "stage": f"{start}-{end}",
                "delta_clean_pp": mean(values[i] - clean[i] for i in indices),
                "delta_m2_pp": mean(values[i] - m2[i] for i in indices),
            }
        )
    return output


def make_report(payload: dict[str, Any]) -> str:
    refs = payload["references"]
    rows = [
        "# M2 Functional Recovery: Probe Horizon",
        "",
        "## Controlled change",
        "",
        "The experiment keeps M2 mean aggregation, two independent materialized "
        "support/query batches, raw functional recovery G, Hungarian assignment, "
        "formal LocalTrainer, FedAvg, and faithful FedPhoenix 2/64 unchanged. Only "
        "the number of SGD updates on each fixed Probe support batch changes.",
        "",
        "## Accuracy",
        "",
        "| Probe steps | Peak | Peak round | Delta Clean | Delta M2 | Delta 74.42 | Final | Last20 | Last50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 1 (M2) | {refs['m2']['peak']:.2f}% | {refs['m2']['peak_round']} | +0.64 pp | — | -0.08 pp | {refs['m2']['final']:.2f}% | {refs['m2']['last20']:.3f}% | {refs['m2']['last50']:.3f}% |",
    ]
    for item in payload["variants"]:
        best = item["steps"] == 5
        step_label = "**5**" if best else str(item["steps"])
        peak = f"**{item['peak']:.2f}%**" if best else f"{item['peak']:.2f}%"
        rows.append(
            f"| {step_label} | {peak} | {item['peak_round']} | "
            f"{item['delta_clean_pp']:+.2f} pp | {item['delta_m2_pp']:+.2f} pp | "
            f"{item['delta_record_pp']:+.2f} pp | {item['final']:.2f}% | "
            f"{item['last20']:.3f}% | {item['last50']:.3f}% |"
        )
    rows.extend(
        [
            "",
            "Five steps reaches 74.70% at round 164: +0.36 pp over M2 and "
            "+0.28 pp over the previous nominal record. Two steps is slightly below "
            "M2 and three steps is substantially worse, so the depth effect is "
            "non-monotonic.",
            "",
            "## Probe and assignment behavior",
            "",
            "| Steps | Score SE | Replicate-to-mean overlap | Reassignment rate | Assignment margin | Assignment overlap with 1-step M2 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["variants"]:
        rows.append(
            f"| {item['steps']} | {item['score_standard_error']:.5f} | "
            f"{100*item['replicate_assignment_overlap']:.2f}% | "
            f"{100*item['assignment_change_rate']:.2f}% | "
            f"{item['assignment_margin']:.5f} | "
            f"{100*item['assignment_overlap_with_m2']:.2f}% |"
        )
    rows.extend(["", "## Stage-wise mean accuracy delta", ""])
    for item in payload["variants"]:
        rows.extend(
            [
                f"### {item['steps']}-step",
                "",
                "| Rounds | Delta Clean | Delta M2 |",
                "|---|---:|---:|",
            ]
        )
        for stage in item["stage_deltas"]:
            rows.append(
                f"| {stage['stage']} | {stage['delta_clean_pp']:+.3f} pp | {stage['delta_m2_pp']:+.3f} pp |"
            )
        rows.append("")
    rows.extend(
        [
            "## Decision",
            "",
            "Retain 5-step M2 Functional Recovery as the new seed-1 peak leader and "
            "stop the Probe-depth sweep at the pre-specified success condition. The "
            "gain is a peak-specific result: its Final, Last20, and Last50 do not beat "
            "plain M2. The increased score uncertainty but larger assignment margin "
            "suggest that the benefit comes from measuring a different recovery "
            "horizon, not from denoising the Probe.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    multiprobe = read_json(
        Path("results/multiprobe_functional_recovery_2of64/result_summary.json")
    )
    reliability = read_json(
        Path("results/reliability_aware_functional_recovery_2of64/result_summary.json")
    )
    clean_ref = multiprobe["references"]["clean"]
    m2_ref = multiprobe["variants"][0]
    half_ref = reliability["variants"][0]
    m2_run = next(
        Path("results/multiprobe_functional_recovery_2of64").glob(
            "*multiprobe_m2_seed1_2of64_200r"
        )
    )
    clean_path = next(
        Path("results/fedrad_fidelity_2of64").glob(
            "clean_fedphoenix_*clean_seed1_2of64_200r_faithful/rounds.jsonl"
        )
    )
    clean_rounds = read_jsonl(clean_path)
    m2_rounds = read_jsonl(m2_run / "rounds.jsonl")
    variants: list[dict[str, Any]] = []
    for depth in DEPTHS:
        run = complete_run(depth)
        rounds = read_jsonl(run / "rounds.jsonl")
        summary = read_json(run / "summary.json")
        item = accuracy_metrics(rounds)
        item.update(
            {
                "steps": depth,
                "run_directory": run.as_posix(),
                "delta_clean_pp": item["peak"] - clean_ref["peak"],
                "delta_m2_pp": item["peak"] - m2_ref["peak"],
                "delta_record_pp": item["peak"] - half_ref["peak"],
                "score_standard_error": summary[
                    "mean_pairwise_score_standard_error"
                ],
                "replicate_assignment_overlap": summary[
                    "mean_replicate_assignment_overlap"
                ],
                "assignment_change_rate": summary[
                    "mean_assignment_change_rate"
                ],
                "assignment_margin": summary["mean_assignment_margin"],
                "assignment_overlap_with_m2": assignment_overlap(
                    run / "assignments.jsonl", m2_run / "assignments.jsonl"
                ),
                "stage_deltas": stage_deltas(rounds, clean_rounds, m2_rounds),
            }
        )
        variants.append(item)
    payload = {
        "experiment": "M2 Functional Recovery Probe Horizon",
        "setting": "CIFAR-10 ResNet18 beta=0.3 seed=1 200 rounds faithful FedPhoenix reset_ratio=2/64",
        "references": {"clean": clean_ref, "m2": m2_ref, "half_se": half_ref},
        "variants": variants,
        "best_variant": "M2 5-step",
        "best_peak": variants[-1]["peak"],
        "delta_m2_pp": variants[-1]["delta_m2_pp"],
        "delta_previous_record_pp": variants[-1]["delta_record_pp"],
        "decision": "Retain 5-step as the seed-1 peak leader and stop the Probe-depth sweep.",
    }
    (ROOT / "result_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "probe_horizon_report.md").write_text(
        make_report(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
