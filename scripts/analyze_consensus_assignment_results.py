from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/consensus_assignment_functional_recovery_2of64")
MODES = ("consensus_lock", "union_restrict", "bilateral_gain")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def completed_run(mode: str) -> Path:
    matches: list[Path] = []
    for run in ROOT.iterdir():
        if not run.is_dir() or not (run / "summary.json").is_file():
            continue
        config = read_json(run / "config.json")
        summary = read_json(run / "summary.json")
        if (
            config.get("functional_assignment_mode") == mode
            and summary.get("status") == "complete"
            and summary.get("rounds") == 200
        ):
            matches.append(run)
    if not matches:
        raise FileNotFoundError(f"No complete run for {mode}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def accuracy(rounds: list[dict[str, Any]]) -> dict[str, float | int]:
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
    if len(left_rows) != len(right_rows):
        raise ValueError("Controlled assignment logs differ in length")
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
    return mean(values)


def stage_deltas(
    rounds: list[dict[str, Any]],
    clean: list[dict[str, Any]],
    m2: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    values = [float(row["diagnostic_accuracy"]) for row in rounds]
    clean_values = [float(row["diagnostic_accuracy"]) for row in clean]
    m2_values = [float(row["diagnostic_accuracy"]) for row in m2]
    result: list[dict[str, float | str]] = []
    for start, end in ((1, 40), (41, 80), (81, 120), (121, 160), (161, 200)):
        indices = range(start - 1, end)
        result.append(
            {
                "stage": f"{start}-{end}",
                "delta_clean_pp": mean(values[i] - clean_values[i] for i in indices),
                "delta_m2_pp": mean(values[i] - m2_values[i] for i in indices),
            }
        )
    return result


def report(payload: dict[str, Any]) -> str:
    references = payload["references"]
    rows = [
        "# Consensus-Aware M2 Assignment",
        "",
        "## Controlled variants",
        "",
        "All variants retain M2's two independent raw-G probes and mean score. "
        "Only the one-to-one assignment feasible set changes:",
        "",
        "- `consensus_lock`: lock edges shared by the two Probe-specific Hungarian permutations, then solve the remainder on mean G.",
        "- `union_restrict`: allow baseline edges and edges selected by either Probe-specific Hungarian permutation.",
        "- `bilateral_gain`: allow a reassignment edge only when both probes score it above that client's baseline edge.",
        "",
        "## Accuracy",
        "",
        "| Variant | Peak | Peak round | Delta Clean | Delta M2 | Delta 74.42 | Final | Last20 | Last50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Clean | {references['clean']['peak']:.2f}% | {references['clean']['peak_round']} | — | -0.64 pp | -0.72 pp | {references['clean']['final']:.2f}% | {references['clean']['last20']:.3f}% | {references['clean']['last50']:.3f}% |",
        f"| M2 mean | {references['m2']['peak']:.2f}% | {references['m2']['peak_round']} | +0.64 pp | — | -0.08 pp | {references['m2']['final']:.2f}% | {references['m2']['last20']:.3f}% | {references['m2']['last50']:.3f}% |",
        f"| half-SE | {references['half_se']['peak']:.2f}% | {references['half_se']['peak_round']} | +0.72 pp | +0.08 pp | — | {references['half_se']['final']:.2f}% | {references['half_se']['last20']:.3f}% | {references['half_se']['last50']:.3f}% |",
    ]
    for item in payload["variants"]:
        rows.append(
            f"| {item['mode']} | {item['peak']:.2f}% | {item['peak_round']} | "
            f"{item['delta_clean_pp']:+.2f} pp | {item['delta_m2_pp']:+.2f} pp | "
            f"{item['delta_best_pp']:+.2f} pp | {item['final']:.2f}% | "
            f"{item['last20']:.3f}% | {item['last50']:.3f}% |"
        )
    rows.extend(
        [
            "",
            "No structural variant beats M2 or the nominal half-SE peak. "
            "Consensus-Lock is best of this group but remains 0.24 pp below M2.",
            "",
            "## Assignment behavior",
            "",
            "| Variant | Reassignment rate | Shared Probe edges/10 | Final supported by both | Final supported by either | Reassigned edges with bilateral gain | Overlap with M2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["variants"]:
        rows.append(
            f"| {item['mode']} | {100*item['assignment_change_rate']:.2f}% | "
            f"{item['consensus_pairs']:.2f} | {100*item['supported_both']:.2f}% | "
            f"{100*item['supported_either']:.2f}% | {100*item['bilateral_gain_rate']:.2f}% | "
            f"{100*item['assignment_overlap_with_m2']:.2f}% |"
        )
    rows.extend(["", "## Stage-wise mean accuracy delta", ""])
    for item in payload["variants"]:
        rows.extend(
            [
                f"### {item['mode']}",
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
            "Stop the consensus/partial-Hungarian direction. The light constraint "
            "barely changes the global reassignment rate and loses peak accuracy; "
            "the stronger feasible-set restrictions reduce or certify reassignments "
            "as intended but degrade accuracy further. Return to plain M2 as the "
            "clean primary method. The 74.42% half-SE result remains only a nominal "
            "seed-1 peak, not sufficient evidence to complicate the main method.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    prior = read_json(
        Path("results/multiprobe_functional_recovery_2of64/result_summary.json")
    )
    reliability = read_json(
        Path("results/reliability_aware_functional_recovery_2of64/result_summary.json")
    )
    clean_ref = prior["references"]["clean"]
    m2_ref = prior["variants"][0]
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
    for mode in MODES:
        run = completed_run(mode)
        rounds = read_jsonl(run / "rounds.jsonl")
        metrics = accuracy(rounds)
        metrics.update(
            {
                "mode": mode,
                "run_directory": run.as_posix(),
                "delta_clean_pp": metrics["peak"] - clean_ref["peak"],
                "delta_m2_pp": metrics["peak"] - m2_ref["peak"],
                "delta_best_pp": metrics["peak"] - half_ref["peak"],
                "assignment_change_rate": mean(
                    float(row["assignment_change_rate"]) for row in rounds
                ),
                "consensus_pairs": mean(
                    float(row["replicate_consensus_pair_count"]) for row in rounds
                ),
                "supported_both": mean(
                    float(row["final_pair_supported_by_both_rate"])
                    for row in rounds
                ),
                "supported_either": mean(
                    float(row["final_pair_supported_by_either_rate"])
                    for row in rounds
                ),
                "bilateral_gain_rate": mean(
                    float(row["reassigned_pair_bilateral_gain_rate"])
                    for row in rounds
                ),
                "assignment_overlap_with_m2": assignment_overlap(
                    run / "assignments.jsonl", m2_run / "assignments.jsonl"
                ),
                "stage_deltas": stage_deltas(rounds, clean_rounds, m2_rounds),
            }
        )
        variants.append(metrics)
    payload = {
        "experiment": "Consensus-Aware M2 Assignment",
        "setting": "CIFAR-10 ResNet18 beta=0.3 seed=1 200 rounds faithful FedPhoenix reset_ratio=2/64",
        "references": {"clean": clean_ref, "m2": m2_ref, "half_se": half_ref},
        "variants": variants,
        "best_structural_variant": "consensus_lock",
        "best_structural_peak": variants[0]["peak"],
        "beats_m2": False,
        "beats_current_best": False,
        "decision": "Stop consensus/partial assignment optimization and retain plain M2 as the clean primary method.",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "result_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "consensus_assignment_report.md").write_text(
        report(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
