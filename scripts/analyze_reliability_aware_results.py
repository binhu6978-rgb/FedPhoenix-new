from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _completed_run(root: Path, mode: str) -> Path:
    candidates: list[Path] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        config_path = directory / "config.json"
        summary_path = directory / "summary.json"
        rounds_path = directory / "rounds.jsonl"
        if not all(path.is_file() for path in (config_path, summary_path, rounds_path)):
            continue
        config = _read_json(config_path)
        summary = _read_json(summary_path)
        if (
            config.get("functional_reliability_mode") == mode
            and config.get("functional_probe_replicates") == 2
            and summary.get("status") == "complete"
            and summary.get("rounds") == 200
        ):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(f"No complete 200-round {mode} run below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _accuracy_metrics(rounds: list[dict[str, Any]]) -> dict[str, float | int]:
    accuracy = [float(row["diagnostic_accuracy"]) for row in rounds]
    peak = max(accuracy)
    return {
        "peak": peak,
        "peak_round": accuracy.index(peak) + 1,
        "final": accuracy[-1],
        "last20": mean(accuracy[-20:]),
        "last50": mean(accuracy[-50:]),
    }


def _mean_field(rounds: list[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rounds)


def _assignment_overlap(left_path: Path, right_path: Path) -> float:
    left = _read_jsonl(left_path)
    right = _read_jsonl(right_path)
    if len(left) != len(right):
        raise ValueError("Assignment logs must contain the same number of rounds")
    overlaps: list[float] = []
    for left_round, right_round in zip(left, right):
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
        overlaps.append(
            sum(left_map[client] == right_map[client] for client in left_map)
            / len(left_map)
        )
    return mean(overlaps)


def _stage_deltas(
    variant: list[dict[str, Any]],
    clean: list[dict[str, Any]],
    m2: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    stages = ((1, 40), (41, 80), (81, 120), (121, 160), (161, 200))
    values = [float(row["diagnostic_accuracy"]) for row in variant]
    clean_values = [float(row["diagnostic_accuracy"]) for row in clean]
    m2_values = [float(row["diagnostic_accuracy"]) for row in m2]
    rows: list[dict[str, float | str]] = []
    for start, end in stages:
        indices = range(start - 1, end)
        rows.append(
            {
                "stage": f"{start}-{end}",
                "delta_clean_pp": mean(values[i] - clean_values[i] for i in indices),
                "delta_m2_pp": mean(values[i] - m2_values[i] for i in indices),
            }
        )
    return rows


def _variant(
    *,
    label: str,
    run: Path,
    m2_reference: dict[str, Any],
    clean_reference: dict[str, Any],
    clean_rounds: list[dict[str, Any]],
    m2_rounds: list[dict[str, Any]],
    m2_assignments: Path,
) -> dict[str, Any]:
    rounds = _read_jsonl(run / "rounds.jsonl")
    metrics = _accuracy_metrics(rounds)
    metrics.update(
        {
            "label": label,
            "run_directory": run.relative_to(Path.cwd().resolve()).as_posix(),
            "delta_clean_pp": metrics["peak"] - float(clean_reference["peak"]),
            "delta_m2_pp": metrics["peak"] - float(m2_reference["peak"]),
            "mean_score_variance": _mean_field(
                rounds, "mean_score_variance_across_replicates"
            ),
            "mean_pairwise_score_standard_error": _mean_field(
                rounds, "mean_pairwise_score_standard_error"
            ),
            "mean_replicate_assignment_overlap": _mean_field(
                rounds, "mean_replicate_assignment_overlap"
            ),
            "mean_assignment_change_rate": _mean_field(
                rounds, "assignment_change_rate"
            ),
            "mean_assignment_margin": _mean_field(rounds, "assignment_margin"),
            "mean_reliability_penalty": _mean_field(
                rounds, "mean_reliability_penalty"
            ),
            "mean_selected_reliability_penalty": _mean_field(
                rounds, "mean_selected_reliability_penalty"
            ),
            "assignment_overlap_with_m2": _assignment_overlap(
                run / "assignments.jsonl", m2_assignments
            ),
            "stage_deltas": _stage_deltas(rounds, clean_rounds, m2_rounds),
        }
    )
    return metrics


def _report(payload: dict[str, Any]) -> str:
    references = payload["references"]
    variants = payload["variants"]
    rows = [
        "# Reliability-Aware M2 Functional Recovery",
        "",
        "## Controlled change",
        "",
        "Both variants retain the accepted two independent raw-G probes, formal "
        "LocalTrainer, original TaskSpec reload, Hungarian matching, FedAvg, client "
        "sampling, and faithful FedPhoenix 2/64 setup. Only the matrix supplied to "
        "Hungarian changes:",
        "",
        "- `half_se_lcb`: `Q = mean(G1,G2) - 0.5 * |G1-G2|/2`.",
        "- `one_se_lcb`: `Q = mean(G1,G2) - |G1-G2|/2 = min(G1,G2)`.",
        "",
        "These are fixed mild and strict reliability corrections, not a lambda sweep.",
        "",
        "## Accuracy",
        "",
        "| Variant | Peak | Peak round | Delta Clean | Delta M2 | Final | Last20 | Last50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (
        ("Clean", references["clean"]),
        ("M1", references["full_ours"]),
        ("M2 mean", references["m2"]),
    ):
        delta_clean = float(item["peak"]) - float(references["clean"]["peak"])
        delta_m2 = float(item["peak"]) - float(references["m2"]["peak"])
        rows.append(
            f"| {label} | {item['peak']:.2f}% | {item['peak_round']} | "
            f"{delta_clean:+.2f} pp | {delta_m2:+.2f} pp | {item['final']:.2f}% | "
            f"{item['last20']:.3f}% | {item['last50']:.3f}% |"
        )
    for item in variants:
        is_best = item["label"] == "half_se_lcb"
        label = "**M2 half-SE LCB**" if is_best else "M2 one-SE LCB"
        peak = f"**{item['peak']:.2f}%**" if is_best else f"{item['peak']:.2f}%"
        rows.append(
            f"| {label} | {peak} | {item['peak_round']} | "
            f"{item['delta_clean_pp']:+.2f} pp | {item['delta_m2_pp']:+.2f} pp | "
            f"{item['final']:.2f}% | {item['last20']:.3f}% | {item['last50']:.3f}% |"
        )
    rows.extend(
        [
            "",
            "The mild correction is the nominal new peak leader at 74.42%, but its "
            "gain over M2 is only 0.08 pp. The strict correction loses 0.46 pp versus "
            "M2, so stronger disagreement penalization is not supported.",
            "",
            "## Assignment diagnostics",
            "",
            "| Variant | Replicate overlap | Change rate | Margin | Penalty (all) | Penalty (selected) | Overlap with M2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in variants:
        rows.append(
            f"| {item['label']} | {100*item['mean_replicate_assignment_overlap']:.2f}% | "
            f"{100*item['mean_assignment_change_rate']:.2f}% | {item['mean_assignment_margin']:.5f} | "
            f"{item['mean_reliability_penalty']:.5f} | "
            f"{item['mean_selected_reliability_penalty']:.5f} | "
            f"{100*item['assignment_overlap_with_m2']:.2f}% |"
        )
    rows.extend(["", "## Stage-wise mean accuracy delta", ""])
    for item in variants:
        rows.extend(
            [
                f"### {item['label']}",
                "",
                "| Rounds | Delta Clean | Delta M2 |",
                "|---|---:|---:|",
            ]
        )
        for stage in item["stage_deltas"]:
            rows.append(
                f"| {stage['stage']} | {stage['delta_clean_pp']:+.3f} pp | "
                f"{stage['delta_m2_pp']:+.3f} pp |"
            )
        rows.append("")
    rows.extend(
        [
            "## Decision",
            "",
            "Retain `half_se_lcb` as the simple seed-1 peak leader, while describing "
            "the +0.08 pp improvement as marginal rather than robust. Stop the "
            "reliability-strength search: the strict analytic endpoint is clearly worse, "
            "and neither rule reduces the approximately nine-of-ten reassignment rate. "
            "The next assignment study, if pursued, should target structural stability "
            "of the permutation rather than apply a larger elementwise uncertainty penalty.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/reliability_aware_functional_recovery_2of64"),
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    prior_root = Path("results/multiprobe_functional_recovery_2of64").resolve()
    prior = _read_json(prior_root / "result_summary.json")
    references = {
        "clean": prior["references"]["clean"],
        "full_ours": prior["references"]["full_ours"],
        "m2": prior["variants"][0],
    }
    clean_path = next(
        Path("results/fedrad_fidelity_2of64").glob(
            "clean_fedphoenix_*clean_seed1_2of64_200r_faithful/rounds.jsonl"
        )
    )
    m2_run = next(prior_root.glob("*multiprobe_m2_seed1_2of64_200r"))
    clean_rounds = _read_jsonl(clean_path)
    m2_rounds = _read_jsonl(m2_run / "rounds.jsonl")
    variants = [
        _variant(
            label=mode,
            run=_completed_run(output_root, mode),
            m2_reference=references["m2"],
            clean_reference=references["clean"],
            clean_rounds=clean_rounds,
            m2_rounds=m2_rounds,
            m2_assignments=m2_run / "assignments.jsonl",
        )
        for mode in ("half_se_lcb", "one_se_lcb")
    ]
    payload = {
        "experiment": "Reliability-Aware M2 Functional Recovery",
        "setting": "CIFAR-10 ResNet18 beta=0.3 seed=1 200 rounds faithful FedPhoenix reset_ratio=2/64",
        "references": references,
        "variants": variants,
        "best_variant": "M2 half-SE LCB",
        "best_peak": variants[0]["peak"],
        "delta_m2_pp": variants[0]["delta_m2_pp"],
        "decision": "Retain half-SE LCB as a marginal seed-1 peak improvement; stop reliability-penalty tuning.",
    }
    (output_root / "result_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "reliability_report.md").write_text(
        _report(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
