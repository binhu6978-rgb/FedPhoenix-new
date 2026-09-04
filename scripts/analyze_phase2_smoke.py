from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean_row_std": float(array.std(axis=1, ddof=0).mean()),
        "mean_column_std": float(array.std(axis=0, ddof=0).mean()),
    }


def _correlations(matrices: dict[str, np.ndarray]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    names = tuple(matrices)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            x = matrices[left].reshape(-1)
            y = matrices[right].reshape(-1)
            key = f"{left}_{right}"
            result[key] = (
                None
                if float(x.std()) == 0.0 or float(y.std()) == 0.0
                else float(np.corrcoef(x, y)[0, 1])
            )
    return result


def analyze(run_dir: Path) -> dict[str, object]:
    rounds = [
        json.loads(line)
        for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assignments = [
        json.loads(line)
        for line in (run_dir / "assignments.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    with (run_dir / "probe_pairs.csv").open(newline="", encoding="utf-8") as handle:
        probes = list(csv.DictReader(handle))

    report: dict[str, object] = {"run_dir": str(run_dir.resolve()), "rounds": []}
    per_round: list[dict[str, object]] = []
    for round_record, assignment in zip(rounds, assignments):
        round_number = int(round_record["round_number"])
        score_path = run_dir / "scores" / f"round_{round_number:04d}.npz"
        score_file = np.load(score_path, allow_pickle=False)
        matrices = {
            name: np.asarray(score_file[name], dtype=np.float64)
            for name in ("G", "A", "D", "C")
        }
        Q = np.asarray(score_file["Q"], dtype=np.float64)
        round_probes = [row for row in probes if int(row["round"]) == round_number]
        per_client: dict[int, list[dict[str, str]]] = {}
        for row in round_probes:
            per_client.setdefault(int(row["client_id"]), []).append(row)
        probe_reuse_valid = all(
            len({row["support_hash"] for row in rows}) == 1
            and len({row["query_hash"] for row in rows}) == 1
            and len({row["probe_seed"] for row in rows}) == 1
            and len({row["global_loss"] for row in rows}) == 1
            for rows in per_client.values()
        )
        hash_by_task = {
            task_id: task_hash
            for task_id, task_hash in enumerate(round_record["task_hashes"])
        }
        formal_hashes_match = all(
            round_record["formal_initial_hashes"][row]
            == hash_by_task[pair[1]]
            for row, pair in enumerate(round_record["assignments"])
        )
        baseline_tasks = [pair["task_id"] for pair in assignment["baseline_pairs"]]
        hungarian_tasks = [pair["task_id"] for pair in assignment["hungarian_pairs"]]
        row_choices = np.argmax(Q, axis=1)
        column_choices = np.argmax(Q, axis=0)
        round_report = {
            "round": round_number,
            "accuracy": round_record["diagnostic_accuracy"],
            "selected_clients": round_record["selected_clients"],
            "task_hash_prefixes": [value[:12] for value in round_record["task_hashes"]],
            "support_hash_prefixes": [
                next(iter({row["support_hash"] for row in per_client[client_id]}))[:12]
                for client_id in round_record["selected_clients"]
            ],
            "query_hash_prefixes": [
                next(iter({row["query_hash"] for row in per_client[client_id]}))[:12]
                for client_id in round_record["selected_clients"]
            ],
            "probe_reuse_and_global_reference_valid": probe_reuse_valid,
            "formal_initial_hashes_match_tasks": formal_hashes_match,
            "component_stats": {name: _stats(value) for name, value in matrices.items()},
            "D_positive_pairs": int(np.count_nonzero(matrices["D"] > 0.0)),
            "alignment_invalid_pairs": sum(
                row["alignment_valid"].lower() != "true" for row in round_probes
            ),
            "component_correlations": _correlations(matrices),
            "component_degenerate": score_file["component_degenerate"].tolist(),
            "Q_stats": _stats(Q),
            "row_argmax_largest_task_demand": int(
                np.bincount(row_choices, minlength=Q.shape[1]).max()
            ),
            "column_argmax_largest_client_demand": int(
                np.bincount(column_choices, minlength=Q.shape[0]).max()
            ),
            "baseline_tasks": baseline_tasks,
            "hungarian_tasks": hungarian_tasks,
            "changed_pairs": sum(
                left != right for left, right in zip(baseline_tasks, hungarian_tasks)
            ),
            "baseline_score": assignment["baseline_score"],
            "hungarian_score": assignment["hungarian_score"],
            "Gamma": assignment["Gamma"],
            "gate_passed": assignment["gate_passed"],
            "fallback_reason": assignment["fallback_reason"],
            "probe_seconds": round_record["probe_seconds"],
            "matching_seconds": round_record["matching_seconds"],
            "formal_training_seconds": round_record["formal_training_seconds"],
            "peak_gpu_memory_bytes": round_record["peak_gpu_memory_bytes"],
        }
        per_round.append(round_report)
        score_file.close()
    report["rounds"] = per_round
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

