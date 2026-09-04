from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare(left: Path, right: Path) -> dict[str, object]:
    left_rounds = _jsonl(left / "rounds.jsonl")
    right_rounds = _jsonl(right / "rounds.jsonl")
    deterministic_round_fields = (
        "selected_clients", "assignments", "task_seeds", "task_hashes",
        "local_seeds", "global_state_hash", "diagnostic_accuracy",
        "diagnostic_loss", "mean_client_train_loss", "baseline_assignments",
        "hungarian_assignments", "formal_initial_hashes", "Gamma",
        "baseline_score", "hungarian_score", "gate_tau", "gate_passed",
        "fallback_reason", "component_degenerate",
    )
    round_fields_equal = len(left_rounds) == len(right_rounds) and all(
        all(left_row[field] == right_row[field] for field in deterministic_round_fields)
        for left_row, right_row in zip(left_rounds, right_rounds)
    )

    score_arrays_equal = True
    score_names = (
        "client_order", "task_order", "G", "A", "D", "C", "ZG", "ZA",
        "ZD", "ZC", "Q", "component_names", "component_degenerate",
    )
    for round_number in range(1, len(left_rounds) + 1):
        left_file = np.load(left / "scores" / f"round_{round_number:04d}.npz")
        right_file = np.load(right / "scores" / f"round_{round_number:04d}.npz")
        score_arrays_equal = score_arrays_equal and all(
            np.array_equal(left_file[name], right_file[name]) for name in score_names
        )
        left_file.close()
        right_file.close()

    left_summary = json.loads((left / "summary.json").read_text(encoding="utf-8"))
    right_summary = json.loads((right / "summary.json").read_text(encoding="utf-8"))
    return {
        "round_deterministic_fields_equal": round_fields_equal,
        "task_logs_equal": _jsonl(left / "tasks.jsonl") == _jsonl(right / "tasks.jsonl"),
        "probe_pair_logs_equal": _csv(left / "probe_pairs.csv") == _csv(right / "probe_pairs.csv"),
        "assignment_logs_equal": _jsonl(left / "assignments.jsonl")
        == _jsonl(right / "assignments.jsonl"),
        "score_arrays_bitwise_equal": bool(score_arrays_equal),
        "initial_hash_equal": left_summary["initial_state_hash"]
        == right_summary["initial_state_hash"],
        "final_hash_equal": left_summary["final_state_hash"]
        == right_summary["final_state_hash"],
        "final_hash": right_summary["final_state_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

