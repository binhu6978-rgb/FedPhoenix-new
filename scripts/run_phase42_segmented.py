from __future__ import annotations

"""Resumable runner for the Phase 4.2 diagnostic on hosts with process limits."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedrad.aggregation import weighted_fedavg
from fedrad.data import load_cifar10
from fedrad.local_trainer import LocalTrainer
from fedrad.models import build_model
from fedrad.rng import RNGStreams
from fedrad.task_bank import build_task_bank
from fedrad.types import clone_state_dict, state_dict_hash
from scripts.run_phase42_utility_fidelity import (
    RANDOM_PERMUTATIONS, REFERENCE_SEED, SNAPSHOTS, _analysis_rows,
    _configure_determinism, _evaluate_snapshot, _jsonl, _load_config,
    _sha256, _verify_task_bank, _write_csv, _write_report, _reference_batch,
    candidate_scores,
)


def _source(source: Path) -> tuple[Any, Any, Any, dict[int, dict[str, Any]], list[dict[str, Any]]]:
    config = _load_config(source)
    _configure_determinism(config.seed)
    rngs = RNGStreams(config.seed)
    data = load_cifar10(config, rngs)
    model = build_model(config, rngs.model_seed)
    rounds = {int(row["round_number"]): row for row in _jsonl(source / "rounds.jsonl")}
    tasks = _jsonl(source / "tasks.jsonl")
    return config, data, model, rounds, tasks


def capture(source: Path, output: Path, start: int, end: int, continuation_in: Path | None, snapshots: set[int]) -> None:
    config, data, model, source_rounds, source_tasks = _source(source)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rngs = RNGStreams(config.seed)
    if continuation_in is None:
        global_state = clone_state_dict(model.state_dict())
    else:
        saved = torch.load(continuation_in, map_location="cpu", weights_only=False)
        if int(saved["next_round"]) != start:
            raise ValueError("continuation next_round mismatch")
        global_state = saved["global_state"]
        rngs._client_rng.set_state(saved["client_rng_state"])
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    validation = output / "replay_validation.jsonl"
    trainer = LocalTrainer(config=config, model_template=model, train_dataset=data.train_dataset, device=device)
    for round_number in range(start, end + 1):
        idx = round_number - 1
        saved = source_rounds[round_number]
        clients = rngs.sample_clients(config.num_users, config.clients_per_round)
        bank = build_task_bank(config=config, model_template=model, global_state=global_state, round_idx=idx, task_count=config.clients_per_round, rngs=rngs)
        _verify_task_bank(bank, [row for row in source_tasks if int(row["round_idx"]) == idx], round_number)
        if list(clients) != saved["selected_clients"] or [task.reset_seed for task in bank.tasks] != saved["task_seeds"] or [task.state_hash for task in bank.tasks] != saved["task_hashes"]:
            raise RuntimeError(f"round {round_number}: source replay metadata mismatch")
        if round_number in snapshots:
            snapshot = work / f"pre_round_{round_number:04d}.pt"
            torch.save({"global_state": clone_state_dict(global_state), "global_hash": state_dict_hash(global_state)}, snapshot)
        updates = []
        local_seeds = []
        for client_id, task in zip(clients, bank.tasks):
            seed = rngs.local_seed(idx, client_id)
            update = trainer.train_client(initial_state=task.clone_reset_state(), client_id=client_id, task_id=task.task_id, client_indices=data.partitions[client_id], round_idx=idx, local_seed=seed)
            if update.initial_state_hash != task.state_hash:
                raise RuntimeError("replay local start mismatch")
            updates.append(update); local_seeds.append(seed)
        if local_seeds != saved["local_seeds"]:
            raise RuntimeError(f"round {round_number}: source local seed mismatch")
        global_state = weighted_fedavg(updates)
        actual = state_dict_hash(global_state)
        if actual != saved["global_state_hash"]:
            raise RuntimeError(f"round {round_number}: source global hash mismatch")
        with validation.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"round": round_number, "global_hash": actual, "all_replay_assertions_passed": True}) + "\n")
    torch.save({"global_state": global_state, "client_rng_state": rngs._client_rng.get_state(), "next_round": end + 1}, work / f"continuation_after_{end:04d}.pt")


def oracle(source: Path, output: Path, round_number: int) -> None:
    config, data, model, source_rounds, source_tasks = _source(source)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    snapshot = torch.load(output / "work" / f"pre_round_{round_number:04d}.pt", map_location="cpu", weights_only=True)
    global_state = snapshot["global_state"]
    idx = round_number - 1
    rngs = RNGStreams(config.seed)
    clients = tuple(int(value) for value in source_rounds[round_number]["selected_clients"])
    bank = build_task_bank(config=config, model_template=model, global_state=global_state, round_idx=idx, task_count=config.clients_per_round, rngs=rngs)
    _verify_task_bank(bank, [row for row in source_tasks if int(row["round_idx"]) == idx], round_number)
    images, labels, weights, reference_meta = _reference_batch(data)
    raw, probe_rows, full_rows, value = _evaluate_snapshot(round_number=round_number, global_state=global_state, bank=bank, clients=clients, data=data, config=config, model_template=model, rngs=rngs, device=device, reference=(images, labels, weights))
    scores = candidate_scores(raw, config)
    correlations, assignments, random_values = _analysis_rows(round_number=round_number, scores=scores, value=value, client_order=clients, task_order=tuple(task.task_id for task in bank.tasks), random_seed=REFERENCE_SEED + round_number)
    matrix_dir = output / "matrices"; matrix_dir.mkdir(exist_ok=True)
    matrix_path = matrix_dir / f"round_{round_number:04d}.npz"
    np.savez_compressed(matrix_path, client_order=np.asarray(clients), task_order=np.asarray([task.task_id for task in bank.tasks]), V_full=value, **raw, **{f"Q_{key}": val for key, val in scores.items()})
    payload = {"round": round_number, "pre_round_global_hash": snapshot["global_hash"], "matrix_file": str(matrix_path.relative_to(output)), "matrix_sha256": _sha256(matrix_path), "reference": reference_meta, "probe_rows": probe_rows, "full_rows": full_rows, "correlations": correlations, "assignments": assignments, "random_values": random_values.tolist()}
    (output / "work" / f"oracle_{round_number:04d}.json").write_text(json.dumps(payload), encoding="utf-8")


def finalize(source: Path, output: Path) -> None:
    payloads = [json.loads((output / "work" / f"oracle_{round_number:04d}.json").read_text(encoding="utf-8")) for round_number in SNAPSHOTS]
    correlations = [row for item in payloads for row in item["correlations"]]
    assignments = [row for item in payloads for row in item["assignments"]]
    probes = [row for item in payloads for row in item["probe_rows"]]
    full = [row for item in payloads for row in item["full_rows"]]
    snapshots = [{"round": item["round"], "pre_round_global_hash": item["pre_round_global_hash"], "clients": 10, "tasks": 10, "full_pairs": len(item["full_rows"]), "reference_samples": item["reference"]["total_reference_samples"], "matrix_file": item["matrix_file"], "matrix_sha256": item["matrix_sha256"]} for item in payloads]
    _write_csv(output / "probe_pairs.csv", probes); _write_csv(output / "full_training_pairs.csv", full)
    _write_csv(output / "correlations.csv", correlations); _write_csv(output / "assignment_oracle.csv", assignments)
    np.savez_compressed(output / "random_assignment_values.npz", **{f"round_{item['round']:04d}": np.asarray(item["random_values"]) for item in payloads})
    summary = {"source_clean_run": str(source), "source_rounds_sha256": _sha256(source / "rounds.jsonl"), "source_tasks_sha256": _sha256(source / "tasks.jsonl"), "snapshots": snapshots, "reference_objective": payloads[0]["reference"], "probe_protocol": "64/32/1-step", "uses_test_set": False, "random_permutations_per_snapshot": RANDOM_PERMUTATIONS, "replay_segmented_for_host_process_limit": True}
    (output / "phase42_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output / "phase42_report.md", correlations, assignments, snapshots)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "oracle", "finalize")); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int); parser.add_argument("--end", type=int); parser.add_argument("--continuation-in", type=Path); parser.add_argument("--snapshot", type=int)
    args = parser.parse_args(); source=args.source.resolve(); output=args.output.resolve()
    if args.mode == "capture":
        if args.start is None or args.end is None: raise ValueError("capture needs --start/--end")
        capture(source, output, args.start, args.end, args.continuation_in.resolve() if args.continuation_in else None, set(SNAPSHOTS).intersection(range(args.start, args.end + 1)))
    elif args.mode == "oracle":
        if args.snapshot not in SNAPSHOTS: raise ValueError("oracle needs a Phase 4.2 snapshot")
        oracle(source, output, args.snapshot)
    else: finalize(source, output)
    return 0

if __name__ == "__main__": raise SystemExit(main())
