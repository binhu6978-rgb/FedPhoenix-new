from __future__ import annotations

"""Strict two-replicate conditional-utility measurements on one frozen 2/64 run.

This is an offline diagnostic.  It never updates the completed FL trajectory,
and its only train calls are the 10 x 10 formal LocalTrainer evaluations for a
requested snapshot and replicate.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedrad.config import FedRADConfig
from fedrad.data import load_cifar10
from fedrad.local_trainer import LocalTrainer
from fedrad.models import build_model
from fedrad.rng import RNGStreams
from fedrad.task_bank import build_task_bank
from fedrad.aggregation import weighted_fedavg
from fedrad.types import clone_state_dict, state_dict_hash
from scripts.run_phase42_utility_fidelity import (
    _configure_determinism,
    _jsonl,
    _reference_batch,
    _sha256,
    _verify_task_bank,
    federation_reference_loss,
)


PRIMARY_SNAPSHOTS = (20, 160)
REPLAY_ONLY_SNAPSHOTS = (40, 120)
REFERENCE_DESCRIPTION = (
    "V_full(i,j) = L_fed_ref(reset_j) - L_fed_ref(w_full(i,j)); "
    "L_fed_ref is sample-size-weighted stratified empirical cross entropy on "
    "a deterministic 16 training examples from every one of 100 clients "
    "(1600 examples total), evaluated in model.eval() mode; no test samples."
)
SEED_NAMESPACE = 700_000
SEED_REPLICATE_STRIDE = 15_485_863


def full_local_seed(base_seed: int, round_number: int, client_id: int, replicate_id: int) -> int:
    """Seed differs only by replicate for a fixed round/client (not TaskSpec)."""
    if replicate_id < 1:
        raise ValueError("replicate_id must be positive")
    return int((int(base_seed) + SEED_NAMESPACE + int(round_number) * 100_003
                + int(client_id) * 997 + int(replicate_id) * SEED_REPLICATE_STRIDE) % (2**32 - 1))


def _partition_hash(indices: list[int]) -> str:
    payload = json.dumps([int(value) for value in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_config(source: Path) -> FedRADConfig:
    payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(FedRADConfig)}
    kwargs = {key: payload[key] for key in allowed if key in payload}
    for key in ("data_root", "partition_path", "output_root", "warmup_reference_rounds_path"):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = Path(kwargs[key])
    kwargs["diagnostic_probe_rounds"] = tuple(kwargs.get("diagnostic_probe_rounds", ()))
    kwargs["download"] = False
    kwargs["regenerate_partition"] = False
    config = FedRADConfig(**kwargs)
    config.validate()
    return config


def _source(source: Path) -> tuple[FedRADConfig, Any, torch.nn.Module, dict[int, dict[str, Any]], list[dict[str, Any]]]:
    config = _load_config(source)
    _configure_determinism(config.seed)
    rngs = RNGStreams(config.seed)
    data = load_cifar10(config, rngs)
    model = build_model(config, rngs.model_seed)
    rounds = {int(row["round_number"]): row for row in _jsonl(source / "rounds.jsonl")}
    return config, data, model, rounds, _jsonl(source / "tasks.jsonl")


def _snapshot_path(source: Path, round_number: int) -> Path:
    return source / "diagnostic_checkpoints" / f"pre_round_{round_number:04d}_global_state.pt"


def _task_rows(source_tasks: list[dict[str, Any]], round_number: int) -> list[dict[str, Any]]:
    rows = [row for row in source_tasks if int(row["round_idx"]) == round_number - 1]
    return sorted(rows, key=lambda row: int(row["task_id"]))


def _formal_hyperparameters(config: FedRADConfig) -> dict[str, Any]:
    payload = config.as_serializable_dict()
    return {key: payload[key] for key in (
        "dataset", "model", "num_users", "clients_per_round", "dirichlet_beta", "seed",
        "local_epochs", "local_batch_size", "learning_rate", "momentum", "weight_decay",
        "num_workers", "reset_ratio", "reset_method", "fp_conv_rounds", "eval_batch_size",
        "partition_path",
    )}


def _load_snapshot(path: Path, expected_parent_hash: str) -> Mapping[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{path} is not a raw state dictionary")
    observed = state_dict_hash(state)
    if observed != expected_parent_hash:
        raise RuntimeError(f"checkpoint state hash mismatch: {observed} != {expected_parent_hash}")
    return state


def _resolved_snapshot(output: Path, source: Path, item: Mapping[str, Any]) -> tuple[Mapping[str, torch.Tensor], Path, str]:
    """Use the frozen direct file when present, otherwise a hash-checked replay file."""
    round_number = int(item["round"])
    expected = str(item["expected_pre_round_global_hash"])
    direct = _snapshot_path(source, round_number)
    if bool(item["direct_checkpoint_available"]):
        state = _load_snapshot(direct, expected)
        return state, direct, "direct"
    replay = output / "work" / f"pre_round_{round_number:04d}.pt"
    if not replay.is_file():
        raise RuntimeError("deferred snapshot requires the documented hash-exact replay artifact")
    payload = torch.load(replay, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "global_state" not in payload or "global_hash" not in payload:
        raise RuntimeError("replay snapshot has an invalid format")
    if str(payload["global_hash"]) != expected or state_dict_hash(payload["global_state"]) != expected:
        raise RuntimeError("replay snapshot global hash differs from frozen trajectory")
    return payload["global_state"], replay, "hash_exact_replay"


def initialize(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config, data, model, source_rounds, source_tasks = _source(source)
    del data, model
    if config.reset_ratio != 2.0 / 64.0:
        raise RuntimeError(f"faithful run must use reset_ratio=2/64, got {config.reset_ratio}")
    if (config.seed, config.rounds, config.clients_per_round, config.dirichlet_beta) != (1, 200, 10, 0.3):
        raise RuntimeError("source run does not match frozen faithful trajectory settings")
    config_path = source / "config.json"
    provenance_rounds: list[dict[str, Any]] = []
    for round_number in PRIMARY_SNAPSHOTS + REPLAY_ONLY_SNAPSHOTS:
        round_row = source_rounds[round_number]
        tasks = _task_rows(source_tasks, round_number)
        if len(tasks) != 10 or len(round_row["selected_clients"]) != 10:
            raise RuntimeError(f"round {round_number}: source is not a complete 10x10 snapshot")
        expected_parent = str(tasks[0]["parent_state_hash"])
        if any(str(row["parent_state_hash"]) != expected_parent for row in tasks):
            raise RuntimeError(f"round {round_number}: inconsistent task parents")
        checkpoint = _snapshot_path(source, round_number)
        item: dict[str, Any] = {
            "round": round_number,
            "selected_clients": [int(value) for value in round_row["selected_clients"]],
            "task_ids": [int(row["task_id"]) for row in tasks],
            "task_seeds": [int(row["reset_seed"]) for row in tasks],
            "task_state_hashes": [str(row["state_hash"]) for row in tasks],
            "expected_pre_round_global_hash": expected_parent,
            "expected_post_round_global_hash": str(round_row["global_state_hash"]),
            "direct_checkpoint_available": checkpoint.is_file(),
            "checkpoint_path": str(checkpoint.resolve()) if checkpoint.is_file() else None,
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
            "checkpoint_global_state_hash": None,
            "hash_exact_replay_capability": True,
            "replay_source": "same frozen rounds.jsonl/tasks.jsonl/config and deterministic RNGStreams/TaskBank/LocalTrainer/FedAvg",
        }
        if checkpoint.is_file():
            item["checkpoint_global_state_hash"] = state_dict_hash(
                _load_snapshot(checkpoint, expected_parent)
            )
        provenance_rounds.append(item)
    output.mkdir(parents=True)
    manifest = {
        "experiment": "faithful FedPhoenix 2/64 conditional utility validation",
        "source_clean_run": str(source.resolve()),
        "source_run_id": source.name,
        "source_config_sha256": _sha256(config_path),
        "source_rounds_sha256": _sha256(source / "rounds.jsonl"),
        "source_tasks_sha256": _sha256(source / "tasks.jsonl"),
        "frozen_trajectory": {"reset_ratio": config.reset_ratio, "seed": config.seed, "rounds": config.rounds,
                              "clients_per_round": config.clients_per_round, "beta": config.dirichlet_beta,
                              **_formal_hyperparameters(config)},
        "primary_utility": {"formula": "V_full(i,j) = L_fed_ref(reset_j) - L_fed_ref(w_full(i,j))",
                            "description": REFERENCE_DESCRIPTION, "uses_test_set": False,
                            "only_difference_between_replicates": "formal local-training stochastic seed"},
        "replicate_seed_formula": "(base_seed + 700000 + round*100003 + client_id*997 + replicate_id*15485863) mod (2**32-1); no task-position term",
        "primary_snapshots": list(PRIMARY_SNAPSHOTS),
        "deferred_snapshots": list(REPLAY_ONLY_SNAPSHOTS),
        "snapshots": provenance_rounds,
        "old_1over64_results_used": False,
        "runs_new_fl_trajectory": False,
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps(_formal_hyperparameters(config), indent=2) + "\n", encoding="utf-8")


def measure(source: Path, output: Path, round_number: int, replicate_id: int) -> None:
    if round_number not in PRIMARY_SNAPSHOTS + REPLAY_ONLY_SNAPSHOTS:
        raise ValueError("unapproved snapshot")
    if replicate_id not in (1, 2):
        raise ValueError("replicate must be 1 or 2")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    item = next(row for row in provenance["snapshots"] if int(row["round"]) == round_number)
    config, data, model, source_rounds, source_tasks = _source(source)
    _configure_determinism(config.seed)
    if str(source.resolve()) != provenance["source_clean_run"] or _sha256(source / "rounds.jsonl") != provenance["source_rounds_sha256"]:
        raise RuntimeError("source trajectory provenance changed")
    global_state, checkpoint, checkpoint_kind = _resolved_snapshot(output, source, item)
    index = round_number - 1
    clients = tuple(int(value) for value in source_rounds[round_number]["selected_clients"])
    bank = build_task_bank(config=config, model_template=model, global_state=global_state, round_idx=index,
                          task_count=config.clients_per_round, rngs=RNGStreams(config.seed))
    expected = _task_rows(source_tasks, round_number)
    _verify_task_bank(bank, expected, round_number)
    if tuple(task.state_hash for task in bank.tasks) != tuple(item["task_state_hashes"]):
        raise RuntimeError("TaskSpec replay hash mismatch")
    if tuple(clients) != tuple(item["selected_clients"]):
        raise RuntimeError("client order mismatch")
    images, labels, weights, reference_meta = _reference_batch(data)
    reference = (images, labels, weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reset_losses = np.asarray([
        federation_reference_loss(state=task.clone_reset_state(), model_template=model, reference=reference,
                                  device=device, batch_size=config.eval_batch_size)
        for task in bank.tasks
    ], dtype=np.float64)
    trainer = LocalTrainer(config=config, model_template=model, train_dataset=data.train_dataset, device=device)
    matrix = np.empty((10, 10), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for row_index, client_id in enumerate(clients):
        seed = full_local_seed(config.seed, round_number, client_id, replicate_id)
        other_seed = full_local_seed(config.seed, round_number, client_id, 3 - replicate_id)
        if seed == other_seed:
            raise RuntimeError("replicate seeds collided")
        partition_hash = _partition_hash(data.partitions[client_id])
        for column, task in enumerate(bank.tasks):
            update = trainer.train_client(initial_state=task.clone_reset_state(), client_id=client_id, task_id=task.task_id,
                                          client_indices=data.partitions[client_id], round_idx=index, local_seed=seed)
            if update.initial_state_hash != task.state_hash or update.local_seed != seed:
                raise RuntimeError("formal local input invariant failed")
            after = federation_reference_loss(state=update.clone_state(), model_template=model, reference=reference,
                                              device=device, batch_size=config.eval_batch_size)
            matrix[row_index, column] = float(reset_losses[column] - after)
            records.append({"round": round_number, "replicate": replicate_id, "client_id": client_id,
                            "task_id": task.task_id, "task_seed": task.reset_seed, "task_state_hash": task.state_hash,
                            "global_hash": state_dict_hash(global_state), "partition_hash": partition_hash,
                            "local_seed": seed, "other_replicate_seed": other_seed,
                            "reset_reference_loss": float(reset_losses[column]), "full_reference_loss": after,
                            "V_full": float(matrix[row_index, column]), "full_state_hash": update.state_hash,
                            "all_input_assertions_passed": True})
    result_dir = output / "matrices"
    result_dir.mkdir(exist_ok=True)
    np.savez_compressed(result_dir / f"round_{round_number:04d}_V{replicate_id}.npz", V_full=matrix,
                        client_order=np.asarray(clients), task_order=np.asarray([task.task_id for task in bank.tasks]),
                        reset_seeds=np.asarray([task.reset_seed for task in bank.tasks]))
    with (result_dir / f"round_{round_number:04d}_V{replicate_id}_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (result_dir / f"round_{round_number:04d}_V{replicate_id}_metadata.json").write_text(
        json.dumps({"round": round_number, "replicate": replicate_id, "reference": reference_meta,
                    "global_hash": state_dict_hash(global_state), "checkpoint_sha256": _sha256(checkpoint), "checkpoint_kind": checkpoint_kind,
                    "formal_hyperparameters": _formal_hyperparameters(config), "pairs": len(records)}, indent=2) + "\n",
        encoding="utf-8")


def replay_deferred(source: Path, output: Path) -> None:
    """Replay rounds 20..120 from the direct r20 checkpoint and assert every hash."""
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    config, data, model, source_rounds, source_tasks = _source(source)
    item20 = next(row for row in provenance["snapshots"] if int(row["round"]) == 20)
    global_state = _load_snapshot(_snapshot_path(source, 20), str(item20["expected_pre_round_global_hash"]))
    rngs = RNGStreams(config.seed)
    # Client sampling is stateful, unlike TaskSpec seeds.  Advance exactly through
    # rounds 1..19 so the direct pre-r20 checkpoint and client stream align.
    for _ in range(19):
        rngs.sample_clients(config.num_users, config.clients_per_round)
    work = output / "work"; work.mkdir(exist_ok=True)
    validation_path = output / "replay_verification.jsonl"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = LocalTrainer(config=config, model_template=model, train_dataset=data.train_dataset, device=device)
    for round_number in range(20, 121):
        index = round_number - 1
        source_round = source_rounds[round_number]
        clients = rngs.sample_clients(config.num_users, config.clients_per_round)
        bank = build_task_bank(config=config, model_template=model, global_state=global_state, round_idx=index,
                              task_count=config.clients_per_round, rngs=rngs)
        _verify_task_bank(bank, _task_rows(source_tasks, round_number), round_number)
        if list(clients) != source_round["selected_clients"] or [task.state_hash for task in bank.tasks] != source_round["task_hashes"]:
            raise RuntimeError(f"round {round_number}: replay metadata mismatch")
        if round_number in REPLAY_ONLY_SNAPSHOTS:
            torch.save({"global_state": clone_state_dict(global_state), "global_hash": state_dict_hash(global_state)}, work / f"pre_round_{round_number:04d}.pt")
        updates = []
        for client_id, task in zip(clients, bank.tasks):
            seed = rngs.local_seed(index, client_id)
            update = trainer.train_client(initial_state=task.clone_reset_state(), client_id=client_id, task_id=task.task_id,
                                          client_indices=data.partitions[client_id], round_idx=index, local_seed=seed)
            if update.initial_state_hash != task.state_hash:
                raise RuntimeError(f"round {round_number}: replay local start mismatch")
            updates.append(update)
        global_state = weighted_fedavg(updates)
        actual = state_dict_hash(global_state)
        if actual != source_round["global_state_hash"]:
            raise RuntimeError(f"round {round_number}: replay global hash mismatch: {actual}")
        with validation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"round": round_number, "global_hash": actual, "all_replay_assertions_passed": True}) + "\n")
    for item in provenance["snapshots"]:
        if int(item["round"]) in REPLAY_ONLY_SNAPSHOTS:
            replay = work / f"pre_round_{int(item['round']):04d}.pt"
            payload = torch.load(replay, map_location="cpu", weights_only=True)
            item["replay_checkpoint_path"] = str(replay.resolve())
            item["replay_checkpoint_sha256"] = _sha256(replay)
            item["replay_checkpoint_global_state_hash"] = str(payload["global_hash"])
            item["replay_hash_verified"] = True
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initialize", "measure", "replay-deferred"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args.source.resolve(), args.output.resolve())
    elif args.mode == "measure":
        if args.round is None or args.replicate is None:
            raise ValueError("measure requires --round and --replicate")
        measure(args.source.resolve(), args.output.resolve(), args.round, args.replicate)
    else:
        replay_deferred(args.source.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
