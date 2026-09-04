from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedrad.config import FedRADConfig
from fedrad.data import FederatedData, load_cifar10
from fedrad.models import build_model
from fedrad.probe import ProbeRunner
from fedrad.rng import RNGStreams, isolated_python_numpy_rng, isolated_torch_rng
from fedrad.scoring import build_score_matrices
from fedrad.task_bank import TaskBank, build_task_bank
from fedrad.types import ProbeBatch, ProbeResult, state_dict_hash


PROTOCOLS = {
    "P0_32_32": (32, 32),
    "P1_32_64": (32, 64),
    "P2_64_32": (64, 32),
    "P3_64_64": (64, 64),
}


def nested_probe_indices(
    client_indices: Sequence[int], *, probe_seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indices = np.asarray(tuple(int(value) for value in client_indices), dtype=np.int64)
    if len(indices) < 128:
        raise ValueError("nested 64/64 probe requires at least 128 client samples")
    order = np.random.default_rng(int(probe_seed)).permutation(indices)
    return (
        tuple(int(value) for value in order[:64]),
        tuple(int(value) for value in order[64:128]),
    )


def _hash_batch(
    indices: Sequence[int], images: torch.Tensor, labels: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(struct.pack("<q", int(index)))
    for tensor in (images, labels):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _materialize(
    dataset: object, indices: tuple[int, ...], seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    with isolated_python_numpy_rng(seed), isolated_torch_rng(seed):
        for index in indices:
            image, label = dataset[index]  # type: ignore[index]
            if not isinstance(image, torch.Tensor):
                raise TypeError("probe dataset must return Tensor images")
            images.append(image.detach().to("cpu").clone())
            labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def materialize_nested_batches(
    *,
    data: FederatedData,
    client_id: int,
    round_idx: int,
    replicate: int,
    probe_seed: int,
) -> dict[str, ProbeBatch]:
    support_pool, query_pool = nested_probe_indices(
        data.partitions[client_id], probe_seed=probe_seed
    )
    support_images, support_labels = _materialize(
        data.train_dataset, support_pool, probe_seed
    )
    query_images, query_labels = _materialize(
        data.train_dataset, query_pool, probe_seed
    )
    output: dict[str, ProbeBatch] = {}
    for protocol, (support_size, query_size) in PROTOCOLS.items():
        support_indices = support_pool[:support_size]
        query_indices = query_pool[:query_size]
        selected_support_images = support_images[:support_size].clone()
        selected_support_labels = support_labels[:support_size].clone()
        selected_query_images = query_images[:query_size].clone()
        selected_query_labels = query_labels[:query_size].clone()
        output[protocol] = ProbeBatch(
            round_idx=round_idx,
            client_id=client_id,
            support_indices=support_indices,
            query_indices=query_indices,
            support_images=selected_support_images,
            support_labels=selected_support_labels,
            query_images=selected_query_images,
            query_labels=selected_query_labels,
            support_hash=_hash_batch(
                support_indices, selected_support_images, selected_support_labels
            ),
            query_hash=_hash_batch(
                query_indices, selected_query_images, selected_query_labels
            ),
            probe_seed=probe_seed,
            probe_replicate=replicate,
        )
    return output


def _load_config(run_dir: Path) -> FedRADConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    names = {field.name for field in fields(FedRADConfig)}
    kwargs = {name: payload[name] for name in names if name in payload}
    for name in ("data_root", "partition_path", "output_root"):
        kwargs[name] = Path(kwargs[name])
    kwargs["diagnostic_probe_rounds"] = tuple(kwargs["diagnostic_probe_rounds"])
    kwargs["download"] = False
    kwargs["regenerate_partition"] = False
    config = FedRADConfig(**kwargs)
    config.validate()
    return config


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        handle.flush()


def _loss_matrix(
    results: Sequence[ProbeResult],
    clients: Sequence[int],
    tasks: Sequence[int],
    field: str,
) -> np.ndarray:
    by_pair = {
        (int(result.client_id), int(result.task_id)): float(getattr(result, field))
        for result in results
    }
    matrix = np.asarray(
        [[by_pair[(int(client), int(task))] for task in tasks] for client in clients],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError(f"non-finite {field} matrix")
    return matrix


def _save_grid(
    path: Path,
    *,
    scores: object,
    results: Sequence[ProbeResult],
) -> None:
    client_order = tuple(int(value) for value in scores.client_order)  # type: ignore[attr-defined]
    task_order = tuple(int(value) for value in scores.task_order)  # type: ignore[attr-defined]
    np.savez_compressed(
        path,
        client_order=np.asarray(client_order, dtype=np.int64),
        task_order=np.asarray(task_order, dtype=np.int64),
        G=scores.G, A=scores.A, D=scores.D, C=scores.C,  # type: ignore[attr-defined]
        ZG=scores.ZG, ZA=scores.ZA, ZD=scores.ZD, ZC=scores.ZC, Q=scores.Q,  # type: ignore[attr-defined]
        global_loss=_loss_matrix(results, client_order, task_order, "global_loss"),
        reset_loss=_loss_matrix(results, client_order, task_order, "reset_loss"),
        adapted_loss=_loss_matrix(results, client_order, task_order, "adapted_loss"),
    )


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _verify_bank(
    bank: TaskBank, expected_rows: list[dict[str, Any]], round_number: int
) -> None:
    ordered = sorted(expected_rows, key=lambda row: int(row["task_id"]))
    if len(ordered) != len(bank.tasks):
        raise ValueError(f"task count mismatch at round {round_number}")
    for task, expected in zip(bank.tasks, ordered):
        if (
            task.reset_seed != int(expected["reset_seed"])
            or task.state_hash != expected["state_hash"]
            or task.parent_state_hash != expected["parent_state_hash"]
        ):
            raise RuntimeError(f"TaskBank replay mismatch at round {round_number}")


def run(source_run: Path, output_dir: Path, rounds: tuple[int, ...]) -> None:
    config = _load_config(source_run)
    _configure_determinism(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rngs = RNGStreams(config.seed)
    data = load_cifar10(config, rngs)
    model = build_model(config, rngs.model_seed)
    runner = ProbeRunner(config=config, model_template=model, device=device)
    round_rows = {
        int(row["round_number"]): row for row in _jsonl(source_run / "rounds.jsonl")
    }
    task_rows = _jsonl(source_run / "tasks.jsonl")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    metadata_path = output_dir / "probe_grids.jsonl"
    batch_path = output_dir / "nested_batches.jsonl"

    client_counts: list[dict[str, Any]] = []
    for round_number in rounds:
        clients = tuple(int(value) for value in round_rows[round_number]["selected_clients"])
        for client_id in clients:
            client_counts.append(
                {
                    "round": round_number,
                    "client_id": client_id,
                    "samples": len(data.partitions[client_id]),
                    "supports_64_64": len(data.partitions[client_id]) >= 128,
                }
            )
    if not all(row["supports_64_64"] for row in client_counts):
        raise ValueError("at least one selected client cannot support 64/64")
    with (output_dir / "client_sample_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(client_counts[0]))
        writer.writeheader()
        writer.writerows(client_counts)

    provenance = {
        "source_run": str(source_run.resolve()),
        "rounds": list(rounds),
        "protocols": {name: list(value) for name, value in PROTOCOLS.items()},
        "replicates": 3,
        "device": str(device),
        "partition_fingerprint": data.partition_fingerprint,
        "nested_rule": (
            "one deterministic client/round/replicate permutation; support is "
            "prefix[0:32/64], query is disjoint prefix[64:96/128]"
        ),
        "probe_steps": config.probe_steps,
        "probe_learning_rate": config.probe_learning_rate,
        "probe_optimizer": "plain SGD, momentum=0, weight_decay=0",
        "batch_norm_mode": "eval",
        "uses_test_accuracy": False,
        "formal_training_performed": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for round_number in rounds:
        round_idx = round_number - 1
        checkpoint = source_run / "diagnostic_checkpoints" / (
            f"pre_round_{round_number:04d}_global_state.pt"
        )
        global_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_parent = state_dict_hash(global_state)
        bank = build_task_bank(
            config=config,
            model_template=model,
            global_state=global_state,
            round_idx=round_idx,
            task_count=config.clients_per_round,
            rngs=rngs,
        )
        expected_tasks = [
            row for row in task_rows if int(row["round_idx"]) == round_idx
        ]
        _verify_bank(bank, expected_tasks, round_number)
        if bank.parent_state_hash != expected_parent:
            raise RuntimeError("checkpoint hash differs from rebuilt TaskBank parent")
        clients = tuple(
            int(value) for value in round_rows[round_number]["selected_clients"]
        )
        task_ids = tuple(task.task_id for task in bank.tasks)
        task_hashes = tuple(task.state_hash for task in bank.tasks)

        for replicate in range(3):
            batches_by_protocol: dict[str, dict[int, ProbeBatch]] = {
                protocol: {} for protocol in PROTOCOLS
            }
            for client_id in clients:
                probe_seed = rngs.probe_replicate_seed(
                    round_idx, client_id, replicate
                )
                materialized = materialize_nested_batches(
                    data=data,
                    client_id=client_id,
                    round_idx=round_idx,
                    replicate=replicate,
                    probe_seed=probe_seed,
                )
                for protocol, batch in materialized.items():
                    batches_by_protocol[protocol][client_id] = batch
                    _append(
                        batch_path,
                        {
                            "protocol": protocol,
                            "round": round_number,
                            "replicate": replicate,
                            "client_id": client_id,
                            "probe_seed": probe_seed,
                            "support_indices": list(batch.support_indices),
                            "query_indices": list(batch.query_indices),
                            "support_hash": batch.support_hash,
                            "query_hash": batch.query_hash,
                        },
                    )

            for protocol in PROTOCOLS:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                results: list[ProbeResult] = []
                for client_id in clients:
                    batch = batches_by_protocol[protocol][client_id]
                    global_loss = runner.global_reference(
                        global_state=global_state, batch=batch
                    )
                    for task in bank.tasks:
                        results.append(
                            runner.run_pair(
                                task=task,
                                batch=batch,
                                global_loss=global_loss,
                            )
                        )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                )
                if state_dict_hash(global_state) != expected_parent:
                    raise RuntimeError("targeted probe mutated global state")
                if tuple(task.state_hash for task in bank.tasks) != task_hashes:
                    raise RuntimeError("targeted probe mutated TaskBank")
                scores = build_score_matrices(
                    selected_clients=clients,
                    task_ids=task_ids,
                    results=results,
                    config=config,
                )
                output_path = raw_dir / (
                    f"{protocol}_round_{round_number:04d}_rep_{replicate}.npz"
                )
                _save_grid(output_path, scores=scores, results=results)
                _append(
                    metadata_path,
                    {
                        "protocol": protocol,
                        "round": round_number,
                        "replicate": replicate,
                        "support_size": PROTOCOLS[protocol][0],
                        "query_size": PROTOCOLS[protocol][1],
                        "probe_seconds": seconds,
                        "peak_gpu_memory_bytes": peak_memory,
                        "global_state_hash": expected_parent,
                        "task_hashes": list(task_hashes),
                        "file": str(output_path.resolve()),
                    },
                )
                print(
                    f"PROBE_SIZE protocol={protocol} round={round_number} "
                    f"replicate={replicate} seconds={seconds:.3f}",
                    flush=True,
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", default="1,20,40")
    args = parser.parse_args()
    rounds = tuple(int(value.strip()) for value in args.rounds.split(","))
    run(args.source_run.resolve(), args.output_dir.resolve(), rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
