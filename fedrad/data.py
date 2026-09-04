from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from fedrad.config import FedRADConfig
from fedrad.rng import RNGStreams


@dataclass(frozen=True)
class FederatedData:
    train_dataset: Dataset
    test_dataset: Dataset
    partitions: Mapping[int, tuple[int, ...]]
    partition_fingerprint: str
    partition_path: Path


def partition_fingerprint(partitions: Mapping[int, Sequence[int]]) -> str:
    """Hash partition membership independently of JSON/key/list ordering."""
    digest = hashlib.sha256()
    for client_id in sorted(partitions):
        digest.update(struct.pack("<q", int(client_id)))
        indices = sorted(int(index) for index in partitions[client_id])
        digest.update(struct.pack("<q", len(indices)))
        for index in indices:
            digest.update(struct.pack("<q", index))
    return digest.hexdigest()


def validate_partition(
    partitions: Mapping[int, Sequence[int]],
    *,
    num_clients: int,
    dataset_size: int,
    min_client_samples: int,
    require_full_coverage: bool = True,
) -> dict[int, tuple[int, ...]]:
    expected_clients = set(range(num_clients))
    actual_clients = {int(client_id) for client_id in partitions}
    if actual_clients != expected_clients:
        missing = sorted(expected_clients - actual_clients)
        extra = sorted(actual_clients - expected_clients)
        raise ValueError(f"Invalid partition clients; missing={missing}, extra={extra}")

    normalized: dict[int, tuple[int, ...]] = {}
    all_seen: set[int] = set()
    for client_id in range(num_clients):
        indices = tuple(int(index) for index in partitions[client_id])
        if len(indices) < min_client_samples:
            raise ValueError(
                f"Client {client_id} has {len(indices)} samples; "
                f"minimum is {min_client_samples}"
            )
        local_seen = set(indices)
        if len(local_seen) != len(indices):
            raise ValueError(f"Client {client_id} contains duplicate indices")
        invalid = [index for index in indices if not 0 <= index < dataset_size]
        if invalid:
            raise ValueError(
                f"Client {client_id} has out-of-range indices: {invalid[:5]}"
            )
        overlap = all_seen.intersection(local_seen)
        if overlap:
            raise ValueError(
                f"Client {client_id} overlaps previous clients: {sorted(overlap)[:5]}"
            )
        all_seen.update(local_seen)
        normalized[client_id] = indices

    if require_full_coverage and len(all_seen) != dataset_size:
        raise ValueError(
            f"Partition covers {len(all_seen)} of {dataset_size} training samples"
        )
    return normalized


def generate_dirichlet_partition(
    labels: Sequence[int] | np.ndarray,
    *,
    num_clients: int,
    beta: float,
    min_client_samples: int,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> dict[int, tuple[int, ...]]:
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if beta <= 0:
        raise ValueError("beta must be positive")

    classes = np.unique(labels_array)
    target_size = len(labels_array) / num_clients
    for _ in range(max_attempts):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        for class_id in classes:
            class_indices = np.flatnonzero(labels_array == class_id)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, beta, dtype=np.float64))
            capacity = np.asarray(
                [len(bucket) < target_size for bucket in buckets], dtype=np.float64
            )
            proportions *= capacity
            if proportions.sum() <= 0:
                proportions.fill(1.0 / num_clients)
            else:
                proportions /= proportions.sum()
            split_points = (
                np.cumsum(proportions) * len(class_indices)
            ).astype(np.int64)[:-1]
            for client_id, split in enumerate(np.split(class_indices, split_points)):
                buckets[client_id].extend(int(index) for index in split.tolist())

        if min(len(bucket) for bucket in buckets) >= min_client_samples:
            result: dict[int, tuple[int, ...]] = {}
            for client_id, bucket in enumerate(buckets):
                rng.shuffle(bucket)
                result[client_id] = tuple(bucket)
            return result

    raise RuntimeError(
        "Unable to generate a valid Dirichlet partition in "
        f"{max_attempts} attempts"
    )


def _read_partition(path: Path) -> tuple[dict[int, Sequence[int]], dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("train_data", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Partition file {path} has no train_data mapping")
    partitions = {int(key): value for key, value in raw.items()}
    return partitions, payload


def _write_partition(
    path: Path,
    partitions: Mapping[int, Sequence[int]],
    config: FedRADConfig,
    fingerprint: str,
    partition_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "dataset": "cifar10",
        "num_users": config.num_users,
        "iid": 0,
        "noniid_case": 5,
        "data_beta": config.dirichlet_beta,
        "min_client_samples": config.min_client_samples,
        "partition_seed": int(partition_seed),
        "fingerprint": fingerprint,
        "train_data": {
            str(client_id): [int(index) for index in partitions[client_id]]
            for client_id in sorted(partitions)
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_cifar10(config: FedRADConfig, rngs: RNGStreams) -> FederatedData:
    if config.dataset != "cifar10":
        raise ValueError("Phase 1 data loader only supports CIFAR-10")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    data_root = Path(config.data_root)
    legacy_root = data_root / "cifar10"
    dataset_root = (
        legacy_root
        if (legacy_root / "cifar-10-batches-py").is_dir()
        else data_root
    )
    train_dataset = datasets.CIFAR10(
        str(dataset_root), train=True, download=config.download, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        str(dataset_root), train=False, download=config.download, transform=transform
    )

    path = Path(config.partition_path)
    if path.exists() and not config.regenerate_partition:
        raw_partitions, metadata = _read_partition(path)
        if "num_users" in metadata and int(metadata["num_users"]) != config.num_users:
            raise ValueError("Partition metadata num_users does not match config")
        if "data_beta" in metadata and not np.isclose(
            float(metadata["data_beta"]), config.dirichlet_beta
        ):
            raise ValueError("Partition metadata beta does not match config")
    else:
        raw_partitions = generate_dirichlet_partition(
            train_dataset.targets,
            num_clients=config.num_users,
            beta=config.dirichlet_beta,
            min_client_samples=config.min_client_samples,
            rng=rngs.partition_rng(),
        )

    partitions = validate_partition(
        raw_partitions,
        num_clients=config.num_users,
        dataset_size=len(train_dataset),
        min_client_samples=config.min_client_samples,
    )
    fingerprint = partition_fingerprint(partitions)

    if not path.exists() or config.regenerate_partition:
        _write_partition(
            path,
            partitions,
            config,
            fingerprint,
            rngs.base_seed + 100_000,
        )
    elif isinstance(metadata, dict) and metadata.get("fingerprint") not in {
        None,
        fingerprint,
    }:
        raise ValueError("Stored partition fingerprint does not match its contents")

    return FederatedData(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        partitions=partitions,
        partition_fingerprint=fingerprint,
        partition_path=path.resolve(),
    )

