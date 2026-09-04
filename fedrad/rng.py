from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Iterator

import numpy as np
import torch


_UINT32_MODULUS = 2**32 - 1


def _bounded_seed(value: int) -> int:
    return int(value % _UINT32_MODULUS)


class RNGStreams:
    """Independent deterministic random streams for every FedRAD concern."""

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)
        # Preserve the historical first/next client draws without sharing the
        # stream with partition generation, task reset, or local training.
        self._client_rng = np.random.RandomState(_bounded_seed(self.base_seed))

    @property
    def model_seed(self) -> int:
        return _bounded_seed(self.base_seed)

    @property
    def evaluation_seed(self) -> int:
        return _bounded_seed(self.base_seed + 600_000)

    def partition_rng(self) -> np.random.Generator:
        return np.random.default_rng(_bounded_seed(self.base_seed + 100_000))

    def sample_clients(self, num_users: int, count: int) -> tuple[int, ...]:
        selected = self._client_rng.choice(num_users, count, replace=False)
        return tuple(int(item) for item in selected.tolist())

    def task_seed(self, round_idx: int, task_id: int) -> int:
        return _bounded_seed(
            self.base_seed + 200_000 + int(round_idx) * 100_003 + int(task_id) * 997
        )

    def local_seed(self, round_idx: int, client_id: int) -> int:
        return _bounded_seed(
            self.base_seed + 300_000 + int(round_idx) * 100_003 + int(client_id) * 997
        )

    def probe_seed(self, round_idx: int, client_id: int) -> int:
        # Deliberately independent of task_id and assignment position so one
        # materialized ProbeBatch can be reused for every task.
        return _bounded_seed(
            self.base_seed + 400_000 + int(round_idx) * 100_003 + int(client_id) * 997
        )

    def probe_replicate_seed(
        self, round_idx: int, client_id: int, replicate: int
    ) -> int:
        if replicate < 0:
            raise ValueError("probe replicate must be non-negative")
        if replicate == 0:
            return self.probe_seed(round_idx, client_id)
        return _bounded_seed(
            self.base_seed
            + 500_000
            + int(round_idx) * 100_003
            + int(client_id) * 997
            + int(replicate) * 15_485_863
        )


@contextmanager
def isolated_torch_rng(seed: int, device: torch.device | None = None) -> Iterator[None]:
    devices: list[int] = []
    if device is not None and device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        devices = [int(index)]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        yield


@contextmanager
def isolated_python_numpy_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(int(seed))
        np.random.seed(_bounded_seed(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
