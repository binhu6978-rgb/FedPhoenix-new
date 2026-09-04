from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from fedrad.config import FedRADConfig
from fedrad.data import FederatedData, partition_fingerprint


class TinyCIFAR(Dataset):
    def __init__(self, size: int = 40, seed: int = 123):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(size, 3, 8, 8, generator=generator)
        self.targets = [index % 10 for index in range(size)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index].clone(), self.targets[index]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, inputs):
        features = self.pool(self.relu(self.bn(self.conv(inputs))))
        representation = torch.flatten(features, 1)
        logits = self.fc(representation)
        return {"representation": representation, "output": logits}


def tiny_config(**overrides) -> FedRADConfig:
    config = replace(
        FedRADConfig(),
        num_users=4,
        clients_per_round=2,
        rounds=2,
        min_client_samples=2,
        local_epochs=1,
        local_batch_size=5,
        eval_batch_size=10,
        num_workers=0,
        device="cpu",
        data_root=Path("unused"),
        partition_path=Path("unused.json"),
        output_root=Path("unused-results"),
    )
    return replace(config, **overrides)


def tiny_data() -> FederatedData:
    train = TinyCIFAR(size=40, seed=123)
    test = TinyCIFAR(size=20, seed=456)
    partitions = {
        client_id: tuple(range(client_id * 10, (client_id + 1) * 10))
        for client_id in range(4)
    }
    return FederatedData(
        train_dataset=train,
        test_dataset=test,
        partitions=partitions,
        partition_fingerprint=partition_fingerprint(partitions),
        partition_path=Path("synthetic-partition.json"),
    )


def initialized_tiny_model(seed: int = 1) -> TinyModel:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return TinyModel()

