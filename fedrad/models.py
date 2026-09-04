from __future__ import annotations

import torch
from torch import nn

from fedrad.config import FedRADConfig
from fedrad.rng import isolated_torch_rng
from models.resnetcifar import ResNet18_cifar10


def build_model(config: FedRADConfig, initialization_seed: int | None = None) -> nn.Module:
    if config.dataset != "cifar10" or config.model != "resnet18":
        raise ValueError("Phase 1 only supports CIFAR-10 with ResNet18")
    if initialization_seed is None:
        return ResNet18_cifar10(num_classes=config.num_classes)
    with isolated_torch_rng(initialization_seed):
        return ResNet18_cifar10(num_classes=config.num_classes)


def extract_logits(output: object) -> torch.Tensor:
    if not isinstance(output, dict) or "output" not in output:
        raise TypeError("FedRAD models must return a dict containing 'output'")
    logits = output["output"]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output['output'] must be a Tensor")
    return logits

