from __future__ import annotations

import copy
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedrad.config import FedRADConfig
from fedrad.models import extract_logits
from fedrad.rng import isolated_torch_rng
from fedrad.types import EvaluationResult, clone_state_dict, state_dict_hash


class Evaluator:
    def __init__(
        self,
        *,
        config: FedRADConfig,
        model_template: nn.Module,
        dataset: Dataset,
        device: torch.device,
        evaluation_seed: int,
    ):
        self.config = config
        self.model_template = copy.deepcopy(model_template).to("cpu")
        self.dataset = dataset
        self.device = device
        self.evaluation_seed = int(evaluation_seed)

    def evaluate(self, state: Mapping[str, torch.Tensor]) -> EvaluationResult:
        state_copy = clone_state_dict(state)
        state_hash_before = state_dict_hash(state_copy)
        model = copy.deepcopy(self.model_template)
        model.load_state_dict(state_copy, strict=True)
        model.to(self.device)
        model.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.evaluation_seed)
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            generator=generator,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )

        loss_sum = 0.0
        correct = 0
        count = 0
        with isolated_torch_rng(self.evaluation_seed, self.device), torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = extract_logits(model(images))
                loss_sum += float(
                    F.cross_entropy(logits, labels, reduction="sum").item()
                )
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                count += int(labels.shape[0])

        if state_dict_hash(state_copy) != state_hash_before:
            raise RuntimeError("Evaluation mutated its input state")
        del loader, model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return EvaluationResult(
            loss=loss_sum / max(count, 1),
            accuracy=100.0 * correct / max(count, 1),
            num_examples=count,
        )

