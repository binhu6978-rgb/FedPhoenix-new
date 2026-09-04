from __future__ import annotations

import copy
import random
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from fedrad.config import FedRADConfig
from fedrad.models import extract_logits
from fedrad.rng import isolated_python_numpy_rng, isolated_torch_rng
from fedrad.types import (
    ClientUpdate,
    assert_finite_state,
    clone_state_dict,
    state_dict_hash,
)


def _seed_worker(_: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32 - 1))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class LocalTrainer:
    def __init__(
        self,
        *,
        config: FedRADConfig,
        model_template: nn.Module,
        train_dataset: Dataset,
        device: torch.device,
    ):
        self.config = config
        self.model_template = copy.deepcopy(model_template).to("cpu")
        self.train_dataset = train_dataset
        self.device = device

    def train_client(
        self,
        *,
        initial_state: Mapping[str, torch.Tensor],
        client_id: int,
        task_id: int,
        client_indices: Sequence[int],
        round_idx: int,
        local_seed: int,
    ) -> ClientUpdate:
        del round_idx  # Encoded in local_seed by RNGStreams; kept in the interface.
        if not client_indices:
            raise ValueError(f"Client {client_id} has no training samples")

        initial_copy = clone_state_dict(initial_state)
        initial_hash = state_dict_hash(initial_copy)
        model = copy.deepcopy(self.model_template)
        model.load_state_dict(initial_copy, strict=True)
        model.to(self.device)
        model.train()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(local_seed))
        loader = DataLoader(
            Subset(self.train_dataset, [int(index) for index in client_indices]),
            batch_size=self.config.local_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            generator=generator,
            worker_init_fn=_seed_worker if self.config.num_workers > 0 else None,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        loss_sum = 0.0
        example_visits = 0
        with isolated_python_numpy_rng(local_seed), isolated_torch_rng(
            local_seed, self.device
        ):
            for _ in range(self.config.local_epochs):
                for images, labels in loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = extract_logits(model(images))
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    batch_size = int(labels.shape[0])
                    loss_sum += float(loss.item()) * batch_size
                    example_visits += batch_size

        state = clone_state_dict(model.state_dict())
        assert_finite_state(state, f"client_{client_id}_update")
        update_hash = state_dict_hash(state)
        del optimizer, loader, model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return ClientUpdate(
            client_id=int(client_id),
            task_id=int(task_id),
            initial_state_hash=initial_hash,
            num_examples=len(client_indices),
            mean_train_loss=loss_sum / max(example_visits, 1),
            local_seed=int(local_seed),
            state_hash=update_hash,
            _state=state,
        )

