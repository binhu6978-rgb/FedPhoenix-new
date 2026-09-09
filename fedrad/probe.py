from __future__ import annotations

import copy
import hashlib
import math
import struct
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from fedrad.config import FedRADConfig
from fedrad.models import extract_logits
from fedrad.rng import isolated_python_numpy_rng, isolated_torch_rng
from fedrad.types import (
    ProbeBatch,
    ProbeResult,
    TaskSpec,
    clone_state_dict,
    state_dict_hash,
)


def _hash_materialized_batch(
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


def materialize_probe_batch(
    *,
    dataset: Dataset,
    client_indices: Sequence[int],
    client_id: int,
    round_idx: int,
    probe_seed: int,
    support_limit: int,
    query_limit: int,
    probe_replicate: int = 0,
    support_coverage: str = "fixed",
    query_batches: int = 1,
) -> ProbeBatch:
    indices = tuple(int(index) for index in client_indices)
    if len(indices) < 2:
        raise ValueError(
            f"Client {client_id} needs at least two samples for a disjoint probe"
        )
    support_size = min(int(support_limit), len(indices) // 2)
    query_size = min(int(query_limit), len(indices) - support_size)
    if support_size < 1 or query_size < 1:
        raise RuntimeError("Probe support/query split unexpectedly became empty")

    generator = np.random.default_rng(int(probe_seed))
    permutation = generator.permutation(np.asarray(indices, dtype=np.int64))
    chosen = permutation[: support_size + query_size]
    support_indices = tuple(int(index) for index in chosen[:support_size])
    query_indices = tuple(int(index) for index in chosen[support_size:])
    if set(support_indices).intersection(query_indices):
        raise RuntimeError("Probe support and query indices overlap")

    def materialize(selected: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        images: list[torch.Tensor] = []
        labels: list[int] = []
        # Future stochastic transforms are also isolated from formal training.
        with isolated_python_numpy_rng(probe_seed), isolated_torch_rng(probe_seed):
            for index in selected:
                image, label = dataset[index]
                if not isinstance(image, torch.Tensor):
                    raise TypeError("Probe dataset transforms must produce tensors")
                images.append(image.detach().to("cpu").clone())
                labels.append(int(label))
        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)

    support_images, support_labels = materialize(support_indices)
    query_images, query_labels = materialize(query_indices)
    if support_coverage not in {"fixed", "fresh", "refresh_once"}:
        raise ValueError("unknown support coverage")
    batch_indices = []
    batch_images = []
    batch_labels = []
    batch_hashes = []
    step_ids = ()
    support_hash = _hash_materialized_batch(support_indices, support_images, support_labels)
    if support_coverage != "fixed":
        step_ids = (0, 1, 2, 3, 4) if support_coverage == "fresh" else (0, 0, 0, 1, 1)
        # Preserve legacy batch 1 and query exactly. Traverse the remaining
        # shuffled non-query pool first, cycling only after it is exhausted.
        pool = tuple(int(i) for i in permutation[support_size + query_size:]) + support_indices
        for batch_id in range(max(step_ids) + 1):
            selected = support_indices if batch_id == 0 else tuple(
                pool[((batch_id - 1) * support_size + offset) % len(pool)]
                for offset in range(support_size)
            )
            if len(set(selected)) != support_size or set(selected).intersection(query_indices):
                raise RuntimeError("Invalid multi-batch support/query split")
            images, labels = (support_images, support_labels) if batch_id == 0 else materialize(selected)
            batch_indices.append(selected)
            batch_images.append(images)
            batch_labels.append(labels)
            batch_hashes.append(_hash_materialized_batch(selected, images, labels))
        support_hash = hashlib.sha256(
            repr((tuple(batch_hashes), step_ids)).encode("ascii")
        ).hexdigest()
    query_indices_list, query_images_list, query_labels_list, query_hashes = [], [], [], []
    query_hash = _hash_materialized_batch(query_indices, query_images, query_labels)
    if query_batches not in {1, 2, 4} or (query_batches > 1 and support_coverage != "fixed"):
        raise ValueError("Invalid query coverage configuration")
    if query_batches > 1:
        pool = tuple(int(i) for i in permutation[support_size + query_size:]) + query_indices
        for batch_id in range(query_batches):
            selected = query_indices if batch_id == 0 else tuple(
                pool[((batch_id - 1) * query_size + offset) % len(pool)]
                for offset in range(query_size)
            )
            if set(selected) & set(support_indices) or len(set(selected)) != query_size:
                raise RuntimeError("Invalid query batch")
            images, labels = (query_images, query_labels) if batch_id == 0 else materialize(selected)
            query_indices_list.append(selected)
            query_images_list.append(images)
            query_labels_list.append(labels)
            query_hashes.append(_hash_materialized_batch(selected, images, labels))
        query_hash = hashlib.sha256(repr(tuple(query_hashes)).encode("ascii")).hexdigest()
    return ProbeBatch(
        round_idx=int(round_idx),
        client_id=int(client_id),
        support_indices=support_indices,
        query_indices=query_indices,
        support_images=support_images,
        support_labels=support_labels,
        query_images=query_images,
        query_labels=query_labels,
        support_hash=support_hash,
        query_hash=query_hash,
        probe_seed=int(probe_seed),
        probe_replicate=int(probe_replicate),
        support_batch_indices=tuple(batch_indices),
        support_batch_images=tuple(batch_images),
        support_batch_labels=tuple(batch_labels),
        support_batch_hashes=tuple(batch_hashes),
        support_step_batch_ids=step_ids,
        query_batch_indices=tuple(query_indices_list),
        query_batch_images=tuple(query_images_list),
        query_batch_labels=tuple(query_labels_list),
        query_batch_hashes=tuple(query_hashes),
    )


def _query_loss(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> float:
    with torch.no_grad():
        loss = F.cross_entropy(extract_logits(model(images)), labels)
    value = float(loss.item())
    if not math.isfinite(value):
        raise ValueError("Probe query loss is NaN or Inf")
    return value


def _isolated_query_loss(model, images, labels, *, training, device):
    """Query a private copy; preserve BN state, module modes and RNG exactly."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        query_model = copy.deepcopy(model)
        query_model.train(training)
        return _query_loss(query_model, images, labels)


def _multi_query_loss(model, batch, *, training, device):
    """Keep 32-example BN batches separate; each query starts from the same state."""
    if not batch.query_batch_images:
        raise ValueError("Missing multi-query batches")
    return float(np.mean([
        _isolated_query_loss(model, images.to(device), labels.to(device), training=training, device=device)
        for images, labels in zip(batch.query_batch_images, batch.query_batch_labels, strict=True)
    ]))


def _side_effect_free_training_query_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> float:
    """Measure train-mode query loss without altering later adaptation steps."""
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    with isolated_torch_rng(seed, device):
        value = _query_loss(model, images, labels)
    current_buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, saved in buffers.items():
            current_buffers[name].copy_(saved)
    return value


def _alignment_from_first_backward(
    model: nn.Module,
    task: TaskSpec,
    *,
    norm_eps: float,
    device: torch.device,
) -> tuple[float, float, float, bool]:
    parameters = dict(model.named_parameters())
    delta = task.clone_delta()
    dot = torch.zeros((), dtype=torch.float64, device=device)
    gradient_square = torch.zeros((), dtype=torch.float64, device=device)
    delta_square = torch.zeros((), dtype=torch.float64, device=device)

    for reset_slice in task.omega:
        if reset_slice.parameter_name not in parameters:
            raise ValueError(
                f"Omega references missing parameter {reset_slice.parameter_name}"
            )
        parameter = parameters[reset_slice.parameter_name]
        if parameter.grad is None:
            raise RuntimeError(
                f"No support gradient for {reset_slice.parameter_name}"
            )
        selected = list(reset_slice.output_indices)
        negative_gradient = -parameter.grad.detach()[selected].to(torch.float64)
        reset_delta = delta[reset_slice.parameter_name].to(
            device=device, dtype=torch.float64
        )
        if negative_gradient.shape != reset_delta.shape:
            raise ValueError("Gradient/delta shapes differ on omega")
        dot += torch.sum(negative_gradient * reset_delta)
        gradient_square += torch.sum(negative_gradient * negative_gradient)
        delta_square += torch.sum(reset_delta * reset_delta)

    gradient_norm = math.sqrt(float(gradient_square.item()))
    delta_norm = math.sqrt(float(delta_square.item()))
    if gradient_norm <= norm_eps or delta_norm <= norm_eps:
        return 0.0, gradient_norm, delta_norm, False
    cosine = float(dot.item()) / (gradient_norm * delta_norm)
    if not math.isfinite(cosine):
        raise ValueError("Recovery alignment is NaN or Inf")
    # Roundoff can only push a mathematically valid cosine a few ulps outside.
    cosine = min(1.0, max(-1.0, cosine))
    return cosine, gradient_norm, delta_norm, True


class ProbeRunner:
    def __init__(
        self,
        *,
        config: FedRADConfig,
        model_template: nn.Module,
        device: torch.device,
    ):
        self.config = config
        self.model_template = copy.deepcopy(model_template).to("cpu")
        self.device = device

    def global_reference(
        self,
        *,
        global_state: Mapping[str, torch.Tensor],
        batch: ProbeBatch,
    ) -> float:
        state = clone_state_dict(global_state)
        before = state_dict_hash(state)
        model = copy.deepcopy(self.model_template)
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        images = batch.query_images.to(self.device, non_blocking=True)
        labels = batch.query_labels.to(self.device, non_blocking=True)
        with isolated_python_numpy_rng(batch.probe_seed), isolated_torch_rng(
            batch.probe_seed, self.device
        ):
            loss = (_multi_query_loss(model, batch, training=False, device=self.device)
                    if self.config.probe_query_batches > 1 else _query_loss(model, images, labels))
        if state_dict_hash(state) != before:
            raise RuntimeError("Global reference evaluation mutated global state")
        del model, images, labels
        return loss

    def run_pair(
        self,
        *,
        task: TaskSpec,
        batch: ProbeBatch,
        global_loss: float,
    ) -> ProbeResult:
        task.verify_hash()
        original = task.clone_reset_state()
        if state_dict_hash(original) != task.state_hash:
            raise RuntimeError("Probe did not receive the original TaskSpec state")
        model = copy.deepcopy(self.model_template)
        model.load_state_dict(original, strict=True)
        model.to(self.device)
        # This is deliberately the same training-mode semantics as LocalTrainer's
        # first formal SGD step.  The probe model is private and discarded after
        # scoring, so its BN buffers can never enter either TaskSpec or FedAvg.
        model.train()
        support_images = batch.support_images.to(self.device, non_blocking=True)
        support_labels = batch.support_labels.to(self.device, non_blocking=True)
        query_images = batch.query_images.to(self.device, non_blocking=True)
        query_labels = batch.query_labels.to(self.device, non_blocking=True)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.probe_learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        with isolated_python_numpy_rng(batch.probe_seed), isolated_torch_rng(
            batch.probe_seed, self.device
        ):
            # Keep the historical eval/train pair under legacy. New controls
            # use a private query copy with the same mode before and after SGD.
            model.eval()
            if self.config.probe_query_batches > 1:
                if len(batch.query_batch_images) != self.config.probe_query_batches:
                    raise ValueError("Wrong number of query batches")
                reset_loss = _multi_query_loss(model, batch, training=False, device=self.device)
            elif self.config.probe_query_mode == "legacy":
                reset_loss = _query_loss(model, query_images, query_labels)
            else:
                reset_loss = _isolated_query_loss(
                    model, query_images, query_labels,
                    training=self.config.probe_query_mode == "train_train",
                    device=self.device,
                )
            model.train()
            alignment = 0.0
            gradient_norm = 0.0
            delta_norm = float(task.delta_norm)
            alignment_valid = False
            trajectory_losses: list[float] = []
            for step in range(self.config.probe_steps):
                if self.config.probe_support_coverage != "fixed":
                    if len(batch.support_step_batch_ids) != self.config.probe_steps:
                        raise ValueError("Multi-batch Probe schedule is missing or has wrong length")
                    batch_id = batch.support_step_batch_ids[step]
                    support_images = batch.support_batch_images[batch_id].to(self.device, non_blocking=True)
                    support_labels = batch.support_batch_labels[batch_id].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                support_loss = F.cross_entropy(
                    extract_logits(model(support_images)), support_labels
                )
                if not torch.isfinite(support_loss):
                    raise ValueError("Probe support loss is NaN or Inf")
                support_loss.backward()
                if step == 0:
                    (
                        alignment,
                        gradient_norm,
                        delta_norm,
                        alignment_valid,
                    ) = _alignment_from_first_backward(
                        model,
                        task,
                        norm_eps=self.config.norm_eps,
                        device=self.device,
                    )
                optimizer.step()
                if self.config.probe_recovery_measurement != "terminal":
                    trajectory_losses.append(
                        _side_effect_free_training_query_loss(
                            model,
                            query_images,
                            query_labels,
                            seed=batch.probe_seed + step + 1,
                            device=self.device,
                        )
                    )
            if self.config.probe_recovery_measurement == "terminal":
                if self.config.probe_query_batches > 1:
                    adapted_loss = _multi_query_loss(model, batch, training=True, device=self.device)
                elif self.config.probe_query_mode == "legacy":
                    adapted_loss = _query_loss(model, query_images, query_labels)
                else:
                    adapted_loss = _isolated_query_loss(
                        model, query_images, query_labels,
                        training=self.config.probe_query_mode == "train_train",
                        device=self.device,
                    )
                utility_loss = adapted_loss
            else:
                adapted_loss = trajectory_losses[-1]
                if self.config.probe_recovery_measurement == "trajectory_mean":
                    utility_loss = float(np.mean(trajectory_losses))
                else:
                    utility_loss = 0.5 * (
                        trajectory_losses[0] + trajectory_losses[-1]
                    )

        G = reset_loss - utility_loss
        A = float(global_loss) - adapted_loss
        D = max(-A, 0.0)
        values = (global_loss, reset_loss, adapted_loss, G, A, D, alignment)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("ProbeResult contains NaN or Inf")
        result = ProbeResult(
            round_idx=task.round_idx,
            client_id=batch.client_id,
            task_id=task.task_id,
            task_state_hash=task.state_hash,
            support_hash=batch.support_hash,
            query_hash=batch.query_hash,
            global_loss=float(global_loss),
            reset_loss=reset_loss,
            adapted_loss=adapted_loss,
            G=G,
            A=A,
            D=D,
            C=alignment,
            gradient_norm=gradient_norm,
            delta_norm=delta_norm,
            alignment_valid=alignment_valid,
            probe_seed=batch.probe_seed,
            probe_replicate=batch.probe_replicate,
        )
        del optimizer, model, support_images, support_labels, query_images, query_labels
        task.verify_hash()
        return result
