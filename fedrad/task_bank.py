from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from Algorithm.Phoenix_util import reset_kernels_for_task
from fedrad.config import FedRADConfig
from fedrad.rng import RNGStreams
from fedrad.types import (
    ResetSlice,
    StateDict,
    TaskSpec,
    assert_finite_state,
    clone_state_dict,
    state_dict_hash,
)


def _parameter_name(layer_name: str) -> str:
    return f"{layer_name}.weight" if layer_name else "weight"


def _omega_from_trace(
    trace: Mapping[str, object], state: Mapping[str, torch.Tensor]
) -> tuple[ResetSlice, ...]:
    omega: list[ResetSlice] = []
    layers = trace.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("reset trace layers must be a list")
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("reset trace layer entries must be mappings")
        parameter_name = _parameter_name(str(layer["name"]))
        if parameter_name not in state:
            raise ValueError(f"Trace references missing parameter {parameter_name}")
        indices = tuple(int(index) for index in layer.get("reset_indices", []))
        if len(set(indices)) != len(indices):
            raise ValueError(f"Trace repeats reset indices for {parameter_name}")
        tensor = state[parameter_name]
        if any(index < 0 or index >= tensor.shape[0] for index in indices):
            raise ValueError(f"Trace has invalid reset index for {parameter_name}")
        if indices:
            omega.append(
                ResetSlice(
                    parameter_name=parameter_name,
                    output_indices=indices,
                    parameter_shape=tuple(tensor.shape),
                )
            )
    expected = int(trace.get("num_reset_kernels", 0))
    actual = sum(len(item.output_indices) for item in omega)
    if expected != actual:
        raise ValueError(f"Trace reset count mismatch: {expected} != {actual}")
    return tuple(omega)


def _build_delta_and_validate_support(
    global_state: Mapping[str, torch.Tensor],
    reset_state: Mapping[str, torch.Tensor],
    omega: tuple[ResetSlice, ...],
) -> tuple[StateDict, float]:
    if global_state.keys() != reset_state.keys():
        raise ValueError("Global and reset states have different keys")
    reset_by_parameter = {item.parameter_name: item for item in omega}
    delta: StateDict = {}
    norm_square = 0.0

    for key in global_state:
        global_tensor = global_state[key]
        reset_tensor = reset_state[key]
        if global_tensor.shape != reset_tensor.shape or global_tensor.dtype != reset_tensor.dtype:
            raise ValueError(f"State metadata differs for {key}")
        reset_slice = reset_by_parameter.get(key)
        if reset_slice is None:
            if not torch.equal(global_tensor, reset_tensor):
                raise ValueError(f"Reset changed {key} outside recorded omega")
            continue

        selected = list(reset_slice.output_indices)
        mask = torch.ones(global_tensor.shape[0], dtype=torch.bool)
        mask[selected] = False
        if mask.any() and not torch.equal(global_tensor[mask], reset_tensor[mask]):
            raise ValueError(f"Reset changed unselected kernels in {key}")

        # Paper direction: delta = global state - reset state.
        parameter_delta = (
            global_tensor[selected] - reset_tensor[selected]
        ).detach().to("cpu").clone()
        if not torch.equal(
            parameter_delta, global_tensor[selected] - reset_tensor[selected]
        ):
            raise RuntimeError(f"Delta construction failed for {key}")
        delta[key] = parameter_delta
        norm_square += float(
            torch.sum(parameter_delta.to(torch.float64) ** 2).item()
        )

    return delta, math.sqrt(norm_square)


def _make_reset_state(
    model_template: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    config: FedRADConfig,
    round_idx: int,
    seed: int,
) -> tuple[StateDict, dict[str, object]]:
    model = copy.deepcopy(model_template).to("cpu")
    model.load_state_dict(clone_state_dict(global_state), strict=True)
    trace = reset_kernels_for_task(
        model,
        reset_ratio=config.reset_ratio,
        seed=seed,
        layer_scope="all",
        init_method=config.reset_method,
        at_least_one=False,
        current_iter=round_idx,
        conv_transition_period=config.fp_conv_rounds,
    )
    reset_state = clone_state_dict(model.state_dict())
    del model
    return reset_state, trace


@dataclass(frozen=True)
class TaskBank:
    round_idx: int
    parent_state_hash: str
    tasks: tuple[TaskSpec, ...]

    def task(self, task_id: int) -> TaskSpec:
        if not 0 <= task_id < len(self.tasks):
            raise IndexError(f"Invalid task_id {task_id}")
        task = self.tasks[task_id]
        if task.task_id != task_id:
            raise RuntimeError("TaskBank task ordering is corrupt")
        task.verify_hash()
        return task


def build_task_bank(
    *,
    config: FedRADConfig,
    model_template: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    round_idx: int,
    task_count: int,
    rngs: RNGStreams,
) -> TaskBank:
    if task_count < 1:
        raise ValueError("task_count must be positive")
    frozen_global = clone_state_dict(global_state)
    assert_finite_state(frozen_global, "global_state")
    parent_hash = state_dict_hash(frozen_global)
    tasks: list[TaskSpec] = []

    for task_id in range(task_count):
        seed = rngs.task_seed(round_idx, task_id)
        reset_state, trace = _make_reset_state(
            model_template, frozen_global, config, round_idx, seed
        )
        omega = _omega_from_trace(trace, reset_state)
        delta, delta_norm = _build_delta_and_validate_support(
            frozen_global, reset_state, omega
        )
        assert_finite_state(reset_state, f"task_{task_id}")
        state_hash = state_dict_hash(reset_state)

        if config.verify_task_replay:
            replay_state, replay_trace = _make_reset_state(
                model_template, frozen_global, config, round_idx, seed
            )
            if state_dict_hash(replay_state) != state_hash or replay_trace != trace:
                raise RuntimeError(
                    f"Task replay failed for round={round_idx}, task={task_id}"
                )

        task = TaskSpec(
            round_idx=int(round_idx),
            task_id=task_id,
            reset_seed=seed,
            parent_state_hash=parent_hash,
            state_hash=state_hash,
            omega=omega,
            delta_norm=delta_norm,
            reset_trace=copy.deepcopy(trace),
            active=bool(omega) and delta_norm > 0.0,
            _reset_state=clone_state_dict(reset_state),
            _delta=clone_state_dict(delta),
        )
        task.verify_hash()
        tasks.append(task)

    if state_dict_hash(global_state) != parent_hash:
        raise RuntimeError("Global state changed while building TaskBank")
    return TaskBank(
        round_idx=int(round_idx),
        parent_state_hash=parent_hash,
        tasks=tuple(tasks),
    )

