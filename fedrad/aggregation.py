from __future__ import annotations

from collections.abc import Sequence

import torch

from fedrad.types import ClientUpdate, StateDict, assert_finite_state


def weighted_fedavg(updates: Sequence[ClientUpdate]) -> StateDict:
    if not updates:
        raise ValueError("weighted_fedavg requires at least one ClientUpdate")
    if any(update.num_examples <= 0 for update in updates):
        raise ValueError("All ClientUpdate sample counts must be positive")

    states = [update.clone_state() for update in updates]
    reference_keys = tuple(states[0].keys())
    for update, state in zip(updates, states):
        if tuple(state.keys()) != reference_keys:
            raise ValueError(f"Client {update.client_id} state keys/order differ")
        assert_finite_state(state, f"client_{update.client_id}")
        for key in reference_keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(f"Client {update.client_id} shape differs for {key}")
            if state[key].dtype != states[0][key].dtype:
                raise ValueError(f"Client {update.client_id} dtype differs for {key}")

    counts = torch.tensor(
        [update.num_examples for update in updates], dtype=torch.float64
    )
    weights = counts / counts.sum()
    result: StateDict = {}

    for key in reference_keys:
        values = [state[key].detach().to("cpu") for state in states]
        reference = values[0]
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for weight, value in zip(weights, values):
                accumulator.add_(value.to(torch.float64), alpha=float(weight.item()))
            result[key] = accumulator.to(dtype=reference.dtype)
        elif key == "num_batches_tracked" or key.endswith(".num_batches_tracked"):
            result[key] = torch.stack(values).max(dim=0).values.to(reference.dtype)
        else:
            if not all(torch.equal(reference, value) for value in values[1:]):
                raise ValueError(
                    f"Unsupported non-floating buffer {key} differs across clients"
                )
            result[key] = reference.clone()

    assert_finite_state(result, "aggregated_state")
    return result

