"""Motivation experiment for reset-conditioned client recovery.

This script isolates three claims behind recovery-aware client--reset matching:

H1. The same reset task is recovered differently by different non-IID clients.
H2. The client ranking changes across reset tasks (a client--task interaction).
H3. With the client set fixed, changing only the client--task assignment changes
    the aggregated model.

The experiment first obtains a non-random global checkpoint with FedAvg, then
trains every probe client from every deterministic FedPhoenix reset copy.  The
complete client x task matrix makes the claims falsifiable and avoids mixing
client selection gains with assignment gains.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.stats import f as f_distribution
from scipy.stats import kendalltau
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from Algorithm.Phoenix_util import reset_kernels_for_task
from models.Nets import CNNCifar
from models.resnetcifar import ResNet18_cifar10


class SyntheticCIFAR(Dataset):
    """Small CIFAR-shaped dataset used only for an offline smoke test."""

    def __init__(self, size: int, num_classes: int, seed: int):
        prototype_generator = torch.Generator().manual_seed(90210)
        generator = torch.Generator().manual_seed(seed)
        prototypes = torch.randn(
            num_classes, 3, 32, 32, generator=prototype_generator
        ) * 0.7
        labels = torch.randint(
            0, num_classes, (size,), generator=generator
        )
        noise = torch.randn(size, 3, 32, 32, generator=generator) * 0.6
        self.data = prototypes[labels] + noise
        self.targets = labels.tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test whether FedPhoenix reset recovery depends on client-task matching."
    )
    parser.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--dirichlet-beta", type=float, default=0.1)
    parser.add_argument("--min-client-samples", type=int, default=20)

    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--warmup-clients", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument(
        "--checkpoint-selection-samples",
        type=int,
        default=1000,
        help=(
            "Held-out samples used only to select the warm-up checkpoint. "
            "They are disjoint from the final evaluation samples."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--probe-clients", type=int, default=5)
    parser.add_argument(
        "--probe-selection",
        choices=["coverage", "random"],
        default="coverage",
        help=(
            "coverage fixes a collectively representative client set so the "
            "assignment test is not dominated by an accidentally skewed set"
        ),
    )
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reset-ratio", type=float, default=0.25)
    parser.add_argument(
        "--task-scopes",
        type=str,
        default="all",
        help="Comma-separated scopes cycled across tasks: all,early,middle,late.",
    )
    parser.add_argument(
        "--reset-init",
        choices=["uniform", "ori_normal", "kaiming_uniform", "kaiming_normal"],
        default="uniform",
    )
    parser.add_argument("--assignment-trials", type=int, default=20)
    parser.add_argument(
        "--exact-assignment-limit",
        type=int,
        default=720,
        help=(
            "Enumerate every client-task permutation when factorial(num_tasks) "
            "does not exceed this limit; otherwise sample assignment-trials."
        ),
    )

    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--significance-alpha", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--deterministic",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use deterministic PyTorch algorithms when available.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/motivation")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate summary report and figure from an existing output directory.",
    )
    parser.add_argument("--synthetic-train-size", type=int, default=3000)
    parser.add_argument("--synthetic-test-size", type=int, default=1000)
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def split_holdout_indices(
    dataset_size: int,
    selection_samples: int,
    eval_samples: int,
    seed: int,
):
    """Create deterministic, disjoint checkpoint-selection and evaluation sets."""
    if dataset_size < 2:
        raise ValueError("The holdout dataset needs at least two samples")
    if selection_samples < 1 or eval_samples < 1:
        raise ValueError("selection_samples and eval_samples must be positive")

    requested_total = selection_samples + eval_samples
    if requested_total <= dataset_size:
        selection_size = selection_samples
        eval_size = eval_samples
    else:
        selection_size = max(
            1,
            int(round(dataset_size * selection_samples / requested_total)),
        )
        selection_size = min(selection_size, dataset_size - 1)
        eval_size = dataset_size - selection_size

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dataset_size)
    selection = permutation[:selection_size].astype(np.int64).tolist()
    evaluation = permutation[
        selection_size : selection_size + eval_size
    ].astype(np.int64).tolist()
    return selection, evaluation


def indices_fingerprint(indices):
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def partition_fingerprint(partitions):
    digest = hashlib.sha256()
    for client_id in sorted(partitions):
        digest.update(np.asarray([client_id], dtype=np.int64).tobytes())
        digest.update(
            np.asarray(sorted(partitions[client_id]), dtype=np.int64).tobytes()
        )
    return digest.hexdigest()


def resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_datasets(args):
    if args.dataset == "synthetic":
        train = SyntheticCIFAR(
            args.synthetic_train_size, args.num_classes, args.seed
        )
        test = SyntheticCIFAR(
            args.synthetic_test_size, args.num_classes, args.seed + 1
        )
        return train, test

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    train = datasets.CIFAR10(
        args.data_root, train=True, download=True, transform=transform
    )
    test = datasets.CIFAR10(
        args.data_root, train=False, download=True, transform=transform
    )
    return train, test


def dirichlet_partition(labels, num_users: int, beta: float, min_size: int, seed: int):
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    for _ in range(1000):
        partitions = [[] for _ in range(num_users)]
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_users, beta))
            # Avoid repeatedly overfilling the same clients.
            capacity_mask = np.array(
                [len(part) < len(labels) / num_users for part in partitions],
                dtype=np.float64,
            )
            proportions *= capacity_mask
            if proportions.sum() == 0:
                proportions = np.ones(num_users) / num_users
            else:
                proportions /= proportions.sum()
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            for user_id, split in enumerate(np.split(class_indices, split_points)):
                partitions[user_id].extend(split.tolist())

        if min(len(part) for part in partitions) >= min_size:
            for part in partitions:
                rng.shuffle(part)
            return {user_id: part for user_id, part in enumerate(partitions)}

    raise RuntimeError(
        "Could not create a Dirichlet partition with the requested minimum size; "
        "increase the dataset size, beta, or reduce num_users/min-client-samples."
    )


def build_model(args):
    if args.model == "cnn":
        return CNNCifar(args)
    return ResNet18_cifar10(num_classes=args.num_classes)


def logits_from_output(output):
    return output["output"] if isinstance(output, dict) else output


def state_dict_to_cpu(model: nn.Module):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def average_state_dicts(state_dicts, weights):
    if len(state_dicts) != len(weights) or not state_dicts:
        raise ValueError("state_dicts and weights must be non-empty and aligned")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    result = {}
    for key, first_value in state_dicts[0].items():
        if torch.is_floating_point(first_value):
            value = first_value * float(normalized[0])
            for state, weight in zip(state_dicts[1:], normalized[1:]):
                value = value + state[key] * float(weight)
            result[key] = value.to(dtype=first_value.dtype)
        else:
            # BatchNorm counters and other integer buffers should not be averaged.
            result[key] = first_value.clone()
    return result


def train_local(args, initial_state, dataset, indices, device, seed):
    seed_everything(seed, bool(args.deterministic))
    model = build_model(args)
    model.load_state_dict(initial_state)
    model.to(device)
    model.train()

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    loss_sum = 0.0
    sample_count = 0
    for _ in range(args.local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = logits_from_output(model(images))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            sample_count += labels.size(0)

    state = state_dict_to_cpu(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, loss_sum / max(sample_count, 1)


def evaluate(args, state, dataset, indices, device):
    model = build_model(args)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    loss_sum = 0.0
    correct = 0
    sample_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = logits_from_output(model(images))
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += labels.size(0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "loss": loss_sum / max(sample_count, 1),
        "accuracy": 100.0 * correct / max(sample_count, 1),
        "samples": sample_count,
    }


def load_checkpoint(path, args):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    model = build_model(args)
    model.load_state_dict(payload)
    return state_dict_to_cpu(model)


def warmup_fedavg(args, initial_state, train_dataset, partitions, eval_dataset, eval_indices, device):
    state = initial_state
    rng = np.random.default_rng(args.seed + 17)
    available = np.array(sorted(partitions.keys()), dtype=np.int64)
    clients_per_round = min(args.warmup_clients, len(available))
    initial_metrics = evaluate(args, state, eval_dataset, eval_indices, device)
    best_state = copy.deepcopy(state)
    best_loss = initial_metrics["loss"]
    best_round = 0
    history = [{"round": 0, **initial_metrics}]

    for round_id in range(args.warmup_rounds):
        selected = rng.choice(available, clients_per_round, replace=False)
        local_states = []
        local_weights = []
        for position, client_id in enumerate(selected):
            local_state, _ = train_local(
                args,
                state,
                train_dataset,
                partitions[int(client_id)],
                device,
                seed=args.seed + round_id * 1000 + position,
            )
            local_states.append(local_state)
            local_weights.append(len(partitions[int(client_id)]))
        state = average_state_dicts(local_states, local_weights)

        if round_id == 0 or (round_id + 1) % 5 == 0 or round_id + 1 == args.warmup_rounds:
            metrics = evaluate(args, state, eval_dataset, eval_indices, device)
            history.append({"round": round_id + 1, **metrics})
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_state = copy.deepcopy(state)
                best_round = round_id + 1
            print(
                f"Warm-up round {round_id + 1:03d}/{args.warmup_rounds}: "
                f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.2f}%"
            )
    print(f"Selected warm-up checkpoint from round {best_round} (loss={best_loss:.4f}).")
    return best_state, history, best_round


def two_way_anova_with_replication(values):
    """Two-way ANOVA for values shaped [client, task, repeat]."""
    values = np.asarray(values, dtype=np.float64)
    clients, tasks, repeats = values.shape
    grand = values.mean()
    client_means = values.mean(axis=(1, 2))
    task_means = values.mean(axis=(0, 2))
    cell_means = values.mean(axis=2)

    ss_client = tasks * repeats * np.square(client_means - grand).sum()
    ss_task = clients * repeats * np.square(task_means - grand).sum()
    interaction_residual = (
        cell_means - client_means[:, None] - task_means[None, :] + grand
    )
    ss_interaction = repeats * np.square(interaction_residual).sum()
    ss_error = np.square(values - cell_means[:, :, None]).sum()
    ss_total = np.square(values - grand).sum()

    df_client = clients - 1
    df_task = tasks - 1
    df_interaction = df_client * df_task
    df_error = clients * tasks * (repeats - 1)

    def f_test(ss_effect, df_effect):
        if df_effect <= 0 or df_error <= 0 or ss_error <= 0:
            return None, None
        statistic = (ss_effect / df_effect) / (ss_error / df_error)
        return float(statistic), float(f_distribution.sf(statistic, df_effect, df_error))

    client_f, client_p = f_test(ss_client, df_client)
    task_f, task_p = f_test(ss_task, df_task)
    interaction_f, interaction_p = f_test(ss_interaction, df_interaction)
    return {
        "ss_client": float(ss_client),
        "ss_task": float(ss_task),
        "ss_interaction": float(ss_interaction),
        "ss_error": float(ss_error),
        "ss_total": float(ss_total),
        "client_f": client_f,
        "client_p": client_p,
        "task_f": task_f,
        "task_p": task_p,
        "interaction_f": interaction_f,
        "interaction_p": interaction_p,
        "interaction_eta_squared": (
            float(ss_interaction / ss_total) if ss_total > 0 else 0.0
        ),
        "interaction_partial_eta_squared": (
            float(ss_interaction / (ss_interaction + ss_error))
            if ss_interaction + ss_error > 0
            else 0.0
        ),
    }


def bootstrap_mean_interval(values, confidence_level, samples, seed):
    """Percentile bootstrap interval for a scalar mean."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "confidence_level": float(confidence_level),
        "bootstrap_samples": int(samples),
        "observations": int(values.size),
    }


def bootstrap_interaction_effect_interval(
    values,
    confidence_level,
    samples,
    seed,
):
    """Bootstrap ANOVA interaction effect sizes by resampling cell repeats."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError("values must have shape [client, task, repeat]")
    rng = np.random.default_rng(seed)
    eta_squared = []
    partial_eta_squared = []
    repeats = values.shape[2]
    for _ in range(samples):
        repeat_indices = rng.integers(
            0,
            repeats,
            size=values.shape,
        )
        sampled = np.take_along_axis(values, repeat_indices, axis=2)
        effect = two_way_anova_with_replication(sampled)
        eta_squared.append(effect["interaction_eta_squared"])
        partial_eta_squared.append(effect["interaction_partial_eta_squared"])

    tail = (1.0 - confidence_level) / 2.0

    def interval(items):
        return {
            "lower": float(np.quantile(items, tail)),
            "upper": float(np.quantile(items, 1.0 - tail)),
            "confidence_level": float(confidence_level),
            "bootstrap_samples": int(samples),
        }

    return {
        "interaction_eta_squared": interval(eta_squared),
        "interaction_partial_eta_squared": interval(partial_eta_squared),
    }


def holm_adjusted_pvalues(named_pvalues):
    """Holm family-wise error correction for pre-specified hypotheses."""
    valid = [
        (name, float(value))
        for name, value in named_pvalues.items()
        if value is not None and np.isfinite(value)
    ]
    valid.sort(key=lambda item: item[1])
    adjusted = {name: None for name in named_pvalues}
    running = 0.0
    count = len(valid)
    for rank, (name, value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = float(running)
    return adjusted


def ranking_summary(score_matrix, client_ids):
    score_matrix = np.asarray(score_matrix)
    correlations = []
    for first, second in itertools.combinations(range(score_matrix.shape[1]), 2):
        tau = kendalltau(score_matrix[:, first], score_matrix[:, second]).statistic
        if not np.isnan(tau):
            correlations.append(float(tau))
    best_positions = np.argmax(score_matrix, axis=0)
    return {
        "mean_pairwise_kendall_tau": (
            float(np.mean(correlations)) if correlations else None
        ),
        "min_pairwise_kendall_tau": (
            float(np.min(correlations)) if correlations else None
        ),
        "unique_best_clients": int(len(np.unique(best_positions))),
        "best_client_by_task": [int(client_ids[position]) for position in best_positions],
    }


def interaction_residual(score_matrix):
    """Remove general client and task effects, retaining pair-specific value."""
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    return (
        score_matrix
        - score_matrix.mean(axis=1, keepdims=True)
        - score_matrix.mean(axis=0, keepdims=True)
        + score_matrix.mean()
    )


def select_probe_clients(args, eligible, partitions, labels, rng):
    if args.probe_selection == "random":
        return sorted(
            int(item)
            for item in rng.choice(eligible, args.probe_clients, replace=False)
        )

    # Greedily form a set whose sample-weighted label distribution matches the
    # population distribution.  The set remains fixed for every assignment,
    # so this controls selection quality rather than giving any assignment an
    # advantage.
    labels = np.asarray(labels, dtype=np.int64)
    target = np.bincount(labels, minlength=args.num_classes).astype(np.float64)
    target /= target.sum()
    client_counts = {
        int(client_id): np.bincount(
            labels[np.asarray(partitions[int(client_id)], dtype=np.int64)],
            minlength=args.num_classes,
        ).astype(np.float64)
        for client_id in eligible
    }

    selected = []
    accumulated = np.zeros(args.num_classes, dtype=np.float64)
    candidates = set(int(item) for item in eligible)
    while len(selected) < args.probe_clients:
        scored = []
        for client_id in candidates:
            combined = accumulated + client_counts[client_id]
            distribution = combined / combined.sum()
            # L1 is stable when highly non-IID clients contain zero classes.
            distance = np.abs(distribution - target).sum()
            scored.append((float(distance), int(client_id)))
        best_distance = min(item[0] for item in scored)
        tied = [item[1] for item in scored if abs(item[0] - best_distance) < 1e-12]
        chosen = int(rng.choice(tied))
        selected.append(chosen)
        candidates.remove(chosen)
        accumulated += client_counts[chosen]
    return sorted(selected)


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_results(output_dir, score_matrix, client_ids, label_hist, assignment_rows):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    image = axes[0].imshow(score_matrix, aspect="auto", cmap="coolwarm")
    axes[0].set_title("Reset advantage vs. unreset control")
    axes[0].set_xlabel("Reset task")
    axes[0].set_ylabel("Client")
    axes[0].set_yticks(range(len(client_ids)), [str(item) for item in client_ids])
    axes[0].set_xticks(range(score_matrix.shape[1]))
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    image = axes[1].imshow(label_hist, aspect="auto", cmap="Blues")
    axes[1].set_title("Probe-client label distributions")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Client")
    axes[1].set_yticks(range(len(client_ids)), [str(item) for item in client_ids])
    figure.colorbar(image, ax=axes[1], fraction=0.046)

    labels = [row["assignment"] for row in assignment_rows]
    has_exact_population = any(
        row.get("assignment_kind") == "exact_population"
        for row in assignment_rows
    )
    losses = [row["aggregate_loss"] for row in assignment_rows]
    colors = [
        "tab:green" if label == "oracle_score_matched" else
        "tab:red" if label == "oracle_score_antimatched" else
        "tab:blue" if label.startswith(("random", "exact")) else
        "tab:gray"
        for label in labels
    ]
    axes[2].scatter(range(len(losses)), losses, c=colors, alpha=0.8)
    axes[2].set_title("Fixed clients, different task assignments")
    axes[2].set_xlabel("Assignment trial")
    axes[2].set_ylabel("Aggregated global loss")
    axes[2].legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green", label="score matched"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", label="score anti-matched"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:blue",
                label=("all assignments" if has_exact_population else "sampled assignments"),
            ),
        ],
        loc="best",
        frameon=False,
    )

    figure.tight_layout()
    figure.savefig(output_dir / "motivation_summary.png", dpi=200)
    plt.close(figure)


def write_markdown_report(output_dir, summary):
    h1 = summary["hypothesis_1_same_task_differs_by_client"]
    h2 = summary["hypothesis_2_client_ranking_depends_on_task"]
    h3 = summary["hypothesis_3_assignment_changes_aggregate"]
    interpretation = summary["interpretation"]
    ci = h1["mean_client_range_confidence_interval"]
    lines = [
        "# Reset-recovery motivation experiment",
        "",
        "This report describes one deterministic global seed. Checkpoint selection "
        "and final hypothesis evaluation use disjoint holdout samples.",
        "",
        "## Main results",
        "",
        (
            f"- H1 client effect: mean within-task client range "
            f"`{h1['mean_client_range_across_tasks']:.6f}` loss "
            f"({summary['config']['confidence_level']:.0%} bootstrap CI "
            f"`[{ci['lower']:.6f}, {ci['upper']:.6f}]`); Holm-adjusted "
            f"`p={h1['client_main_effect_holm_p']:.3e}`."
        ),
        (
            f"- H2 client-task interaction: eta-squared "
            f"`{h2['client_task_interaction_eta_squared']:.6f}`, partial "
            f"eta-squared `{h2['client_task_interaction_partial_eta_squared']:.6f}`; "
            f"Holm-adjusted `p={h2['client_task_interaction_holm_p']:.3e}`."
        ),
        (
            f"- H3 assignment effect: `{h3['assignment_population_mode']}` "
            f"assignment population with `{h3['assignment_population_count']}` "
            f"permutations; loss range "
            f"`{h3['random_assignment_loss_range']:.6f}`; matched-minus-population-mean "
            f"`{h3['oracle_matched_minus_random_mean']:.6f}`."
        ),
        (
            f"- H3 score-matched lower-tail probability: "
            f"`{h3['oracle_matched_lower_tail_probability']:.6f}` "
            "(lower loss is better)."
        ),
        "",
        "## Interpretation",
        "",
        f"- H1 supported: `{interpretation['h1_supported_at_0_05']}`",
        f"- H2 supported: `{interpretation['h2_supported_at_0_05']}`",
        f"- H3 assignment effect observed: `{interpretation['h3_assignment_effect_observed']}`",
        f"- Matched assignment beats assignment mean: `{interpretation['h3_oracle_matching_beats_random_mean']}`",
        f"- Matched assignment is in the alpha lower tail: `{interpretation['h3_score_matched_lower_tail_at_alpha']}`",
        "",
        "## Boundary",
        "",
        interpretation["warning"],
        "",
        "Plot-ready files: `figure_recovery_matrix.csv`, "
        "`figure_assignment_losses.csv`, and `figure_key_metrics.csv`.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def postprocess_existing_output(output_dir):
    """Regenerate human-readable outputs without rerunning model training."""
    output_dir = Path(output_dir)
    with (output_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    h3 = summary["hypothesis_3_assignment_changes_aggregate"]
    alpha = float(summary["config"].get("significance_alpha", 0.05))
    lower_tail = h3.get("oracle_matched_lower_tail_probability")
    summary["interpretation"]["h3_score_matched_lower_tail_at_alpha"] = (
        lower_tail is not None and float(lower_tail) <= alpha
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    client_ids = [int(item) for item in summary["probe_clients"]]
    matrix_rows = read_csv(output_dir / "figure_recovery_matrix.csv")
    task_ids = sorted({int(row["task_id"]) for row in matrix_rows})
    score_lookup = {
        (int(row["client_id"]), int(row["task_id"])): float(
            row["mean_reset_advantage_vs_unreset_loss"]
        )
        for row in matrix_rows
    }
    score_matrix = np.asarray(
        [
            [score_lookup[(client_id, task_id)] for task_id in task_ids]
            for client_id in client_ids
        ],
        dtype=np.float64,
    )

    label_rows = read_csv(output_dir / "probe_client_label_histograms.csv")
    label_lookup = {int(row["client_id"]): row for row in label_rows}
    label_hist = np.asarray(
        [
            [
                int(label_lookup[client_id][f"class_{class_id}"])
                for class_id in range(int(summary["config"]["num_classes"]))
            ]
            for client_id in client_ids
        ],
        dtype=np.int64,
    )
    assignment_rows = read_csv(output_dir / "figure_assignment_losses.csv")
    for row in assignment_rows:
        row["aggregate_loss"] = float(row["aggregate_loss"])

    write_markdown_report(output_dir, summary)
    plot_results(
        output_dir,
        score_matrix,
        client_ids,
        label_hist,
        assignment_rows,
    )
    print(f"Regenerated report and figure in {output_dir.resolve()}")


def state_path(directory: str, prefix: str, client_position: int, task_id=None):
    suffix = f"_task_{task_id}" if task_id is not None else ""
    return Path(directory) / f"{prefix}_client_{client_position}{suffix}.pt"


def evaluate_assignments(
    args,
    temporary_directory,
    client_ids,
    partitions,
    score_matrix,
    eval_dataset,
    eval_indices,
    device,
):
    num_clients, num_tasks = score_matrix.shape
    if num_clients != num_tasks:
        print("Skipping assignment test because probe-clients != num-tasks.")
        return []

    best_rows, best_cols = linear_sum_assignment(-score_matrix)
    worst_rows, worst_cols = linear_sum_assignment(score_matrix)
    best_permutation = np.empty(num_clients, dtype=np.int64)
    worst_permutation = np.empty(num_clients, dtype=np.int64)
    best_permutation[best_rows] = best_cols
    worst_permutation[worst_rows] = worst_cols

    rng = np.random.default_rng(args.seed + 300000)
    assignments = [
        ("oracle_score_matched", "oracle_matched", best_permutation),
        ("oracle_score_antimatched", "oracle_antimatched", worst_permutation),
    ]
    assignment_count = math.factorial(num_tasks)
    if assignment_count <= args.exact_assignment_limit:
        for index, permutation in enumerate(itertools.permutations(range(num_tasks))):
            assignments.append(
                (
                    f"exact_{index:03d}",
                    "exact_population",
                    np.asarray(permutation, dtype=np.int64),
                )
            )
    else:
        for trial in range(args.assignment_trials):
            assignments.append(
                (
                    f"random_{trial:03d}",
                    "sampled_assignment",
                    rng.permutation(num_tasks),
                )
            )

    cache = {}
    rows = []
    client_weights = [len(partitions[int(client_id)]) for client_id in client_ids]
    for label, assignment_kind, permutation in assignments:
        key = tuple(int(item) for item in permutation)
        if key not in cache:
            states = [
                torch.load(
                    state_path(temporary_directory, "pair", client_position, int(task_id)),
                    map_location="cpu",
                    weights_only=True,
                )
                for client_position, task_id in enumerate(permutation)
            ]
            aggregate = average_state_dicts(states, client_weights)
            cache[key] = evaluate(args, aggregate, eval_dataset, eval_indices, device)
        metrics = cache[key]
        rows.append(
            {
                "assignment": label,
                "assignment_kind": assignment_kind,
                "permutation_client_to_task": json.dumps(key),
                "oracle_pair_score_sum": float(
                    sum(score_matrix[i, task_id] for i, task_id in enumerate(key))
                ),
                "aggregate_loss": metrics["loss"],
                "aggregate_accuracy": metrics["accuracy"],
            }
        )
    return rows


def main():
    args = parse_args()
    if args.postprocess_only:
        postprocess_existing_output(args.output_dir)
        return
    if args.probe_clients < 2 or args.num_tasks < 2:
        raise ValueError("Use at least two clients and two tasks")
    if args.repeats < 1:
        raise ValueError("repeats must be at least one")
    if args.dirichlet_beta <= 0:
        raise ValueError("dirichlet-beta must be positive")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence-level must be between zero and one")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    if not 0.0 < args.significance_alpha < 1.0:
        raise ValueError("significance-alpha must be between zero and one")
    if args.exact_assignment_limit < 1:
        raise ValueError("exact-assignment-limit must be positive")

    seed_everything(args.seed, bool(args.deterministic))
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Results: {output_dir.resolve()}")

    train_dataset, test_dataset = load_datasets(args)
    partitions = dirichlet_partition(
        train_dataset.targets,
        args.num_users,
        args.dirichlet_beta,
        args.min_client_samples,
        args.seed,
    )

    rng = np.random.default_rng(args.seed)
    selection_indices, eval_indices = split_holdout_indices(
        len(test_dataset),
        args.checkpoint_selection_samples,
        args.eval_samples,
        args.seed + 700000,
    )
    print(
        "Holdout split: "
        f"checkpoint-selection={len(selection_indices)}, "
        f"final-evaluation={len(eval_indices)}, overlap=0"
    )

    initial_model = build_model(args)
    initial_state = state_dict_to_cpu(initial_model)
    warmup_history = []
    selected_warmup_round = None
    if args.checkpoint:
        global_state = load_checkpoint(args.checkpoint, args)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        global_state, warmup_history, selected_warmup_round = warmup_fedavg(
            args,
            initial_state,
            train_dataset,
            partitions,
            test_dataset,
            selection_indices,
            device,
        )
        write_csv(output_dir / "warmup_history.csv", warmup_history)
        torch.save(
            {
                "model_state": global_state,
                "config": vars(args),
                "selected_warmup_round": selected_warmup_round,
                "checkpoint_selection_indices_sha256": indices_fingerprint(
                    selection_indices
                ),
            },
            output_dir / "warmup_checkpoint.pt",
        )

    checkpoint_selection_metrics = evaluate(
        args, global_state, test_dataset, selection_indices, device
    )
    base_metrics = evaluate(
        args, global_state, test_dataset, eval_indices, device
    )
    print(
        f"Base checkpoint: loss={base_metrics['loss']:.4f}, "
        f"acc={base_metrics['accuracy']:.2f}%"
    )

    eligible = [
        client_id
        for client_id, indices in partitions.items()
        if len(indices) >= args.min_client_samples
    ]
    if len(eligible) < args.probe_clients:
        raise RuntimeError("Not enough eligible clients for the requested probe set")
    probe_clients = select_probe_clients(
        args,
        eligible,
        partitions,
        train_dataset.targets,
        rng,
    )
    print(f"Probe clients: {probe_clients}")

    labels = np.asarray(train_dataset.targets)
    label_hist = np.zeros((len(probe_clients), args.num_classes), dtype=np.int64)
    label_rows = []
    for position, client_id in enumerate(probe_clients):
        counts = np.bincount(
            labels[np.asarray(partitions[client_id], dtype=np.int64)],
            minlength=args.num_classes,
        )
        label_hist[position] = counts
        row = {"client_id": client_id, "num_samples": int(counts.sum())}
        row.update({f"class_{class_id}": int(count) for class_id, count in enumerate(counts)})
        label_rows.append(row)
    write_csv(output_dir / "probe_client_label_histograms.csv", label_rows)

    scopes = [item.strip() for item in args.task_scopes.split(",") if item.strip()]
    if not scopes or any(scope not in {"all", "early", "middle", "late"} for scope in scopes):
        raise ValueError("task-scopes must contain all/early/middle/late")

    task_states = []
    task_rows = []
    for task_id in range(args.num_tasks):
        task_model = build_model(args)
        task_model.load_state_dict(global_state)
        trace = reset_kernels_for_task(
            task_model,
            reset_ratio=args.reset_ratio,
            seed=args.seed + 200000 + task_id,
            layer_scope=scopes[task_id % len(scopes)],
            init_method=args.reset_init,
        )
        task_state = state_dict_to_cpu(task_model)
        task_states.append(task_state)
        metrics = evaluate(args, task_state, test_dataset, eval_indices, device)
        task_rows.append(
            {
                "task_id": task_id,
                "scope": trace["layer_scope"],
                "seed": trace["seed"],
                "num_reset_layers": trace["num_reset_layers"],
                "num_reset_kernels": trace["num_reset_kernels"],
                "reset_loss": metrics["loss"],
                "reset_accuracy": metrics["accuracy"],
                "loss_damage": metrics["loss"] - base_metrics["loss"],
                "accuracy_damage": base_metrics["accuracy"] - metrics["accuracy"],
                "trace": json.dumps(trace, ensure_ascii=False),
            }
        )
    write_csv(output_dir / "reset_tasks.csv", task_rows)

    recovery_values = np.zeros(
        (args.probe_clients, args.num_tasks, args.repeats), dtype=np.float64
    )
    reset_advantage_values = np.zeros_like(recovery_values)
    cell_rows = []
    control_rows = []

    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_directory:
        control_losses = np.zeros((args.probe_clients, args.repeats), dtype=np.float64)
        for client_position, client_id in enumerate(probe_clients):
            for repeat in range(args.repeats):
                local_seed = args.seed + 100000 + client_position * 100 + repeat
                control_state, train_loss = train_local(
                    args,
                    global_state,
                    train_dataset,
                    partitions[client_id],
                    device,
                    local_seed,
                )
                metrics = evaluate(
                    args, control_state, test_dataset, eval_indices, device
                )
                control_losses[client_position, repeat] = metrics["loss"]
                control_rows.append(
                    {
                        "client_id": client_id,
                        "repeat": repeat,
                        "local_train_loss": train_loss,
                        "global_eval_loss": metrics["loss"],
                        "global_eval_accuracy": metrics["accuracy"],
                    }
                )
                if repeat == 0:
                    torch.save(
                        control_state,
                        state_path(temporary_directory, "control", client_position),
                    )
        write_csv(output_dir / "unreset_client_controls.csv", control_rows)

        for client_position, client_id in enumerate(probe_clients):
            for task_id, task_state in enumerate(task_states):
                reset_loss = task_rows[task_id]["reset_loss"]
                reset_accuracy = task_rows[task_id]["reset_accuracy"]
                damage = reset_loss - base_metrics["loss"]
                for repeat in range(args.repeats):
                    # The same seed is used for this client's control and all tasks,
                    # so minibatch order does not masquerade as a task effect.
                    local_seed = args.seed + 100000 + client_position * 100 + repeat
                    trained_state, train_loss = train_local(
                        args,
                        task_state,
                        train_dataset,
                        partitions[client_id],
                        device,
                        local_seed,
                    )
                    metrics = evaluate(
                        args, trained_state, test_dataset, eval_indices, device
                    )
                    recovery = reset_loss - metrics["loss"]
                    reset_advantage = (
                        control_losses[client_position, repeat] - metrics["loss"]
                    )
                    recovery_values[client_position, task_id, repeat] = recovery
                    reset_advantage_values[
                        client_position, task_id, repeat
                    ] = reset_advantage
                    cell_rows.append(
                        {
                            "client_id": client_id,
                            "task_id": task_id,
                            "repeat": repeat,
                            "task_scope": task_rows[task_id]["scope"],
                            "local_train_loss": train_loss,
                            "post_train_global_loss": metrics["loss"],
                            "post_train_global_accuracy": metrics["accuracy"],
                            "functional_recovery_loss": recovery,
                            "functional_recovery_accuracy": metrics["accuracy"] - reset_accuracy,
                            "reset_advantage_vs_unreset_loss": reset_advantage,
                            "recovery_fraction_of_damage": (
                                recovery / damage if abs(damage) > 1e-12 else None
                            ),
                        }
                    )
                    if repeat == 0:
                        torch.save(
                            trained_state,
                            state_path(
                                temporary_directory,
                                "pair",
                                client_position,
                                task_id,
                            ),
                        )
                print(
                    f"Completed client {client_id} x task {task_id}: "
                    f"mean recovery={recovery_values[client_position, task_id].mean():.5f}"
                )
        write_csv(output_dir / "client_task_recovery.csv", cell_rows)

        mean_recovery = recovery_values.mean(axis=2)
        mean_reset_advantage = reset_advantage_values.mean(axis=2)
        assignment_rows = evaluate_assignments(
            args,
            temporary_directory,
            probe_clients,
            partitions,
            mean_reset_advantage,
            test_dataset,
            eval_indices,
            device,
        )
        write_csv(output_dir / "assignment_results.csv", assignment_rows)

        # Aggregate the same clients from the unreset checkpoint as a reference.
        control_states = [
            torch.load(
                state_path(temporary_directory, "control", position),
                map_location="cpu",
                weights_only=True,
            )
            for position in range(len(probe_clients))
        ]
        control_aggregate = average_state_dicts(
            control_states,
            [len(partitions[client_id]) for client_id in probe_clients],
        )
        control_aggregate_metrics = evaluate(
            args, control_aggregate, test_dataset, eval_indices, device
        )

    mean_recovery = recovery_values.mean(axis=2)
    mean_reset_advantage = reset_advantage_values.mean(axis=2)
    pair_specific_score = interaction_residual(mean_reset_advantage)
    anova = two_way_anova_with_replication(reset_advantage_values)
    raw_ranks = ranking_summary(mean_reset_advantage, probe_clients)
    pair_specific_ranks = ranking_summary(pair_specific_score, probe_clients)
    client_spread_by_task = (
        mean_reset_advantage.max(axis=0) - mean_reset_advantage.min(axis=0)
    )
    client_std_by_task = mean_reset_advantage.std(axis=0, ddof=1)

    client_spread_interval = bootstrap_mean_interval(
        client_spread_by_task,
        args.confidence_level,
        args.bootstrap_samples,
        args.seed + 410000,
    )
    interaction_effect_intervals = bootstrap_interaction_effect_interval(
        reset_advantage_values,
        args.confidence_level,
        args.bootstrap_samples,
        args.seed + 420000,
    )
    adjusted_pvalues = holm_adjusted_pvalues(
        {
            "h1_client_main_effect": anova["client_p"],
            "h2_client_task_interaction": anova["interaction_p"],
        }
    )

    assignment_population_rows = [
        row
        for row in assignment_rows
        if row["assignment_kind"]
        in {"exact_population", "sampled_assignment"}
    ]
    matched_row = next(
        (row for row in assignment_rows if row["assignment"] == "oracle_score_matched"),
        None,
    )
    antimatched_row = next(
        (row for row in assignment_rows if row["assignment"] == "oracle_score_antimatched"),
        None,
    )
    random_losses = [row["aggregate_loss"] for row in assignment_population_rows]
    assignment_mode = (
        "exact"
        if assignment_population_rows
        and assignment_population_rows[0]["assignment_kind"] == "exact_population"
        else "sampled"
    )
    matched_lower_tail_probability = (
        float(
            np.mean(
                np.asarray(random_losses, dtype=np.float64)
                <= float(matched_row["aggregate_loss"]) + 1e-12
            )
        )
        if matched_row and random_losses
        else None
    )
    matched_beats_or_ties_fraction = (
        float(
            np.mean(
                float(matched_row["aggregate_loss"])
                <= np.asarray(random_losses, dtype=np.float64) + 1e-12
            )
        )
        if matched_row and random_losses
        else None
    )
    assignment_mean_interval = bootstrap_mean_interval(
        random_losses,
        args.confidence_level,
        args.bootstrap_samples,
        args.seed + 430000,
    )

    figure_matrix_rows = []
    for client_position, client_id in enumerate(probe_clients):
        for task_id in range(args.num_tasks):
            figure_matrix_rows.append(
                {
                    "client_id": int(client_id),
                    "task_id": int(task_id),
                    "task_scope": task_rows[task_id]["scope"],
                    "mean_functional_recovery_loss": float(
                        mean_recovery[client_position, task_id]
                    ),
                    "mean_reset_advantage_vs_unreset_loss": float(
                        mean_reset_advantage[client_position, task_id]
                    ),
                    "pair_specific_interaction": float(
                        pair_specific_score[client_position, task_id]
                    ),
                }
            )
    write_csv(output_dir / "figure_recovery_matrix.csv", figure_matrix_rows)
    write_csv(output_dir / "figure_assignment_losses.csv", assignment_rows)

    summary = {
        "config": vars(args),
        "device": str(device),
        "probe_clients": probe_clients,
        "selected_warmup_round": selected_warmup_round,
        "checkpoint_selection_metrics": checkpoint_selection_metrics,
        "base_checkpoint": base_metrics,
        "unreset_same_clients_aggregate": control_aggregate_metrics,
        "hypothesis_1_same_task_differs_by_client": {
            "metric": "reset advantage versus the same client's unreset control",
            "mean_client_range_across_tasks": float(client_spread_by_task.mean()),
            "mean_client_range_confidence_interval": client_spread_interval,
            "mean_client_std_across_tasks": float(client_std_by_task.mean()),
            "client_main_effect_p": anova["client_p"],
            "client_main_effect_holm_p": adjusted_pvalues[
                "h1_client_main_effect"
            ],
        },
        "hypothesis_2_client_ranking_depends_on_task": {
            "metric": "client-task interaction after removing row/column main effects",
            "raw_reset_advantage_ranking": raw_ranks,
            "pair_specific_ranking": pair_specific_ranks,
            "client_task_interaction_p": anova["interaction_p"],
            "client_task_interaction_holm_p": adjusted_pvalues[
                "h2_client_task_interaction"
            ],
            "client_task_interaction_eta_squared": anova["interaction_eta_squared"],
            "client_task_interaction_partial_eta_squared": anova[
                "interaction_partial_eta_squared"
            ],
            "interaction_effect_confidence_intervals": (
                interaction_effect_intervals
            ),
        },
        "hypothesis_3_assignment_changes_aggregate": {
            "assignment_population_mode": assignment_mode,
            "assignment_population_count": len(random_losses),
            "random_assignment_loss_mean": (
                float(np.mean(random_losses)) if random_losses else None
            ),
            "random_assignment_loss_mean_confidence_interval": (
                assignment_mean_interval
            ),
            "random_assignment_loss_std": (
                float(np.std(random_losses, ddof=1)) if len(random_losses) > 1 else None
            ),
            "random_assignment_loss_range": (
                float(np.max(random_losses) - np.min(random_losses))
                if random_losses
                else None
            ),
            "oracle_matched_loss": matched_row["aggregate_loss"] if matched_row else None,
            "oracle_antimatched_loss": (
                antimatched_row["aggregate_loss"] if antimatched_row else None
            ),
            "oracle_matched_minus_random_mean": (
                matched_row["aggregate_loss"] - float(np.mean(random_losses))
                if matched_row and random_losses
                else None
            ),
            "oracle_matched_lower_tail_probability": (
                matched_lower_tail_probability
            ),
            "oracle_matched_beats_or_ties_fraction": (
                matched_beats_or_ties_fraction
            ),
        },
        "two_way_anova": anova,
        "validity_checks": {
            "base_accuracy_above_chance": (
                base_metrics["accuracy"] > 100.0 / args.num_classes
            ),
            "tasks_with_positive_loss_damage_fraction": float(
                np.mean([row["loss_damage"] > 0 for row in task_rows])
            ),
            "tasks_with_positive_accuracy_damage_fraction": float(
                np.mean([row["accuracy_damage"] > 0 for row in task_rows])
            ),
            "checkpoint_selection_evaluation_overlap": int(
                len(set(selection_indices).intersection(eval_indices))
            ),
        },
        "reproducibility": {
            "single_global_seed": int(args.seed),
            "deterministic_algorithms": bool(args.deterministic),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "checkpoint_selection_samples": len(selection_indices),
            "final_evaluation_samples": len(eval_indices),
            "checkpoint_selection_indices_sha256": indices_fingerprint(
                selection_indices
            ),
            "final_evaluation_indices_sha256": indices_fingerprint(eval_indices),
            "client_partition_sha256": partition_fingerprint(partitions),
        },
        "interpretation": {
            "h1_supported_at_0_05": (
                adjusted_pvalues["h1_client_main_effect"] is not None
                and adjusted_pvalues["h1_client_main_effect"]
                < args.significance_alpha
            ),
            "h2_supported_at_0_05": (
                adjusted_pvalues["h2_client_task_interaction"] is not None
                and adjusted_pvalues["h2_client_task_interaction"]
                < args.significance_alpha
                and pair_specific_ranks["unique_best_clients"] > 1
            ),
            "h3_assignment_effect_observed": (
                bool(random_losses)
                and float(np.max(random_losses) - np.min(random_losses)) > 1e-12
            ),
            "h3_oracle_matching_beats_random_mean": (
                matched_row is not None
                and bool(random_losses)
                and matched_row["aggregate_loss"] < float(np.mean(random_losses))
            ),
            "h3_score_matched_lower_tail_at_alpha": (
                matched_lower_tail_probability is not None
                and matched_lower_tail_probability <= args.significance_alpha
            ),
            "warning": (
                "The oracle matching uses the completed recovery matrix and is a "
                "motivation upper bound, not a deployable selector. This run uses "
                "one global seed and supports a within-run mechanism claim, not a "
                "cross-seed robustness claim."
            ),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_csv(
        output_dir / "figure_key_metrics.csv",
        [
            {
                "seed": int(args.seed),
                "h1_mean_client_range": float(client_spread_by_task.mean()),
                "h1_range_ci_lower": client_spread_interval["lower"],
                "h1_range_ci_upper": client_spread_interval["upper"],
                "h1_holm_p": adjusted_pvalues["h1_client_main_effect"],
                "h2_interaction_eta_squared": anova[
                    "interaction_eta_squared"
                ],
                "h2_interaction_partial_eta_squared": anova[
                    "interaction_partial_eta_squared"
                ],
                "h2_holm_p": adjusted_pvalues[
                    "h2_client_task_interaction"
                ],
                "h3_assignment_mode": assignment_mode,
                "h3_assignment_count": len(random_losses),
                "h3_assignment_loss_mean": (
                    float(np.mean(random_losses)) if random_losses else None
                ),
                "h3_assignment_loss_range": (
                    float(np.max(random_losses) - np.min(random_losses))
                    if random_losses
                    else None
                ),
                "h3_matched_loss": (
                    matched_row["aggregate_loss"] if matched_row else None
                ),
                "h3_matched_minus_assignment_mean": (
                    matched_row["aggregate_loss"] - float(np.mean(random_losses))
                    if matched_row and random_losses
                    else None
                ),
                "h3_matched_lower_tail_probability": (
                    matched_lower_tail_probability
                ),
            }
        ],
    )
    write_markdown_report(output_dir, summary)

    plot_results(
        output_dir,
        mean_reset_advantage,
        probe_clients,
        label_hist,
        assignment_rows,
    )
    print(json.dumps(summary["interpretation"], ensure_ascii=False, indent=2))
    print(f"Finished. See {output_dir.resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
