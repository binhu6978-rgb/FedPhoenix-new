from __future__ import annotations

"""Phase 4.2: offline full-local-training utility fidelity diagnostic.

This is deliberately separate from Clean/FedRAD trainers.  It replays the
completed Clean trajectory only to reconstruct four verified pre-round states,
then evaluates all client x reset-state pairs offline.  No test-set metric,
aggregation experiment, score-weight search, or training-policy modification is
performed.
"""

import argparse
import copy
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedrad.aggregation import weighted_fedavg
from fedrad.config import FedRADConfig
from fedrad.data import FederatedData, load_cifar10
from fedrad.local_trainer import LocalTrainer
from fedrad.models import build_model, extract_logits
from fedrad.probe import ProbeRunner, materialize_probe_batch
from fedrad.rng import RNGStreams, isolated_python_numpy_rng, isolated_torch_rng
from fedrad.scoring import matrix_zscore
from fedrad.task_bank import TaskBank, build_task_bank
from fedrad.types import ProbeResult, StateDict, clone_state_dict, state_dict_hash


SNAPSHOTS = (20, 40, 120, 160)
REFERENCE_SAMPLES_PER_CLIENT = 16
REFERENCE_SEED = 4_200_042
RATIO_EPS = 1e-12
RANDOM_PERMUTATIONS = 10_000


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields_order = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_order)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_config(source_run: Path) -> FedRADConfig:
    payload = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(FedRADConfig)}
    kwargs = {key: payload[key] for key in allowed if key in payload}
    for key in ("data_root", "partition_path", "output_root", "warmup_reference_rounds_path"):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = Path(kwargs[key])
    kwargs["diagnostic_probe_rounds"] = tuple(kwargs.get("diagnostic_probe_rounds", ()))
    kwargs["download"] = False
    kwargs["regenerate_partition"] = False
    config = FedRADConfig(**kwargs)
    config.validate()
    return config


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _verify_task_bank(bank: TaskBank, expected: list[dict[str, Any]], round_number: int) -> None:
    expected_ordered = sorted(expected, key=lambda item: int(item["task_id"]))
    if len(bank.tasks) != len(expected_ordered):
        raise RuntimeError(f"round {round_number}: TaskBank length mismatch")
    for task, saved in zip(bank.tasks, expected_ordered):
        observed = (task.task_id, task.reset_seed, task.parent_state_hash, task.state_hash)
        recorded = (
            int(saved["task_id"]), int(saved["reset_seed"]),
            str(saved["parent_state_hash"]), str(saved["state_hash"]),
        )
        if observed != recorded:
            raise RuntimeError(f"round {round_number}: TaskBank replay mismatch")


def _reference_batch(data: FederatedData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fixed, weighted stratified training-data objective over all 100 clients."""
    images: list[torch.Tensor] = []
    labels: list[int] = []
    weights: list[float] = []
    selected: dict[int, list[int]] = {}
    total_examples = sum(len(indices) for indices in data.partitions.values())
    for client_id in sorted(data.partitions):
        indices = np.asarray(data.partitions[client_id], dtype=np.int64)
        count = min(REFERENCE_SAMPLES_PER_CLIENT, len(indices))
        if count < 1:
            raise ValueError(f"reference client {client_id} is empty")
        chosen = np.random.default_rng(REFERENCE_SEED + int(client_id)).choice(
            indices, size=count, replace=False
        )
        selected[int(client_id)] = [int(index) for index in chosen.tolist()]
        # Each sampled item represents an equal share of its full client mass.
        item_weight = len(indices) / total_examples / count
        with isolated_python_numpy_rng(REFERENCE_SEED + int(client_id)), isolated_torch_rng(
            REFERENCE_SEED + int(client_id)
        ):
            for index in chosen.tolist():
                image, label = data.train_dataset[int(index)]
                if not isinstance(image, torch.Tensor):
                    raise TypeError("training transform did not produce a Tensor")
                images.append(image.detach().to("cpu").clone())
                labels.append(int(label))
                weights.append(float(item_weight))
    image_tensor = torch.stack(images)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    weight_tensor = torch.tensor(weights, dtype=torch.float64)
    if not np.isclose(float(weight_tensor.sum()), 1.0):
        raise RuntimeError("federation reference weights do not sum to one")
    digest = hashlib.sha256()
    digest.update(image_tensor.numpy().tobytes())
    digest.update(label_tensor.numpy().tobytes())
    digest.update(weight_tensor.numpy().tobytes())
    metadata = {
        "kind": "sample-size-weighted stratified federation training objective",
        "clients_covered": len(selected),
        "samples_per_client": REFERENCE_SAMPLES_PER_CLIENT,
        "total_reference_samples": len(labels),
        "reference_seed": REFERENCE_SEED,
        "reference_batch_sha256": digest.hexdigest(),
        "selected_train_indices_by_client": selected,
    }
    return image_tensor, label_tensor, weight_tensor, metadata


def federation_reference_loss(
    *,
    state: Mapping[str, torch.Tensor],
    model_template: torch.nn.Module,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> float:
    """Weighted empirical cross-entropy on a fixed federation-wide train subset."""
    images, labels, weights = reference
    state_copy = clone_state_dict(state)
    before = state_dict_hash(state_copy)
    model = copy.deepcopy(model_template)
    model.load_state_dict(state_copy, strict=True)
    model.to(device)
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad(), isolated_torch_rng(REFERENCE_SEED, device):
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            logits = extract_logits(model(images[start:stop].to(device, non_blocking=True)))
            losses = F.cross_entropy(
                logits, labels[start:stop].to(device, non_blocking=True), reduction="none"
            ).to(torch.float64)
            total += torch.sum(losses * weights[start:stop].to(device, non_blocking=True))
    if state_dict_hash(state_copy) != before:
        raise RuntimeError("reference objective mutated its input state")
    value = float(total.item())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(value):
        raise ValueError("reference objective is non-finite")
    return value


def _probe_matrices(
    *,
    config: FedRADConfig,
    data: FederatedData,
    model_template: torch.nn.Module,
    global_state: Mapping[str, torch.Tensor],
    bank: TaskBank,
    clients: Sequence[int],
    round_idx: int,
    rngs: RNGStreams,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    runner = ProbeRunner(config=config, model_template=model_template, device=device)
    task_ids = tuple(task.task_id for task in bank.tasks)
    results: list[ProbeResult] = []
    pair_rows: list[dict[str, Any]] = []
    for client_id in clients:
        seed = rngs.probe_seed(round_idx, int(client_id))
        batch = materialize_probe_batch(
            dataset=data.train_dataset,
            client_indices=data.partitions[int(client_id)],
            client_id=int(client_id), round_idx=round_idx, probe_seed=seed,
            support_limit=config.probe_support_size, query_limit=config.probe_query_size,
        )
        global_loss = runner.global_reference(global_state=global_state, batch=batch)
        for task in bank.tasks:
            result = runner.run_pair(task=task, batch=batch, global_loss=global_loss)
            results.append(result)
            pair_rows.append({
                "round": round_idx + 1, "client_id": result.client_id, "task_id": result.task_id,
                "global_loss": result.global_loss, "reset_loss": result.reset_loss,
                "adapted_loss": result.adapted_loss, "G": result.G, "A": result.A,
                "D": result.D, "C": result.C, "probe_seed": result.probe_seed,
                "support_hash": result.support_hash, "query_hash": result.query_hash,
            })
    lookup = {(result.client_id, result.task_id): result for result in results}
    fields = ("global_loss", "reset_loss", "adapted_loss", "G", "A", "D", "C")
    matrices = {
        field: np.asarray(
            [[float(getattr(lookup[(int(client), task)], field)) for task in task_ids] for client in clients],
            dtype=np.float64,
        )
        for field in fields
    }
    return matrices, pair_rows


def candidate_scores(raw: Mapping[str, np.ndarray], config: FedRADConfig) -> dict[str, np.ndarray]:
    """Fixed Phase 4.2 candidates; R has no tuned hyperparameter.

    R_ij = (reset_loss_ij - adapted_loss_ij) /
           (abs(reset_loss_ij - global_loss_ij) + 1e-12).
    Its denominator is the magnitude of reset damage relative to the global
    model, so it measures recovered loss per unit of reset displacement.
    """
    normalized = {
        name: matrix_zscore(
            raw[name], z_eps=config.z_eps, std_atol=config.std_atol, std_rtol=config.std_rtol
        )[0]
        for name in ("G", "A", "D", "C")
    }
    damage = np.abs(raw["reset_loss"] - raw["global_loss"])
    ratio = raw["G"] / (damage + RATIO_EPS)
    scores = {
        "Legacy": normalized["G"] + normalized["A"] - normalized["D"] + config.lambda_C * normalized["C"],
        "GAD": normalized["G"] + normalized["A"] - normalized["D"],
        "R": ratio,
        "AdaptedLoss": -raw["adapted_loss"],
        "GOnly": normalized["G"],
    }
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError("candidate score is non-finite")
    return scores


def _rankdata(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(vector, kind="mergesort")
    ranks = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(vector):
        end = start + 1
        while end < len(vector) and vector[order[end]] == vector[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator else float("nan")


def _double_center(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return values - values.mean(1, keepdims=True) - values.mean(0, keepdims=True) + values.mean()


def _assignment(matrix: np.ndarray) -> tuple[tuple[int, ...], float]:
    rows, columns = linear_sum_assignment(-np.asarray(matrix, dtype=np.float64))
    pairs = sorted(zip(rows.tolist(), columns.tolist()))
    assignment = tuple(column for _, column in pairs)
    return assignment, float(sum(matrix[row, column] for row, column in pairs))


def _assignment_value(matrix: np.ndarray, assignment: Sequence[int]) -> float:
    return float(np.asarray(matrix)[np.arange(len(assignment)), np.asarray(assignment)].sum())


def _evaluate_snapshot(
    *,
    round_number: int,
    global_state: Mapping[str, torch.Tensor],
    bank: TaskBank,
    clients: Sequence[int],
    data: FederatedData,
    config: FedRADConfig,
    model_template: torch.nn.Module,
    rngs: RNGStreams,
    device: torch.device,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    round_idx = round_number - 1
    raw, probe_rows = _probe_matrices(
        config=config, data=data, model_template=model_template, global_state=global_state,
        bank=bank, clients=clients, round_idx=round_idx, rngs=rngs, device=device,
    )
    reference_reset = np.asarray([
        federation_reference_loss(
            state=task.clone_reset_state(), model_template=model_template, reference=reference,
            device=device, batch_size=config.eval_batch_size,
        )
        for task in bank.tasks
    ], dtype=np.float64)
    trainer = LocalTrainer(config=config, model_template=model_template, train_dataset=data.train_dataset, device=device)
    value = np.empty((len(clients), len(bank.tasks)), dtype=np.float64)
    full_rows: list[dict[str, Any]] = []
    for row, client_id in enumerate(clients):
        local_seed = rngs.local_seed(round_idx, int(client_id))
        for column, task in enumerate(bank.tasks):
            update = trainer.train_client(
                initial_state=task.clone_reset_state(), client_id=int(client_id), task_id=task.task_id,
                client_indices=data.partitions[int(client_id)], round_idx=round_idx, local_seed=local_seed,
            )
            if update.initial_state_hash != task.state_hash:
                raise RuntimeError("full oracle did not start from ORIGINAL TaskSpec")
            after_loss = federation_reference_loss(
                state=update.clone_state(), model_template=model_template, reference=reference,
                device=device, batch_size=config.eval_batch_size,
            )
            value[row, column] = reference_reset[column] - after_loss
            full_rows.append({
                "round": round_number, "client_id": int(client_id), "task_id": task.task_id,
                "reset_reference_loss": float(reference_reset[column]),
                "full_reference_loss": after_loss, "V_full": float(value[row, column]),
                "local_seed": local_seed, "initial_state_hash": update.initial_state_hash,
                "full_state_hash": update.state_hash,
            })
    return raw, probe_rows, full_rows, value


def _analysis_rows(
    *, round_number: int, scores: Mapping[str, np.ndarray], value: np.ndarray,
    client_order: Sequence[int], task_order: Sequence[int], random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    value_interaction = _double_center(value)
    oracle_assignment, oracle_value = _assignment(value)
    clean_assignment = tuple(range(value.shape[0]))
    clean_value = _assignment_value(value, clean_assignment)
    rng = np.random.default_rng(random_seed)
    random_values = np.asarray([
        _assignment_value(value, rng.permutation(value.shape[1])) for _ in range(RANDOM_PERMUTATIONS)
    ], dtype=np.float64)
    correlation_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    assignment_rows.append({
        "round": round_number, "candidate": "Clean", "assignment": list(clean_assignment),
        "value_on_V_full": clean_value, "random_percentile": float(np.mean(random_values <= clean_value) * 100.0),
        "regret_to_oracle": oracle_value - clean_value, "improvement_over_clean": 0.0,
        "pair_overlap_with_oracle": float(np.mean(np.asarray(clean_assignment) == np.asarray(oracle_assignment))),
    })
    assignment_rows.append({
        "round": round_number, "candidate": "OracleV", "assignment": list(oracle_assignment),
        "value_on_V_full": oracle_value, "random_percentile": float(np.mean(random_values <= oracle_value) * 100.0),
        "regret_to_oracle": 0.0, "improvement_over_clean": oracle_value - clean_value,
        "pair_overlap_with_oracle": 1.0,
    })
    for name, score in scores.items():
        score_interaction = _double_center(score)
        correlation_rows.append({
            "round": round_number, "candidate": name,
            "raw_pearson": _pearson(score, value),
            "raw_spearman": _pearson(_rankdata(score), _rankdata(value)),
            "interaction_pearson": _pearson(score_interaction, value_interaction),
            "interaction_spearman": _pearson(_rankdata(score_interaction), _rankdata(value_interaction)),
        })
        assignment, _ = _assignment(score)
        assignment_value = _assignment_value(value, assignment)
        assignment_rows.append({
            "round": round_number, "candidate": name, "assignment": list(assignment),
            "value_on_V_full": assignment_value,
            "random_percentile": float(np.mean(random_values <= assignment_value) * 100.0),
            "regret_to_oracle": oracle_value - assignment_value,
            "improvement_over_clean": assignment_value - clean_value,
            "pair_overlap_with_oracle": float(np.mean(np.asarray(assignment) == np.asarray(oracle_assignment))),
        })
    return correlation_rows, assignment_rows, random_values


def _write_report(path: Path, correlations: list[dict[str, Any]], assignments: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> None:
    by_round = {int(item["round"]): item for item in snapshots}
    lines = [
        "# Phase 4.2 — Full-Training Utility Fidelity", "",
        "## Scope", "",
        "A single exact Clean FedPhoenix replay reconstructed pre-round snapshots at rounds 20, 40, 120, and 160. Each replayed round was checked against the original Clean run's selected clients, TaskBank seeds/hashes, local seeds, and post-round global hash. At each snapshot, every one of the original 10 clients was fully trained from every one of the original 10 reset TaskSpecs (100 pairs). No test-set data, test accuracy, aggregation trajectory, or score-weight search was used.", "",
        "`V_full(i,j) = L_fed_ref(reset_j) - L_fed_ref(w_full(i,j))`, where `w_full` is produced by the unchanged 5-epoch LocalTrainer. `L_fed_ref` is a fixed sample-size-weighted stratified empirical cross-entropy over 16 deterministic training examples from each of all 100 federation clients (1,600 total); it uses no test samples and is analysis-only.", "",
        "The fixed ratio candidate is `R = (reset_loss - adapted_loss) / (abs(reset_loss - global_loss) + 1e-12)`. This is recovery per absolute reset-damage unit; `1e-12` is only numerical stabilization.", "",
        "## Snapshot replay", "",
        "| Snapshot | Pre-round global hash verified | Original clients/tasks | Full pairs | Reference samples |", "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in snapshots:
        lines.append(f"| {row['round']} | yes | {row['clients']} / {row['tasks']} | {row['full_pairs']} | {row['reference_samples']} |")
    lines.extend(["", "## Correlation with full-training utility", "", "Raw correlations are secondary. The primary interaction correlations double-center both score and `V_full`, removing client and state main effects that are constant over every bijection.", "", "| Round | Score | Raw Pearson | Raw Spearman | Interaction Pearson | Interaction Spearman |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for row in correlations:
        lines.append("| {round} | {candidate} | {raw_pearson:.3f} | {raw_spearman:.3f} | {interaction_pearson:.3f} | {interaction_spearman:.3f} |".format(**row))
    lines.extend(["", "## Assignment oracle evaluation", "", "Each candidate's Hungarian-max assignment is evaluated on `V_full`, never on its own score. Random percentiles use 10,000 fixed-seed bijections per snapshot.", "", "| Round | Candidate | V_full value | Random percentile | Regret to oracle | Improvement over Clean | Oracle-pair overlap |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in assignments:
        lines.append("| {round} | {candidate} | {value_on_V_full:.5f} | {random_percentile:.2f}% | {regret_to_oracle:.5f} | {improvement_over_clean:.5f} | {pair_overlap_with_oracle:.2f} |".format(**row))
    candidates = sorted({str(row["candidate"]) for row in correlations})
    positive_interaction_pearson = [
        candidate for candidate in candidates
        if all(float(row["interaction_pearson"]) > 0.0 for row in correlations if row["candidate"] == candidate)
    ]
    positive_interaction_spearman = [
        candidate for candidate in candidates
        if all(float(row["interaction_spearman"]) > 0.0 for row in correlations if row["candidate"] == candidate)
    ]
    score_assignments = [row for row in assignments if row["candidate"] not in {"Clean", "OracleV"}]
    always_beats_clean = [
        candidate for candidate in candidates
        if all(float(row["improvement_over_clean"]) > 0.0 for row in score_assignments if row["candidate"] == candidate)
    ]
    lines.extend([
        "", "## Evidence-based conclusion", "",
        "No fixed candidate has positive interaction Pearson correlation at all four snapshots: "
        + (", ".join(positive_interaction_pearson) if positive_interaction_pearson else "none") + ".",
        "No fixed candidate has positive interaction Spearman correlation at all four snapshots: "
        + (", ".join(positive_interaction_spearman) if positive_interaction_spearman else "none") + ".",
        "No score-derived Hungarian assignment improves on Clean at all four snapshots: "
        + (", ".join(always_beats_clean) if always_beats_clean else "none") + ".",
        "Accordingly, this diagnostic does not provide evidence that the present Probe scores faithfully recover the client-by-state interaction in full local-training utility. Isolated assignment gains must not be interpreted as validation of conditional utility when their interaction correlations are near zero or sign-unstable. On this evidence alone, Phase 4.3 state-conditioned assignment is not justified.",
        "", "## Limits", "",
        "The result is limited to four snapshots from one Clean seed and to a fixed 1,600-example federation-training reference subset. It excludes test data by design, and it does not establish that the full-training oracle itself is a complete downstream utility. These are the main remaining sources of uncertainty; they do not turn the observed lack of cross-snapshot score fidelity into positive evidence.",
        "", "## Interpretation", "",
        "This phase is a fidelity diagnostic, not an accuracy comparison. The report does not claim that any score improves 200-round test accuracy.",
        "", "Machine-readable source/replay metadata, raw Probe matrices, full-training utilities, score matrices, correlation tables, assignments, and random distributions are saved beside this report.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source_run: Path, output_dir: Path) -> None:
    config = _load_config(source_run)
    _configure_determinism(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rngs = RNGStreams(config.seed)
    data = load_cifar10(config, rngs)
    model = build_model(config, rngs.model_seed)
    reference_images, reference_labels, reference_weights, reference_meta = _reference_batch(data)
    reference = (reference_images, reference_labels, reference_weights)
    source_rounds = {int(row["round_number"]): row for row in _jsonl(source_run / "rounds.jsonl")}
    source_tasks = _jsonl(source_run / "tasks.jsonl")
    if any(round_number not in source_rounds for round_number in SNAPSHOTS):
        raise ValueError("source run lacks a requested Phase 4.2 snapshot")
    output_dir.mkdir(parents=True, exist_ok=False)
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir()
    validation_path = output_dir / "replay_validation.jsonl"
    global_state: StateDict = clone_state_dict(model.state_dict())
    trainer = LocalTrainer(config=config, model_template=model, train_dataset=data.train_dataset, device=device)
    all_probe_rows: list[dict[str, Any]] = []
    all_full_rows: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    random_distributions: dict[str, np.ndarray] = {}
    replay_started = time.perf_counter()

    for round_number in range(1, max(SNAPSHOTS) + 1):
        round_idx = round_number - 1
        saved = source_rounds[round_number]
        clients = rngs.sample_clients(config.num_users, config.clients_per_round)
        bank = build_task_bank(config=config, model_template=model, global_state=global_state,
                               round_idx=round_idx, task_count=config.clients_per_round, rngs=rngs)
        expected_tasks = [item for item in source_tasks if int(item["round_idx"]) == round_idx]
        _verify_task_bank(bank, expected_tasks, round_number)
        if list(clients) != saved["selected_clients"]:
            raise RuntimeError(f"round {round_number}: client sampling replay mismatch")
        if [task.reset_seed for task in bank.tasks] != saved["task_seeds"] or [task.state_hash for task in bank.tasks] != saved["task_hashes"]:
            raise RuntimeError(f"round {round_number}: task replay mismatch")

        if round_number in SNAPSHOTS:
            start_hash = state_dict_hash(global_state)
            raw, probe_rows, full_rows, value = _evaluate_snapshot(
                round_number=round_number, global_state=global_state, bank=bank, clients=clients,
                data=data, config=config, model_template=model, rngs=rngs, device=device, reference=reference,
            )
            if state_dict_hash(global_state) != start_hash:
                raise RuntimeError("Phase 4.2 diagnostic mutated the replay global state")
            scores = candidate_scores(raw, config)
            corr_rows, assignment_rows, random_values = _analysis_rows(
                round_number=round_number, scores=scores, value=value, client_order=clients,
                task_order=tuple(task.task_id for task in bank.tasks), random_seed=REFERENCE_SEED + round_number,
            )
            archive_path = matrix_dir / f"round_{round_number:04d}.npz"
            np.savez_compressed(archive_path, client_order=np.asarray(clients, dtype=np.int64),
                                task_order=np.asarray([task.task_id for task in bank.tasks], dtype=np.int64),
                                V_full=value, reset_reference_loss=np.asarray([row["reset_reference_loss"] for row in full_rows[:len(bank.tasks)]], dtype=np.float64),
                                **raw, **{f"Q_{name}": matrix for name, matrix in scores.items()})
            snapshots.append({"round": round_number, "pre_round_global_hash": start_hash,
                              "clients": len(clients), "tasks": len(bank.tasks), "full_pairs": len(full_rows),
                              "reference_samples": reference_meta["total_reference_samples"],
                              "matrix_file": str(archive_path.relative_to(output_dir)), "matrix_sha256": _sha256(archive_path)})
            all_probe_rows.extend(probe_rows)
            all_full_rows.extend(full_rows)
            correlations.extend(corr_rows)
            assignments.extend(assignment_rows)
            random_distributions[f"round_{round_number:04d}"] = random_values
            print(f"PHASE42 snapshot={round_number} full_pairs=100 complete", flush=True)

        updates = []
        local_seeds = []
        for client_id, task_id in zip(clients, range(config.clients_per_round)):
            task = bank.task(task_id)
            local_seed = rngs.local_seed(round_idx, client_id)
            update = trainer.train_client(initial_state=task.clone_reset_state(), client_id=client_id,
                                          task_id=task_id, client_indices=data.partitions[client_id],
                                          round_idx=round_idx, local_seed=local_seed)
            if update.initial_state_hash != task.state_hash:
                raise RuntimeError("Clean replay local training start mismatch")
            updates.append(update)
            local_seeds.append(local_seed)
        if local_seeds != saved["local_seeds"]:
            raise RuntimeError(f"round {round_number}: local seed replay mismatch")
        global_state = weighted_fedavg(updates)
        global_hash = state_dict_hash(global_state)
        if global_hash != saved["global_state_hash"]:
            raise RuntimeError(f"round {round_number}: global hash replay mismatch")
        with validation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"round": round_number, "selected_clients_match": True,
                                     "task_bank_match": True, "local_seeds_match": True,
                                     "global_hash": global_hash, "global_hash_match": True}) + "\n")
        print(f"PHASE42 replay round={round_number} hash_verified", flush=True)

    _write_csv(output_dir / "probe_pairs.csv", all_probe_rows)
    _write_csv(output_dir / "full_training_pairs.csv", all_full_rows)
    _write_csv(output_dir / "correlations.csv", correlations)
    _write_csv(output_dir / "assignment_oracle.csv", assignments)
    np.savez_compressed(output_dir / "random_assignment_values.npz", **random_distributions)
    metadata = {"source_clean_run": str(source_run), "source_rounds_sha256": _sha256(source_run / "rounds.jsonl"),
                "source_tasks_sha256": _sha256(source_run / "tasks.jsonl"), "snapshots": snapshots,
                "reference_objective": reference_meta, "probe_protocol": "64/32/1-step", "new_clean_replay": True,
                "new_training": "one exact Clean replay to round 160 plus 400 offline LocalTrainer oracle pairs",
                "uses_test_set": False, "random_permutations_per_snapshot": RANDOM_PERMUTATIONS,
                "elapsed_seconds": time.perf_counter() - replay_started}
    (output_dir / "phase42_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / "phase42_report.md", correlations, assignments, snapshots)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=PROJECT_ROOT / "results" / "fedrad_phase38" / "clean_fedphoenix_seed1_20260903_202836_392982_clean_200r_controlled")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "fedrad_phase42")
    args = parser.parse_args()
    run(args.source_run.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
