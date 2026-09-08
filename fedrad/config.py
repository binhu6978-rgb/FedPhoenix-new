from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FedRADConfig:
    """Configuration shared by clean FedPhoenix and the Phase-2 FedRAD path."""

    dataset: str = "cifar10"
    model: str = "resnet18"
    num_classes: int = 10
    num_users: int = 100
    clients_per_round: int = 10
    rounds: int = 1200
    dirichlet_beta: float = 0.3
    min_client_samples: int = 10
    seed: int = 1

    local_epochs: int = 5
    local_batch_size: int = 50
    learning_rate: float = 0.01
    momentum: float = 0.5
    weight_decay: float = 0.0
    num_workers: int = 0

    # Faithful original FedPhoenix main-experiment setting (AutoRun.py).
    reset_ratio: float = 2.0 / 64.0
    fp_conv_rounds: int = 1000
    reset_method: str = "ori_normal"

    eval_batch_size: int = 256
    eval_every: int = 1
    device: str = "auto"
    deterministic: bool = True

    data_root: Path = Path("data")
    partition_path: Path = Path(
        "data/cifar10_100_noniidCase5_beta0.3.json"
    )
    output_root: Path = Path("results/fedrad_phase1")
    download: bool = True
    regenerate_partition: bool = False
    verify_task_replay: bool = True

    # Phase-2 probe defaults. These values are frozen for development smoke
    # tests; they do not change the Phase-1 clean baseline path.
    probe_support_size: int = 64
    probe_query_size: int = 32
    probe_steps: int = 1
    probe_learning_rate: float = 0.01
    # How the Functional Recovery utility summarizes losses observed along
    # the fixed-support adaptation path. ``terminal`` preserves M2 exactly.
    probe_recovery_measurement: str = "terminal"
    probe_query_mode: str = "legacy"
    # Number of independent Functional Recovery estimates averaged before
    # Hungarian assignment.  One exactly preserves the accepted Ours path.
    functional_probe_replicates: int = 1
    # Reliability rule applied to the independent raw-G estimates. ``mean``
    # exactly preserves Multi-Probe M2. The LCB variants are fixed,
    # interpretable candidates rather than a free lambda sweep.
    functional_reliability_mode: str = "mean"
    # Assignment-level use of the two independent Functional Recovery probes.
    # ``hungarian`` exactly preserves the accepted M2/LCB paths.
    functional_assignment_mode: str = "hungarian"

    # Development scoring defaults, not final calibrated values.
    lambda_G: float = 1.0
    lambda_A: float = 1.0
    lambda_D: float = 1.0
    lambda_C: float = 0.25
    scoring_weights_status: str = (
        "development defaults; not final calibrated values"
    )
    z_eps: float = 1e-12
    std_atol: float = 1e-12
    std_rtol: float = 1e-7
    norm_eps: float = 1e-12

    # Positive conservative placeholder. Formal calibration is intentionally
    # deferred to a later phase and must not use test accuracy here.
    gate_tau: float = 0.25
    gate_tau_status: str = "development placeholder; not calibrated"
    development_force_hungarian: bool = False

    # Phase-3.9 timing/score controls. Round numbers are 1-based. A delayed
    # run must replay and assert against a completed Clean rounds.jsonl file.
    matching_start_round: int = 1
    score_mode: str = "full"
    warmup_reference_rounds_path: Path | None = None

    # Optional Phase-3 diagnostics. Empty/default values add no work to the
    # accepted Phase-2 path.
    diagnostic_probe_rounds: tuple[int, ...] = ()
    diagnostic_probe_replicates: int = 1
    diagnostic_save_checkpoints: bool = False

    @property
    def probe_protocol(self) -> str:
        protocol = (
            f"support{self.probe_support_size}_query{self.probe_query_size}"
            f"_steps{self.probe_steps}"
        )
        if self.probe_recovery_measurement != "terminal":
            protocol += f"_{self.probe_recovery_measurement}"
        if self.probe_query_mode != "legacy":
            protocol += f"_{self.probe_query_mode}"
        return protocol

    def validate(self) -> None:
        if self.dataset != "cifar10" or self.model != "resnet18":
            raise ValueError("Phase 1 only supports CIFAR-10 with ResNet18")
        if self.num_classes != 10:
            raise ValueError("CIFAR-10 requires num_classes=10")
        if self.num_users < 1:
            raise ValueError("num_users must be positive")
        if not 1 <= self.clients_per_round <= self.num_users:
            raise ValueError("clients_per_round must be in [1, num_users]")
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.dirichlet_beta <= 0:
            raise ValueError("dirichlet_beta must be positive")
        if self.min_client_samples < 1:
            raise ValueError("min_client_samples must be positive")
        if self.local_epochs < 1 or self.local_batch_size < 1:
            raise ValueError("local training epochs and batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.momentum < 0 or self.weight_decay < 0:
            raise ValueError("momentum and weight_decay must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not 0 <= self.reset_ratio <= 1:
            raise ValueError("reset_ratio must be in [0, 1]")
        if self.fp_conv_rounds < 1:
            raise ValueError("fp_conv_rounds must be positive")
        if self.reset_method != "ori_normal":
            raise ValueError("Phase 1 fixes reset_method to ori_normal")
        if self.eval_batch_size < 1 or self.eval_every < 1:
            raise ValueError("evaluation batch size/frequency must be positive")
        if self.probe_support_size < 1 or self.probe_query_size < 1:
            raise ValueError("probe support/query sizes must be positive")
        if self.probe_steps < 1 or self.probe_learning_rate <= 0:
            raise ValueError("probe steps and learning rate must be positive")
        if self.probe_query_mode not in {"legacy", "eval_eval", "train_train"}:
            raise ValueError("unknown probe_query_mode")
        if self.probe_query_mode != "legacy" and (
            self.probe_recovery_measurement != "terminal" or self.score_mode != "functional"
        ):
            raise ValueError("query-mode experiments require terminal Functional Recovery")
        if self.probe_recovery_measurement not in {
            "terminal",
            "trajectory_mean",
            "endpoints_mean",
        }:
            raise ValueError(
                "probe_recovery_measurement must be terminal, trajectory_mean, "
                "or endpoints_mean"
            )
        if self.probe_recovery_measurement != "terminal" and self.probe_steps < 2:
            raise ValueError("trajectory recovery measurement requires probe_steps >= 2")
        if self.functional_probe_replicates < 1:
            raise ValueError("functional_probe_replicates must be positive")
        if self.functional_reliability_mode not in {
            "mean",
            "half_se_lcb",
            "one_se_lcb",
        }:
            raise ValueError(
                "functional_reliability_mode must be mean, half_se_lcb, or one_se_lcb"
            )
        if self.functional_assignment_mode not in {
            "hungarian",
            "consensus_lock",
            "union_restrict",
            "bilateral_gain",
        }:
            raise ValueError(
                "functional_assignment_mode must be hungarian, consensus_lock, "
                "union_restrict, or bilateral_gain"
            )
        if not math.isclose(
            self.probe_learning_rate,
            self.learning_rate,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("probe learning rate must equal current formal local LR")
        if min(self.lambda_G, self.lambda_A, self.lambda_D, self.lambda_C) < 0:
            raise ValueError("scoring weights must be non-negative")
        if self.z_eps <= 0 or self.norm_eps <= 0:
            raise ValueError("normalization epsilons must be positive")
        if self.std_atol < 0 or self.std_rtol < 0:
            raise ValueError("standard-deviation tolerances must be non-negative")
        if self.gate_tau <= 0:
            raise ValueError("gate_tau must be positive and conservative")
        if not 1 <= self.matching_start_round <= self.rounds + 1:
            raise ValueError("matching_start_round must lie in [1, rounds + 1]")
        if self.score_mode not in {"full", "g_only", "functional"}:
            raise ValueError("score_mode must be full, g_only, or functional")
        if self.functional_probe_replicates > 1 and self.score_mode != "functional":
            raise ValueError(
                "multiple functional probes require score_mode=functional"
            )
        if self.functional_reliability_mode != "mean":
            if self.score_mode != "functional":
                raise ValueError(
                    "functional reliability requires score_mode=functional"
                )
            if self.functional_probe_replicates != 2:
                raise ValueError(
                    "functional reliability variants require exactly two probes"
                )
        if self.functional_assignment_mode != "hungarian":
            if self.score_mode != "functional":
                raise ValueError(
                    "functional structural assignment requires score_mode=functional"
                )
            if self.functional_probe_replicates != 2:
                raise ValueError(
                    "functional structural assignment requires exactly two probes"
                )
            if self.functional_reliability_mode != "mean":
                raise ValueError(
                    "functional structural assignment requires the M2 mean score"
                )
        if self.matching_start_round > 1:
            if self.warmup_reference_rounds_path is None:
                raise ValueError(
                    "delayed matching requires warmup_reference_rounds_path"
                )
            if not self.warmup_reference_rounds_path.is_file():
                raise ValueError("warmup reference rounds file does not exist")
        if self.diagnostic_probe_replicates < 1:
            raise ValueError("diagnostic_probe_replicates must be positive")
        if len(set(self.diagnostic_probe_rounds)) != len(
            self.diagnostic_probe_rounds
        ):
            raise ValueError("diagnostic probe rounds must be unique")
        if any(
            round_number < 1 or round_number > self.rounds
            for round_number in self.diagnostic_probe_rounds
        ):
            raise ValueError("diagnostic probe rounds must lie within the run")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")

    def as_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["probe_protocol"] = self.probe_protocol
        return payload
