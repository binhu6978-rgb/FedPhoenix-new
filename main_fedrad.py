#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys

# Optional read-only dependency fallback for hosts whose Conda environment
# delegates scientific packages to an unavailable/incomplete user site.
_FALLBACK_SITE = os.environ.get("FEDRAD_FALLBACK_SITE_PACKAGES")
_FALLBACK_DLLS = os.environ.get("FEDRAD_FALLBACK_DLL_DIRS", "")
if _FALLBACK_SITE:
    sys.path.insert(0, _FALLBACK_SITE)
if os.name == "nt":
    for _directory in filter(None, _FALLBACK_DLLS.split(os.pathsep)):
        os.add_dll_directory(_directory)

import numpy as np
import torch
import torchvision

from fedrad.config import FedRADConfig
from fedrad.data import load_cifar10
from fedrad.logger import RunLogger
from fedrad.models import build_model
from fedrad.rng import RNGStreams
from fedrad.trainer import CleanFedPhoenixTrainer, FedRADTrainer


PROJECT_ROOT = Path(__file__).resolve().parent


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_round_numbers(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean FedPhoenix baseline and Phase-2 FedRAD"
    )
    parser.add_argument(
        "--algorithm", choices=("clean", "fedrad", "ours"), default="clean"
    )
    parser.add_argument(
        "--diagnostic-probe-rounds",
        default="",
        help="comma-separated 1-based rounds for Phase-3 probe replicates",
    )
    parser.add_argument("--diagnostic-probe-replicates", type=int, default=1)
    parser.add_argument("--diagnostic-save-checkpoints", action="store_true")
    parser.add_argument(
        "--probe-support-size",
        type=int,
        default=None,
        help="override development support size; pass 32 to replay historical 32/32 runs",
    )
    parser.add_argument(
        "--probe-query-size",
        type=int,
        default=None,
        help="override development query size; historical value is 32",
    )
    parser.add_argument(
        "--functional-probe-replicates",
        type=int,
        default=1,
        help="independent raw-G probes averaged before Functional Recovery assignment",
    )
    parser.add_argument(
        "--functional-reliability-mode",
        choices=("mean", "half_se_lcb", "one_se_lcb"),
        default="mean",
        help="fixed reliability rule for Functional Recovery raw-G probes",
    )
    parser.add_argument(
        "--functional-assignment-mode",
        choices=(
            "hungarian",
            "consensus_lock",
            "union_restrict",
            "bilateral_gain",
        ),
        default="hungarian",
        help="assignment-level structural use of two Functional Recovery probes",
    )
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument(
        "--reset-ratio",
        type=float,
        default=None,
        help="FedPhoenix reset ratio; default is the verified AutoRun main setting (2/64)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--partition-path",
        type=Path,
        default=Path("data/cifar10_100_noniidCase5_beta0.3.json"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--regenerate-partition", action="store_true")
    parser.add_argument("--no-task-replay-check", action="store_true")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gate-tau", type=float, default=0.25)
    parser.add_argument(
        "--development-force-hungarian",
        action="store_true",
        help="debug/smoke only; bypasses the gate and is not the paper method",
    )
    parser.add_argument(
        "--matching-start-round",
        type=int,
        default=1,
        help="1-based first round that executes Probe and Hungarian matching",
    )
    parser.add_argument(
        "--score-mode", choices=("full", "g_only", "functional"), default="full"
    )
    parser.add_argument(
        "--warmup-reference-rounds",
        type=Path,
        default=None,
        help="Clean rounds.jsonl used for strict delayed-matching replay assertions",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> int:
    cli = parse_args()
    defaults = FedRADConfig()
    default_output = Path(
        "results/fedrad_phase2" if cli.algorithm == "fedrad" else "results/fedrad_phase1"
    )
    config = replace(
        defaults,
        rounds=cli.rounds,
        reset_ratio=(defaults.reset_ratio if cli.reset_ratio is None else cli.reset_ratio),
        seed=cli.seed,
        device=cli.device,
        num_workers=cli.num_workers,
        eval_every=cli.eval_every,
        data_root=_absolute(cli.data_root),
        partition_path=_absolute(cli.partition_path),
        output_root=_absolute(cli.output_root or default_output),
        download=not cli.no_download,
        regenerate_partition=cli.regenerate_partition,
        verify_task_replay=not cli.no_task_replay_check,
        gate_tau=cli.gate_tau,
        development_force_hungarian=cli.development_force_hungarian,
        matching_start_round=cli.matching_start_round,
        score_mode=("functional" if cli.algorithm == "ours" else cli.score_mode),
        warmup_reference_rounds_path=(
            None
            if cli.warmup_reference_rounds is None
            else _absolute(cli.warmup_reference_rounds)
        ),
        diagnostic_probe_rounds=_parse_round_numbers(
            cli.diagnostic_probe_rounds
        ),
        diagnostic_probe_replicates=cli.diagnostic_probe_replicates,
        diagnostic_save_checkpoints=cli.diagnostic_save_checkpoints,
        functional_probe_replicates=cli.functional_probe_replicates,
        functional_reliability_mode=cli.functional_reliability_mode,
        functional_assignment_mode=cli.functional_assignment_mode,
        probe_support_size=(
            defaults.probe_support_size
            if cli.probe_support_size is None
            else cli.probe_support_size
        ),
        probe_query_size=(
            defaults.probe_query_size
            if cli.probe_query_size is None
            else cli.probe_query_size
        ),
    )
    config.validate()
    configure_determinism(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    rngs = RNGStreams(config.seed)
    data = load_cifar10(config, rngs)
    model = build_model(config, rngs.model_seed)
    logger = RunLogger(
        config,
        cli.run_name or None,
        algorithm=("functional_recovery" if cli.algorithm == "ours" else ("fedrad" if cli.algorithm == "fedrad" else "clean_fedphoenix")),
    )
    (logger.run_dir / "runtime_environment.json").write_text(
        json.dumps(
            {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "numpy_file": np.__file__,
                "torch_version": torch.__version__,
                "torch_file": torch.__file__,
                "torchvision_version": torchvision.__version__,
                "torchvision_file": torchvision.__file__,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "fallback_site_packages": _FALLBACK_SITE,
                "fallback_dll_dirs": _FALLBACK_DLLS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Run directory: {logger.run_dir}", flush=True)
    print(
        f"Partition fingerprint: {data.partition_fingerprint}; "
        f"device={device}",
        flush=True,
    )
    trainer_class = FedRADTrainer if cli.algorithm in {"fedrad", "ours"} else CleanFedPhoenixTrainer
    trainer = trainer_class(
        config=config,
        data=data,
        model_template=model,
        rngs=rngs,
        device=device,
        logger=logger,
    )
    result = trainer.run()
    torch.save(result.clone_final_state(), logger.run_dir / "final_state.pt")
    print(f"FINAL_STATE_HASH {result.final_state_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
