from __future__ import annotations

"""Offline Phase 4.1 analysis of client-by-transient-state utility.

This script intentionally consumes only the saved Phase 3.6 P2 (64/32) Probe
replicates.  It neither constructs a model nor invokes a trainer.  The primary
quantity is G = reset_loss - adapted_loss, a direct per-client/per-reset-state
functional recovery gain.  The other saved components are retained as
secondary sensitivity analyses.
"""

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RAW_PROTOCOL = "P2_64_32"
PRIMARY_METRIC = "G"
SECONDARY_METRICS = ("A", "D", "C", "adapted_utility", "Q")
T975_DF2 = 4.302652729911275


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_replicates(raw_dir: Path) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """Load and validate the saved 64/32 matrices grouped by round/replicate."""
    loaded: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    expected_fields = {
        "client_order", "task_order", "G", "A", "D", "C", "Q",
        "global_loss", "reset_loss", "adapted_loss",
    }
    pattern = f"{RAW_PROTOCOL}_round_*_rep_*.npz"
    paths = sorted(raw_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {RAW_PROTOCOL} files under {raw_dir}")
    for path in paths:
        pieces = path.stem.split("_")
        # P2_64_32_round_0001_rep_0
        if len(pieces) != 7 or "_".join(pieces[:3]) != RAW_PROTOCOL:
            raise ValueError(f"unexpected raw filename: {path.name}")
        round_number = int(pieces[4])
        replicate = int(pieces[6])
        with np.load(path, allow_pickle=False) as archive:
            missing = expected_fields - set(archive.files)
            if missing:
                raise ValueError(f"{path.name} misses fields: {sorted(missing)}")
            payload = {
                name: np.asarray(archive[name], dtype=np.float64)
                for name in expected_fields
                if name not in {"client_order", "task_order"}
            }
            payload["client_order"] = np.asarray(archive["client_order"], dtype=np.int64)
            payload["task_order"] = np.asarray(archive["task_order"], dtype=np.int64)
        size = len(payload["client_order"])
        if size != len(payload["task_order"]) or size < 2:
            raise ValueError(f"{path.name} is not a non-trivial square client/task grid")
        if any(value.shape != (size, size) for key, value in payload.items()
               if key not in {"client_order", "task_order"}):
            raise ValueError(f"{path.name} has inconsistent matrix dimensions")
        if not all(np.isfinite(value).all() for key, value in payload.items()
                   if key not in {"client_order", "task_order"}):
            raise ValueError(f"{path.name} contains a non-finite value")
        loaded.setdefault(round_number, {})[replicate] = payload

    for round_number, replicates in loaded.items():
        if set(replicates) != {0, 1, 2}:
            raise ValueError(f"round {round_number} lacks exactly replicates 0,1,2")
        first = replicates[0]
        for replicate, payload in replicates.items():
            for order_name in ("client_order", "task_order"):
                if not np.array_equal(first[order_name], payload[order_name]):
                    raise ValueError(
                        f"round {round_number} replicate {replicate} changes {order_name}"
                    )
    return loaded


def metric_matrix(payload: dict[str, np.ndarray], metric: str) -> np.ndarray:
    if metric == "adapted_utility":
        return -np.asarray(payload["adapted_loss"], dtype=np.float64)
    if metric in {"G", "A", "D", "C", "Q"}:
        return np.asarray(payload[metric], dtype=np.float64)
    raise ValueError(metric)


def two_way_decomposition(matrix: np.ndarray) -> dict[str, Any]:
    """Balanced two-way ANOVA-style orthogonal decomposition without fitting.

    Variances use divisor I*J, so total variance is exactly the sum of the
    client, state, and interaction components up to floating-point error.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("two_way_decomposition requires a finite matrix of at least 2x2")
    grand_mean = float(values.mean())
    client_effect = values.mean(axis=1, keepdims=True) - grand_mean
    state_effect = values.mean(axis=0, keepdims=True) - grand_mean
    interaction = values - grand_mean - client_effect - state_effect
    total_variance = float(np.mean((values - grand_mean) ** 2))
    client_variance = float(np.mean(client_effect**2))
    state_variance = float(np.mean(state_effect**2))
    interaction_variance = float(np.mean(interaction**2))
    return {
        "grand_mean": grand_mean,
        "total_variance": total_variance,
        "client_main_effect_variance": client_variance,
        "state_main_effect_variance": state_variance,
        "interaction_variance": interaction_variance,
        "interaction_fraction_total": (
            interaction_variance / total_variance if total_variance > 0 else float("nan")
        ),
        "variance_reconstruction_error": float(
            total_variance - client_variance - state_variance - interaction_variance
        ),
        "interaction_matrix": interaction,
    }


def interaction_noise_variance(
    replicate_stack: np.ndarray,
) -> dict[str, float]:
    """Estimate interaction-specific Probe noise from independent replicates.

    The same two-way projection is applied to every replicate before taking
    between-replicate variance.  This directly measures noise left in the
    interaction subspace and does not incorrectly treat shared client/state
    Probe variation as independent cell noise.  Averaging R replicates divides
    the expected interaction-noise variance by R.
    """
    values = np.asarray(replicate_stack, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or min(values.shape[1:]) < 2:
        raise ValueError("need R x I x J replicate matrices")
    if not np.isfinite(values).all():
        raise ValueError("replicate stack contains non-finite values")
    repeats, clients, states = values.shape
    cell_noise_variance = float(np.var(values, axis=0, ddof=1).mean())
    interaction_stack = np.stack(
        [two_way_decomposition(values[index])["interaction_matrix"] for index in range(repeats)],
        axis=0,
    )
    interaction_noise_single = float(
        np.var(interaction_stack, axis=0, ddof=1).mean()
    )
    interaction_noise_mean = interaction_noise_single / repeats
    return {
        "raw_cell_measurement_noise_variance": cell_noise_variance,
        "interaction_measurement_noise_variance_single_replicate": interaction_noise_single,
        "expected_interaction_noise_variance_replicate_mean": interaction_noise_mean,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks, implemented locally to avoid an extra dependency."""
    vector = np.asarray(values, dtype=np.float64)
    ordering = np.argsort(vector, kind="mergesort")
    ranks = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(vector):
        end = start + 1
        while end < len(vector) and vector[ordering[end]] == vector[ordering[start]]:
            end += 1
        ranks[ordering[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    ranks_a = _rankdata(values_a)
    ranks_b = _rankdata(values_b)
    return pearson(ranks_a, ranks_b)


def pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Dependency-light Pearson correlation that avoids np.corrcoef's BLAS path."""
    left = np.asarray(values_a, dtype=np.float64).ravel()
    right = np.asarray(values_b, dtype=np.float64).ravel()
    if left.shape != right.shape or not len(left):
        raise ValueError("Pearson inputs must be equally shaped and non-empty")
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = float(
        np.sqrt(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def cross_state_rank_correlations(replicate_stack: np.ndarray) -> list[float]:
    """Spearman correlations of client rankings between every pair of states."""
    values = np.asarray(replicate_stack, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("need R x I x J replicate matrices")
    average = values.mean(axis=0)
    return [
        spearman(average[:, left], average[:, right])
        for left, right in combinations(range(average.shape[1]), 2)
    ]


def _machine_zero(values: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(values))))
    return np.finfo(np.float64).eps * scale * 16.0


def stable_order_sign(replicate_differences: np.ndarray) -> int:
    """Return reproducible ordering sign, or 0 when the order is not stable.

    An ordering must (i) have the same sign in all three independent Probe
    replicates and (ii) have a conventional two-sided 95% t interval (df=2)
    for its mean difference that excludes zero.  The only numeric zero tolerance
    is a scale-aware floating-point tolerance, not an effect-size threshold.
    """
    values = np.asarray(replicate_differences, dtype=np.float64)
    if values.ndim != 1 or len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("stable ordering requires exactly three finite differences")
    tolerance = _machine_zero(values)
    if np.all(values > tolerance):
        proposed = 1
    elif np.all(values < -tolerance):
        proposed = -1
    else:
        return 0
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation == 0.0:
        return proposed
    lower = mean - T975_DF2 * standard_deviation / np.sqrt(len(values))
    upper = mean + T975_DF2 * standard_deviation / np.sqrt(len(values))
    if proposed > 0 and lower > tolerance:
        return 1
    if proposed < 0 and upper < -tolerance:
        return -1
    return 0


def stable_reversal_summary(replicate_stack: np.ndarray) -> dict[str, int | float]:
    """Count cross-state client-rank reversals that reproduce across Probes."""
    values = np.asarray(replicate_stack, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("stable reversal analysis requires exactly three replicates")
    _, clients, states = values.shape
    candidate_count = 0
    stable_comparable_count = 0
    stable_reversal_count = 0
    for client_left, client_right in combinations(range(clients), 2):
        differences = values[:, client_left, :] - values[:, client_right, :]
        for state_left, state_right in combinations(range(states), 2):
            candidate_count += 1
            left_sign = stable_order_sign(differences[:, state_left])
            right_sign = stable_order_sign(differences[:, state_right])
            if left_sign and right_sign:
                stable_comparable_count += 1
                if left_sign != right_sign:
                    stable_reversal_count += 1
    return {
        "candidate_client_pair_state_pair_count": candidate_count,
        "stable_comparable_count": stable_comparable_count,
        "stable_reversal_count": stable_reversal_count,
        "stable_comparable_rate": stable_comparable_count / candidate_count,
        "stable_reversal_rate_all_candidates": stable_reversal_count / candidate_count,
        "stable_reversal_rate_among_stable_comparable": (
            stable_reversal_count / stable_comparable_count
            if stable_comparable_count else float("nan")
        ),
    }


def _mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()) if len(array) else float("nan")


def analyze_round(
    *,
    round_number: int,
    replicates: dict[int, dict[str, np.ndarray]],
    metric: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stack = np.stack(
        [metric_matrix(replicates[index], metric) for index in sorted(replicates)], axis=0
    )
    averaged = stack.mean(axis=0)
    decomposition = two_way_decomposition(averaged)
    noise = interaction_noise_variance(stack)
    interaction_variance = float(decomposition["interaction_variance"])
    expected_noise = noise["expected_interaction_noise_variance_replicate_mean"]
    corrected = interaction_variance - expected_noise
    pairwise_rows: list[dict[str, Any]] = []
    for left, right in combinations(range(stack.shape[0]), 2):
        interaction_left = two_way_decomposition(stack[left])["interaction_matrix"]
        interaction_right = two_way_decomposition(stack[right])["interaction_matrix"]
        pairwise_rows.append(
            {
                "round": round_number,
                "metric": metric,
                "left_replicate": left,
                "right_replicate": right,
                "interaction_pearson": float(
                    pearson(interaction_left.ravel(), interaction_right.ravel())
                ),
            }
        )
    rank_correlations = cross_state_rank_correlations(stack)
    reversal = stable_reversal_summary(stack) if metric == PRIMARY_METRIC else {}
    summary: dict[str, Any] = {
        "round": round_number,
        "metric": metric,
        "replicate_count": int(stack.shape[0]),
        "client_count": int(stack.shape[1]),
        "state_count": int(stack.shape[2]),
        **{
            key: value
            for key, value in decomposition.items()
            if key != "interaction_matrix"
        },
        **noise,
        "interaction_variance_minus_expected_noise": corrected,
        "interaction_signal_variance_nonnegative": max(0.0, corrected),
        "interaction_to_expected_noise_ratio": (
            interaction_variance / expected_noise if expected_noise > 0 else float("inf")
        ),
        "interaction_signal_to_expected_noise_ratio": (
            max(0.0, corrected) / expected_noise if expected_noise > 0 else float("inf")
        ),
        "mean_pairwise_interaction_pearson": _mean_or_nan(
            row["interaction_pearson"] for row in pairwise_rows
        ),
        "cross_state_client_rank_spearman_mean": _mean_or_nan(rank_correlations),
        "cross_state_client_rank_spearman_min": float(np.min(rank_correlations)),
        "cross_state_client_rank_spearman_max": float(np.max(rank_correlations)),
        **reversal,
    }
    return summary, pairwise_rows


def _format(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _write_report(
    path: Path,
    primary_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    source_manifest: dict[str, Any],
) -> None:
    lines = [
        "# Phase 4.1 — Client–State Interaction",
        "",
        "## Scope",
        "",
        "This is an offline mechanism analysis. It reuses only the saved Phase "
        "3.6 `P2_64_32` (support=64/query=32) Probe matrices at rounds 1, 20, "
        "and 40, with three independent Probe replicates per round. No model was "
        "loaded for training, no new Probe forward pass was run, and no 200-round "
        "trajectory was changed.",
        "",
        "The primary pairwise utility is `H_ij = G_ij = reset_loss_ij - "
        "adapted_loss_ij`: one-step functional recovery gain for client `i` on "
        "transient reset state `j`. `A`, `D`, `C`, `-adapted_loss`, and `Q` are "
        "secondary sensitivity analyses only. The source NPZ archives contain "
        "client IDs, task IDs, global/reset/adapted losses, G/A/D/C, and Q.",
        "",
        "## Estimands and noise accounting",
        "",
        "For the mean of the three independent replicate matrices in a round, we "
        "use the exact balanced two-way decomposition `H_ij = mu + a_i + b_j + "
        "e_ij`. Reported variances use divisor `I*J`, so total variance equals "
        "the client-main, state-main, and interaction variances (up to numerical "
        "roundoff).",
        "",
        "Probe noise is estimated directly in the relevant subspace: each "
        "replicate is first double-centered with the same two-way projection, "
        "then its per-cell between-replicate variance is averaged and divided "
        "by 3 for the replicate mean. This avoids treating shared client/state "
        "Probe variation as independent interaction noise and does not use an "
        "arbitrary percentage threshold. Raw cell-level replicate variance is "
        "also retained in the CSV/JSON as an auxiliary diagnostic.",
        "",
        "## Primary result: G recovery utility",
        "",
        "| Round | Total var | Client var | State var | Interaction var | Interaction / total | Expected interaction noise (mean of 3 reps) | Interaction / noise | Mean cross-replicate interaction r | Mean cross-state client rank rho | Stable reversal rate (among stable comparable) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary_rows:
        lines.append(
            "| {round} | {total} | {client} | {state} | {interaction} | "
            "{fraction} | {noise} | {ratio} | {pearson} | {rho} | {reversal} |".format(
                round=row["round"],
                total=_format(row["total_variance"]),
                client=_format(row["client_main_effect_variance"]),
                state=_format(row["state_main_effect_variance"]),
                interaction=_format(row["interaction_variance"]),
                fraction=_format(row["interaction_fraction_total"]),
                noise=_format(row["expected_interaction_noise_variance_replicate_mean"]),
                ratio=_format(row["interaction_to_expected_noise_ratio"], 2),
                pearson=_format(row["mean_pairwise_interaction_pearson"], 3),
                rho=_format(row["cross_state_client_rank_spearman_mean"], 3),
                reversal=_format(row["stable_reversal_rate_among_stable_comparable"], 3),
            )
        )
    lines.extend([
        "",
        "A stable ordering requires all three replicate client-pair differences "
        "to have the same nonzero sign **and** the conventional two-sided 95% "
        "t interval (df=2) for their mean to exclude zero. A stable reversal is "
        "an opposite stable order for the same client pair in two states. The "
        "analysis also reports its denominator, so a rate cannot hide a scarcity "
        "of stable orderings.",
        "",
        "## Stable ranking reversals for G",
        "",
        "| Round | All client-pair × state-pair candidates | Stable comparable | Stable comparable rate | Stable reversals | Reversal / all candidates | Reversal / stable comparable |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in primary_rows:
        lines.append(
            "| {round} | {candidates} | {comparable} | {comparable_rate} | "
            "{reversals} | {all_rate} | {stable_rate} |".format(
                round=row["round"],
                candidates=row["candidate_client_pair_state_pair_count"],
                comparable=row["stable_comparable_count"],
                comparable_rate=_format(row["stable_comparable_rate"], 3),
                reversals=row["stable_reversal_count"],
                all_rate=_format(row["stable_reversal_rate_all_candidates"], 3),
                stable_rate=_format(row["stable_reversal_rate_among_stable_comparable"], 3),
            )
        )
    lines.extend([
        "",
        "## Secondary decomposition sensitivity",
        "",
        "| Metric | Round | Interaction / total | Interaction / expected noise | Mean replicate interaction r | Mean cross-state rank rho |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in secondary_rows:
        lines.append(
            "| {metric} | {round} | {fraction} | {ratio} | {pearson} | {rho} |".format(
                metric=row["metric"],
                round=row["round"],
                fraction=_format(row["interaction_fraction_total"]),
                ratio=_format(row["interaction_to_expected_noise_ratio"], 2),
                pearson=_format(row["mean_pairwise_interaction_pearson"], 3),
                rho=_format(row["cross_state_client_rank_spearman_mean"], 3),
            )
        )
    primary_signal = all(
        row["interaction_variance_minus_expected_noise"] > 0 for row in primary_rows
    )
    any_reversals = any(row["stable_reversal_count"] > 0 for row in primary_rows)
    lines.extend([
        "",
        "## Answers to the Phase 4.1 questions",
        "",
        "1. **Does client × state interaction exist?** "
        + (
            "Yes in these saved rounds: the G interaction component is nonzero after "
            "removing client and state main effects."
            if any(row["interaction_variance"] > 0 for row in primary_rows)
            else "No measurable interaction component was found."
        ),
        "2. **Is it identifiable against Probe noise?** "
        + (
            "For all analysed rounds, observed interaction variance of the three-replicate "
            "mean exceeds the directly estimated expected measurement-noise variance."
            if primary_signal
            else "Not uniformly: at least one analysed round does not exceed the projected "
            "expected measurement-noise variance."
        ),
        "3. **Do stable ranking reversals exist?** "
        + (
            "Yes: at least one client-pair/state-pair reversal meets the strict three-replicate "
            "definition."
            if any_reversals
            else "No: none meets the strict three-replicate definition in these rounds."
        ),
        "4. **Does this support Q_t(i,j), rather than only Q_t(i)?** "
        + (
            "The evidence supports conditioning recovery utility on transient state in this "
            "controlled setting, because interaction remains after main effects and is not "
            "explained by the estimated Probe noise alone. It does not establish a universal "
            "or multi-seed claim."
            if primary_signal and any_reversals
            else "The evidence is incomplete: an interaction component alone does not yet "
            "demonstrate reproducible state-dependent client ranking."
        ),
        "5. **Weakest evidence.** Only three replicate Probes and three early/mid rounds "
        "are available, cells within a matrix are not independent observations, and the "
        "noise correction assumes replicate differences are measurement noise. This is a "
        "single-seed mechanism analysis, not an accuracy or generalization result.",
        "6. **Phase 4.2 decision.** "
        + (
            "The mechanism evidence is sufficient to justify considering a narrowly scoped "
            "Phase 4.2 proposal, but no Phase 4.2 experiment is run here."
            if primary_signal and any_reversals
            else "Do not advance on this evidence alone; first improve/expand the measurement "
            "evidence rather than changing training."
        ),
        "",
        "## Reproducibility",
        "",
        "The exact source archive paths and SHA-256 digests are recorded in "
        "`phase41_sources.json`; machine-readable results are in "
        "`phase41_decomposition.csv`, `phase41_interaction_reliability.csv`, "
        "`phase41_rank_correlations.csv`, `phase41_reversals.csv`, and "
        "`phase41_summary.json`.",
        "",
        f"Source archive count: {len(source_manifest['source_archives'])}.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "fedrad_phase36" / "probe_size_seed1" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "fedrad_phase41",
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_replicates(raw_dir)
    source_paths = sorted(raw_dir.glob(f"{RAW_PROTOCOL}_round_*_rep_*.npz"))
    source_manifest = {
        "analysis": "Phase 4.1 offline client-state interaction",
        "source_protocol": RAW_PROTOCOL,
        "source_archives": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
        "rounds": sorted(raw),
        "replicates_per_round": [0, 1, 2],
        "primary_quantity": "G = reset_loss - adapted_loss",
        "secondary_quantities": list(SECONDARY_METRICS),
        "new_training_or_forward_passes": False,
    }
    (output_dir / "phase41_sources.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    decomposition_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    reversal_rows: list[dict[str, Any]] = []
    for metric in (PRIMARY_METRIC, *SECONDARY_METRICS):
        for round_number, replicates in sorted(raw.items()):
            summary, pairwise = analyze_round(
                round_number=round_number, replicates=replicates, metric=metric
            )
            decomposition_rows.append(summary)
            reliability_rows.extend(pairwise)
            stack = np.stack(
                [metric_matrix(replicates[index], metric) for index in sorted(replicates)], axis=0
            )
            for state_left, state_right in combinations(range(stack.shape[2]), 2):
                rank_rows.append(
                    {
                        "round": round_number,
                        "metric": metric,
                        "task_left": int(replicates[0]["task_order"][state_left]),
                        "task_right": int(replicates[0]["task_order"][state_right]),
                        "client_rank_spearman": spearman(
                            stack.mean(axis=0)[:, state_left], stack.mean(axis=0)[:, state_right]
                        ),
                    }
                )
            if metric == PRIMARY_METRIC:
                reversal_rows.append(
                    {
                        key: value for key, value in summary.items()
                        if key.startswith("stable_") or key.startswith("candidate_") or key == "round"
                    }
                )

    _write_csv(output_dir / "phase41_decomposition.csv", decomposition_rows)
    _write_csv(output_dir / "phase41_interaction_reliability.csv", reliability_rows)
    _write_csv(output_dir / "phase41_rank_correlations.csv", rank_rows)
    _write_csv(output_dir / "phase41_reversals.csv", reversal_rows)
    primary_rows = [row for row in decomposition_rows if row["metric"] == PRIMARY_METRIC]
    secondary_rows = [row for row in decomposition_rows if row["metric"] != PRIMARY_METRIC]
    summary_payload = {
        "scope": source_manifest,
        "primary_G_rows": primary_rows,
        "secondary_rows": secondary_rows,
        "stable_reversal_definition": (
            "same nonzero sign in all three independent replicates and a two-sided "
            "95% t interval (df=2) for the mean pairwise difference excluding zero"
        ),
        "noise_estimator": (
            "mean unbiased per-cell variance across independently double-centered "
            "replicate interaction matrices, divided by replicate count"
        ),
    }
    (output_dir / "phase41_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        output_dir / "phase41_report.md",
        primary_rows=primary_rows,
        secondary_rows=secondary_rows,
        source_manifest=source_manifest,
    )
    print(f"Phase 4.1 offline analysis complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
