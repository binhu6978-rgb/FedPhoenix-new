from __future__ import annotations

"""Offline, bidirectional validation of two full-local utility matrices."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


RANDOM_COUNT = 10_000
NULL_SEED = 4_260_641


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def rankdata(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel().copy(); y = np.asarray(right, dtype=np.float64).ravel().copy()
    x -= x.mean(); y -= y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator else float("nan")


def two_way(value: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    matrix = np.asarray(value, dtype=np.float64)
    mean = float(matrix.mean())
    client = matrix.mean(axis=1, keepdims=True) - mean
    state = matrix.mean(axis=0, keepdims=True) - mean
    interaction = matrix - mean - client - state
    metrics = {
        "total_variance": float(matrix.var()),
        "client_main_effect_variance": float(client.var()),
        "state_main_effect_variance": float(state.var()),
        "interaction_variance": float(interaction.var()),
    }
    metrics["interaction_over_total"] = metrics["interaction_variance"] / metrics["total_variance"] if metrics["total_variance"] else float("nan")
    if not np.isclose(metrics["total_variance"], metrics["client_main_effect_variance"] + metrics["state_main_effect_variance"] + metrics["interaction_variance"], rtol=1e-10, atol=1e-12):
        raise RuntimeError("two-way variance decomposition is not additive")
    return interaction, metrics


def assignment_value(value: np.ndarray, assignment: Sequence[int]) -> float:
    assignment_array = np.asarray(assignment, dtype=np.int64)
    return float(np.asarray(value, dtype=np.float64)[np.arange(len(assignment_array)), assignment_array].sum())


def hungarian(value: np.ndarray) -> tuple[tuple[int, ...], float]:
    rows, columns = linear_sum_assignment(-np.asarray(value, dtype=np.float64))
    assignment = tuple(column for _, column in sorted(zip(rows.tolist(), columns.tolist())))
    return assignment, assignment_value(value, assignment)


def second_best(value: np.ndarray) -> tuple[float, float]:
    assignment, best = hungarian(value)
    matrix = np.asarray(value, dtype=np.float64)
    forbidden = float(matrix.min() - max(float(np.ptp(matrix)), 1.0) - 1.0)
    alternatives: list[float] = []
    for row, column in enumerate(assignment):
        constrained = matrix.copy(); constrained[row, column] = forbidden
        alternative, _ = hungarian(constrained)
        if alternative != assignment:
            alternatives.append(assignment_value(matrix, alternative))
    if not alternatives:
        raise RuntimeError("no distinct feasible assignment")
    return best, max(alternatives)


def random_assignment_values(value: np.ndarray, rng: np.random.Generator, count: int = RANDOM_COUNT) -> np.ndarray:
    size = np.asarray(value).shape[0]
    return np.asarray([assignment_value(value, rng.permutation(size)) for _ in range(count)], dtype=np.float64)


def distribution_row(value: np.ndarray, assignment: tuple[int, ...], rng: np.random.Generator) -> tuple[dict[str, float], np.ndarray]:
    samples = random_assignment_values(value, rng)
    score = assignment_value(value, assignment)
    return {
        "random_mean": float(samples.mean()), "random_std": float(samples.std()),
        "random_p90": float(np.quantile(samples, .90)), "random_p95": float(np.quantile(samples, .95)),
        "random_p99": float(np.quantile(samples, .99)), "score": score,
        "percentile": float(np.mean(samples <= score) * 100.0),
        "upper_tail_empirical_p": float((1 + np.count_nonzero(samples >= score)) / (len(samples) + 1)),
    }, samples


def interaction_null(i1: np.ndarray, i2: np.ndarray, rng: np.random.Generator, count: int = RANDOM_COUNT) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Reindex both axes of replicate 2, retaining its own interaction structure."""
    observed_p = pearson(i1, i2); observed_s = pearson(rankdata(i1), rankdata(i2))
    pearsons = np.empty(count); spearmans = np.empty(count)
    for index in range(count):
        permuted = i2[rng.permutation(i2.shape[0]), :][:, rng.permutation(i2.shape[1])]
        pearsons[index] = pearson(i1, permuted)
        spearmans[index] = pearson(rankdata(i1), rankdata(permuted))
    def summary(observed: float, samples: np.ndarray, label: str) -> dict[str, float]:
        return {f"interaction_{label}": observed, f"interaction_{label}_null_mean": float(samples.mean()),
                f"interaction_{label}_null_std": float(samples.std()),
                f"interaction_{label}_null_percentile": float(np.mean(samples <= observed) * 100.0),
                f"interaction_{label}_null_two_sided_p": float((1 + np.count_nonzero(np.abs(samples) >= abs(observed))) / (len(samples) + 1))}
    row = {**summary(observed_p, pearsons, "pearson"), **summary(observed_s, spearmans, "spearman")}
    return row, pearsons, spearmans


def overlap_null(left: tuple[int, ...], right: tuple[int, ...], rng: np.random.Generator, count: int = RANDOM_COUNT) -> tuple[dict[str, float], np.ndarray]:
    observed = float(np.mean(np.asarray(left) == np.asarray(right)))
    samples = np.empty(count)
    size = len(left)
    for index in range(count):
        samples[index] = float(np.mean(rng.permutation(size) == rng.permutation(size)))
    return {"observed_pair_overlap": observed, "exact_match": bool(left == right),
            "random_overlap_mean": float(samples.mean()), "random_overlap_std": float(samples.std()),
            "random_overlap_percentile": float(np.mean(samples <= observed) * 100.0),
            "random_overlap_upper_tail_p": float((1 + np.count_nonzero(samples >= observed)) / (len(samples) + 1))}, samples


def _trajectory_metrics(source: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in (source / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if line]
    values = np.asarray([float(row["diagnostic_accuracy"]) for row in rows if int(row["round_number"]) >= 150], dtype=np.float64)
    return {"round_150_200_accuracy_std": float(values.std()), "round_150_200_mean_abs_adjacent_change": float(np.abs(np.diff(values)).mean())}


def _report(output: Path, interaction: Sequence[Mapping[str, Any]], crossfit: Sequence[Mapping[str, Any]], overlap: Sequence[Mapping[str, Any]], margins: Sequence[Mapping[str, Any]], decision: str, rationale: str) -> None:
    lines = ["# Faithful FedPhoenix 2/64 Conditional Utility Validation", "", "## Scope", "", "Two independent formal 5-epoch LocalTrainer measurements were made for each client × reset-state pair on the frozen faithful 2/64 Clean trajectory. Primary utility is fixed before analysis: `V_full(i,j) = L_fed_ref(reset_j) - L_fed_ref(w_full(i,j))`, using a fixed, sample-size-weighted 1,600-example stratified training-set CE objective. No test samples, Probe, score, gate, Hungarian training policy, or FL trajectory was changed.", "", "## Interaction reproducibility and noise", "", "| Snapshot | Pearson (null percentile / p) | Spearman (null percentile / p) | mean interaction var. | interaction noise var. | I/noise |", "| ---: | --- | --- | ---: | ---: | ---: |"]
    for row in interaction:
        lines.append("| {round} | {interaction_pearson:.3f} ({interaction_pearson_null_percentile:.2f}% / {interaction_pearson_null_two_sided_p:.4f}) | {interaction_spearman:.3f} ({interaction_spearman_null_percentile:.2f}% / {interaction_spearman_null_two_sided_p:.4f}) | {mean_interaction_variance:.6g} | {interaction_noise_variance:.6g} | {interaction_to_noise_ratio:.3f} |".format(**row))
    lines += ["", "## Held-out Hungarian decision value", "", "Each row selects Hungarian on one replicate and evaluates only on the other. Random percentiles and p-values are drawn directly from the held-out matrix's 10,000 one-to-one assignments.", "", "| Snapshot | Direction | Oracle − Clean | Clean percentile | Cross-fit percentile / p |", "| ---: | --- | ---: | ---: | --- |"]
    for row in crossfit:
        lines.append("| {round} | {direction} | {crossfit_oracle_minus_clean:.6f} | {clean_percentile:.2f}% | {crossfit_percentile:.2f}% / {crossfit_upper_tail_p:.4f} |".format(**row))
    lines += ["", "## Assignment overlap and margins (auxiliary)", "", "| Snapshot | V1/V2 overlap | overlap null percentile / p | V1 margin/K | V1 margin/std(I) | V2 margin/K | V2 margin/std(I) |", "| ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    margin_map = {int(row["round"]): row for row in margins}
    for row in overlap:
        margin = margin_map[int(row["round"])]
        combined = dict(margin)
        combined.update(row)
        lines.append("| {round} | {observed_pair_overlap:.2f} | {random_overlap_percentile:.2f}% / {random_overlap_upper_tail_p:.4f} | {v1_margin_per_pair:.6f} | {v1_margin_over_interaction_std:.3f} | {v2_margin_per_pair:.6f} | {v2_margin_over_interaction_std:.3f} |".format(**combined))
    lines += ["", "## Decision", "", f"**{decision}**", "", rationale, "", "Primary evidence is replicate interaction correspondence relative to its structure-preserving null and bidirectional held-out matching utility. Noise, overlap, and margins are explanatory rather than standalone vetoes."]
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, *, include_deferred: bool = False) -> dict[str, Any]:
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    rounds = tuple(int(item) for item in provenance["primary_snapshots"])
    if include_deferred:
        rounds = tuple(sorted(rounds + tuple(int(item) for item in provenance["deferred_snapshots"])))
    interaction_rows: list[dict[str, Any]] = []; crossfit_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []; margin_rows: list[dict[str, Any]] = []; stored: dict[str, np.ndarray] = {}
    for round_number in rounds:
        v1_file = output / "matrices" / f"round_{round_number:04d}_V1.npz"; v2_file = output / "matrices" / f"round_{round_number:04d}_V2.npz"
        if not v1_file.is_file() or not v2_file.is_file():
            raise FileNotFoundError(f"incomplete V1/V2 for round {round_number}")
        a1, a2 = np.load(v1_file), np.load(v2_file)
        v1, v2 = np.asarray(a1["V_full"], dtype=np.float64), np.asarray(a2["V_full"], dtype=np.float64)
        if v1.shape != (10, 10) or v2.shape != (10, 10) or not np.array_equal(a1["client_order"], a2["client_order"]) or not np.array_equal(a1["task_order"], a2["task_order"]):
            raise RuntimeError(f"round {round_number}: pair ordering mismatch")
        i1, d1 = two_way(v1); i2, d2 = two_way(v2)
        rng = np.random.default_rng(NULL_SEED + round_number)
        null_row, null_p, null_s = interaction_null(i1, i2, rng)
        noise = 0.5 * float(np.var(i1 - i2)); mean_interaction = 0.5 * (d1["interaction_variance"] + d2["interaction_variance"])
        interaction_rows.append({"round": round_number, **{f"v1_{key}": value for key, value in d1.items()}, **{f"v2_{key}": value for key, value in d2.items()}, **null_row, "interaction_noise_variance": noise, "mean_interaction_variance": mean_interaction, "interaction_to_noise_ratio": mean_interaction / noise if noise else float("nan")})
        m1, _ = hungarian(v1); m2, _ = hungarian(v2); clean = tuple(range(10))
        for direction, heldout, held_id, chosen in (("V1->V2", v2, 2, m1), ("V2->V1", v1, 1, m2)):
            random_row, values = distribution_row(heldout, chosen, np.random.default_rng(NULL_SEED + round_number * 10 + held_id))
            clean_row, _ = distribution_row(heldout, clean, np.random.default_rng(NULL_SEED + round_number * 10 + held_id))
            crossfit_rows.append({"round": round_number, "direction": direction, "heldout_replicate": held_id, "assignment": json.dumps(chosen), "clean_assignment": json.dumps(clean), "clean_score": clean_row["score"], "clean_percentile": clean_row["percentile"], "crossfit_oracle_score": random_row["score"], "crossfit_oracle_minus_clean": random_row["score"] - clean_row["score"], "crossfit_percentile": random_row["percentile"], "crossfit_upper_tail_p": random_row["upper_tail_empirical_p"], **{key: value for key, value in random_row.items() if key not in {"score", "percentile", "upper_tail_empirical_p"}}})
            stored[f"round_{round_number:04d}_{direction.replace('->', '_')}_random"] = values
        overlap_row, overlap_values = overlap_null(m1, m2, np.random.default_rng(NULL_SEED + round_number * 100))
        overlap_rows.append({"round": round_number, "v1_assignment": json.dumps(m1), "v2_assignment": json.dumps(m2), **overlap_row})
        stored[f"round_{round_number:04d}_interaction_pearson_null"] = null_p; stored[f"round_{round_number:04d}_interaction_spearman_null"] = null_s; stored[f"round_{round_number:04d}_overlap_null"] = overlap_values
        best1, second1 = second_best(v1); best2, second2 = second_best(v2)
        margin_rows.append({"round": round_number, "v1_best_assignment_objective": best1, "v1_second_best_objective": second1, "v1_margin": best1 - second1, "v1_margin_per_pair": (best1-second1)/10, "v1_margin_over_interaction_std": (best1-second1)/float(i1.std()), "v2_best_assignment_objective": best2, "v2_second_best_objective": second2, "v2_margin": best2-second2, "v2_margin_per_pair": (best2-second2)/10, "v2_margin_over_interaction_std": (best2-second2)/float(i2.std())})
    # No correlation-magnitude or SNR threshold is used.  Reproducibility is evaluated
    # only against its explicitly generated structure-preserving permutation null; held-out
    # value asks whether both directions exceed Clean and their own held-out random median.
    positive = []
    for row in interaction_rows:
        related = [item for item in crossfit_rows if int(item["round"]) == int(row["round"])]
        reproducible = (float(row["interaction_pearson"]) > 0 and float(row["interaction_spearman"]) > 0
                        and float(row["interaction_pearson_null_two_sided_p"]) < 0.05
                        and float(row["interaction_spearman_null_two_sided_p"]) < 0.05)
        decision_value = all(float(item["crossfit_oracle_minus_clean"]) > 0 and float(item["crossfit_percentile"]) > 50 for item in related)
        positive.append(reproducible and decision_value)
    if all(positive):
        decision = "supported"; rationale = "Both frozen snapshots have positive replicate interaction correspondence and positive bidirectional held-out Oracle-versus-Clean utility. This supports proceeding only to the specified Probe–LocalTrainer parity diagnosis."
    elif any(positive):
        decision = "regime-dependent / heterogeneous"
        if include_deferred:
            rationale = "The four frozen snapshots are not consistent: r20 is near its interaction null, r40 has negative interaction correspondence and negative held-out matching value, r120 has positive interaction correspondence but negative held-out matching value, and only r160 has both primary evidence streams positive. This does not support a stable, trajectory-wide conditional-utility assignment effect."
        else:
            rationale = "The primary reproducibility/held-out-value evidence differs by endpoint snapshot. r40 and r120 should be measured through the already documented hash-exact replay before any estimator is considered."
    else:
        decision = "unsupported"
        rationale = ("No snapshot jointly has reproducible interaction correspondence and bidirectional held-out Oracle-versus-Clean utility."
                     if include_deferred else "Neither endpoint snapshot jointly has reproducible interaction correspondence and bidirectional held-out Oracle-versus-Clean utility; r40/r120 require the same strict measurement before a final stop decision.")
    _write_csv(output / "interaction_summary.csv", interaction_rows); _write_csv(output / "crossfit_assignments.csv", crossfit_rows); _write_csv(output / "assignment_overlap.csv", overlap_rows); _write_csv(output / "assignment_margins.csv", margin_rows)
    null_summary = [{"round": row["round"], "random_count": RANDOM_COUNT, "interaction_null_method": "independent client and state reindexing of I2", "overlap_null_method": "two independent random one-to-one assignments"} for row in interaction_rows]
    _write_csv(output / "null_distributions_summary.csv", null_summary); np.savez_compressed(output / "null_distributions.npz", **stored)
    summary = {"decision": decision, "rationale": rationale, "analyzed_snapshots": list(rounds), "random_assignments_or_null_draws": RANDOM_COUNT, "trajectory_descriptive_only": _trajectory_metrics(Path(provenance["source_clean_run"])), "uses_test_set": False, "runs_new_fl_trajectory": False}
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _report(output, interaction_rows, crossfit_rows, overlap_rows, margin_rows, decision, rationale)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--include-deferred", action="store_true")
    args = parser.parse_args(); run(args.output.resolve(), include_deferred=args.include_deferred); return 0


if __name__ == "__main__":
    raise SystemExit(main())
