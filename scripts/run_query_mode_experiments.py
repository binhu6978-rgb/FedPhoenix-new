"""Two simultaneous trainers; block on exits without polling training logs."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--support-coverage", action="store_true", help="run fresh and refresh_once with legacy queries")
    args = parser.parse_args()
    rounds = 1 if args.smoke else 200
    experiment = "support_coverage" if args.support_coverage else "query_mode"
    root = PROJECT / "results" / (f"{experiment}_smoke" if args.smoke else f"{experiment}_functional_recovery_2of64") / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    children = []
    modes = ("fresh", "refresh_once") if args.support_coverage else ("eval_eval", "train_train")
    option = "--probe-support-coverage" if args.support_coverage else "--probe-query-mode"
    for mode in modes:
        output = root / mode
        output.mkdir()
        stdout = (output / "training.log").open("w", encoding="utf-8")
        stderr = (output / "training.err.log").open("w", encoding="utf-8")
        command = [sys.executable, str(PROJECT / "main_fedrad.py"),
                   "--algorithm", "ours", "--rounds", str(rounds), "--seed", "1",
                   "--device", "cuda:0", "--num-workers", "0", "--eval-every", "1",
                   "--no-download", "--data-root", str(PROJECT.parent / "FedPhoenix-main" / "data"),
                   "--partition-path", str(PROJECT / "data/cifar10_100_noniidCase5_beta0.3.json"),
                   "--output-root", str(output), "--reset-ratio", "0.03125",
                   "--functional-probe-replicates", "2", "--probe-steps", "5",
                   option, mode, "--run-name", f"m2_steps5_{mode}_seed1_{rounds}r"]
        process = subprocess.Popen(command, cwd=PROJECT, stdout=stdout, stderr=stderr)
        children.append((mode, process, stdout, stderr, output))
        print(f"START {mode} pid={process.pid}", flush=True)
    results = []
    for mode, process, stdout, stderr, output in children:
        code = process.wait()
        stdout.close()
        stderr.close()
        print(f"DONE {mode} exit={code}", flush=True)
        summaries = list(output.glob("*/summary.json"))
        if code or len(summaries) != 1:
            results.append({"mode": mode, "status": "failed", "exit_code": code})
            continue
        run = summaries[0].parent
        summary = json.loads(summaries[0].read_text())
        rows = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines() if line]
        if summary.get("status") != "complete" or len(rows) != rounds:
            raise RuntimeError(f"Incomplete run: {run}")
        values = [row["diagnostic_accuracy"] for row in rows]
        if args.support_coverage:
            references = list((PROJECT / "results/probe_horizon_functional_recovery_2of64").glob("*m2_steps5_seed1_2of64_200r/rounds.jsonl"))
            if len(references) != 1:
                raise RuntimeError("Expected one authoritative 5-step reference")
            reference = [json.loads(line) for line in references[0].read_text().splitlines() if line]
            for row, ref in zip(rows, reference[:rounds], strict=True):
                for field in ("selected_clients", "task_seeds", "local_seeds"):
                    if row[field] != ref[field]:
                        raise RuntimeError(f"Fairness mismatch: {mode} {field}")
            coverage = [json.loads(line) for line in (run / "support_coverage.jsonl").read_text().splitlines() if line]
            if len(coverage) != rounds * 10 * 2:
                raise RuntimeError("Incomplete support coverage audit")
            for row in coverage:
                if set(row["query_indices"]) & {i for batch in row["support_batch_indices"] for i in batch}:
                    raise RuntimeError("Support/query overlap in coverage audit")
            print(f"FAIRNESS_PASS {mode}; mean_unique_support={mean(row['unique_support_count'] for row in coverage):.3f}", flush=True)
        peak = max(values)
        results.append(dict(mode=mode, status="complete", rounds=len(rows), peak=peak,
                            peak_round=values.index(peak)+1, final=values[-1],
                            last20=mean(values[-20:]), last50=mean(values[-50:]),
                            delta_vs_7470=peak-74.70, run_directory=str(run)))
    (root / "result_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any(row["status"] != "complete" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
