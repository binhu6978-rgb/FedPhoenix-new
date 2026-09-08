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
    args = parser.parse_args()
    rounds = 1 if args.smoke else 200
    root = PROJECT / "results" / ("query_mode_smoke" if args.smoke else "query_mode_functional_recovery_2of64") / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    children = []
    for mode in ("eval_eval", "train_train"):
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
                   "--probe-query-mode", mode, "--run-name", f"m2_steps5_{mode}_seed1_{rounds}r"]
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
