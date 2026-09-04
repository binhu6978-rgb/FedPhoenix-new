from __future__ import annotations

import ast
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "results" / "fedrad_phase37" / "analysis_seed1"


def _csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    for relative in (
        "scripts/analyze_phase35_cross_probe.py",
        "scripts/analyze_phase37.py",
        "scripts/run_regression_tests.py",
    ):
        path = PROJECT_ROOT / relative
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    report = json.loads(
        (ANALYSIS_DIR / "phase37_analysis.json").read_text(encoding="utf-8")
    )
    assert report["accuracy"]["clean"]["peak"] == 48.46
    assert report["accuracy"]["fedrad"]["peak"] == 48.72
    assert report["accuracy"]["delta"]["peak"] > 0.25
    assert report["heldout"]["heldout_observations"] == 10
    assert report["gate_identifiability"]["verdict"] == "Gamma not predictive"
    assert report["final_conclusion"] == "matching useful but gate invalid"
    assert _csv_rows(ANALYSIS_DIR / "accuracy_curves.csv") == 40
    assert _csv_rows(ANALYSIS_DIR / "dynamics_64_32.csv") == 40
    assert _csv_rows(ANALYSIS_DIR / "primary_to_heldout.csv") == 10
    markdown = (ANALYSIS_DIR / "phase37_report.md").read_text(encoding="utf-8")
    assert "48.7200 (r36)" in markdown
    assert "Phase 4 was not started" in markdown
    print("Phase 3.7 artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
