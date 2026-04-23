"""
report.py — Prints and saves a human-readable backtest summary.
"""

from typing import Dict


def print_report(metrics: Dict, run_label: str = "") -> None:
    header = f"=== Backtest Report {run_label} ===".strip()
    print(header)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")


def save_report(metrics: Dict, path: str) -> None:
    import json
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Report saved → {path}")
