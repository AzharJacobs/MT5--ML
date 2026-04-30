"""
run_pipeline.py — Ordered entry point for the full trading pipeline.

Steps:
  a. Model health check  (scripts/check_model_health.py)
  b. Label logic tests   (tests/test_label_logic)
  c. Backtest            (backtest.engine --timeframe 15min ...)
  d. Final summary

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-checks
    python scripts/run_pipeline.py --timeframe 5min --cash 5000
"""

import argparse
import subprocess
import sys
import os
import time

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _run(cmd: list, label: str) -> subprocess.CompletedProcess:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=False,  # stream output directly to terminal
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - t0
    status  = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {label} ({elapsed:.1f}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ML trading pipeline")
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip model health check and label logic tests; go straight to backtest"
    )
    parser.add_argument("--timeframe",   default="15min")
    parser.add_argument("--cash",        type=float, default=1000.0)
    parser.add_argument("--start-date",  default="2024-11-08")
    parser.add_argument("--confidence",  type=float, default=0.52,
                        help="Confidence threshold (0.52 = use saved optimal_threshold)")
    args = parser.parse_args()

    python = sys.executable
    summary = {
        "model_health": None,
        "label_tests":  None,
        "backtest_pnl": None,
        "total_trades": None,
        "win_rate":     None,
    }

    # ── a. Model health check ──────────────────────────────────────────
    if not args.skip_checks:
        res_a = _run(
            [python, os.path.join(PROJECT_ROOT, "scripts", "check_model_health.py")],
            "Step a: Model health check"
        )
        summary["model_health"] = "OK" if res_a.returncode == 0 else "NEEDS RETRAINING"
        if res_a.returncode != 0:
            print("\n  !! Model health check failed — continuing anyway (use "
                  "--skip-checks to suppress)")
    else:
        summary["model_health"] = "skipped"

    # ── b. Label logic tests ───────────────────────────────────────────
    if not args.skip_checks:
        res_b = _run(
            [python, "-m", "tests.test_label_logic"],
            "Step b: Label logic tests"
        )
        summary["label_tests"] = "passed" if res_b.returncode == 0 else "FAILED"
        if res_b.returncode != 0:
            print("\n  !! Label logic tests failed — check signal_generator.py")
    else:
        summary["label_tests"] = "skipped"

    # ── c. Backtest ────────────────────────────────────────────────────
    backtest_cmd = [
        python, "-m", "backtest.engine",
        "--timeframe",   args.timeframe,
        "--cash",        str(args.cash),
        "--start-date",  args.start_date,
        "--confidence",  str(args.confidence),
    ]
    res_c = _run(backtest_cmd, f"Step c: Backtest ({args.timeframe}, cash={args.cash:,.0f})")
    if res_c.returncode != 0:
        print("\n  !! Backtest exited with error — check logs above")

    # ── d. Final summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Model health  : {summary['model_health']}")
    print(f"  Label tests   : {summary['label_tests']}")
    print(f"  Backtest      : {'completed' if res_c.returncode == 0 else 'FAILED'}")
    print(f"  Timeframe     : {args.timeframe}")
    print(f"  Start date    : {args.start_date}")
    print(f"  Starting cash : {args.cash:,.2f}")
    print()
    print("  Note: detailed PnL / trade count / win rate are printed above")
    print("        in the backtest output section.")
    print(f"{'='*60}\n")

    # Exit non-zero if any critical step failed
    if res_c.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
