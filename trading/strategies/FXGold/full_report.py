#!/usr/bin/env python3
"""
full_report.py — FXGold backtest with gate-attrition instrumentation and the
extended HTML report (report.py).

Wraps entry.evaluate_entry to (a) independently verify, via is_rejection_touch,
whether each call was a genuine candidate touch, and (b) bucket every
rejection reason into a gate (bias / touch_count / secret_pattern /
confirmation / sl_geometry / rr / entry_confirmed), keeping both a per-gate
total and a per-gate breakdown of the raw reason strings.

Usage:
    python trading/strategies/FXGold/full_report.py
    python trading/strategies/FXGold/full_report.py --start 2020-01-01 --end 2024-12-31 --load_start 2019-07-01
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import trading.strategies.FXGold.engine as eng
import trading.strategies.FXGold.entry as entry_mod
from trading.strategies.FXGold.zones import is_rejection_touch
from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.report import save_extended_html_report

REPORTS_DIR = _PROJECT_ROOT / "trading" / "Reports"

_GATE_BY_REASON_PREFIX = (
    ("bias_", "bias"),
    ("sub_min_rr_", "rr"),
)
_GATE_BY_REASON_EXACT = {
    "skip_3rd_touch":            "touch_count",
    "breakout_expected_4th_plus": "touch_count",
    "no_secret_pattern":         "secret_pattern",
    "no_confirmation_pattern":   "confirmation",
    "zero_sl_dist":              "sl_geometry",
}


def _gate_for_reason(reason: str) -> str:
    if reason in _GATE_BY_REASON_EXACT:
        return _GATE_BY_REASON_EXACT[reason]
    for prefix, gate in _GATE_BY_REASON_PREFIX:
        if reason.startswith(prefix):
            return gate
    return f"other:{reason}"


def run_with_instrumentation(
    cfg: FXGoldConfig, start: str, end: str, cash: float, fixed_lot: float,
    data_source: str, load_start: str,
):
    gate_counts: Counter = Counter()
    reason_counts: dict = defaultdict(Counter)
    candidate_touches = 0

    _orig_evaluate_entry = entry_mod.evaluate_entry

    def _instrumented_evaluate_entry(**kwargs):
        nonlocal candidate_touches
        bar_high, bar_low, bar_close = kwargs["bar_high"], kwargs["bar_low"], kwargs["bar_close"]
        zone = kwargs["zone"]
        is_touch = is_rejection_touch(bar_high, bar_low, bar_close, zone)

        result = _orig_evaluate_entry(**kwargs)

        if not is_touch:
            gate_counts["not_a_touch_this_bar"] += 1
            return result

        candidate_touches += 1
        if result.valid:
            gate_counts["entry_confirmed"] += 1
            return result

        gate = _gate_for_reason(result.reason)
        gate_counts[gate] += 1
        reason_counts[gate][result.reason] += 1
        return result

    # engine.py did `from ...entry import evaluate_entry` — patch the name IN
    # the engine module, since that's the reference actually called inside
    # run_backtest's loop (Python resolves bare globals by name at call time).
    eng.evaluate_entry = _instrumented_evaluate_entry
    try:
        trades, equity_curve, df_h1 = eng.run_backtest(
            cfg, start=start, end=end, cash=cash, fixed_lot=fixed_lot,
            data_source=data_source, load_start=load_start,
        )
    finally:
        eng.evaluate_entry = _orig_evaluate_entry

    return trades, equity_curve, gate_counts, dict(reason_counts), candidate_touches


def main() -> None:
    parser = argparse.ArgumentParser(description="FXGold extended report — PDF-spec backtest")
    parser.add_argument("--start",      default="2020-01-01")
    parser.add_argument("--end",        default="2024-12-31")
    parser.add_argument("--load_start", default="2019-07-01",
                        help="Data is loaded from this date for warmup, so --start trades clean")
    parser.add_argument("--cash",       type=float, default=150.0)
    parser.add_argument("--fixed_lot",  type=float, default=0.01)
    parser.add_argument("--data_source", default="mt5", choices=["db", "mt5"])
    parser.add_argument("--out", default=None,
                        help="HTML report output path (default: trading/Reports/fxgold_extended_report.html)")
    args = parser.parse_args()

    cfg = FXGoldConfig()
    print(f"FXGold extended report  {args.start} -> {args.end}  (warmup from {args.load_start})")
    print(f"  cash=${args.cash:.2f}  fixed_lot={args.fixed_lot}  data_source={args.data_source}")
    print(f"  tf: high={cfg.tf_high} mid={cfg.tf_mid} entry={cfg.tf_entry}  "
          f"bias_mode={cfg.bias_mode}  d1_bias_method={cfg.d1_bias_method}  "
          f"secret_pattern_enabled={cfg.secret_pattern_enabled}\n")

    trades, equity_curve, gate_counts, reason_counts, candidate_touches = run_with_instrumentation(
        cfg, args.start, args.end, args.cash, args.fixed_lot, args.data_source, args.load_start,
    )

    print(f"\nTrades: {len(trades)}   Candidate touches: {candidate_touches}")
    for gate, count in gate_counts.most_common():
        pct = count / candidate_touches * 100 if candidate_touches and gate != "not_a_touch_this_bar" else 0.0
        print(f"  {gate:<20} {count:>8}   ({pct:.1f}% of candidate touches)" if gate != "not_a_touch_this_bar"
              else f"  {gate:<20} {count:>8}")

    if not trades:
        print("\nNo trades generated — report will still render (funnel/reasons still useful).")

    df_t = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["date", "exit_date", "side", "entry", "sl", "tp", "exit", "outcome", "lot",
                 "pnl", "equity", "pattern", "tp_mode", "zone_tf", "zone_kind", "zone_top",
                 "zone_bottom", "zone_score", "live_touch_count", "d1_bias", "h4_bias"]
    )

    out_path = args.out or str(REPORTS_DIR / "fxgold_extended_report.html")
    save_extended_html_report(
        df_t, gate_counts, reason_counts, candidate_touches,
        start=args.start, end=args.end, load_start=args.load_start, cash=args.cash,
        out_path=out_path,
        meta={
            "data source":     args.data_source,
            "fixed lot":       str(args.fixed_lot),
            "bias_mode":       cfg.bias_mode,
            "d1_bias_method":  cfg.d1_bias_method,
            "tf":              f"{cfg.tf_high}/{cfg.tf_mid}/{cfg.tf_entry}",
        },
    )


if __name__ == "__main__":
    main()
