#!/usr/bin/env python3
"""
FXGold backtest — config-driven CLI (mirrors trading/strategies/zz/ustec/backtest.py).

Data source can be the PostgreSQL OHLCV database (default) or a live MT5
terminal (--data_source mt5), same as the USTEC zone-to-zone backtest.
MT5 mode requires the terminal to be open and logged in (.env: MT5_LOGIN /
MT5_PASSWORD / MT5_SERVER) and pulls XAUUSDm bars directly off the chart.

Examples:
    python trading/strategies/FXGold/backtest.py --start 2023-01-01 --end 2024-01-01
    python trading/strategies/FXGold/backtest.py --data_source mt5 --start 2024-01-01 --end 2025-01-01
    python trading/strategies/FXGold/backtest.py --data_source mt5 --chart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.engine import run_backtest
from trading.shared.backtest.report import print_report, save_report
from trading.shared.backtest.chart import plot_trades


def _build_metrics(
    df_t: pd.DataFrame, start: str, end: str, cash: float, equity: float, data_source: str,
) -> dict:
    total    = len(df_t)
    tp_hits  = int((df_t["outcome"] ==  1).sum())
    sl_hits  = int((df_t["outcome"] == -1).sum())
    expired  = int((df_t["outcome"] ==  0).sum())
    win_rate = tp_hits / max(total, 1) * 100
    winners  = df_t[df_t["pnl"] > 0]
    losers   = df_t[df_t["pnl"] < 0]
    eq_s     = pd.Series([cash] + df_t["equity"].tolist())
    max_dd   = round(((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100, 2)
    buy_df   = df_t[df_t["side"] == "buy"]
    sell_df  = df_t[df_t["side"] == "sell"]

    return {
        "symbol":          "XAUUSD",
        "period":          f"{start} to {end}",
        "data_source":     data_source,
        "start_cash":      f"${cash:,.2f}",
        "final_equity":    f"${equity:,.2f}",
        "net_pnl":         f"${equity - cash:,.2f}",
        "total_trades":    total,
        "tp_hits":         tp_hits,
        "sl_hits":         sl_hits,
        "expired":         expired,
        "win_rate_%":      f"{win_rate:.1f}",
        "buy_trades":      len(buy_df),
        "buy_wins":        int((buy_df["outcome"] == 1).sum()),
        "sell_trades":     len(sell_df),
        "sell_wins":       int((sell_df["outcome"] == 1).sum()),
        "avg_win_$":       f"${winners['pnl'].mean():.2f}" if len(winners) else "$0.00",
        "avg_loss_$":      f"${losers['pnl'].mean():.2f}"  if len(losers)  else "$0.00",
        "largest_win_$":   f"${df_t['pnl'].max():.2f}",
        "largest_loss_$":  f"${df_t['pnl'].min():.2f}",
        "max_drawdown_%":  f"{max_dd:.2f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest: FXGold zone strategy (XAUUSD)")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2025-12-31")
    parser.add_argument("--cash",  type=float, default=150.0)
    parser.add_argument("--fixed_lot", type=float, default=0.01)
    parser.add_argument("--data_source", default="db", choices=["db", "mt5"],
                        help="Data source: 'db' = PostgreSQL (default), 'mt5' = live MT5 terminal")
    parser.add_argument("--save",  default=None)
    parser.add_argument("--chart", action="store_true")
    args = parser.parse_args()

    cfg = FXGoldConfig()

    trades, equity_curve, df_h1 = run_backtest(
        cfg,
        start=args.start,
        end=args.end,
        cash=args.cash,
        fixed_lot=args.fixed_lot,
        data_source=args.data_source,
    )

    if not trades:
        print("No trades generated. Widen date range, or check zone/bias filters in config.py.")
        return

    df_t   = pd.DataFrame(trades)
    equity = equity_curve[-1]
    metrics = _build_metrics(df_t, args.start, args.end, args.cash, equity, args.data_source)

    print_report(metrics, run_label="FXGold Zone Strategy")

    print(f"\n  Pattern breakdown:")
    for pattern, grp in df_t.groupby("pattern"):
        wr = (grp["outcome"] == 1).mean() * 100
        print(f"    {pattern:<14} : {len(grp):>4} trades  WR={wr:.0f}%")

    print(f"\n  Zone timeframe:")
    for tf, grp in df_t.groupby("zone_tf"):
        wr = (grp["outcome"] == 1).mean() * 100
        print(f"    {tf:<8} : {len(grp):>4} trades  WR={wr:.0f}%")

    print()
    cols = ["date", "side", "pattern", "zone_tf", "zone_kind",
            "entry", "sl", "tp", "exit", "pnl", "outcome"]
    present = [c for c in cols if c in df_t.columns]
    print(df_t[present].to_string(index=False))
    print()

    if args.save:
        save_report(metrics, args.save)

    if args.chart:
        title = (
            f"FXGold Zone Strategy — XAUUSD  {args.start}→{args.end}  "
            f"{metrics['tp_hits']}W/{metrics['sl_hits']}L/{metrics['expired']}E  "
            f"WR{metrics['win_rate_%']}%"
        )
        fig = plot_trades(df_h1, trades, title=title, start_cash=args.cash)
        fig.show()


if __name__ == "__main__":
    main()
