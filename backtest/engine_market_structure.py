"""
engine_market_structure.py — Backtest engine for the Market Structure Trading Plan.

Pure rules-based simulation — no ML model, no feature engineering.
Loads raw 4H + 15M OHLCV from the DB, applies the strategy signal bar-by-bar,
and simulates TP/SL outcomes using forward price action.

Usage (via scripts/backtest_market_structure.py):
    python scripts/backtest_market_structure.py --start 2024-01-01 --end 2025-01-01
    python scripts/backtest_market_structure.py --cash 5000 --min_rr 2.0 --max_bars 30
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from data.loader import get_connection
from strategy.market_structure import (
    apply_market_structure_signal,
    M15_BARS   as M15_WINDOW,
    H4_BARS    as H4_WINDOW,
)
from backtest.report import print_report, save_report
from backtest.chart_market_structure import plot_trades


# 4H strategy targets can be $30-50 away — needs 1-2 days to reach.
# 200 bars × 15 min = 50 hours (~2 trading days). Configurable via --max_bars.
MAX_FORWARD_BARS = 350

# Zone re-entry cooldown — outcome-based (15M bars).
# Win: zone consumed, block for the rest of the sim.
# Loss: short retry window (48 bars = 12 h). Expired: no cooldown.
COOLDOWN_WIN     = 9999
COOLDOWN_LOSS    = 48
COOLDOWN_EXPIRED = 0

# XAUUSD: 1 lot = 100 oz, so each $1 price move = $100 P&L per lot.
XAUUSD_CONTRACT = 100.0

# Risk 1% of current equity per trade.
RISK_PCT = 0.01


# Per-symbol settings: (db_name, table_name, contract_size)
# contract_size: dollar value of 1 full lot per 1 point move
SYMBOL_CONFIG = {
    "xauusd": ("XAUUSD",      "xauusd_ohlcv", 100.0),  # 1 lot = 100 oz, $1/oz = $100/lot
    "ustech": ("ustech_ohlcv","ustech_ohlcv",   1.0),  # 1 lot = 1 unit, $1/pt = $1/lot
}

# Default spread in points per symbol (applied to entry: buy+spread, sell-spread).
SYMBOL_SPREAD = {
    "xauusd": 0.0,   # SL buffer already covers it
    "ustech": 2.0,   # typical 2-point spread
}


def _load_ohlcv(db, table: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    query = (
        f"SELECT * FROM {table} WHERE timeframe = %s "
        f"AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC"
    )
    df = db.fetch_dataframe(query, (timeframe, start, end))
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _atr(df_15m: pd.DataFrame, period: int = 14) -> float:
    """14-period ATR on the 15M window."""
    high  = df_15m["high"].values
    low   = df_15m["low"].values
    close = df_15m["close"].values
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 1.0
    return float(np.mean(tr[-period:]))


def _lot_size(equity: float, sl_dist: float, contract_size: float = XAUUSD_CONTRACT) -> float:
    """Position size that risks RISK_PCT of equity given an SL distance (price points)."""
    if sl_dist <= 0:
        return 0.01
    risk_dollars = equity * RISK_PCT
    lot = risk_dollars / (sl_dist * contract_size)
    return max(round(lot, 2), 0.01)


def run_backtest(
    start: str = "2024-01-01",
    end: str = "2025-01-01",
    cash: float = 10000.0,
    min_rr: float = 1.5,
    max_forward_bars: int = MAX_FORWARD_BARS,
    sl_atr_mult: float = 0.0,
    symbol: str = "xauusd",
    spread: Optional[float] = None,
    save_path: Optional[str] = None,
    chart: bool = False,
) -> dict:
    """
    Run the Market Structure backtest and print a summary report.

    Returns the metrics dict (also saved to save_path as JSON if provided).
    """
    sym = symbol.lower()
    if sym not in SYMBOL_CONFIG:
        raise ValueError(f"Unknown symbol '{symbol}'. Choose from: {list(SYMBOL_CONFIG)}")

    db_name, table, contract_size = SYMBOL_CONFIG[sym]
    eff_spread = spread if spread is not None else SYMBOL_SPREAD.get(sym, 0.0)

    # Connect to the correct database for this symbol
    db = get_connection()
    db.database = db_name
    db.connect()

    print(f"\nSymbol: {sym.upper()}  |  DB: {db_name}  |  Table: {table}  |  Contract: ${contract_size}/pt/lot  |  Spread: {eff_spread} pts")
    print(f"Loading 15M data  {start} → {end} ...")
    df_15m = _load_ohlcv(db, table, "15min", start, end)
    print(f"Loading 4H  data  {start} → {end} ...")
    df_4h  = _load_ohlcv(db, table, "4H",    start, end)

    try:
        db.connection.close()
    except Exception:
        pass

    if df_15m.empty or df_4h.empty:
        print("ERROR: no data returned — check DB connection and date range.")
        return {}

    print(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")
    print(f"Running backtest  min_rr={min_rr}  max_hold={max_forward_bars} bars ...\n")

    equity        = cash
    equity_curve  = [cash]
    trades        = []
    skip_until    = -1
    zone_cooldown: dict = {}
    n             = len(df_15m)
    warmup        = max(M15_WINDOW, H4_WINDOW, 15)

    # Filter breakdown counters
    filters = {
        "in_position":              0,
        "4H bias neutral":          0,
        "4H zone invalidated":      0,
        "zone not tapped recently": 0,
        "no 15M BOS":               0,
        "15M bias conflicts":       0,
        "15M post-BOS broken":      0,
        "no 15M swing for SL":      0,
        "SL geometry invalid":      0,
        "TP geometry invalid":      0,
        "SL too tight":             0,
        "15M BOS stale":            0,
        "RR too low":               0,
        "zone cooldown":            0,
        "other neutral":            0,
    }

    def _bucket(reason: str) -> str:
        r = reason.lower()
        if "bias neutral"            in r: return "4H bias neutral"
        if "invalidated"             in r: return "4H zone invalidated"
        if "not tapped"              in r: return "zone not tapped recently"
        if "15m" in r and "bos"  in r and "stale"    in r: return "15M BOS stale"
        if "15m" in r and "bos"  in r and "broken"   in r: return "15M post-BOS broken"
        if "15m" in r and "bos"  in r:                     return "no 15M BOS"
        if "conflicts"               in r: return "15M bias conflicts"
        if "swing" in r and "sl" in r: return "no 15M swing for SL"
        if "sl at or"                in r: return "SL geometry invalid"
        if "tp at or"                in r: return "TP geometry invalid"
        if "sl too tight"            in r: return "SL too tight"
        if "rr"                      in r: return "RR too low"
        return "other neutral"

    for i in range(warmup, n - max_forward_bars):
        if i <= skip_until:
            filters["in_position"] += 1
            continue

        ts_now = df_15m["timestamp"].iloc[i]

        df_4h_w = df_4h[df_4h["timestamp"] <= ts_now].tail(H4_WINDOW).reset_index(drop=True)
        if len(df_4h_w) < 10:
            continue

        df_15m_w = df_15m.iloc[i - M15_WINDOW: i + 1].reset_index(drop=True)

        sig = apply_market_structure_signal(df_15m_w, df_4h_w, min_rr=min_rr)

        if sig["signal"] == "neutral":
            filters[_bucket(sig["reason"])] += 1
            continue

        # Zone cooldown
        zone = sig["structure"].get("zone")
        if zone is not None:
            zone_key = (round(zone[0], 1), round(zone[1], 1))
            if zone_cooldown.get(zone_key, -1) >= i:
                filters["zone cooldown"] += 1
                continue

        # Entry = open of the next bar (realistic fill after signal bar closes)
        entry  = float(df_15m["open"].iloc[i + 1])
        sl     = sig["sl"]
        tp     = sig["tp"]
        side   = sig["signal"]
        reason = sig["reason"]

        # Apply spread cost (buy pays ask, sell receives bid)
        if eff_spread > 0:
            entry = entry + eff_spread if side == "buy" else entry - eff_spread

        # Re-validate geometry at actual entry (next-bar open may differ from signal close)
        if side == "buy":
            if sl >= entry:
                filters["SL geometry invalid"] += 1
                continue
            if tp <= entry:
                filters["TP geometry invalid"] += 1
                continue
        else:
            if sl <= entry:
                filters["SL geometry invalid"] += 1
                continue
            if tp >= entry:
                filters["TP geometry invalid"] += 1
                continue

        # ATR-based SL override: replaces 15M swing SL with entry ± N×ATR
        if sl_atr_mult > 0:
            atr_val = _atr(df_15m_w)
            if side == "buy":
                sl = entry - sl_atr_mult * atr_val
            else:
                sl = entry + sl_atr_mult * atr_val
            # Re-check geometry and RR with new SL
            risk   = abs(entry - sl)
            reward = abs(tp - entry)
            if risk <= 0 or reward / risk < min_rr:
                filters["RR too low"] += 1
                continue
            reason += f" | ATR-SL({sl_atr_mult}×{atr_val:.2f})"

        lot    = _lot_size(equity, abs(entry - sl), contract_size)

        # ---- simulate forward price action ----
        outcome        = 0
        exit_price     = entry
        exit_bar       = i + max_forward_bars
        max_favourable = 0.0
        max_adverse    = 0.0

        for j in range(i + 1, min(i + 1 + max_forward_bars, n)):
            fh = float(df_15m["high"].iloc[j])
            fl = float(df_15m["low"].iloc[j])

            if side == "buy":
                max_favourable = max(max_favourable, fh - entry)
                max_adverse    = max(max_adverse,    entry - fl)
                if fh >= tp:
                    outcome = 1;  exit_price = tp;  exit_bar = j; break
                if fl <= sl:
                    outcome = -1; exit_price = sl;  exit_bar = j; break
            else:
                max_favourable = max(max_favourable, entry - fl)
                max_adverse    = max(max_adverse,    fh - entry)
                if fl <= tp:
                    outcome = 1;  exit_price = tp;  exit_bar = j; break
                if fh >= sl:
                    outcome = -1; exit_price = sl;  exit_bar = j; break

        # ---- P&L ----
        if outcome != 0:
            if side == "buy":
                pnl = (exit_price - entry) * lot * contract_size
            else:
                pnl = (entry - exit_price) * lot * contract_size
        else:
            pnl = 0.0   # expired without hitting TP or SL

        equity += pnl
        equity_curve.append(equity)
        skip_until = i + max_forward_bars

        # Outcome-based zone cooldown: win=consumed, loss=short retry, expired=none
        if zone is not None:
            zone_key = (round(zone[0], 1), round(zone[1], 1))
            if outcome == 1:
                cooldown = COOLDOWN_WIN
            elif outcome == -1:
                cooldown = COOLDOWN_LOSS
            else:
                cooldown = COOLDOWN_EXPIRED
            if cooldown > 0:
                zone_cooldown[zone_key] = exit_bar + cooldown

        exit_ts = df_15m["timestamp"].iloc[min(exit_bar, n - 1)]
        trades.append({
            "date":          ts_now,
            "exit_date":     exit_ts,
            "side":          side,
            "entry":         entry,
            "sl":            sl,
            "tp":            tp,
            "exit":          exit_price,
            "outcome":       outcome,
            "lot":           lot,
            "pnl":           round(pnl, 2),
            "equity":        round(equity, 2),
            "max_favour":    round(max_favourable, 2),
            "max_adverse":   round(max_adverse, 2),
            "playbook":      sig.get("playbook", "A"),
            "reason":        reason,
        })

    # ---- build report ----
    if not trades:
        print("No trades generated. Try a wider date range or lower min_rr.")
        return {}

    df_t = pd.DataFrame(trades)

    total      = len(df_t)
    tp_hits    = int((df_t["outcome"] ==  1).sum())
    sl_hits    = int((df_t["outcome"] == -1).sum())
    expired    = int((df_t["outcome"] ==  0).sum())
    win_rate   = tp_hits / max(total, 1) * 100
    winners    = df_t[df_t["pnl"] > 0]
    losers     = df_t[df_t["pnl"] < 0]

    eq_s       = pd.Series(equity_curve)
    max_dd_pct = round(((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100, 2)

    buy_df     = df_t[df_t["side"] == "buy"]
    sell_df    = df_t[df_t["side"] == "sell"]

    metrics = {
        "strategy":         "Market Structure Trading Plan — Playbook A",
        "symbol":           sym.upper(),
        "period":           f"{start} to {end}",
        "start_cash":       f"${cash:,.2f}",
        "final_equity":     f"${equity:,.2f}",
        "net_pnl":          f"${equity - cash:,.2f}",
        "total_trades":     total,
        "tp_hits":          tp_hits,
        "sl_hits":          sl_hits,
        "expired_no_fill":  expired,
        "win_rate_%":       f"{win_rate:.1f}",
        "buy_trades":       len(buy_df),
        "buy_wins":         int((buy_df["outcome"] == 1).sum()),
        "sell_trades":      len(sell_df),
        "sell_wins":        int((sell_df["outcome"] == 1).sum()),
        "avg_win_$":        f"${winners['pnl'].mean():.2f}" if len(winners) else "$0.00",
        "avg_loss_$":       f"${losers['pnl'].mean():.2f}"  if len(losers)  else "$0.00",
        "largest_win_$":    f"${df_t['pnl'].max():.2f}",
        "largest_loss_$":   f"${df_t['pnl'].min():.2f}",
        "max_drawdown_%":   f"{max_dd_pct:.2f}",
        "spread_pts":       eff_spread,
        "sl_mode":          f"ATR×{sl_atr_mult}" if sl_atr_mult > 0 else "15M swing",
        "min_rr_filter":    min_rr,
        "max_hold_bars":    max_forward_bars,
    }

    print_report(metrics, run_label="Market Structure")

    # Filter breakdown
    evaluated = n - warmup - max_forward_bars
    print(f"\n{'─'*45}")
    print(f"  Filter breakdown  ({evaluated:,} bars evaluated)")
    print(f"{'─'*45}")
    for label, count in sorted(filters.items(), key=lambda x: -x[1]):
        if count == 0:
            continue
        pct = count / max(evaluated, 1) * 100
        print(f"  {label:<30} {count:>7,}  ({pct:.1f}%)")
    print(f"  {'signals fired':<30} {total:>7,}")
    print(f"{'─'*45}\n")

    # Trade logs separated by playbook
    cols = ["date", "side", "entry", "sl", "tp", "exit", "pnl", "outcome",
            "max_favour", "max_adverse"]
    for pb in sorted(df_t["playbook"].unique()):
        pb_df   = df_t[df_t["playbook"] == pb]
        pb_wins = int((pb_df["outcome"] == 1).sum())
        pb_loss = int((pb_df["outcome"] == -1).sum())
        pb_exp  = int((pb_df["outcome"] == 0).sum())
        print(f"\nPlaybook {pb}  ({len(pb_df)} trades  |  {pb_wins}W / {pb_loss}L / {pb_exp}E):")
        print(pb_df[cols].to_string(index=False))
    print()

    if save_path:
        save_report(metrics, save_path)

    if chart:
        chart_title = (
            f"Market Structure — {sym.upper()}  |  {start} → {end}  |  "
            f"{tp_hits}W / {sl_hits}L / {expired}E  |  WR {win_rate:.1f}%"
        )
        fig = plot_trades(df_15m, trades, title=chart_title, start_cash=cash)
        fig.show()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest: Market Structure Trading Plan (PDF strategy)"
    )
    parser.add_argument("--symbol",   default="xauusd",
                        choices=list(SYMBOL_CONFIG),
                        help="Symbol to backtest (default: xauusd)")
    parser.add_argument("--start",    default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default="2025-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--cash",     type=float, default=10000.0, help="Starting equity ($)")
    parser.add_argument("--min_rr",   type=float, default=1.5,     help="Min risk:reward ratio")
    parser.add_argument("--max_bars", type=int,   default=350,
                        help="Max 15M bars to wait for TP/SL (350 bars ≈ 87h / ~4 trading days)")
    parser.add_argument("--sl_atr",   type=float, default=0.0,
                        help="ATR multiplier for SL (e.g. 1.5 = SL at 1.5×ATR from entry). "
                             "0 = use 15M swing low/high (default)")
    parser.add_argument("--spread",   type=float, default=None,
                        help="Spread in points applied at entry (default: 0 for xauusd, 2 for ustech)")
    parser.add_argument("--save",     default=None,
                        help="Optional path to save JSON report (e.g. results/ms_backtest.json)")
    parser.add_argument("--chart",    action="store_true",
                        help="Open interactive Plotly chart after backtest completes")
    args = parser.parse_args()

    run_backtest(
        start=args.start,
        end=args.end,
        cash=args.cash,
        min_rr=args.min_rr,
        max_forward_bars=args.max_bars,
        sl_atr_mult=args.sl_atr,
        symbol=args.symbol,
        spread=args.spread,
        save_path=args.save,
        chart=args.chart,
    )


if __name__ == "__main__":
    main()
