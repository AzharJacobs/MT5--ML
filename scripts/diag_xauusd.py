#!/usr/bin/env python3
"""
diag_xauusd.py — XAUUSD Zone-to-Zone backtest diagnostic (2023-2025).

Runs 3 annual backtests and prints:
  • Core stats table (trades, WR, payoff, avg win/loss, net P&L, max DD)
  • H4 bias distribution
  • Zone stats (height $, height ATR, strength, freshness)
  • Outcome split (TP / SL / expired)
  • First-tap vs re-test WR + P&L
  • Confirmation signal breakdown
  • Avg RR achieved vs planned

Usage:
    python scripts/diag_xauusd.py
    python scripts/diag_xauusd.py --min_rr 1.5 --spread 0.5
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.engine_zones import run_backtest

YEARS = [
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

SEP = "=" * 70
SEP2 = "-" * 70


def _atr14(df_4h: pd.DataFrame) -> float:
    """Rough 14-period ATR from 4H data as a price-context scalar."""
    if len(df_4h) < 15:
        return float("nan")
    hi = df_4h["high"].values[-15:]
    lo = df_4h["low"].values[-15:]
    cl = df_4h["close"].values[-15:]
    tr = np.maximum(hi[1:] - lo[1:],
         np.maximum(np.abs(hi[1:] - cl[:-1]),
                    np.abs(lo[1:] - cl[:-1])))
    return float(np.mean(tr))


def _payoff(df: pd.DataFrame) -> float:
    wins   = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] < 0]["pnl"]
    if losses.empty or wins.empty:
        return float("nan")
    return abs(wins.mean() / losses.mean())


def _rr_planned(row) -> float:
    risk   = abs(row["entry"] - row["sl"])
    reward = abs(row["tp"]    - row["entry"])
    if risk == 0:
        return float("nan")
    return reward / risk


def _rr_achieved(row) -> float:
    risk = abs(row["entry"] - row["sl"])
    if risk == 0:
        return float("nan")
    if row["side"] == "buy":
        return (row["exit"] - row["entry"]) / risk
    else:
        return (row["entry"] - row["exit"]) / risk


def core_stats(df: pd.DataFrame, year_label: str, start_cash: float = 10_000.0) -> dict:
    total    = len(df)
    tp_hits  = int((df["outcome"] == 1).sum())
    sl_hits  = int((df["outcome"] == -1).sum())
    expired  = int((df["outcome"] == 0).sum())
    wr       = tp_hits / max(total, 1) * 100
    winners  = df[df["pnl"] > 0]
    losers   = df[df["pnl"] < 0]
    net_pnl  = df["pnl"].sum()

    eq = [start_cash] + list(df["equity"])
    eq_s   = pd.Series(eq)
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100

    return {
        "year":       year_label,
        "trades":     total,
        "tp":         tp_hits,
        "sl":         sl_hits,
        "exp":        expired,
        "wr_%":       wr,
        "payoff":     _payoff(df),
        "avg_win":    winners["pnl"].mean() if len(winners) else float("nan"),
        "avg_loss":   losers["pnl"].mean()  if len(losers)  else float("nan"),
        "net_pnl":    net_pnl,
        "max_dd_%":   max_dd,
    }


def diag_bias(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for bias, grp in df.groupby("h4_bias"):
        rows.append({
            "bias":     bias,
            "count":    len(grp),
            "pct_%":    len(grp) / max(total, 1) * 100,
            "wr_%":     (grp["outcome"] == 1).mean() * 100,
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def diag_zones(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["zone_height_$"] = df["zone_top"] - df["zone_bottom"]
    return {
        "avg_strength":     df["zone_strength"].mean(),
        "avg_height_$":     df["zone_height_$"].mean(),
        "med_height_$":     df["zone_height_$"].median(),
        "supply_pct_%":     (df["zone_kind"] == "supply").mean() * 100,
        "demand_pct_%":     (df["zone_kind"] == "demand").mean() * 100,
    }


def diag_outcomes(df: pd.DataFrame) -> dict:
    total = len(df)
    return {
        "tp_pct_%":      (df["outcome"] == 1).sum() / max(total, 1) * 100,
        "sl_pct_%":      (df["outcome"] == -1).sum() / max(total, 1) * 100,
        "expired_pct_%": (df["outcome"] == 0).sum() / max(total, 1) * 100,
    }


def diag_tap_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, grp in [("first_tap",  df[~df["is_retest"]]),
                        ("re_test",    df[ df["is_retest"]])]:
        if len(grp) == 0:
            continue
        rows.append({
            "type":    label,
            "trades":  len(grp),
            "wr_%":    (grp["outcome"] == 1).mean() * 100,
            "net_pnl": grp["pnl"].sum(),
        })
    return pd.DataFrame(rows)


def diag_confirmations(df: pd.DataFrame) -> pd.DataFrame:
    all_signals = []
    for sigs in df["signals"]:
        all_signals.extend(sigs.split("|") if sigs else [])
    unique_signals = sorted(set(s for s in all_signals if s))

    rows = []
    for sig in unique_signals:
        mask = df["signals"].str.contains(sig, na=False)
        grp  = df[mask]
        rows.append({
            "signal":  sig,
            "fires":   len(grp),
            "pct_%":   len(grp) / max(len(df), 1) * 100,
            "wr_%":    (grp["outcome"] == 1).mean() * 100,
            "net_pnl": grp["pnl"].sum(),
        })
    return pd.DataFrame(rows).sort_values("fires", ascending=False)


def diag_rr(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["rr_planned"]  = df.apply(_rr_planned,  axis=1)
    df["rr_achieved"] = df.apply(_rr_achieved, axis=1)
    wins  = df[df["outcome"] == 1]
    losses = df[df["outcome"] == -1]
    return {
        "avg_rr_planned":        df["rr_planned"].mean(),
        "avg_rr_achieved_wins":  wins["rr_achieved"].mean()   if len(wins)   else float("nan"),
        "avg_rr_achieved_loss":  losses["rr_achieved"].mean() if len(losses) else float("nan"),
        "avg_rr_achieved_all":   df["rr_achieved"].mean(),
    }


def print_core_table(stats_list: list) -> None:
    print(f"\n{'YEAR':<10} {'TRADES':>7} {'TP':>5} {'SL':>5} {'EXP':>5} "
          f"{'WR%':>7} {'PAYOFF':>7} {'AVG WIN':>10} {'AVG LOSS':>10} "
          f"{'NET P&L':>12} {'MAX DD%':>9}")
    print(SEP2)
    for s in stats_list:
        print(
            f"{s['year']:<10} {s['trades']:>7} {s['tp']:>5} {s['sl']:>5} {s['exp']:>5} "
            f"{s['wr_%']:>6.1f}% {s['payoff']:>7.2f} "
            f"${s['avg_win']:>9.2f} ${s['avg_loss']:>9.2f} "
            f"${s['net_pnl']:>11.2f} {s['max_dd_%']:>8.2f}%"
        )
    print()


def run_all(args) -> None:
    all_dfs   = []
    stats_list = []
    start_cash = 10_000.0

    print(f"\n{SEP}")
    print(f"  XAUUSD Zone-to-Zone Diagnostic  |  min_rr={args.min_rr}  "
          f"spread={args.spread}  min_conf={args.min_confirmations}")
    print(SEP)

    running_cash = start_cash
    for start, end in YEARS:
        label = start[:4]
        print(f"\n{'─'*50}")
        print(f"  Running {label}  ({start} → {end})")
        print(f"{'─'*50}")

        result = run_backtest(
            start=start,
            end=end,
            cash=running_cash,
            symbol="xauusd",
            min_rr=args.min_rr,
            spread=args.spread,
            min_confirmations=args.min_confirmations,
        )
        if isinstance(result, tuple):
            metrics, df_t = result
        else:
            print("  [!] Engine returned no trades — skipping year.")
            continue

        if df_t is None or df_t.empty:
            print("  [!] Empty trade DataFrame — skipping year.")
            continue

        all_dfs.append(df_t)
        stats_list.append(core_stats(df_t, label, running_cash))

        # carry equity forward for realistic compounding
        if "equity" in df_t.columns and len(df_t):
            running_cash = float(df_t["equity"].iloc[-1])

    if not all_dfs:
        print("\nNo trades across any year. Abort.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    stats_list.append(core_stats(df_all, "3YR COMBINED", start_cash))

    # ─── Core table ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CORE PERFORMANCE TABLE")
    print(SEP)
    print_core_table(stats_list)

    # ─── Diagnostic 1: H4 bias distribution ───────────────────────────────
    print(f"{SEP}")
    print("  DIAG 1 — H4 BIAS DISTRIBUTION  (3-yr combined)")
    print(SEP)
    bias_df = diag_bias(df_all)
    print(bias_df.to_string(index=False))
    print()

    print("  Per-year bias:")
    for df_y, (s, _) in zip(all_dfs, YEARS):
        label = s[:4]
        bd = diag_bias(df_y)
        print(f"\n  {label}:")
        print(bd.to_string(index=False))
    print()

    # ─── Diagnostic 2: Zone stats ──────────────────────────────────────────
    print(f"{SEP}")
    print("  DIAG 2 — ZONE STATS")
    print(SEP)
    zs = diag_zones(df_all)
    print(f"  Avg zone strength   : {zs['avg_strength']:.2f}  (impulse / ATR; min_strength=1.5)")
    print(f"  Avg zone height $   : ${zs['avg_height_$']:.2f}")
    print(f"  Median zone height $: ${zs['med_height_$']:.2f}")
    print(f"  Demand zones traded : {zs['demand_pct_%']:.1f}%")
    print(f"  Supply zones traded : {zs['supply_pct_%']:.1f}%")
    print()

    # ─── Diagnostic 3: Outcome breakdown ──────────────────────────────────
    print(f"{SEP}")
    print("  DIAG 3 — OUTCOME SPLIT  (TP / SL / Expired/Never-reached)")
    print(SEP)
    print(f"  {'YEAR':<10} {'TP%':>7} {'SL%':>7} {'EXPIRED%':>10}")
    print(SEP2)
    for df_y, (s, _) in zip(all_dfs, YEARS):
        od = diag_outcomes(df_y)
        print(f"  {s[:4]:<10} {od['tp_pct_%']:>6.1f}% {od['sl_pct_%']:>6.1f}% "
              f"{od['expired_pct_%']:>9.1f}%")
    od_all = diag_outcomes(df_all)
    print(f"  {'COMBINED':<10} {od_all['tp_pct_%']:>6.1f}% {od_all['sl_pct_%']:>6.1f}% "
          f"{od_all['expired_pct_%']:>9.1f}%")
    print()

    # ─── Diagnostic 4: First-tap vs re-test ───────────────────────────────
    print(f"{SEP}")
    print("  DIAG 4 — FIRST-TAP vs RE-TEST")
    print(SEP)
    print(f"  3-yr combined:")
    tap_df = diag_tap_split(df_all)
    print(tap_df.to_string(index=False))
    print()
    print(f"  Prior-outcome bucket breakdown (3-yr):")
    for pb in ("first_attempt", "post_win", "post_loss", "post_expired"):
        g = df_all[df_all["prior_bucket"] == pb]
        if len(g) == 0:
            continue
        wr  = (g["outcome"] == 1).mean() * 100
        net = g["pnl"].sum()
        print(f"    {pb:<16}: {len(g):>3} trades  WR={wr:.0f}%  net=${net:+.2f}")
    print()

    # ─── Diagnostic 5: Confirmation signal breakdown ───────────────────────
    print(f"{SEP}")
    print("  DIAG 5 — CONFIRMATION SIGNAL BREAKDOWN")
    print(SEP)
    conf_df = diag_confirmations(df_all)
    print(conf_df.to_string(index=False))
    print()

    # ─── Diagnostic 6: RR achieved vs planned ─────────────────────────────
    print(f"{SEP}")
    print("  DIAG 6 — RR ACHIEVED vs PLANNED")
    print(SEP)
    print(f"  {'YEAR':<10} {'PLANNED':>9} {'WINS ACHVD':>12} {'LOSS ACHVD':>12} {'ALL ACHVD':>11}")
    print(SEP2)
    for df_y, (s, _) in zip(all_dfs, YEARS):
        rr = diag_rr(df_y)
        print(f"  {s[:4]:<10} {rr['avg_rr_planned']:>9.2f} "
              f"{rr['avg_rr_achieved_wins']:>12.2f} "
              f"{rr['avg_rr_achieved_loss']:>12.2f} "
              f"{rr['avg_rr_achieved_all']:>11.2f}")
    rr_all = diag_rr(df_all)
    print(f"  {'COMBINED':<10} {rr_all['avg_rr_planned']:>9.2f} "
          f"{rr_all['avg_rr_achieved_wins']:>12.2f} "
          f"{rr_all['avg_rr_achieved_loss']:>12.2f} "
          f"{rr_all['avg_rr_achieved_all']:>11.2f}")
    print()
    print(f"  NOTE: Planned RR = (TP-entry)/(entry-SL).  "
          f"Achieved RR uses actual exit price.")
    print(f"  A 'achieved wins' < 'planned' means price hits TP but spread/slippage eats margin.")
    print(f"  A negative 'all achieved' confirms SL losses dominate over wins.\n")


def main():
    parser = argparse.ArgumentParser(description="XAUUSD Z&Z diagnostic")
    parser.add_argument("--min_rr",           type=float, default=1.5)
    parser.add_argument("--spread",           type=float, default=0.0)
    parser.add_argument("--min_confirmations", type=int,  default=1)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
