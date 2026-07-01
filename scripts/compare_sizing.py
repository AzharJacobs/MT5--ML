"""
compare_sizing.py — Tag-based position sizing comparison.

Scenarios (all on USTEC 2022-01-01 → 2025-01-01):
  A  baseline          drift_nc_mult=1.0  clean_conf_mult=1.0
  B  sized 1.5x        drift_nc_mult=1.5  clean_conf_mult=1.0
  C  sized 2.0x        drift_nc_mult=2.0  clean_conf_mult=1.0

For the split-half validation, df_t is sliced by entry date — no extra
re-runs needed.  Equity carry-over means lot sizes in H2 differ slightly
from a cold H2 start, but trade-level WR and relative PnL stats are valid.

Output per scenario:
  1. Month-by-month trade log (entry-date grouping)
  2. Overall totals
Then:
  3. drift × non_confluent split-half comparison across all scenarios
"""

import argparse
import sys

import pandas as pd

from trading.strategies.zz.ustec.engine import run_backtest


# ── shared kwargs ──────────────────────────────────────────────────────────────

SHARED = dict(
    symbol              = "ustech",
    cash                = 10_000.0,
    min_rr              = 1.5,
    nml_filter          = False,   # tags only — no blocking
    stacked_confluence  = False,   # tags only — no blocking
    silent              = True,
)

H1_END   = "2023-06-30"
H2_START = "2023-07-01"


# ── helpers ────────────────────────────────────────────────────────────────────

def _run(label: str, start: str, end: str, drift_nc_mult: float, clean_conf_mult: float = 1.0):
    print(f"  running {label} ({start} → {end}) ...", flush=True)
    result = run_backtest(
        start=start, end=end,
        drift_nc_mult=drift_nc_mult,
        clean_conf_mult=clean_conf_mult,
        **SHARED,
    )
    if not result or not isinstance(result, tuple):
        return None, None
    return result  # (metrics, df_t)


def _monthly_table(df_t: pd.DataFrame, start_cash: float) -> pd.DataFrame:
    df = df_t.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    rows = []
    running = start_cash
    for month, grp in df.sort_values("date").groupby("month", sort=True):
        tp  = int((grp["outcome"] == 1).sum())
        sl  = int((grp["outcome"] == -1).sum())
        exp = int((grp["outcome"] == 0).sum())
        n   = len(grp)
        wr  = tp / max(n, 1) * 100
        net = grp["pnl"].sum()
        running += net
        rows.append({
            "Month":     str(month),
            "Trades":    n,
            "TP":        tp,
            "SL":        sl,
            "Exp":       exp,
            "WR%":       round(wr, 1),
            "Net PnL":   round(net, 2),
            "Equity":    round(running, 2),
        })
    return pd.DataFrame(rows)


def _print_monthly(label: str, metrics: dict, df_t: pd.DataFrame, start_cash: float):
    monthly = _monthly_table(df_t, start_cash)
    total   = len(df_t)
    tp_tot  = int((df_t["outcome"] == 1).sum())
    sl_tot  = int((df_t["outcome"] == -1).sum())
    exp_tot = int((df_t["outcome"] == 0).sum())
    wr_tot  = tp_tot / max(total, 1) * 100
    net_tot = df_t["pnl"].sum()

    W = 72
    print(f"\n{'═'*W}")
    print(f"  SCENARIO: {label}")
    drift_nc = df_t[(df_t["clean_tap"] == False) & (df_t["m15_confluent"] == False)] if "clean_tap" in df_t.columns else pd.DataFrame()
    print(f"  drift×NC trades: {len(drift_nc)} of {total}  "
          f"| drift×NC sizing: {metrics.get('drift_nc_mult','?')}×")
    print(f"{'═'*W}")
    print(f"  {'Month':<10} {'Trades':>7} {'TP':>4} {'SL':>4} {'Exp':>4} "
          f"{'WR%':>6} {'Net PnL':>10} {'Equity':>12}")
    print(f"  {'─'*W}")
    for _, row in monthly.iterrows():
        print(f"  {row['Month']:<10} {row['Trades']:>7} {row['TP']:>4} "
              f"{row['SL']:>4} {row['Exp']:>4} "
              f"{row['WR%']:>5.1f}% "
              f"{row['Net PnL']:>+10.2f} "
              f"{row['Equity']:>12,.2f}")
    print(f"  {'─'*W}")
    final_eq = start_cash + net_tot
    print(f"  {'TOTAL':<10} {total:>7} {tp_tot:>4} {sl_tot:>4} {exp_tot:>4} "
          f"{wr_tot:>5.1f}% "
          f"{net_tot:>+10.2f} "
          f"{final_eq:>12,.2f}")
    print(f"  Max drawdown: {metrics.get('max_drawdown_%', '?')}%  |  "
          f"Avg win: {metrics.get('avg_win_$', '?')}  |  "
          f"Avg loss: {metrics.get('avg_loss_$', '?')}")
    print()


def _drift_nc_stats(df_t: pd.DataFrame, label: str, period: str) -> dict:
    if "clean_tap" not in df_t.columns or "m15_confluent" not in df_t.columns:
        return {}
    g = df_t[(df_t["clean_tap"] == False) & (df_t["m15_confluent"] == False)]
    if len(g) == 0:
        return {"label": label, "period": period, "n": 0, "tp": 0, "sl": 0,
                "wr": 0.0, "net": 0.0, "avg": 0.0}
    tp  = int((g["outcome"] == 1).sum())
    sl  = int((g["outcome"] == -1).sum())
    wr  = tp / max(len(g), 1) * 100
    net = g["pnl"].sum()
    avg = g["pnl"].mean()
    return {"label": label, "period": period, "n": len(g), "tp": tp, "sl": sl,
            "wr": wr, "net": net, "avg": avg}


def _print_split_validation(rows: list):
    W = 80
    print(f"\n{'═'*W}")
    print(f"  drift × non_confluent bucket — SPLIT-HALF VALIDATION")
    print(f"  (all stats from trades in that bucket only; equity carry-over in H2 is expected)")
    print(f"{'═'*W}")
    print(f"  {'Scenario':<22} {'Period':<10} {'N':>5} {'TP':>4} {'SL':>4} "
          f"{'WR%':>6} {'Net PnL':>11} {'Avg/trade':>10}")
    print(f"  {'─'*W}")
    prev_label = None
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['label']:<22} {r['period']:<10}  — (no drift×NC trades in this window)")
            continue
        sep = f"  {'─'*W}" if prev_label and r["label"] != prev_label else ""
        if sep:
            print(sep)
        prev_label = r["label"]
        print(f"  {r['label']:<22} {r['period']:<10} {r['n']:>5} {r['tp']:>4} "
              f"{r['sl']:>4} {r['wr']:>5.1f}% "
              f"{r['net']:>+11.2f} {r['avg']:>+10.2f}")
    print(f"  {'─'*W}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tag-based sizing comparison + monthly breakdown")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end",   default="2025-01-01")
    args = ap.parse_args()

    START      = args.start
    END        = args.end
    START_CASH = 10_000.0

    print(f"\n{'═'*72}")
    print(f"  TAG-BASED SIZING COMPARISON  {START} → {END}")
    print(f"  Sizing rule: drift×NC → mult  |  clean×conf → 1.0×  |  else → 1.0×")
    print(f"{'═'*72}")
    print()
    print("Loading scenarios (this takes a few minutes each)...")

    # ── run all three scenarios ────────────────────────────────────────────────
    scenarios = [
        ("baseline",    1.0),
        ("sized_1.5x",  1.5),
        ("sized_2.0x",  2.0),
    ]

    results = {}
    for label, mult in scenarios:
        m, df = _run(label, START, END, drift_nc_mult=mult)
        if m is not None:
            # stash mult in metrics for display
            m["drift_nc_mult"] = mult
            results[label] = (m, df)

    if not results:
        print("No results returned — check DB connection.")
        sys.exit(1)

    # ── monthly breakdown for each scenario ───────────────────────────────────
    for label, mult in scenarios:
        if label not in results:
            continue
        m, df = results[label]
        _print_monthly(label, m, df, START_CASH)

    # ── split-half validation for drift×NC bucket ─────────────────────────────
    split_rows = []
    for label, _ in scenarios:
        if label not in results:
            continue
        _, df = results[label]

        df_h1 = df[pd.to_datetime(df["date"]) <= pd.Timestamp(H1_END)]
        df_h2 = df[pd.to_datetime(df["date"]) >  pd.Timestamp(H1_END)]

        split_rows.append(_drift_nc_stats(df,    label, f"Full ({START[:4]}–{END[:4]})"))
        split_rows.append(_drift_nc_stats(df_h1, label, f"H1 ({START[:4]}–23-Jun)"))
        split_rows.append(_drift_nc_stats(df_h2, label, f"H2 (23-Jul–{END[:4]})"))

    _print_split_validation(split_rows)

    # ── overall summary comparison ─────────────────────────────────────────────
    W = 72
    print(f"{'═'*W}")
    print(f"  OVERALL SUMMARY  {START} → {END}")
    print(f"{'═'*W}")
    print(f"  {'Scenario':<18} {'Trades':>7} {'TP':>4} {'SL':>4} "
          f"{'WR%':>6} {'Net PnL':>11} {'Max DD':>8}")
    print(f"  {'─'*W}")
    for label, _ in scenarios:
        if label not in results:
            continue
        m, df = results[label]
        total = int(m.get("total_trades", 0))
        tp    = int(m.get("tp_hits", 0))
        sl    = int(m.get("sl_hits", 0))
        wr    = m.get("win_rate_%", "0")
        net   = m.get("net_pnl", "$0")
        dd    = m.get("max_drawdown_%", "0")
        print(f"  {label:<18} {total:>7} {tp:>4} {sl:>4} "
              f"{wr:>6}% {net:>11} {dd:>7}%")
    print(f"  {'─'*W}")
    print()


if __name__ == "__main__":
    main()
