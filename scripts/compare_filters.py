"""
compare_filters.py — 3-scenario comparison: baseline vs NML vs NML+confluence.

Scenarios
---------
  A  baseline              nml_filter=False  stacked_confluence=False
  B  NML only              nml_filter=True   stacked_confluence=False
  C  NML + confluence      nml_filter=True   stacked_confluence=True

All scenarios tag every trade with clean_tap + m15_confluent so the split
stats are visible even in scenario A.

Run:
  python -m scripts.compare_filters
  python -m scripts.compare_filters --start 2022-01-01 --end 2025-01-01
"""

import argparse
import sys

from trading.strategies.zz.ustec.engine import run_backtest


# ── shared kwargs for all three scenarios ──────────────────────────────────────

SHARED = dict(
    symbol          = "ustech",
    cash            = 10_000.0,
    min_rr          = 1.5,
    silent          = True,   # suppress per-scenario verbose output; we print our own
)


def _scenario(label: str, start: str, end: str, **extra) -> dict | None:
    print(f"\n{'═'*64}")
    print(f"  Scenario {label}   {start} → {end}")
    print(f"{'═'*64}")
    result = run_backtest(start=start, end=end, **SHARED, **extra)
    if not result or not isinstance(result, tuple):
        print("  No trades generated.")
        return None
    metrics, df_t = result
    return {"metrics": metrics, "df": df_t}


def _pct(v, total):
    return f"{v/max(total,1)*100:.1f}%"


def _split_report(df_t, col: str, true_label: str, false_label: str, header: str):
    if col not in df_t.columns:
        return
    print(f"\n  {header}")
    print(f"  {'─'*60}")
    print(f"  {'Group':<16} {'Trades':>7} {'TP':>5} {'SL':>5} {'Exp':>5} {'WR%':>6} {'Net PnL':>11}")
    print(f"  {'─'*60}")
    for val, lbl in [(True, true_label), (False, false_label)]:
        g = df_t[df_t[col] == val]
        if len(g) == 0:
            continue
        tp  = int((g["outcome"] == 1).sum())
        sl  = int((g["outcome"] == -1).sum())
        exp = int((g["outcome"] == 0).sum())
        wr  = tp / max(len(g), 1) * 100
        net = g["pnl"].sum()
        print(f"  {lbl:<16} {len(g):>7} {tp:>5} {sl:>5} {exp:>5} {wr:>5.1f}% {net:>+11.2f}")
    print(f"  {'─'*60}")


def _summary_row(label: str, m: dict, df_t):
    total = int(m.get("total_trades", 0))
    tp    = int(m.get("tp_hits", 0))
    sl    = int(m.get("sl_hits", 0))
    exp   = int(m.get("expired", 0))
    wr    = m.get("win_rate_%", "0")
    net   = m.get("net_pnl", "$0.00")
    dd    = m.get("max_drawdown_%", "0")
    print(f"  {label:<26} {total:>6}  {tp:>4}TP {sl:>4}SL {exp:>4}Exp  "
          f"WR={wr:>5}  {net:>10}  DD={dd}%")


def main():
    ap = argparse.ArgumentParser(description="3-scenario NML + confluence filter comparison")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end",   default="2025-01-01")
    args = ap.parse_args()

    print(f"\nRunning 3-year backtest  {args.start} → {args.end}")
    print("Scenarios: A=baseline  B=NML  C=NML+confluence")

    A = _scenario("A  (baseline)",        args.start, args.end,
                  nml_filter=False, stacked_confluence=False)
    B = _scenario("B  (NML filter)",      args.start, args.end,
                  nml_filter=True,  stacked_confluence=False)
    C = _scenario("C  (NML+confluence)",  args.start, args.end,
                  nml_filter=True,  stacked_confluence=True)

    # ── overall comparison table ──────────────────────────────────────────────
    print(f"\n\n{'═'*64}")
    print(f"  OVERALL COMPARISON  ({args.start} → {args.end})")
    print(f"{'═'*64}")
    print(f"  {'Scenario':<26} {'Trades':>6}  {'Exits':>16}  {'WR':>6}  {'Net PnL':>10}  {'MaxDD'}")
    print(f"  {'─'*64}")
    for label, res in [("A: baseline", A), ("B: NML filter", B), ("C: NML+confluence", C)]:
        if res is None:
            print(f"  {label:<26}  (no trades)")
            continue
        _summary_row(label, res["metrics"], res["df"])

    # ── clean_tap split (from scenario A — all trades tagged) ────────────────
    if A:
        _split_report(A["df"], "clean_tap",
                      "clean_tap",  "nml_drift",
                      "Win-rate by tap quality  (scenario A — all trades, unfiltered)")

    # ── m15_confluent split (from scenario A) ────────────────────────────────
    if A:
        _split_report(A["df"], "m15_confluent",
                      "confluent",  "non_confluent",
                      "Win-rate by M15 confluence  (scenario A — all trades, unfiltered)")

    # ── cross-split: clean_tap × m15_confluent (from scenario A) ─────────────
    if A and "clean_tap" in A["df"].columns and "m15_confluent" in A["df"].columns:
        df_t = A["df"]
        print(f"\n  Cross-split: clean_tap × m15_confluent  (scenario A)")
        print(f"  {'─'*68}")
        print(f"  {'Group':<28} {'Trades':>7} {'TP':>5} {'WR%':>6} {'Net PnL':>11}")
        print(f"  {'─'*68}")
        for ct in [True, False]:
            for mc in [True, False]:
                g = df_t[(df_t["clean_tap"] == ct) & (df_t["m15_confluent"] == mc)]
                if len(g) == 0:
                    continue
                lbl = f"{'clean' if ct else 'drift'} × {'conf' if mc else 'no_conf'}"
                tp  = int((g["outcome"] == 1).sum())
                wr  = tp / max(len(g), 1) * 100
                net = g["pnl"].sum()
                print(f"  {lbl:<28} {len(g):>7} {tp:>5} {wr:>5.1f}% {net:>+11.2f}")
        print(f"  {'─'*68}")

    print()


if __name__ == "__main__":
    main()
