"""
monthly_trades.py — Full per-trade breakdown organised by month.

For each scenario (baseline / 1.5x / 2.0x) prints every trade inside
its calendar month with date, side, entry/SL/TP/exit, outcome, PnL,
clean_tap (CT), m15_confluent (MC), tag_mult, and signals.
"""

import sys
import pandas as pd

from trading.strategies.zz.ustec.engine import run_backtest

SHARED = dict(
    symbol             = "ustech",
    cash               = 10_000.0,
    min_rr             = 1.5,
    nml_filter         = False,
    stacked_confluence = False,
    silent             = True,
)
START      = "2022-01-01"
END        = "2025-01-01"
START_CASH = 10_000.0

SCENARIOS = [
    ("baseline",   1.0),
    ("sized_1.5x", 1.5),
    ("sized_2.0x", 2.0),
]


def _out(o):
    return "TP " if o == 1 else ("SL " if o == -1 else "Exp")


def _run(label, mult):
    print(f"  running {label} ...", flush=True)
    res = run_backtest(start=START, end=END, drift_nc_mult=mult, **SHARED)
    if not res or not isinstance(res, tuple):
        return None, None
    return res


def _print_scenario(label, mult, metrics, df_raw):
    df = df_raw.copy()
    df["_dt"]   = pd.to_datetime(df["date"])
    df["month"] = df["_dt"].dt.to_period("M")
    df = df.sort_values("_dt").reset_index(drop=True)

    W = 108
    SEP = "─" * W

    print(f"\n\n{'═'*W}")
    print(f"  SCENARIO : {label}   (drift×NC sizing = {mult}×)")
    print(f"  Period   : {START} → {END}   |   {len(df)} trades total")
    print(f"{'═'*W}")

    running  = START_CASH
    tot_tp = tot_sl = tot_exp = 0

    for month, grp in df.groupby("month", sort=True):
        grp   = grp.sort_values("_dt")
        tp    = int((grp["outcome"] == 1).sum())
        sl    = int((grp["outcome"] == -1).sum())
        exp   = int((grp["outcome"] == 0).sum())
        n     = len(grp)
        wr    = tp / max(n, 1) * 100
        net   = grp["pnl"].sum()
        running += net
        tot_tp  += tp
        tot_sl  += sl
        tot_exp += exp

        # ── month header ──────────────────────────────────────────────────────
        hdr = (f" {month}   {n} trade{'s' if n>1 else ''}  "
               f"{tp}W / {sl}L  WR={wr:.0f}%  "
               f"month PnL={net:+.2f}  running equity={running:,.2f} ")
        print(f"\n┌{'─'*(W-2)}┐")
        print(f"│{hdr:<{W-2}}│")
        print(f"└{'─'*(W-2)}┘")

        # ── column header ─────────────────────────────────────────────────────
        print(f"  {'Date':<11} {'Side':<5} {'Entry':>8} {'SL':>8} {'TP':>8} "
              f"{'Exit':>8} {'Out':<4} {'PnL':>9}  "
              f"{'CT':<3}{'MC':<3}{'Mult':<5}  Signals")
        print(f"  {SEP[:W-2]}")

        # ── individual trades ─────────────────────────────────────────────────
        for _, row in grp.iterrows():
            ct   = "Y" if row.get("clean_tap",     False) else "N"
            mc   = "Y" if row.get("m15_confluent", False) else "N"
            tmlt = f"{row.get('tag_mult', 1.0):.1f}x"
            sigs = str(row.get("signals", "")).replace("|", " | ")
            date_s = str(row["_dt"])[:10]

            print(f"  {date_s:<11} {str(row['side']).upper():<5} "
                  f"{row['entry']:>8.1f} {row['sl']:>8.1f} "
                  f"{row['tp']:>8.1f} {row['exit']:>8.1f} "
                  f"{_out(row['outcome']):<4} {row['pnl']:>+9.2f}  "
                  f"{ct:<3}{mc:<3}{tmlt:<5}  {sigs}")

    # ── grand total ───────────────────────────────────────────────────────────
    tot_n  = len(df)
    tot_wr = tot_tp / max(tot_n, 1) * 100
    net_tot = df["pnl"].sum()
    print(f"\n  {'═'*W}")
    print(f"  TOTAL    {tot_n} trades    {tot_tp}W / {tot_sl}L / {tot_exp}Exp    "
          f"WR={tot_wr:.1f}%    Net PnL={net_tot:+.2f}    "
          f"Final Equity={START_CASH+net_tot:,.2f}")
    print(f"  Max drawdown: {metrics.get('max_drawdown_%','?')}%    "
          f"Avg win: {metrics.get('avg_win_$','?')}    "
          f"Avg loss: {metrics.get('avg_loss_$','?')}")
    print(f"  {'═'*W}")


def main():
    print(f"\nLoading {len(SCENARIOS)} scenarios  {START} → {END} ...\n")

    for label, mult in SCENARIOS:
        m, df = _run(label, mult)
        if m is None:
            print(f"  {label}: no results\n")
            continue
        _print_scenario(label, mult, m, df)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
