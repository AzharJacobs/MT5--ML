"""
USTEC ZZ 3-Year Report — 2 positions, lot=0.02, full table format.

Per-month:  trades | W/L + WR% | net PnL | long breakdown | short breakdown
Per-trade:  short note (signal, arrival) + duration
Overall:    WR, avg RR, net PnL, max DD, profit factor, avg win/loss,
            largest win/loss, max consecutive losses
"""

import sys
from pathlib import Path
from itertools import groupby

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, MAX_FORWARD_BARS, MIN_SL_PCT,
    ZONE_MAX_LOSSES, H4_REGIME_FILTER,
    ENABLE_TRAILING, BE_TRIGGER_PTS, BE_BUFFER_PTS, ATR_TRAIL_MULT,
    EXCLUDED_FROM_COUNT, load_raw,
)
from trading.strategies.zz.ustec.engine import run_backtest

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

START      = "2023-01-01"
END        = "2026-01-01"
CASH       = 10_000.0
LOT        = 0.02          # override: live bot config
MAX_POS    = 2             # override: live bot config


def _wr(df):
    return f"{int((df['outcome'] == 1).sum())}W/{int((df['pnl'] < 0).sum())}L " \
           f"{(df['outcome'] == 1).mean() * 100:.0f}%" if len(df) else "—"


def _rr(row):
    sl = abs(row["entry"] - row["sl"])
    tp = abs(row["tp"] - row["entry"])
    return tp / sl if sl > 0 else 0.0


def _note(row):
    arr  = row.get("arrival_type", "?")
    sigs = row.get("signals", "")
    sig1 = sigs.split("|")[0] if sigs else "?"
    return f"{arr[:4]}/{sig1[:8]}"


def _max_dd(equity_list):
    peak = equity_list[0]
    mdd  = 0.0
    for e in equity_list:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def _max_consec_losses(outcomes):
    best = cur = 0
    for o in outcomes:
        if o < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main():
    _cfg      = load_raw()
    trade_cfg = _cfg.get("trade_setup", {})
    use_m15_sl       = bool(trade_cfg.get("use_m15_sl", False))
    m15_sl_atr_floor = float(trade_cfg.get("m15_sl_atr_floor_mult", 0.5))
    max_zone_ht_atr  = float(_cfg.get("zone", {}).get("max_zone_height_atr", 0.0))

    print(f"\nRunning 3-year backtest ({START} → {END})")
    print(f"  lot={LOT}  max_positions={MAX_POS}  spread={SPREAD_PTS}pts  "
          f"min_rr={MIN_RR}  trailing={ENABLE_TRAILING}  be={BE_TRIGGER_PTS}pts")

    result = run_backtest(
        start=START,
        end=END,
        cash=CASH,
        min_rr=MIN_RR,
        max_forward_bars=MAX_FORWARD_BARS,
        symbol="ustech",
        spread=SPREAD_PTS,
        fixed_lot=LOT,
        directional_filter=True,
        allow_neutral=True,
        h4_swing_left=2,
        h4_swing_right=2,
        min_confirmations=1,
        excluded_from_count=list(EXCLUDED_FROM_COUNT),
        zone_max_losses=ZONE_MAX_LOSSES,
        h4_regime_filter=H4_REGIME_FILTER,
        min_sl_pct=MIN_SL_PCT,
        enable_trailing=ENABLE_TRAILING,
        be_trigger_pts=BE_TRIGGER_PTS,
        be_buffer_pts=BE_BUFFER_PTS,
        atr_trail_mult=ATR_TRAIL_MULT,
        use_m15_sl=use_m15_sl,
        m15_sl_atr_floor_mult=m15_sl_atr_floor,
        max_zone_height_atr=max_zone_ht_atr,
        silent=True,
        max_positions=MAX_POS,
    )

    if not result or isinstance(result, dict):
        print("ERROR: no result returned.")
        return

    _, df = result
    df = df.copy()
    df["entry_dt"]   = pd.to_datetime(df["date"])
    df["exit_dt"]    = pd.to_datetime(df["exit_date"])
    df["year"]       = df["entry_dt"].dt.year
    df["month"]      = df["entry_dt"].dt.month
    df["duration_h"] = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 3600
    df["rr_actual"]  = df.apply(_rr, axis=1)

    equity_list = df["equity"].dropna().tolist()
    equity_list = [CASH] + equity_list

    # ══════════════════════════════════════════════════════════════════════════
    # PER-MONTH TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 100)
    print("  MONTHLY BREAKDOWN")
    print("═" * 100)
    hdr = (f"  {'Month':<10}{'Trades':>7}{'W/L':>10}{'WR%':>7}{'Net PnL':>11}"
           f"{'Longs':>14}{'L-WR%':>8}{'Shorts':>14}{'S-WR%':>8}")
    print(hdr)
    print("  " + "-" * 98)

    year_totals: dict = {}

    for year in [2023, 2024, 2025]:
        ydf = df[df["year"] == year]
        if ydf.empty:
            continue
        for m in range(1, 13):
            mdf = ydf[ydf["month"] == m]
            if mdf.empty:
                continue
            total  = len(mdf)
            wins   = int((mdf["outcome"] == 1).sum())
            losses = int((mdf["pnl"] < 0).sum())
            wr     = wins / total * 100
            net    = mdf["pnl"].sum()

            ldf = mdf[mdf["side"] == "buy"]
            sdf = mdf[mdf["side"] == "sell"]
            lw  = int((ldf["outcome"] == 1).sum()) if len(ldf) else 0
            sw  = int((sdf["outcome"] == 1).sum()) if len(sdf) else 0
            lwr = lw / len(ldf) * 100 if len(ldf) else float("nan")
            swr = sw / len(sdf) * 100 if len(sdf) else float("nan")

            lstr = f"{len(ldf)}T {lw}W/{len(ldf)-lw}L"
            sstr = f"{len(sdf)}T {sw}W/{len(sdf)-sw}L"

            print(f"  {MONTH_NAMES[m]} {year:<7}"
                  f"{total:>7}"
                  f"   {wins}W/{losses}L{'':<2}"
                  f"{wr:>7.0f}%"
                  f"  ${net:>+9.2f}"
                  f"  {lstr:>14}"
                  f"  {lwr:>6.0f}%" if not pd.isna(lwr) else
                  f"  {MONTH_NAMES[m]} {year:<7}{total:>7}   {wins}W/{losses}L  "
                  f"{wr:>6.0f}%  ${net:>+9.2f}  {lstr:>14}       —"
                  , end="")
            if not pd.isna(swr):
                print(f"  {sstr:>14}  {swr:>6.0f}%")
            else:
                print(f"  {'—':>14}       —")

            if year not in year_totals:
                year_totals[year] = {"t": 0, "w": 0, "l": 0, "pnl": 0.0}
            year_totals[year]["t"]   += total
            year_totals[year]["w"]   += wins
            year_totals[year]["l"]   += losses
            year_totals[year]["pnl"] += net

        # Year sub-total
        yt = year_totals.get(year, {})
        if yt:
            print("  " + "-" * 98)
            print(f"  {year} TOTAL  "
                  f"{yt['t']:>7}   {yt['w']}W/{yt['l']}L  "
                  f"{yt['w']/max(yt['t'],1)*100:>6.0f}%  ${yt['pnl']:>+9.2f}")
            print()

    # ══════════════════════════════════════════════════════════════════════════
    # PER-TRADE TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print("═" * 100)
    print("  TRADE LOG")
    print("═" * 100)
    hdr2 = (f"  {'#':>4}  {'Date':>11}  {'Side':>5}  {'Arriv/Sig':>14}  "
            f"{'Entry':>8}  {'SL':>8}  {'TP':>8}  {'Exit':>8}  "
            f"{'Outc':>5}  {'PnL':>9}  {'Dur h':>6}  Note")
    print(hdr2)
    print("  " + "-" * 98)

    for idx, (_, t) in enumerate(df.sort_values("entry_dt").iterrows(), 1):
        oc_str = "WIN " if t["outcome"] == 1 else ("LOSS" if t["pnl"] < 0 else "BE  ")
        note   = _note(t)
        print(f"  {idx:>4}  {t['entry_dt'].strftime('%Y-%m-%d'):>11}  "
              f"{'LONG' if t['side']=='buy' else 'SHORT':>5}  "
              f"{note:>14}  "
              f"{t['entry']:>8.1f}  {t['sl']:>8.1f}  {t['tp']:>8.1f}  {t['exit']:>8.1f}  "
              f"{oc_str:>5}  ${t['pnl']:>+8.2f}  {t['duration_h']:>6.1f}h")

    # ══════════════════════════════════════════════════════════════════════════
    # OVERALL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    total  = len(df)
    wins   = int((df["outcome"] == 1).sum())
    losses = int((df["pnl"] < 0).sum())
    wr_pct = wins / total * 100

    winning_trades = df[df["pnl"] > 0]
    losing_trades  = df[df["pnl"] < 0]
    gross_profit   = winning_trades["pnl"].sum()
    gross_loss     = abs(losing_trades["pnl"].sum())
    profit_factor  = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win  = winning_trades["pnl"].mean() if len(winning_trades) else 0.0
    avg_loss = losing_trades["pnl"].mean()  if len(losing_trades)  else 0.0
    big_win  = df["pnl"].max()
    big_loss = df["pnl"].min()
    net_pnl  = df["pnl"].sum()
    avg_rr   = df["rr_actual"].mean()

    max_dd   = _max_dd(equity_list)
    consec_l = _max_consec_losses(df.sort_values("entry_dt")["pnl"].tolist())

    print("\n" + "═" * 60)
    print("  OVERALL SUMMARY")
    print("═" * 60)
    print(f"  {'Period':<28}  {START} – {END}")
    print(f"  {'Lot size':<28}  {LOT}")
    print(f"  {'Max simultaneous positions':<28}  {MAX_POS}")
    print(f"  {'Cash':<28}  ${CASH:,.0f}")
    print("  " + "-" * 58)
    print(f"  {'Total trades':<28}  {total}")
    print(f"  {'Wins / Losses / BE':<28}  {wins}W / {losses}L / {total-wins-losses}BE")
    print(f"  {'Win rate':<28}  {wr_pct:.1f}%")
    print(f"  {'Avg planned RR':<28}  {avg_rr:.2f}:1")
    print(f"  {'Net PnL':<28}  ${net_pnl:+,.2f}")
    print(f"  {'Max Drawdown':<28}  {max_dd:.2f}%")
    print(f"  {'Profit Factor':<28}  {profit_factor:.2f}")
    print(f"  {'Avg Win':<28}  ${avg_win:+.2f}")
    print(f"  {'Avg Loss':<28}  ${avg_loss:+.2f}")
    print(f"  {'Largest Win':<28}  ${big_win:+.2f}")
    print(f"  {'Largest Loss':<28}  ${big_loss:+.2f}")
    print(f"  {'Max Consecutive Losses':<28}  {consec_l}")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
