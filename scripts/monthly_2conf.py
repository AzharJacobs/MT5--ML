import sys, pandas as pd
from collections import defaultdict
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from trading.strategies.zz.ustec.engine import run_backtest
from trading.strategies.zz.ustec.strategy import MIN_RR, MIN_SL_PCT, SPREAD_PTS, MAX_FORWARD_BARS

CASH = 150.0
result = run_backtest(
    start="2023-01-01", end="2026-01-01", cash=CASH,
    symbol="ustech", spread=SPREAD_PTS,
    fixed_lot=0, risk_pct=0.01,
    min_rr=MIN_RR, min_sl_pct=MIN_SL_PCT, max_forward_bars=MAX_FORWARD_BARS,
    gradual_filter="all", min_confirmations=2, silent=True,
)

if not result:
    print("No trades"); sys.exit()

m, df = result
df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
monthly = defaultdict(list)
for _, row in df.sort_values("date").iterrows():
    monthly[row["month"]].append(row)

running = CASH; peak = CASH; tw = 0; tt = 0
SEP = "-" * 72
print(f"\n3-Year Monthly Backtest | min_confirmations=2 | Start=${CASH:.2f}\n")
print(f"{'Month':<10} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'MonthPnL':>10} {'EndEquity':>10} {'DD%':>6}")
print(SEP)
for p in sorted(monthly):
    rows = monthly[p]
    t = len(rows); w = sum(1 for r in rows if r.outcome == 1)
    pnl = sum(r.pnl for r in rows)
    running = max(running + pnl, 0.0); peak = max(peak, running)
    dd = (running - peak) / peak * 100 if peak else 0
    tw += w; tt += t
    blown = " BLOWN" if running <= 0.01 else ""
    print(f"{str(p):<10} {t:>7} {w:>5} {w/t*100:>5.1f}% {pnl:>+10.2f} {running:>10,.2f} {dd:>5.1f}%{blown}")

print(SEP)
wr = tw / tt * 100 if tt else 0
print(f"{'TOTAL':<10} {tt:>7} {tw:>5} {wr:>5.1f}% {running-CASH:>+10.2f} {running:>10,.2f}")
print(SEP)
dd_key = "max_drawdown_%"
print(f"\nStart   : ${CASH:,.2f}")
print(f"Final   : ${running:,.2f}")
print(f"Net PnL : ${running-CASH:+,.2f}")
print(f"Peak    : ${peak:,.2f}")
print(f"Max DD  : {m[dd_key]}%")
