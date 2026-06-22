"""Run monthly backtest and write output to a file."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUTPUT = _ROOT / "monthly_out.txt"

from trading.strategies.zz.ustec.strategy import MIN_RR, MIN_SL_PCT, SPREAD_PTS, MAX_FORWARD_BARS
from trading.strategies.zz.ustec.engine import run_backtest
import pandas as pd

START      = "2023-01-01"
END        = "2026-01-01"
CASH       = 150.0
MIN_CONF   = int(sys.argv[1]) if len(sys.argv) > 1 else 2

result = run_backtest(
    start=START, end=END, cash=CASH,
    symbol="ustech", spread=SPREAD_PTS,
    fixed_lot=0, risk_pct=0.01,
    min_rr=MIN_RR, min_sl_pct=MIN_SL_PCT, max_forward_bars=MAX_FORWARD_BARS,
    gradual_filter="all",
    min_confirmations=MIN_CONF,
    silent=True,
)

lines = []
lines.append(f"\nMonthly backtest  {START} to {END}  (min_confirmations={MIN_CONF})")
lines.append(f"  Cash=${CASH:.2f}  risk=1.00%  spread={SPREAD_PTS}pts\n")

if not result:
    lines.append("  No trades generated.")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    sys.exit()

metrics, df = result

df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
df_sorted   = df.sort_values("date").reset_index(drop=True)

from collections import defaultdict
monthly = defaultdict(list)
for _, row in df_sorted.iterrows():
    monthly[row["month"]].append(row)

W   = 12
SEP = "─" * (12 + W * 7 + 6)
lines.append(SEP)
lines.append(f"  {'Month':<10} {'Trades':>{W}} {'Wins':>{W}} {'WR%':>{W}} "
             f"{'Month PnL':>{W}} {'End Equity':>{W}} {'DD%':>{W}}")
lines.append(SEP)

running_eq   = CASH
peak_eq      = CASH
total_trades = 0
total_wins   = 0

for period in sorted(monthly.keys()):
    rows       = monthly[period]
    month_pnl  = sum(r["pnl"] for r in rows)
    trades     = len(rows)
    wins       = sum(1 for r in rows if r["outcome"] == 1)
    wr         = wins / trades * 100 if trades else 0.0
    running_eq = max(running_eq + month_pnl, 0.0)
    peak_eq    = max(peak_eq, running_eq)
    dd_pct     = (running_eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0
    total_trades += trades
    total_wins   += wins
    blown = " BLOWN" if running_eq <= 0.01 else ""
    lines.append(f"  {str(period):<10} {trades:>{W}} {wins:>{W}} {wr:>{W-1}.1f}% "
                 f"{month_pnl:>+{W}.2f} {running_eq:>{W},.2f} {dd_pct:>{W-1}.2f}%{blown}")

lines.append(SEP)
total_wr  = total_wins / total_trades * 100 if total_trades else 0.0
total_pnl = running_eq - CASH
lines.append(f"  {'TOTAL':<10} {total_trades:>{W}} {total_wins:>{W}} {total_wr:>{W-1}.1f}% "
             f"{total_pnl:>+{W}.2f} {running_eq:>{W},.2f}")
lines.append(SEP)
lines.append(f"\n  Start : ${CASH:,.2f}")
lines.append(f"  Final : ${running_eq:,.2f}")
lines.append(f"  Net   : ${total_pnl:+,.2f}")
lines.append(f"  Peak  : ${peak_eq:,.2f}")
lines.append(f"  MaxDD : {metrics.get('max_drawdown_%', 'n/a')}%")
lines.append("")

text = "\n".join(lines)
OUTPUT.write_text(text, encoding="utf-8")
print(text)
