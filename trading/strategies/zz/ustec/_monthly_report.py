"""Monthly breakdown for any backtest run."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

start, end, cash = sys.argv[1], sys.argv[2], float(sys.argv[3])

result = run_backtest(start=start, end=end, cash=cash, fixed_lot=0.05, silent=True)
if not result or isinstance(result, dict):
    sys.exit("No trades.")

_, df = result
df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

monthly = (
    df.groupby("month")
    .apply(lambda g: pd.Series({
        "trades":  len(g),
        "wins":    int((g["outcome"] == 1).sum()),
        "losses":  int((g["outcome"] == -1).sum()),
        "expired": int((g["outcome"] == 0).sum()),
        "WR%":     round((g["outcome"] == 1).sum() / max((g["outcome"] != 0).sum(), 1) * 100, 1),
        "net_pnl": round(g["pnl"].sum(), 2),
    }), include_groups=False)
    .reset_index()
)

year_groups = monthly.copy()
year_groups["year"] = monthly["month"].dt.year

print(f"\n{'='*70}")
print(f"  Monthly Results — USTEC  {start} to {end}  (${cash:.0f} start, 0.05 lot)")
print(f"{'='*70}")
print(f"  {'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'Exp':>4} {'WR%':>6} {'Net PnL':>11}")
print("  " + "-"*60)

prev_year = None
for _, r in monthly.iterrows():
    yr = r["month"].year
    if prev_year is not None and yr != prev_year:
        yr_rows = monthly[monthly["month"].dt.year == prev_year]
        yr_net = yr_rows["net_pnl"].sum()
        yr_w   = yr_rows["wins"].sum()
        yr_t   = yr_rows["trades"].sum()
        yr_wr  = round(yr_w / max((yr_t - yr_rows["expired"].sum()), 1) * 100, 1)
        print(f"  {'─'*60}")
        print(f"  {str(prev_year)+' TOTAL':<10} {int(yr_t):>7} {int(yr_w):>5} {int(yr_rows['losses'].sum()):>5} "
              f"{int(yr_rows['expired'].sum()):>4} {yr_wr:>6.1f} {yr_net:>+11.2f}")
        print(f"  {'─'*60}")
    prev_year = yr
    print(f"  {str(r['month']):<10} {int(r['trades']):>7} {int(r['wins']):>5} {int(r['losses']):>5} "
          f"{int(r['expired']):>4} {r['WR%']:>6.1f} {r['net_pnl']:>+11.2f}")

# Last year total
yr_rows = monthly[monthly["month"].dt.year == prev_year]
yr_net  = yr_rows["net_pnl"].sum()
yr_w    = yr_rows["wins"].sum()
yr_t    = yr_rows["trades"].sum()
yr_wr   = round(yr_w / max((yr_t - yr_rows["expired"].sum()), 1) * 100, 1)
print(f"  {'─'*60}")
print(f"  {str(prev_year)+' TOTAL':<10} {int(yr_t):>7} {int(yr_w):>5} {int(yr_rows['losses'].sum()):>5} "
      f"{int(yr_rows['expired'].sum()):>4} {yr_wr:>6.1f} {yr_net:>+11.2f}")
print(f"  {'='*60}")

total_net  = monthly["net_pnl"].sum()
total_t    = monthly["trades"].sum()
total_w    = monthly["wins"].sum()
total_exp  = monthly["expired"].sum()
total_wr   = round(total_w / max((total_t - total_exp), 1) * 100, 1)
print(f"  {'OVERALL':<10} {int(total_t):>7} {int(total_w):>5} {int(monthly['losses'].sum()):>5} "
      f"{int(total_exp):>4} {total_wr:>6.1f} {total_net:>+11.2f}")
print(f"  {'='*70}")
