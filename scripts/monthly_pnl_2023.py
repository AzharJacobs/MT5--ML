import sys
sys.path.insert(0, '.')
from trading.strategies.zz.ustec.engine import run_backtest

result = run_backtest(
    start="2023-01-01", end="2024-01-01", cash=150.0,
    symbol="ustech", spread=2.0, fixed_lot=0.05, min_rr=1.5,
    min_sl_pct=0.10, zone_max_losses=3, directional_filter=True,
    require_leave_and_return=True, cooldown_bars=15,
    min_confirmations=1, h4_regime_filter=False,
    contract_size_override=1.0,
)
if not result:
    print("No trades")
    sys.exit()

metrics, df = result
df["month"] = df["date"].dt.to_period("M")
monthly = df.groupby("month").agg(
    trades=("pnl", "count"),
    wins=("outcome", lambda x: (x == 1).sum()),
    pnl=("pnl", "sum"),
).reset_index()
monthly["wr"]  = (monthly["wins"] / monthly["trades"] * 100).round(1)
monthly["pnl"] = monthly["pnl"].round(2)

print()
print("  Month        Trades  Wins    WR%       PnL    Equity")
print("  " + "-" * 56)
cum = 150.0
for _, r in monthly.iterrows():
    cum += r["pnl"]
    sign = "+" if r["pnl"] >= 0 else ""
    print(
        f"  {str(r['month']):<12} {int(r['trades']):>5}  {int(r['wins']):>4}"
        f"  {r['wr']:>5.1f}%  {sign}{r['pnl']:>7.2f}   ${cum:.2f}"
    )
print("  " + "-" * 56)
print(f"  TOTAL        {int(monthly['trades'].sum()):>5}  {int(monthly['wins'].sum()):>4}          ${df['pnl'].sum():>+7.2f}")
