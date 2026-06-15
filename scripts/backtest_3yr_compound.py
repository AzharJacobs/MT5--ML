"""
3-year compounding backtest — $150 start, equity carries over each year.
Applies all 4 changes: structural SL, min_sl_pct=0.10, zone_exhausted, TP headroom.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.zz.ustec.engine import run_backtest

YEARS = [
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

COMMON = dict(
    symbol                  = "ustech",
    spread                  = 2.0,
    fixed_lot               = 0.05,
    min_rr                  = 1.5,
    min_sl_pct              = 0.10,
    zone_max_losses         = 3,
    directional_filter      = True,
    require_leave_and_return= True,
    cooldown_bars           = 15,
    min_confirmations       = 1,
    h4_regime_filter        = False,
    contract_size_override  = 1.0,   # Exness USTECm actual contract_size (matches live bot)
)

equity = 150.0
year_results = []

for start, end in YEARS:
    print(f"\n{'='*60}")
    print(f"  YEAR {start[:4]}  |  starting equity = ${equity:,.2f}")
    print(f"{'='*60}")
    result = run_backtest(start=start, end=end, cash=equity, **COMMON)
    if not result:
        print(f"No trades — equity unchanged at ${equity:,.2f}")
        year_results.append({"year": start[:4], "start": equity, "end": equity, "pnl": 0.0})
        continue
    metrics, df_t = result
    final = df_t["equity"].iloc[-1]
    pnl   = final - equity
    year_results.append({"year": start[:4], "start": equity, "end": final, "pnl": pnl})
    equity = final

print(f"\n{'='*60}")
print(f"  3-YEAR COMPOUND SUMMARY")
print(f"{'='*60}")
print(f"  {'Year':<8} {'Start':>12} {'End':>12} {'PnL':>12}")
print(f"  {'-'*46}")
for r in year_results:
    print(f"  {r['year']:<8} ${r['start']:>10,.2f} ${r['end']:>10,.2f} ${r['pnl']:>+10,.2f}")
print(f"  {'-'*46}")
start_cash = year_results[0]["start"]
final_cash = year_results[-1]["end"]
total_pnl  = final_cash - start_cash
total_ret  = (final_cash / start_cash - 1) * 100 if start_cash > 0 else 0
print(f"  {'TOTAL':<8} ${start_cash:>10,.2f} ${final_cash:>10,.2f} ${total_pnl:>+10,.2f}  ({total_ret:+.1f}%)")
print()
