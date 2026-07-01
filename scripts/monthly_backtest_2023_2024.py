#!/usr/bin/env python3
"""
Monthly breakdown backtest: USTECm, Jan 2023 – Dec 2024.

For each calendar month:
  - Total trades / Wins / Losses / Expired
  - Win rate
  - Avg theoretical RR  |  Avg realized RR
  - TP-hit / SL-hit / Expired counts
  - Net PnL
  - Max intra-month drawdown (equity peak-to-trough from trades closed that month)

Usage:
    python scripts/monthly_backtest_2023_2024.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

# ── Run the full 2-year backtest ──────────────────────────────────────────────
print("Running backtest  Jan 2023 → Dec 2024 …  (this takes ~60-90 s)\n")

result = run_backtest(
    start="2023-01-01",
    end="2025-01-01",
    cash=150.0,
    symbol="ustech",
    be_trigger_r=1.0,
    lock_trigger_r=2.0,
    be_buffer_pts=5.0,
    silent=True,
)

if not result:
    print("Backtest returned no data. Check DB connection and date range.")
    sys.exit(1)

metrics, df = result

# ── Pre-compute per-trade RR fields ──────────────────────────────────────────
# Theoretical RR: distance to TP / distance to SL (at signal price)
df["rr_theory"] = abs(df["tp"] - df["entry"]) / abs(df["entry"] - df["sl"]).replace(0, float("nan"))

# Realized RR: signed — positive = moved in our favour, negative = moved against
def _realized_rr(row):
    sl_dist = abs(row["entry"] - row["sl"])
    if sl_dist == 0:
        return 0.0
    if row["side"] == "buy":
        return (row["exit"] - row["entry"]) / sl_dist
    else:
        return (row["entry"] - row["exit"]) / sl_dist

df["rr_realized"] = df.apply(_realized_rr, axis=1)

# Group by entry month
df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

# Build the equity curve starting from the initial cash
start_cash = float(metrics["start_cash"].replace("$", "").replace(",", ""))


def _monthly_maxdd(grp_equity: pd.Series, equity_before: float) -> float:
    """
    Peak-to-trough drawdown (%) across trades closed in this month.
    equity_before is the running equity at the start of the month.
    """
    curve = [equity_before] + list(grp_equity)
    s = pd.Series(curve)
    dd = ((s - s.cummax()) / s.cummax()).min() * 100
    return round(dd, 2)


# ── Build monthly stats ───────────────────────────────────────────────────────
# All 24 calendar months Jan 2023 – Dec 2024 (including zero-trade months)
all_months = pd.period_range("2023-01", "2024-12", freq="M")
trade_months = set(df["month"].unique())

# We need the running equity at the START of each month.
# df["equity"] is the equity AFTER each trade closes (sorted by exit_date / close order).
# Sort trades by exit_date to reconstruct the equity progression.
df_sorted = df.sort_values("exit_date").reset_index(drop=True)

# Map each trade to its entry-month
df_sorted["entry_month"] = pd.to_datetime(df_sorted["date"]).dt.to_period("M")

rows = []
running_equity = start_cash   # equity carried into the current month

for m in all_months:
    grp = df_sorted[df_sorted["entry_month"] == m]

    if m not in trade_months or grp.empty:
        rows.append({
            "month": str(m), "trades": 0, "wins": 0, "losses": 0, "expired": 0,
            "wr": float("nan"),
            "tp_hits": 0, "sl_hits": 0, "exp_hits": 0,
            "avg_rr_theory": float("nan"),
            "avg_rr_realized": float("nan"),
            "net_pnl": 0.0,
            "max_dd_pct": 0.0,
            "flag": "NO SETUPS",
        })
        continue

    total   = len(grp)
    wins    = int((grp["outcome"] == 1).sum())
    losses  = int((grp["outcome"] == -1).sum())
    expired = int((grp["outcome"] == 0).sum())
    wr      = wins / total * 100 if total > 0 else float("nan")

    avg_rr_theory   = grp["rr_theory"].mean()
    avg_rr_realized = grp["rr_realized"].mean()

    net_pnl = grp["pnl"].sum()

    # Max intra-month drawdown from equity curve
    max_dd = _monthly_maxdd(grp["equity"], running_equity)

    # Advance running equity to end of this month
    running_equity = float(grp["equity"].iloc[-1])

    flag = ""
    if wins == 0 and losses > 0:
        flag = "100% LOSS RATE"

    rows.append({
        "month": str(m),
        "trades": total,
        "wins": wins,
        "losses": losses,
        "expired": expired,
        "wr": wr,
        "tp_hits": wins,
        "sl_hits": losses,
        "exp_hits": expired,
        "avg_rr_theory": avg_rr_theory,
        "avg_rr_realized": avg_rr_realized,
        "net_pnl": net_pnl,
        "max_dd_pct": max_dd,
        "flag": flag,
    })

result_df = pd.DataFrame(rows)

# ── Print table ───────────────────────────────────────────────────────────────
DIVIDER = "─" * 108
HEADER = (
    f"{'Month':<10} "
    f"{'Tr':>4} {'W':>3} {'L':>3} {'E':>3} "
    f"{'WR%':>6}  "
    f"{'TP':>3}{'SL':>4}{'Exp':>4}  "
    f"{'AvgRR(T)':>9}{'AvgRR(R)':>10}  "
    f"{'Net PnL':>10}  "
    f"{'MaxDD%':>7}  "
    f"{'Flag'}"
)

print("\n" + DIVIDER)
print("  USTECm  |  Backtest Monthly Report  |  Jan 2023 – Dec 2024  |  BE@1R / Lock@2R")
print(DIVIDER)
print(HEADER)
print(DIVIDER)

flagged_months = []

for _, r in result_df.iterrows():
    wr_str   = f"{r['wr']:5.1f}%" if not pd.isna(r["wr"]) else "   N/A"
    rr_t_str = f"{r['avg_rr_theory']:8.2f}" if not pd.isna(r["avg_rr_theory"]) else "      N/A"
    rr_r_str = f"{r['avg_rr_realized']:8.2f}" if not pd.isna(r["avg_rr_realized"]) else "      N/A"
    pnl_str  = f"${r['net_pnl']:+9.2f}"
    dd_str   = f"{r['max_dd_pct']:6.2f}%"

    line = (
        f"{r['month']:<10} "
        f"{int(r['trades']):>4} {int(r['wins']):>3} {int(r['losses']):>3} {int(r['expired']):>3} "
        f"{wr_str}  "
        f"{int(r['tp_hits']):>3}{int(r['sl_hits']):>4}{int(r['exp_hits']):>4}  "
        f"{rr_t_str}{rr_r_str}  "
        f"{pnl_str}  "
        f"{dd_str}  "
        f"{r['flag']}"
    )
    print(line)

    # Year separator
    if r["month"] in ("2023-12", "2024-12"):
        print(DIVIDER)

# ── 2-year totals ─────────────────────────────────────────────────────────────
total_trades   = int(result_df["trades"].sum())
total_wins     = int(result_df["wins"].sum())
total_losses   = int(result_df["losses"].sum())
total_expired  = int(result_df["expired"].sum())
total_wr       = total_wins / total_trades * 100 if total_trades else 0.0
total_pnl      = result_df["net_pnl"].sum()
avg_rr_t_all   = df_sorted["rr_theory"].mean()
avg_rr_r_all   = df_sorted["rr_realized"].mean()

# Full-period max drawdown from reported metrics
full_dd = metrics["max_drawdown_%"]

print(f"  {'2-YEAR TOTAL':<9} "
      f"{total_trades:>4} {total_wins:>3} {total_losses:>3} {total_expired:>3} "
      f"{total_wr:5.1f}%  "
      f"{total_wins:>3}{total_losses:>4}{total_expired:>4}  "
      f"{avg_rr_t_all:8.2f}{avg_rr_r_all:8.2f}  "
      f"${total_pnl:+9.2f}  "
      f"{full_dd:>6}%")
print(DIVIDER)

# ── Flagged months ────────────────────────────────────────────────────────────
no_setup_months  = result_df[result_df["flag"] == "NO SETUPS"]["month"].tolist()
loss_only_months = result_df[result_df["flag"] == "100% LOSS RATE"]["month"].tolist()

print()
if no_setup_months:
    print(f"  ⚑  Months with 0 trades (no setups): {', '.join(no_setup_months)}")
else:
    print("  ✓  No months with 0 trades")

if loss_only_months:
    print(f"  ⚑  Months with 100% loss rate:       {', '.join(loss_only_months)}")
else:
    print("  ✓  No months with 100% loss rate")

print()
print("  Legend:")
print("    Tr=Total trades  W=Wins  L=Losses  E=Expired (time-out)")
print("    TP/SL/Exp = TP hits / SL hits / Expired counts")
print("    AvgRR(T) = avg theoretical RR (TP dist / SL dist at entry)")
print("    AvgRR(R) = avg realized RR (exit dist / SL dist, signed)")
print("    MaxDD%   = peak-to-trough % drawdown from trades in that month")
print()
print(f"  Full-period settings: {metrics.get('period','?')} | "
      f"cash={metrics.get('start_cash','?')} | "
      f"final={metrics.get('final_equity','?')} | "
      f"max_dd={full_dd}%")
print()
