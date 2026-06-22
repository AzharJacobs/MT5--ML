"""One-off: analyse how far trades travel toward TP over the 3-year run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from trading.strategies.zz.ustec.strategy import MIN_RR, MIN_SL_PCT, SPREAD_PTS, MAX_FORWARD_BARS
from trading.strategies.zz.ustec.engine import run_backtest

result = run_backtest(
    start="2023-01-01", end="2026-01-01", cash=150,
    symbol="ustech", spread=SPREAD_PTS,
    fixed_lot=0, risk_pct=0.01,
    min_rr=MIN_RR, min_sl_pct=MIN_SL_PCT, max_forward_bars=MAX_FORWARD_BARS,
    gradual_filter="all",
    silent=True,
)

if not result:
    print("No trades")
    sys.exit()

_, df = result

df["tp_dist"]    = abs(df["tp"] - df["entry"])
df["favour_pct"] = df["max_favour"] / df["tp_dist"] * 100

total  = len(df)
r50    = int((df["favour_pct"] >= 50).sum())
r75    = int((df["favour_pct"] >= 75).sum())
r100   = int((df["outcome"] == 1).sum())
below50 = total - r50
only_50 = int(((df["favour_pct"] >= 50) & (df["favour_pct"] < 75)).sum())
only_75 = int(((df["favour_pct"] >= 75) & (df["favour_pct"] < 100)).sum())

print(f"\nTotal trades   : {total}")
print(f"")
print(f"Reached 50%+   : {r50:>4}  ({r50/total*100:.1f}%)")
print(f"Reached 75%+   : {r75:>4}  ({r75/total*100:.1f}%)")
print(f"Hit TP (100%)  : {r100:>4}  ({r100/total*100:.1f}%)")
print(f"")
print(f"50%+ but no TP : {r50 - r100:>4}  ({(r50-r100)/total*100:.1f}% of all trades)")
print(f"75%+ but no TP : {r75 - r100:>4}  ({(r75-r100)/total*100:.1f}% of all trades)")
print(f"")
print(f"--- Bucket breakdown ---")
print(f"< 50%           : {below50:>4}  ({below50/total*100:.1f}%)")
print(f"50% - 74%       : {only_50:>4}  ({only_50/total*100:.1f}%)")
print(f"75% - 99%       : {only_75:>4}  ({only_75/total*100:.1f}%)")
print(f"100% (TP hit)   : {r100:>4}  ({r100/total*100:.1f}%)")

# SL adverse analysis per TP-favour bucket
df["sl_dist"]    = abs(df["sl"] - df["entry"])
df["adverse_pct"] = df["max_adverse"] / df["sl_dist"] * 100

buckets = [
    ("< 50% of TP",   df["favour_pct"] <  50),
    ("50% - 74%",    (df["favour_pct"] >= 50) & (df["favour_pct"] < 75)),
    ("75% - 99%",    (df["favour_pct"] >= 75) & (df["favour_pct"] < 100)),
    ("Hit TP 100%",   df["outcome"] == 1),
]

W = 8
print(f"\n{'─'*72}")
print(f"  Max adverse move as % of SL distance — by TP-favour bucket")
print(f"{'─'*72}")
print(f"  {'Bucket':<14} {'Trades':>{W}} {'Avg%':>{W}} {'Median%':>{W}} {'Min%':>{W}} {'Max%':>{W}} {'<25%SL':>{W}} {'>75%SL':>{W}}")
print(f"{'─'*72}")
for label, mask in buckets:
    g = df[mask]
    if len(g) == 0:
        print(f"  {label:<14} {'0':>{W}}")
        continue
    ap = g["adverse_pct"]
    lt25 = int((ap < 25).sum())
    gt75 = int((ap > 75).sum())
    print(f"  {label:<14} {len(g):>{W}} {ap.mean():>{W}.1f} {ap.median():>{W}.1f} {ap.min():>{W}.1f} {ap.max():>{W}.1f} {lt25:>{W}} {gt75:>{W}}")
print(f"{'─'*72}")

# ── Pattern investigation: what separates winners from losers ─────────────────
import pandas as pd

def wr_table(label, col, df_all):
    print(f"\n  {'─'*58}")
    print(f"  {label}")
    print(f"  {'─'*58}")
    print(f"  {'Value':<22} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'Net PnL':>10}")
    print(f"  {'─'*58}")
    for val, grp in df_all.groupby(col, sort=False):
        t  = len(grp)
        w  = int((grp["outcome"] == 1).sum())
        wr = w / t * 100
        net = grp["pnl"].sum()
        print(f"  {str(val):<22} {t:>7} {w:>6} {wr:>6.1f}% {net:>+10.2f}")
    print(f"  {'─'*58}")

df["hour"]    = pd.to_datetime(df["date"]).dt.hour
df["session"] = df["hour"].apply(
    lambda h: "Asia    (00-07)" if h < 7
    else ("London  (07-12)" if h < 12
    else ("NY-Open (12-17)" if h < 17
    else ("NY-Late (17-21)" if h < 21
    else "After   (21-24)")))
)
df["dow"] = pd.to_datetime(df["date"]).dt.day_name()

wr_table("BY SESSION (entry hour UTC)", "session", df)
wr_table("BY DAY OF WEEK", "dow", df)
wr_table("BY DIRECTION", "side", df)
wr_table("BY H4 BIAS", "h4_bias", df)
wr_table("BY ARRIVAL TYPE", "arrival_type", df)
wr_table("BY ZONE FRESH", "zone_fresh", df)
wr_table("BY CONFIRMATION COUNT", "confirmations", df)
wr_table("BY PRIOR OUTCOME", "prior_bucket", df)

print(f"\n  {'─'*58}")
print(f"  BY SIGNAL PRESENT IN TRADE")
print(f"  {'─'*58}")
print(f"  {'Signal':<22} {'Present':>7} {'Wins':>6} {'WR%':>7} {'Net PnL':>10}")
print(f"  {'─'*58}")
all_sigs = set()
for s in df["signals"]:
    all_sigs.update(s.split("|"))
all_sigs.discard("")
for sig in sorted(all_sigs):
    mask = df["signals"].str.contains(sig, regex=False)
    grp  = df[mask]
    t    = len(grp)
    w    = int((grp["outcome"] == 1).sum())
    wr   = w / t * 100
    net  = grp["pnl"].sum()
    print(f"  {sig:<22} {t:>7} {w:>6} {wr:>6.1f}% {net:>+10.2f}")
print(f"  {'─'*58}")
