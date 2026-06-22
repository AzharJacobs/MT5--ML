"""Quick analysis of retest vs gradual trade breakdown."""
import sys
sys.path.insert(0, '.')
from trading.strategies.zz.ustec.engine import run_backtest

r = run_backtest(start='2023-01-01', end='2026-01-01', silent=True)
df = r['trades']
rt = df[df['arrival_type'] == 'retest'].copy()
gr = df[df['arrival_type'] == 'gradual'].copy()

print("=== RETEST DEEP DIVE ===")
print(f"Total retest: {len(rt)}  Wins: {(rt.outcome==1).sum()}  WR: {100*(rt.outcome==1).mean():.1f}%  Net: ${rt.pnl.sum():.0f}")
print(f"Total gradual: {len(gr)}  Wins: {(gr.outcome==1).sum()}  WR: {100*(gr.outcome==1).mean():.1f}%  Net: ${gr.pnl.sum():.0f}")

print("\nRETEST — by signal fired:")
sig_wr = {}
for _, row in rt.iterrows():
    for s in str(row['signals']).split('|'):
        s = s.strip()
        if s not in sig_wr:
            sig_wr[s] = [0, 0]
        sig_wr[s][1] += 1
        if row['outcome'] == 1:
            sig_wr[s][0] += 1
for s, (w, t) in sorted(sig_wr.items(), key=lambda x: -x[1][1]):
    print(f"  {s:<22} {t:>3} trades  WR={100*w/t:.0f}%")

print("\nRETEST — by confirmation count:")
for n in sorted(rt['confirmations'].unique()):
    g = rt[rt['confirmations'] == n]
    wr = 100 * (g['outcome'] == 1).mean()
    print(f"  {n} conf   {len(g):>3} trades  WR={wr:.0f}%  net=${g['pnl'].sum():.0f}")

print("\nGRADUAL — by confirmation count:")
for n in sorted(gr['confirmations'].unique()):
    g = gr[gr['confirmations'] == n]
    wr = 100 * (g['outcome'] == 1).mean()
    print(f"  {n} conf   {len(g):>3} trades  WR={wr:.0f}%  net=${g['pnl'].sum():.0f}")

print("\nRETEST — by h4_bias:")
for b in sorted(rt['h4_bias'].unique()):
    g = rt[rt['h4_bias'] == b]
    wr = 100 * (g['outcome'] == 1).mean()
    print(f"  {b:<14}  {len(g):>3} trades  WR={wr:.0f}%  net=${g['pnl'].sum():.0f}")

print("\nRETEST — by prior_bucket:")
for b in sorted(rt['prior_bucket'].unique()):
    g = rt[rt['prior_bucket'] == b]
    wr = 100 * (g['outcome'] == 1).mean()
    print(f"  {b:<16}  {len(g):>3} trades  WR={wr:.0f}%  net=${g['pnl'].sum():.0f}")

print("\nRETEST — by zone_fresh:")
for fresh in [True, False]:
    g = rt[rt['zone_fresh'] == fresh]
    if len(g) == 0:
        continue
    wr = 100 * (g['outcome'] == 1).mean()
    label = "fresh" if fresh else "stale"
    print(f"  {label:<8}  {len(g):>3} trades  WR={wr:.0f}%  net=${g['pnl'].sum():.0f}")

print("\nAvg RR at entry:")
rt2 = rt.copy()
rt2['rr_entry'] = abs(rt2['tp'] - rt2['entry']) / (abs(rt2['entry'] - rt2['sl']).clip(lower=0.01))
gr2 = gr.copy()
gr2['rr_entry'] = abs(gr2['tp'] - gr2['entry']) / (abs(gr2['entry'] - gr2['sl']).clip(lower=0.01))
print(f"  Retest  avg RR: {rt2['rr_entry'].mean():.2f}  median: {rt2['rr_entry'].median():.2f}")
print(f"  Gradual avg RR: {gr2['rr_entry'].mean():.2f}  median: {gr2['rr_entry'].median():.2f}")
