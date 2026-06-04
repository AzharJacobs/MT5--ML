"""
analyse_prior_outcome.py

Reads the leave-and-return baseline trade logs and splits trades into three buckets
based on the prior outcome on the same zone:
  1. first_attempt  — no prior trade on this zone
  2. post_win       — most recent prior trade on this zone was a WIN  (outcome=1)
  3. post_loss      — most recent prior trade on this zone was a LOSS (outcome=-1)

Zone identity is approximated from the printed SL value.
  buy  trade: zone.bottom ≈ sl * 1.002  (SL was set just below zone.bottom)
  sell trade: zone.top    ≈ sl / 1.002  (SL was set just above zone.top)
Zones are clustered at ±100-pt resolution (200-pt buckets) to absorb the small
rebase shift between entries on the same zone.

Expired trades (outcome=0) are tagged post_expired and excluded from the three
main buckets to keep comparisons clean.
"""

import re, pathlib

FILES = {
    '2023': r'C:\Users\AZHARJ~1\AppData\Local\Temp\claude\f--MT5--ML\f161b91a-785e-4060-90fa-30e0efd89fe4\tasks\bfso8f1lq.output',
    '2024': r'C:\Users\AZHARJ~1\AppData\Local\Temp\claude\f--MT5--ML\f161b91a-785e-4060-90fa-30e0efd89fe4\tasks\bx12ozkoe.output',
    '2025': r'C:\Users\AZHARJ~1\AppData\Local\Temp\claude\f--MT5--ML\f161b91a-785e-4060-90fa-30e0efd89fe4\tasks\bx2069g5n.output',
}

BUCKET_W = 200   # price-point width of zone cluster bucket

# Updated regex: date time  side  bias  signals  confs  is_retest  entry  sl  tp  exit  pnl  outcome
PAT = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
    r'(buy|sell)\s+'
    r'(\S+)\s+'
    r'([\w|]+)\s+'
    r'(\d+)\s+'
    r'(True|False)\s+'
    r'([\d.]+)\s+'    # entry
    r'([\d.]+)\s+'    # sl
    r'([\d.]+)\s+'    # tp
    r'([\d.]+)\s+'    # exit
    r'([-\d.]+)\s+'   # pnl
    r'([-\d]+)'       # outcome
)

def zone_key(side, sl):
    """Approximate zone boundary and cluster into 200-pt buckets."""
    if side == 'buy':
        approx_bottom = sl * 1.002
        return ('buy', round(approx_bottom / BUCKET_W) * BUCKET_W)
    else:
        approx_top = sl / 1.002
        return ('sell', round(approx_top / BUCKET_W) * BUCKET_W)

def stats(trades):
    if not trades:
        return dict(n=0, wins=0, wr=0.0, avg_win=0.0, avg_loss=0.0, payoff=0.0, net=0.0)
    wins   = [t for t in trades if t['outcome'] ==  1]
    losses = [t for t in trades if t['outcome'] == -1]
    wr     = len(wins) / len(trades) * 100
    avg_w  = sum(t['pnl'] for t in wins)   / max(len(wins),   1)
    avg_l  = sum(t['pnl'] for t in losses) / max(len(losses), 1)
    payoff = avg_w / abs(avg_l) if avg_l != 0 else 0.0
    net    = sum(t['pnl'] for t in trades)
    return dict(n=len(trades), wins=len(wins), wr=wr,
                avg_win=avg_w, avg_loss=avg_l, payoff=payoff, net=net)

def row(label, s, indent=4):
    pad = ' ' * indent
    if s['n'] == 0:
        return f"{pad}{label:<16}  n=  0  —"
    return (
        f"{pad}{label:<16}  n={s['n']:3d}  W={s['wins']:2d}  "
        f"WR={s['wr']:5.1f}%  "
        f"avgW=${s['avg_win']:8.2f}  avgL=${s['avg_loss']:9.2f}  "
        f"payoff={s['payoff']:.2f}x  net=${s['net']:+8.2f}"
    )

all_buckets = {'first_attempt': [], 'post_win': [], 'post_loss': []}

for year, path in FILES.items():
    text = pathlib.Path(path).read_text(encoding='utf-8', errors='replace')
    trades = []
    for line in text.splitlines():
        m = PAT.match(line.strip())
        if not m:
            continue
        side    = m.group(2)
        sl      = float(m.group(8))
        outcome = int(m.group(12))
        pnl     = float(m.group(11))
        trades.append(dict(side=side, sl=sl, outcome=outcome, pnl=pnl,
                           zk=zone_key(side, sl)))

    # Replay chronologically; tag each trade by most recent prior outcome on same zone
    zone_last: dict = {}   # zk -> most recent outcome (-1, 0, or 1)
    yr_buckets = {'first_attempt': [], 'post_win': [], 'post_loss': []}
    expired = []

    for t in trades:
        zk = t['zk']
        if t['outcome'] == 0:
            # Expired: update zone history but exclude from analysis buckets
            expired.append(t)
            zone_last[zk] = 0
            continue

        prior = zone_last.get(zk)   # None = no prior trade on this zone

        if prior is None:
            bucket = 'first_attempt'
        elif prior == 1:
            bucket = 'post_win'
        elif prior == -1:
            bucket = 'post_loss'
        else:
            # prior was expired (0) — treat as first meaningful attempt
            bucket = 'first_attempt'

        yr_buckets[bucket].append(t)
        zone_last[zk] = t['outcome']

    print(f'=== {year}  ({sum(len(v) for v in yr_buckets.values())} trades, {len(expired)} expired excluded) ===')
    for k in ('first_attempt', 'post_win', 'post_loss'):
        print(row(k, stats(yr_buckets[k])))
    print()

    for k in all_buckets:
        all_buckets[k] += yr_buckets[k]

print('=== 3-YEAR COMBINED ===')
total = sum(len(v) for v in all_buckets.values())
print(f'  ({total} trades across 2023–2025)')
for k in ('first_attempt', 'post_win', 'post_loss'):
    print(row(k, stats(all_buckets[k])))
