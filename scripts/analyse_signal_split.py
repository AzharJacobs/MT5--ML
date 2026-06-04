"""Parses backtest output files and reports BOS-only / CHoCH / both stats."""
import re, pathlib

FILES = {
    '2023': r'C:\Users\AZHARJ~1\AppData\Local\Temp\claude\f--MT5--ML\f161b91a-785e-4060-90fa-30e0efd89fe4\tasks\bw2n3ms3h.output',
    '2024': r'C:\Users\AZHARJ~1\AppData\Local\Temp\claude\f--MT5--ML\f161b91a-785e-4060-90fa-30e0efd89fe4\tasks\b0b7y36wa.output',
    '2025': r'C:\Users\AZHARJ~1\AppData\Local\Temp\claude\f--MT5--ML\f161b91a-785e-4060-90fa-30e0efd89fe4\tasks\bl01z8imi.output',
}

PAT = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
    r'(buy|sell)\s+'
    r'(\S+)\s+'
    r'([\w|]+)\s+'
    r'(\d+)\s+'
    r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
    r'([-\d.]+)\s+([-\d]+)'
)

def stats(trades):
    if not trades:
        return dict(n=0, wins=0, wr=0.0, avg_win=0.0, avg_loss=0.0, payoff=0.0, net=0.0)
    wins   = [t for t in trades if t['outcome'] == 1]
    losses = [t for t in trades if t['outcome'] == -1]
    wr     = len(wins) / len(trades) * 100
    avg_w  = sum(t['pnl'] for t in wins)   / max(len(wins),   1)
    avg_l  = sum(t['pnl'] for t in losses) / max(len(losses), 1)
    payoff = avg_w / abs(avg_l) if avg_l != 0 else 0.0
    net    = sum(t['pnl'] for t in trades)
    return dict(n=len(trades), wins=len(wins), wr=wr,
                avg_win=avg_w, avg_loss=avg_l, payoff=payoff, net=net)

def row(label, s):
    return (
        f"  {label:<12} n={s['n']:3d}  W={s['wins']:2d}  "
        f"WR={s['wr']:5.1f}%  "
        f"avgW=${s['avg_win']:8.2f}  avgL=${s['avg_loss']:9.2f}  "
        f"payoff={s['payoff']:.2f}x  net=${s['net']:+8.2f}"
    )

buckets = {'bos_only': [], 'choch': [], 'both': [], 'other': []}

for year, path in FILES.items():
    text = pathlib.Path(path).read_text(encoding='utf-8', errors='replace')
    yr_trades = {'bos_only': [], 'choch': [], 'both': [], 'other': []}

    for line in text.splitlines():
        m = PAT.match(line.strip())
        if not m:
            continue
        sigs_str = m.group(4)
        sigs     = set(sigs_str.split('|'))
        outcome  = int(m.group(11))
        pnl      = float(m.group(10))
        t        = dict(sigs=sigs_str, outcome=outcome, pnl=pnl)

        has_bos   = 'bos_msb' in sigs
        has_choch = 'choch'   in sigs

        if has_bos and has_choch:
            yr_trades['both'].append(t)
            yr_trades['choch'].append(t)   # choch-tagged includes both-fired
        elif has_bos:
            yr_trades['bos_only'].append(t)
        elif has_choch:
            yr_trades['choch'].append(t)
        else:
            yr_trades['other'].append(t)

    print(f'=== {year} ===')
    for k in ('bos_only', 'choch', 'both', 'other'):
        print(row(k, stats(yr_trades[k])))
        buckets[k] += yr_trades[k]
    print()

print('=== 3-YEAR COMBINED ===')
for k in ('bos_only', 'choch', 'both', 'other'):
    print(row(k, stats(buckets[k])))
