"""
compare_bias_levers.py — Compare H4 swing-width (Lever 1) and allow_neutral (Lever 2)
against the baseline for the Z&Z strategy on USTECH, 2023–2024.

All other params are pinned to the validated live-bot baseline.

Usage:
    python -X utf8 scripts/compare_bias_levers.py

Prints full per-run detail (bias distribution, trade log, etc.) then a
clean summary comparison table at the end.
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trading.strategies.zz.ustec.engine import run_backtest

START = "2023-01-01"
END   = "2025-01-01"

# All non-lever params pinned to live-bot baseline
COMMON = dict(
    start                  = START,
    end                    = END,
    symbol                 = "ustech",
    cash                   = 10_000.0,
    min_rr                 = 1.5,
    directional_filter     = True,
    min_confirmations      = 1,
    aggressive_entry       = False,
    midline_tp             = False,
    sl_buffer_pct          = 0.002,
    require_leave_and_return = True,
    cooldown_bars          = 15,
    realistic              = False,
    min_sl_pct             = 0.0,
)

CONFIGS = [
    ("C0  baseline     sw=2 neutral=Y", dict(h4_swing_left=2, h4_swing_right=2, allow_neutral=True)),
    ("C1  sw=3         sw=3 neutral=Y", dict(h4_swing_left=3, h4_swing_right=3, allow_neutral=True)),
    ("C2  sw=4         sw=4 neutral=Y", dict(h4_swing_left=4, h4_swing_right=4, allow_neutral=True)),
    ("C3  no_neutral   sw=2 neutral=N", dict(h4_swing_left=2, h4_swing_right=2, allow_neutral=False)),
    ("C4  sw3+no_neut  sw=3 neutral=N", dict(h4_swing_left=3, h4_swing_right=3, allow_neutral=False)),
    ("C5  sw4+no_neut  sw=4 neutral=N", dict(h4_swing_left=4, h4_swing_right=4, allow_neutral=False)),
]


def _bias_dist(df_t) -> str:
    """Compact bias distribution string, e.g. 'bull=8 bear=5 nup=3 ndn=1'"""
    abbr = {
        "bullish":      "bull",
        "bearish":      "bear",
        "neutral_up":   "nup",
        "neutral_down": "ndn",
        "neutral":      "neut",
    }
    parts = []
    for bias, grp in df_t.groupby("h4_bias"):
        wr = (grp["outcome"] == 1).mean() * 100
        parts.append(f"{abbr.get(bias, bias)}={len(grp)}(WR{wr:.0f}%)")
    return "  ".join(parts)


def _run_one(label: str, extra: dict):
    print("\n" + "=" * 90)
    print(f"  RUN: {label}")
    print("=" * 90)
    result = run_backtest(**COMMON, **extra)
    if not result:
        return None, None
    metrics, df_t = result
    return metrics, df_t


def _pct(num, denom):
    return f"{num/denom*100:.0f}%" if denom else "n/a"


def main():
    rows = []

    for label, extra in CONFIGS:
        metrics, df_t = _run_one(label, extra)
        if metrics is None:
            rows.append((label, {}, None))
            continue
        rows.append((label, metrics, df_t))

    # ── Summary comparison table ──────────────────────────────────────────────
    SEP = "─" * 130
    print("\n\n" + "=" * 130)
    print("  SUMMARY COMPARISON TABLE  —  USTECH  2023-01-01 → 2025-01-01")
    print("=" * 130)
    hdr = (
        f"{'Config':<28}  {'Trades':>6}  {'WR':>5}  {'Net PnL':>10}  "
        f"{'Buys':>5}  {'Buy WR':>6}  {'Sells':>5}  {'Sell WR':>7}  "
        f"Bias distribution at entry"
    )
    print(hdr)
    print(SEP)

    for label, metrics, df_t in rows:
        if df_t is None or df_t.empty:
            print(f"  {label:<28}  NO TRADES")
            continue

        total  = int(metrics["total_trades"])
        wr     = metrics["win_rate_%"]
        net    = metrics["net_pnl"]
        buys   = int(metrics["buy_trades"])
        bwins  = int(metrics["buy_wins"])
        sells  = int(metrics["sell_trades"])
        swins  = int(metrics["sell_wins"])
        bwr    = _pct(bwins, buys)
        swr    = _pct(swins, sells)
        bdist  = _bias_dist(df_t)

        print(
            f"  {label:<28}  {total:>6}  {wr:>5}%  {net:>10}  "
            f"{buys:>5}  {bwr:>6}  {sells:>5}  {swr:>7}  {bdist}"
        )

    print(SEP)
    print()

    # ── Per-bias buy/sell breakdown for each config ───────────────────────────
    print("\n  DIRECTION × BIAS BREAKDOWN  (how many buys vs sells per bias state)")
    print(SEP)
    dir_hdr = f"  {'Config':<28}  {'bias':<14}  {'trades':>6}  {'buys':>5}  {'sells':>5}  {'WR':>5}"
    print(dir_hdr)
    print(SEP)

    for label, metrics, df_t in rows:
        if df_t is None or df_t.empty:
            continue
        first = True
        for bias_val in ["bullish", "bearish", "neutral_up", "neutral_down", "neutral"]:
            grp = df_t[df_t["h4_bias"] == bias_val]
            if grp.empty:
                continue
            n_b   = int((grp["side"] == "buy").sum())
            n_s   = int((grp["side"] == "sell").sum())
            wr_g  = f"{(grp['outcome']==1).mean()*100:.0f}%"
            lbl   = label if first else ""
            print(f"  {lbl:<28}  {bias_val:<14}  {len(grp):>6}  {n_b:>5}  {n_s:>5}  {wr_g:>5}")
            first = False
        print(SEP)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
