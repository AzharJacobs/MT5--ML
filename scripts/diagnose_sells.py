"""
diagnose_sells.py — Diagnostic only. No logic changes.

Pools all SELL trades from 2024/2025/2026 backtests, merges each entry
with the full feature row at that bar, then compares winners vs losers
feature-by-feature to find what separates them.
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, r"f:\MT5- ML")
os.chdir(r"f:\MT5- ML")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data.loader import get_connection
from data.feature_engineer import build_features
from backtest.engine import run_backtest


PERIODS = [
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-05-31"),
]


def _to_naive_min(ts_series):
    """Convert a timestamp Series to naive UTC floored to minute."""
    s = pd.to_datetime(ts_series)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.floor("min")


def _build_features_for_period(start, end):
    db = get_connection()
    db.connect()
    dfs = {}
    for tf in ("15min", "1H", "4H"):
        q = ("SELECT * FROM xauusd_ohlcv WHERE timeframe=%s "
             "AND date>=%s AND date<=%s ORDER BY timestamp ASC")
        dfs[tf] = db.fetch_dataframe(q, (tf, start, end))
    db.disconnect()
    for tf in dfs:
        if not dfs[tf].empty:
            dfs[tf]["timestamp"] = pd.to_datetime(dfs[tf]["timestamp"])
    ltf = dfs["15min"]
    h1  = dfs["1H"] if not dfs["1H"].empty else None
    h4  = dfs["4H"] if not dfs["4H"].empty else None
    feat = build_features(ltf, h1_df=h1, h4_df=h4, include_london_ny=False)
    feat["_dt"] = _to_naive_min(feat["timestamp"])
    return feat.set_index("_dt")


# ── 1. Pool sell trades across all periods ──────────────────────────────────

all_sells = []

for label, start, end in PERIODS:
    print(f"\n>>> {label}: running backtest...", flush=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_backtest(
            timeframe="15min", start_date=start, end_date=end,
            cash=50, stake=0.15, use_pct_stake=True,
            confidence=0.52, commission=0.0, no_trail=True,
        )
    sells_log = [t for t in result.trade_log if t.get("side") == "sell"]
    print(f"    {len(sells_log)} sell entries in trade_log", flush=True)
    if not sells_log:
        continue

    print(f">>> {label}: building features for match...", flush=True)
    feat_idx = _build_features_for_period(start, end)

    # Debug: show what the timestamps look like on each side
    sample_ed  = sells_log[0].get("entry_date") if sells_log else None
    sample_idx = feat_idx.index[0] if not feat_idx.empty else None
    print(f"    entry_date sample : {sample_ed!r}  type={type(sample_ed).__name__}")
    print(f"    feat index sample : {sample_idx!r}  type={type(sample_idx).__name__}")

    # Build a normalised lookup: strip tz, floor to minute
    def _norm(ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t.floor("min")

    feat_lookup = {_norm(idx): feat_idx.loc[idx] for idx in feat_idx.index}

    matched = 0
    for t in sells_log:
        ed = t.get("entry_date")
        if ed is None:
            continue
        dt = _norm(ed)
        frow = feat_lookup.get(dt)
        if frow is None:
            continue
        if isinstance(frow, pd.DataFrame):
            frow = frow.iloc[-1]

        ep   = float(frow.get("close",    np.nan))
        atr  = float(frow.get("atr_14",   np.nan))
        ir   = float(t.get("initial_risk", np.nan) or np.nan)
        d_top = float(frow.get("demand_zone_top", np.nan))

        # RR at entry: (entry - demand_zone_top) / SL_distance
        rr_entry = ((ep - d_top) / ir
                    if (not np.isnan(d_top) and ir > 0 and ep > d_top) else np.nan)

        # Is entry price within the 1H HTF supply zone?
        htf_sb = float(frow.get("htf_supply_zone_bottom", np.nan))
        htf_st = float(frow.get("htf_supply_zone_top",    np.nan))
        in_htf = (1.0 if (not np.isnan(htf_sb) and not np.isnan(htf_st)
                          and htf_sb <= ep <= htf_st) else 0.0)

        all_sells.append({
            "period":               label,
            "winner":               1 if t["pnl"] > 0 else 0,
            "close_reason":         t.get("close_reason", "?"),
            "prob":                 float(t.get("prob",        np.nan) or np.nan),
            "zone_quality":         float(t.get("zone_quality", np.nan) or np.nan),
            # confirmation
            "sell_confirm_score":   float(frow.get("sell_confirmation_score", np.nan)),
            "bearish_engulfing":    float(frow.get("bearish_engulfing",       np.nan)),
            "bos_bearish":          float(frow.get("bos_bearish",             np.nan)),
            # supply zone
            "supply_touches":       float(frow.get("supply_zone_touches",     np.nan)),
            "supply_fresh":         float(frow.get("supply_zone_fresh",       np.nan)),
            "in_htf_supply":        in_htf,
            # htf trend
            "htf_4h_bias":          float(frow.get("htf_4h_bias",            np.nan)),
            "htf_1h_bias":          float(frow.get("htf_1h_bias",            np.nan)),
            "rule_htf_aligned_sell":float(frow.get("rule_htf_aligned_sell",  np.nan)),
            # distance / RR
            "demand_dist_atr":      float(frow.get("nearest_demand_dist_atr", np.nan)),
            "rr_entry":             rr_entry,
            # volatility
            "atr_14":               atr,
            # session
            "session_id":           float(frow.get("session_id", np.nan)),
            "hour":                 float(frow.get("hour",       np.nan)),
        })
        matched += 1

    print(f"    Matched {matched}/{len(sells_log)} entries to feature rows", flush=True)


df = pd.DataFrame(all_sells)

if df.empty:
    print("\nERROR: no sell trades matched — check timestamp debug output above.")
    sys.exit(1)

# ATR regime: current ATR vs median across all sell entries
atr_med = max(float(df["atr_14"].median()), 1e-6)
df["atr_ratio"] = df["atr_14"] / atr_med

W = df[df["winner"] == 1].copy()
L = df[df["winner"] == 0].copy()
nw, nl = len(W), len(L)


# ── 2. Helpers ───────────────────────────────────────────────────────────────

def _sep(wv, lv):
    """YES / WEAK / NO based on relative difference of means."""
    wv = wv.dropna(); lv = lv.dropna()
    if len(wv) == 0 or len(lv) == 0:
        return "N/A "
    wm, lm = wv.mean(), lv.mean()
    scale = max(abs(wm), abs(lm), 1e-6)
    r = abs(wm - lm) / scale
    return "YES  <<<" if r > 0.30 else ("WEAK    " if r > 0.12 else "NO      ")

def prow(label, col, fmt=".2f", pct=False):
    wv = W[col].dropna(); lv = L[col].dropna()
    if len(wv) == 0 or len(lv) == 0:
        print(f"  {label:<42}  {'N/A':>9}  {'N/A':>9}  N/A")
        return
    wm, lm = wv.mean(), lv.mean()
    s = _sep(wv, lv)
    ws = f"{wm*100:.1f}%" if pct else f"{wm:{fmt}}"
    ls = f"{lm*100:.1f}%" if pct else f"{lm:{fmt}}"
    print(f"  {label:<42}  {ws:>9}  {ls:>9}  {s}")


# ── 3. Print report ──────────────────────────────────────────────────────────

print(f"\n{'='*76}")
print(f"  SELL TRADE DIAGNOSTIC — {len(df)} pooled sells  ({nw} W / {nl} L)")
print(f"  Periods: "
      f"2024 n={(df['period']=='2024').sum()}  "
      f"2025 n={(df['period']=='2025').sum()}  "
      f"2026 n={(df['period']=='2026').sum()}")
print(f"  Overall sell WR: {nw/max(len(df),1)*100:.1f}%")
print(f"{'='*76}")
print()
print(f"  All entries have choch_bearish=1 at the entry bar (gate requirement).")
print(f"  It cannot separate winners from losers — all values are 1.0.")

print(f"\n  {'Feature':<42}  {'Winners':>9}  {'Losers':>9}  Separates?")
print(f"  {'-'*42}  {'-'*9}  {'-'*9}  {'-'*10}")

prow("sell_confirmation_score (mean)",      "sell_confirm_score",    ".2f")
prow("  bearish_engulfing component",       "bearish_engulfing",     ".2f")
prow("  bos_bearish component",             "bos_bearish",           ".2f")
print()
prow("supply zone touches (mean)",          "supply_touches",        ".1f")
prow("supply zone fresh flag (mean %)",     "supply_fresh",          ".0f", pct=True)
prow("in HTF (1H) supply zone (%)",         "in_htf_supply",         ".0f", pct=True)
print()
prow("htf_4h_bias at entry (mean)",         "htf_4h_bias",           ".3f")
prow("htf_1h_bias at entry (mean)",         "htf_1h_bias",           ".3f")
prow("rule_htf_aligned_sell (%)",           "rule_htf_aligned_sell", ".0f", pct=True)
print()
prow("demand_dist_atr (TP dist in ATR)",    "demand_dist_atr",       ".2f")
prow("RR at entry (approx)",                "rr_entry",              ".2f")
print()
prow("ATR regime (current / median ATR)",   "atr_ratio",             ".3f")
print()
prow("model confidence prob (mean)",        "prob",                  ".3f")
prow("zone_quality at entry (mean)",        "zone_quality",          ".2f")


# ── Session breakdown ────────────────────────────────────────────────────────
SESSION_MAP = {0.0: "Off", 1.0: "London", 2.0: "NY", 3.0: "Overlap"}
df["session"] = df["session_id"].map(SESSION_MAP).fillna("Other")
W2 = df[df["winner"] == 1]; L2 = df[df["winner"] == 0]

print(f"\n  SESSION BREAKDOWN")
print(f"  {'Session':<10}  {'Total':>6}  {'WR':>6}  {'W':>5}  {'L':>5}")
print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
for sess in ["London", "NY", "Overlap", "Off", "Other"]:
    sub = df[df["session"] == sess]
    n = len(sub); w = sub["winner"].sum(); l = n - w
    wr = w / n * 100 if n > 0 else 0
    print(f"  {sess:<10}  {n:>6}  {wr:>5.0f}%  {w:>5}  {l:>5}")


# ── Confidence bucket breakdown ──────────────────────────────────────────────
print(f"\n  CONFIDENCE BUCKET BREAKDOWN")
print(f"  {'Bucket':<12}  {'Total':>6}  {'WR':>6}  {'W':>5}  {'L':>5}")
print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
for lo, hi, label in [
    (0.30, 0.45, "0.30-0.45"),
    (0.45, 0.55, "0.45-0.55"),
    (0.55, 0.65, "0.55-0.65"),
    (0.65, 1.00, "0.65+    "),
]:
    mask = (df["prob"] >= lo) & (df["prob"] < hi)
    sub = df[mask]; n = len(sub); w = int(sub["winner"].sum())
    wr = w / n * 100 if n > 0 else 0
    print(f"  {label:<12}  {n:>6}  {wr:>5.0f}%  {w:>5}  {n-w:>5}")


# ── HTF bias breakdown ───────────────────────────────────────────────────────
print(f"\n  HTF 4H BIAS AT SELL ENTRY")
print(f"  {'Bias':>6}  {'Total':>6}  {'WR':>6}  {'W':>5}  {'L':>5}")
print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
for bias_val, bias_lbl in [(-1.0, "bearish"), (0.0, "neutral"), (1.0, "bullish")]:
    sub = df[df["htf_4h_bias"] == bias_val]
    n = len(sub); w = int(sub["winner"].sum())
    wr = w / n * 100 if n > 0 else 0
    print(f"  {bias_lbl:>8}  {n:>6}  {wr:>5.0f}%  {w:>5}  {n-w:>5}")


# ── Close reason breakdown ───────────────────────────────────────────────────
print(f"\n  CLOSE REASON (sells only)")
print(f"  {'Reason':<12}  {'Total':>6}  {'is_win':>7}")
print(f"  {'-'*12}  {'-'*6}  {'-'*7}")
for reason in ["tp", "sl", "breakeven", "trail", "?"]:
    sub = df[df["close_reason"] == reason]
    n = len(sub); w = int(sub["winner"].sum())
    print(f"  {reason:<12}  {n:>6}  {w:>7}")


print(f"\n{'='*76}")
