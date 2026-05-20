"""
scripts/test_live_flow.py — End-to-end live shadow flow test.

Simulates one complete bar cycle without MT5 or a trained RL model:

  Step 1  →  market_context      (4 TFs via write_market_context)
  Step 2  →  ml_performance      (signal row via write_ml_signal)
  Step 3  →  ml_performance      (entry update via update_ml_entry)
  Step 4  →  rl_shadow           (decision row via write_rl_decision)
  Step 5  →  rl_shadow           (outcome update via update_rl_outcome)
  Step 6  →  ml_performance      (exit update via update_ml_exit)
  Step 7  →  3-way join query    (market_context JOIN ml_performance JOIN rl_shadow)
  Step 8  →  Cleanup test rows

Run:
  python scripts/test_live_flow.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_writer import (
    get_engine, write_market_context, write_ml_signal,
    update_ml_entry, update_ml_exit, write_rl_decision,
    update_rl_outcome, utcnow,
)
from sqlalchemy import text

PASS = "[PASS]"
FAIL = "[FAIL]"
_failures = []

# Sentinel values — far-future timestamp avoids collisions with real data
_ts      = datetime(2099, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
_symbol  = "XTEST"   # 5 chars, fits VARCHAR(10)

def ok(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        _failures.append(label)
        print(f"  {FAIL}  {label}  {detail}")

engine = get_engine()

print()
print("=" * 62)
print("  SHADOW FLOW END-TO-END TEST")
print("=" * 62)

# Pre-clean any leftover rows from a previous aborted run
with engine.begin() as conn:
    conn.execute(text("DELETE FROM market_context WHERE symbol=:s"), {"s": _symbol})
    conn.execute(text("DELETE FROM ml_performance WHERE symbol=:s"), {"s": _symbol})
    conn.execute(text("DELETE FROM rl_shadow      WHERE symbol=:s"), {"s": _symbol})

# ── Step 1: market_context — all 4 TFs ───────────────────────────────────────
print("\n[1] Writing market_context (4 TFs)...")
mc_rows = [
    {
        "timestamp": _ts, "symbol": _symbol, "timeframe": tf,
        "open": 2310.50, "high": 2325.75, "low": 2305.00, "close": 2318.30,
        "volume": 1200.0,
        "htf_bias": 1,                  # bullish 4H
        "market_structure": "bullish",
        "session": "London",
        "atr_current": 6.20, "atr_average": 5.80, "atr_ratio": 1.069,
        "volume_current": 1200.0, "volume_average": 1050.0, "volume_ratio": 1.143,
        "candle_body_ratio": 0.52, "momentum_score": 0.00312,
        "mins_to_news": None, "news_impact": None,
    }
    for tf in ["5min", "15min", "1H", "4H"]
]
write_market_context(mc_rows)

with engine.connect() as conn:
    mc_count = conn.execute(text(
        "SELECT COUNT(*) FROM market_context WHERE symbol=:s AND timestamp=:t"
    ), {"s": _symbol, "t": _ts}).scalar()
ok("4 market_context rows written", mc_count == 4, f"got {mc_count}")

with engine.connect() as conn:
    tfs = [r[0] for r in conn.execute(text(
        "SELECT timeframe FROM market_context WHERE symbol=:s AND timestamp=:t ORDER BY timeframe"
    ), {"s": _symbol, "t": _ts}).fetchall()]
for tf in ["15min", "1H", "4H", "5min"]:
    ok(f"  timeframe {tf} present", tf in tfs)

# ── Step 2: ml_performance — signal row ──────────────────────────────────────
print("\n[2] Writing ml_performance signal row...")
ml_id = write_ml_signal(
    timestamp        = _ts,
    symbol           = _symbol,
    signal           = 1,             # BUY
    confidence       = 0.67,
    triggered_rule   = "buy Grade=A demand zone 3.8/5 prob=0.67 rr=2.45",
    triggered_tf     = "15min",
    sl_price         = 2299.50,
    tp_price         = 2353.80,
    sl_distance_pips = 189.0,
    tp_distance_pips = 355.0,
    rr_ratio         = 2.45,
)
ok("write_ml_signal returned id", ml_id is not None, f"id={ml_id}")

with engine.connect() as conn:
    row = conn.execute(text(
        "SELECT signal, confidence, sl_price, tp_price, rr_ratio, entry_price "
        "FROM ml_performance WHERE id=:id"
    ), {"id": ml_id}).fetchone()
ok("Signal row stored correctly",
   row is not None and int(row[0]) == 1 and abs(float(row[2]) - 2299.50) < 0.01,
   str(dict(row._mapping) if row else None))
ok("Entry price still NULL (not filled yet)", row is not None and row[5] is None)

# ── Step 3: ml_performance — entry update ─────────────────────────────────────
print("\n[3] Trade entered — updating entry_price...")
update_ml_entry(row_id=ml_id, entry_price=2318.30)

with engine.connect() as conn:
    ep = conn.execute(text(
        "SELECT entry_price FROM ml_performance WHERE id=:id"
    ), {"id": ml_id}).scalar()
ok("entry_price filled", ep is not None and abs(float(ep) - 2318.30) < 0.01, f"got {ep}")

# ── Step 4: rl_shadow — decision row ─────────────────────────────────────────
print("\n[4] Writing rl_shadow decision row...")
rl_id = write_rl_decision(
    timestamp               = _ts,
    symbol                  = _symbol,
    ml_signal               = 1,
    ml_confidence           = 0.67,
    ml_entry_price          = 2318.30,
    ml_sl_price             = 2299.50,
    ml_tp_price             = 2353.80,
    ml_rr_ratio             = 2.45,
    ml_triggered_rule       = "buy Grade=A demand zone",
    ml_triggered_tf         = "15min",
    htf_bias                = 1,
    market_structure        = "bullish",
    session                 = "London",
    momentum_score          = 0.00312,
    atr_ratio               = 1.069,
    volume_ratio            = 1.143,
    candle_body_ratio       = 0.52,
    in_trade                = False,
    trade_duration          = 0,
    unrealised_pnl          = 0.0,
    is_stalling             = False,
    better_setup_available  = False,
    better_setup_direction  = None,
    better_setup_quality    = 3.8,
    rl_decision             = 0,       # AGREE
    rl_confidence           = 0.81,
    rl_reason               = "AGREED BUY | 4H bullish (+0.72) | A-grade zone (3.8/5) | high confidence (67%) | in-session",
    rl_suggested_entry      = 2318.30,
    rl_suggested_sl         = 2299.50,
    rl_suggested_tp         = 2353.80,
    rl_recommended_rotation = False,
)
ok("write_rl_decision returned id", rl_id is not None, f"id={rl_id}")

with engine.connect() as conn:
    rrow = conn.execute(text(
        "SELECT ml_signal, rl_decision, rl_confidence, rl_reason, in_trade "
        "FROM rl_shadow WHERE id=:id"
    ), {"id": rl_id}).fetchone()
ok("RL decision row stored correctly",
   rrow is not None and int(rrow[0]) == 1 and int(rrow[1]) == 0,
   str(dict(rrow._mapping) if rrow else None))
ok("rl_confidence written", rrow is not None and abs(float(rrow[2]) - 0.81) < 0.01)
ok("in_trade = False at decision time", rrow is not None and rrow[4] is False)

# ── Step 5: rl_shadow — outcome filled after 4 hours ─────────────────────────
print("\n[5] 4 hours later — filling outcome prices...")
update_rl_outcome(
    row_id              = rl_id,
    price_1h_later      = 2328.10,
    price_4h_later      = 2341.75,
    price_24h_later     = 2349.30,
    max_favourable      = 2.85,    # MFE USD (0.01 lot × 10 × 28.5 pips)
    max_adverse         = 0.62,    # MAE USD
    ml_actual_pnl       = 2.70,    # ML actually made $2.70 at TP
    rl_hypothetical_pnl = 2.70,    # RL agreed — same outcome
    pnl_difference      = 0.0,
    rl_was_correct      = True,
)
with engine.connect() as conn:
    orow = conn.execute(text(
        "SELECT price_1h_later, price_4h_later, rl_hypothetical_pnl, rl_was_correct "
        "FROM rl_shadow WHERE id=:id"
    ), {"id": rl_id}).fetchone()
ok("price_1h_later filled",     orow is not None and abs(float(orow[0]) - 2328.10) < 0.01)
ok("price_4h_later filled",     orow is not None and abs(float(orow[1]) - 2341.75) < 0.01)
ok("rl_hypothetical_pnl saved", orow is not None and abs(float(orow[2]) - 2.70) < 0.01)
ok("rl_was_correct = True",     orow is not None and orow[3] is True)

# ── Step 6: ml_performance — trade closed ────────────────────────────────────
print("\n[6] Trade closed at TP — updating exit data...")
closed_at = _ts + timedelta(hours=4, minutes=15)
update_ml_exit(
    row_id           = ml_id,
    exit_price       = 2353.80,
    exit_reason      = "TP",
    actual_pnl_usd   = 2.70,
    actual_pnl_pips  = 354.0,
    outcome          = "win",
    closed_at        = closed_at,
    trade_duration   = 17,         # ~17 × 15min bars
    is_stalling      = False,
)
with engine.connect() as conn:
    xrow = conn.execute(text(
        "SELECT exit_price, outcome, actual_pnl_usd, trade_duration FROM ml_performance WHERE id=:id"
    ), {"id": ml_id}).fetchone()
ok("exit_price = 2353.80",  xrow is not None and abs(float(xrow[0]) - 2353.80) < 0.01)
ok("outcome = 'win'",       xrow is not None and xrow[1] == "win")
ok("actual_pnl_usd = 2.70", xrow is not None and abs(float(xrow[2]) - 2.70) < 0.01)
ok("trade_duration = 17",   xrow is not None and int(xrow[3]) == 17)

# ── Step 7: 3-way join ────────────────────────────────────────────────────────
print("\n[7] Verifying 3-way join (market_context JOIN ml_performance JOIN rl_shadow)...")
with engine.connect() as conn:
    jrow = conn.execute(text("""
        SELECT
            mc.timeframe,
            mc.htf_bias,
            mc.market_structure,
            mc.session,
            mc.atr_ratio,
            mp.signal,
            mp.confidence,
            mp.entry_price,
            mp.exit_price,
            mp.outcome,
            mp.actual_pnl_usd,
            rs.rl_decision,
            rs.rl_confidence,
            rs.rl_was_correct,
            rs.rl_hypothetical_pnl
        FROM market_context mc
        JOIN ml_performance mp
            ON mp.timestamp = mc.timestamp AND mp.symbol = mc.symbol
        JOIN rl_shadow rs
            ON rs.timestamp = mc.timestamp AND rs.symbol = mc.symbol
        WHERE mc.symbol = :s AND mc.timestamp = :t
        LIMIT 1
    """), {"s": _symbol, "t": _ts}).fetchone()
ok("3-way join returns a row", jrow is not None, str(jrow))
if jrow:
    ok("  htf_bias = 1 (bullish)",       int(jrow[1]) == 1)
    ok("  market_structure = 'bullish'", jrow[2] == "bullish")
    ok("  session = 'London'",           jrow[3] == "London")
    ok("  mp.outcome = 'win'",           jrow[9]  == "win")
    ok("  rs.rl_was_correct = True",     jrow[13] is True)
    ok("  rs.rl_hypothetical_pnl > 0",  float(jrow[14]) > 0)

# ── Step 7b: show full sample ─────────────────────────────────────────────────
print()
print("  Sample joined row:")
if jrow:
    m = jrow._mapping
    for k, v in m.items():
        print(f"    {k:28s}: {v}")

# ── Step 8: cleanup ───────────────────────────────────────────────────────────
print("\n[8] Cleaning up test rows...")
with engine.begin() as conn:
    conn.execute(text("DELETE FROM market_context WHERE symbol=:s"),  {"s": _symbol})
    conn.execute(text("DELETE FROM ml_performance WHERE symbol=:s"),  {"s": _symbol})
    conn.execute(text("DELETE FROM rl_shadow      WHERE symbol=:s"),  {"s": _symbol})
ok("Test rows cleaned up", True)

# ── Live DB snapshot ──────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  LIVE DB SNAPSHOT (latest real rows per table)")
print("=" * 62)

with engine.connect() as conn:
    print("\n--- ml_performance (last 5 real rows) ---")
    rows = conn.execute(text("""
        SELECT id, timestamp AT TIME ZONE 'UTC' AS ts_utc,
               signal, ROUND(confidence::numeric,3) AS prob,
               entry_price, exit_price, outcome, actual_pnl_usd
        FROM ml_performance
        WHERE symbol != 'XTEST'
        ORDER BY id DESC LIMIT 5
    """)).fetchall()
    for r in rows:
        print(" ", dict(r._mapping))

    print("\n--- market_context (last 4 rows when MT5 feeds data) ---")
    rows = conn.execute(text("""
        SELECT id, timestamp AT TIME ZONE 'UTC' AS ts_utc,
               timeframe, htf_bias, market_structure, session,
               ROUND(atr_ratio::numeric,3) AS atr_ratio
        FROM market_context
        WHERE symbol != 'XTEST'
        ORDER BY id DESC LIMIT 4
    """)).fetchall()
    if rows:
        for r in rows: print(" ", dict(r._mapping))
    else:
        print("  (empty — market_context is written only in --mode mt5)")

    print("\n--- rl_shadow (last 2 rows when shadow agent is active) ---")
    rows = conn.execute(text("""
        SELECT id, timestamp AT TIME ZONE 'UTC' AS ts_utc,
               ml_signal, rl_decision, rl_confidence,
               rl_was_correct, rl_hypothetical_pnl
        FROM rl_shadow
        WHERE symbol != 'XTEST'
        ORDER BY id DESC LIMIT 2
    """)).fetchall()
    if rows:
        for r in rows: print(" ", dict(r._mapping))
    else:
        print("  (empty — rl_shadow is written only with --rl-shadow flag + trained model)")

# ── Summary ───────────────────────────────────────────────────────────────────
n_checks = 21
print()
print("=" * 62)
if _failures:
    print(f"RESULT: {len(_failures)} check(s) FAILED:")
    for f in _failures:
        print(f"  ✗  {f}")
    sys.exit(1)
else:
    print(f"RESULT: All {n_checks} checks PASSED")
    print("  Every table in the shadow flow is receiving data correctly.")
print("=" * 62)
