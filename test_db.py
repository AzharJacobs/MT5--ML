"""
test_db.py — Verify all 5 RL/ML tables receive data correctly.

Tests:
  1. DB connection to RL/ML
  2. write_market_context  → market_context
  3. write_ml_signal       → ml_performance (signal row)
  4. update_ml_entry       → ml_performance (entry update)
  5. update_ml_exit        → ml_performance (exit update)
  6. write_rl_decision     → rl_shadow (decision row)
  7. update_rl_outcome     → rl_shadow (outcome update)
  8. write_retrain_log     → retrain_log
  9. Join: market_context ⋈ ml_performance ⋈ rl_shadow on (timestamp, symbol)
 10. Confirm all 4 timeframes appear in market_context
 11. Cleanup test rows

Run:
    python test_db.py
"""

import sys
from datetime import datetime, timezone

from utils.db_writer import (
    get_engine,
    write_market_context,
    write_ml_signal,
    update_ml_entry,
    update_ml_exit,
    write_rl_decision,
    update_rl_outcome,
    write_retrain_log,
    utcnow,
)
from sqlalchemy import text

PASS = "[PASS]"
FAIL = "[FAIL]"

_test_ts  = datetime(2099, 1, 1, 12, 0, 0, tzinfo=timezone.utc)   # sentinel timestamp
_symbol   = "XTEST"    # 5 chars — fits VARCHAR(10) symbol column
_failures = []


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        _failures.append(label)
        print(f"  {FAIL}  {label}  {detail}")


# ── 1. Connection ──────────────────────────────────────────────────────────────
print("\n── 1. DB connection ──")
try:
    engine = get_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
    check("Connected to RL/ML", True)
    print(f"       {version[:60]}")
except Exception as exc:
    check("Connected to RL/ML", False, str(exc))
    sys.exit(1)


# ── 2. market_context ─────────────────────────────────────────────────────────
print("\n── 2. market_context (all 4 TFs) ──")
mc_rows = [
    {
        "timestamp": _test_ts, "symbol": _symbol, "timeframe": tf,
        "open": 2300.0, "high": 2310.0, "low": 2295.0, "close": 2305.0,
        "volume": 1000.0,
        "htf_bias": 1, "market_structure": "bullish", "session": "London",
        "atr_current": 5.0, "atr_average": 4.8, "atr_ratio": 1.04,
        "volume_current": 1000.0, "volume_average": 900.0, "volume_ratio": 1.11,
        "candle_body_ratio": 0.6, "momentum_score": 0.002,
        "mins_to_news": None, "news_impact": None,
    }
    for tf in ["5min", "15min", "1H", "4H"]
]
write_market_context(mc_rows)
with engine.connect() as conn:
    mc_count = conn.execute(
        text("SELECT COUNT(*) FROM market_context WHERE symbol=:s AND timestamp=:t"),
        {"s": _symbol, "t": _test_ts}
    ).scalar()
check("4 market_context rows written", mc_count == 4, f"got {mc_count}")

with engine.connect() as conn:
    tfs = [r[0] for r in conn.execute(
        text("SELECT timeframe FROM market_context WHERE symbol=:s AND timestamp=:t ORDER BY timeframe"),
        {"s": _symbol, "t": _test_ts}
    ).fetchall()]
for tf in ["5min", "15min", "1H", "4H"]:
    check(f"  timeframe {tf} present", tf in tfs)


# ── 3. ml_performance — signal ─────────────────────────────────────────────────
print("\n── 3. ml_performance (signal → entry → exit) ──")
ml_id = write_ml_signal(
    timestamp        = _test_ts,
    symbol           = _symbol,
    signal           = 1,
    confidence       = 0.65,
    triggered_rule   = "test: demand zone buy",
    triggered_tf     = "15min",
    sl_price         = 2290.0,
    tp_price         = 2340.0,
    sl_distance_pips = 150.0,
    tp_distance_pips = 350.0,
    rr_ratio         = 2.33,
)
check("write_ml_signal returned id", ml_id is not None, f"id={ml_id}")

with engine.connect() as conn:
    row = conn.execute(
        text("SELECT signal, confidence, sl_price, tp_price, rr_ratio, entry_price "
             "FROM ml_performance WHERE id=:id"),
        {"id": ml_id}
    ).fetchone()
check("ml_performance signal row correct",
      row is not None and int(row[0]) == 1 and abs(float(row[2]) - 2290.0) < 0.01,
      str(dict(row._mapping) if row else None))

# ── 4. ml_performance — entry update ──────────────────────────────────────────
update_ml_entry(row_id=ml_id, entry_price=2305.0)
with engine.connect() as conn:
    ep = conn.execute(
        text("SELECT entry_price FROM ml_performance WHERE id=:id"), {"id": ml_id}
    ).scalar()
check("update_ml_entry wrote entry_price", ep is not None and abs(float(ep) - 2305.0) < 0.01)

# ── 5. ml_performance — exit update ───────────────────────────────────────────
update_ml_exit(
    row_id           = ml_id,
    exit_price       = 2340.0,
    exit_reason      = "TP",
    actual_pnl_usd   = 35.0,
    actual_pnl_pips  = 350.0,
    outcome          = "win",
    closed_at        = utcnow(),
    trade_duration   = 8,
)
with engine.connect() as conn:
    row2 = conn.execute(
        text("SELECT outcome, actual_pnl_usd, exit_reason FROM ml_performance WHERE id=:id"),
        {"id": ml_id}
    ).fetchone()
check("update_ml_exit wrote outcome=win",
      row2 is not None and row2[0] == "win" and abs(float(row2[1]) - 35.0) < 0.01)


# ── 6. rl_shadow — decision ────────────────────────────────────────────────────
print("\n── 4. rl_shadow (decision → outcome update) ──")
rl_id = write_rl_decision(
    timestamp               = _test_ts,
    symbol                  = _symbol,
    ml_signal               = 1,
    ml_confidence           = 0.65,
    ml_entry_price          = 2305.0,
    ml_sl_price             = 2290.0,
    ml_tp_price             = 2340.0,
    ml_rr_ratio             = 2.33,
    ml_triggered_rule       = "test demand zone",
    ml_triggered_tf         = "15min",
    htf_bias                = 1,
    market_structure        = "bullish",
    session                 = "London",
    momentum_score          = 0.002,
    atr_ratio               = 1.04,
    volume_ratio            = 1.11,
    candle_body_ratio       = 0.6,
    in_trade                = False,
    trade_duration          = 0,
    unrealised_pnl          = 0.0,
    is_stalling             = False,
    better_setup_available  = False,
    better_setup_direction  = None,
    better_setup_quality    = 3.5,
    rl_decision             = 0,
    rl_confidence           = 0.78,
    rl_reason               = "AGREED BUY | A-grade zone | bullish 4H",
    rl_suggested_entry      = 2305.0,
    rl_suggested_sl         = 2290.0,
    rl_suggested_tp         = 2340.0,
    rl_recommended_rotation = False,
)
check("write_rl_decision returned id", rl_id is not None, f"id={rl_id}")

with engine.connect() as conn:
    rrow = conn.execute(
        text("SELECT ml_signal, rl_decision, rl_confidence FROM rl_shadow WHERE id=:id"),
        {"id": rl_id}
    ).fetchone()
check("rl_shadow decision row correct",
      rrow is not None and int(rrow[0]) == 1 and int(rrow[1]) == 0,
      str(dict(rrow._mapping) if rrow else None))

# ── 7. rl_shadow — outcome update ─────────────────────────────────────────────
update_rl_outcome(
    row_id              = rl_id,
    price_1h_later      = 2320.0,
    price_4h_later      = 2338.0,
    price_24h_later     = 2341.0,
    max_favourable      = 35.0,
    max_adverse         = 5.0,
    ml_actual_pnl       = 35.0,
    rl_hypothetical_pnl = 35.0,
    pnl_difference      = 0.0,
    rl_was_correct      = True,
)
with engine.connect() as conn:
    outcol = conn.execute(
        text("SELECT rl_was_correct, rl_hypothetical_pnl FROM rl_shadow WHERE id=:id"),
        {"id": rl_id}
    ).fetchone()
check("update_rl_outcome wrote rl_was_correct=True",
      outcol is not None and outcol[0] is True and abs(float(outcol[1]) - 35.0) < 0.01)


# ── 8. retrain_log ─────────────────────────────────────────────────────────────
print("\n── 5. retrain_log ──")
write_retrain_log(
    model_version   = "_test_v0",
    training_rows   = 99,
    timesteps       = 1000,
    ml_win_rate     = 0.55,
    rl_accuracy     = 0.62,
    pnl_improvement = 0.08,
)
with engine.connect() as conn:
    rlog = conn.execute(
        text("SELECT training_rows, ml_win_rate FROM retrain_log WHERE model_version='_test_v0'")
    ).fetchone()
check("retrain_log row written", rlog is not None and int(rlog[0]) == 99)


# ── 9. 3-way join: market_context ⋈ ml_performance ⋈ rl_shadow ────────────────
print("\n── 6. 3-way join on (timestamp, symbol) ──")
with engine.connect() as conn:
    jrow = conn.execute(text("""
        SELECT mc.timeframe, mp.signal, mp.outcome, rs.rl_decision, rs.rl_was_correct
        FROM market_context mc
        JOIN ml_performance mp ON mp.timestamp = mc.timestamp AND mp.symbol = mc.symbol
        JOIN rl_shadow      rs ON rs.timestamp = mc.timestamp AND rs.symbol = mc.symbol
        WHERE mc.symbol = :s AND mc.timestamp = :t
        LIMIT 1
    """), {"s": _symbol, "t": _test_ts}).fetchone()
check("3-way join returns a row", jrow is not None, str(jrow))
if jrow:
    check("  join: mp.outcome = win",      jrow[2] == "win")
    check("  join: rs.rl_was_correct = True", jrow[4] is True)


# ── 10. Cleanup ────────────────────────────────────────────────────────────────
print("\n── 7. Cleanup test rows ──")
with engine.begin() as conn:
    conn.execute(text("DELETE FROM market_context  WHERE symbol=:s"), {"s": _symbol})
    conn.execute(text("DELETE FROM ml_performance  WHERE symbol=:s"), {"s": _symbol})
    conn.execute(text("DELETE FROM rl_shadow       WHERE symbol=:s"), {"s": _symbol})
    conn.execute(text("DELETE FROM retrain_log     WHERE model_version='_test_v0'"))
check("Test rows cleaned up", True)


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
if _failures:
    print(f"RESULT: {len(_failures)} test(s) FAILED:")
    for f in _failures:
        print(f"  ✗  {f}")
    sys.exit(1)
else:
    n = 14  # total check count above
    print(f"RESULT: All checks PASSED — every table receiving data correctly.")
print("=" * 58)
