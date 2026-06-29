from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FXGoldConfig:
    # ─── Zone detection ───────────────────────────────────────────────────────
    fractal_window: int = 5          # bars each side required for a swing high/low pivot

    # ─── Zone grading ─────────────────────────────────────────────────────────
    min_rejections_strong: int = 1   # wick rejections needed to tag zone "strong"
    strong_tfs: Tuple[str, ...] = ("H1", "4H", "1D")

    # Score = rejection_count + tf_weight[origin_tf]
    tf_weights: dict = field(default_factory=lambda: {
        "1D":  3.0,
        "4H":  2.0,
        "H1":  1.0,
        "M30": 0.5,
        "M15": 0.3,
    })

    # ─── Touch-count entry rules ───────────────────────────────────────────────
    # live_touch_count tracks retests AFTER the zone is classified as strong.
    # live == 1  →  guide's "2nd touch"  →  TAKE
    # live == 2  →  guide's "3rd touch"  →  SKIP
    # live >= 3  →  guide's "4th touch"  →  SKIP (breakout expected)
    max_live_touches_entry: int = 1

    # ─── Candlestick pattern thresholds ───────────────────────────────────────
    pin_bar_wick_ratio: float = 0.60  # rejection wick / total range
    engulf_body_ratio:  float = 0.50  # body / range for engulfing candle
    doji_body_ratio:    float = 0.25  # body / range <= this = middle star candle
    star_close_ratio:   float = 0.50  # 3rd bar must close past this fraction of bar 1

    # ─── Risk ─────────────────────────────────────────────────────────────────
    sl_buffer_pct: float = 0.001      # extra gap beyond wick tip for SL (0.1%)
    min_rr:        float = 1.5        # minimum R:R to take any trade
    fallback_rr:   float = 2.0        # R:R used when no opposing zone is found

    # ─── Directional bias filter ──────────────────────────────────────────────
    bias_swing_count:      int  = 2         # last N swing highs/lows used for structure read
    bias_mode:             str  = "d1_only"  # "strict" (D1+H4+H1 all agree), "loose" (D1+H4), "d1_only" (D1 alone)
    allow_sideways_trades: bool = False     # if True, sideways bias does not block entry

    # ─── Engine / backtest ────────────────────────────────────────────────────
    max_forward_bars: int   = 168     # H1 bars to look forward for SL/TP (168 = 1 week)
    h1_window:        int   = 50      # H1 bars fed to confirmation + bias
    h4_window:        int   = 150     # H4 bars fed to zone detection
    d1_window:        int   = 200     # D1 bars fed to zone detection

    # ─── DB / data ────────────────────────────────────────────────────────────
    db_name:  str = "XAUUSD"
    table:    str = "xauusd_ohlcv"
    tf_high:  str = "1D"    # higher TF for zone scanning  (DB value)
    tf_mid:   str = "4H"    # mid TF for zone scanning     (DB value)
    tf_entry: str = "1H"    # entry confirmation TF        (DB value)
