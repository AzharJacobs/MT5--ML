from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FXGoldConfig:
    # ─── Zone detection ───────────────────────────────────────────────────────
    fractal_window: int = 5          # bars each side required for a swing high/low pivot

    # ─── Zone grading ─────────────────────────────────────────────────────────
    min_rejections_strong: int = 1   # wick rejections needed to tag zone "strong"
    strong_tfs: Tuple[str, ...] = ("1H", "4H")

    # Score = rejection_count + tf_weight[origin_tf]
    tf_weights: dict = field(default_factory=lambda: {
        "4H":   3.0,
        "1H":   2.0,
        "15min": 1.0,
        "M30":  0.5,
        "M15":  0.3,
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
    bias_swing_count:      int  = 2          # last N swing highs/lows used for structure read
    bias_mode:             str  = "d1_only"  # "d1_only" now reads 4H (tf_high) as the bias TF
    allow_sideways_trades: bool = False      # if True, sideways bias does not block entry

    # ─── Engine / backtest ────────────────────────────────────────────────────
    max_forward_bars: int   = 672    # 15min bars to look forward for SL/TP (672 = 1 week)
    h1_window:        int   = 200    # 15min bars fed to confirmation + bias  (≈50 H1 bars)
    h4_window:        int   = 600    # 1H bars fed to zone detection          (≈150 4H bars)
    d1_window:        int   = 1200   # 4H bars fed to zone detection          (≈200 D1 bars)

    # ─── DB / data ────────────────────────────────────────────────────────────
    db_name:  str = "XAUUSD"
    table:    str = "xauusd_ohlcv"
    tf_high:  str = "4H"     # high TF for zones + bias  (was 1D)
    tf_mid:   str = "1H"     # mid TF for zone scanning  (was 4H)
    tf_entry: str = "15min"  # entry confirmation TF     (was 1H)
