from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FXGoldConfig:
    # ─── Zone detection ───────────────────────────────────────────────────────
    fractal_window: int = 5          # bars each side required for a swing high/low pivot

    # ─── Zone grading ─────────────────────────────────────────────────────────
    min_rejections_strong: int = 1   # wick rejections needed to tag zone "strong"
    strong_tfs: Tuple[str, ...] = ("4H", "1D")

    # Score = rejection_count + tf_weight[origin_tf]
    tf_weights: dict = field(default_factory=lambda: {
        "1D":   4.0,
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
    bias_mode:             str  = "loose"    # PDF default: D1 (tf_high) sets direction; H4 (tf_mid)
                                              # only BLOCKS when it actively opposes D1 — H4 sideways
                                              # no longer blocks, it just doesn't confirm (see bias.py
                                              # get_aligned_bias docstring for the data behind this).
                                              # "strict" adds H1 (tf_entry) requiring exact 3-way match;
                                              # "d1_only" reads only D1 (tf_high) — kept for experiments.
    allow_sideways_trades: bool = False      # if True, sideways bias does not block entry

    # D1's own bias read — independent of bias_mode above (see bias.py d1_bias).
    # "ema" (default) softens D1 to a price-vs-EMA read: swing-structure read
    # "sideways" on D1 50.3% of the time in a 1440-touch live sample, starving
    # every bias_mode of a directional D1 read. "swing" is the old behaviour.
    d1_bias_method:     str = "ema"
    d1_bias_ema_period: int = 50             # EMA span, in D1 (tf_high) bars

    # ─── SRR / RSS "secret pattern" (SOP 3) ────────────────────────────────────
    secret_pattern_enabled:        bool = True
    secret_pattern_fractal_window: int  = 3    # pivot window used for the sweep scan (smaller/tighter than fractal_window)
    secret_pattern_min_breakouts:  int  = 2    # consecutive prior swing levels that must be swept before the reversal

    # ─── SOP 4: pullback entry ────────────────────────────────────────────────
    entry_mode: str = "pullback"        # "pullback" = wait for retrace to zone edge; "market" = old next-bar-open
    pullback_max_wait_bars: int = 12    # entry-TF bars to wait for the pullback before cancelling

    # ─── Engine / backtest ────────────────────────────────────────────────────
    # Window sizes carry the SAME calendar span as the pre-PDF low-TF config
    # (see FXGoldConfigLowTF below) — only the bar size changed, so bar counts
    # were rescaled to match:
    max_forward_bars: int   = 168    # 1H bars to look forward for SL/TP
                                      # 168 × 1H = 168h = 7 days  ≈  old 672 × 15min = 10080min = 168h
    h1_window:        int   = 50     # entry-tf (1H) bars fed to confirmation + bias
                                      # 50 × 1H = 50h            ≈  old 200 × 15min = 3000min  = 50h
    h4_window:        int   = 150    # tf_mid (4H) bars fed to zone detection
                                      # 150 × 4H = 600h = 25 days ≈  old 600 × 1H   = 600h      = 25 days
    d1_window:        int   = 200    # tf_high (1D) bars fed to zone detection
                                      # 200 × 1D = 200 days      ≈  old 1200 × 4H  = 4800h      = 200 days

    # ─── DB / data ────────────────────────────────────────────────────────────
    db_name:  str = "XAUUSD"
    table:    str = "xauusd_ohlcv"
    tf_high:  str = "1D"     # high TF for zones + bias  (was "4H")
    tf_mid:   str = "4H"     # mid TF for zone scanning  (was "1H")
    tf_entry: str = "1H"     # entry confirmation TF     (was "15min")


@dataclass
class FXGoldConfigLowTF(FXGoldConfig):
    """
    Pre-PDF-alignment preset — zones on 1H/4H, bias reading 4H (tf_high), entries
    on 15min. Kept only so backtests can A/B the PDF-spec default against the
    prior low-TF behaviour (see engine.py regression backtest).
    """
    strong_tfs: Tuple[str, ...] = ("1H", "4H")
    tf_weights: dict = field(default_factory=lambda: {
        "4H":   3.0,
        "1H":   2.0,
        "15min": 1.0,
        "M30":  0.5,
        "M15":  0.3,
    })

    max_forward_bars: int = 672
    h1_window:        int = 200
    h4_window:        int = 600
    d1_window:        int = 1200

    tf_high:  str = "4H"
    tf_mid:   str = "1H"
    tf_entry: str = "15min"

    # SOP 3 didn't exist in the pre-PDF code path — disabled so this preset
    # reproduces that exact old behaviour for the Task 6 regression compare.
    secret_pattern_enabled: bool = False

    # "d1_only" was the base config's default before Task 3 changed it to
    # "loose" — pinned here so this preset still reads only tf_high (4H, in
    # this preset) as the bias TF, matching the true pre-Task-3 behaviour.
    bias_mode: str = "d1_only"

    # EMA-softened D1 bias didn't exist in the pre-PDF code path — pinned to
    # the old swing-structure read so this preset reproduces that behaviour.
    d1_bias_method: str = "swing"
