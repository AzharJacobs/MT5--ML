"""
Candlestick reversal confirmation patterns for FXGold.

All pattern functions operate on bar rows with keys: open, high, low, close.
They are checked on the entry TF (H1 by default) with bars that are INSIDE
or touching the zone.

Supported patterns:
  - Bullish / Bearish Engulfing
  - Bullish / Bearish Pin Bar  (hammer / shooting star)
  - Morning Star               (3-bar bullish reversal)
  - Evening Star               (3-bar bearish reversal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from trading.strategies.FXGold.config import FXGoldConfig


@dataclass
class ConfirmationResult:
    valid: bool
    pattern: Optional[str] = None
    direction: Optional[str] = None


# ─── Internal ─────────────────────────────────────────────────────────────────

def _range(bar) -> float:
    return float(bar["high"]) - float(bar["low"])


# ─── Engulfing ────────────────────────────────────────────────────────────────

def is_bullish_engulfing(prev, curr, cfg: FXGoldConfig) -> bool:
    """
    Current bull candle's body fully engulfs the previous bear candle's body.
    curr.open <= prev.close  AND  curr.close >= prev.open.
    Body must be at least engulf_body_ratio of the candle range.
    """
    if not (float(prev["close"]) < float(prev["open"])):   # prev must be bear
        return False
    if not (float(curr["close"]) > float(curr["open"])):   # curr must be bull
        return False
    body = abs(float(curr["close"]) - float(curr["open"]))
    r = _range(curr)
    if r == 0 or body / r < cfg.engulf_body_ratio:
        return False
    return (float(curr["open"]) <= float(prev["close"])
            and float(curr["close"]) >= float(prev["open"]))


def is_bearish_engulfing(prev, curr, cfg: FXGoldConfig) -> bool:
    """
    Current bear candle's body fully engulfs the previous bull candle's body.
    curr.open >= prev.close  AND  curr.close <= prev.open.
    """
    if not (float(prev["close"]) > float(prev["open"])):   # prev must be bull
        return False
    if not (float(curr["close"]) < float(curr["open"])):   # curr must be bear
        return False
    body = abs(float(curr["close"]) - float(curr["open"]))
    r = _range(curr)
    if r == 0 or body / r < cfg.engulf_body_ratio:
        return False
    return (float(curr["open"]) >= float(prev["close"])
            and float(curr["close"]) <= float(prev["open"]))


# ─── Pin bar ──────────────────────────────────────────────────────────────────

def is_bullish_pin_bar(bar, cfg: FXGoldConfig) -> bool:
    """
    Hammer: long lower wick, small body near the top of the range.
    lower_wick / total_range >= pin_bar_wick_ratio.
    """
    r = _range(bar)
    if r == 0:
        return False
    body_bot = min(float(bar["open"]), float(bar["close"]))
    lower_wick = body_bot - float(bar["low"])
    return lower_wick / r >= cfg.pin_bar_wick_ratio


def is_bearish_pin_bar(bar, cfg: FXGoldConfig) -> bool:
    """
    Shooting star: long upper wick, small body near the bottom of the range.
    upper_wick / total_range >= pin_bar_wick_ratio.
    """
    r = _range(bar)
    if r == 0:
        return False
    body_top = max(float(bar["open"]), float(bar["close"]))
    upper_wick = float(bar["high"]) - body_top
    return upper_wick / r >= cfg.pin_bar_wick_ratio


# ─── Morning / Evening star ───────────────────────────────────────────────────

def is_morning_star(bars: pd.DataFrame, cfg: FXGoldConfig) -> bool:
    """
    3-bar bullish reversal at a support zone:
      bar[-3]: bearish candle
      bar[-2]: small-body candle (doji-ish, body/range <= doji_body_ratio)
      bar[-1]: bullish candle closing above the midpoint of bar[-3]
    """
    if len(bars) < 3:
        return False
    b1, b2, b3 = bars.iloc[-3], bars.iloc[-2], bars.iloc[-1]

    b1_bear = float(b1["close"]) < float(b1["open"])
    r2 = _range(b2)
    b2_small = r2 == 0 or abs(float(b2["close"]) - float(b2["open"])) / r2 <= cfg.doji_body_ratio
    b3_bull = float(b3["close"]) > float(b3["open"])
    midpoint = (float(b1["open"]) + float(b1["close"])) / 2

    return b1_bear and b2_small and b3_bull and float(b3["close"]) > midpoint


def is_evening_star(bars: pd.DataFrame, cfg: FXGoldConfig) -> bool:
    """
    3-bar bearish reversal at a resistance zone:
      bar[-3]: bullish candle
      bar[-2]: small-body candle (doji-ish)
      bar[-1]: bearish candle closing below the midpoint of bar[-3]
    """
    if len(bars) < 3:
        return False
    b1, b2, b3 = bars.iloc[-3], bars.iloc[-2], bars.iloc[-1]

    b1_bull = float(b1["close"]) > float(b1["open"])
    r2 = _range(b2)
    b2_small = r2 == 0 or abs(float(b2["close"]) - float(b2["open"])) / r2 <= cfg.doji_body_ratio
    b3_bear = float(b3["close"]) < float(b3["open"])
    midpoint = (float(b1["open"]) + float(b1["close"])) / 2

    return b1_bull and b2_small and b3_bear and float(b3["close"]) < midpoint


# ─── Unified check ────────────────────────────────────────────────────────────

def check_confirmation(
    df_entry_tf: pd.DataFrame,
    direction: str,
    cfg: FXGoldConfig,
) -> ConfirmationResult:
    """
    Scan the last 3 bars of the entry-TF DataFrame for a reversal pattern
    matching the expected trade direction.

    Pattern priority (first match wins): engulfing → pin bar → star.

    Args:
        df_entry_tf: OHLCV bars at the entry timeframe, sorted ascending.
                     Last bar = most recently closed candle.
        direction:   "buy" (zone is support) or "sell" (zone is resistance).
        cfg:         Strategy configuration.

    Returns:
        ConfirmationResult — valid=True and pattern name if a pattern fires.
    """
    if len(df_entry_tf) < 2:
        return ConfirmationResult(valid=False)

    last = df_entry_tf.iloc[-1]
    prev = df_entry_tf.iloc[-2]

    if direction == "buy":
        if is_bullish_engulfing(prev, last, cfg):
            return ConfirmationResult(valid=True, pattern="engulfing", direction="buy")
        if is_bullish_pin_bar(last, cfg):
            return ConfirmationResult(valid=True, pattern="pin_bar", direction="buy")
        if len(df_entry_tf) >= 3 and is_morning_star(df_entry_tf, cfg):
            return ConfirmationResult(valid=True, pattern="morning_star", direction="buy")
    else:
        if is_bearish_engulfing(prev, last, cfg):
            return ConfirmationResult(valid=True, pattern="engulfing", direction="sell")
        if is_bearish_pin_bar(last, cfg):
            return ConfirmationResult(valid=True, pattern="pin_bar", direction="sell")
        if len(df_entry_tf) >= 3 and is_evening_star(df_entry_tf, cfg):
            return ConfirmationResult(valid=True, pattern="evening_star", direction="sell")

    return ConfirmationResult(valid=False)
