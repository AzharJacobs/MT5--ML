"""
drift_detector.py — Detects when model performance is degrading vs baseline.
Compares rolling live win-rate to backtest win-rate; alerts when gap exceeds threshold.
"""

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, baseline_win_rate: float, window: int = 50, threshold: float = 0.15):
        self.baseline = baseline_win_rate
        self.threshold = threshold
        self._recent: deque = deque(maxlen=window)

    def update(self, win: bool) -> Optional[str]:
        self._recent.append(int(win))
        if len(self._recent) < self._recent.maxlen:
            return None
        live_wr = sum(self._recent) / len(self._recent)
        drift = self.baseline - live_wr
        if drift > self.threshold:
            msg = f"DRIFT ALERT: live win-rate={live_wr:.2%} vs baseline={self.baseline:.2%} (gap={drift:.2%})"
            logger.warning(msg)
            return msg
        return None
