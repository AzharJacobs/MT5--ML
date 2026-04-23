"""
performance_tracker.py — Tracks live trade results vs backtest expectations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, log_path: str = "experiments/runs/live_results.json"):
        self.log_path = Path(log_path)
        self._trades: List[Dict] = []
        if self.log_path.exists():
            self._trades = json.loads(self.log_path.read_text())

    def record_trade(self, trade: Dict) -> None:
        self._trades.append(trade)
        self.log_path.write_text(json.dumps(self._trades, indent=2))

    def summary(self) -> Dict:
        if not self._trades:
            return {}
        wins = [t for t in self._trades if t.get("pnl", 0) > 0]
        return {
            "total_trades": len(self._trades),
            "win_rate": round(len(wins) / len(self._trades), 4),
            "total_pnl": round(sum(t.get("pnl", 0) for t in self._trades), 2),
        }
