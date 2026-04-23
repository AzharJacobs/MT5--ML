"""
portfolio_manager.py — Tracks and enforces rules across all open positions.
Enforces max_concurrent_positions and daily_loss_limit from config/risk.yaml.
"""

from typing import Dict, List


class PortfolioManager:
    def __init__(self, max_positions: int = 2, daily_loss_limit: float = 0.03):
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self._positions: Dict[str, dict] = {}
        self._daily_pnl: float = 0.0

    def can_open(self) -> bool:
        return (
            len(self._positions) < self.max_positions
            and self._daily_pnl > -self.daily_loss_limit
        )

    def open_position(self, ticket: str, details: dict) -> None:
        self._positions[ticket] = details

    def close_position(self, ticket: str, pnl_pct: float) -> None:
        self._positions.pop(ticket, None)
        self._daily_pnl += pnl_pct

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0

    @property
    def open_positions(self) -> List[str]:
        return list(self._positions.keys())
