"""
paper_trader.py — Simulates live trading without sending real orders to MT5.
Use this to validate the full live.py pipeline before going live.
"""

import logging
from typing import Optional
from execution.broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class PaperTrader(BrokerInterface):
    def __init__(self, starting_equity: float = 10_000.0):
        self._equity = starting_equity
        self._positions: dict = {}
        self._ticket_counter = 1000

    def connect(self) -> bool:
        logger.info("PaperTrader connected (equity=%.2f)", self._equity)
        return True

    def disconnect(self) -> None:
        logger.info("PaperTrader disconnected")

    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> Optional[int]:
        ticket = self._ticket_counter
        self._ticket_counter += 1
        self._positions[ticket] = {"symbol": symbol, "direction": direction, "volume": volume, "sl": sl, "tp": tp}
        logger.info("[PAPER] %s %s %.2f lots | SL=%.5f TP=%.5f ticket=%d", direction, symbol, volume, sl, tp, ticket)
        return ticket

    def close_order(self, ticket: int) -> bool:
        if ticket in self._positions:
            pos = self._positions.pop(ticket)
            logger.info("[PAPER] closed ticket=%d %s", ticket, pos)
            return True
        return False

    def get_account_info(self) -> dict:
        return {"equity": self._equity, "open_positions": len(self._positions)}
