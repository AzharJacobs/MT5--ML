"""
broker_interface.py — Abstract broker layer so the execution backend can be swapped.
Implement this interface for MT5, paper trading, or any other broker.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BrokerInterface(ABC):

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> Optional[int]: ...

    @abstractmethod
    def close_order(self, ticket: int) -> bool: ...

    @abstractmethod
    def get_account_info(self) -> dict: ...

    def get_closed_deal_info(self, ticket: int) -> dict:
        """Optional: return exit details for a closed position. MT5 only."""
        return {}

    def get_entry_deal_price(self, ticket: int) -> Optional[float]:
        """Optional: return the actual fill price for the opening deal. MT5 only."""
        return None

    @abstractmethod
    def disconnect(self) -> None: ...
