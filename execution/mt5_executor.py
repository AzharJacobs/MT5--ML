"""
mt5_executor.py — Sends buy/sell orders through MT5 to Exness.
Implements BrokerInterface using the MetaTrader5 Python package.
"""

import logging
from typing import Optional
from execution.broker_interface import BrokerInterface
from execution.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)


class MT5Executor(BrokerInterface):
    def __init__(self, connector: MT5Connector):
        self._conn = connector

    def connect(self) -> bool:
        return self._conn.connect()

    def disconnect(self) -> None:
        self._conn.disconnect()

    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> Optional[int]:
        import MetaTrader5 as mt5
        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if direction == "buy" else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed: %s", result.comment)
            return None
        logger.info("Order placed: ticket=%s %s %s vol=%.2f", result.order, direction, symbol, volume)
        return result.order

    def close_order(self, ticket: int) -> bool:
        # MT5 close via reverse order — implementation depends on position tracking
        raise NotImplementedError("close_order not yet implemented")

    def get_account_info(self) -> dict:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        return info._asdict() if info else {}
