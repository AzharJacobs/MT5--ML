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
        import MetaTrader5 as mt5
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning("close_order: ticket=%d not found in open positions", ticket)
            return False
        pos = position[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      pos.symbol,
            "volume":      pos.volume,
            "type":        order_type,
            "position":    ticket,
            "price":       price,
            "comment":     "live_trader_close",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("close_order failed: ticket=%d %s", ticket, result.comment)
            return False
        logger.info("Position closed: ticket=%d", ticket)
        return True

    def get_account_info(self) -> dict:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        return info._asdict() if info else {}

    def get_closed_deal_info(self, ticket: int) -> dict:
        """
        Fetch exit details for a position that has been closed (SL/TP/manual).
        Returns dict with exit_price, pnl, close_reason, close_time — or {} on failure.
        """
        import MetaTrader5 as mt5
        from datetime import datetime
        deals = mt5.history_deals_get(position=ticket)
        if not deals:
            return {}
        reason_map = {
            mt5.DEAL_REASON_SL:     "SL",
            mt5.DEAL_REASON_TP:     "TP",
            mt5.DEAL_REASON_CLIENT: "manual",
            mt5.DEAL_REASON_MOBILE: "manual",
            mt5.DEAL_REASON_WEB:    "manual",
            mt5.DEAL_REASON_EXPERT: "manual",
        }
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT:
                return {
                    "exit_price":   deal.price,
                    "pnl":          deal.profit,
                    "close_reason": reason_map.get(deal.reason, "unknown"),
                    "close_time":   datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                }
        return {}
