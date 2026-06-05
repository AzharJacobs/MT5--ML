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

    @staticmethod
    def _get_tick(symbol: str):
        """
        Fetch a live tick, ensuring the symbol is selected in Market Watch first.
        Exness symbols are often suffixed (e.g. USTECm) — symbol_info_tick returns
        None if the name is wrong, the symbol isn't selected, or the market is closed.
        Returns the tick or None.
        """
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            # Symbol may not be enabled in Market Watch — try selecting it, then retry.
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            # Log diagnostics to help identify the correct symbol name on this server.
            err = mt5.last_error()
            info = mt5.symbol_info(symbol)
            trade_mode = getattr(info, "trade_mode", "N/A") if info else "no symbol_info"
            similar = [s.name for s in (mt5.symbols_get() or [])
                       if symbol.upper()[:5] in s.name.upper() or "NAS" in s.name.upper()][:10]
            logger.debug(
                "_get_tick: %s → None | last_error=%s | trade_mode=%s | similar=%s",
                symbol, err, trade_mode, similar,
            )
        return tick

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

        tick = self._get_tick(symbol)
        if tick is None:
            logger.error(
                "place_order: no tick for %s — check symbol name/Market Watch/market hours. "
                "Order NOT placed.", symbol
            )
            return None

        price = tick.ask if direction == "buy" else tick.bid
        if not price:
            logger.error("place_order: tick for %s has no valid price (bid=%s ask=%s). Order NOT placed.",
                         symbol, tick.bid, tick.ask)
            return None

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
        if result is None:
            logger.error("place_order: order_send returned None for %s — last_error=%s",
                         symbol, mt5.last_error())
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed: retcode=%s %s", result.retcode, result.comment)
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

        tick = self._get_tick(pos.symbol)
        if tick is None:
            logger.error(
                "close_order: no tick for %s (ticket=%d) — check symbol/Market Watch/market hours. "
                "Position NOT closed.", pos.symbol, ticket
            )
            return False

        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        if not price:
            logger.error("close_order: tick for %s has no valid price (bid=%s ask=%s). Position NOT closed.",
                         pos.symbol, tick.bid, tick.ask)
            return False

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
        if result is None:
            logger.error("close_order: order_send returned None for ticket=%d — last_error=%s",
                         ticket, mt5.last_error())
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("close_order failed: ticket=%d retcode=%s %s", ticket, result.retcode, result.comment)
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