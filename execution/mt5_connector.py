"""
mt5_connector.py — Connects Python to the running MT5 terminal via MetaTrader5 package.
"""

import logging

logger = logging.getLogger(__name__)


class MT5Connector:
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self._connected = False

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                logger.error("MT5 initialize() failed")
                return False
            if not mt5.login(self.login, password=self.password, server=self.server):
                logger.error("MT5 login failed: %s", mt5.last_error())
                return False
            self._connected = True
            logger.info("Connected to MT5 as %s on %s", self.login, self.server)
            return True
        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False

    def disconnect(self) -> None:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
