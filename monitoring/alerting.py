"""
alerting.py — Sends alerts when the bot needs attention.
Currently logs to console; extend with email/Telegram as needed.
"""

import logging

logger = logging.getLogger(__name__)


def send_alert(message: str, level: str = "warning") -> None:
    if level == "critical":
        logger.critical(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)
    # TODO: add Telegram / email notification here
