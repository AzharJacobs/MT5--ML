"""
position_sizer.py — Calculates position size based on account equity and risk rules.
Config: config/risk.yaml → position_sizing
"""


def calculate_position_size(
    account_equity: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    pip_value: float,
) -> float:
    """Return lot size so that a SL hit loses exactly risk_per_trade * account_equity."""
    risk_amount = account_equity * risk_per_trade
    if stop_loss_pips <= 0 or pip_value <= 0:
        return 0.0
    return round(risk_amount / (stop_loss_pips * pip_value), 2)
