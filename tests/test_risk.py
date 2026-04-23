"""
test_risk.py — Tests position sizing and stop loss logic.
"""

import pytest
from risk.position_sizer import calculate_position_size
from risk.stop_loss import get_dynamic_stop_loss
from risk.portfolio_manager import PortfolioManager


def test_position_size_basic():
    size = calculate_position_size(
        account_equity=10_000,
        risk_per_trade=0.01,
        stop_loss_pips=20,
        pip_value=10,
    )
    assert size == 0.5


def test_position_size_zero_sl():
    size = calculate_position_size(10_000, 0.01, 0, 10)
    assert size == 0.0


def test_stop_loss_buy_below_entry():
    sl = get_dynamic_stop_loss(entry_price=100.0, direction="buy", atr=0.2)
    assert sl < 100.0


def test_stop_loss_sell_above_entry():
    sl = get_dynamic_stop_loss(entry_price=100.0, direction="sell", atr=0.2)
    assert sl > 100.0


def test_portfolio_manager_max_positions():
    pm = PortfolioManager(max_positions=2)
    pm.open_position("t1", {})
    pm.open_position("t2", {})
    assert not pm.can_open()


def test_portfolio_manager_daily_limit():
    pm = PortfolioManager(daily_loss_limit=0.03)
    pm.close_position("t1", -0.04)
    assert not pm.can_open()
