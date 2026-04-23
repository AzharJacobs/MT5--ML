"""
test_backtest.py — Tests backtest metrics produce expected results.
"""

import pandas as pd
import numpy as np
import pytest
from backtest.metrics import compute_metrics


def _make_equity(n: int = 100, trend: float = 0.001) -> pd.Series:
    rng = pd.date_range("2024-01-01", periods=n, freq="D")
    equity = 10_000 * (1 + trend) ** np.arange(n)
    return pd.Series(equity, index=rng)


def _make_trades(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"pnl": np.random.randn(n) * 100})


def test_compute_metrics_keys():
    metrics = compute_metrics(_make_equity(), _make_trades())
    for key in ("sharpe_ratio", "max_drawdown", "win_rate", "total_pnl", "num_trades"):
        assert key in metrics


def test_win_rate_range():
    metrics = compute_metrics(_make_equity(), _make_trades())
    assert 0.0 <= metrics["win_rate"] <= 1.0


def test_num_trades():
    trades = _make_trades(15)
    metrics = compute_metrics(_make_equity(), trades)
    assert metrics["num_trades"] == 15
