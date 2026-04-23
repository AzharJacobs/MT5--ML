"""
metrics.py — Backtest performance metrics: Sharpe, drawdown, win rate, P&L.
Called by backtest/engine.py and backtest/report.py.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> Dict:
    returns = equity_curve.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    winners = trades[trades["pnl"] > 0]
    win_rate = len(winners) / len(trades) if len(trades) > 0 else 0.0
    total_pnl = trades["pnl"].sum() if "pnl" in trades.columns else 0.0
    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "num_trades": len(trades),
    }
