"""
visualization.py — Plots equity curve and trade entry/exit markers.
Requires matplotlib.
"""

import pandas as pd


def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    equity_curve.plot(ax=ax, label="Equity")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    plt.tight_layout()
    plt.show()
