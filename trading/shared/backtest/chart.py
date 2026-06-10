"""
chart_market_structure.py — Interactive Plotly chart for Market Structure backtest results.

Opens in browser. Shows:
  - Top panel : 15M OHLCV candlesticks with entry markers, SL/TP dotted lines, exit crosses
  - Bottom panel: Equity curve

Color coding:
  Green triangle  = winning entry (TP hit)
  Red   triangle  = losing entry  (SL hit)
  Grey  circle    = expired       (no fill within max_bars)
  Green dotted    = TP level line
  Red   dotted    = SL level line
  X marker        = exit point
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


_WIN        = "#00e676"              # green
_LOSS       = "#f44336"              # red
_EXPIRED    = "#9e9e9e"              # grey
_TP_LINE    = "rgba(0,230,118,0.65)"
_SL_LINE    = "rgba(244,67,54,0.65)"
_EQ_LINE    = "#2196f3"
_STRONG_SW  = "rgba(255,152,0,0.85)"   # amber  — 4H strong swing level
_WEAK_SW    = "rgba(33,150,243,0.85)"  # blue   — 4H weak swing level
_ZONE_FILL  = "rgba(255,152,0,0.07)"   # faint amber fill for 4H zone box
_ZONE_EDGE  = "rgba(255,152,0,0.40)"   # amber border for 4H zone box


def plot_trades(
    df_15m: pd.DataFrame,
    trades: list[dict],
    title: str = "Market Structure Backtest — 15M",
    start_cash: float = 10_000.0,
) -> go.Figure:
    """
    Build and return a Plotly Figure.

    Parameters
    ----------
    df_15m      : full 15M OHLCV dataframe used in the backtest
    trades      : list of trade dicts from run_backtest (must include 'exit_date')
    title       : chart window title
    start_cash  : initial equity (used as the first equity-curve point)
    """
    if not trades:
        raise ValueError("No trades to plot — run the backtest first.")

    df = df_15m.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.02,
        subplot_titles=["15M Price  •  Entries / SL / TP", "Equity Curve"],
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="15M",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
            whiskerwidth=0.3,
        ),
        row=1, col=1,
    )

    # ── Equity curve ─────────────────────────────────────────────────────────
    eq_x = [trades[0]["date"]] + [t["exit_date"] for t in trades]
    eq_y = [start_cash]        + [t["equity"]    for t in trades]

    fig.add_trace(
        go.Scatter(
            x=eq_x, y=eq_y,
            mode="lines+markers",
            name="Equity",
            line=dict(color=_EQ_LINE, width=2),
            marker=dict(size=4, color=_EQ_LINE),
        ),
        row=2, col=1,
    )

    # ── Entry markers — grouped by playbook × outcome ────────────────────────
    # Playbook A: triangle / circle shapes (original visual language)
    # Playbook C: diamond / square shapes  (counter-trend, distinct at a glance)
    _PB_SHAPES = {
        ("A",  1): ("triangle-up",   8),
        ("A", -1): ("triangle-down", 8),
        ("A",  0): ("circle",        6),
        ("C",  1): ("diamond",       9),
        ("C", -1): ("diamond",       8),
        ("C",  0): ("square",        7),
    }
    _OUTCOME_COLOR = { 1: _WIN,  -1: _LOSS,  0: _EXPIRED}
    _OUTCOME_LABEL = { 1: "Win", -1: "Loss", 0: "Exp"}
    # Playbook C gets a thicker border so it reads differently when overlapping A
    _PB_BORDER = {"A": ("white", 0.8), "C": ("white", 1.8)}

    for pb in ["A", "C"]:
        for outcome in [1, -1, 0]:
            group = [t for t in trades
                     if t.get("playbook", "A") == pb and t["outcome"] == outcome]
            if not group:
                continue
            symbol, size   = _PB_SHAPES[(pb, outcome)]
            color          = _OUTCOME_COLOR[outcome]
            bcol, bw       = _PB_BORDER[pb]
            y_markers = []
            for t in group:
                offset = abs(t["entry"] - t["sl"]) * 0.15
                y_markers.append(
                    t["entry"] - offset if t["side"] == "buy" else t["entry"] + offset
                )
            fig.add_trace(
                go.Scatter(
                    x=[t["date"] for t in group],
                    y=y_markers,
                    mode="markers",
                    name=f"{pb}-{_OUTCOME_LABEL[outcome]}",
                    marker=dict(
                        symbol=symbol, size=size, color=color,
                        line=dict(color=bcol, width=bw),
                    ),
                ),
                row=1, col=1,
            )

    # ── Exit crosses ─────────────────────────────────────────────────────────
    exit_colors = [
        _WIN if t["outcome"] == 1 else _LOSS if t["outcome"] == -1 else _EXPIRED
        for t in trades
    ]
    fig.add_trace(
        go.Scatter(
            x=[t["exit_date"] for t in trades],
            y=[t["exit"]      for t in trades],
            mode="markers",
            name="Exit",
            showlegend=True,
            marker=dict(
                symbol="x", size=9, color=exit_colors,
                line=dict(color="white", width=0.6),
            ),
        ),
        row=1, col=1,
    )

    # ── 4H structure overlay: zone box + strong/weak swing lines ─────────────
    # Drawn per trade for the trade's duration; zone renders below candles,
    # swing lines render above.  struct_shapes is merged before sl_tp_shapes
    # so that SL/TP lines always appear on top.
    struct_shapes = []
    for t in trades:
        struct = t.get("structure", {})
        zone   = struct.get("zone")
        strong = struct.get("strong_swing")
        weak   = struct.get("weak_swing")
        x0, x1 = str(t["date"]), str(t["exit_date"])

        if zone is not None:
            struct_shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=zone[0], y1=zone[1],
                fillcolor=_ZONE_FILL,
                line=dict(color=_ZONE_EDGE, width=0.8, dash="dot"),
                layer="below",
            ))
        if strong is not None:
            struct_shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=x0, x1=x1, y0=strong, y1=strong,
                line=dict(color=_STRONG_SW, width=1.2, dash="dashdot"),
            ))
        if weak is not None:
            struct_shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=x0, x1=x1, y0=weak, y1=weak,
                line=dict(color=_WEAK_SW, width=1.2, dash="dashdot"),
            ))

    # ── SL / TP dotted lines (shapes) ────────────────────────────────────────
    sl_tp_shapes = []
    for t in trades:
        x0, x1 = str(t["date"]), str(t["exit_date"])
        sl_tp_shapes.append(dict(
            type="line", xref="x", yref="y",
            x0=x0, x1=x1, y0=t["tp"], y1=t["tp"],
            line=dict(color=_TP_LINE, width=1.2, dash="dot"),
        ))
        sl_tp_shapes.append(dict(
            type="line", xref="x", yref="y",
            x0=x0, x1=x1, y0=t["sl"], y1=t["sl"],
            line=dict(color=_SL_LINE, width=1.2, dash="dot"),
        ))

    shapes = struct_shapes + sl_tp_shapes

    # ── Legend entries for structure colours (dummy traces — shapes have none) ─
    for leg_name, leg_color, leg_dash in [
        ("Strong Swing", _STRONG_SW, "dashdot"),
        ("Weak Swing",   _WEAK_SW,   "dashdot"),
        ("4H Zone",      _ZONE_EDGE, "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                name=leg_name,
                line=dict(color=leg_color, width=1.8, dash=leg_dash),
                showlegend=True,
            ),
            row=1, col=1,
        )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark",
        height=820,
        shapes=shapes,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode="x unified",
    )

    # Hide weekend gaps
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")

    return fig
