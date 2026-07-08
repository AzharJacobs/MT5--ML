"""
report.py — Extended HTML report for the FXGold zone strategy.

Unlike trading/Reports/report_html.py (shared with USTEC, month-grouped,
fixed column set), this renders FXGold's fuller per-trade schema plus
diagnostics that only make sense for FXGold's gate pipeline:

  - Per-trade: entry/exit timestamp+price, SL, TP, outcome, pnl, confirmation
    pattern, tp_mode, zone TF + top/bottom + score, live touch count at
    entry, and the individual D1 + H4 bias readings at entry — so every
    trade shows why it was taken.
  - Per-year summary table (trades, WR, profit factor, max DD, net PnL).
  - Gate-attrition funnel: of all real candidate touches (rejection-touch
    events), how many were killed at each gate.
  - Reasons-blocked breakdown: within each gate, the specific reason strings
    that fired and how often.

Usage:
    from trading.strategies.FXGold.report import save_extended_html_report
    save_extended_html_report(
        df_t, gate_counts, reason_counts, candidate_touches,
        start=start, end=end, load_start=load_start, cash=cash,
        out_path="report.html", meta={...},
    )
"""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Dict

import pandas as pd


def _fmt(v, dp=2):
    try:
        return f"{float(v):,.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def _year_stats(df: pd.DataFrame, cash: float, start: str, end: str) -> pd.DataFrame:
    """Per-calendar-year trades/WR/PF/maxDD/netPnL, walking the equity curve
    chronologically so each year's drawdown is measured from the correct
    running peak rather than reset to that year's own trades in isolation.

    Includes every year in [start, end], even ones with zero trades — a
    year with nothing in it is itself a data point (e.g. a gate blocking
    everything that year), not something to silently omit."""
    df = df.sort_values("date").reset_index(drop=True)
    by_year = {int(y): g for y, g in df.groupby(df["date"].dt.year)}
    all_years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)

    rows = []
    running_eq = cash
    peak_eq    = cash
    empty = df.iloc[0:0]
    for year in all_years:
        g = by_year.get(year, empty)
        start_eq = running_eq
        year_dd  = 0.0
        for pnl in g["pnl"]:
            running_eq += pnl
            peak_eq = max(peak_eq, running_eq)
            dd = (running_eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0
            year_dd = min(year_dd, dd)
        wins  = int((g["outcome"] == 1).sum())
        total = len(g)
        gross_win  = g.loc[g["pnl"] > 0, "pnl"].sum() if total else 0.0
        gross_loss = g.loc[g["pnl"] < 0, "pnl"].sum() if total else 0.0
        if total == 0:
            pf = None
        elif gross_loss < 0:
            pf = gross_win / abs(gross_loss)
        else:
            pf = float("inf") if gross_win > 0 else None
        rows.append({
            "year": year, "trades": total, "wins": wins,
            "wr": wins / total * 100 if total else 0.0,
            "pf": pf, "max_dd": year_dd, "net_pnl": g["pnl"].sum() if total else 0.0,
            "start_eq": start_eq, "end_eq": running_eq,
        })
    return pd.DataFrame(rows)


# Gate buckets, in funnel order, mapped to a short human label.
_GATE_ORDER = [
    ("bias",            "Bias (D1/H4 alignment)"),
    ("touch_count",     "Touch count"),
    ("secret_pattern",  "Secret pattern (SOP 3)"),
    ("confirmation",    "Candlestick confirmation"),
    ("sl_geometry",     "SL geometry"),
    ("rr",              "Min R:R"),
    ("entry_confirmed", "Entry confirmed (trade taken)"),
]


def save_extended_html_report(
    df_t: pd.DataFrame,
    gate_counts: Dict[str, int],
    reason_counts: Dict[str, Dict[str, int]],
    candidate_touches: int,
    start: str,
    end: str,
    load_start: str,
    cash: float,
    out_path: str = "report.html",
    meta: dict | None = None,
    title: str = "FXGold Extended Backtest Report",
    eyebrow: str = "ZONE-TO-ZONE · XAUUSD · PDF-SPEC BACKTEST",
) -> str:
    df = df_t.copy()
    df["date"]      = pd.to_datetime(df["date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year

    total   = len(df)
    wins    = int((df["outcome"] == 1).sum())
    wr      = wins / max(total, 1) * 100
    net     = float(df["pnl"].sum())
    final   = cash + net
    gross_win  = df.loc[df["pnl"] > 0, "pnl"].sum()
    gross_loss = df.loc[df["pnl"] < 0, "pnl"].sum()
    pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_win > 0 else 0.0)
    eq = pd.concat([pd.Series([cash]), cash + df["pnl"].cumsum()])
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)

    years_df = _year_stats(df, cash, start, end)

    # ── Meta line ────────────────────────────────────────────────────────────
    meta = meta or {}
    meta_html = " &middot; ".join(
        [f"Start cash <b>${_fmt(cash)}</b>", f"Warmup from <b>{load_start}</b>"] +
        [f"{k} <b>{_html.escape(str(v))}</b>" for k, v in meta.items()]
    )

    # ── Per-year table ───────────────────────────────────────────────────────
    year_rows = []
    for _, r in years_df.iterrows():
        if pd.isna(r["pf"]):
            pf_str = "&mdash;"
        elif r["pf"] == float("inf"):
            pf_str = "&infin;"
        else:
            pf_str = f"{r['pf']:.2f}"
        pnl_cls = "pos" if r["net_pnl"] >= 0 else "neg"
        year_rows.append(
            "<tr>"
            f"<td class='d'>{int(r['year'])}</td>"
            f"<td>{int(r['trades'])}</td>"
            f"<td>{int(r['wins'])}</td>"
            f"<td>{r['wr']:.1f}%</td>"
            f"<td>{pf_str}</td>"
            f"<td class='neg'>{r['max_dd']:.2f}%</td>"
            f"<td class='{pnl_cls}'>{r['net_pnl']:+.2f}</td>"
            f"<td>${_fmt(r['start_eq'])} &rarr; ${_fmt(r['end_eq'])}</td>"
            "</tr>"
        )
    year_table = (
        "<table><thead><tr><th>YEAR</th><th>TRADES</th><th>WINS</th><th>WR</th>"
        "<th>PF</th><th>MAX DD</th><th>NET PNL</th><th>EQUITY</th></tr></thead>"
        f"<tbody>{''.join(year_rows)}</tbody></table>"
    )

    # ── Gate-attrition funnel ────────────────────────────────────────────────
    not_touch = gate_counts.get("not_a_touch_this_bar", 0)
    total_calls = candidate_touches + not_touch
    funnel_rows = []
    for key, label in _GATE_ORDER:
        v = gate_counts.get(key, 0)
        pct = (v / candidate_touches * 100) if candidate_touches else 0.0
        bar_w = max(1, pct)
        cls = "gate-pass" if key == "entry_confirmed" else "gate-block"
        funnel_rows.append(
            "<tr>"
            f"<td class='d'>{_html.escape(label)}</td>"
            f"<td>{v:,}</td>"
            f"<td>{pct:.1f}%</td>"
            f"<td class='bar-cell'><div class='bar-track'><div class='bar-fill {cls}' "
            f"style='width:{bar_w:.1f}%'></div></div></td>"
            "</tr>"
        )
    funnel_table = (
        "<table><thead><tr><th>GATE</th><th>COUNT</th><th>% OF CANDIDATE TOUCHES</th>"
        "<th>&nbsp;</th></tr></thead>"
        f"<tbody>{''.join(funnel_rows)}</tbody></table>"
    )

    # ── Reasons-blocked breakdown per gate ───────────────────────────────────
    reason_sections = []
    for key, label in _GATE_ORDER:
        reasons = reason_counts.get(key, {})
        if not reasons:
            continue
        rows = "".join(
            f"<tr><td>{_html.escape(reason)}</td><td>{count:,}</td></tr>"
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])
        )
        reason_sections.append(
            f"<div class='reason-block'><h3>{_html.escape(label)}</h3>"
            f"<table><thead><tr><th>REASON</th><th>COUNT</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    reasons_html = "".join(reason_sections) or "<p class='empty'>No rejection reasons recorded.</p>"

    # ── tp_mode split ────────────────────────────────────────────────────────
    tp_counts = df["tp_mode"].value_counts()
    tp_html = " &middot; ".join(f"<b>{_html.escape(str(k))}</b>: {v}" for k, v in tp_counts.items()) or "n/a"

    # ── Per-year trade tables ────────────────────────────────────────────────
    cols   = ["date", "side", "entry", "sl", "tp", "exit", "exit_date", "outcome", "pnl",
              "pattern", "tp_mode", "zone_tf", "zone_kind", "zone_top", "zone_bottom",
              "zone_score", "live_touch_count", "d1_bias", "h4_bias"]
    heads  = ["ENTRY DATE", "SIDE", "ENTRY", "SL", "TP", "EXIT", "EXIT DATE", "OUTCOME", "PNL",
              "PATTERN", "TP MODE", "ZONE TF", "ZONE KIND", "ZONE TOP", "ZONE BOT",
              "ZONE SCORE", "TOUCH #", "D1 BIAS", "H4 BIAS"]

    year_sections = []
    for year, g in df.groupby("year"):
        rows = []
        for _, r in g.iterrows():
            outcome = {1: "TP", -1: "SL", 0: "EXP"}.get(r["outcome"], r["outcome"])
            pnl_cls = "pos" if r["pnl"] >= 0 else "neg"
            side_cls = r["side"]
            cells = [
                f"<td class='d'>{r['date'].strftime('%Y-%m-%d %H:%M')}</td>",
                f"<td class='{side_cls}'>{r['side']}</td>",
                f"<td>{_fmt(r['entry'])}</td>",
                f"<td>{_fmt(r['sl'])}</td>",
                f"<td>{_fmt(r['tp'])}</td>",
                f"<td>{_fmt(r['exit'])}</td>",
                f"<td>{r['exit_date'].strftime('%Y-%m-%d %H:%M')}</td>",
                f"<td>{outcome}</td>",
                f"<td class='{pnl_cls}'>{r['pnl']:+.2f}</td>",
                f"<td>{_html.escape(str(r.get('pattern','')))}</td>",
                f"<td>{_html.escape(str(r.get('tp_mode','')))}</td>",
                f"<td>{_html.escape(str(r.get('zone_tf','')))}</td>",
                f"<td>{_html.escape(str(r.get('zone_kind','')))}</td>",
                f"<td>{_fmt(r.get('zone_top'))}</td>",
                f"<td>{_fmt(r.get('zone_bottom'))}</td>",
                f"<td>{_fmt(r.get('zone_score'))}</td>",
                f"<td>{r.get('live_touch_count','')}</td>",
                f"<td>{_html.escape(str(r.get('d1_bias','')))}</td>",
                f"<td>{_html.escape(str(r.get('h4_bias','')))}</td>",
            ]
            rows.append("<tr>" + "".join(cells) + "</tr>")
        wins_y = int((g["outcome"] == 1).sum())
        year_pnl = g["pnl"].sum()
        year_pnl_cls = "pos" if year_pnl >= 0 else "neg"
        year_sections.append(
            f"<section class='month' id='y{int(year)}'>"
            f"<div class='mh'><h2>{int(year)}</h2>"
            f"<div class='mstats'>{len(g)} trades &middot; {wins_y}/{len(g)} won &middot; "
            f"{wins_y/len(g)*100:.0f}% WR &middot; PnL "
            f"<b class='{year_pnl_cls}'>{year_pnl:+.2f}</b></div></div>"
            f"<div class='tbl'><table><thead><tr>"
            + "".join(f"<th>{h}</th>" for h in heads)
            + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            "</section>"
        )

    chips = "".join(
        f'<a class="chip has" href="#y{int(y)}">{int(y)}</a>' for y in sorted(df["year"].unique())
    )

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
:root {{
  --bg:#111110; --panel:#191816; --line:#2a2825;
  --text:#e8e4dc; --dim:#8f8a80; --gold:#d9a441;
  --green:#5fbf7a; --red:#e2685f;
  font-size:15px;
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--bg); color:var(--text);
  font:400 1rem/1.5 "Segoe UI",system-ui,sans-serif; padding:40px 16px; }}
.wrap {{ max-width:1100px; margin:0 auto; }}
.eyebrow {{ color:var(--gold); letter-spacing:.14em; font-size:.72rem; font-weight:600; }}
h1 {{ font-size:2rem; margin:6px 0 4px; }}
h2 {{ font-size:1.05rem; }}
h3 {{ font-size:.85rem; color:var(--gold); margin:0 0 8px; letter-spacing:.04em; }}
.meta {{ color:var(--dim); font-size:.85rem; border-bottom:1px solid var(--line);
  padding-bottom:18px; margin-bottom:24px; }}
.meta b {{ color:var(--text); font-weight:600; }}

.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  border:1px solid var(--line); border-radius:8px; overflow:hidden; margin-bottom:24px; }}
.card {{ background:var(--panel); padding:16px 18px; border-right:1px solid var(--line); }}
.card:last-child {{ border-right:0; }}
.card .lbl {{ color:var(--dim); font-size:.68rem; letter-spacing:.1em; }}
.card .val {{ font-size:1.35rem; font-weight:700; margin-top:4px; }}
.pos {{ color:var(--green); }} .neg {{ color:var(--red); }}

.section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:18px; margin-bottom:20px; }}
.section .lbl {{ color:var(--dim); font-size:.68rem; letter-spacing:.1em; margin-bottom:14px;
  text-transform:uppercase; }}

.chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:26px; }}
.chip {{ color:var(--dim); text-decoration:none; font-size:.75rem;
  border:1px solid var(--line); border-radius:6px; padding:4px 10px; }}
.chip.has {{ color:var(--text); border-color:var(--gold); }}

.month {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:18px; margin-bottom:14px; }}
.mh {{ display:flex; justify-content:space-between; align-items:baseline;
  flex-wrap:wrap; gap:8px; }}
.mstats {{ color:var(--dim); font-size:.8rem; }}

.tbl {{ overflow-x:auto; margin-top:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:.76rem; white-space:nowrap; }}
th {{ color:var(--dim); font-size:.63rem; letter-spacing:.06em; text-align:left;
  padding:6px 10px; border-bottom:1px solid var(--line); }}
td {{ padding:7px 10px; border-bottom:1px solid var(--line); }}
tr:last-child td {{ border-bottom:0; }}
td.buy {{ color:var(--green); }} td.sell {{ color:var(--red); }}
td.d {{ font-weight:600; }}
.empty {{ color:var(--dim); font-style:italic; }}

.bar-cell {{ width:200px; }}
.bar-track {{ background:var(--line); border-radius:4px; height:10px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:4px; }}
.bar-fill.gate-block {{ background:var(--red); }}
.bar-fill.gate-pass  {{ background:var(--green); }}

.reasons-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; }}
.reason-block table {{ font-size:.72rem; }}
</style></head><body><div class="wrap">
<div class="eyebrow">{_html.escape(eyebrow)}</div>
<h1>{_html.escape(title)}</h1>
<div class="meta">{start} &rarr; {end} &middot; {meta_html}</div>

<div class="cards">
  <div class="card"><div class="lbl">START EQUITY</div><div class="val">${_fmt(cash)}</div></div>
  <div class="card"><div class="lbl">FINAL EQUITY</div><div class="val">${_fmt(final)}</div></div>
  <div class="card"><div class="lbl">NET PNL</div><div class="val {'pos' if net>=0 else 'neg'}">{net:+,.2f}</div></div>
  <div class="card"><div class="lbl">TRADES</div><div class="val">{total}</div></div>
  <div class="card"><div class="lbl">WIN RATE</div><div class="val">{wr:.1f}%</div></div>
  <div class="card"><div class="lbl">PROFIT FACTOR</div><div class="val">{'&infin;' if pf==float('inf') else f'{pf:.2f}'}</div></div>
  <div class="card"><div class="lbl">MAX DRAWDOWN</div><div class="val neg">{max_dd:.2f}%</div></div>
</div>

<div class="section"><div class="lbl">Per-year summary</div>{year_table}</div>

<div class="section"><div class="lbl">Gate attrition funnel &mdash; {candidate_touches:,} candidate touches
  ({total_calls:,} total evaluate_entry() calls, {not_touch:,} not a touch this bar)</div>{funnel_table}</div>

<div class="section"><div class="lbl">Reasons blocked, per gate</div>
  <div class="reasons-grid">{reasons_html}</div>
</div>

<div class="section"><div class="lbl">TP mode split</div>{tp_html}</div>

<div class="chips">{chips}</div>
{''.join(year_sections)}
</div></body></html>"""

    p = Path(out_path)
    p.write_text(page, encoding="utf-8")
    print(f"Extended report saved: {p.resolve()}")
    return str(p)
