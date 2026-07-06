"""
report_html.py — Monthly Trade Report generator (Zone-to-Zone).

Usage from your backtest:

    from report_html import save_html_report
    metrics, df_t = run_backtest(...)
    save_html_report(df_t, start="2023-01-01", end="2025-01-01",
                     cash=150.0, out_path="report.html",
                     meta={"risk": "1.00%/trade", "spread": "2.0 pts",
                           "min confirmations": "2"})

Requires df_t with columns: date, exit_date (optional), side, h4_bias,
signals, confirmations, zone_fresh, zone_kind, entry, sl, tp, exit,
pnl, outcome, equity.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd


def _fmt(v, dp=2):
    try:
        return f"{float(v):,.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def save_html_report(df_t: pd.DataFrame, start: str, end: str, cash: float,
                     out_path: str = "report.html", meta: dict | None = None,
                     title: str = "Monthly Trade Report",
                     eyebrow: str = "ZONE-TO-ZONE · USTECH · BACKTEST") -> str:
    df = df_t.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.strftime("%Y-%m")

    # ── Summary metrics ──────────────────────────────────────────────────────
    total   = len(df)
    wins    = int((df["outcome"] == 1).sum())
    wr      = wins / max(total, 1) * 100
    net     = float(df["pnl"].sum())
    final   = cash + net
    eq      = pd.concat([pd.Series([cash]), cash + df["pnl"].cumsum()])
    max_dd  = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)

    # ── Month list covering full range (incl. no-trade months) ──────────────
    months = pd.period_range(start=start, end=end, freq="M").strftime("%Y-%m").tolist()
    by_month = {m: df[df["month"] == m] for m in months}
    month_pnl = {m: float(g["pnl"].sum()) for m, g in by_month.items()}
    max_abs = max((abs(v) for v in month_pnl.values()), default=1) or 1

    # running equity at end of each month
    run_eq, cur = {}, cash
    for m in months:
        cur += month_pnl[m]
        run_eq[m] = cur

    # ── Bar chart ────────────────────────────────────────────────────────────
    bars = []
    for m in months:
        v = month_pnl[m]
        h = max(2, abs(v) / max_abs * 70)
        cls = "up" if v > 0 else ("dn" if v < 0 else "flat")
        bars.append(
            f'<div class="bar-wrap" title="{m}: {v:+.2f}">'
            f'<div class="bar {cls}" style="height:{h:.0f}px"></div></div>'
        )
    labels = "".join(f"<span>{m[2:].replace('-','-')}</span>" for m in months[::3])

    # ── Meta line ────────────────────────────────────────────────────────────
    meta = meta or {}
    meta_html = " &middot; ".join(
        [f"Start cash <b>${_fmt(cash)}</b>"] +
        [f"{k} <b>{_html.escape(str(v))}</b>" for k, v in meta.items()]
    )

    # ── Month sections ───────────────────────────────────────────────────────
    cols = ["date", "side", "h4_bias", "signals", "confirmations", "zone_fresh",
            "zone_kind", "entry", "sl", "tp", "exit", "pnl", "outcome"]
    heads = ["DATE", "SIDE", "H4 BIAS", "SIGNALS", "CONF", "FRESH", "ZONE",
             "ENTRY", "SL", "TP", "EXIT", "PNL", "OUTCOME"]

    sections = []
    dd_run = 0.0
    peak = cash
    for m in months:
        g = by_month[m]
        peak = max(peak, run_eq[m])
        dd_run = (run_eq[m] - peak) / peak * 100
        head_right = (
            f'<span class="pill">no trades</span>' if g.empty else
            f'<span class="wr {"bad" if (g["outcome"]==1).sum()==0 else "ok"}">'
            f'{int((g["outcome"]==1).sum())}/{len(g)} won · '
            f'{(g["outcome"]==1).mean()*100:.0f}% WR</span>'
        )
        stats = (f'PnL <b class="{"pos" if month_pnl[m] >= 0 else "neg"}">'
                 f'{month_pnl[m]:+.2f}</b> &nbsp; End Eq ${_fmt(run_eq[m])} '
                 f'&nbsp; DD {dd_run:.2f}%')

        if g.empty:
            body = '<p class="empty">No trades this month.</p>'
        else:
            rows = []
            for _, r in g.iterrows():
                pnl_cls = "pos" if r["pnl"] >= 0 else "neg"
                outcome = {1: "TP", -1: "SL", 0: "EXP"}.get(r["outcome"], r["outcome"])
                fresh = "fresh" if r.get("zone_fresh") else "tapped"
                cells = [
                    f'<td class="d">{pd.to_datetime(r["date"]).strftime("%Y-%m-%d %H:%M")}</td>',
                    f'<td class="{r["side"]}">{r["side"]}</td>',
                    f'<td>{r.get("h4_bias","")}</td>',
                    f'<td class="sig">{_html.escape(str(r.get("signals","")))}</td>',
                    f'<td>{r.get("confirmations","")}</td>',
                    f'<td>{fresh}</td>',
                    f'<td>{r.get("zone_kind","")}</td>',
                    f'<td>{_fmt(r["entry"])}</td>',
                    f'<td>{_fmt(r["sl"])}</td>',
                    f'<td>{_fmt(r["tp"])}</td>',
                    f'<td>{_fmt(r["exit"])}</td>',
                    f'<td class="{pnl_cls}">{r["pnl"]:+.2f}</td>',
                    f'<td>{outcome}</td>',
                ]
                rows.append("<tr>" + "".join(cells) + "</tr>")
            body = (
                '<div class="tbl"><table><thead><tr>'
                + "".join(f"<th>{h}</th>" for h in heads)
                + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            )

        sections.append(
            f'<section class="month" id="m{m}">'
            f'<div class="mh"><h2>{m}</h2>'
            f'<div class="mstats">{head_right} &nbsp; {stats}</div></div>'
            f'{body}</section>'
        )

    # month nav chips
    chips = "".join(
        f'<a class="chip {"has" if not by_month[m].empty else ""}" href="#m{m}">{m}</a>'
        for m in months
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
.wrap {{ max-width:960px; margin:0 auto; }}
.eyebrow {{ color:var(--gold); letter-spacing:.14em; font-size:.72rem; font-weight:600; }}
h1 {{ font-size:2rem; margin:6px 0 4px; }}
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

.chart {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:18px; margin-bottom:20px; }}
.chart .lbl {{ color:var(--dim); font-size:.68rem; letter-spacing:.1em; margin-bottom:14px; }}
.bars {{ display:flex; align-items:center; gap:4px; height:150px; }}
.bar-wrap {{ flex:1; display:flex; align-items:center; height:100%; }}
.bar {{ width:100%; border-radius:2px; }}
.bar.up   {{ background:var(--green); align-self:flex-end; transform:translateY(-50%); }}
.bar.dn   {{ background:var(--red);   align-self:flex-start; transform:translateY(50%); }}
.bar.flat {{ background:var(--line); height:2px !important; align-self:center; }}
.bars {{ position:relative; }}
.bars::after {{ content:""; position:absolute; left:0; right:0; top:50%;
  border-top:1px solid var(--line); }}
.xlabels {{ display:flex; justify-content:space-between; color:var(--dim);
  font-size:.65rem; margin-top:8px; }}

.chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:26px; }}
.chip {{ color:var(--dim); text-decoration:none; font-size:.75rem;
  border:1px solid var(--line); border-radius:6px; padding:4px 10px; }}
.chip.has {{ color:var(--text); border-color:var(--gold); }}

.month {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:18px; margin-bottom:14px; }}
.mh {{ display:flex; justify-content:space-between; align-items:baseline;
  flex-wrap:wrap; gap:8px; }}
.mh h2 {{ font-size:1.05rem; }}
.mstats {{ color:var(--dim); font-size:.8rem; }}
.wr.bad {{ color:var(--red); }} .wr.ok {{ color:var(--green); }}
.pill {{ border:1px solid var(--line); border-radius:20px; padding:2px 10px;
  font-size:.72rem; color:var(--dim); }}
.empty {{ color:var(--dim); font-style:italic; margin-top:10px; font-size:.85rem; }}

.tbl {{ overflow-x:auto; margin-top:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:.78rem; white-space:nowrap; }}
th {{ color:var(--dim); font-size:.65rem; letter-spacing:.08em; text-align:left;
  padding:6px 10px; border-bottom:1px solid var(--line); }}
td {{ padding:7px 10px; border-bottom:1px solid var(--line); }}
tr:last-child td {{ border-bottom:0; }}
td.buy {{ color:var(--green); }} td.sell {{ color:var(--red); }}
td.d {{ font-weight:600; }} td.sig {{ color:var(--dim); }}
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
  <div class="card"><div class="lbl">MAX DRAWDOWN</div><div class="val neg">{max_dd:.2f}%</div></div>
</div>

<div class="chart"><div class="lbl">MONTH PNL ($)</div>
  <div class="bars">{''.join(bars)}</div>
  <div class="xlabels">{labels}</div>
</div>

<div class="chips">{chips}</div>
{''.join(sections)}
</div></body></html>"""

    p = Path(out_path)
    p.write_text(page, encoding="utf-8")
    print(f"Report saved: {p.resolve()}")
    return str(p)
