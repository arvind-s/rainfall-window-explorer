#!/usr/bin/env python3
"""Generate a static HTML dashboard per polygon (unique id) — same content as the app.

For every unique block in the shapefile / blocks.geojson it writes a self-contained
`<id>.html` reproducing the Streamlit output: per rainfall source (GSMaP / IMD, as
tabs) the May–Dec 10-year rainy-days & cumulative-rainfall heatmaps, the current-year
"this season so far" bars, a WeatherNext forecast (fetched live at generation time),
a block map, and the weekly table. Plus an `index.html` linking them all.

Usage:
    python generate_html.py                       # all blocks, both sources, threshold 1mm
    python generate_html.py --sources gsmap --threshold 2.5
    python generate_html.py --id-col block_id --no-forecast --limit 3
"""
from __future__ import annotations

import argparse
import html as _html
import os
import re

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd

from rainfall import metrics, figures
from rainfall import weathernext as wnx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "out")
SM, EM = 5, 12
SRC_NAME = {"gsmap": "🛰️ JAXA GSMaP (satellite, ~11 km)",
            "imd": "🌧️ IMD gauge grid (~28 km)"}
DAYS_RAMP, RAIN_RAMP = figures.DAYS_RAMP, figures.RAIN_RAMP

PAGE_CSS = """
:root{--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--bg:#f8fafc}
*{box-sizing:border-box}
body{font-family:'Source Sans Pro','Segoe UI',sans-serif;color:var(--ink);margin:0;background:#fff}
.wrap{max-width:1050px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:1.9rem;margin:.2em 0 .1em}
.sub{color:var(--mut);font-size:.95rem;margin-bottom:14px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
.kpi{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:12px 18px;min-width:150px}
.kpi .l{color:var(--mut);font-size:.8rem}
.kpi .v{font-size:1.6rem;font-weight:600}
.tabs{display:flex;gap:6px;border-bottom:2px solid var(--line);margin:18px 0 8px}
.tab{padding:8px 14px;cursor:pointer;font-weight:600;color:var(--mut);border:none;background:none;font-size:1rem}
.tab.active{color:var(--ink);border-bottom:3px solid #2b6cb0;margin-bottom:-2px}
.panel{display:none}.panel.active{display:block}
h3{margin:22px 0 2px}.cap{color:var(--mut);font-size:.88rem;margin-bottom:6px}
table{border-collapse:collapse;font-size:.85rem;width:100%}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:right}th{background:var(--bg)}
td:first-child,th:first-child{text-align:left}
.note{color:var(--mut);font-size:.82rem;margin-top:6px}
a{color:#2b6cb0}
"""

TAB_JS = """
<script>
function showSrc(el,grp,src){
 document.querySelectorAll('.tab[data-grp="'+grp+'"]').forEach(t=>t.classList.remove('active'));
 document.querySelectorAll('.panel[data-grp="'+grp+'"]').forEach(p=>p.classList.remove('active'));
 el.classList.add('active');
 document.querySelector('.panel[data-grp="'+grp+'"][data-src="'+src+'"]').classList.add('active');
}
</script>
"""


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_") or "block"


def plot_div(fig, state):
    inc = "cdn" if not state["loaded"] else False
    state["loaded"] = True
    return fig.to_html(full_html=False, include_plotlyjs=inc, config={"displayModeBar": False})


def kpi(label, value):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>'


def source_panel(block, src, wby, clim, cur, threshold, state, active):
    wb = wby[wby["block"] == block].sort_values(["week", "year"])
    cb = clim[clim["block"] == block].sort_values("week")
    full = cb[cb["is_full_week"]]
    wettest = full.loc[full["mean_rainy_days"].idxmax()] if len(full) else None
    kpis = "".join([
        kpi("Rainy days (season)", f"{full['mean_rainy_days'].sum():.0f}"),
        kpi("Rainfall (season)", f"{full['mean_total_mm'].sum():.0f} mm"),
        kpi("Wettest week", metrics.week_label(int(wettest['week']), SM, EM) if wettest is not None else "–"),
        kpi("Longest dry run", f"{full['mean_longest_dry_run'].max():.0f} d"),
    ])
    hm_days = plot_div(figures.heatmap(wb, "rainy_days", DAYS_RAMP, "days",
                                       lambda v: f"{v:.0f}", zmax=7, sm=SM, em=EM), state)
    hm_rain = plot_div(figures.heatmap(wb, "total_mm", RAIN_RAMP, "mm",
                                       lambda v: f"{v:.0f}", sm=SM, em=EM), state)
    n_years = wb["year"].nunique()
    parts = [f'<div class="kpis">{kpis}</div>',
             f"<h3>Rainy days per week &middot; &gt; {threshold:g} mm</h3>",
             f'<div class="cap">May–Dec, {n_years} years. Rows: weeks May 1 → Dec 31 · '
             f'columns: each year + {n_years}-yr average.</div>', hm_days,
             "<h3>Cumulative rainfall per week &middot; mm</h3>", hm_rain]
    # current-year "this season so far"
    if cur is not None:
        cc = cur[cur["block"] == block].sort_values("week")
        if len(cc):
            cy = int(cc["year"].iloc[0])
            parts += [f"<h3>This year: {cy} so far</h3>",
                      plot_div(figures.year_bars(cc, "rainy_days", "#2b6cb0", "days",
                                                 lambda v: f"{v:.0f}",
                                                 f"Rainy days per week (> {threshold:g} mm)", SM, EM), state),
                      plot_div(figures.year_bars(cc, "total_mm", "#128577", "mm",
                                                 lambda v: f"{v:.0f}",
                                                 "Cumulative rainfall per week", SM, EM), state)]
    # weekly table (season averages)
    t = cb.copy()
    t["Week"] = t["week"].apply(lambda w: metrics.week_label(int(w), SM, EM))
    tbl = t[["Week", "mean_rainy_days", "mean_total_mm", "mean_extreme_days",
             "mean_dry_days", "mean_longest_dry_run"]].round(1).rename(columns={
        "mean_rainy_days": "Rainy days", "mean_total_mm": "Rain (mm)",
        "mean_extreme_days": "Extreme days", "mean_dry_days": "Dry days",
        "mean_longest_dry_run": "Longest dry run"})
    parts += ["<h3>Weekly detail (season averages)</h3>",
              tbl.to_html(index=False, border=0)]
    cls = "panel active" if active else "panel"
    return f'<div class="{cls}" data-grp="src" data-src="{src}">' + "".join(parts) + "</div>"


def forecast_section(block, blocks_geo, threshold, state):
    cen = blocks_geo[blocks_geo["block"] == block].geometry.centroid
    lat, lon = float(cen.y.iloc[0]), float(cen.x.iloc[0])
    try:
        fb = wnx.forecast_point(round(lat, 3), round(lon, 3), rainy_mm=threshold)
        fb = fb[fb["rain_mm"].notna()].reset_index(drop=True)
    except Exception:
        fb = None
    if fb is None or fb.empty:
        return '<h3>🔮 Forecast</h3><div class="note">WeatherNext forecast unavailable at build time.</div>'
    rainy_ahead = int((fb["rain_mm"] > threshold).sum())
    kpis = "".join([kpi("Forecast rainfall", f"{fb['rain_mm'].sum():.0f} mm"),
                    kpi("Rainy days ahead", f"{rainy_ahead}"),
                    kpi("Wettest day", f"{pd.to_datetime(fb.loc[fb['rain_mm'].idxmax(),'date']):%d %b}")])
    return (f"<h3>🔮 Rainfall forecast &middot; WeatherNext 2 (Google DeepMind)</h3>"
            f'<div class="cap">64-member ensemble via Open-Meteo · next {len(fb)} days from '
            f'{fb["date"].min():%d %b %Y} · baked at generation time.</div>'
            f'<div class="kpis">{kpis}</div>' + plot_div(figures.forecast_bars(fb), state))


def map_section(block, blocks_geo, agg_by_src, src0):
    b = blocks_geo.merge(agg_by_src[src0], on="block", how="left").copy()
    b["geometry"] = b.geometry.simplify(0.01)          # ~1 km — keeps files light
    sel = b[b["block"] == block]
    cen = sel.geometry.centroid
    m = folium.Map(location=[cen.y.iloc[0], cen.x.iloc[0]], zoom_start=8, tiles="CartoDB positron")
    vmin, vmax = float(b["season_rainy"].min()), float(b["season_rainy"].max())
    if vmax - vmin < 1e-9:
        vmax = vmin + 1
    cmap = cm.LinearColormap(DAYS_RAMP, vmin=vmin, vmax=vmax, caption="Rainy days (season avg)")
    folium.GeoJson(b.__geo_interface__, style_function=lambda f: {
        "fillColor": cmap(f["properties"].get("season_rainy") or vmin),
        "color": "#94a3b8", "weight": .5, "fillOpacity": .75},
        tooltip=folium.GeoJsonTooltip(fields=["block", "district", "state"])).add_to(m)
    folium.GeoJson(sel.__geo_interface__, style_function=lambda f: {
        "color": "#dc2626", "weight": 3, "fillOpacity": 0}).add_to(m)
    cmap.add_to(m)
    return "<h3>🗺️ Location</h3>" + m._repr_html_()


def build_html(block, blocks_geo, data, agg_by_src, threshold, do_forecast):
    row = blocks_geo[blocks_geo["block"] == block].iloc[0]
    state = {"loaded": False}
    srcs = list(data)
    tabs = "".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-grp="src" '
        f'onclick="showSrc(this,\'src\',\'{s}\')">{SRC_NAME[s]}</button>'
        for i, s in enumerate(srcs))
    panels = "".join(
        source_panel(block, s, data[s][0], data[s][1], data[s][2], threshold, state, i == 0)
        for i, s in enumerate(srcs))
    fc = forecast_section(block, blocks_geo, threshold, state) if do_forecast else ""
    mp = map_section(block, blocks_geo, agg_by_src, srcs[0])
    title = _html.escape(str(block))
    sub = _html.escape(f"{row.get('district','')}, {row.get('state','')}")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Rainfall</title><style>{PAGE_CSS}</style>{TAB_JS}</head><body><div class="wrap">
<h1>🌧️ {title}</h1>
<div class="sub">{sub} &nbsp;|&nbsp; May–Dec 2016–2025 (10 yrs) · rainy day = &gt; {threshold:g} mm ·
<a href="index.html">← all blocks</a></div>
<div class="tabs">{tabs}</div>{panels}
{fc}{mp}
<div class="note">Sources: JAXA GSMaP · IMD Pune · WeatherNext 2 (Open-Meteo). Static export of the
Rainfall Window Explorer.</div>
</div></body></html>"""


def build_index(blocks_geo, agg, out_dir, files):
    rows = blocks_geo[blocks_geo["block"].isin(files)].merge(
        agg, on="block", how="left").sort_values(["state", "district", "block"])
    trs = "".join(
        f'<tr><td><a href="{files[r.block]}">{_html.escape(str(r.block))}</a></td>'
        f'<td>{_html.escape(str(r.district))}</td><td>{_html.escape(str(r.state))}</td>'
        f'<td>{(r.season_rainy or 0):.0f}</td><td>{(r.season_mm or 0):.0f}</td></tr>'
        for r in rows.itertuples())
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Rainfall — all blocks</title>
<style>{PAGE_CSS}</style></head><body><div class="wrap">
<h1>🌧️ Rainfall Window Explorer</h1>
<div class="sub">{len(files)} blocks · May–Dec 2016–2025 · click a block for its dashboard.</div>
<table><tr><th>Block</th><th>District</th><th>State</th><th>Rainy days</th><th>Rainfall (mm)</th></tr>
{trs}</table></div></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", choices=["gsmap", "imd"], default=["gsmap", "imd"])
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--id-col", default="block", help="column used for the HTML filename")
    ap.add_argument("--blocks", default=os.path.join(OUT, "blocks.geojson"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "html_out"))
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N blocks (testing)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    blocks_geo = gpd.read_file(args.blocks)
    if args.id_col not in blocks_geo.columns:
        raise SystemExit(f"--id-col '{args.id_col}' not in {list(blocks_geo.columns)}")

    # per-source: (weekly, climatology, current-year weekly); plus agg for the map
    data, agg_by_src = {}, {}
    for s in args.sources:
        daily = pd.read_parquet(os.path.join(OUT, f"daily_{s}.parquet"))
        wby = metrics.weekly_by_year(daily, start_month=SM, rainy_mm=args.threshold)
        clim = metrics.climatology(wby)
        agg_by_src[s] = clim.groupby("block").agg(
            season_rainy=("mean_rainy_days", "sum"),
            season_mm=("mean_total_mm", "sum")).reset_index()
        cur_path = os.path.join(OUT, f"daily_current_{s}.parquet")
        cur = (metrics.weekly_by_year(pd.read_parquet(cur_path), start_month=SM, rainy_mm=args.threshold)
               if os.path.exists(cur_path) else None)
        data[s] = (wby, clim, cur)

    names = list(blocks_geo["block"])
    if args.limit:
        names = names[:args.limit]
    files = {}
    for i, block in enumerate(names, 1):
        idval = blocks_geo.loc[blocks_geo["block"] == block, args.id_col].iloc[0]
        fname = f"{slug(idval)}.html"
        if fname in files.values():                     # keep filenames unique
            fname = f"{slug(idval)}_{i}.html"
        html_doc = build_html(block, blocks_geo, data, agg_by_src, args.threshold,
                              not args.no_forecast)
        with open(os.path.join(args.out_dir, fname), "w") as f:
            f.write(html_doc)
        files[block] = fname
        print(f"  [{i}/{len(names)}] {block} -> {fname}", flush=True)

    build_index(blocks_geo, agg_by_src[args.sources[0]], args.out_dir, files)
    print(f"Done -> {args.out_dir} ({len(files)} blocks + index.html)")


if __name__ == "__main__":
    main()
