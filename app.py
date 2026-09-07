#!/usr/bin/env python3
"""Rainfall Window Explorer -- JAXA GSMaP + IMD May-Dec analysis for field staff.

A clean, one-view-at-a-time dashboard. Choose a rainfall source at the top:
    * JAXA GSMaP -- satellite estimate, gauge-calibrated, ~11 km
    * IMD        -- rain-gauge gridded, ~28 km (ground reference for India)

Then, per block, over the historical record:
    * Rainy days per week (> 1 mm)      -- weekly heatmap
    * Cumulative rainfall per week (mm) -- weekly heatmap
    * A map of all blocks, and the raw weekly data table.

Pure data -- no scoring or 'suitability' judgement.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from rainfall import metrics
from rainfall import weathernext as wnx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RAINFALL_OUT", os.path.join(HERE, "data", "out"))

st.set_page_config(page_title="Rainfall Window Explorer",
                   page_icon="🌧️", layout="wide",
                   initial_sidebar_state="expanded")

# ---- rainfall sources -----------------------------------------------------
SOURCES = {
    "🛰️ JAXA GSMaP": {"key": "gsmap", "res": "satellite estimate, gauge-calibrated · ~11 km"},
    "🌧️ IMD gauge grid": {"key": "imd", "res": "rain-gauge gridded, IMD Pune · ~28 km"},
}

# ---- palette (sequential, single-hue ramps) -------------------------------
INK, MUTED, LINE = "#1f2937", "#64748b", "#cbd5e1"
DAYS_RAMP = ["#eef4fb", "#cfe1f2", "#9dc3e3", "#5a9bd4", "#2b6cb0", "#1a4f8a"]   # blues
RAIN_RAMP = ["#eff7f6", "#cdeae6", "#93d3cb", "#4db6ac", "#128577", "#0b5c52"]   # teals
PLOT_FONT = dict(family="Source Sans Pro, Segoe UI, sans-serif", color=INK, size=13)

MAP_METRICS = {
    "Rainy days (May–Dec avg)": ("season_rainy", "days", DAYS_RAMP),
    "Rainfall (May–Dec avg)": ("season_mm", "mm", RAIN_RAMP),
}

# single continuous historical season: May 1 – Dec 31
SEASON_SM, SEASON_EM = 5, 12

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1150px;}
h1, h2, h3 {font-family: 'Source Sans Pro', 'Segoe UI', sans-serif; color: #0f172a;}
div[data-testid="stMetric"] {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px;}
div[data-testid="stMetricLabel"] p {color: #64748b; font-size: .8rem;}
div[data-testid="stMetricValue"] {color: #0f172a; font-weight: 600; font-size: 1.7rem;}
button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.app-caption {color:#64748b; font-size:.9rem; margin:-6px 0 4px 0;}
</style>
"""


@st.cache_data(show_spinner=False)
def load_blocks():
    return gpd.read_file(os.path.join(OUT, "blocks.geojson"))


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_forecast(lat, lon, rainy_mm):
    """Live WeatherNext 2 forecast for a point (cached 1h). None on failure."""
    try:
        return wnx.forecast_point(round(lat, 3), round(lon, 3), rainy_mm=rainy_mm)
    except Exception:  # noqa: BLE001 -- network/API hiccup; caller shows a message
        return None


@st.cache_data(show_spinner=False)
def load_daily(key, suffix):
    """Daily block rainfall table for daily{suffix}_{key}.parquet, or None.

    suffix: '' = May–Dec historical, '_current' = current-year (May–present).
    """
    path = os.path.join(OUT, f"daily{suffix}_{key}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def weekly_hist(key, suffix, start_month, rainy_mm):
    """Historical weekly metrics recomputed from daily data at the chosen threshold."""
    daily = load_daily(key, suffix)
    if daily is None:
        return None, None, None
    wby = metrics.weekly_by_year(daily, start_month=start_month, rainy_mm=rainy_mm)
    clim = metrics.climatology(wby)
    agg = clim.groupby("block").agg(
        season_rainy=("mean_rainy_days", "sum"),
        season_mm=("mean_total_mm", "sum")).reset_index()
    return wby, clim, agg


@st.cache_data(show_spinner=False)
def weekly_current(key, rainy_mm):
    """Current-year (May–present) weekly metrics (single year), or None if not built."""
    daily = load_daily(key, "_current")
    if daily is None or daily.empty:
        return None
    return metrics.weekly_by_year(daily, start_month=5, rainy_mm=rainy_mm)


def styled_heatmap(wby_block, value_col, ramp, unit, val_fmt, zmax=None, sm=9, em=12):
    """Professional week (rows) x year (cols) heatmap for a season (sm..em)."""
    pivot = wby_block.pivot_table(index="week", columns="year",
                                  values=value_col, aggfunc="first").sort_index()
    years = [str(int(y)) for y in pivot.columns]
    avg = pivot.mean(axis=1)
    z = [list(r) + [a] for r, a in zip(pivot.values, avg.values)]
    xcols = years + [f"{len(years)}-yr avg"]
    ylabels = [metrics.week_label(int(w), start_month=sm, end_month=em) for w in pivot.index]
    zt = float(zmax) if zmax else float(np.nanmax(z) or 1)

    fig = go.Figure(go.Heatmap(
        z=z, x=xcols, y=ylabels, colorscale=ramp, zmin=0, zmax=zt,
        xgap=3, ygap=3, hoverongaps=False,
        colorbar=dict(title=dict(text=unit, side="right"), thickness=13,
                      len=0.9, outlinewidth=0, tickcolor=LINE, ticklen=4),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f} " + unit + "<extra></extra>"))

    anns = []  # per-cell labels; anchor by INTEGER index (year strings would
    for i, wk in enumerate(ylabels):          # coerce to numbers and break the axis)
        for j in range(len(xcols)):
            v = z[i][j]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            frac = v / zt if zt else 0
            anns.append(dict(x=j, y=i, xref="x", yref="y", text=val_fmt(v),
                             showarrow=False,
                             font=dict(color="white" if frac > 0.6 else INK, size=11)))

    fig.update_layout(
        annotations=anns, font=PLOT_FONT, height=26 * len(ylabels) + 120,
        margin=dict(t=52, b=10, l=8, r=8), paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(autorange="reversed", showgrid=False, ticks="", tickfont=dict(color=MUTED)),
        xaxis=dict(type="category", side="top", showgrid=False, ticks="",
                   tickfont=dict(color=INK, size=12)))
    fig.add_vline(x=len(years) - 0.5, line_width=1.5, line_color="#94a3b8", line_dash="dot")
    return fig


def year_bars(wby_block, value_col, color, unit, val_fmt, title):
    """Single-year weekly bar chart (used for the current-year section)."""
    g = wby_block.sort_values("week")
    labels = [metrics.week_label(int(w), start_month=5, end_month=12) for w in g["week"]]
    vals = g[value_col].tolist()
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=color, marker_line_width=0,
        text=[val_fmt(v) for v in vals], textposition="outside",
        textfont=dict(color=INK, size=11),
        hovertemplate="<b>%{x}</b><br>%{y:.1f} " + unit + "<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_layout(
        title=title, font=PLOT_FONT, height=330,
        margin=dict(t=52, b=70, l=8, r=8), paper_bgcolor="white", plot_bgcolor="white",
        bargap=0.25,
        yaxis=dict(title=unit, showgrid=True, gridcolor="#eef2f7", zeroline=False,
                   tickfont=dict(color=MUTED)),
        xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(color=INK, size=11)))
    return fig


# --------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

if not os.path.exists(os.path.join(OUT, "daily_gsmap.parquet")):
    st.title("🌧️ Rainfall Window Explorer")
    st.warning("Dataset not built yet. Run `python build_dataset.py` first.")
    st.stop()

blocks_geo = load_blocks()
block_names = sorted(blocks_geo["block"].dropna().unique())
info = blocks_geo.set_index("block")[["state", "district"]].to_dict("index")

# ---- selection state (dropdowns + map click share one selected block) -----
pending = st.session_state.pop("_pending_block", None)
if pending is None:
    qp_block = st.query_params.get("block")
    if qp_block in info and "sel_block" not in st.session_state:
        pending = qp_block
if pending in info:
    st.session_state["sel_state"] = info[pending]["state"]
    st.session_state["sel_district"] = info[pending]["district"]
    st.session_state["sel_block"] = pending

with st.sidebar:
    st.markdown("### 🌧️ Rainfall Window")
    st.caption("May–Dec · 10 yrs + this year")
    st.divider()
    st.markdown("**Select a block**")
    states = sorted(blocks_geo["state"].dropna().unique())
    state = st.selectbox("State", states, key="sel_state")

    dsub = blocks_geo[blocks_geo["state"] == state]
    districts = sorted(dsub["district"].dropna().unique())
    if st.session_state.get("sel_district") not in districts:
        st.session_state["sel_district"] = districts[0]
    district = st.selectbox("District", districts, key="sel_district")

    bsub = dsub[dsub["district"] == district]
    bopts = sorted(bsub["block"].dropna().unique())
    if st.session_state.get("sel_block") not in bopts:
        st.session_state["sel_block"] = bopts[0]
    block = st.selectbox("Block", bopts, key="sel_block")

    st.divider()
    rainy_mm = st.number_input("Rainy-day threshold (mm)", min_value=0.0, max_value=50.0,
                               value=1.0, step=0.5,
                               help="A day counts as 'rainy' if its rainfall exceeds this. "
                                    "Rainy/dry days recompute live; cumulative rainfall is unaffected.")

st.query_params["block"] = block

# ---- header + data-source switch ------------------------------------------
st.title("Rainfall Window Explorer")
src_label = st.segmented_control("Rainfall source", list(SOURCES),
                                 default=list(SOURCES)[0], label_visibility="collapsed")
if not src_label:
    src_label = list(SOURCES)[0]
src = SOURCES[src_label]
wby, clim, agg = weekly_hist(src["key"], "", SEASON_SM, rainy_mm)
blocks = blocks_geo.merge(agg, on="block", how="left")

# year coverage of the historical (May–Dec) record, derived from the data
years_present = sorted(int(y) for y in wby["year"].unique())
n_years = len(years_present)
yr_range = f"{years_present[0]}–{years_present[-1]}"

st.markdown(
    f"<div class='app-caption'><b>{block}</b> · {district}, {state} &nbsp;|&nbsp; "
    f"Source: <b>{src_label.split(' ', 1)[1]}</b> ({src['res']}) · "
    f"May–Dec {yr_range} ({n_years} yrs) · rainy day = &gt; {rainy_mm:g} mm</div>",
    unsafe_allow_html=True)
st.write("")

clim_b = clim[clim["block"] == block].sort_values("week")
wby_b = wby[wby["block"] == block].sort_values(["week", "year"])
full = clim_b[clim_b["is_full_week"]]

# ---- KPI strip ------------------------------------------------------------
wettest = full.loc[full["mean_rainy_days"].idxmax()] if len(full) else None
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rainy days", f"{full['mean_rainy_days'].sum():.0f}",
          help=f"Total days >{rainy_mm:g}mm across May–Dec, {n_years}-yr average")
k2.metric("Rainfall", f"{full['mean_total_mm'].sum():.0f} mm",
          help=f"Total rainfall across May–Dec, {n_years}-yr average")
k3.metric("Wettest week", metrics.week_label(int(wettest["week"]), SEASON_SM, SEASON_EM) if wettest is not None else "–",
          help="Week with the most rainy days on average")
k4.metric("Longest dry run", f"{full['mean_longest_dry_run'].max():.0f} days",
          help=f"Longest consecutive dry-day run in any week, {n_years}-yr average")
st.write("")

# ---- tabs (one view at a time) --------------------------------------------
tab_days, tab_rain, tab_now, tab_fc, tab_map, tab_data = st.tabs(
    ["🔵  Rainy days", "🟢  Cumulative rainfall", "☀️  This year (so far)",
     "🔮  Forecast", "🗺️  Map", "📋  Data"])

with tab_days:
    st.markdown(f"##### Rainy days per week &nbsp;·&nbsp; days with rainfall &gt; {rainy_mm:g} mm")
    st.caption(f"May–Dec, {n_years} years. Rows: weeks May 1 → Dec 31 · Columns: each year and "
               f"the {n_years}-year average (right of the dotted line). Darker = more rainy days.")
    st.plotly_chart(styled_heatmap(wby_b, "rainy_days", DAYS_RAMP, "days",
                                   lambda v: f"{v:.0f}", zmax=7, sm=SEASON_SM, em=SEASON_EM),
                    use_container_width=True, config={"displayModeBar": False})

with tab_now:
    cur = weekly_current(src["key"], rainy_mm)
    if cur is None:
        st.info("Current-year (May–present) data isn't built for this source yet. "
                "Run `python build_dataset.py` to add it.")
    else:
        cb = cur[cur["block"] == block].sort_values("week")
        dcur = load_daily(src["key"], "_current")
        dcb = dcur[dcur["block"] == block]
        cy = int(pd.to_datetime(dcb["date"]).dt.year.max())
        last = pd.to_datetime(dcb["date"]).max()
        st.markdown(f"##### This year: {cy} so far — actual rainfall (May onward)")
        st.caption(f"This season so far, {src_label} · May 1 through {last:%d %b %Y}. "
                   "Single year (not an average) — the rain that has actually fallen.")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Rainy days so far", f"{cb['rainy_days'].sum():.0f}",
                   help=f"Days >{rainy_mm:g}mm, May 1 → latest available")
        cc2.metric("Rainfall so far", f"{cb['total_mm'].sum():.0f} mm")
        cc3.metric("Wettest week",
                   metrics.week_label(int(cb.loc[cb['rainy_days'].idxmax(), 'week']),
                                      start_month=5, end_month=12) if len(cb) else "–")
        st.plotly_chart(
            year_bars(cb, "rainy_days", "#2b6cb0", "days", lambda v: f"{v:.0f}",
                      f"Rainy days per week (&gt; {rainy_mm:g} mm)"),
            use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(
            year_bars(cb, "total_mm", "#128577", "mm", lambda v: f"{v:.0f}",
                      "Cumulative rainfall per week"),
            use_container_width=True, config={"displayModeBar": False})

with tab_fc:
    st.markdown("##### Rainfall forecast &nbsp;·&nbsp; WeatherNext 2 (Google DeepMind)")
    cen = blocks_geo[blocks_geo["block"] == block].geometry.centroid
    lat, lon = float(cen.y.iloc[0]), float(cen.x.iloc[0])
    fb = fetch_forecast(lat, lon, rainy_mm)
    if fb is not None:
        fb = fb[fb["rain_mm"].notna()].reset_index(drop=True)
    if fb is None or fb.empty:
        st.warning("Couldn't reach the WeatherNext forecast service just now. "
                   "It's a free live API (Open-Meteo) — try again in a moment.")
    else:
        horizon = len(fb)
        st.caption(f"WeatherNext 2 ensemble ({int(fb['n_members'].iloc[0])} members) via Open-Meteo · "
                   f"next {horizon} days from {fb['date'].min():%d %b} · block centroid "
                   f"{lat:.2f}, {lon:.2f}. Live forecast — updates through the day.")
        rainy_ahead = int((fb["rain_mm"] > rainy_mm).sum())
        g1, g2, g3 = st.columns(3)
        g1.metric("Forecast rainfall", f"{fb['rain_mm'].sum():.0f} mm",
                  help=f"Ensemble-mean total over the next {horizon} days")
        g2.metric("Rainy days ahead", f"{rainy_ahead}",
                  help=f"Days with mean rainfall >{rainy_mm:g}mm")
        g3.metric("Wettest day", f"{pd.to_datetime(fb.loc[fb['rain_mm'].idxmax(), 'date']):%d %b}")

        labels = [d.strftime("%d %b") for d in fb["date"].dt.date]
        vals = fb["rain_mm"].round(1).tolist()
        prob = fb["prob_rain"].round(0).tolist()
        # asymmetric p10–p90 uncertainty whiskers around the ensemble mean
        up = (fb["p90"] - fb["rain_mm"]).clip(lower=0).round(1).tolist()
        dn = (fb["rain_mm"] - fb["p10"]).clip(lower=0).round(1).tolist()
        custom = list(zip(prob, fb["p10"].round(1), fb["p90"].round(1)))
        fig = go.Figure(go.Bar(
            x=labels, y=vals, marker_color="#7c3aed", marker_line_width=0,
            error_y=dict(type="data", symmetric=False, array=up, arrayminus=dn,
                         color="#94a3b8", thickness=1.3, width=3),
            text=[f"{v:.0f}" for v in vals], textposition="outside",
            textfont=dict(color=INK, size=11),
            customdata=custom,
            hovertemplate="<b>%{x}</b><br>%{y:.1f} mm mean · %{customdata[0]:.0f}% chance of rain"
                          "<br>range %{customdata[1]:.0f}–%{customdata[2]:.0f} mm (p10–p90)<extra></extra>"))
        fig.update_traces(marker_cornerradius=3)
        fig.update_layout(title="Daily rainfall forecast — ensemble mean ± p10–p90 (mm)",
                          font=PLOT_FONT, height=360, margin=dict(t=52, b=50, l=8, r=8),
                          bargap=0.3, paper_bgcolor="white", plot_bgcolor="white",
                          yaxis=dict(title="mm", showgrid=True, gridcolor="#eef2f7",
                                     zeroline=False, tickfont=dict(color=MUTED)),
                          xaxis=dict(tickangle=-45, showgrid=False,
                                     tickfont=dict(color=INK, size=11)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Bar = 64-member ensemble mean; whiskers = p10–p90 spread (forecast "
                   "confidence); hover for each day's chance of rain. Model output — guidance, not certainty.")

with tab_rain:
    st.markdown("##### Cumulative rainfall per week &nbsp;·&nbsp; total mm")
    st.caption(f"Rows: weeks May 1 → Dec 31 · Columns: each year and the {n_years}-year average "
               "(right of the dotted line). Darker = more rainfall.")
    st.plotly_chart(styled_heatmap(wby_b, "total_mm", RAIN_RAMP, "mm", lambda v: f"{v:.0f}", sm=SEASON_SM, em=SEASON_EM),
                    use_container_width=True, config={"displayModeBar": False})

with tab_map:
    st.markdown("##### Block map")
    mlabel = st.radio("Colour by", list(MAP_METRICS), horizontal=True, label_visibility="collapsed")
    mcol, munit, mramp = MAP_METRICS[mlabel]
    st.caption(f"Colour = {mlabel.lower()} · {src_label}. Click a block to select it.")

    vmin, vmax = float(blocks[mcol].min()), float(blocks[mcol].max())
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    colormap = cm.LinearColormap(mramp, vmin=vmin, vmax=vmax, caption=mlabel)
    sel = blocks[blocks["block"] == block]
    center = [sel.geometry.centroid.y.iloc[0], sel.geometry.centroid.x.iloc[0]]
    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
    folium.GeoJson(
        blocks.__geo_interface__,
        style_function=lambda f: {
            "fillColor": colormap(f["properties"][mcol]) if f["properties"].get(mcol) is not None else "#e2e8f0",
            "color": "#94a3b8", "weight": 0.6, "fillOpacity": 0.8},
        highlight_function=lambda f: {"weight": 2.5, "color": "#0f172a", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(fields=["block", "district", "state", mcol],
                                      aliases=["Block", "District", "State", mlabel], localize=True),
    ).add_to(fmap)
    folium.GeoJson(sel.__geo_interface__,
                   style_function=lambda f: {"color": "#dc2626", "weight": 3, "fillOpacity": 0.0}).add_to(fmap)
    colormap.add_to(fmap)
    out = st_folium(fmap, height=560, use_container_width=True,
                    returned_objects=["last_active_drawing"])
    clicked = (out or {}).get("last_active_drawing")
    if clicked and clicked.get("properties", {}).get("block") in block_names:
        cb = clicked["properties"]["block"]
        if cb != block:
            st.session_state["_pending_block"] = cb
            st.rerun()

with tab_data:
    st.markdown("##### Weekly detail")
    avg_label = f"{n_years}-year average"
    view = st.radio("View", [avg_label, "Year by year"], horizontal=True,
                    label_visibility="collapsed")
    if view == avg_label:
        t = clim_b.copy()
        t["Week"] = t["week"].apply(lambda w: metrics.week_label(int(w), SEASON_SM, SEASON_EM))
        show = t[["Week", "mean_rainy_days", "mean_total_mm", "mean_extreme_days",
                  "mean_dry_days", "mean_longest_dry_run", "n_years"]].round(1).rename(columns={
            "mean_rainy_days": "Rainy days", "mean_total_mm": "Rain (mm)",
            "mean_extreme_days": "Extreme days", "mean_dry_days": "Dry days",
            "mean_longest_dry_run": "Longest dry run", "n_years": "Years"})
    else:
        t = wby_b.copy()
        t["Week"] = t["week"].apply(lambda w: metrics.week_label(int(w), SEASON_SM, SEASON_EM))
        show = t[["year", "Week", "rainy_days", "total_mm", "extreme_days",
                  "dry_days", "longest_dry_run", "n_days"]].round(1).rename(columns={
            "year": "Year", "rainy_days": "Rainy days", "total_mm": "Rain (mm)",
            "extreme_days": "Extreme days", "dry_days": "Dry days",
            "longest_dry_run": "Longest dry run", "n_days": "Days"})
    st.dataframe(show, use_container_width=True, hide_index=True)

    dl = wby_b[["year", "week", "rainy_days", "total_mm", "extreme_days",
                "dry_days", "longest_dry_run", "n_days"]].copy()
    dl.insert(2, "week_range", dl["week"].apply(lambda w: metrics.week_label(int(w), SEASON_SM, SEASON_EM)))
    st.download_button(f"⬇️  Download {block} · {src['key'].upper()} weekly table (CSV)",
                       dl.to_csv(index=False).encode(),
                       file_name=f"rainfall_{src['key']}_{block.replace(' ', '_')}.csv",
                       mime="text/csv")
