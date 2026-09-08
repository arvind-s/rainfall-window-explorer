"""Plotly figure builders shared by the Streamlit app and the static HTML export.

Pure functions: take data, return a go.Figure. No Streamlit dependency.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import metrics

INK, MUTED, LINE = "#1f2937", "#64748b", "#cbd5e1"
DAYS_RAMP = ["#eef4fb", "#cfe1f2", "#9dc3e3", "#5a9bd4", "#2b6cb0", "#1a4f8a"]   # blues
RAIN_RAMP = ["#eff7f6", "#cdeae6", "#93d3cb", "#4db6ac", "#128577", "#0b5c52"]   # teals
PLOT_FONT = dict(family="Source Sans Pro, Segoe UI, sans-serif", color=INK, size=13)


def heatmap(wby_block, value_col, ramp, unit, val_fmt, zmax=None, sm=9, em=12):
    """Week (rows) x year (cols) heatmap for a season (months sm..em)."""
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

    anns = []  # anchor labels by integer index (numeric-looking years break the axis)
    for i, wk in enumerate(ylabels):
        for j in range(len(xcols)):
            v = z[i][j]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            frac = v / zt if zt else 0
            anns.append(dict(x=j, y=i, xref="x", yref="y", text=val_fmt(v), showarrow=False,
                             font=dict(color="white" if frac > 0.6 else INK, size=11)))

    fig.update_layout(
        annotations=anns, font=PLOT_FONT, height=26 * len(ylabels) + 120,
        margin=dict(t=52, b=10, l=8, r=8), paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(autorange="reversed", showgrid=False, ticks="", tickfont=dict(color=MUTED)),
        xaxis=dict(type="category", side="top", showgrid=False, ticks="",
                   tickfont=dict(color=INK, size=12)))
    fig.add_vline(x=len(years) - 0.5, line_width=1.5, line_color="#94a3b8", line_dash="dot")
    return fig


def year_bars(wby_block, value_col, color, unit, val_fmt, title, sm=5, em=12):
    """Single-year weekly bar chart (current-year 'this season so far')."""
    g = wby_block.sort_values("week")
    labels = [metrics.week_label(int(w), start_month=sm, end_month=em) for w in g["week"]]
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


def forecast_bars(fb):
    """WeatherNext forecast: ensemble-mean bars + p10–p90 whiskers + chance of rain."""
    labels = [d.strftime("%d %b") for d in fb["date"].dt.date]
    vals = fb["rain_mm"].round(1).tolist()
    prob = fb["prob_rain"].round(0).tolist()
    up = (fb["p90"] - fb["rain_mm"]).clip(lower=0).round(1).tolist()
    dn = (fb["rain_mm"] - fb["p10"]).clip(lower=0).round(1).tolist()
    custom = list(zip(prob, fb["p10"].round(1), fb["p90"].round(1)))
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color="#7c3aed", marker_line_width=0,
        error_y=dict(type="data", symmetric=False, array=up, arrayminus=dn,
                     color="#94a3b8", thickness=1.3, width=3),
        text=[f"{v:.0f}" for v in vals], textposition="outside",
        textfont=dict(color=INK, size=11), customdata=custom,
        hovertemplate="<b>%{x}</b><br>%{y:.1f} mm mean · %{customdata[0]:.0f}% chance of rain"
                      "<br>range %{customdata[1]:.0f}–%{customdata[2]:.0f} mm (p10–p90)<extra></extra>"))
    fig.update_traces(marker_cornerradius=3)
    fig.update_layout(title="Daily rainfall forecast — ensemble mean ± p10–p90 (mm)",
                      font=PLOT_FONT, height=360, margin=dict(t=52, b=50, l=8, r=8),
                      bargap=0.3, paper_bgcolor="white", plot_bgcolor="white",
                      yaxis=dict(title="mm", showgrid=True, gridcolor="#eef2f7",
                                 zeroline=False, tickfont=dict(color=MUTED)),
                      xaxis=dict(tickangle=-45, showgrid=False, tickfont=dict(color=INK, size=11)))
    return fig
