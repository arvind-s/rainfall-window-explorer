# 🌧️ Rainfall Window Explorer

A Streamlit app for field staff to plan **plantation timing and field travel**
from satellite/gauge rainfall, analysing the continuous **May–Dec** season over the
**last 10 years (2016–2025)** for each project block. Two rainfall sources,
switchable at the top of the app:

- **🛰️ JAXA GSMaP** — satellite estimate, gauge-calibrated, ~11 km
- **🌧️ IMD** — India Meteorological Department rain-gauge grid, ~28 km (ground reference)

Built for the South India Phase-3 agroforestry blocks, but works with any block
polygon shapefile.

## What it shows (per block)

Pure data — no scoring or "suitability" judgement. A **data-source switch** at the
top toggles every view between JAXA GSMaP and IMD, and a **user-defined rainy-day
threshold** (sidebar, default 1 mm) recomputes rainy/dry days live.

- **Rainy days per week** heatmap — days above the chosen threshold (May–Dec, 10 yrs)
- **Cumulative rainfall per week** heatmap — total mm (May–Dec, 10 yrs)
- **This year** — the *current year's* actual weekly rainfall so far (May onward)
  (single year, not an average) for the pre-planting monsoon months
- **Forecast** (🔮) — WeatherNext 2 (Google DeepMind) 16-day daily rainfall forecast,
  fetched live per block (ensemble mean, p10–p90 spread, and chance of rain)
- A **map** of all blocks coloured by season rainy days / rainfall
- A weekly detail **table** (also extreme days > 5 mm, dry days, longest dry run)
  and a **CSV download**

Rows = weeks (7-day bins), columns = each year + a multi-year average. Field staff read
the patterns directly to plan plantation and travel.

## Data

**JAXA GSMaP** — satellite:
- *Gauge-calibrated* daily product, v6 standard, pulled **directly from the JAXA
  FTP** (`hokusai.eorc.jaxa.jp`) — no Earth Engine. 0.1° (~11 km).
- The raw grid stores daily-mean rain *rate* in mm/hr; the pipeline converts to
  **mm/day** (× 24). Negative values (−999.9) are treated as missing.

**IMD** — rain-gauge grid:
- IMD Pune 0.25° (~28 km) gauge-based daily rainfall (Pai et al. 2014), pulled via
  the **`imdlib`** package. Value is **mm/day** directly. Nodata (−999, over sea) →
  NaN; coastal blocks whose cell falls on water are snapped to the nearest land cell.

Each block's daily value is the **mean of the grid cells inside the polygon**
(nearest cell for blocks smaller than one grid cell).

## Setup

```bash
pip install -r requirements.txt
```

## 1. Build the dataset (run once; yearly to refresh)

```bash
python build_dataset.py --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

Builds **both sources** by default (`--sources gsmap imd` to pick). GSMaP is
FTP-bound (~1 hr first run); IMD downloads ten small year-files. Writes per-source
tables to `data/out/`:

| file | contents |
|------|----------|
| `daily_<src>.parquet` | block × date daily rainfall, May–Dec (mm) |
| `daily_current_<src>.parquet` | block × date daily rainfall, **current year, May onward** |
| `weekly_by_year_<src>.parquet` | block × year × week metrics |
| `climatology_<src>.parquet` | block × week multi-year averages |
| `blocks.geojson` | block boundaries (shared) |

The app recomputes weekly rainy/dry days from the `daily_*` files at runtime so the
**rainy-day threshold is user-adjustable**; the `weekly_*`/`climatology_*` files are
convenience caches. Re-run the build to refresh the **current-year** data
(it changes as the season progresses); `--no-current` skips it.

where `<src>` is `gsmap` or `imd`. GSMaP windows cache under `data/raw_cache/`
(~few MB) and IMD year-files under `data/imd_cache/`, so re-runs skip
already-downloaded data and the multi-GB global GSMaP files are never stored.

Quick validation run (10 days of one year, one source):

```bash
python build_dataset.py --years 2024 --sources imd --max-days 10 --out data/out_test
```

## 2. Run the app

```bash
streamlit run app.py
```

Pick **State → District → Block** (or click a block on the map) and toggle the
**rainfall source** (JAXA GSMaP / IMD) at the top. The URL carries
`?block=<name>` so a specific block view can be shared.

## 3. Deploy to Streamlit Community Cloud

The app reads the **baked `data/out/` files** (committed to this repo, ~2.6 MB) —
it does **not** download rainfall at runtime, so no FTP/IMD access is needed on the
server. To deploy:

1. Push this directory to a GitHub repo (see below).
2. On [share.streamlit.io](https://share.streamlit.io) → **New app**, pick the repo,
   set **Main file path** to `app.py`, and deploy.

Streamlit Cloud installs `requirements.txt` (runtime only). To refresh the data
later, rebuild locally with `requirements-build.txt` and push the updated
`data/out/` files.

## 4. WeatherNext forecast (live, no setup)

The **🔮 Forecast** tab shows a forward-looking daily rainfall forecast from
**WeatherNext 2** (Google DeepMind) — distinct from the historical/current
*observations* above. It is fetched **live** from the free **Open-Meteo Ensemble
API** (`google_weathernext2_ensemble`, 64 members, 0.25°) for the selected block's
centroid — no API key, no Earth Engine, no baked file, and nothing to configure on
Streamlit Cloud (the server just makes an outbound HTTPS call, cached ~1 h).

Per day it shows the **ensemble mean** rainfall (mm), the **p10–p90 spread**
(forecast confidence), and the **chance of rain** (share of the 64 members above the
rainy-day threshold). Horizon ≈ 16 days. See `rainfall/weathernext.py`.

> WeatherNext 2 is also available via Google **BigQuery** and **Weather Lab**, but
> those need GCP auth/billing (BigQuery) or are a web viewer (Weather Lab); Open-Meteo
> serves the same model with the best fit for a credential-free deployed app.

## 5. Static HTML export (one file per block)

The **same dashboard** can be exported as standalone HTML — one `<id>.html` per
polygon — for sharing offline or hosting as a static site (no Streamlit server):

```bash
python generate_html.py                         # all blocks, both sources, 1 mm
python generate_html.py --threshold 2.5 --sources gsmap
python generate_html.py --id-col block_id --no-forecast --limit 5   # testing
```

Writes to `html_out/`: one page per block (GSMaP/IMD source tabs, the two 10-year
heatmaps, the current-year bars, a live-baked WeatherNext forecast, a location map,
and the weekly table) plus an `index.html` linking them all. `--id-col` picks the
shapefile/geojson column used for the filename (default `block`). The app and this
exporter share the plotting code in `rainfall/figures.py`, so both stay in sync.

Serve locally with `python -m http.server` inside `html_out/`, or publish the folder
(e.g. GitHub Pages / any static host). The threshold is fixed at generation time
(static pages have no live slider); re-run to change it.

## Weeks

Fixed 7-day bins from May 1 (Week 1 = May 1–7, …). The May–Dec season is 245 days
= exactly 35 full weeks (Week 35 = Dec 25–31).

## Project layout

```
rainfall_app/
├── rainfall/           # library
│   ├── gsmap.py        #   JAXA FTP access + binary grid parsing
│   ├── imd.py          #   IMD gridded rainfall via imdlib
│   ├── weathernext.py  #   WeatherNext 2 forecast via Open-Meteo (live)
│   ├── figures.py      #   Plotly figures shared by app + HTML export
│   ├── metrics.py      #   weekly binning, thresholds, dry spells
│   └── blocks.py       #   shapefile loading + polygon→pixel mapping
├── build_dataset.py       # offline data-prep pipeline (both sources)
├── generate_html.py       # static HTML export — one page per block
├── app.py                 # Streamlit app (entry point)
├── tests/                 # unit tests (pytest)
├── data/out/              # baked dataset (committed -- read at runtime)
├── .streamlit/config.toml # theme
├── requirements.txt       # runtime deps (Streamlit Cloud)
└── requirements-build.txt # + imdlib, for rebuilding data locally
```

## Tests

```bash
python -m pytest tests/ -q
```
