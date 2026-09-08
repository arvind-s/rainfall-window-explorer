#!/usr/bin/env python3
"""Build the baked rainfall datasets the Streamlit app reads.

Two rainfall sources, each fetched directly (no Earth Engine):
  * gsmap -- JAXA GSMaP gauge-calibrated daily, 0.1deg, via JAXA FTP
  * imd   -- IMD Pune gauge-based daily, 0.25deg, via the imdlib package

For the Sep-Dec season of each requested year, computes block-mean daily rainfall
for every block in the shapefile, then writes per-source tables to data/out/:

    daily_<src>.parquet          block x date daily rainfall (mm)
    weekly_by_year_<src>.parquet block x year x week metrics
    climatology_<src>.parquet    block x week multi-year averages (descriptive)
    blocks.geojson               block boundaries (shared, geometry + names)

Usage:
    python build_dataset.py                          # both sources, 2016-2025 (10 yrs)
    python build_dataset.py --sources imd
    python build_dataset.py --sources gsmap --max-days 10   # quick validation
"""
from __future__ import annotations

import argparse
import ftplib
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from rainfall import gsmap, imd, metrics, blocks as blk

DEFAULT_SHP = ("/Users/mipl/Documents/Agroforestry/South India/Phase 3/"
               "Stratification/Input_boundary/Blocks/SI_blocks_3rd_phase_updated_6.shp")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "raw_cache")
IMD_CACHE = os.path.join(HERE, "data", "imd_cache")
IMD_RT_CACHE = os.path.join(HERE, "data", "imd_rt_cache")
OUT = os.path.join(HERE, "data", "out")


def _block_means(grid, pixel_map) -> dict:
    """Mean over each block's pixels for one 2-D (lat, lon) rainfall grid."""
    out = {}
    for name, px in pixel_map.items():
        vals = np.array([grid[r, c] for r, c in px], dtype="float64")
        out[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    return out


def _prefetch_gsmap_parallel(dates, bbox, workers=8):
    """Warm the GSMaP window cache in parallel (each worker its own FTP)."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [d for d in dates
            if not os.path.exists(gsmap._cache_path(CACHE, d, bbox))]
    if not todo:
        return
    print(f"  prefetching {len(todo)} GSMaP days with {workers} workers...", flush=True)
    done = [0]

    def work(chunk):
        ftp = ftplib.FTP(gsmap.FTP_HOST, timeout=120)
        ftp.login(gsmap.FTP_USER, gsmap.FTP_PASS)
        try:
            for d in chunk:
                try:
                    gsmap.fetch_window(d, bbox, CACHE, ftp=ftp)
                except FileNotFoundError:
                    pass
                except (ftplib.error_perm, EOFError, OSError):
                    try:
                        ftp.quit()
                    except Exception:
                        pass
                    ftp = ftplib.FTP(gsmap.FTP_HOST, timeout=120)
                    ftp.login(gsmap.FTP_USER, gsmap.FTP_PASS)
                done[0] += 1
                if done[0] % 100 == 0:
                    print(f"    prefetched {done[0]}/{len(todo)}", flush=True)
        finally:
            try:
                ftp.quit()
            except Exception:
                pass

    chunks = [todo[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, chunks))


def build_daily_gsmap(years, blocks, bbox, start_month=9, end_month=12,
                      max_days=None, max_date=None, workers=1) -> pd.DataFrame:
    if workers > 1:
        all_dates = []
        for year in years:
            ds = metrics.season_dates(year, start_month, end_month)
            if max_date:
                ds = [d for d in ds if d <= max_date]
            all_dates += ds
        _prefetch_gsmap_parallel(all_dates, bbox, workers=workers)
    ftp = ftplib.FTP(gsmap.FTP_HOST, timeout=120)
    ftp.login(gsmap.FTP_USER, gsmap.FTP_PASS)
    pixel_map, records = None, []
    try:
        for year in years:
            dates = metrics.season_dates(year, start_month, end_month)
            if max_date:
                dates = [d for d in dates if d <= max_date]
            if max_days:
                dates = dates[:max_days]
            t0 = time.time()
            for i, d in enumerate(dates):
                try:
                    win = gsmap.fetch_window(d, bbox, CACHE, ftp=ftp)
                except FileNotFoundError as e:
                    print(f"  ! {d} missing on server ({e})")
                    continue
                except (ftplib.error_perm, EOFError, OSError) as e:
                    print(f"  ! {d} conn error ({e.__class__.__name__}); reconnecting")
                    try:
                        ftp.quit()
                    except Exception:
                        pass
                    ftp = ftplib.FTP(gsmap.FTP_HOST, timeout=120)
                    ftp.login(gsmap.FTP_USER, gsmap.FTP_PASS)
                    gsmap._DIR_CACHE.clear()
                    continue
                if pixel_map is None:
                    pixel_map = blk.build_pixel_map(blocks, win["lats"], win["lons"])
                    npx = {k: len(v) for k, v in pixel_map.items()}
                    print(f"  gsmap pixel map: {min(npx.values())}-{max(npx.values())} cells/block")
                for name, m in _block_means(win["data"], pixel_map).items():
                    records.append((name, d, m))
                if (i + 1) % 30 == 0 or i == len(dates) - 1:
                    print(f"  gsmap {year}: {i+1}/{len(dates)} days ({time.time()-t0:.0f}s)", flush=True)
    finally:
        try:
            ftp.quit()
        except Exception:
            pass
    df = pd.DataFrame(records, columns=["block", "date", "rain_mm"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_daily_imd(years, blocks, bbox, start_month=9, end_month=12, max_days=None) -> pd.DataFrame:
    months = list(range(start_month, end_month + 1))
    pixel_map, records = None, []
    for year in years:
        da = imd.season_window(year, bbox, IMD_CACHE, months=months)
        lats, lons = imd.window_axes(da)
        data = da.values  # (time, lat, lon)
        if pixel_map is None:
            # IMD is nodata over sea -> restrict to land cells (valid on any day)
            valid = np.isfinite(data).any(axis=0)
            pixel_map = blk.build_pixel_map(blocks, lats, lons, valid_mask=valid)
            npx = {k: len(v) for k, v in pixel_map.items()}
            print(f"  imd pixel map: {min(npx.values())}-{max(npx.values())} cells/block "
                  f"({int(valid.sum())} land cells)")
        times = imd.season_times(da)
        if max_days:
            times, data = times[:max_days], data[:max_days]
        for ti, t in enumerate(times):
            for name, m in _block_means(data[ti], pixel_map).items():
                records.append((name, t, m))
        print(f"  imd {year}: {len(times)} days", flush=True)
    df = pd.DataFrame(records, columns=["block", "date", "rain_mm"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_daily_imd_realtime(start_date, end_date, blocks, bbox) -> pd.DataFrame:
    """Current (incomplete) year IMD via the real-time API, day-by-day.

    Robust to the flaky IMD server: each day is fetched with retries and
    cached; a day that persistently fails is skipped (the season just ends at
    the last available day).
    """
    days = pd.date_range(start_date, end_date, freq="D")
    pixel_map, records, skipped = None, [], 0
    t0 = time.time()
    for i, ts in enumerate(days):
        day = ts.strftime("%Y-%m-%d")
        try:
            arr, lats, lons = imd.realtime_day(day, bbox, IMD_RT_CACHE)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print(f"  ! imd {day} skipped ({e.__class__.__name__})", flush=True)
            continue
        if pixel_map is None:
            # a single day's valid cells suffice as the land mask
            pixel_map = blk.build_pixel_map(blocks, lats, lons,
                                            valid_mask=np.isfinite(arr))
            npx = {k: len(v) for k, v in pixel_map.items()}
            print(f"  imd-rt pixel map: {min(npx.values())}-{max(npx.values())} cells/block")
        for name, m in _block_means(arr, pixel_map).items():
            records.append((name, ts, m))
        if (i + 1) % 20 == 0 or i == len(days) - 1:
            print(f"  imd-rt: {i+1}/{len(days)} days ({time.time()-t0:.0f}s, {skipped} skipped)", flush=True)
    df = pd.DataFrame(records, columns=["block", "date", "rain_mm"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _summarize(daily, tag):
    print(f"  {tag}: {len(daily)} rows | NaN {daily['rain_mm'].isna().sum()} "
          f"| mm/day {daily['rain_mm'].min():.1f}..{daily['rain_mm'].max():.1f} "
          f"| dates {daily['date'].min():%Y-%m-%d}..{daily['date'].max():%Y-%m-%d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+",
                    default=list(range(2016, 2026)))   # 10 years: 2016–2025
    ap.add_argument("--sources", nargs="+", choices=["gsmap", "imd"], default=["gsmap", "imd"])
    ap.add_argument("--shp", default=DEFAULT_SHP)
    ap.add_argument("--block-col", default=blk.BLOCK_COL, help="polygon name/id column")
    ap.add_argument("--district-col", default=blk.DISTRICT_COL, help="optional district column")
    ap.add_argument("--state-col", default=blk.STATE_COL, help="optional state column")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--current-year", type=int, default=date.today().year,
                    help="year for the May–Aug current-season section")
    ap.add_argument("--current-lag", type=int, default=0,
                    help="days back from today to stop (0 = through today; unavailable "
                         "recent days are skipped automatically)")
    ap.add_argument("--no-current", action="store_true", help="skip the May–Aug current build")
    ap.add_argument("--only-current", action="store_true",
                    help="build ONLY the May–Aug current data (skip the historical rebuild)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel FTP connections for the GSMaP download")
    args = ap.parse_args()

    # single continuous historical season: May 1 – Dec 31
    SEASON_START, SEASON_END = 5, 12

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(IMD_CACHE, exist_ok=True)
    os.makedirs(IMD_RT_CACHE, exist_ok=True)
    blocks = blk.load_blocks(args.shp, args.block_col, args.district_col, args.state_col)
    bbox = blk.season_bbox(blocks)
    print(f"Loaded {len(blocks)} blocks; window bbox = {tuple(round(v,2) for v in bbox)}")
    skip_hist = args.only_current
    if not skip_hist:
        blocks.to_file(os.path.join(args.out, "blocks.geojson"), driver="GeoJSON")

    # ---- historical (10 yrs): continuous May–Dec ----
    for src in ([] if skip_hist else args.sources):
        print(f"=== {src}: May–Dec {args.years[0]}–{args.years[-1]} ===")
        if src == "gsmap":
            daily = build_daily_gsmap(args.years, blocks, bbox, SEASON_START, SEASON_END,
                                      max_days=args.max_days, workers=args.workers)
        else:
            daily = build_daily_imd(args.years, blocks, bbox, SEASON_START, SEASON_END,
                                    max_days=args.max_days)
        _summarize(daily, src)
        wby = metrics.weekly_by_year(daily, start_month=SEASON_START)
        clim = metrics.climatology(wby)
        daily.to_parquet(os.path.join(args.out, f"daily_{src}.parquet"), index=False)
        wby.to_parquet(os.path.join(args.out, f"weekly_by_year_{src}.parquet"), index=False)
        clim.to_parquet(os.path.join(args.out, f"climatology_{src}.parquet"), index=False)

    # ---- current-year May–present actual ----
    if not args.no_current:
        cy = args.current_year
        end = date.today() - timedelta(days=args.current_lag)
        end = min(end, date(cy, 12, 31))
        start = date(cy, 5, 1)
        print(f"=== current-year May–present {cy}: {start} .. {end} ===")
        for src in args.sources:
            if src == "gsmap":
                daily_c = build_daily_gsmap([cy], blocks, bbox, 5, 12, max_date=end,
                                            workers=args.workers)
            else:
                daily_c = build_daily_imd_realtime(start.isoformat(), end.isoformat(), blocks, bbox)
            _summarize(daily_c, f"{src} current")
            daily_c.to_parquet(os.path.join(args.out, f"daily_current_{src}.parquet"), index=False)

    # (The WeatherNext forecast is fetched live in the app via Open-Meteo — not baked.)
    print("Done ->", args.out)


if __name__ == "__main__":
    sys.exit(main())
