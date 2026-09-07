"""Access IMD (India Meteorological Department) gridded daily rainfall.

Source: IMD Pune 0.25deg x 0.25deg gauge-based daily rainfall (Pai et al. 2014),
downloaded directly from IMD via the ``imdlib`` package. This is the gauge-based
reference product for India -- coarser than GSMaP (~28 km vs ~11 km) but ground-
truth rather than satellite estimate.

Grid: lat 6.5-38.5N (129), lon 66.5-100.0E (135). Value = daily total rainfall in
mm (NO x24 conversion, unlike GSMaP). Nodata (ocean / outside India) = -999.
"""
from __future__ import annotations

import os
import socket
import time

import imdlib
import numpy as np
import pandas as pd

socket.setdefaulttimeout(60)   # bound hangs against the flaky IMD real-time server

FILL_BELOW = -900.0   # -999 nodata sentinel
SEASON_MONTHS = [9, 10, 11, 12]


def season_window(year: int, bbox: tuple[float, float, float, float], cache_dir: str,
                  months: list[int] | None = None):
    """IMD rainfall (mm/day) over ``bbox`` for one year, restricted to ``months``.

    Returns an xarray DataArray (time, lat, lon) with nodata as NaN. Uses the
    local cache if present (``open_data``), otherwise downloads (``get_data``).
    """
    if months is None:
        months = SEASON_MONTHS
    try:
        d = imdlib.open_data("rain", year, year, "yearwise", file_dir=cache_dir)
    except Exception:
        d = imdlib.get_data("rain", year, year, fn_format="yearwise", file_dir=cache_dir)
    da = d.get_xarray()["rain"]
    da = da.where(da > FILL_BELOW)
    minx, miny, maxx, maxy = bbox
    da = da.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))
    da = da.sel(time=da["time"].dt.month.isin(months))
    return da


def _rt_cache_path(cache_dir: str, day: str, bbox: tuple) -> str:
    tag = "_".join(f"{v:.2f}" for v in bbox)
    return os.path.join(cache_dir, f"imd_rt_{day}_{tag}.npz")


def realtime_day(day: str, bbox: tuple[float, float, float, float],
                 cache_dir: str, retries: int = 3):
    """One day of IMD real-time rainfall (mm/day) over ``bbox``: (data2d, lats, lons).

    imdlib's yearly reader cannot open the incomplete current year, so the
    current season is fetched day-by-day from the real-time API. Each day's
    small window is cached (.npz) so re-runs are instant and resumable, and
    transient timeouts against the flaky IMD server are retried.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cp = _rt_cache_path(cache_dir, day, bbox)
    if os.path.exists(cp):
        z = np.load(cp)
        return z["data"], z["lats"], z["lons"]
    minx, miny, maxx, maxy = bbox
    last = None
    for _ in range(retries):
        try:
            d = imdlib.get_real_data("rain", day, day, file_dir=cache_dir)
            da = d.get_xarray()["rain"]
            da = da.where(da > FILL_BELOW).sel(lat=slice(miny, maxy), lon=slice(minx, maxx))
            arr = np.asarray(da.values[0], dtype="float64")
            lats = np.asarray(da["lat"].values, dtype="float64")
            lons = np.asarray(da["lon"].values, dtype="float64")
            np.savez_compressed(cp, data=arr, lats=lats, lons=lons)
            return arr, lats, lons
        except Exception as e:  # noqa: BLE001 -- server is flaky; retry then skip
            last = e
            time.sleep(2)
    raise last


def window_axes(da):
    """(lats, lons) 1-D arrays for a season-window DataArray."""
    return np.asarray(da["lat"].values, dtype="float64"), np.asarray(da["lon"].values, dtype="float64")


def season_times(da) -> pd.DatetimeIndex:
    return pd.to_datetime(da["time"].values)
