"""Access and parse JAXA GSMaP daily gauge-calibrated rainfall.

Data source (direct from JAXA, no Earth Engine):
    ftp://hokusai.eorc.jaxa.jp/standard/v6/daily_G/00Z-23Z/YYYYMM/
    gsmap_gauge.YYYYMMDD.0.1d.daily.00Z-23Z.v6.6133.0.dat.gz

File format (verified against a real file, 2024-09-15):
    * 3600 x 1200 grid, 0.1 deg global, little-endian float32 (17,280,000 bytes)
    * row-major, longitude varies fastest; reshape to (nlat=1200, nlon=3600)
    * row 0 = northernmost (59.95 N), going south;  col 0 = 0.05 E, going east (0..360)
    * value is DAILY-MEAN RAIN RATE in mm/hr; daily total mm = value * 24
    * missing / no-retrieval = negative (-999.9)
"""
from __future__ import annotations

import ftplib
import gzip
import io
import os
from datetime import date

import numpy as np

# ---- grid constants -------------------------------------------------------
NLON = 3600
NLAT = 1200
RES = 0.1
LAT_ORIGIN = 59.95   # centre of first (northern-most) row
LON_ORIGIN = 0.05    # centre of first column (degrees east, 0..360)
MISSING_BELOW = -0.001  # anything < 0 is missing
MM_PER_HR_TO_MM_PER_DAY = 24.0

# ---- FTP constants --------------------------------------------------------
FTP_HOST = "hokusai.eorc.jaxa.jp"
FTP_USER = "rainmap"
FTP_PASS = "Niskur+1404"  # JAXA public GSMaP guest account
REMOTE_DIR = "/standard/v6/daily_G/00Z-23Z"


_DIR_CACHE: dict[str, list[str]] = {}


def remote_dir(d: date) -> str:
    return f"{REMOTE_DIR}/{d:%Y%m}"


def resolve_remote_file(d: date, ftp: ftplib.FTP) -> str:
    """Full FTP path of the daily gauge file for ``d``.

    The version tag in the filename is NOT constant across years (e.g. v6.4133
    for the 2021-2022 reanalysis vs v6.6133 for recent data), so resolve it by
    listing the month directory and matching the date prefix. Listings are
    cached per month.
    """
    month = d.strftime("%Y%m")
    if month not in _DIR_CACHE:
        try:
            _DIR_CACHE[month] = ftp.nlst(remote_dir(d))
        except ftplib.error_perm:
            _DIR_CACHE[month] = []
    prefix = f"gsmap_gauge.{d:%Y%m%d}."
    for name in _DIR_CACHE[month]:
        base = name.rsplit("/", 1)[-1]
        if base.startswith(prefix) and base.endswith(".dat.gz"):
            return f"{remote_dir(d)}/{base}"
    raise FileNotFoundError(f"no GSMaP gauge file for {d:%Y-%m-%d}")


# ---- grid geometry --------------------------------------------------------
def lats() -> np.ndarray:
    """Latitude of each grid row (north -> south), length NLAT."""
    return LAT_ORIGIN - RES * np.arange(NLAT)


def lons() -> np.ndarray:
    """Longitude of each grid column (0..360, west -> east), length NLON."""
    return LON_ORIGIN + RES * np.arange(NLON)


def row_of_lat(lat: float) -> int:
    return int(round((LAT_ORIGIN - lat) / RES))


def col_of_lon(lon: float) -> int:
    lon360 = lon % 360.0
    return int(round((lon360 - LON_ORIGIN) / RES))


def window_slices(bbox: tuple[float, float, float, float]) -> tuple[slice, slice]:
    """Return (row_slice, col_slice) covering a lon/lat bbox (minx,miny,maxx,maxy)."""
    minx, miny, maxx, maxy = bbox
    r0, r1 = row_of_lat(maxy), row_of_lat(miny)   # north row has smaller index
    c0, c1 = col_of_lon(minx), col_of_lon(maxx)
    r0, r1 = max(0, r0 - 1), min(NLAT - 1, r1 + 1)
    c0, c1 = max(0, c0 - 1), min(NLON - 1, c1 + 1)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


# ---- parsing --------------------------------------------------------------
def parse_grid(raw: bytes) -> np.ndarray:
    """Decode raw (already-decompressed) bytes to a (NLAT, NLON) mm/day array.

    Negative (missing) values become NaN. Returned units are mm/day.
    """
    if len(raw) != NLON * NLAT * 4:
        raise ValueError(
            f"unexpected size {len(raw)} bytes, expected {NLON * NLAT * 4}"
        )
    arr = np.frombuffer(raw, dtype="<f4").astype("float64").reshape(NLAT, NLON)
    arr = np.where(arr < MISSING_BELOW, np.nan, arr)
    return arr * MM_PER_HR_TO_MM_PER_DAY


def parse_gz(gz_bytes: bytes) -> np.ndarray:
    """Decompress a .dat.gz payload and parse to a mm/day grid."""
    return parse_grid(gzip.decompress(gz_bytes))


# ---- download + cache -----------------------------------------------------
def _cache_path(cache_dir: str, d: date, bbox: tuple) -> str:
    tag = "_".join(f"{v:.2f}" for v in bbox)
    return os.path.join(cache_dir, f"gsmap_gauge_{d:%Y%m%d}_{tag}.npz")


def fetch_window(
    d: date,
    bbox: tuple[float, float, float, float],
    cache_dir: str,
    ftp: ftplib.FTP | None = None,
) -> dict:
    """Return the mm/day window over ``bbox`` for date ``d``.

    Result dict: {"data": 2-D mm/day array, "lats": 1-D, "lons": 1-D}.
    Only the small window is cached to disk (as .npz), so re-runs are cheap
    and the multi-GB global files are never stored.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cp = _cache_path(cache_dir, d, bbox)
    if os.path.exists(cp):
        z = np.load(cp)
        return {"data": z["data"], "lats": z["lats"], "lons": z["lons"]}

    own = ftp is None
    if own:
        ftp = ftplib.FTP(FTP_HOST, timeout=120)
        ftp.login(FTP_USER, FTP_PASS)
    try:
        buf = io.BytesIO()
        ftp.retrbinary(f"RETR {resolve_remote_file(d, ftp)}", buf.write)
    finally:
        if own:
            ftp.quit()

    grid = parse_gz(buf.getvalue())
    rs, cs = window_slices(bbox)
    data = grid[rs, cs]
    wl, wo = lats()[rs], lons()[cs]
    np.savez_compressed(cp, data=data, lats=wl, lons=wo)
    return {"data": data, "lats": wl, "lons": wo}
