"""Load the block shapefile and map each block polygon to GSMaP grid pixels."""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from . import gsmap

BLOCK_COL = "Block"
DISTRICT_COL = "District"
STATE_COL = "State"

# the shapefile mixes full names and codes -> normalise (matched case-insensitively)
STATE_ALIASES = {"AP": "Andhra Pradesh", "KA": "Karnataka", "TN": "Tamil Nadu",
                 "TS": "Telangana", "TG": "Telangana", "OD": "Odisha", "OR": "Odisha",
                 "KL": "Kerala", "MH": "Maharashtra"}
DISTRICT_ALIASES = {
    "ASR": "Alluri Sitharama Raju", "VSK": "Alluri Sitharama Raju",
    "BDR": "Bidar", "BHD": "Bhadradri Kothagudem", "CTT": "Cuttack", "CUT": "Cuttack",
    "DAV": "Davanagere", "DIN": "Dindigul", "HAV": "Haveri", "MYR": "Mayurbhanj",
    "SND": "Sangareddy", "VIL": "Villupuram", "VJP": "Vijayapura", "VKR": "Vikarabad",
    "PRA": "Prakasam", "WGI": "Eluru", "PLK": "Palakkad",
}


def _alias(series, table):
    """Map codes to full names, matching keys case-insensitively; keep others as-is."""
    up = {k.upper(): v for k, v in table.items()}
    return series.astype(str).str.strip().map(lambda v: up.get(v.upper(), v))


def load_blocks(shp_path: str, block_col: str = BLOCK_COL,
                district_col: str = DISTRICT_COL, state_col: str = STATE_COL) -> gpd.GeoDataFrame:
    """Read polygons, ensure WGS84, add a stable ``block_id`` and clean names.

    ``block_col`` names the polygon-name/id field (required). ``district_col`` and
    ``state_col`` are optional grouping/label fields; when absent they are blank.
    """
    g = gpd.read_file(shp_path)
    if block_col not in g.columns:
        raise ValueError(f"block_col '{block_col}' not in shapefile columns {list(g.columns)}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    bad = g.geometry.isna() | g.geometry.is_empty
    if bad.any():
        names = ", ".join(g.loc[bad, block_col].astype(str))
        print(f"WARNING: dropping {int(bad.sum())} block(s) with no geometry: {names}")
        g = g[~bad]
    g = g.reset_index(drop=True)
    g["block_id"] = g.index.astype(int)
    g["block"] = g[block_col].astype(str).str.strip()
    g["district"] = _alias(g[district_col], DISTRICT_ALIASES) if district_col in g.columns else ""
    g["state"] = _alias(g[state_col], STATE_ALIASES) if state_col in g.columns else ""
    # make block names UNIQUE (they key all downstream data). First qualify
    # duplicates with their district; if a name is still duplicated (same name +
    # district), append the unique block_id so nothing collapses.
    dup = g["block"].duplicated(keep=False)
    if dup.any():
        d = g.loc[dup, "district"].astype(str).str.strip()
        g.loc[dup, "block"] = g.loc[dup, "block"] + d.where(d.eq(""), " (" + d + ")")
    dup2 = g["block"].duplicated(keep=False)
    if dup2.any():
        g.loc[dup2, "block"] = g.loc[dup2, "block"] + " #" + g.loc[dup2, "block_id"].astype(str)
    assert not g["block"].duplicated().any(), "block names still not unique"
    return g


def block_pixels(geom, lats: np.ndarray, lons: np.ndarray,
                 valid_mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Grid (row, col) indices whose cell centre falls inside ``geom``.

    When ``valid_mask`` (a 2-D bool array over the same grid) is given, only
    valid cells are used -- this matters for IMD, which is nodata over the sea:
    a coastal block whose cell falls on water is snapped to the nearest valid
    land cell. Falls back to the single nearest (valid) cell to the polygon
    centroid when no cell centre lies inside the block.
    """
    minx, miny, maxx, maxy = geom.bounds
    rmask = np.where((lats >= miny) & (lats <= maxy))[0]
    cmask = np.where((lons >= minx) & (lons <= maxx))[0]
    px = []
    for r in rmask:
        for c in cmask:
            if geom.contains(Point(lons[c], lats[r])) and (valid_mask is None or valid_mask[r, c]):
                px.append((int(r), int(c)))
    if not px:
        cen = geom.centroid
        if valid_mask is None:
            r = int(np.argmin(np.abs(lats - cen.y)))
            c = int(np.argmin(np.abs(lons - cen.x)))
            px = [(r, c)]
        else:
            vr, vc = np.where(valid_mask)
            dist = (lats[vr] - cen.y) ** 2 + (lons[vc] - cen.x) ** 2
            k = int(np.argmin(dist))
            px = [(int(vr[k]), int(vc[k]))]
    return px


def build_pixel_map(blocks: gpd.GeoDataFrame, lats: np.ndarray, lons: np.ndarray,
                    valid_mask: np.ndarray | None = None) -> dict:
    """Map each block name -> list of (row, col) window indices."""
    return {row["block"]: block_pixels(row.geometry, lats, lons, valid_mask)
            for _, row in blocks.iterrows()}


def season_bbox(blocks: gpd.GeoDataFrame, pad: float = 0.3) -> tuple[float, float, float, float]:
    """Padded lon/lat bbox covering all blocks."""
    minx, miny, maxx, maxy = blocks.total_bounds
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)
