"""Load the block shapefile and map each block polygon to GSMaP grid pixels."""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from . import gsmap

BLOCK_COL = "Block"
DISTRICT_COL = "District_x"
STATE_COL = "State_x"


def load_blocks(shp_path: str) -> gpd.GeoDataFrame:
    """Read blocks, ensure WGS84, add a stable ``block_id`` and clean names."""
    g = gpd.read_file(shp_path)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    bad = g.geometry.isna() | g.geometry.is_empty
    if bad.any():
        names = ", ".join(g.loc[bad, BLOCK_COL].astype(str))
        print(f"WARNING: dropping {int(bad.sum())} block(s) with no geometry: {names}")
        g = g[~bad]
    g = g.reset_index(drop=True)
    g["block_id"] = g.index.astype(int)
    g["block"] = g[BLOCK_COL].astype(str).str.strip()
    g["district"] = g[DISTRICT_COL].astype(str).str.strip()
    g["state"] = g[STATE_COL].astype(str).str.strip()
    # disambiguate any duplicate block names by district
    dup = g["block"].duplicated(keep=False)
    g.loc[dup, "block"] = g.loc[dup, "block"] + " (" + g.loc[dup, "district"] + ")"
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
