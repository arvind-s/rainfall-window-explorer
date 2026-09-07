"""Unit tests for the rainfall analysis library."""
import gzip
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rainfall import gsmap, metrics, blocks as blk


# ---- gsmap grid geometry --------------------------------------------------
def test_grid_dimensions_and_axes():
    assert gsmap.lats().shape == (gsmap.NLAT,)
    assert gsmap.lons().shape == (gsmap.NLON,)
    assert gsmap.lats()[0] == pytest.approx(59.95)      # north first
    assert gsmap.lats()[-1] == pytest.approx(-59.95)    # south last
    assert gsmap.lons()[0] == pytest.approx(0.05)


def test_row_col_lookup_roundtrip():
    # a cell centre near Bangalore (~12.95N, 77.55E)
    r, c = gsmap.row_of_lat(12.95), gsmap.col_of_lon(77.55)
    assert gsmap.lats()[r] == pytest.approx(12.95, abs=0.05)
    assert gsmap.lons()[c] == pytest.approx(77.55, abs=0.05)


def test_window_slices_cover_bbox():
    bbox = (77.0, 12.0, 78.0, 13.0)
    rs, cs = gsmap.window_slices(bbox)
    wl, wo = gsmap.lats()[rs], gsmap.lons()[cs]
    assert wl.min() <= 12.0 and wl.max() >= 13.0
    assert wo.min() <= 77.0 and wo.max() >= 78.0


def test_parse_grid_units_and_missing():
    g = np.zeros((gsmap.NLAT, gsmap.NLON), dtype="<f4")
    g[0, 0] = 2.0           # 2 mm/hr -> 48 mm/day
    g[0, 1] = -999.9        # missing
    arr = gsmap.parse_grid(g.tobytes())
    assert arr[0, 0] == pytest.approx(48.0)      # x24 scaling
    assert np.isnan(arr[0, 1])                   # negatives -> NaN


def test_parse_gz_roundtrip():
    g = np.full((gsmap.NLAT, gsmap.NLON), 0.5, dtype="<f4")
    arr = gsmap.parse_gz(gzip.compress(g.tobytes()))
    assert np.allclose(arr, 12.0)                # 0.5 mm/hr -> 12 mm/day


def test_parse_grid_rejects_bad_size():
    with pytest.raises(ValueError):
        gsmap.parse_grid(b"\x00" * 100)


# ---- season / week binning ------------------------------------------------
def test_season_dates_span_sep_to_dec():
    d = metrics.season_dates(2023)
    assert d[0] == date(2023, 9, 1)
    assert d[-1] == date(2023, 12, 31)
    assert len(d) == 122


def test_week_of_boundaries():
    assert metrics.week_of(date(2023, 9, 1)) == 1
    assert metrics.week_of(date(2023, 9, 7)) == 1
    assert metrics.week_of(date(2023, 9, 8)) == 2
    assert metrics.week_of(date(2023, 12, 29)) == 18   # final partial week


# ---- dry run --------------------------------------------------------------
def test_longest_dry_run_basic():
    # <1mm is dry: [dry, dry, wet, dry, dry, dry, wet]
    assert metrics.longest_dry_run([0.0, 0.5, 3.0, 0.0, 0.2, 0.9, 2.0]) == 3


def test_longest_dry_run_all_wet_and_all_dry():
    assert metrics.longest_dry_run([2, 2, 2]) == 0
    assert metrics.longest_dry_run([0, 0, 0, 0]) == 4


def test_longest_dry_run_ignores_nan_as_break():
    assert metrics.longest_dry_run([0.0, np.nan, 0.0, 0.0]) == 2


# ---- weekly metrics --------------------------------------------------------
def _one_week_daily(block, year, values, start=date(2023, 9, 1)):
    from datetime import timedelta
    return pd.DataFrame({
        "block": block,
        "date": [pd.Timestamp(start + timedelta(days=i)) for i in range(len(values))],
        "rain_mm": values,
    })


def test_weekly_by_year_counts():
    # 7 days: rain = 0, 1.0, 1.5, 6.0, 0.5, 5.0, 12.0
    #  rainy (>1): 1.5, 6.0, 5.0, 12.0            -> 4
    #  extreme(>5): 6.0, 12.0                      -> 2
    #  dry (<1):   0, 0.5                          -> 2
    vals = [0.0, 1.0, 1.5, 6.0, 0.5, 5.0, 12.0]
    wby = metrics.weekly_by_year(_one_week_daily("B", 2023, vals))
    row = wby.iloc[0]
    assert row["week"] == 1 and row["n_days"] == 7
    assert row["rainy_days"] == 4
    assert row["extreme_days"] == 2
    assert row["dry_days"] == 2
    assert row["total_mm"] == pytest.approx(26.0)


def test_threshold_is_strict_greater_than():
    # exactly 1.0 mm is NOT a rainy day (>1); exactly 5.0 NOT extreme (>5)
    vals = [1.0, 5.0]
    wby = metrics.weekly_by_year(_one_week_daily("B", 2023, vals))
    row = wby.iloc[0]
    assert row["rainy_days"] == 1     # only the 5.0 counts as rainy
    assert row["extreme_days"] == 0   # 5.0 is not > 5
    assert row["dry_days"] == 0       # 1.0 is not < 1


# ---- climatology + scoring -------------------------------------------------
def _multiyear(block, per_year_week1):
    """Build daily frames: dict year -> list of 7 daily values for week 1."""
    frames = []
    for y, vals in per_year_week1.items():
        frames.append(_one_week_daily(block, y, vals, start=date(y, 9, 1)))
    return pd.concat(frames, ignore_index=True)


def test_climatology_averages_and_full_week_flag():
    daily = _multiyear("B", {
        2021: [2, 2, 2, 0, 0, 0, 0],
        2022: [3, 3, 0, 0, 0, 0, 0],
    })
    wby = metrics.weekly_by_year(daily)
    clim = metrics.climatology(wby)
    row = clim[clim["week"] == 1].iloc[0]
    assert row["n_years"] == 2
    assert row["is_full_week"]
    # rainy days: 2021 -> 3 (>1: 2,2,2), 2022 -> 2 -> mean 2.5
    assert row["mean_rainy_days"] == pytest.approx(2.5)


def test_climatology_partial_week_flag():
    # final partial week (Dec 29-31) is flagged is_full_week=False
    from datetime import timedelta
    days = [(date(2023, 9, 1) + timedelta(days=i), 4.0) for i in range(7)]
    days += [(d, 4.0) for d in (date(2023, 12, 29), date(2023, 12, 30), date(2023, 12, 31))]
    daily = pd.DataFrame({"block": "B",
                          "date": [pd.Timestamp(d) for d, _ in days],
                          "rain_mm": [v for _, v in days]})
    clim = metrics.climatology(metrics.weekly_by_year(daily))
    assert clim[clim["week"] == 1].iloc[0]["is_full_week"]
    assert not clim[clim["week"] == 18].iloc[0]["is_full_week"]


def test_week_label_format():
    assert metrics.week_label(1) == "Sep 1-7"
    assert metrics.week_label(2) == "Sep 8-14"


# ---- configurable season (May–Aug) + user threshold -----------------------
def test_may_aug_season_dates_and_weeks():
    d = metrics.season_dates(2026, start_month=5, end_month=8)
    assert d[0] == date(2026, 5, 1) and d[-1] == date(2026, 8, 31)
    assert metrics.week_of(date(2026, 5, 1), start_month=5) == 1
    assert metrics.week_of(date(2026, 5, 8), start_month=5) == 2
    assert metrics.week_label(1, start_month=5, end_month=8) == "May 1-7"


def test_user_threshold_changes_rainy_and_dry_counts():
    # daily week: 0.5, 2, 4, 6, 0.5, 3, 8
    vals = [0.5, 2.0, 4.0, 6.0, 0.5, 3.0, 8.0]
    daily = _one_week_daily("B", 2023, vals)
    # threshold 1mm: rainy = 2,4,6,3,8 -> 5 ; dry (<1) = 0.5,0.5 -> 2
    r1 = metrics.weekly_by_year(daily, rainy_mm=1.0).iloc[0]
    assert r1["rainy_days"] == 5 and r1["dry_days"] == 2
    # threshold 5mm: rainy = 6,8 -> 2 ; dry (<5) = 0.5,2,4,0.5,3 -> 5
    r5 = metrics.weekly_by_year(daily, rainy_mm=5.0).iloc[0]
    assert r5["rainy_days"] == 2 and r5["dry_days"] == 5
    # cumulative rainfall is threshold-independent
    assert r1["total_mm"] == pytest.approx(r5["total_mm"])


# ---- block-pixel mapping (IMD nodata/coastal handling) --------------------
def test_block_pixels_snaps_to_valid_cell():
    from shapely.geometry import box
    lats = np.array([10.0, 10.25, 10.5])
    lons = np.array([79.0, 79.25, 79.5])
    geom = box(79.4, 10.15, 79.6, 10.35)      # only cell centre inside = (row1,col2)
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 2] = False                          # that cell is sea/nodata
    px = blk.block_pixels(geom, lats, lons, valid_mask=mask)
    assert px and all(mask[r, c] for r, c in px)   # snapped to a valid land cell


def test_weathernext_aggregate_mean_and_prob():
    from rainfall import weathernext as wnx
    daily = {
        "time": ["2026-09-10", "2026-09-11"],
        "precipitation_sum": [0.0, 10.0],
        "precipitation_sum_member01": [2.0, 20.0],
        "precipitation_sum_member02": [None, 0.0],   # missing on day 0
    }
    df = wnx._aggregate(daily, rainy_mm=1.0)
    # day0 valid members [0.0, 2.0] -> mean 1.0; >1mm: 2.0 -> 1/2 = 50%
    assert df.loc[0, "rain_mm"] == pytest.approx(1.0)
    assert df.loc[0, "prob_rain"] == pytest.approx(50.0)
    # day1 members [10, 20, 0] -> mean 10.0; >1mm: 10,20 -> 2/3
    assert df.loc[1, "rain_mm"] == pytest.approx(10.0)
    assert df.loc[1, "prob_rain"] == pytest.approx(200 / 3)
    assert int(df.loc[0, "n_members"]) == 3


def test_block_pixels_no_mask_uses_incell():
    from shapely.geometry import box
    lats = np.array([10.0, 10.25, 10.5])
    lons = np.array([79.0, 79.25, 79.5])
    geom = box(79.4, 10.15, 79.6, 10.35)
    assert blk.block_pixels(geom, lats, lons) == [(1, 2)]
