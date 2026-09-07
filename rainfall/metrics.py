"""Rainfall insight metrics: weekly binning, thresholds, dry spells, scoring.

All rainfall inputs are DAILY TOTALS in mm (block-mean already applied).

Definitions (as specified by the field team):
    * season      : Sep 1 - Dec 31
    * week        : fixed 7-day bins from Sep 1 (Week 1 = Sep 1-7, ...).
                    The final bin (Dec 29-31) is a 3-day partial week and is
                    flagged n_days<7 and excluded from the plantation ranking.
    * rainy day   : daily rainfall  > 1 mm
    * extreme day : daily rainfall  > 5 mm
    * dry day     : daily rainfall  < 1 mm
    * dry spell   : reported two ways per week -- count of dry days, and the
                    longest run of consecutive dry days within the week.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

SEASON_START_MONTH = 9   # September
SEASON_END_MONTH = 12    # December
RAINY_MM = 1.0
EXTREME_MM = 5.0
DRY_MM = 1.0
FULL_WEEK_DAYS = 7


# Named seasons (start/end month, inclusive). Weeks are fixed 7-day bins from
# the 1st of the start month.
SEASONS = {
    "planting": {"start_month": 9, "end_month": 12, "label": "Sep–Dec"},   # 5-yr analysis
    "monsoon": {"start_month": 5, "end_month": 8, "label": "May–Aug"},      # current-year actual
}
_LAST_DAY = {5: 31, 8: 31, 9: 30, 12: 31}


def season_dates(year: int, start_month: int = SEASON_START_MONTH,
                 end_month: int = SEASON_END_MONTH) -> list[date]:
    """All calendar dates from ``start_month`` 1 to ``end_month`` end of ``year``."""
    d = date(year, start_month, 1)
    end = date(year, end_month, _LAST_DAY[end_month])
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def week_of(d: date, start_month: int = SEASON_START_MONTH) -> int:
    """1-based fixed-7-day week index counted from ``start_month`` 1 of that year."""
    offset = (d - date(d.year, start_month, 1)).days
    return offset // FULL_WEEK_DAYS + 1


def longest_dry_run(daily_mm, dry_mm: float = DRY_MM) -> int:
    """Longest run of consecutive dry days (rainfall < ``dry_mm``) in a sequence."""
    best = run = 0
    for v in daily_mm:
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and v < dry_mm:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def weekly_by_year(daily: pd.DataFrame, start_month: int = SEASON_START_MONTH,
                   rainy_mm: float = RAINY_MM, extreme_mm: float = EXTREME_MM,
                   dry_mm: float | None = None) -> pd.DataFrame:
    """Per (block, year, week) metrics from a daily block-rainfall table.

    ``daily`` columns: block, date (datetime), rain_mm (daily total mm).
    ``rainy_mm`` is the (user-configurable) rainy-day threshold; a dry day is
    below ``dry_mm`` (defaults to the same threshold, so rainy/dry are
    complementary). ``extreme_mm`` stays a fixed heavy-rain marker.
    """
    if dry_mm is None:
        dry_mm = rainy_mm
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].apply(lambda t: week_of(t.date(), start_month))

    rows = []
    for (block, year, week), g in df.groupby(["block", "year", "week"], sort=True):
        vals = g.sort_values("date")["rain_mm"].to_numpy(dtype="float64")
        valid = vals[~np.isnan(vals)]
        rows.append({
            "block": block,
            "year": year,
            "week": int(week),
            "n_days": len(vals),
            "rainy_days": int((valid > rainy_mm).sum()),
            "extreme_days": int((valid > extreme_mm).sum()),
            "dry_days": int((valid < dry_mm).sum()),
            "longest_dry_run": longest_dry_run(vals, dry_mm),
            "total_mm": float(np.nansum(vals)),
        })
    return pd.DataFrame(rows).sort_values(["block", "year", "week"]).reset_index(drop=True)


def climatology(wby: pd.DataFrame) -> pd.DataFrame:
    """5-year averages per (block, week). Plain descriptive statistics only --
    no scoring or 'suitability' judgement; the raw weekly data speaks for itself.
    """
    rows = []
    for (block, week), g in wby.groupby(["block", "week"], sort=True):
        n_days = int(g["n_days"].max())
        rows.append({
            "block": block,
            "week": int(week),
            "n_days": n_days,
            "is_full_week": n_days >= FULL_WEEK_DAYS,
            "n_years": len(g),
            "mean_rainy_days": float(g["rainy_days"].mean()),
            "mean_extreme_days": float(g["extreme_days"].mean()),
            "mean_dry_days": float(g["dry_days"].mean()),
            "mean_longest_dry_run": float(g["longest_dry_run"].mean()),
            "mean_total_mm": float(g["total_mm"].mean()),
            "std_total_mm": float(g["total_mm"].std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values(["block", "week"]).reset_index(drop=True)


def week_label(week: int, start_month: int = SEASON_START_MONTH,
               end_month: int = SEASON_END_MONTH, year: int = 2001) -> str:
    """Human date range for a week index, e.g. 'Sep 1-7' or 'May 1-7'."""
    start = date(year, start_month, 1) + timedelta(days=(week - 1) * FULL_WEEK_DAYS)
    end = min(start + timedelta(days=FULL_WEEK_DAYS - 1), date(year, end_month, _LAST_DAY[end_month]))
    if start.month == end.month:
        return f"{start:%b} {start.day}-{end.day}"
    return f"{start:%b %d}-{end:%b %d}"
