"""WeatherNext (Google DeepMind) rainfall forecast via the Open-Meteo API.

Open-Meteo serves Google's WeatherNext 2 ensemble (64 members, 0.25 deg) as a
free, no-auth HTTPS API, so the forecast is fetched LIVE (no Earth Engine, no
access request, no baked file) and works from Streamlit Cloud with no credentials.

Endpoint: https://ensemble-api.open-meteo.com/v1/ensemble
Model:    google_weathernext2_ensemble
Variable: precipitation_sum (mm/day), one column per ensemble member.

Per day we report the ensemble MEAN rainfall (mm) and the PROBABILITY of rain
(fraction of members exceeding the rainy-day threshold).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODEL = "google_weathernext2_ensemble"
MAX_DAYS = 16


def _aggregate(daily: dict, rainy_mm: float) -> pd.DataFrame:
    """Ensemble mean + prob-of-rain per day from an Open-Meteo `daily` block.

    Split out from the HTTP call so it can be unit-tested without a network.
    """
    dates = daily["time"]
    member_keys = [k for k in daily if k.startswith("precipitation_sum")]
    arr = np.array(
        [[np.nan if v is None else v for v in daily[k]] for k in member_keys],
        dtype="float64")                              # (members, days)
    valid = np.isfinite(arr)
    has = valid.any(axis=0)
    masked = np.where(valid, arr, np.nan)
    with np.errstate(invalid="ignore", all="ignore"):
        mean = np.where(has, np.nanmean(masked, axis=0), np.nan)
        p10 = np.where(has, np.nanpercentile(masked, 10, axis=0), np.nan)
        p90 = np.where(has, np.nanpercentile(masked, 90, axis=0), np.nan)
    n_valid = np.maximum(valid.sum(axis=0), 1)
    prob = np.where(has, np.nansum((arr > rainy_mm) & valid, axis=0) / n_valid * 100.0, np.nan)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "rain_mm": mean,        # ensemble mean
        "p10": p10,             # 10th percentile (drier members)
        "p90": p90,             # 90th percentile (wetter members)
        "prob_rain": prob,      # % of members above the rainy-day threshold
        "n_members": len(member_keys),
    })


def forecast_point(lat: float, lon: float, horizon_days: int = MAX_DAYS,
                   rainy_mm: float = 1.0, timeout: int = 30) -> pd.DataFrame:
    """Live WeatherNext 2 daily rainfall forecast for one lat/lon.

    Returns DataFrame [date, rain_mm (ensemble mean), prob_rain %, n_members].
    Raises on network/HTTP error (the caller shows a friendly message).
    """
    params = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "daily": "precipitation_sum", "models": MODEL,
        "forecast_days": int(min(horizon_days, MAX_DAYS)), "timezone": "auto"})
    with urllib.request.urlopen(f"{ENDPOINT}?{params}", timeout=timeout) as r:
        payload = json.load(r)
    if "daily" not in payload:
        raise RuntimeError(f"Open-Meteo returned no daily data: {str(payload)[:200]}")
    return _aggregate(payload["daily"], rainy_mm)
