"""Hourly weather for Paris from Open-Meteo (CC BY 4.0, free API for non-commercial use).

Two endpoints: the archive API for history (reanalysis, final values) and the forecast
API with `past_days` for the last hours, which the archive may not cover yet. Rows carry a
`source` column so archive values can replace forecast-API values later, never the reverse.
"""

from datetime import date

import httpx
import pandas as pd

from ingest import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "cloud_cover",
]


def parse_hourly(payload: dict, source: str) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({v: hourly[v] for v in VARIABLES})
    # Requests use timezone=GMT, so timestamps are UTC without an offset.
    df.insert(0, "ts", pd.to_datetime(hourly["time"]).tz_localize("UTC"))
    df["source"] = source
    return df.dropna(subset=VARIABLES, how="all").reset_index(drop=True)


def _params(**extra) -> dict:
    return {
        "latitude": config.PARIS_LAT,
        "longitude": config.PARIS_LON,
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
        **extra,
    }


def archive(client: httpx.Client, start: date, end: date) -> pd.DataFrame:
    resp = client.get(
        ARCHIVE_URL, params=_params(start_date=start.isoformat(), end_date=end.isoformat())
    )
    resp.raise_for_status()
    return parse_hourly(resp.json(), "archive")


def recent(client: httpx.Client, past_days: int = 2) -> pd.DataFrame:
    """Observed hours of the last `past_days` days from the forecast API (no future hours)."""
    resp = client.get(FORECAST_URL, params=_params(past_days=past_days, forecast_days=1))
    resp.raise_for_status()
    df = parse_hourly(resp.json(), "forecast")
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return df[df["ts"] <= now].reset_index(drop=True)
