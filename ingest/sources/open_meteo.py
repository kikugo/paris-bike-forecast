"""Hourly weather for Paris from Open-Meteo (CC BY 4.0, free API for non-commercial use).

Two endpoints: the archive API for history (reanalysis, final values) and the forecast
API with `past_days` for the last hours, which the archive may not cover yet. Rows carry a
`source` column so archive values can replace forecast-API values later, never the reverse.
"""

from datetime import date

import httpx
import numpy as np
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


def observed_only(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop hours after `now`: both endpoints also return model values for later today."""
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    return df[df["ts"] <= now].reset_index(drop=True)


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
    return observed_only(parse_hourly(resp.json(), "archive"))


def recent(client: httpx.Client, past_days: int = 2) -> pd.DataFrame:
    """Observed hours of the last `past_days` days from the forecast API (no future hours)."""
    resp = client.get(FORECAST_URL, params=_params(past_days=past_days, forecast_days=1))
    resp.raise_for_status()
    return observed_only(parse_hourly(resp.json(), "forecast"))


# Weather forecasts as they were issued, from the Previous Runs API: `<var>_previous_day<d>` is the
# value forecast d x 24 hours before the valid hour. Forecasting models must use these rather than
# observed weather, which is not known at the forecast cutoff.
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
]
LEAD_DAYS = range(1, 8)


def parse_previous_runs(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    ts = pd.to_datetime(hourly["time"]).tz_localize("UTC")
    parts = []
    for d in LEAD_DAYS:
        part = pd.DataFrame({v: hourly[f"{v}_previous_day{d}"] for v in FORECAST_VARIABLES})
        part.insert(0, "lead_day", d)
        part.insert(0, "ts", ts)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    return df.dropna(subset=FORECAST_VARIABLES, how="all").sort_values(["ts", "lead_day"]).reset_index(drop=True)


def previous_runs(client: httpx.Client, start: date, end: date) -> pd.DataFrame:
    hourly = ",".join(f"{v}_previous_day{d}" for v in FORECAST_VARIABLES for d in LEAD_DAYS)
    resp = client.get(
        PREVIOUS_RUNS_URL,
        params=_params(start_date=start.isoformat(), end_date=end.isoformat()) | {"hourly": hourly},
    )
    resp.raise_for_status()
    return parse_previous_runs(resp.json())


def forecast_at_issue(client: httpx.Client, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """The current weather forecast for the `horizon` hours after `origin`, in the shape the
    model expects: one row per hour with the lead day the backtest used for that hour.

    This is the live counterpart of the Previous Runs data: a forecast that exists when the
    weekly forecast is issued.
    """
    resp = client.get(FORECAST_URL, params=_params(forecast_days=9) | {"hourly": ",".join(FORECAST_VARIABLES)})
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df = pd.DataFrame({v: hourly[v] for v in FORECAST_VARIABLES})
    df.insert(0, "ts", pd.to_datetime(hourly["time"]))  # naive UTC, like the forecast package
    lead = (df["ts"] - origin) / pd.Timedelta(hours=1)
    df = df[(lead >= 0) & (lead < horizon)].copy()
    df.insert(1, "lead_day", np.maximum(1, np.ceil(lead[df.index] / 24)).astype(int))
    return df.reset_index(drop=True)
