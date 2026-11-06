"""Features for a forecast made at an origin (the cutoff), for each of the next `horizon` hours.

Every feature is something known at the origin:
- count lags only read hours before the origin (for an hour 3 days ahead the "24h lag" is in the
  future, so the same-hour value of the last day before the origin is used instead);
- weather comes from forecasts issued at or before the origin, never observed weather;
- calendar and public holidays are known in advance.
Hour, day of week and holidays use Paris local time, which is what cyclists live by.
"""

import warnings

import numpy as np
import pandas as pd

from forecast.data import HOUR

TZ = "Europe/Paris"
WEATHER = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover", "relative_humidity_2m"]
FEATURES = [
    "counter",
    "lead_hours",
    "hour",
    "day_of_week",
    "month",
    "is_holiday",
    "day_before_holiday",
    "day_after_holiday",
    "lag_168",
    "last_day_same_hour",
    "mean_same_hour_of_week_4w",
    "mean_24h_before",
    "mean_168h_before",
    *WEATHER,
]


def lead_day(lead_hours: np.ndarray) -> np.ndarray:
    """Forecast lead in days such that the forecast was issued at or before the origin."""
    return np.maximum(1, np.ceil(lead_hours / 24)).astype(int)


def issued_by(weather_fc: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Weather forecasts that existed at the cutoff."""
    issued = weather_fc["ts"] - pd.to_timedelta(weather_fc["lead_day"] * 24, unit="h")
    return weather_fc[issued <= cutoff]


def origins_before(cutoff: pd.Timestamp, first: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
    """Training origins: Mondays 00:00 with four weeks of history and a horizon ending before the cutoff."""
    start = (first + pd.Timedelta(weeks=4)).normalize()
    start += pd.Timedelta(days=(7 - start.day_of_week) % 7)
    last = cutoff - pd.Timedelta(hours=horizon)
    return list(pd.date_range(start, last, freq="7D")) if start <= last else []


def build(
    counts: pd.DataFrame,
    origins: list[pd.Timestamp],
    horizon: int,
    weather_fc: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
    counters: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (origin, counter, target hour) with FEATURES and the target `y` (NaN if unknown).

    `counts` uses the panel format (unique_id, ds naive UTC, y). Only rows strictly before each
    origin feed that origin's count features.
    """
    counters = counters or sorted(counts["unique_id"].unique())
    if not origins or not counters:
        return pd.DataFrame(columns=["unique_id", "origin", "ds", "y", *FEATURES])

    grid_start = counts["ds"].min().floor("D") - pd.Timedelta(weeks=5)
    grid_end = max(origins) + pd.Timedelta(hours=horizon)
    n_hours = int((grid_end - grid_start) / HOUR) + 1
    index = {c: i for i, c in enumerate(counters)}
    Y = np.full((len(counters), n_hours), np.nan, dtype=np.float64)
    known = counts[counts["unique_id"].isin(index) & (counts["ds"] <= grid_end)]
    rows = known["unique_id"].map(index).to_numpy()
    cols = ((known["ds"] - grid_start) / HOUR).astype(int).to_numpy()
    Y[rows, cols] = known["y"].to_numpy(dtype=float)

    lead = np.arange(horizon)
    k_last_day = lead // 24 + 1  # whole days back so that t - 24k is before the origin
    parts = []
    for origin in origins:
        o = int((origin - grid_start) / HOUR)
        idx = o + lead
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN windows
            frame = {
                "y": Y[:, idx],
                "lag_168": Y[:, idx - 168],
                "last_day_same_hour": Y[:, idx - 24 * k_last_day],
                "mean_same_hour_of_week_4w": np.nanmean(
                    np.stack([Y[:, idx - 168 * k] for k in range(1, 5)]), axis=0
                ),
                "mean_24h_before": np.repeat(np.nanmean(Y[:, o - 24 : o], axis=1)[:, None], horizon, 1),
                "mean_168h_before": np.repeat(np.nanmean(Y[:, o - 168 : o], axis=1)[:, None], horizon, 1),
            }
        part = pd.DataFrame({k: v.ravel() for k, v in frame.items()})
        part.insert(0, "lead_hours", np.tile(lead, len(counters)))
        part.insert(0, "ds", np.tile(origin + pd.to_timedelta(lead, unit="h"), len(counters)))
        part.insert(0, "origin", origin)
        part.insert(0, "unique_id", np.repeat(counters, horizon))
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)

    local = df["ds"].dt.tz_localize("UTC").dt.tz_convert(TZ)
    df["hour"] = local.dt.hour
    df["day_of_week"] = local.dt.day_of_week
    df["month"] = local.dt.month
    days = pd.DatetimeIndex([]) if holidays is None else pd.DatetimeIndex(pd.to_datetime(holidays["day"]))
    local_day = local.dt.tz_localize(None).dt.normalize()
    one_day = pd.Timedelta(days=1)
    df["is_holiday"] = local_day.isin(days).astype(int)
    df["day_before_holiday"] = (local_day + one_day).isin(days).astype(int)
    df["day_after_holiday"] = (local_day - one_day).isin(days).astype(int)

    df["lead_day"] = lead_day(df["lead_hours"].to_numpy())
    if weather_fc is not None and not weather_fc.empty:
        wx = weather_fc[["ts", "lead_day", *WEATHER]].rename(columns={"ts": "ds"})
        df = df.merge(wx, on=["ds", "lead_day"], how="left")
        # Belt and braces: drop any weather value issued after its origin.
        late = df["ds"] - pd.to_timedelta(df["lead_day"] * 24, unit="h") > df["origin"]
        df.loc[late, WEATHER] = np.nan
    else:
        for col in WEATHER:
            df[col] = np.nan

    df["counter"] = pd.Categorical(df["unique_id"], categories=counters)
    return df.drop(columns=["lead_day"])

