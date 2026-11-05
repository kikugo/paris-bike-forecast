import pandas as pd
import pytest

from ingest.expectations import DataQualityError, require, validate


def _counts(**overrides):
    df = pd.DataFrame(
        {
            "counter_id": ["a", "a", "b"],
            "ts": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T00:00Z"]),
            "count": [1, 0, 3],
        }
    )
    for col, (idx, value) in overrides.items():
        df.loc[idx, col] = value
    return df


def test_clean_counts_pass():
    assert validate("counts_hourly", _counts()).success


def test_negative_count_fails():
    v = validate("counts_hourly", _counts(count=(1, -1)))
    assert not v.success
    assert any("between" in f for f in v.failures)
    with pytest.raises(DataQualityError):
        require(v)


def test_duplicate_counter_hour_fails():
    df = _counts(counter_id=(2, "a"))
    v = validate("counts_hourly", df)
    assert not v.success
    assert any("unique" in f for f in v.failures)


def test_timestamp_off_the_hour_fails():
    df = _counts(ts=(2, pd.Timestamp("2026-01-01T00:15Z")))
    assert not validate("counts_hourly", df).success


def test_weather_rules():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T01:00Z"]),
            "temperature_2m": [5.0, 6.0],
            "relative_humidity_2m": [80, 85],
            "precipitation": [0.0, 0.2],
            "rain": [0.0, 0.2],
            "wind_speed_10m": [10.0, 12.0],
            "cloud_cover": [100, 90],
            "source": ["archive", "forecast"],
        }
    )
    assert validate("weather_hourly", df).success
    df.loc[1, "precipitation"] = -0.1
    assert not validate("weather_hourly", df).success


def test_holidays_duplicate_day_fails():
    df = pd.DataFrame({"day": [pd.Timestamp("2026-01-01").date()] * 2, "name": ["x", "y"]})
    assert not validate("holidays", df).success


def test_weather_forecast_rules():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00Z"] * 2),
            "lead_day": [1, 2],
            "temperature_2m": [5.0, 6.0],
            "precipitation": [0.0, 0.1],
            "wind_speed_10m": [10.0, 12.0],
            "cloud_cover": [100, 90],
            "relative_humidity_2m": [80, 85],
        }
    )
    assert validate("weather_forecast", df).success
    df.loc[1, "lead_day"] = 1
    assert not validate("weather_forecast", df).success
