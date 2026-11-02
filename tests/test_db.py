import json

import pandas as pd
import pytest
from sqlalchemy import select

from ingest import db, gaps, schema


@pytest.fixture
def engine(tmp_path):
    eng = db.get_engine(f"sqlite:///{tmp_path / 'bikes.db'}")
    db.init_db(eng)
    return eng


def _weather(source, temp):
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00Z"]),
            "temperature_2m": [temp],
            "relative_humidity_2m": [80.0],
            "precipitation": [0.0],
            "rain": [0.0],
            "wind_speed_10m": [5.0],
            "cloud_cover": [50.0],
            "source": [source],
        }
    )


def test_upsert_is_idempotent(engine):
    df = pd.DataFrame(
        {"counter_id": ["a", "a"], "ts": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T01:00Z"]), "count": [1, 2]}
    )
    db.upsert(engine, schema.counts_hourly, df, ["counter_id", "ts"])
    df["count"] = [3, 4]
    db.upsert(engine, schema.counts_hourly, df, ["counter_id", "ts"])
    with engine.connect() as conn:
        rows = conn.execute(select(schema.counts_hourly.c.count).order_by("ts")).scalars().all()
    assert rows == [3, 4]


def test_forecast_weather_never_overwrites_archive(engine):
    only_forecast = "weather_hourly.source = 'forecast'"
    db.upsert(engine, schema.weather_hourly, _weather("forecast", 1.0), ["ts"], only_if=only_forecast)
    db.upsert(engine, schema.weather_hourly, _weather("archive", 2.0), ["ts"])
    db.upsert(engine, schema.weather_hourly, _weather("forecast", 9.0), ["ts"], only_if=only_forecast)
    with engine.connect() as conn:
        row = conn.execute(select(schema.weather_hourly)).one()
    assert (row.source, row.temperature_2m) == ("archive", 2.0)


def test_ingest_log(engine):
    assert not db.already_loaded(engine, "counters_history", "2024")
    db.log_ingest(engine, "counters_history", "2024", 10)
    assert db.already_loaded(engine, "counters_history", "2024")


def test_gap_refresh_end_to_end(engine, tmp_path, monkeypatch):
    start = pd.Timestamp("2023-01-01T00:00Z")
    hours = pd.date_range(start, periods=200, freq="h", tz="UTC")
    good = pd.DataFrame({"counter_id": "good", "ts": hours, "count": 5})
    # 'dead' reports zeros for 48 hours, so 24% of the window is missing.
    dead = good.assign(counter_id="dead")
    dead.loc[50:97, "count"] = 0
    db.upsert(engine, schema.counts_hourly, pd.concat([good, dead]), ["counter_id", "ts"])

    summary = gaps.refresh(engine, results_dir=str(tmp_path))

    assert summary["counters_total"] == 2
    assert summary["counters_kept"] == 1
    saved = json.loads((tmp_path / "gap_report.json").read_text())
    assert [c["counter_id"] for c in saved["counters"] if c["excluded"]] == ["dead"]
    with engine.connect() as conn:
        runs = conn.execute(select(schema.zero_runs)).all()
    assert len(runs) == 1 and runs[0].hours == 48
