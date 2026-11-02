import pandas as pd

from ingest.sources import holidays, open_meteo


def test_parse_hourly_weather():
    payload = {
        "hourly": {
            "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
            **{v: [1.0, None] for v in open_meteo.VARIABLES},
        }
    }
    df = open_meteo.parse_hourly(payload, "archive")
    # The hour with no values at all is dropped.
    assert len(df) == 1
    assert df.ts.iloc[0] == pd.Timestamp("2026-01-01T00:00Z")
    assert df.source.iloc[0] == "archive"


def test_observed_only_drops_future_hours():
    df = pd.DataFrame({"ts": pd.date_range("2026-11-02T10:00Z", periods=4, freq="h")})
    kept = open_meteo.observed_only(df, now=pd.Timestamp("2026-11-02T11:40Z"))
    assert kept.ts.max() == pd.Timestamp("2026-11-02T11:00Z")


def test_parse_holidays_sorted_as_dates():
    df = holidays.parse({"2026-12-25": "Jour de Noël", "2026-01-01": "1er janvier"})
    assert [str(d) for d in df.day] == ["2026-01-01", "2026-12-25"]
