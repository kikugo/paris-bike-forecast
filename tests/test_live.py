import httpx
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from forecast import live
from ingest import db, schema
from ingest.sources import open_meteo
from tests.forecast_fixtures import START, weekly_panel

ORIGIN = START + pd.Timedelta(weeks=5)


class FakeModel:
    def predict(self, X):
        base = X["lag_168"].fillna(0).to_numpy()
        return pd.DataFrame({"q10": base * 0.8, "q50": base, "q90": base * 1.2})


def _inputs():
    return {"counts": weekly_panel(weeks=5), "holidays": pd.DataFrame({"day": []}),
            "school_holidays": pd.DataFrame(columns=["first_day", "last_day", "description"])}


def test_week_origin_is_monday_midnight():
    assert live.week_origin(pd.Timestamp("2026-11-06 17:30")) == pd.Timestamp("2026-11-02")
    assert live.week_origin(pd.Timestamp("2026-11-02 00:00")) == pd.Timestamp("2026-11-02")


def test_issue_refuses_an_origin_the_model_was_trained_on(tmp_path):
    engine = db.get_engine(f"sqlite:///{tmp_path / 'l.db'}")
    with pytest.raises(live.OriginTooEarly):
        live.issue(engine, FakeModel(), "3", ORIGIN + pd.Timedelta(hours=1), ORIGIN, None, _inputs())


def test_issue_writes_every_counter_hour(tmp_path):
    engine = db.get_engine(f"sqlite:///{tmp_path / 'l.db'}")
    db.init_db(engine)
    n = live.issue(engine, FakeModel(), "3", ORIGIN, ORIGIN, None, _inputs())
    assert n == 3 * 168
    out = pd.read_sql(select(schema.forecasts), engine)
    assert set(out.source) == {"live"} and set(out.model_version) == {"3"}
    assert (out.q10 <= out.q50).all() and (out.q50 <= out.q90).all()


def test_trained_until_is_the_hour_after_the_snapshot():
    assert live.trained_until("2026-11-05T22:00:00+00:00") == pd.Timestamp("2026-11-05 23:00")


def test_live_weather_gets_the_backtest_lead_days():
    origin = pd.Timestamp("2026-11-09")
    hours = pd.date_range(origin - pd.Timedelta(days=1), periods=24 * 10, freq="h")
    payload = {"hourly": {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
                          **{v: list(np.arange(len(hours), dtype=float)) for v in open_meteo.FORECAST_VARIABLES}}}
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    df = open_meteo.forecast_at_issue(client, origin, 168)
    assert len(df) == 168 and df.ts.min() == origin
    lead = {0: 1, 24: 1, 25: 2, 167: 7}
    for hours_ahead, day in lead.items():
        assert df.loc[df.ts == origin + pd.Timedelta(hours=hours_ahead), "lead_day"].item() == day
