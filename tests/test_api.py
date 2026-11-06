from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main
from ingest import db, schema

NOW = datetime(2026, 11, 6, 17, 30, tzinfo=UTC)


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = db.get_engine(f"sqlite:///{tmp_path / 'api.db'}")
    db.init_db(engine)
    origin = pd.Timestamp("2026-11-02", tz="UTC")
    ts = pd.date_range(origin, periods=168, freq="h")
    db.upsert(engine, schema.counters, pd.DataFrame({"counter_id": ["a"], "counter_name": ["Quai A"]}), ["counter_id"])
    fc = pd.DataFrame({"source": "live", "origin": origin, "counter_id": "a", "ts": ts, "q10": 1.0, "q50": 2.0,
                       "q90": 3.0, "model_version": "3", "issued_at": origin + pd.Timedelta(hours=9)})
    db.upsert(engine, schema.forecasts, fc, ["source", "origin", "counter_id", "ts"])
    monkeypatch.setattr(main, "_engine", engine)
    monkeypatch.setattr(main, "now", lambda: NOW)
    return TestClient(main.app)


def test_next_24_hours_from_now(client):
    r = client.get("/forecast/a")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "3" and body["counter_name"] == "Quai A"
    assert len(body["hours"]) == 24
    assert body["hours"][0]["ts"].startswith("2026-11-06T17:00")
    assert all(h["q10"] <= h["q50"] <= h["q90"] for h in body["hours"])


def test_hours_past_the_issued_week_are_not_invented(client):
    r = client.get("/forecast/a", params={"hours": 168})
    # The week issued on 2 Nov ends on Sunday 8 Nov 23:00.
    assert len(r.json()["hours"]) == (pd.Timestamp("2026-11-09", tz="UTC") - pd.Timestamp(NOW).floor("h")) // pd.Timedelta(hours=1)


def test_unknown_counter_is_404(client):
    assert client.get("/forecast/nope").status_code == 404


def test_no_forecast_for_now_is_503(client, monkeypatch):
    monkeypatch.setattr(main, "now", lambda: datetime(2026, 11, 9, 3, tzinfo=UTC))
    assert client.get("/forecast/a").status_code == 503


def test_bad_hours_is_422(client):
    assert client.get("/forecast/a", params={"hours": 0}).status_code == 422
