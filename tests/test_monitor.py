from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, text

from forecast import monitor
from ingest import db, schema


def test_psi_is_zero_for_the_same_distribution_and_grows_with_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert monitor.psi(ref, rng.normal(0, 1, 20_000)) < 0.01
    small, large = monitor.psi(ref, rng.normal(0.3, 1, 20_000)), monitor.psi(ref, rng.normal(1.5, 1, 20_000))
    assert 0.05 < small < large
    assert large > monitor.PSI_ALERT


def test_psi_hand_computed_two_bins():
    # Reference median splits at 0.5: half and half. Current: 80% above.
    ref = np.array([0, 0, 1, 1], dtype=float)
    cur = np.array([0, 1, 1, 1, 1], dtype=float)
    p, q = np.array([0.5, 0.5]), np.array([0.2, 0.8])
    assert monitor.psi(ref, cur, bins=2) == pytest.approx(np.sum((q - p) * np.log(q / p)))


def test_score_frame():
    ts = pd.to_datetime(["2026-11-02T00:00Z", "2026-11-02T01:00Z", "2026-11-02T02:00Z"])
    fc = pd.DataFrame({"source": "live", "model_version": "3", "counter_id": "a", "ts": ts,
                       "q10": [0.0, 5.0, 10.0], "q50": [2.0, 10.0, 20.0], "q90": [4.0, 15.0, 30.0]})
    actual = pd.DataFrame({"counter_id": "a", "ts": ts, "count": [3.0, 20.0, np.nan]})
    s = monitor.score_frame(fc, actual).iloc[0]
    assert s["rows"] == 2  # the unknown actual is not scored
    assert s["mae"] == pytest.approx((1 + 10) / 2)
    assert s["coverage_80"] == 0.5
    assert s["interval_width"] == pytest.approx((4 + 10) / 2)


@pytest.fixture
def engine(tmp_path):
    eng = db.get_engine(f"sqlite:///{tmp_path / 'm.db'}")
    db.init_db(eng)
    with eng.begin() as conn:  # sqlite stand-in for the Postgres view
        conn.execute(text("CREATE TABLE counts_clean (counter_id TEXT, ts TIMESTAMP, count INTEGER)"))
    return eng


def test_score_days_only_scores_new_days(engine):
    ts = pd.date_range("2026-11-02", periods=48, freq="h", tz="UTC")
    fc = pd.DataFrame({"source": "live", "origin": ts[0], "counter_id": "a", "ts": ts, "q10": 0.0,
                       "q50": 10.0, "q90": 20.0, "model_version": "3", "issued_at": ts[0]})
    db.upsert(engine, schema.forecasts, fc, ["source", "origin", "counter_id", "ts"])
    pd.DataFrame({"counter_id": "a", "ts": ts, "count": 12}).to_sql("counts_clean", engine, if_exists="append", index=False)

    assert monitor.score_days(engine, date(2026, 11, 2)) == 1
    assert monitor.score_days(engine, date(2026, 11, 2)) == 0  # already scored
    assert monitor.score_days(engine, date(2026, 11, 3)) == 1
    with engine.connect() as conn:
        rows = conn.execute(select(schema.forecast_scores)).all()
    assert [(str(r.day), r.mae, r.coverage_80) for r in rows] == [("2026-11-02", 2.0, 1.0), ("2026-11-03", 2.0, 1.0)]


def test_drift_reference_is_the_same_weeks_in_earlier_years():
    ts = pd.Series(pd.date_range("2024-01-01", "2026-11-01 23:00", freq="h", tz="UTC"))
    start, end = pd.Timestamp("2026-10-05", tz="UTC"), pd.Timestamp("2026-11-02", tz="UTC")
    ref = ts[monitor.seasonal_reference(ts, start, end)]
    assert set(ref.dt.year) == {2024, 2025}
    assert ref.dt.month.isin([9, 10, 11]).all()


def test_seasonal_weather_is_not_drift_but_an_odd_week_is():
    # A yearly temperature cycle: a normal November week is not drift; a heatwave in November is.
    ts = pd.date_range("2024-01-01", "2026-11-01 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    temp = np.asarray(12 - 10 * np.cos(2 * np.pi * (ts.day_of_year - 15) / 365)) + rng.normal(0, 2, len(ts))
    normal = monitor.drift_for_day({"t": pd.DataFrame({"ts": ts, "value": temp})}, date(2026, 11, 1))
    assert normal[0]["psi"] < monitor.PSI_ALERT
    hot = temp.copy()
    hot[ts >= pd.Timestamp("2026-10-05", tz="UTC")] += 12
    odd = monitor.drift_for_day({"t": pd.DataFrame({"ts": ts, "value": hot})}, date(2026, 11, 1))
    assert odd[0]["psi"] > monitor.PSI_ALERT
    assert odd[0]["current_rows"] == monitor.DRIFT_DAYS * 24
