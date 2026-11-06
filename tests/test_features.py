import numpy as np
import pandas as pd

from forecast import features
from tests.forecast_fixtures import START


def _ramp(weeks=6, counters=("a",)):
    """y = hours since START, so every lag is easy to check by hand."""
    ds = pd.date_range(START, periods=168 * weeks, freq="h")
    return pd.concat(
        [pd.DataFrame({"unique_id": c, "ds": ds, "y": np.arange(len(ds), dtype=float)}) for c in counters],
        ignore_index=True,
    )


ORIGIN = START + pd.Timedelta(weeks=5)
O = 168 * 5  # hour index of the origin in _ramp


def _row(df, lead):
    return df[df.lead_hours == lead].iloc[0]


def test_count_features_only_read_before_the_origin():
    df = features.build(_ramp(), [ORIGIN], horizon=168)
    for lead in (0, 23, 24, 71, 167):
        r = _row(df, lead)
        assert r.lag_168 == O + lead - 168
        # Same hour on the last day before the origin.
        assert r.last_day_same_hour == O + lead - 24 * (lead // 24 + 1)
        assert r.last_day_same_hour < O
        assert r.mean_same_hour_of_week_4w == np.mean([O + lead - 168 * k for k in range(1, 5)])
        assert r.mean_24h_before == np.mean(np.arange(O - 24, O))
        assert r.mean_168h_before == np.mean(np.arange(O - 168, O))
        assert r.y == O + lead


def test_count_features_do_not_change_when_the_future_is_deleted():
    data = _ramp(weeks=7)
    full = features.build(data, [ORIGIN], horizon=168)
    cut = features.build(data[data.ds < ORIGIN], [ORIGIN], horizon=168)
    cols = [c for c in features.FEATURES if c != "counter"]
    pd.testing.assert_frame_equal(full[cols], cut[cols])


def test_weather_comes_from_forecasts_issued_by_the_origin():
    ds = pd.date_range(ORIGIN, periods=168, freq="h")
    wx = pd.concat(
        [pd.DataFrame({"ts": ds, "lead_day": d, **{v: float(d) for v in features.WEATHER}}) for d in range(1, 8)],
        ignore_index=True,
    )
    df = features.build(_ramp(), [ORIGIN], horizon=168, weather_fc=wx)
    # Lead 0-24h uses the 1-day-ahead forecast, 25-48h the 2-day one, ..., up to 7 days.
    for lead, day in [(0, 1), (24, 1), (25, 2), (48, 2), (167, 7)]:
        assert _row(df, lead).temperature_2m == day
    # Nothing issued after the origin is ever used.
    assert (features.issued_by(wx, ORIGIN).shape[0] < wx.shape[0])


def test_holiday_flags_use_paris_dates():
    holidays = pd.DataFrame({"day": [ORIGIN.date() + pd.Timedelta(days=2)]})
    df = features.build(_ramp(), [ORIGIN], horizon=168, holidays=holidays)
    # 2026-02-09 00:00 UTC is 01:00 in Paris; the holiday is Wednesday local time.
    wed = df[(df.ds >= ORIGIN + pd.Timedelta(days=2)) & (df.ds < ORIGIN + pd.Timedelta(days=3))]
    assert wed.is_holiday.sum() == 23  # 23:00 UTC on Wednesday is already Thursday in Paris
    assert df[df.ds == ORIGIN + pd.Timedelta(days=1, hours=12)].day_before_holiday.item() == 1
    assert df[df.ds == ORIGIN + pd.Timedelta(days=3, hours=12)].day_after_holiday.item() == 1


def test_training_origins_are_mondays_that_end_before_the_cutoff():
    origins = features.origins_before(ORIGIN, START, horizon=168)
    assert origins[0] == START + pd.Timedelta(weeks=4)
    assert origins[-1] == ORIGIN - pd.Timedelta(weeks=1)
    assert all(o.day_of_week == 0 for o in origins)


def test_school_holiday_flag_uses_paris_dates():
    school = pd.DatetimeIndex([ORIGIN + pd.Timedelta(days=3)])  # Thursday
    df = features.build(_ramp(), [ORIGIN], horizon=168, school_days=school)
    thu = df[(df.ds >= ORIGIN + pd.Timedelta(days=3)) & (df.ds < ORIGIN + pd.Timedelta(days=4))]
    assert thu.is_school_holiday.sum() == 23  # 23:00 UTC is already Friday in Paris (winter)
    assert df.is_school_holiday.sum() == 24
