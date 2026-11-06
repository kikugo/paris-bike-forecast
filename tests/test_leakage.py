"""A forecast made at the cutoff must not change if all data after the cutoff is deleted."""

import numpy as np
import pandas as pd
import pytest

from forecast import baselines, features
from forecast.backtest import forecast_unchanged_after_deletion
from forecast.lgbm import LightGBMQuantile
from tests.forecast_fixtures import START, weekly_panel

CUTOFF = START + pd.Timedelta(weeks=5)


def _data_with_extreme_future():
    data = weekly_panel(weeks=7, noise=2.0)
    data.loc[data.ds >= CUTOFF, "y"] = 1e6
    return data


@pytest.mark.parametrize("name", sorted(baselines.MODELS))
def test_registered_models_do_not_look_past_the_cutoff(name):
    model = baselines.MODELS[name]()
    assert forecast_unchanged_after_deletion(model, _data_with_extreme_future(), CUTOFF)


class LeakyMean:
    """Predicts each counter's mean over all rows it is given, future included."""

    name = "leaky"
    label = "leaky"

    def fit_predict(self, history, cutoff, horizon, exog=None):
        means = history.groupby("unique_id")["y"].mean()
        hours = pd.date_range(cutoff, periods=horizon, freq="h")
        return pd.DataFrame(
            [(uid, ds, m) for uid, m in means.items() for ds in hours], columns=["unique_id", "ds", "yhat"]
        )


def test_the_check_catches_a_leaky_model():
    assert not forecast_unchanged_after_deletion(LeakyMean(), _data_with_extreme_future(), CUTOFF)


def _exog():
    """Issued weather forecasts for every hour and lead day; the ones issued after the cutoff are wild."""
    ds = pd.date_range(START, CUTOFF + pd.Timedelta(weeks=2), freq="h")
    rng = np.random.default_rng(0)
    wx = pd.concat(
        [
            pd.DataFrame({"ts": ds, "lead_day": d, **{v: rng.normal(10, 3, len(ds)) for v in features.WEATHER}})
            for d in range(1, 8)
        ],
        ignore_index=True,
    )
    issued = wx.ts - pd.to_timedelta(wx.lead_day * 24, unit="h")
    wx.loc[issued > CUTOFF, features.WEATHER] = 1e6
    holidays = pd.DataFrame({"day": [pd.Timestamp("2026-02-11")]})
    school = pd.DataFrame(
        {"first_day": [pd.Timestamp("2026-02-07").date()], "last_day": [pd.Timestamp("2026-02-22").date()],
         "description": ["Vacances d'Hiver"]}
    )
    return {"weather_fc": wx, "holidays": holidays, "school_holidays": school}


def test_lightgbm_does_not_look_past_the_cutoff():
    model = LightGBMQuantile(params={"n_estimators": 20, "min_child_samples": 5})
    assert forecast_unchanged_after_deletion(model, _data_with_extreme_future(), CUTOFF, exog=_exog())
