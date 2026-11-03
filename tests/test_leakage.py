"""A forecast made at the cutoff must not change if all data after the cutoff is deleted."""

import pandas as pd
import pytest

from forecast import baselines
from forecast.backtest import forecast_unchanged_after_deletion
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

    def fit_predict(self, history, cutoff, horizon):
        means = history.groupby("unique_id")["y"].mean()
        hours = pd.date_range(cutoff, periods=horizon, freq="h")
        return pd.DataFrame(
            [(uid, ds, m) for uid, m in means.items() for ds in hours], columns=["unique_id", "ds", "yhat"]
        )


def test_the_check_catches_a_leaky_model():
    assert not forecast_unchanged_after_deletion(LeakyMean(), _data_with_extreme_future(), CUTOFF)
