import pandas as pd

from forecast.lgbm import LightGBMQuantile
from tests.forecast_fixtures import START, weekly_panel


def test_quantiles_are_ordered_and_cover_the_horizon():
    data = weekly_panel(weeks=8, noise=2.0)
    cutoff = START + pd.Timedelta(weeks=7)
    fc = LightGBMQuantile(params={"n_estimators": 30, "min_child_samples": 5}).fit_predict(data, cutoff, 168)
    assert len(fc) == 3 * 168
    assert (fc.q10 <= fc.q50).all() and (fc.q50 <= fc.q90).all()
    assert (fc.q10 >= 0).all()
    assert (fc.yhat == fc.q50).all()
    assert fc.ds.min() == cutoff
