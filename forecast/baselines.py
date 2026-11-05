"""Seasonal-naive baselines through statsforecast."""

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, SeasonalWindowAverage

from forecast.data import HOUR, fill_history, to_grid

WEEK_HOURS = 168


class StatsBaseline:
    def __init__(self, name: str, label: str, model):
        self.name = name
        self.label = label
        self.model = model

    def fit_predict(self, history: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
        past = history[history["ds"] < cutoff]
        grid = fill_history(to_grid(past, end=cutoff - HOUR))
        sf = StatsForecast(models=[self.model], freq="h", n_jobs=1)
        fc = sf.forecast(df=grid, h=horizon)
        model_col = next(c for c in fc.columns if c not in ("unique_id", "ds"))
        return fc.rename(columns={model_col: "yhat"})[["unique_id", "ds", "yhat"]]


def seasonal_naive() -> StatsBaseline:
    return StatsBaseline(
        "seasonal_naive", "Seasonal naive (same hour last week)", SeasonalNaive(season_length=WEEK_HOURS)
    )


def seasonal_naive_4w() -> StatsBaseline:
    return StatsBaseline(
        "seasonal_naive_4w",
        "Seasonal naive, 4-week average",
        SeasonalWindowAverage(season_length=WEEK_HOURS, window_size=4),
    )


MODELS = {"seasonal_naive": seasonal_naive, "seasonal_naive_4w": seasonal_naive_4w}
