"""One global LightGBM model across counters, with 0.1 / 0.5 / 0.9 quantiles.

Training examples mirror the forecast: each Monday 00:00 in the history is an origin, and the
following hours are targets with features built relative to that origin (forecast/features.py).
The 0.5 quantile is the point forecast; 0.1 and 0.9 bound an 80% interval.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from forecast import features

QUANTILES = (0.1, 0.5, 0.9)
PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "colsample_bytree": 0.9,
    "random_state": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


class LightGBMQuantile:
    name = "lightgbm"
    label = "LightGBM (median)"

    def __init__(self, params: dict | None = None, quantiles=QUANTILES):
        self.params = {**PARAMS, **(params or {})}
        self.quantiles = tuple(quantiles)
        self.models: dict[float, LGBMRegressor] = {}
        self.counters: list[str] = []

    def fit(self, history: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, exog: dict | None = None):
        exog = exog or {}
        history = history[history["ds"] < cutoff]
        self.counters = sorted(history["unique_id"].unique())
        origins = features.origins_before(cutoff, history["ds"].min(), horizon)
        train = features.build(
            history, origins, horizon, _weather(exog, cutoff), exog.get("holidays"), self.counters
        )
        train = train[train["y"].notna()]
        X, y = train[features.FEATURES], train["y"]
        self.models = {
            q: LGBMRegressor(objective="quantile", alpha=q, **self.params).fit(X, y) for q in self.quantiles
        }
        return self

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int, exog: dict | None = None):
        exog = exog or {}
        history = history[history["ds"] < origin]
        X = features.build(
            history, [origin], horizon, _weather(exog, origin), exog.get("holidays"), self.counters
        )
        preds = np.column_stack([self.models[q].predict(X[features.FEATURES]) for q in self.quantiles])
        # Separate quantile models can cross; sorting each row keeps q10 <= q50 <= q90.
        preds = np.sort(np.clip(preds, 0, None), axis=1)
        out = X[["unique_id", "ds"]].copy()
        for i, q in enumerate(self.quantiles):
            out[f"q{int(q * 100)}"] = preds[:, i]
        out["yhat"] = out["q50"]
        return out.reset_index(drop=True)

    def fit_predict(self, history, cutoff, horizon, exog=None):
        return self.fit(history, cutoff, horizon, exog).predict(history, cutoff, horizon, exog)


def _weather(exog: dict, cutoff: pd.Timestamp) -> pd.DataFrame | None:
    wx = exog.get("weather_fc")
    return None if wx is None else features.issued_by(wx, cutoff)


MODELS = {"lightgbm": LightGBMQuantile}
