"""Rolling-origin backtest.

Each fold trains on every hour before its cutoff and forecasts the next `horizon` hours. Cutoffs
are Mondays 00:00 UTC, one week apart, ending with the last week that is complete in the
snapshot. Scores cover (counter, hour) pairs with a known actual.

Run: uv run python -m forecast.backtest [--models seasonal_naive ...] [--folds 8]
"""

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from forecast import metrics
from forecast.baselines import MODELS as BASELINES
from forecast.data import HOUR
from forecast.features import issued_by
from forecast.lgbm import MODELS as LGBM

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "snapshot"
RESULTS = ROOT / "results"
HORIZON = 168
FOLDS = 8


class Forecaster(Protocol):
    name: str
    label: str

    def fit_predict(
        self, history: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, exog: dict | None = None
    ) -> pd.DataFrame: ...


def fold_cutoffs(data_end: pd.Timestamp, n_folds: int = FOLDS, horizon: int = HORIZON) -> list[pd.Timestamp]:
    """Weekly Monday cutoffs; the last one leaves a full `horizon` of data after it."""
    if horizon > 168:
        raise ValueError("test windows longer than a week would overlap")
    latest_start = data_end + HOUR - pd.Timedelta(hours=horizon)
    day = latest_start.normalize()
    last = day - pd.Timedelta(days=day.day_of_week)
    return [last - pd.Timedelta(weeks=n_folds - 1 - i) for i in range(n_folds)]


def exog_known_at(exog: dict | None, cutoff: pd.Timestamp) -> dict | None:
    """Exogenous data as it existed at the cutoff: weather forecasts issued later are removed.
    Public holidays are fixed in advance and kept."""
    if not exog:
        return exog
    out = dict(exog)
    if exog.get("weather_fc") is not None:
        out["weather_fc"] = issued_by(exog["weather_fc"], cutoff)
    return out


def forecast_unchanged_after_deletion(
    model: Forecaster,
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int = HORIZON,
    exog: dict | None = None,
) -> bool:
    """The leakage check: a forecast made at the cutoff must not change when every row after
    the cutoff is deleted, counts and issued weather forecasts alike. The model gets the full
    data on purpose."""
    full = model.fit_predict(data, cutoff, horizon, exog)
    full = full.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    cut = model.fit_predict(data[data["ds"] < cutoff], cutoff, horizon, exog_known_at(exog, cutoff))
    cut = cut.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return full[["unique_id", "ds"]].equals(cut[["unique_id", "ds"]]) and np.allclose(
        full["yhat"].to_numpy(), cut["yhat"].to_numpy(), equal_nan=True
    )


def run(
    models: Iterable[Forecaster],
    data: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    horizon: int = HORIZON,
    exog: dict | None = None,
) -> dict:
    models = list(models)
    folds, pooled = [], {m.name: [] for m in models}
    for i, cutoff in enumerate(cutoffs, start=1):
        history = data[data["ds"] < cutoff]  # the harness never shows a model anything later
        end = cutoff + pd.Timedelta(hours=horizon)
        actual = data[(data["ds"] >= cutoff) & (data["ds"] < end) & data["y"].notna()]
        actual = actual[actual["unique_id"].isin(history["unique_id"].unique())]
        fold = {
            "fold": i,
            "cutoff": cutoff.isoformat(),
            "test_end": (end - HOUR).isoformat(),
            "counters": int(actual["unique_id"].nunique()),
            "models": {},
        }
        for model in models:
            fc = model.fit_predict(history, cutoff, horizon, exog_known_at(exog, cutoff))
            scored = actual.merge(fc, on=["unique_id", "ds"], how="left")
            if scored["yhat"].isna().any():
                raise RuntimeError(f"{model.name} left hours without a forecast at fold {i}")
            fold["models"][model.name] = _score(scored)
            pooled[model.name].append(scored)
        folds.append(fold)
    overall = {}
    for model in models:
        overall[model.name] = _score(pd.concat(pooled[model.name], ignore_index=True))
    return {
        "horizon_hours": horizon,
        "models": {m.name: m.label for m in models},
        "folds": folds,
        "overall": overall,
    }


def _score(scored: pd.DataFrame) -> dict:
    out = metrics.score(scored["y"], scored["yhat"])
    if {"q10", "q50", "q90"} <= set(scored.columns):
        out |= metrics.interval_score(
            scored["y"],
            scored["q10"],
            scored["q90"],
            {0.1: scored["q10"], 0.5: scored["q50"], 0.9: scored["q90"]},
        )
    return out


def load_exog(directory: Path = SNAPSHOT_DIR) -> dict:
    wx = pd.read_parquet(directory / "weather_forecast.parquet")
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True).dt.tz_localize(None)
    return {"weather_fc": wx, "holidays": pd.read_parquet(directory / "holidays.parquet")}


def load_snapshot(directory: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    df = pd.read_parquet(directory / "counts.parquet")
    df = df.rename(columns={"counter_id": "unique_id", "ts": "ds", "count": "y"})
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def main(argv: list[str] | None = None, registry: dict[str, Callable[[], Forecaster]] | None = None) -> None:
    registry = registry or (BASELINES | LGBM)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(registry))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--out", default=str(RESULTS / "backtest.json"))
    args = parser.parse_args(argv)

    snapshot = json.loads((RESULTS / "snapshot.json").read_text())
    data = load_snapshot()
    cutoffs = fold_cutoffs(data["ds"].max(), args.folds)
    exog = load_exog()
    result = run([registry[name]() for name in args.models], data, cutoffs, exog=exog)
    result = {"snapshot": snapshot, **result}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    for name, s in result["overall"].items():
        print(f"{name}: MAE {s['mae']:.2f}  RMSE {s['rmse']:.2f}  sMAPE {s['smape']:.1f}%")


if __name__ == "__main__":
    main()
