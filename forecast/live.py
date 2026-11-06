"""Issue the weekly forecast the way it was backtested.

The model was trained and scored on forecasts made at Monday 00:00 UTC for the next 168 hours, so
that is what gets served: once a week, after Monday's data load, every counter gets q10 / q50 / q90
for each hour of the week, written to the `forecasts` table. The API reads that table and the
daily scoring DAG checks it against what actually happened.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import Engine, select, text

from forecast import features
from forecast.data import HOUR
from ingest import config, db, schema
from ingest.sources import school_holidays

HORIZON = 168


class OriginTooEarly(RuntimeError):
    """The model has seen data at or after this origin, so its forecast could not be scored fairly."""


def week_origin(now: pd.Timestamp) -> pd.Timestamp:
    """Monday 00:00 UTC of the week containing `now` (naive UTC)."""
    day = now.floor("D")
    return day - pd.Timedelta(days=day.day_of_week)


def _naive_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def load_inputs(engine: Engine, origin: pd.Timestamp) -> dict:
    """Clean counts before the origin, public holidays and school holidays from Postgres."""
    start = pd.Timestamp(config.WINDOW_START)
    counts = pd.read_sql(
        text(
            "SELECT counter_id AS unique_id, ts AS ds, count AS y FROM counts_clean "
            "WHERE ts >= :start AND ts < :origin"
        ),
        engine,
        params={"start": start.tz_localize("UTC").to_pydatetime(), "origin": origin.tz_localize("UTC").to_pydatetime()},
    )
    counts["ds"] = _naive_utc(counts["ds"])
    counts["y"] = counts["y"].astype(float)
    holidays = pd.read_sql(select(schema.holidays.c.day), engine)
    school = pd.read_sql(select(schema.school_holidays), engine)
    return {"counts": counts, "holidays": holidays, "school_holidays": school}


def build_features(inputs: dict, origin: pd.Timestamp, weather_fc: pd.DataFrame) -> pd.DataFrame:
    counts = inputs["counts"]
    counts = counts[counts["ds"] < origin]
    X = features.build(
        counts,
        [origin],
        HORIZON,
        weather_fc,
        inputs["holidays"],
        school_days=school_holidays.days(inputs["school_holidays"]),
    )
    return X


def issue(
    engine: Engine,
    model,
    model_version: str,
    trained_until: pd.Timestamp,
    origin: pd.Timestamp,
    weather_fc: pd.DataFrame,
    inputs: dict | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """Forecast every counter for the week starting at `origin` and store it.

    `model.predict(X)` takes the feature frame and returns q10, q50, q90. `trained_until` is the
    first hour the model has not seen; an origin before it is refused.
    """
    if origin < trained_until:
        raise OriginTooEarly(
            f"origin {origin} is before the end of the model's training data ({trained_until}); "
            "the forecast would cover hours the model was trained on"
        )
    inputs = inputs or load_inputs(engine, origin)
    X = build_features(inputs, origin, weather_fc)
    model_input = X[features.FEATURES].copy()
    model_input["counter"] = model_input["counter"].astype(str)
    preds = model.predict(model_input)
    out = pd.DataFrame(
        {
            "source": "live",
            "origin": origin.tz_localize("UTC"),
            "counter_id": X["unique_id"].to_numpy(),
            "ts": X["ds"].dt.tz_localize("UTC").to_numpy(),
            "q10": preds["q10"].to_numpy(),
            "q50": preds["q50"].to_numpy(),
            "q90": preds["q90"].to_numpy(),
            "model_version": model_version,
            "issued_at": now(),
        }
    )
    db.upsert(engine, schema.forecasts, out, ["source", "origin", "counter_id", "ts"])
    return len(out)


def trained_until(snapshot_end: str) -> pd.Timestamp:
    """First hour after a model's training data, from its `snapshot_end` tag (naive UTC)."""
    return pd.Timestamp(snapshot_end).tz_convert("UTC").tz_localize(None) + HOUR


class _Retrained:
    """Adapter so a freshly fitted LightGBMQuantile answers like the registered pyfunc model."""

    def __init__(self, model):
        self.model = model

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        X = X.copy()
        X["counter"] = pd.Categorical(X["counter"], categories=self.model.counters)
        preds = np.column_stack([self.model.models[q].predict(X) for q in self.model.quantiles])
        preds = np.sort(np.clip(preds, 0, None), axis=1)
        return pd.DataFrame(preds, columns=["q10", "q50", "q90"], index=X.index)


def main() -> None:
    """Issue a week's forecast from the command line.

    --retrain fits a LightGBM on data before the origin only, with weather forecasts issued by
    then (from the snapshot), and issues with it. Use it when the champion has already seen the
    week: the forecast stays honest and is labelled with its own model version.
    """
    import argparse

    from forecast import backtest
    from forecast.features import issued_by
    from forecast.lgbm import LightGBMQuantile
    from forecast.snapshot import database_url

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--origin", required=True, help="Monday 00:00 UTC, e.g. 2026-11-02")
    parser.add_argument("--retrain", action="store_true", required=True)
    args = parser.parse_args()
    origin = pd.Timestamp(args.origin)
    if origin != week_origin(origin):
        raise SystemExit("origin must be a Monday 00:00 UTC")

    engine = db.get_engine(database_url())
    db.init_db(engine)
    inputs = load_inputs(engine, origin)
    exog = backtest.load_exog()
    weather = issued_by(exog["weather_fc"], origin)
    model = LightGBMQuantile().fit(inputs["counts"], origin, HORIZON, {**exog, "weather_fc": weather})
    rows = issue(engine, _Retrained(model), f"retrained-{origin.date()}", origin, origin, weather, inputs)
    print(f"issued {rows} rows for the week from {origin}")


if __name__ == "__main__":
    main()
