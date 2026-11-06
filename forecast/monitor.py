"""Daily monitoring: score issued forecasts against actuals, and measure feature drift with PSI.

Scoring covers every UTC day that has forecasts and at least one known count but no score yet,
per source (`live`, `backtest`) and model version: MAE of the median, 80% interval coverage and
mean interval width, written to `forecast_scores`.

Drift compares the last 28 days with the same weeks of the year in earlier years (the window's
calendar days plus or minus 14 days), for the inputs the model's features are made of: hourly
counts (behind the lag and rolling-mean features) and the day-ahead weather forecasts.
- The reference is seasonal: November weather always differs from a whole year of weather, so
  comparing against all history would raise an alarm every week.
- The window is 28 days, not 7: one week of weather is mostly one weather pattern, and 7-day PSI
  flagged ordinary weeks as drift.
PSI uses the reference deciles as bins; rule of thumb: below 0.1 stable, 0.1 to 0.2 moderate,
above 0.2 drift.

Run (stack up): uv run python -m forecast.monitor
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select, text

from ingest import config, db, schema

EPS = 1e-4
BINS = 10
DRIFT_DAYS = 28
SEASON_MARGIN_DAYS = 14
WEATHER = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover", "relative_humidity_2m"]
PSI_ALERT = 0.2


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    """Population stability index of `current` against `reference`, with reference-quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    ref = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1)
    cur = np.bincount(np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1)
    p = np.maximum(ref / ref.sum(), EPS)
    q = np.maximum(cur / cur.sum(), EPS)
    return float(np.sum((q - p) * np.log(q / p)))


def _utc(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], utc=True)


def score_frame(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Daily scores per source and model version. Both frames use counter_id and ts (UTC)."""
    merged = forecasts.merge(actuals, on=["counter_id", "ts"])
    merged = merged[merged["count"].notna()]
    if merged.empty:
        return pd.DataFrame(columns=["day", "source", "model_version", "rows", "mae", "coverage_80", "interval_width"])
    merged["day"] = merged["ts"].dt.date
    merged["abs_err"] = (merged["count"] - merged["q50"]).abs()
    merged["inside"] = (merged["count"] >= merged["q10"]) & (merged["count"] <= merged["q90"])
    merged["width"] = merged["q90"] - merged["q10"]
    out = merged.groupby(["day", "source", "model_version"]).agg(
        rows=("abs_err", "size"), mae=("abs_err", "mean"), coverage_80=("inside", "mean"), interval_width=("width", "mean")
    )
    return out.reset_index().round({"mae": 4, "coverage_80": 4, "interval_width": 4})


def score_days(engine: Engine, until: date) -> int:
    """Score every day up to `until` that has forecasts and no score yet."""
    f = schema.forecasts
    with engine.connect() as conn:
        scored = {
            (r.day, r.source, r.model_version)
            for r in conn.execute(select(schema.forecast_scores.c.day, schema.forecast_scores.c.source,
                                         schema.forecast_scores.c.model_version))
        }
    end = pd.Timestamp(until, tz="UTC") + pd.Timedelta(days=1)
    forecasts = pd.read_sql(
        select(f.c.source, f.c.model_version, f.c.counter_id, f.c.ts, f.c.q10, f.c.q50, f.c.q90).where(
            f.c.ts < end.to_pydatetime()
        ),
        engine,
    )
    if forecasts.empty:
        return 0
    forecasts["ts"] = _utc(forecasts, "ts")
    forecasts["day"] = forecasts["ts"].dt.date
    keep = [(d, s, v) not in scored for d, s, v in zip(forecasts["day"], forecasts["source"], forecasts["model_version"], strict=True)]
    forecasts = forecasts[keep].drop(columns="day")
    if forecasts.empty:
        return 0
    actuals = pd.read_sql(
        text("SELECT counter_id, ts, count FROM counts_clean WHERE ts >= :start AND ts < :end"),
        engine,
        params={"start": forecasts["ts"].min().to_pydatetime(), "end": end.to_pydatetime()},
    )
    actuals["ts"] = _utc(actuals, "ts")
    scores = score_frame(forecasts, actuals)
    if scores.empty:
        return 0
    scores["scored_at"] = datetime.now(UTC)
    db.upsert(engine, schema.forecast_scores, scores, ["day", "source", "model_version"])
    return len(scores)


def seasonal_reference(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Rows from earlier years whose day of year falls within the [start, end) window's days,
    widened by SEASON_MARGIN_DAYS on each side."""
    centre = start + (end - start) / 2
    half = (end - start) / 2 + pd.Timedelta(days=SEASON_MARGIN_DAYS)
    diff = (ts.dt.day_of_year - centre.day_of_year).abs()
    diff = np.minimum(diff, 365 - diff)  # wrap around the new year
    earlier = ts < start - pd.Timedelta(days=365) + half
    return earlier & (diff <= half.days)


def drift_for_day(series: dict[str, pd.DataFrame], day: date) -> list[dict]:
    """PSI per input for the DRIFT_DAYS ending on `day` against the same weeks in earlier years.

    `series` maps a feature name to a frame with `ts` (UTC) and `value`.
    """
    end = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=DRIFT_DAYS)
    rows = []
    for name, df in series.items():
        ref = df.loc[seasonal_reference(df["ts"], start, end), "value"].to_numpy()
        cur = df.loc[(df["ts"] >= start) & (df["ts"] < end), "value"].to_numpy()
        if len(cur) == 0 or len(ref) == 0:
            continue
        rows.append({"day": day, "feature": name, "psi": round(psi(ref, cur), 4),
                     "reference_rows": len(ref), "current_rows": len(cur)})
    return rows


def load_drift_inputs(engine: Engine) -> dict[str, pd.DataFrame]:
    start = pd.Timestamp(config.WINDOW_START, tz="UTC").to_pydatetime()
    counts = pd.read_sql(
        text("SELECT ts, count AS value FROM counts_clean WHERE ts >= :start AND count IS NOT NULL"),
        engine, params={"start": start},
    )
    counts["ts"] = _utc(counts, "ts")
    wx = pd.read_sql(
        text(f"SELECT ts, {', '.join(WEATHER)} FROM weather_forecast WHERE lead_day = 1 AND ts >= :start"),
        engine, params={"start": start},
    )
    wx["ts"] = _utc(wx, "ts")
    out = {"count": counts}
    for col in WEATHER:
        out[f"weather_{col}"] = wx[["ts", col]].rename(columns={col: "value"})
    return out


def drift_days(engine: Engine, days: list[date]) -> int:
    inputs = load_drift_inputs(engine)
    rows = [r for d in days for r in drift_for_day(inputs, d)]
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["computed_at"] = datetime.now(UTC)
    db.upsert(engine, schema.feature_drift, df, ["day", "feature"])
    return len(df)


def summary(engine: Engine) -> dict:
    """Small summary for the README: days scored and mean scores per source, latest PSI per input."""
    scores = pd.read_sql(select(schema.forecast_scores), engine)
    drift = pd.read_sql(select(schema.feature_drift), engine)
    by_source = {}
    for source, g in scores.groupby("source"):
        w = g["rows"]
        by_source[source] = {
            "days": int(g["day"].nunique()),
            "first_day": str(g["day"].min()),
            "last_day": str(g["day"].max()),
            "mae": round(float((g["mae"] * w).sum() / w.sum()), 4),
            "coverage_80": round(float((g["coverage_80"] * w).sum() / w.sum()), 4),
        }
    latest = {}
    if not drift.empty:
        last_day = drift["day"].max()
        latest = {
            "day": str(last_day),
            "psi": {r.feature: r.psi for r in drift[drift["day"] == last_day].itertuples()},
            "alert_threshold": PSI_ALERT,
        }
    return {"scores": by_source, "drift": latest}


def run(engine: Engine, until: date | None = None, drift_back_days: int = 1) -> dict:
    until = until or (datetime.now(UTC).date() - timedelta(days=1))
    scored = score_days(engine, until)
    days = [until - timedelta(days=i) for i in range(drift_back_days)]
    drifted = drift_days(engine, days)
    return {"scored_rows": scored, "drift_rows": drifted}


def main() -> None:
    import argparse

    from forecast.snapshot import database_url

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-back-days", type=int, default=70, help="days of drift history to (re)compute")
    args = parser.parse_args()
    engine = db.get_engine(database_url())
    db.init_db(engine)
    print(run(engine, drift_back_days=args.drift_back_days))
    out = Path(__file__).resolve().parents[1] / "results" / "monitoring.json"
    out.write_text(json.dumps(summary(engine), indent=2, default=str) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
